"""Four adversarial fact-checking agents powered by Gemini."""
import json
import re

from google import genai
from config import get_gemini_api_key, GEMINI_MODEL, SOURCE_SCORE_THRESHOLD
from search_tools import search_ddg, fetch_url_text
from source_scorer import score_source

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_gemini_api_key())
    return _client


def _call_gemini(system_prompt, user_prompt):
    """Call Gemini and return text response."""
    client = _get_client()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )
    return resp.text


def _call_gemini_stream(system_prompt, user_prompt, stream_cb=None):
    """Call Gemini with streaming; calls stream_cb(chunk_text) for each chunk."""
    if stream_cb is None:
        return _call_gemini(system_prompt, user_prompt)
    client = _get_client()
    full_text = ""
    for chunk in client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    ):
        if chunk.text:
            full_text += chunk.text
            stream_cb(chunk.text)
    return full_text


def _extract_json(text):
    """Extract JSON object from LLM response text."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_response": text}


# ─── AGENT 1: THE RESEARCHER ────────────────────────────────────────────

RESEARCHER_PROMPT = """You are THE RESEARCHER — thorough, eager, and comprehensive.
Your job is to find as much relevant information as possible about a claim.

Steps:
1. Break the claim into individual verifiable sub-claims.
2. For each sub-claim, I will provide web search results. Analyze them carefully.
3. Summarize key findings with source attribution.

Return ONLY valid JSON (no markdown fences) in this format:
{
  "sub_claims": ["claim1", "claim2"],
  "findings": [
    {
      "sub_claim": "...",
      "evidence": [
        {"source_url": "...", "source_title": "...", "snippet": "...", "supports_claim": true}
      ]
    }
  ]
}"""


def run_researcher(claim, stream_cb=None, log_cb=None):
    """Agent 1: Search the web and gather evidence for the claim."""
    if log_cb:
        log_cb("Breaking claim into sub-claims...")

    breakdown_prompt = (
        "Break this claim into verifiable sub-claims (max 3). "
        "Return ONLY a JSON list of strings.\n\nClaim: " + claim
    )
    raw = _call_gemini(
        "You break claims into verifiable sub-claims. Return ONLY a JSON list of strings.",
        breakdown_prompt,
    )
    try:
        sub_claims = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group(0))
    except Exception:
        sub_claims = [claim]

    if log_cb:
        log_cb("Found %d sub-claims" % len(sub_claims))

    all_results = []
    for sc in sub_claims:
        if log_cb:
            log_cb('Searching: "%s"...' % sc[:60])
        search_results = search_ddg(sc)
        if log_cb:
            log_cb("Found %d results" % len(search_results))
        for r in search_results[:3]:
            if r.get("url"):
                content = fetch_url_text(r["url"])
                if content:
                    r["fetched_content"] = content[:1500]
        all_results.append({"sub_claim": sc, "search_results": search_results})

    if log_cb:
        log_cb("Synthesizing findings...")

    synth_prompt = (
        "Claim: " + claim + "\n\n"
        "Sub-claims and search results:\n" +
        json.dumps(all_results, indent=2, default=str)[:12000] +
        "\n\nAnalyze these results and produce your findings JSON."
    )
    response = _call_gemini_stream(RESEARCHER_PROMPT, synth_prompt, stream_cb)
    result = _extract_json(response)
    if "sub_claims" not in result:
        result["sub_claims"] = sub_claims
    return result


# ─── AGENT 2: THE SKEPTIC ───────────────────────────────────────────────

SKEPTIC_PROMPT = """You are THE SKEPTIC — distrustful, nitpicky, always looking for problems.
Your job is to audit the Researcher's sources for reliability, bias, and credibility.

You receive:
- The Researcher's findings with sources
- Credibility scores for each source (pre-computed)

Steps:
1. Review each source and its credibility score.
2. REJECT any source with score < 5 and explain why.
3. Flag contradictions between sources.
4. If fewer than 2 credible sources remain for a sub-claim, note what additional research is needed.

Return ONLY valid JSON (no markdown fences):
{
  "audited_findings": [
    {
      "sub_claim": "...",
      "accepted_sources": [{"url": "...", "title": "...", "score": 7, "snippet": "..."}],
      "rejected_sources": [{"url": "...", "reason": "..."}],
      "contradictions": ["..."],
      "needs_more_research": false,
      "research_suggestions": ""
    }
  ]
}"""


def run_skeptic(researcher_output, stream_cb=None, log_cb=None):
    """Agent 2: Audit sources for credibility."""
    findings = researcher_output.get("findings", [])
    for finding in findings:
        for ev in finding.get("evidence", []):
            url = ev.get("source_url", "")
            sc = score_source(url)
            ev["credibility_score"] = sc["score"]
            ev["credibility_tier"] = sc["tier"]
            ev["domain"] = sc["domain"]
            if log_cb:
                status = "ACCEPTED" if sc["score"] >= SOURCE_SCORE_THRESHOLD else "REJECTED"
                log_cb("Scoring %s → %d/10 (Tier %s) — %s" % (
                    sc["domain"][:30], sc["score"], sc["tier"], status
                ))

    if log_cb:
        log_cb("Analyzing source quality...")

    prompt = (
        "Researcher's findings with credibility scores:\n" +
        json.dumps(researcher_output, indent=2, default=str)[:12000] +
        "\n\nAudit these sources. Reject anything with score < " +
        str(SOURCE_SCORE_THRESHOLD) + "."
    )
    response = _call_gemini_stream(SKEPTIC_PROMPT, prompt, stream_cb)
    return _extract_json(response)


# ─── AGENT 3: THE ADVERSARY ─────────────────────────────────────────────

ADVERSARY_PROMPT = """You are THE ADVERSARY — a devil's advocate and contrarian stress-tester.
Your job is to argue AGAINST the claim and find every possible weakness.

Steps:
1. Sort the audited evidence into FOR and AGAINST piles.
2. Look for: contradictions, missing context, logical fallacies, cherry-picked data,
   outdated info, correlation vs causation errors.
3. Write a critique for each piece of evidence.
4. Identify what's MISSING — what evidence would be needed to fully verify this?

Return ONLY valid JSON (no markdown fences):
{
  "for_evidence": [{"point": "...", "source": "...", "strength": "strong/moderate/weak"}],
  "against_evidence": [{"point": "...", "source": "...", "strength": "strong/moderate/weak"}],
  "critiques": ["..."],
  "missing_evidence": ["..."],
  "logical_issues": ["..."]
}"""


def run_adversary(claim, skeptic_output, stream_cb=None, log_cb=None):
    """Agent 3: Argue against the claim, find weaknesses."""
    if log_cb:
        log_cb("Stress-testing evidence...")

    prompt = (
        "Original claim: " + claim + "\n\n"
        "Skeptic's audited findings:\n" +
        json.dumps(skeptic_output, indent=2, default=str)[:12000] +
        "\n\nNow tear this apart. Find every weakness."
    )
    response = _call_gemini_stream(ADVERSARY_PROMPT, prompt, stream_cb)
    return _extract_json(response)


# ─── AGENT 4: THE JUDGE ─────────────────────────────────────────────────

JUDGE_PROMPT = """You are THE JUDGE — balanced, judicial, fair, and measured.
Your job is to weigh all evidence impartially and deliver a final verdict.

Verdict scale:
- TRUE (confidence 80-100%)
- MOSTLY TRUE (confidence 60-80%)
- PARTIALLY TRUE (confidence 40-60%)
- MOSTLY FALSE (confidence 20-40%)
- FALSE (confidence 0-20%)

Consider: source quality, argument strength, consensus vs outlier, recency.

Return ONLY valid JSON (no markdown fences):
{
  "sub_verdicts": [
    {"sub_claim": "...", "verdict": "...", "confidence": 85, "reasoning": "..."}
  ],
  "overall_verdict": "MOSTLY TRUE",
  "overall_confidence": 72,
  "reasoning": "A clear 2-3 sentence summary of why this verdict was reached.",
  "key_sources": [{"url": "...", "title": "...", "why_important": "..."}]
}"""


def run_judge(claim, adversary_output, skeptic_output, stream_cb=None, log_cb=None):
    """Agent 4: Deliver final verdict."""
    if log_cb:
        log_cb("Weighing all evidence...")

    prompt = (
        "Original claim: " + claim + "\n\n"
        "Audited sources (from Skeptic):\n" +
        json.dumps(skeptic_output, indent=2, default=str)[:6000] +
        "\n\nAdversary's analysis:\n" +
        json.dumps(adversary_output, indent=2, default=str)[:6000] +
        "\n\nDeliver your verdict."
    )
    response = _call_gemini_stream(JUDGE_PROMPT, prompt, stream_cb)
    return _extract_json(response)
