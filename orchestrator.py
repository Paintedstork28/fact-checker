"""Orchestrator: manages the 4-agent fact-checking pipeline."""
from config import MAX_RESEARCHER_RETRIES
from agents import run_researcher, run_skeptic, run_adversary, run_judge


def run_pipeline(claim, status_callback=None):
    """Run the full fact-checking pipeline.

    Args:
        claim: The user's claim to fact-check.
        status_callback: Optional callable(agent_name, message) for progress updates.

    Returns:
        Dict with all agent outputs and final verdict.
    """
    def update(agent, msg):
        if status_callback:
            status_callback(agent, msg)

    results = {"claim": claim, "retries": 0}

    # ── Agent 1: Researcher ──
    update("researcher", "Breaking claim into sub-claims and searching the web...")
    researcher_out = run_researcher(claim)
    results["researcher"] = researcher_out

    # ── Agent 2: Skeptic (with retry loop) ──
    retries = 0
    skeptic_out = None
    current_research = researcher_out

    while retries <= MAX_RESEARCHER_RETRIES:
        update("skeptic", "Auditing source credibility...")
        skeptic_out = run_skeptic(current_research)

        # Check if any sub-claim needs more research
        needs_retry = False
        audited = skeptic_out.get("audited_findings", [])
        for finding in audited:
            if finding.get("needs_more_research", False):
                needs_retry = True
                break

        if not needs_retry or retries >= MAX_RESEARCHER_RETRIES:
            break

        retries += 1
        results["retries"] = retries
        update("researcher", "Re-searching (retry %d/%d) — Skeptic wants better sources..." % (retries, MAX_RESEARCHER_RETRIES))
        # Build refined search instructions from skeptic's suggestions
        suggestions = []
        for f in audited:
            s = f.get("research_suggestions", "")
            if s:
                suggestions.append(s)
        refined_claim = claim + ". Focus on: " + "; ".join(suggestions) if suggestions else claim
        current_research = run_researcher(refined_claim)
        results["researcher"] = current_research

    results["skeptic"] = skeptic_out

    # ── Agent 3: Adversary ──
    update("adversary", "Playing devil's advocate and stress-testing evidence...")
    adversary_out = run_adversary(claim, skeptic_out)
    results["adversary"] = adversary_out

    # ── Agent 4: Judge ──
    update("judge", "Weighing all evidence and delivering verdict...")
    judge_out = run_judge(claim, adversary_out, skeptic_out)
    results["judge"] = judge_out

    return results
