"""Orchestrator: manages the 4-agent fact-checking pipeline.

Designed for exactly 4 Gemini API calls, paced to stay within
the free-tier limit of 5 requests per minute.
"""
from agents import run_researcher, run_skeptic, run_adversary, run_judge


def run_pipeline(claim, callback=None):
    """Run the full fact-checking pipeline.

    Args:
        claim: The user's claim to fact-check.
        callback: Optional callable(agent, event, data) for progress updates.
            Events: "start", "log", "stream", "handover", "done"

    Returns:
        Dict with all agent outputs and final verdict.
    """
    def emit(agent, event, data=""):
        if callback:
            callback(agent, event, data)

    def make_cbs(agent_name):
        """Create stream_cb and log_cb wired to the callback."""
        def stream_cb(chunk):
            emit(agent_name, "stream", chunk)
        def log_cb(msg):
            emit(agent_name, "log", msg)
        return stream_cb, log_cb

    results = {"claim": claim}

    # ── Agent 1: Researcher (Gemini call 1 of 4) ──
    emit("researcher", "start")
    s_cb, l_cb = make_cbs("researcher")
    researcher_out = run_researcher(claim, stream_cb=s_cb, log_cb=l_cb)
    results["researcher"] = researcher_out

    n_sources = sum(
        len(f.get("evidence", []))
        for f in researcher_out.get("findings", [])
    )
    n_subclaims = len(researcher_out.get("sub_claims", []))
    emit("researcher", "done")
    emit("researcher", "handover",
         "Passing %d sources across %d sub-claims to Skeptic" % (n_sources, n_subclaims))

    # ── Agent 2: Skeptic (Gemini call 2 of 4) ──
    emit("skeptic", "start")
    s_cb, l_cb = make_cbs("skeptic")
    skeptic_out = run_skeptic(researcher_out, stream_cb=s_cb, log_cb=l_cb)
    results["skeptic"] = skeptic_out

    audited = skeptic_out.get("audited_findings", [])
    accepted = sum(len(f.get("accepted_sources", [])) for f in audited)
    rejected = sum(len(f.get("rejected_sources", [])) for f in audited)
    emit("skeptic", "done")
    emit("skeptic", "handover",
         "%d accepted, %d rejected → Adversary" % (accepted, rejected))

    # ── Agent 3: Adversary (Gemini call 3 of 4) ──
    emit("adversary", "start")
    s_cb, l_cb = make_cbs("adversary")
    adversary_out = run_adversary(claim, skeptic_out, stream_cb=s_cb, log_cb=l_cb)
    results["adversary"] = adversary_out

    n_for = len(adversary_out.get("for_evidence", []))
    n_against = len(adversary_out.get("against_evidence", []))
    n_critiques = len(adversary_out.get("critiques", []))
    emit("adversary", "done")
    emit("adversary", "handover",
         "%d for, %d against, %d critiques → Judge" % (n_for, n_against, n_critiques))

    # ── Agent 4: Judge (Gemini call 4 of 4) ──
    emit("judge", "start")
    s_cb, l_cb = make_cbs("judge")
    judge_out = run_judge(claim, adversary_out, skeptic_out, stream_cb=s_cb, log_cb=l_cb)
    results["judge"] = judge_out
    emit("judge", "done")

    return results
