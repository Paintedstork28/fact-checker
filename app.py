"""Fact Checker — Multi-Agent Verification Tool (Streamlit UI)."""
import streamlit as st
from orchestrator import run_pipeline

st.set_page_config(page_title="Fact Checker", page_icon="", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("How It Works")
    st.markdown("""
This tool uses **4 adversarial AI agents** to rigorously verify claims:

**1. The Researcher** searches the web and gathers evidence from multiple sources.

**2. The Skeptic** audits every source for credibility, bias, and reliability — rejecting weak sources and demanding re-research if needed.

**3. The Adversary** plays devil's advocate, arguing against the claim and finding every possible weakness in the evidence.

**4. The Judge** weighs all evidence impartially and delivers a final verdict with confidence score.

---
**Powered by:** Gemini 2.5 Flash + DuckDuckGo Search

**Cost:** Completely free
    """)

# ── Main UI ──────────────────────────────────────────────────────────────
st.title("Fact Checker")
st.caption("Enter a claim and 4 adversarial agents will verify it for you.")

# Session state
if "results" not in st.session_state:
    st.session_state.results = None

# ── Verdict color/icon mapping ───────────────────────────────────────────
VERDICT_STYLES = {
    "TRUE": ("#22c55e", "check-circle"),
    "MOSTLY TRUE": ("#84cc16", "check-circle"),
    "PARTIALLY TRUE": ("#eab308", "minus-circle"),
    "MOSTLY FALSE": ("#f97316", "x-circle"),
    "FALSE": ("#ef4444", "x-circle"),
}

AGENT_CONFIG = {
    "researcher": {"label": "The Researcher", "icon": ":material/search:"},
    "skeptic": {"label": "The Skeptic", "icon": ":material/policy:"},
    "adversary": {"label": "The Adversary", "icon": ":material/gavel:"},
    "judge": {"label": "The Judge", "icon": ":material/balance:"},
}


def get_verdict_style(verdict):
    v = (verdict or "").upper().strip()
    return VERDICT_STYLES.get(v, ("#6b7280", "help-circle"))


# ── Claim input ──────────────────────────────────────────────────────────
claim = st.chat_input("Type a claim to fact-check...")

if claim:
    st.session_state.results = None

    st.markdown("**Claim:** " + claim)
    st.divider()

    # Build status containers for each agent
    agent_order = ["researcher", "skeptic", "adversary", "judge"]
    containers = {}   # agent_name -> st.status widget
    placeholders = {} # agent_name -> (log_placeholder, stream_placeholder)
    handovers = {}    # agent_name -> st.empty for handover text
    stream_buffers = {}  # agent_name -> accumulated stream text

    for name in agent_order:
        cfg = AGENT_CONFIG[name]
        containers[name] = st.status(cfg["label"], state="complete")
        handovers[name] = st.empty()

    current_agent = {"name": None}

    def event_callback(agent, event, data=""):
        cfg = AGENT_CONFIG.get(agent, {})

        if event == "start":
            current_agent["name"] = agent
            stream_buffers[agent] = ""
            # Re-create inner placeholders each time agent starts
            status_widget = containers[agent]
            status_widget.update(label=cfg.get("label", agent), state="running")
            # Write log and stream placeholders inside the status
            with status_widget:
                placeholders[agent] = (st.empty(), st.empty())

        elif event == "log":
            if agent in placeholders:
                # Append log line — use the log placeholder
                placeholders[agent][0].markdown(
                    "_" + str(data) + "_"
                )

        elif event == "stream":
            if agent in placeholders:
                stream_buffers[agent] = stream_buffers.get(agent, "") + str(data)
                # Show accumulated stream in the stream placeholder
                # Truncate display to last 2000 chars to keep UI responsive
                display = stream_buffers[agent]
                if len(display) > 2000:
                    display = "..." + display[-2000:]
                placeholders[agent][1].code(display, language=None)

        elif event == "handover":
            handovers[agent].markdown(
                "  **→** " + str(data)
            )

        elif event == "done":
            containers[agent].update(
                label=cfg.get("label", agent),
                state="complete",
            )

    # Run pipeline
    try:
        results = run_pipeline(claim, callback=event_callback)
        st.session_state.results = results
    except Exception as e:
        st.error("Error: " + str(e))
        st.stop()

# ── Display results ──────────────────────────────────────────────────────
results = st.session_state.results
if results:
    judge = results.get("judge", {})
    verdict = judge.get("overall_verdict", "UNKNOWN")
    confidence = judge.get("overall_confidence", 0)
    reasoning = judge.get("reasoning", "No reasoning provided.")
    color, icon = get_verdict_style(verdict)

    # Verdict card
    st.markdown(
        '<div style="background:%s20;border-left:4px solid %s;padding:16px;border-radius:8px;margin:16px 0">'
        '<h2 style="color:%s;margin:0">%s</h2>'
        '<p style="margin:4px 0 0 0;font-size:0.9em">Confidence: <strong>%d%%</strong></p>'
        '</div>' % (color, color, color, verdict, confidence),
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.progress(confidence / 100.0)

    # Reasoning
    st.markdown("### Reasoning")
    st.write(reasoning)

    # Retries info
    if results.get("retries", 0) > 0:
        st.info("The Skeptic sent the Researcher back %d time(s) for better sources." % results["retries"])

    # Sub-verdicts
    sub_verdicts = judge.get("sub_verdicts", [])
    if sub_verdicts:
        st.markdown("### Sub-Claim Verdicts")
        for sv in sub_verdicts:
            sc_color, _ = get_verdict_style(sv.get("verdict", ""))
            st.markdown(
                '<div style="background:#f8f9fa;padding:12px;border-radius:6px;margin:8px 0;'
                'border-left:3px solid %s">'
                '<strong>%s</strong> — <span style="color:%s">%s (%d%%)</span><br>'
                '<small>%s</small></div>'
                % (
                    sc_color,
                    sv.get("sub_claim", ""),
                    sc_color,
                    sv.get("verdict", "?"),
                    sv.get("confidence", 0),
                    sv.get("reasoning", ""),
                ),
                unsafe_allow_html=True,
            )

    # Expandable sections
    st.markdown("---")

    with st.expander("Evidence FOR the claim"):
        for_ev = results.get("adversary", {}).get("for_evidence", [])
        if for_ev:
            for e in for_ev:
                st.markdown("- **[%s]** %s (%s)" % (
                    e.get("strength", "?"),
                    e.get("point", ""),
                    e.get("source", ""),
                ))
        else:
            st.write("No supporting evidence found.")

    with st.expander("Evidence AGAINST the claim"):
        against_ev = results.get("adversary", {}).get("against_evidence", [])
        if against_ev:
            for e in against_ev:
                st.markdown("- **[%s]** %s (%s)" % (
                    e.get("strength", "?"),
                    e.get("point", ""),
                    e.get("source", ""),
                ))
        else:
            st.write("No counter-evidence found.")

    with st.expander("Critiques & Logical Issues"):
        critiques = results.get("adversary", {}).get("critiques", [])
        issues = results.get("adversary", {}).get("logical_issues", [])
        for c in critiques:
            st.markdown("- " + str(c))
        for i in issues:
            st.markdown("- " + str(i))
        if not critiques and not issues:
            st.write("No critiques or logical issues identified.")

    with st.expander("Missing Evidence"):
        missing = results.get("adversary", {}).get("missing_evidence", [])
        if missing:
            for m in missing:
                st.markdown("- " + str(m))
        else:
            st.write("No missing evidence identified.")

    with st.expander("Key Sources"):
        key_sources = judge.get("key_sources", [])
        if key_sources:
            for s in key_sources:
                url = s.get("url", "")
                title = s.get("title", url)
                why = s.get("why_important", "")
                if url:
                    st.markdown("- [%s](%s) — %s" % (title, url, why))
                else:
                    st.markdown("- %s — %s" % (title, why))
        else:
            st.write("No key sources listed.")

    with st.expander("Source Audit Trail (Skeptic)"):
        audited = results.get("skeptic", {}).get("audited_findings", [])
        for af in audited:
            st.markdown("**%s**" % af.get("sub_claim", ""))
            accepted = af.get("accepted_sources", [])
            rejected = af.get("rejected_sources", [])
            if accepted:
                st.markdown("*Accepted:*")
                for a in accepted:
                    st.markdown("- %s (score: %s)" % (a.get("url", a.get("title", "?")), a.get("score", "?")))
            if rejected:
                st.markdown("*Rejected:*")
                for r in rejected:
                    st.markdown("- %s — %s" % (r.get("url", "?"), r.get("reason", "?")))
            contradictions = af.get("contradictions", [])
            if contradictions:
                st.markdown("*Contradictions:*")
                for c in contradictions:
                    st.markdown("- " + str(c))
