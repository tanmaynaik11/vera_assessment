import json
import time
from pathlib import Path

import requests
import streamlit as st

BACKEND = "http://localhost:8000"

VERDICT_SYMBOL = {"ship as is": "✓", "ship with caveat": "~", "do not ship": "✗"}
SEV_ICON = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _post_evaluate(file_bytes: bytes, filename: str) -> dict:
    resp = requests.post(
        f"{BACKEND}/evaluate",
        files={"file": (filename, file_bytes, "application/json")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _get_status(job_id: str) -> dict:
    resp = requests.get(f"{BACKEND}/status/{job_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_result(job_id: str) -> list:
    resp = requests.get(f"{BACKEND}/result/{job_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _list_runs() -> list[str]:
    try:
        resp = requests.get(f"{BACKEND}/runs", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def _load_run(filename: str) -> list:
    resp = requests.get(f"{BACKEND}/runs/{filename}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def _backend_alive() -> bool:
    try:
        return requests.get(f"{BACKEND}/", timeout=3).status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def _symbol(verdict: str) -> str:
    for k, s in VERDICT_SYMBOL.items():
        if verdict.startswith(k):
            return s
    return "?"


def _verdict_badge(verdict: str) -> str:
    if verdict.startswith("ship as is"):
        color = "green"
    elif verdict.startswith("ship with caveat"):
        color = "orange"
    elif verdict.startswith("do not ship"):
        color = "red"
    else:
        color = "gray"
    sym = _symbol(verdict)
    return f":{color}[**{sym} {verdict}**]"


def _render_patient_specific(r: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Programmatic Tier", r.get("programmatic_tier", "—"))
    c2.metric("Faithfulness Issues", r.get("faithfulness_issue_count", "—"),
              help="Abstract-only — lower confidence")
    c3.metric("Safety Issues", r.get("safety_issue_count", "—"))
    c4.metric("High Info Gaps", r.get("high_gap_count", "—"))

    if r.get("information_gaps"):
        st.markdown("**Information gaps (requires more info before answer is complete):**")
        for gap in r["information_gaps"]:
            st.markdown(f"- {gap}")

    if r.get("claims"):
        st.markdown("**Claim-level evaluation:**")
        rows = []
        for c in r["claims"]:
            f = c.get("faithfulness", {})
            s = c.get("safety", {})
            rows.append({
                "ID": c["id"],
                "Type": c["type"],
                "Claim": c["text"][:90],
                "Source Span": c.get("source_span", "")[:60],
                "Faithfulness": f.get("verdict", "skipped") if not f.get("skipped") else "skipped",
                "F. Confidence": f.get("confidence", "—"),
                "Safety": s.get("verdict", "—"),
                "Severity": (s.get("severity") or "—").upper(),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    absent = [
        e for e in (r.get("omissions") or {}).get("expected_claims", [])
        if e.get("status") == "ABSENT"
    ]
    if absent:
        st.markdown("**Absent expected claims:**")
        for e in absent:
            icon = SEV_ICON.get(e.get("severity", ""), "⚪")
            st.markdown(
                f"{icon} **[{e.get('severity','?')}]** `{e.get('category','?')}` — {e.get('description','')}"
            )

    act = r.get("actionability", {})
    if act:
        scores = act.get("scores", {})
        with st.expander("Actionability detail"):
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Context Calibration", f"{scores.get('context_calibration','?')}/5")
            a2.metric("Decision Clarity", f"{scores.get('decision_clarity','?')}/5")
            a3.metric("Acuity Matching", f"{scores.get('acuity_matching','?')}/5")
            a4.metric("Cognitive Load", f"{scores.get('cognitive_load','?')}/5")
            st.caption(act.get("verdict_detail", ""))


def _render_guideline(r: dict):
    scores = r.get("scores", {})
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Accuracy", f"{scores.get('accuracy','?')}/5")
    g2.metric("Currency", f"{scores.get('currency','?')}/5")
    g3.metric("Completeness", f"{scores.get('completeness','?')}/5")
    g4.metric("Groundedness", f"{scores.get('groundedness','?')}/5")

    if r.get("issues"):
        st.markdown("**Issues:**")
        for issue in r["issues"]:
            icon = SEV_ICON.get(issue.get("severity", "").upper(), "⚪")
            st.markdown(
                f"{icon} **[{issue.get('severity','?').upper()}]** "
                f"`{issue.get('dimension','?')}` — {issue.get('location','?')}: {issue.get('detail','')}"
            )


def render_results(results: list[dict]):
    total = len(results)
    ship = sum(1 for r in results if r.get("verdict", "").startswith("ship as is"))
    caveat = sum(1 for r in results if r.get("verdict", "").startswith("ship with caveat"))
    no_ship = sum(1 for r in results if r.get("verdict", "").startswith("do not ship"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", total)
    m2.metric("✓ Ship As Is", ship)
    m3.metric("~ Ship With Caveat", caveat)
    m4.metric("✗ Do Not Ship", no_ship)

    st.divider()

    for r in results:
        if r.get("verdict") == "error":
            with st.expander(f"⚠ ERROR — {r.get('query','')[:75]}"):
                st.error(r.get("error", "Unknown error"))
            continue

        sym = _symbol(r.get("verdict", ""))
        label = f"{sym}  [{r.get('query_type','?')}]  {r.get('query','')[:75]}"
        with st.expander(label):
            st.markdown(_verdict_badge(r.get("verdict", "")))
            st.markdown(r.get("verdict_text", ""))
            st.divider()
            if r.get("query_type") == "PATIENT_SPECIFIC":
                _render_patient_specific(r)
            else:
                _render_guideline(r)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Vera Evaluation Pipeline", layout="wide",
                   initial_sidebar_state="expanded")
st.title("Vera Evaluation Pipeline")
st.caption("Classify → evaluate → audit every clinical answer in your dataset")

# Backend health check
if not _backend_alive():
    st.error(
        "Backend is not running. Start it with:\n\n"
        "```\nvenv/Scripts/uvicorn backend:app --reload\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Upload + trigger
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload dataset JSON", type="json")

if uploaded_file:
    # Cache raw bytes — st.file_uploader can't be re-read across reruns
    if st.session_state.get("_upload_name") != uploaded_file.name:
        st.session_state["_upload_bytes"] = uploaded_file.read()
        st.session_state["_upload_name"] = uploaded_file.name
        st.session_state.pop("job_id", None)
        st.session_state.pop("results", None)

    try:
        record_count = len(json.loads(st.session_state["_upload_bytes"]))
        st.success(f"Loaded **{record_count}** records from `{uploaded_file.name}`")
    except Exception:
        st.error("File is not valid JSON.")
        st.stop()

    if "job_id" not in st.session_state:
        if st.button("Run Evaluation", type="primary"):
            try:
                resp = _post_evaluate(
                    st.session_state["_upload_bytes"],
                    st.session_state["_upload_name"],
                )
                st.session_state["job_id"] = resp["job_id"]
                st.session_state["job_total"] = resp["total"]
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start job: {e}")

# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------

if "job_id" in st.session_state and "results" not in st.session_state:
    job_id = st.session_state["job_id"]
    total = st.session_state.get("job_total", 1)

    try:
        status = _get_status(job_id)
    except Exception as e:
        st.error(f"Lost contact with backend: {e}")
        st.stop()

    progress = status["progress"]
    pct = min(progress / total, 1.0) if total > 0 else 0.0

    st.progress(pct, text=f"Record {progress} of {total}")
    st.caption(
        f"**{status['current_query']}** → {status['current_step']}"
        if status["current_query"] else "Starting..."
    )

    if status["status"] == "done":
        try:
            results = _get_result(job_id)
        except Exception as e:
            st.error(f"Failed to fetch results: {e}")
            st.stop()
        st.session_state["results"] = results
        if status.get("run_file"):
            st.session_state["last_run_file"] = status["run_file"]
        st.rerun()
    else:
        time.sleep(2)
        st.rerun()

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if "results" in st.session_state:
    if st.session_state.get("last_run_file"):
        st.success(f"Audit saved to `runs/{st.session_state['last_run_file']}`")

    audit_json = json.dumps(st.session_state["results"], indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Audit JSON",
        data=audit_json,
        file_name=st.session_state.get("last_run_file", "vera_audit.json"),
        mime="application/json",
    )

    st.divider()
    render_results(st.session_state["results"])

# ---------------------------------------------------------------------------
# Sidebar: past runs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Past Runs")
    run_files = _list_runs()
    if run_files:
        selected = st.selectbox(
            "Select a run",
            options=run_files,
            format_func=lambda f: f.replace("run_", "").replace(".json", ""),
        )
        if st.button("Load Selected Run"):
            try:
                past = _load_run(selected)
                st.session_state["results"] = past
                st.session_state["last_run_file"] = selected
                st.session_state.pop("job_id", None)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load run: {e}")
    else:
        st.info("No past runs yet.")
