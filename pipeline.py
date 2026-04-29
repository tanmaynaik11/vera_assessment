import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

from input_classifier import classify
from guideline_evaluator import evaluate_record as _eval_guideline
from decomposer_A import decompose
from decomposer_B import check_omissions
from faithfulness_judge import judge_record as _judge_faithfulness
from safety_judge import judge_record as _judge_safety
from actionability_judge import judge_record as _judge_actionability
from final_verdict import generate_verdict


def run_record(record: dict, on_status=None) -> dict:
    def status(msg):
        if on_status:
            on_status(msg)

    query = record["input"]["question"]
    record_id = record["id"]

    status("Classifying query...")
    classification = classify(query)
    query_type = classification["classification"]

    if query_type == "GUIDELINE_ONLY":
        status("Running guideline evaluator (fetching references)...")
        eval_result = _eval_guideline(record)
        return {
            "id": record_id,
            "query": query,
            "query_type": "GUIDELINE_ONLY",
            "classification_confidence": classification["confidence"],
            "verdict": eval_result.get("verdict", ""),
            "verdict_text": eval_result.get("verdict_detail", ""),
            "scores": eval_result.get("scores", {}),
            "issues": eval_result.get("issues", []),
            "information_gaps": [],
            "claims": [],
        }

    # PATIENT_SPECIFIC pipeline
    status("Extracting claims (Decomposer A)...")
    a_result = decompose(record)

    status("Checking omissions (Decomposer B)...")
    b_result = check_omissions(a_result)

    status("Running faithfulness, safety, and actionability judges in parallel...")
    question_type = record.get("metadata", {}).get("questionType", "Unknown")

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_f = executor.submit(_judge_faithfulness, a_result, record)
        future_s = executor.submit(_judge_safety, a_result)
        future_a = executor.submit(_judge_actionability, query, record["output"], question_type)

        for future in as_completed([future_f, future_s, future_a]):
            name = {future_f: "faithfulness", future_s: "safety", future_a: "actionability"}[future]
            status(f"Judges running — {name} done")

    faithfulness = future_f.result()
    safety = future_s.result()
    actionability = future_a.result()

    status("Generating final verdict...")
    claim_lookup = {
        c["id"]: {
            "source_span": c.get("source_span", ""),
            "text": c.get("text", ""),
            "type": c.get("type", ""),
        }
        for c in a_result.get("claims", [])
    }
    verdict_result = generate_verdict(
        query, faithfulness, safety, actionability, claim_lookup, b_result
    )

    # Merge per-claim faithfulness + safety into a single claims array for audit
    faithfulness_by_id = {r["claim_id"]: r for r in faithfulness}
    safety_by_id = {r["claim_id"]: r for r in safety}
    merged_claims = [
        {
            **c,
            "faithfulness": faithfulness_by_id.get(c["id"], {}),
            "safety": safety_by_id.get(c["id"], {}),
        }
        for c in a_result.get("claims", [])
    ]

    return {
        "id": record_id,
        "query": query,
        "query_type": "PATIENT_SPECIFIC",
        "classification_confidence": classification["confidence"],
        "verdict": verdict_result.get("verdict", ""),
        "verdict_text": verdict_result.get("verdict_text", ""),
        "information_gaps": verdict_result.get("information_gaps", []),
        "programmatic_tier": verdict_result.get("programmatic_tier"),
        "faithfulness_issue_count": verdict_result.get("faithfulness_issue_count"),
        "safety_issue_count": verdict_result.get("safety_issue_count"),
        "high_gap_count": verdict_result.get("high_gap_count"),
        "actionability": actionability,
        "claims": merged_claims,
        "omissions": b_result,
    }


def run_pipeline(records: list[dict], on_progress=None, on_status=None) -> list[dict]:
    results = []
    for i, record in enumerate(records):
        if on_progress:
            on_progress(i, len(records))
        result = run_record(record, on_status=on_status)
        results.append(result)
    if on_progress:
        on_progress(len(records), len(records))
    return results
