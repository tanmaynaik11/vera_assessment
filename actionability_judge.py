import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a clinical actionability evaluator for Vera, a medical decision-support system. Evaluate a complete clinical answer on four dimensions for the specific clinical context stated in the question.

DIMENSIONS (each scored 1–5)
-----------------------------
CONTEXT_CALIBRATION
Is the response shaped for the clinical setting implied by the question (ED vs outpatient vs inpatient vs acute)?
5 = Explicitly calibrated to the setting — urgency, resources, and timeframes match the stated context
3 = Setting-agnostic — applicable anywhere but lacks setting-specific prioritisation
1 = Calibrated for the wrong setting (e.g. outpatient framing for an acute ED presentation)

DECISION_CLARITY
Can a clinician identify what to do first within 30 seconds of reading?
5 = Immediate action is unambiguous and appears at the top of the response
3 = Action is identifiable but buried or requires reading the full response before the first step is clear
1 = No clear first action — clinician must synthesise before acting

ACUITY_MATCHING
Does the response treat must-not-miss diagnoses and time-critical decisions with appropriate urgency?
5 = Time-critical items explicitly flagged, sequenced first, distinguished from routine items
3 = Urgent items are present but not structurally or visually distinguished from non-urgent ones
1 = Urgent items are missing or given equal weight to routine management

COGNITIVE_LOAD
Is the answer structured to support rather than overwhelm clinical reasoning under pressure?
5 = Clear hierarchy, progressive disclosure, scannable — key facts findable in under 10 seconds
3 = Structured but dense; requires linear reading to extract key decisions
1 = Unstructured wall of text that would impede decision-making

OVERALL VERDICT
---------------
actionable — overall ≥ 3.5 AND no dimension < 2
degraded   — overall 2.5–3.4 OR one dimension < 2
unusable   — overall < 2.5 OR any dimension = 1

OUTPUT FORMAT — JSON only, no markdown fences:
{
  "scores": {
    "context_calibration": <int 1-5>,
    "decision_clarity": <int 1-5>,
    "acuity_matching": <int 1-5>,
    "cognitive_load": <int 1-5>,
    "overall": <float, average of the four dimensions>
  },
  "verdict": "actionable | degraded | unusable",
  "dimension_notes": {
    "context_calibration": "one sentence",
    "decision_clarity": "one sentence",
    "acuity_matching": "one sentence",
    "cognitive_load": "one sentence"
  },
  "verdict_detail": "2-3 sentences: what makes this answer actionable/degraded/unusable, what specific change would move it to the next tier."
}"""

USER_TEMPLATE = """CLINICAL QUESTION (patient profile + implied setting): {query}
QUESTION TYPE: {question_type}

COMPLETE CLINICAL ANSWER:
{answer}

Evaluate this answer on all four actionability dimensions."""


def judge_record(query: str, answer: str, question_type: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                query=query,
                question_type=question_type,
                answer=answer,
            )},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def print_report(result: dict, query: str):
    scores = result.get("scores", {})
    verdict = result.get("verdict", "?")
    symbol = {"actionable": "✓", "degraded": "~", "unusable": "✗"}.get(verdict, "?")
    notes = result.get("dimension_notes", {})

    print(f"\n{'=' * 72}")
    print(f"ACTIONABILITY: {query[:70]}")
    print(f"  {symbol} {verdict.upper()}  |  Overall: {scores.get('overall', '?'):.1f}/5")
    print()
    dims = [
        ("context_calibration", "Context Calibration"),
        ("decision_clarity",    "Decision Clarity   "),
        ("acuity_matching",     "Acuity Matching    "),
        ("cognitive_load",      "Cognitive Load     "),
    ]
    for key, label in dims:
        score = scores.get(key, "?")
        note = notes.get(key, "")
        bar = "█" * int(score) + "░" * (5 - int(score)) if isinstance(score, int) else "?"
        print(f"  {label}: {score}/5 [{bar}]")
        print(f"    {note}")
    print()
    print(f"  Verdict: {result.get('verdict_detail', '')}")
    print("=" * 72)


if __name__ == "__main__":
    with open("decomposition_results.json", encoding="utf-8") as f:
        decomposer_a_results = json.load(f)
    with open("vera_answers_extras.json", encoding="utf-8") as f:
        dataset = {r["id"]: r for r in json.load(f)}

    all_results = {}
    for record in decomposer_a_results:
        rid = record["id"]
        query = record["query"]
        ds = dataset[rid]
        question_type = ds["metadata"].get("questionType", "Unknown")
        answer = ds["output"]

        print(f"\nRunning actionability judge: {query[:60]}...")
        result = judge_record(query, answer, question_type)
        print_report(result, query)
        all_results[rid] = result

    with open("actionability_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nSaved to actionability_results.json")
