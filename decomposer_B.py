import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a clinical safety reviewer for Vera, a medical decision-support system. You receive a patient profile, a clinical question, and a list of claims already present in an answer. Your job is to identify clinically important claims that are ABSENT from the answer — things a complete, safe, guideline-concordant response should contain but does not.

You output JSON only. No preamble, no explanation, no markdown fences.

YOUR TASK
---------
Generate a checklist of expected claims for this patient profile and question. For each expected claim, indicate whether it is PRESENT (found in the extracted claims list) or ABSENT. Flag absent claims with a severity level.

SEVERITY LEVELS
---------------
HIGH:   A clinician acting on this answer without this information could directly harm the patient. Examples: missing contraindication for the recommended drug, missing prerequisite lab before starting a medication, omitted must-not-miss diagnosis in a differential.

MEDIUM: The omission degrades care quality or could lead to suboptimal decisions but is unlikely to cause immediate direct harm. Examples: missing second-line option, missing monitoring requirement, missing relevant comorbidity consideration.

LOW:    The omission is a completeness gap that a thorough answer would include but whose absence does not materially affect the clinical decision. Examples: missing mechanistic explanation, missing epidemiological context.

EXPECTED CLAIM CATEGORIES TO CHECK
------------------------------------
For T1_THERAPY questions, always check for:
  - Prerequisite labs or assessments before starting the recommended drug class
  - Contraindications in the stated patient profile for each recommended drug
  - Metabolic or disease-specific adverse effects relevant to this patient's comorbidities
  - Monitoring requirements after initiation
  - Drug interaction flags if the patient profile implies polypharmacy

For T2_DIAGNOSIS questions, always check for:
  - Must-not-miss diagnoses that could present similarly
  - Time-critical diagnoses requiring immediate action
  - Risk stratification tool or score appropriate to the presentation

For T3_GUIDELINE questions, always check for:
  - Whether the answer covers all major domains the guideline addresses
  - Whether the most recent guideline version is represented

For T6_THRESHOLD claims, always check for:
  - Whether the threshold is stated with its qualifying condition (on-therapy vs off-therapy, specific population)

IMPORTANT: Do not flag a claim as absent if a semantically equivalent claim is present in the extracted list, even if the wording differs. Only flag genuine omissions.

OUTPUT FORMAT
-------------
{
  "expected_claims": [
    {
      "id": "EC01",
      "description": "...",
      "category": "prerequisite_lab | contraindication | adverse_effect | monitoring | must_not_miss | risk_score | guideline_domain | threshold_qualifier | drug_interaction",
      "status": "PRESENT | ABSENT",
      "present_in_claim_id": "C03",
      "severity": "HIGH | MEDIUM | LOW",
      "severity_rationale": "one sentence explaining why this severity",
      "patient_relevance": "why this is specifically relevant to THIS patient profile"
    }
  ],
  "high_severity_absent_count": N,
  "medium_severity_absent_count": N,
  "summary": "one sentence overall assessment of completeness"
}"""

USER_PROMPT_TEMPLATE = """PATIENT PROFILE & QUERY:
{query}

CLAIMS ALREADY PRESENT IN THE ANSWER (output of Decomposer A — treat these as the ground truth of what the answer contains):
{claims_block}

INSTRUCTIONS:
1. For each expected claim category relevant to this patient type and question domain, determine whether it is PRESENT or ABSENT by carefully reading every claim above.
2. A claim is PRESENT if any claim in the list above addresses the same clinical concept — even if the wording differs. Do not mark something ABSENT just because the exact phrase is not there.
3. Only mark ABSENT if no claim above covers that concept at all.
4. For ABSENT claims, only flag them if they could directly affect the physician's clinical decision or patient safety — not for completeness-only gaps.
5. Be specific: name the exact drug, lab, or clinical scenario relevant to THIS patient. Generic "check contraindications" without naming what is contraindicated for this patient does not count as a useful flag.

Generate the expected-claims checklist now."""


def _format_claims_block(claims: list[dict]) -> str:
    lines = []
    for c in claims:
        uncertain = " [UNCERTAIN]" if c.get("uncertain") else ""
        no_cite = " [NO CITATION]" if c.get("citation_absent") else ""
        cond = f" | condition: {c['condition_text']}" if c.get("conditional") else ""
        lines.append(f"{c['id']} [{c['type']}]{uncertain}{no_cite}: {c['text']}{cond}")
    return "\n".join(lines)


def check_omissions(decomposer_a_result: dict) -> dict:
    query = decomposer_a_result["query"]
    claims = decomposer_a_result.get("claims", [])
    claims_block = _format_claims_block(claims)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                query=query,
                claims_block=claims_block,
            )},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    result["id"] = decomposer_a_result["id"]
    result["query"] = query

    # Build a lookup of decomposer_A claims by ID so downstream can resolve references
    claims_by_id = {c["id"]: c for c in decomposer_a_result.get("claims", [])}

    # Enrich every PRESENT expected claim with the full decomposer_A claim it maps to
    for ec in result.get("expected_claims", []):
        if ec["status"] == "PRESENT" and ec.get("present_in_claim_id"):
            ref_id = ec["present_in_claim_id"]
            matched = claims_by_id.get(ref_id)
            if matched:
                ec["covered_by_claim"] = {
                    "id": matched["id"],
                    "type": matched["type"],
                    "text": matched["text"],
                    "citations": matched.get("citations", []),
                    "citation_absent": matched.get("citation_absent", False),
                    "uncertain": matched.get("uncertain", False),
                }

    result["decomposer_a_claims"] = list(claims_by_id.values())
    return result


def print_omission_report(result: dict):
    expected = result.get("expected_claims", [])
    absent = [e for e in expected if e["status"] == "ABSENT"]
    present = [e for e in expected if e["status"] == "PRESENT"]

    high = [e for e in absent if e["severity"] == "HIGH"]
    medium = [e for e in absent if e["severity"] == "MEDIUM"]
    low = [e for e in absent if e["severity"] == "LOW"]

    print("\n" + "=" * 72)
    print(f"OMISSION REPORT: {result['query']}")
    print(f"Record ID : {result['id']}")
    print(f"Checked   : {len(expected)} expected claims  |  "
          f"Present: {len(present)}  |  Absent: {len(absent)} "
          f"(HIGH: {len(high)}, MEDIUM: {len(medium)}, LOW: {len(low)})")
    print(f"Summary   : {result.get('summary', '')}")
    print("=" * 72)

    if absent:
        print("\nABSENT CLAIMS:")
        for e in absent:
            sev_label = {"HIGH": "!! HIGH", "MEDIUM": "   MED", "LOW": "   LOW"}[e["severity"]]
            print(f"\n  {sev_label} | {e['id']} [{e['category']}]")
            print(f"  {e['description']}")
            print(f"  Why relevant : {e['patient_relevance']}")
            print(f"  Why severity : {e['severity_rationale']}")

    if present:
        print("\nPRESENT (confirmed covered by Decomposer A):")
        for e in present:
            covered = e.get("covered_by_claim")
            if covered:
                uncertain_flag = " [UNCERTAIN]" if covered.get("uncertain") else ""
                no_cite_flag = " [NO CITATION]" if covered.get("citation_absent") else ""
                citations = ", ".join(covered.get("citations", [])) or "—"
                print(f"\n  ✓ {e['id']} [{e['category']}] → {covered['id']} [{covered['type']}]{uncertain_flag}{no_cite_flag}")
                print(f"    Expected : {e['description']}")
                print(f"    Covered by: \"{covered['text']}\"")
                print(f"    Citations : {citations}")
            else:
                claim_ref = f" → {e['present_in_claim_id']}" if e.get("present_in_claim_id") else ""
                print(f"\n  ✓ {e['id']} [{e['category']}]{claim_ref}")
                print(f"    Expected : {e['description']}")

    print("\n" + "=" * 72)


def run_all(decomposition_path: str) -> list[dict]:
    with open(decomposition_path, encoding="utf-8") as f:
        decomposer_a_results = json.load(f)

    results = []
    for record in decomposer_a_results:
        print(f"\nChecking omissions: {record['query'][:70]}...")
        result = check_omissions(record)
        print_omission_report(result)
        results.append(result)

    return results


if __name__ == "__main__":
    results = run_all("decomposition_results.json")

    with open("omission_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_absent = sum(len([e for e in r["expected_claims"] if e["status"] == "ABSENT"]) for r in results)
    total_high = sum(r.get("high_severity_absent_count", 0) for r in results)
    total_medium = sum(r.get("medium_severity_absent_count", 0) for r in results)

    print(f"\n{'=' * 72}")
    print("OVERALL OMISSION SUMMARY ACROSS ALL RECORDS")
    print(f"  Records reviewed        : {len(results)}")
    print(f"  Total absent claims     : {total_absent}")
    print(f"  HIGH severity absences  : {total_high}")
    print(f"  MEDIUM severity absences: {total_medium}")
    print(f"\nFull output saved to omission_results.json")
