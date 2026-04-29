import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a clinical claim extraction specialist for Vera, a medical decision-support system. Your job is to decompose a clinical answer into a structured list of atomic, independently verifiable claims.

You output JSON only. No preamble, no explanation, no markdown fences.

WHAT IS AN ATOMIC CLAIM
------------------------
An atomic claim makes exactly one verifiable assertion. It is self-contained — a reader can evaluate it without reading any other claim. It does not use pronouns that reference prior claims ("it", "they", "this drug"). It does not bundle two assertions into one sentence.

WRONG (bundled): "ACE inhibitors are preferred in CKD patients and reduce proteinuria."
RIGHT (atomic):  Claim 1: "ACE inhibitors are preferred over other antihypertensive classes in patients with CKD."
                 Claim 2: "ACE inhibitors reduce proteinuria in patients with CKD."

CLAIM TYPES
-----------
T1_THERAPY     — A recommendation to use, prefer, start, stop, or adjust a treatment for a specific patient or patient profile.
T2_DIAGNOSIS   — A claim about differential diagnosis, diagnostic probability, or test interpretation for a patient.
T3_GUIDELINE   — A statement of what a guideline, consensus body, or evidence base recommends, without patient-specific framing.
T4_DRUG        — A claim about drug mechanism, dosing, interaction, or pharmacology without patient-specific context.
T5_PROCEDURAL  — A claim about a diagnostic workup step, procedure, or monitoring requirement.
T6_THRESHOLD   — A specific numerical value or cutoff (lab value, score, dose, percentage, time window).
T7_CAUSAL      — A mechanistic or causal claim explaining why something works or happens.
T8_SAFETY      — A contraindication, adverse effect, prerequisite check, or monitoring requirement.

ONE CLAIM, ONE TYPE. If a claim could be T1 and T6, split it into two claims.

CONDITIONALITY RULE
--------------------
Every conditional qualifier in the source sentence MUST appear in the extracted claim.
Source: "ACE inhibitors are preferred when albuminuria is present."
WRONG extraction: "ACE inhibitors are preferred."
RIGHT extraction: "ACE inhibitors are preferred when albuminuria (UACR ≥30 mg/g) is present."

If the source sentence contains a qualifier ("when", "if", "in patients with", "unless", "except when"), that qualifier is not optional — it is part of the claim. Dropping it creates a more dangerous, more general claim than the source intended.

UNCERTAINTY FLAG
----------------
Mark uncertain: true if the claim contains hedging language ("may", "consider", "evidence suggests", "associated with") OR if the claim relies on a single small study (n < 100) OR if the clinical area is contested.

CITATION MAPPING
----------------
Map each claim to the DOI(s) cited in the answer for that specific assertion. If a claim has no citation in the answer, set citations to [] and flag citation_absent: true.

GRANULARITY GUIDANCE
--------------------
- One recommendation per claim. "Use X and monitor Y" → two claims.
- One threshold per claim. "Target BP <130/80 and HR <80" → two claims.
- Background/framing sentences that make no verifiable assertion → skip them.
- Do not extract the question restatement or introductory sentences.

OUTPUT FORMAT
-------------
Return a JSON object with this exact structure:
{
  "query": "...",
  "total_claims": N,
  "claims": [
    {
      "id": "C01",
      "text": "...",
      "type": "T1_THERAPY",
      "conditional": true,
      "condition_text": "when UACR ≥30 mg/g is present",
      "citations": ["10.xxxx/xxxxx"],
      "citation_absent": false,
      "uncertain": false,
      "uncertainty_reason": null,
      "source_span": "exact phrase from the answer this claim was extracted from"
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """QUERY: {query}

GENERATED ANSWER:
{answer}

Extract all atomic claims from this answer. Apply all rules strictly: split bundled claims, preserve every conditional qualifier, flag uncertainty, map citations, and note any claim with no citation in the answer."""


def decompose(record: dict) -> dict:
    query = record["input"]["question"]
    answer = record["output"]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                query=query,
                answer=answer,
            )},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    result["id"] = record["id"]
    return result


def print_decomposition(result: dict):
    claims = result.get("claims", [])
    type_counts = {}
    for c in claims:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1

    print("\n" + "=" * 72)
    print(f"DECOMPOSITION: {result['query']}")
    print(f"Record ID    : {result['id']}")
    print(f"Total claims : {result['total_claims']}")
    print(f"Type breakdown: {dict(sorted(type_counts.items()))}")
    print("=" * 72)

    for c in claims:
        uncertain_flag = " [UNCERTAIN]" if c.get("uncertain") else ""
        citation_flag = " [NO CITATION]" if c.get("citation_absent") else ""
        cond_flag = f"  condition: {c['condition_text']}" if c.get("conditional") else ""
        citations = ", ".join(c.get("citations", [])) or "—"

        print(f"\n{c['id']} [{c['type']}]{uncertain_flag}{citation_flag}")
        print(f"  {c['text']}")
        if cond_flag:
            print(f"  {cond_flag}")
        print(f"  citations : {citations}")
        if c.get("uncertainty_reason"):
            print(f"  uncertainty: {c['uncertainty_reason']}")
        print(f"  source    : \"{c.get('source_span', '')[:80]}\"")

    print("\n" + "=" * 72)


def decompose_all(dataset_path: str, patient_specific_ids: list[str] | None = None) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for record in data:
        if patient_specific_ids and record["id"] not in patient_specific_ids:
            continue
        print(f"\nDecomposing: {record['input']['question'][:70]}...")
        result = decompose(record)
        print_decomposition(result)
        results.append(result)

    return results


if __name__ == "__main__":
    # The 5 patient-specific record IDs (stroke is GUIDELINE_ONLY → handled by guideline_evaluator)
    PATIENT_SPECIFIC_IDS = [
        "5a8c1b3e-htn-dm-2026-04-27",
        "9c4d2f1a-uc-flare-2026-04-27",
        "7e9b3a2c-pe-ddx-2026-04-27",
        "8d2a7c4f-gerd-refractory-2026-04-27",
        "3a1f8e6b-afib-rvr-2026-04-27",
    ]

    results = decompose_all("vera_answers_extras.json", patient_specific_ids=PATIENT_SPECIFIC_IDS)

    with open("decomposition_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_claims = sum(r.get("total_claims", 0) for r in results)
    no_citation = sum(
        1 for r in results for c in r.get("claims", []) if c.get("citation_absent")
    )
    uncertain = sum(
        1 for r in results for c in r.get("claims", []) if c.get("uncertain")
    )

    print(f"\nSUMMARY ACROSS ALL PATIENT-SPECIFIC RECORDS")
    print(f"  Records processed : {len(results)}")
    print(f"  Total claims      : {total_claims}")
    print(f"  Uncited claims    : {no_citation}")
    print(f"  Uncertain claims  : {uncertain}")
    print(f"\nFull output saved to decomposition_results.json")
