import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from judge_utils import build_doi_to_ref_map, fetch_sources_for_claim

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a clinical faithfulness evaluator for Vera, a medical decision-support system. You receive an atomic clinical claim and the text of the source documents cited for that claim.

PATTERN DETECTION
-----------------
SINGLE_SOURCE: Exactly one source provided. Apply standard NLI entailment.
CORROBORATION: Multiple sources address the same aspect of the claim. Check for language laundering — sources say "may reduce" but the claim says "reduces".
SYNTHESIS: Multiple sources address different aspects of the claim. Check whether the combination of sources explicitly authorises the full claim as written.

VERDICT OPTIONS
---------------
SINGLE_SOURCE → supported | unsupported | contradicted | overclaimed
CORROBORATION → per-source verdicts + language_laundering flag. Overall verdict is the weakest per-source verdict.
SYNTHESIS     → supported | unauthorised_synthesis | contradicted

OUTPUT FORMAT — JSON only, no markdown fences:
{
  "pattern": "SINGLE_SOURCE | CORROBORATION | SYNTHESIS",
  "verdict": "supported | unsupported | contradicted | overclaimed | unauthorised_synthesis",
  "per_source_verdicts": [
    {"doi": "...", "verdict": "supported | unsupported | contradicted | overclaimed", "reasoning": "one sentence"}
  ],
  "language_laundering": false,
  "language_laundering_detail": null,
  "reasoning": "2-3 sentences explaining the overall verdict",
  "confidence": "HIGH | MEDIUM | LOW"
}"""

USER_TEMPLATE = """CLAIM: {claim_text}
CLAIM TYPE: {claim_type}
CONDITIONAL QUALIFIER: {conditional}

CITED SOURCE DOCUMENTS:
{sources_block}

Detect the pattern, then evaluate whether this claim is faithfully supported by the cited sources."""


def judge_claim(claim: dict, doi_to_ref: dict) -> dict:
    if claim.get("citation_absent") or not claim.get("citations"):
        return {
            "claim_id": claim["id"],
            "claim_text": claim["text"],
            "skipped": True,
            "reason": "no_citations",
            "verdict": None,
        }

    sources = fetch_sources_for_claim(claim, doi_to_ref)
    sources_block = "\n\n".join(
        f"--- [{s['doi']}] {s['title']} ({s['year']})\n{s['content']}"
        for s in sources
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                claim_text=claim["text"],
                claim_type=claim["type"],
                conditional=claim.get("condition_text") or "none",
                sources_block=sources_block,
            )},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)
    result["claim_id"] = claim["id"]
    result["claim_text"] = claim["text"]
    result["skipped"] = False
    return result


def judge_record(decomposer_a_record: dict, dataset_record: dict) -> list[dict]:
    doi_to_ref = build_doi_to_ref_map(dataset_record["metadata"].get("references", []))
    claims = decomposer_a_record["claims"]
    results = []
    for claim in claims:
        cited = claim.get("citations", [])
        if cited:
            print(f"  [{claim['id']}] fetching {len(cited)} source(s)...")
        result = judge_claim(claim, doi_to_ref)
        results.append(result)
    return results


def print_report(results: list[dict], query: str):
    skipped = [r for r in results if r.get("skipped")]
    judged = [r for r in results if not r.get("skipped")]

    verdict_counts: dict[str, int] = {}
    for r in judged:
        v = r.get("verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    laundering = [r for r in judged if r.get("language_laundering")]

    print(f"\n{'=' * 72}")
    print(f"FAITHFULNESS: {query[:70]}")
    print(f"  Judged: {len(judged)}  |  Skipped (no citation): {len(skipped)}")
    print(f"  Verdicts: {verdict_counts}")
    if laundering:
        print(f"  !! Language laundering in {len(laundering)} claim(s)")
    print()

    for r in judged:
        verdict = r.get("verdict", "?")
        pattern = r.get("pattern", "?")
        conf = r.get("confidence", "?")
        ll_flag = "  !! LANGUAGE LAUNDERING" if r.get("language_laundering") else ""
        marker = "  " if verdict == "supported" else "!!"
        print(f"  {marker} [{r['claim_id']}] {verdict.upper()} ({pattern}, {conf}){ll_flag}")
        if verdict != "supported":
            print(f"       {r.get('reasoning', '')[:120]}")
            for psv in r.get("per_source_verdicts", []):
                if psv.get("verdict") != "supported":
                    print(f"       source [{psv['doi']}]: {psv['verdict']} — {psv.get('reasoning', '')[:80]}")
            if r.get("language_laundering_detail"):
                print(f"       laundering: {r['language_laundering_detail'][:100]}")

    if skipped:
        print(f"\n  Skipped (no citations):")
        for r in skipped:
            print(f"    [{r['claim_id']}] {r['claim_text'][:80]}")
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
        print(f"\nRunning faithfulness judge: {query[:60]}...")
        results = judge_record(record, dataset[rid])
        print_report(results, query)
        all_results[rid] = results

    with open("faithfulness_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nSaved to faithfulness_results.json")
