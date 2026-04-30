import json
import logging
import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI
from judge_utils import build_doi_to_ref_map, fetch_sources_for_claim

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
log = logging.getLogger("vera.faithfulness")

SYSTEM_PROMPT = """You are a clinical faithfulness evaluator for Vera, a medical decision-support system. You receive an atomic clinical claim and the text of the source documents cited for that claim.

SOURCE PROVENANCE
-----------------
Each source is prefixed with either:
  [PMC targeted sections — Results + Discussion (≤8,000 chars) | PMCxxxxxx]
    — You are reading the actual Results and Discussion sections of the paper, trimmed to 8,000 characters. Apply strict NLI: contradictions and overclaiming can be assessed with HIGH confidence. Note that the 8,000-char trim may omit later passages — if the claim is plausible but the relevant passage is absent, prefer unsupported over contradicted.
  [PubMed abstract only]
    — You are reading only the abstract. Specific numerical values, effect sizes, and sub-group results may not appear in abstracts. Mark those as unsupported with MEDIUM confidence, not contradicted, unless the abstract itself explicitly states the opposite.

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

CONTRADICTED vs UNSUPPORTED — CRITICAL BOUNDARY
------------------------------------------------
contradicted: The source contains a quotable passage that directly negates what the claim asserts.
  Example — Claim: "Drug X reduces mortality." Source passage: "Drug X showed no reduction in mortality (HR 1.02, p=0.87)." → contradicted.
  You MUST be able to cite the specific passage that negates the claim. If you cannot, do not use contradicted.

unsupported: The source text does not contain evidence that confirms the claim. The source may simply be silent on this specific assertion, or the relevant detail may fall outside the extracted sections or abstract.
  Use unsupported when: the source never mentions the claim's specific assertion, the claim's detail is not covered, or the 8,000-char PMC trim may have excluded the relevant passage.

Do NOT use contradicted just because the source is silent or because the relevant passage is absent. Absence of evidence is unsupported, not contradicted. A contradicted verdict is a hard stop ("do not ship") — apply it only when you have a direct, quotable negation.

CONFIDENCE CALIBRATION
----------------------
HIGH   — PMC targeted sections available AND the relevant passage directly addresses the claim.
MEDIUM — Abstract only, OR PMC sections available but the specific claim detail is not covered in the extracted text.
LOW    — Source fetch failed or content is clearly insufficient to evaluate the claim.

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

Check the provenance prefix of each source before evaluating. Detect the pattern, then evaluate whether this claim is faithfully supported by the cited sources."""


def judge_claim(claim: dict, doi_to_ref: dict) -> dict:
    if claim.get("citation_absent") or not claim.get("citations"):
        return {
            "claim_id": claim["id"],
            "claim_text": claim["text"],
            "skipped": False,
            "verdict": "missing_citation",
            "source_type": "none",
            "reasoning": "Claim makes a verifiable assertion but the generated answer cited no source for it.",
        }

    t0 = time.perf_counter()
    sources = fetch_sources_for_claim(claim, doi_to_ref)

    # Build provenance map: doi → "PMC TARGETED SECTIONS" | "ABSTRACT ONLY" | "UNKNOWN"
    def _detect_provenance(content: str) -> str:
        if content.startswith("[PMC targeted sections"):
            return "PMC TARGETED SECTIONS"
        if content.startswith("[PubMed abstract only"):
            return "ABSTRACT ONLY"
        return "UNKNOWN"

    doi_to_provenance = {s["doi"]: _detect_provenance(s["content"]) for s in sources}

    for s in sources:
        prov = doi_to_provenance[s["doi"]]
        log.info(
            f"  [{claim['id']}] source [{s['doi']}] → {prov} ({len(s['content'])} chars)"
        )

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
    elapsed = time.perf_counter() - t0

    # Attach provenance to each per_source_verdict entry
    for psv in result.get("per_source_verdicts", []):
        psv["provenance"] = doi_to_provenance.get(psv.get("doi", ""), "UNKNOWN")

    # Summarise provenance at claim level
    prov_values = set(doi_to_provenance.values())
    if "FULL TEXT" in prov_values and "ABSTRACT ONLY" in prov_values:
        result["source_type"] = "mixed"
    elif "FULL TEXT" in prov_values:
        result["source_type"] = "full_text"
    elif "ABSTRACT ONLY" in prov_values:
        result["source_type"] = "abstract_only"
    else:
        result["source_type"] = "unknown"

    log.info(
        f"  [{claim['id']}] verdict={result.get('verdict','?').upper()} "
        f"conf={result.get('confidence','?')} pattern={result.get('pattern','?')} "
        f"source_type={result['source_type']} in {elapsed:.2f}s"
    )
    result["claim_id"] = claim["id"]
    result["claim_text"] = claim["text"]
    result["skipped"] = False
    return result


def judge_record(decomposer_a_record: dict, dataset_record: dict) -> list[dict]:
    doi_to_ref = build_doi_to_ref_map(dataset_record["metadata"].get("references", []))
    claims = decomposer_a_record["claims"]
    record_id = decomposer_a_record.get("id", "?")
    log.info(f"Faithfulness judge starting: record={record_id}, claims={len(claims)}")
    t_record = time.perf_counter()
    results = []
    for claim in claims:
        cited = claim.get("citations", [])
        if cited:
            log.debug(f"  [{claim['id']}] fetching {len(cited)} source(s): {cited}")
        result = judge_claim(claim, doi_to_ref)
        results.append(result)
    elapsed = time.perf_counter() - t_record
    verdict_summary = {r.get("verdict"): 0 for r in results}
    for r in results:
        verdict_summary[r.get("verdict")] = verdict_summary.get(r.get("verdict"), 0) + 1
    log.info(
        f"Faithfulness judge done: record={record_id} | {elapsed:.2f}s | "
        f"verdicts={verdict_summary}"
    )
    return results


def print_report(results: list[dict], query: str):
    missing_cit = [r for r in results if r.get("verdict") == "missing_citation"]
    judged = [r for r in results if not r.get("skipped") and r.get("verdict") != "missing_citation"]

    verdict_counts: dict[str, int] = {}
    for r in judged:
        v = r.get("verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    laundering = [r for r in judged if r.get("language_laundering")]

    print(f"\n{'=' * 72}")
    print(f"FAITHFULNESS: {query[:70]}")
    print(f"  Judged: {len(judged)}  |  Missing citation: {len(missing_cit)}")
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

    if missing_cit:
        print(f"\n  Missing citation (generated assertion with no cited source):")
        for r in missing_cit:
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
