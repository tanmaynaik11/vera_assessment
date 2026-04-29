import json
import os
import re
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PUBMED_FETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
)
SEMANTIC_SCHOLAR_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    "?fields=title,abstract,year,authors,externalIds"
)

EVALUATION_SYSTEM_PROMPT = """You are a clinical evidence evaluator auditing AI-generated medical guideline answers. You will be given:
1. The original clinical query
2. The AI-generated answer
3. Retrieved reference context fetched from the cited sources

Evaluate the answer on FOUR dimensions, each scored 1–5:

ACCURACY (1–5)
Does the answer correctly represent what the guidelines say?
5 = All claims are accurate, nothing overclaimed/underclaimed, attributions are correct
4 = Minor imprecision but nothing clinically dangerous
3 = One notable inaccuracy or misattribution
2 = Multiple inaccuracies or one significant clinical error
1 = Substantially wrong or contradicts the guideline

CURRENCY (1–5)
Is the primary grounding the most recent authoritative version of the relevant guideline?
5 = Latest guideline version is the primary source; explicitly supersedes older versions
4 = Mostly current but one key source could be replaced with a newer version
3 = Mix of current and outdated; no explicit acknowledgment of updates
2 = Primarily relying on guidelines that have since been superseded
1 = Outdated sources are the sole basis; no current guideline cited

COMPLETENESS (1–5)
Does the answer cover all major domains the guideline addresses, or are there silent omissions?
5 = All major prevention domains covered (antithrombotic, BP, lipid, lifestyle, cause-specific)
4 = All major domains covered, minor detail gaps only
3 = One entire domain missing or severely underaddressed
2 = Two or more domains missing
1 = Answer addresses only one narrow aspect of a multi-domain guideline

GROUNDEDNESS (1–5)
Is every factual claim in the answer traceable to one of the retrieved reference documents?
Work through each major claim and check whether the retrieved context actually supports it.
5 = Every specific claim (drug names, thresholds, trial names, effect sizes) is directly supported by the retrieved context with correct attribution
4 = Nearly all claims supported; one minor claim cannot be verified from retrieved context but is not contradicted
3 = Several claims are unverifiable from retrieved context, or one DOI citation points to a paper that does not contain the attributed claim
2 = Multiple claims float free of any retrieved document; citations are mismatched or misleading
1 = Answer is largely ungrounded — claims are asserted without support in any retrieved document, or directly contradict retrieved context

OUTPUT FORMAT — JSON only, no markdown fences:
{
  "scores": {
    "accuracy": <int 1-5>,
    "currency": <int 1-5>,
    "completeness": <int 1-5>,
    "groundedness": <int 1-5>,
    "overall": <float, weighted average: accuracy*0.35 + currency*0.25 + completeness*0.2 + groundedness*0.2>
  },
  "verdict": "<ship as is | ship with caveat: X | do not ship — Y is wrong>",
  "verdict_detail": "<3-5 sentences. State what is correct, what is wrong or missing, where the issue lives (which section/claim), and whether it is clinically significant.>",
  "issues": [
    {"dimension": "<accuracy|currency|completeness|groundedness>", "severity": "<critical|major|minor>", "location": "<section or claim>", "detail": "<what is wrong>"}
  ]
}"""

EVALUATION_USER_TEMPLATE = """QUERY: {query}

GENERATED ANSWER:
{answer}

RETRIEVED REFERENCE CONTEXT:
{context}

Now evaluate the generated answer against all four criteria. For groundedness, explicitly check whether the specific claims, thresholds, trial names, and effect sizes cited in the answer actually appear in the retrieved context above."""


def _extract_semantic_scholar_id(url: str) -> str | None:
    match = re.search(r"semanticscholar\.org/paper/([a-f0-9]+)", url)
    return match.group(1) if match else None


def _fetch_pubmed_abstract(pmid: str) -> str:
    try:
        resp = requests.get(PUBMED_FETCH_URL.format(pmid=pmid), timeout=10)
        resp.raise_for_status()
        text = resp.text.strip()
        # Trim to first 1500 chars to keep context manageable
        return text[:1500]
    except Exception as e:
        return f"[fetch failed: {e}]"


def _fetch_semantic_scholar_abstract(paper_id: str) -> str:
    try:
        resp = requests.get(
            SEMANTIC_SCHOLAR_URL.format(paper_id=paper_id),
            timeout=10,
            headers={"User-Agent": "vera-eval/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        title = data.get("title", "")
        year = data.get("year", "")
        abstract = data.get("abstract") or "[no abstract]"
        return f"Title: {title} ({year})\nAbstract: {abstract[:1200]}"
    except Exception as e:
        return f"[fetch failed: {e}]"


def _dois_cited_in_answer(answer: str) -> set[str]:
    return set(re.findall(r"doi:\s*([\w./\-]+)", answer, re.IGNORECASE))


def select_and_fetch_references(refs: list[dict], answer: str, max_refs: int = 6) -> str:
    cited_dois = _dois_cited_in_answer(answer)

    # Score refs: cited in answer > recent year > stroke/prevention keyword in title
    stroke_keywords = {"stroke", "tia", "ischemic", "cerebrovascular", "prevention", "antithrombotic"}

    def score_ref(ref):
        title_lower = ref.get("title", "").lower()
        is_cited = ref.get("doi", "") in cited_dois
        is_stroke = any(kw in title_lower for kw in stroke_keywords)
        year = ref.get("year") or 0
        return (is_cited * 10) + (is_stroke * 5) + (year / 100)

    ranked = sorted(refs, key=score_ref, reverse=True)[:max_refs]

    context_parts = []
    for ref in ranked:
        pmid = ref.get("pmid")
        url = ref.get("url", "")
        title = ref.get("title", "N/A")
        year = ref.get("year", "N/A")
        doi = ref.get("doi", "")

        header = f"--- [{year}] {title}"
        if doi:
            header += f" | doi:{doi}"

        if pmid:
            print(f"  Fetching PubMed {pmid}: {title[:60]}...")
            content = _fetch_pubmed_abstract(pmid)
            time.sleep(0.4)  # NCBI rate limit courtesy
        elif "semanticscholar" in url:
            ss_id = _extract_semantic_scholar_id(url)
            if ss_id:
                print(f"  Fetching SemanticScholar {ss_id[:12]}: {title[:60]}...")
                content = _fetch_semantic_scholar_abstract(ss_id)
                time.sleep(0.3)
            else:
                content = "[no abstract available]"
        else:
            content = "[no abstract available]"

        context_parts.append(f"{header}\n{content}")

    return "\n\n".join(context_parts)


def evaluate_record(record: dict) -> dict:
    query = record["input"]["question"]
    answer = record["output"]
    refs = record["metadata"].get("references", [])

    print(f"\nFetching references for: '{query}'")
    context = select_and_fetch_references(refs, answer, max_refs=6)

    print("\nRunning evaluation...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": EVALUATION_USER_TEMPLATE.format(
                query=query,
                answer=answer,
                context=context,
            )},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    result["id"] = record["id"]
    result["query"] = query
    return result


def print_evaluation(result: dict):
    scores = result["scores"]
    print("\n" + "=" * 70)
    print(f"EVALUATION: {result['query']}")
    print("=" * 70)
    print(f"  Accuracy     : {scores['accuracy']}/5")
    print(f"  Currency     : {scores['currency']}/5")
    print(f"  Completeness : {scores['completeness']}/5")
    print(f"  Groundedness : {scores['groundedness']}/5")
    print(f"  Overall      : {scores['overall']:.2f}/5")
    print()
    print(f"VERDICT: {result['verdict']}")
    print()
    print("DETAIL:")
    print(result["verdict_detail"])
    if result.get("issues"):
        print()
        print("ISSUES:")
        for issue in result["issues"]:
            print(f"  [{issue['severity'].upper()}] {issue['dimension']} — {issue['location']}")
            print(f"    {issue['detail']}")
    print("=" * 70)


if __name__ == "__main__":
    with open("vera_answers_extras.json", encoding="utf-8") as f:
        data = json.load(f)

    stroke_record = next(r for r in data if "stroke" in r["input"]["question"].lower())
    result = evaluate_record(stroke_record)
    print_evaluation(result)

    with open("evaluation_result_stroke.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nFull result saved to evaluation_result_stroke.json")
