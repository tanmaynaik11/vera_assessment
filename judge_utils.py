import re
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

PUBMED_FETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
)
SEMANTIC_SCHOLAR_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    "?fields=title,abstract,year"
)

_abstract_cache: dict[str, str] = {}


def fetch_pubmed(pmid: str) -> str:
    if pmid in _abstract_cache:
        return _abstract_cache[pmid]
    try:
        resp = requests.get(PUBMED_FETCH_URL.format(pmid=pmid), timeout=10)
        resp.raise_for_status()
        text = resp.text.strip()[:1800]
    except Exception as e:
        text = f"[fetch failed: {e}]"
    _abstract_cache[pmid] = text
    time.sleep(0.35)
    return text


def fetch_semantic_scholar(url: str) -> str:
    match = re.search(r"semanticscholar\.org/paper/([a-f0-9]+)", url)
    if not match:
        return "[no abstract available]"
    paper_id = match.group(1)
    if paper_id in _abstract_cache:
        return _abstract_cache[paper_id]
    try:
        resp = requests.get(
            SEMANTIC_SCHOLAR_URL.format(paper_id=paper_id),
            timeout=10,
            headers={"User-Agent": "vera-eval/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        abstract = data.get("abstract") or "[no abstract]"
        text = f"Title: {data.get('title', '')} ({data.get('year', '')})\n{abstract[:1500]}"
    except Exception as e:
        text = f"[fetch failed: {e}]"
    _abstract_cache[paper_id] = text
    time.sleep(0.3)
    return text


def build_doi_to_ref_map(metadata_refs: list[dict]) -> dict[str, dict]:
    return {r["doi"]: r for r in metadata_refs if r.get("doi")}


def fetch_sources_for_claim(claim: dict, doi_to_ref: dict) -> list[dict]:
    sources = []
    for doi in claim.get("citations", []):
        ref = doi_to_ref.get(doi, {})
        pmid = ref.get("pmid")
        url = ref.get("url", "")
        title = ref.get("title", doi)
        year = ref.get("year", "")

        if pmid:
            content = fetch_pubmed(pmid)
        elif "semanticscholar" in url:
            content = fetch_semantic_scholar(url)
        else:
            content = "[no abstract available]"

        sources.append({"doi": doi, "title": title, "year": year, "content": content})
    return sources
