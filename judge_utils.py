import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
import requests

sys.stdout.reconfigure(encoding="utf-8")

log = logging.getLogger("vera.judge_utils")

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

PUBMED_ABSTRACT_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
)
PMC_FULLTEXT_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pmc&id={pmcid_num}&rettype=full&retmode=xml"
)
PMCID_CONV_URL = (
    "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    "?ids={pmid}&format=json&tool=vera-eval"
)
SEMANTIC_SCHOLAR_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    "?fields=title,abstract,year"
)

# Sections most relevant for faithfulness checking
_TARGET_SECTION_TYPES = {
    "results", "result", "findings",
    "discussion", "conclusions", "conclusion",
    "results and discussion", "discussion and conclusions",
}

# Per-paper character budget sent to the faithfulness judge
# (~2 000 tokens at gpt-4o tokenisation; comfortably fits 3 papers in one call)
_FULLTEXT_CHAR_BUDGET = 8_000

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_abstract_cache: dict[str, str] = {}   # pmid / s2-paper-id  → abstract text
_pmcid_cache: dict[str, str | None] = {}  # pmid → PMCID string or None
_fulltext_cache: dict[str, str] = {}   # pmid → extracted sections text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pmid_to_pmcid(pmid: str) -> str | None:
    """Return the PMC ID for a PubMed ID, or None if not in PMC."""
    if pmid in _pmcid_cache:
        log.debug(f"PMCID cache hit: PMID {pmid} → {_pmcid_cache[pmid]}")
        return _pmcid_cache[pmid]
    pmcid = None
    try:
        resp = requests.get(PMCID_CONV_URL.format(pmid=pmid), timeout=10)
        resp.raise_for_status()
        records = resp.json().get("records", [])
        pmcid = records[0].get("pmcid") if records else None
    except Exception as e:
        log.warning(f"PMCID lookup failed for PMID {pmid}: {e}")
    _pmcid_cache[pmid] = pmcid
    result_label = pmcid if pmcid else "not in PMC"
    log.debug(f"PMCID lookup: PMID {pmid} → {result_label}")
    time.sleep(0.35)
    return pmcid


def _xml_to_text(element: ET.Element) -> str:
    """Recursively extract plain text from an XML element."""
    parts = []
    if element.text:
        parts.append(element.text.strip())
    for child in element:
        child_text = _xml_to_text(child)
        if child_text:
            parts.append(child_text)
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


def _is_target_section(sec: ET.Element) -> bool:
    """Return True if this <sec> element is a Results/Discussion/Conclusion section."""
    sec_type = (sec.get("sec-type") or "").lower().strip()
    if any(t in sec_type for t in _TARGET_SECTION_TYPES):
        return True
    title_el = sec.find("title")
    if title_el is not None:
        title = (title_el.text or "").lower().strip()
        if any(t in title for t in _TARGET_SECTION_TYPES):
            return True
    return False


def _extract_target_sections(xml_text: str) -> str:
    """Parse PMC JATS XML and return text from Results/Discussion sections."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.warning("PMC XML parse error — falling back to abstract")
        return ""

    parts = []
    found_sections = []
    for sec in root.iter("sec"):
        if _is_target_section(sec):
            title_el = sec.find("title")
            label = title_el.text if title_el is not None else "Section"
            text = _xml_to_text(sec)
            if text:
                parts.append(f"[{label}]\n{text}")
                found_sections.append(label)

    combined = "\n\n".join(parts)[:_FULLTEXT_CHAR_BUDGET]
    if found_sections:
        log.debug(f"Sections extracted: {found_sections} → {len(combined)} chars")
    else:
        log.debug("No target sections found in PMC XML")
    return combined


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_pubmed(pmid: str) -> str:
    """Fetch PubMed abstract (kept for backward compatibility / fallback)."""
    if pmid in _abstract_cache:
        return _abstract_cache[pmid]
    try:
        resp = requests.get(PUBMED_ABSTRACT_URL.format(pmid=pmid), timeout=10)
        resp.raise_for_status()
        text = resp.text.strip()[:1800]
    except Exception as e:
        text = f"[fetch failed: {e}]"
    _abstract_cache[pmid] = text
    time.sleep(0.35)
    return text


def fetch_pubmed_fulltext(pmid: str) -> str:
    """Fetch PMC full text (Results + Discussion); fall back to abstract.

    Returns a string prefixed with [PMC targeted sections — ...] or
    [PubMed abstract only] so the faithfulness judge knows the provenance
    of the content.
    """
    if pmid in _fulltext_cache:
        log.debug(f"Fulltext cache hit: PMID {pmid}")
        return _fulltext_cache[pmid]

    t0 = time.perf_counter()
    pmcid = _pmid_to_pmcid(pmid)

    if pmcid:
        pmcid_num = pmcid.replace("PMC", "")
        try:
            resp = requests.get(
                PMC_FULLTEXT_URL.format(pmcid_num=pmcid_num),
                timeout=20,
            )
            resp.raise_for_status()
            sections = _extract_target_sections(resp.text)
            if sections:
                elapsed = time.perf_counter() - t0
                log.info(
                    f"PMID {pmid} → {pmcid} | FULL TEXT ({len(sections)} chars) "
                    f"in {elapsed:.2f}s"
                )
                result = f"[PMC targeted sections — Results + Discussion (≤8,000 chars) | {pmcid}]\n\n{sections}"
                _fulltext_cache[pmid] = result
                time.sleep(0.35)
                return result
            else:
                log.info(f"PMID {pmid} → {pmcid} | PMC XML had no target sections → falling back to abstract")
        except Exception as e:
            log.warning(f"PMID {pmid} → {pmcid} | Full text fetch failed: {e} → falling back to abstract")
        time.sleep(0.35)
    else:
        log.info(f"PMID {pmid} | Not in PMC OA → abstract only")

    # Fallback: PubMed abstract
    abstract = fetch_pubmed(pmid)
    elapsed = time.perf_counter() - t0
    log.info(f"PMID {pmid} | ABSTRACT ONLY ({len(abstract)} chars) in {elapsed:.2f}s")
    result = f"[PubMed abstract only]\n\n{abstract}"
    _fulltext_cache[pmid] = result
    return result


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
            content = fetch_pubmed_fulltext(pmid)   # full text with abstract fallback
        elif "semanticscholar" in url:
            content = fetch_semantic_scholar(url)   # abstract only — no S2 full text API
        else:
            content = "[no abstract available]"

        sources.append({"doi": doi, "title": title, "year": year, "content": content})
    return sources
