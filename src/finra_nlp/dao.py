"""Labels from full case documents in FINRA Disciplinary Actions Online (DAO).

Recent monthly summaries often omit rule numbers (the July 2026 report cites them in a
minority of cases), so labels come from the AWC, complaint or decision, where the
violations section names each rule.

The DAO page layout is not verified yet. Run `probe` on one case first and confirm it
finds the document link before running `labels` on everything.

Usage:
    uv run python -m finra_nlp.dao probe --case 2023077018401
    uv run python -m finra_nlp.dao labels --db data/finra.duckdb
"""

from __future__ import annotations

import argparse
import re
import time
from urllib.parse import urljoin

import duckdb
import pandas as pd
from bs4 import BeautifulSoup

from finra_nlp.citations import extract_citations
from finra_nlp.http import PoliteClient
from finra_nlp.monthly import pdf_text

SEARCH_URL = "https://www.finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions?search={case}"
VIOLATION_RE = re.compile(
    r"[^.]*\b(?:violated|violation|violations|in contravention of|constitut\w+)\b[^.]*\.", re.IGNORECASE
)


def document_links(html: bytes, page_url: str, case_no: str) -> list[str]:
    """PDFs whose file name or link text carries the case number. The search is full-text: a case
    that isn't in DAO still returns a page of other cases' documents, so nothing else is taken."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        if href.lower().endswith(".pdf") and (case_no in href or case_no in a.get_text()):
            links.append(href)
    return list(dict.fromkeys(links))


WAIVER_SECTION_RE = re.compile(r"WAIVER\s+OF\s+PROCEDURAL\s+RIGHTS", re.IGNORECASE)
WAIVE_RE = re.compile(r"\bwaive", re.IGNORECASE)


def labels_from_document(text: str) -> list[str]:
    """Citation keys from sentences that state violations, in order of first appearance.

    AWCs end with a procedural-rights waiver that mentions Rules 9143 and 9144 next to
    the word "violated"; text from that heading on is dropped, as is any sentence that waives.
    """
    text = re.sub(r"\s+", " ", text)
    cut = WAIVER_SECTION_RE.search(text)
    if cut:
        text = text[: cut.start()]
    keys: dict[str, None] = {}
    for sent in VIOLATION_RE.finditer(text):
        if WAIVE_RE.search(sent.group(0)):
            continue
        for c in extract_citations(sent.group(0)):
            keys[c.key] = None
    return list(keys)


def probe(client: PoliteClient, case_no: str) -> None:
    url = SEARCH_URL.format(case=case_no)
    status, html = client.get(url, use_cache=False)
    print(f"{status} {url} ({len(html)} bytes)")
    links = document_links(html, url, case_no)
    print("document links:", links or "NONE FOUND - page may be rendered by JavaScript")
    if links:
        s, pdf = client.get(links[0])
        text = pdf_text(pdf)
        print(f"{s} {links[0]} ({len(text)} chars)")
        print("labels:", labels_from_document(text))


def build_labels(client: PoliteClient, db: str, limit: int | None, pdf_client: PoliteClient | None = None,
                 every: int = 10) -> None:
    """pdf_client fetches the documents; they're static files, so they can go faster than the search
    page, which is what draws 429s. Prints progress every `every` cases."""
    pdf_client = pdf_client or client
    con = duckdb.connect(db)
    case_nos = [r[0] for r in con.execute("SELECT case_no FROM raw.monthly_cases ORDER BY case_no").fetchall()]
    if limit:
        case_nos = case_nos[:limit]
    cached = sum(client.cached(SEARCH_URL.format(case=c)) for c in case_nos)
    print(f"{len(case_nos)} cases, {cached} searches already cached (those go fast)", flush=True)
    rows, start, found = [], time.monotonic(), 0
    for i, case_no in enumerate(case_nos, 1):
        url = SEARCH_URL.format(case=case_no)
        status, html = client.get(url)
        if status != 200:  # retried by the client and not cached; a rerun tries again
            print(f"  {case_no}: search returned {status}", flush=True)
        links = document_links(html, url, case_no) if status == 200 else []
        keys, doc_url = [], None
        for link in links:
            s, pdf = pdf_client.get(link)
            if s == 200:
                keys, doc_url = labels_from_document(pdf_text(pdf)), link
                if keys:
                    break
        found += bool(keys)
        rows.append({"case_no": case_no, "doc_url": doc_url, "doc_rules": "|".join(keys), "n_docs": len(links),
                     "search_status": status})
        if i % every == 0 or i == len(case_nos):
            took = time.monotonic() - start
            left = (len(case_nos) - i) * took / i
            print(f"  {i}/{len(case_nos)} cases, {found} labeled | {took / 60:.0f} min so far, about "
                  f"{left / 3600:.1f} h left at this pace | search 1 per {client.min_interval:.1f} s, "
                  f"{client.throttled} 429s", flush=True)
    con.register("df", pd.DataFrame(rows))
    con.execute("CREATE OR REPLACE TABLE raw.case_document_labels AS SELECT * FROM df")
    found = sum(1 for r in rows if r["doc_rules"])
    no_doc = sum(1 for r in rows if r["search_status"] == 200 and not r["n_docs"])
    failed = sum(1 for r in rows if r["search_status"] != 200)
    unlabeled = len(rows) - found - no_doc - failed
    print(f"Labels for {found}/{len(rows)} cases; {no_doc} not in DAO (no document with the case number), "
          f"{unlabeled} with a document but no rule found, {failed} searches failed (rerun to retry)")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--case", required=True)
    lb = sub.add_parser("labels")
    lb.add_argument("--db", default="data/finra.duckdb")
    lb.add_argument("--limit", type=int)
    a = ap.parse_args()
    # 2/s and then 1/s drew 429s on the search page; back off up to 1 request per 4 s, and wait
    # 30, 60, 90 ... s between retries
    def log(msg):
        print(f"  search: {msg}", flush=True)

    # the search page answers 429 for minutes once tripped; speeding back up after 25 successes tripped it
    # again within a minute, so the search pace only recovers after 300 successes in a row
    client = PoliteClient(cache_dir="data/cache/finra", max_per_second=1.0, max_retries=6,
                          retry_wait=30.0, max_interval=4.0, recover_after=300, log=log)
    pdf_client = PoliteClient(cache_dir="data/cache/finra", max_per_second=2.0, max_retries=6, retry_wait=30.0,
                              max_interval=4.0, log=lambda m: print(f"  document: {m}", flush=True))
    if a.cmd == "probe":
        probe(client, a.case)
    else:
        build_labels(client, a.db, a.limit, pdf_client)


if __name__ == "__main__":
    main()
