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
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        if href.lower().endswith(".pdf") and (case_no in href or case_no in a.get_text()):
            links.append(href)
    if not links:  # fall back to any PDF on the page; the search should return one case
        links = [urljoin(page_url, a["href"]) for a in soup.find_all("a", href=True)
                 if a["href"].lower().endswith(".pdf")]
    return list(dict.fromkeys(links))


def labels_from_document(text: str) -> list[str]:
    """Citation keys from sentences that state violations, in order of first appearance."""
    text = re.sub(r"\s+", " ", text)
    keys: dict[str, None] = {}
    for sent in VIOLATION_RE.finditer(text):
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


def build_labels(client: PoliteClient, db: str, limit: int | None) -> None:
    con = duckdb.connect(db)
    case_nos = [r[0] for r in con.execute("SELECT case_no FROM raw.monthly_cases ORDER BY case_no").fetchall()]
    if limit:
        case_nos = case_nos[:limit]
    rows = []
    for i, case_no in enumerate(case_nos, 1):
        url = SEARCH_URL.format(case=case_no)
        status, html = client.get(url)
        links = document_links(html, url, case_no) if status == 200 else []
        keys, doc_url = [], None
        for link in links:
            s, pdf = client.get(link)
            if s == 200:
                keys, doc_url = labels_from_document(pdf_text(pdf)), link
                if keys:
                    break
        rows.append({"case_no": case_no, "doc_url": doc_url, "doc_rules": "|".join(keys), "n_docs": len(links)})
        if i % 100 == 0:
            print(f"  {i}/{len(case_nos)}")
    con.register("df", pd.DataFrame(rows))
    con.execute("CREATE OR REPLACE TABLE raw.case_document_labels AS SELECT * FROM df")
    found = sum(1 for r in rows if r["doc_rules"])
    print(f"Labels for {found}/{len(rows)} cases")
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
    client = PoliteClient(cache_dir="data/cache/finra")
    if a.cmd == "probe":
        probe(client, a.case)
    else:
        build_labels(client, a.db, a.limit)


if __name__ == "__main__":
    main()
