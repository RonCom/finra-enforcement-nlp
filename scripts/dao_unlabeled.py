"""Why cases with a DAO document got no rule: reads cached documents only, never the network.

    uv run python scripts/dao_unlabeled.py            # counts by reason and document type
    uv run python scripts/dao_unlabeled.py --out data/dao_unlabeled.csv

Reasons, per case (the best document wins, in this order):
    not cached    no document was fetched (a 429 or error during the run; a labels rerun retries it)
    no text       under 500 characters of text (a scanned PDF)
    no violation  cites rules, but no sentence before the waiver states a violation
    no citation   states a violation, but no sentence names a rule the extractor knows
    neither       no citation and no violation sentence
    apart         cites rules and states a violation, but never in the same sentence (or only in one that waives)
"""

from __future__ import annotations

import argparse
import re

import duckdb
import pandas as pd

from finra_nlp.citations import extract_citations
from finra_nlp.dao import SEARCH_URL, VIOLATION_RE, WAIVER_SECTION_RE, document_links
from finra_nlp.http import PoliteClient
from finra_nlp.monthly import pdf_text

ORDER = ["apart", "no violation", "no citation", "neither", "no text", "not cached"]


def diagnose(client: PoliteClient, url: str) -> tuple[str, int, str]:
    if not client.cached(url):
        return "not cached", 0, ""
    status, body = client.get(url)
    if status != 200 or not body.startswith(b"%PDF"):
        return "not cached", 0, ""
    text = re.sub(r"\s+", " ", pdf_text(body))
    if len(text) < 500:
        return "no text", len(text), text[:200]
    cut = WAIVER_SECTION_RE.search(text)
    head = text[: cut.start()] if cut else text
    cites = bool(extract_citations(head))
    violation = VIOLATION_RE.search(head)
    if cites and not violation:
        return "no violation", len(text), head[:300]
    if violation and not cites:
        return "no citation", len(text), violation.group(0)[:300]
    if not cites:
        return "neither", len(text), head[:300]
    return "apart", len(text), violation.group(0)[:300]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--out", default="data/dao_unlabeled.csv")
    a = ap.parse_args()
    con = duckdb.connect(a.db, read_only=True)
    cases = con.execute("""SELECT d.case_no, d.doc_url, m.action_date FROM raw.case_document_labels d
                           JOIN raw.monthly_cases m USING (case_no)
                           WHERE coalesce(d.doc_rules, '') = '' AND d.n_docs > 0 ORDER BY d.case_no""").df()
    docs = con.execute("SELECT case_no, doc_url, doc_type, action_date FROM raw.dao_index").df()
    con.close()
    client = PoliteClient(cache_dir="data/cache/finra")
    rows = []
    for c in cases.itertuples(index=False):
        found = docs[docs.case_no == c.case_no]
        candidates = [(d.doc_url, d.doc_type, d.action_date) for d in found.itertuples(index=False)]
        if not candidates:  # found by search, not in the index: links from the cached search page
            url = SEARCH_URL.format(case=c.case_no)
            links = document_links(client.get(url)[1], url, c.case_no) if client.cached(url) else []
            links = links or ([c.doc_url] if c.doc_url else [])
            candidates = [(u, "(searched)", c.action_date) for u in links]
        best = {"case_no": c.case_no, "reason": "not cached", "doc_type": "(no link kept)",
                "action_date": c.action_date, "chars": 0, "doc_url": None, "snippet": ""}
        for doc_url, doc_type, action_date in candidates:
            reason, chars, snippet = diagnose(client, doc_url)
            if best["doc_url"] is None or ORDER.index(reason) < ORDER.index(best["reason"]):
                best = {"case_no": c.case_no, "reason": reason, "doc_type": doc_type, "action_date": action_date,
                        "chars": chars, "doc_url": doc_url, "snippet": snippet}
        rows.append(best)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} cases with a document but no rule; written to {a.out}\n")
    print(df.reason.value_counts().to_string(), "\n")
    print(pd.crosstab(df.doc_type.fillna("(none)"), df.reason).to_string(), "\n")
    df["year"] = pd.to_datetime(df.action_date).dt.year
    print(pd.crosstab(df.year, df.reason).to_string())


if __name__ == "__main__":
    main()
