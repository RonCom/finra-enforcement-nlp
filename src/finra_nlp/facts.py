"""Facts sections of the case documents (spec H6): the classifier input for the full-document comparison.

The labels come from the documents' violation sentences, so a model reading the whole document would read the
answer. This keeps only the narrative of what happened: the text after the FACTS AND VIOLATIVE CONDUCT heading
(or another facts heading), with "(Violation of ...)" sub-headings removed, cut at the first sentence that states
a violation. Rule numbers are masked as in the summaries. Cases whose facts come to under MIN_CHARS characters
are left out and counted. OCR text is used where the OCR pass read the document.

Writes model.facts (case_no, split, facts_text, chars, start). Reads cached documents only.

Usage:
    uv run python -m finra_nlp.facts
    uv run python -m finra_nlp.facts --show 5      # print five extracted sections to read
"""

from __future__ import annotations

import argparse
import re

import duckdb
import pandas as pd

from finra_nlp.citations import leaks, mask
from finra_nlp.dao import VIOLATION_RE, WAIVER_SECTION_RE
from finra_nlp.http import PoliteClient
from finra_nlp.monthly import pdf_text
from finra_nlp.ocr import cache_path

MIN_CHARS = 300
MAX_CHARS = 20000
HEADINGS = [  # tried in order; the first found starts the facts
    ("facts and violative conduct", re.compile(r"FACTS\s+AND\s+VIOLATIVE\s+CONDUCT")),
    ("findings of fact", re.compile(r"FINDINGS\s+OF\s+FACTS?|Findings\s+of\s+Fact\b")),
    ("facts", re.compile(r"\b(?:[IVX]+\.\s*|[A-H]\.\s*)?(?:FACTS|Facts)\b(?=\s+[A-Z])")),
]
SUBHEAD_RE = re.compile(r"\(\s*(?:Alleged\s+)?Violations?\s+of\b.{0,300}?\)", re.IGNORECASE)


def facts_section(text: str) -> tuple[str, str]:
    """(facts text, start heading or '' if none found). The text is unmasked."""
    text = re.sub(r"\s+", " ", text)
    cut = WAIVER_SECTION_RE.search(text)
    if cut:
        text = text[: cut.start()]
    start, kind = None, ""
    for name, pat in HEADINGS:
        m = pat.search(text)
        if m:
            start, kind = m.end(), name
            break
    if start is None:
        return "", ""
    body = SUBHEAD_RE.sub(" ", text[start:])
    v = VIOLATION_RE.search(body)
    if v:
        # VIOLATION_RE's match starts at the previous period; keep everything before that sentence
        body = body[: v.start()]
    return body.strip()[:MAX_CHARS], kind


def document_text(client: PoliteClient, url) -> str:
    if not isinstance(url, str) or not url:  # no document link comes through pandas as NaN
        return ""
    ocr = cache_path(url)
    if ocr.exists():
        return ocr.read_text(encoding="utf-8")
    if not client.cached(url):
        return ""
    status, body = client.get(url)
    return pdf_text(body) if status == 200 and body.startswith(b"%PDF") else ""


def build(db: str, client: PoliteClient | None = None, show: int = 0) -> pd.DataFrame:
    client = client or PoliteClient(cache_dir="data/cache/finra")
    con = duckdb.connect(db)
    cases = con.execute("""SELECT d.case_no, d.split, l.doc_url FROM model.dataset d
                           LEFT JOIN raw.case_document_labels l USING (case_no) ORDER BY d.case_no""").df()
    try:
        ocr_urls = dict(con.execute("SELECT case_no, doc_url FROM raw.case_ocr_labels").fetchall())
    except duckdb.CatalogException:
        ocr_urls = {}
    rows, dropped = [], {"no document": 0, "no facts heading": 0, f"under {MIN_CHARS} characters": 0}
    for i, r in enumerate(cases.itertuples(index=False), 1):
        text = document_text(client, ocr_urls.get(r.case_no) or r.doc_url)
        if not text:
            dropped["no document"] += 1
            continue
        facts, kind = facts_section(text)
        if not kind:
            dropped["no facts heading"] += 1
            continue
        masked = mask(facts)
        if len(masked) < MIN_CHARS:
            dropped[f"under {MIN_CHARS} characters"] += 1
            continue
        rows.append({"case_no": r.case_no, "split": r.split, "facts_text": masked, "chars": len(masked), "start": kind})
        if i % 500 == 0:
            print(f"  {i}/{len(cases)} cases read, {len(rows)} with facts", flush=True)
    df = pd.DataFrame(rows, columns=["case_no", "split", "facts_text", "chars", "start"])
    con.execute("CREATE SCHEMA IF NOT EXISTS model")
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE model.facts AS SELECT * FROM df")
    con.close()

    print(f"\n{len(df)} of {len(cases)} dataset cases have a facts section; left out: "
          + ", ".join(f"{v} {k}" for k, v in dropped.items()))
    if len(df):
        print(df.groupby("split").agg(cases=("case_no", "size"), median_chars=("chars", "median")).to_string())
        print(df.start.value_counts().to_string())
        leaky = df.facts_text.map(lambda t: bool(leaks(t)))
        stated = df.facts_text.str.contains(r"\bviolat(?:ed|ion|ions)\b", case=False)
        print(f"numbered rule references left: {int(leaky.sum())}; texts still saying 'violated'/'violation': "
              f"{int(stated.sum())}")
    for r in df.sample(min(show, len(df)), random_state=0).itertuples(index=False) if show else []:
        print(f"\n--- {r.case_no} ({r.split}, {r.chars} characters, from '{r.start}') ---\n{r.facts_text[:1500]}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--show", type=int, default=0, help="print this many extracted sections")
    a = ap.parse_args()
    build(a.db, show=a.show)


if __name__ == "__main__":
    main()
