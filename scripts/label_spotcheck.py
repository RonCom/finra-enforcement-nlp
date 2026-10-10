"""Spot-check of labels outside the hand-check: a sample of OCR labels, and every citation whose
rule number doesn't exist in the Rulebook's numbering (FINRA above 14999, NASD above 11999).
Reads cached documents and OCR text only, never the network.

    uv run python scripts/label_spotcheck.py                 # writes data/label_spotcheck.json
    uv run python scripts/label_spotcheck.py --n 15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb

from finra_nlp.citations import extract_citations
from finra_nlp.dao import VIOLATION_RE, WAIVER_SECTION_RE
from finra_nlp.http import PoliteClient
from finra_nlp.labels import _set, case_labels
from finra_nlp.monthly import pdf_text
from finra_nlp.ocr import cache_path

LIMIT = {"FINRA": 14999, "NASD": 11999}


def odd(key: str) -> bool:
    family, _, rule = key.partition(":")
    m = re.match(r"(?:IM-)?(\d+)", rule)
    return family in LIMIT and bool(m) and int(m.group(1)) > LIMIT[family]


def violation_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    cut = WAIVER_SECTION_RE.search(text)
    return [s.group(0).strip() for s in VIOLATION_RE.finditer(text[: cut.start()] if cut else text)
            if extract_citations(s.group(0))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--out", default="data/label_spotcheck.json")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    con = duckdb.connect(a.db, read_only=True)
    labels = case_labels(con).set_index("case_no")
    ocr = con.execute("SELECT case_no, doc_url, ocr_rules FROM raw.case_ocr_labels WHERE ocr_rules <> ''").df()
    docs = dict(con.execute("SELECT case_no, doc_url FROM raw.case_document_labels").fetchall())
    texts = dict(con.execute("SELECT case_no, text FROM raw.monthly_cases").fetchall())
    con.close()
    client = PoliteClient(cache_dir="data/cache/finra")

    def source_text(case_no: str, source: str) -> str:
        if source == "summary":
            return texts.get(case_no) or ""
        if source == "ocr":
            row = ocr[ocr.case_no == case_no]
            path = cache_path(row.doc_url.iloc[0]) if len(row) else None
            return path.read_text(encoding="utf-8") if path and path.exists() else ""
        url = docs.get(case_no)
        if url and client.cached(url):
            status, body = client.get(url)
            return pdf_text(body) if status == 200 and body.startswith(b"%PDF") else ""
        return ""

    sample = ocr.sample(n=min(a.n, len(ocr)), random_state=a.seed)
    ocr_rows = [{"case_no": r.case_no, "doc_url": r.doc_url, "ocr_rules": r.ocr_rules,
                 "sentences": violation_sentences(source_text(r.case_no, "ocr"))}
                for r in sample.itertuples(index=False)]

    odd_rows = []
    for case_no, r in labels[labels.rules != ""].iterrows():
        keys = [k for k in _set(r.rules) if odd(k)]
        if not keys:
            continue
        text = re.sub(r"\s+", " ", source_text(case_no, r.label_source))
        nums = [k.partition(":")[2] for k in keys]
        sents = [s for s in (violation_sentences(text) if r.label_source != "summary" else re.split(r"(?<=\.)\s", text))
                 if any(n in s for n in nums)]
        odd_rows.append({"case_no": case_no, "label_source": r.label_source, "odd_keys": keys, "rules": r.rules,
                         "sentences": sents[:3]})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"ocr_sample": ocr_rows, "odd_numbers": odd_rows}, indent=1,
                                      ensure_ascii=False), encoding="utf-8")
    by_source: dict[str, int] = {}
    for x in odd_rows:
        by_source[x["label_source"]] = by_source.get(x["label_source"], 0) + 1
    print(f"Wrote {a.out}: {len(ocr_rows)} OCR cases; {len(odd_rows)} cases with a rule number outside the "
          f"Rulebook's numbering {by_source}")


if __name__ == "__main__":
    main()
