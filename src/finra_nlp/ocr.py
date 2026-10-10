"""OCR for case documents whose text layer gives no rule and whose summary cites none.

Some DAO PDFs have no usable text layer: 2015-2017 scans with an old OCR layer that garbles rule
numbers ("FINRA Rulc 9216", "F1NRA"), 2020-2022 files whose fonts extract as shifted characters
("HKPCPEKCN" for "FINANCIAL"), and scans with no text at all. This pass renders each page and runs
Tesseract on it (through PyMuPDF), then applies the same violation-sentence rules as the document
labels. Results go to raw.case_ocr_labels; labels.case_labels uses them last, after the document
and the summary.

Needs Tesseract installed (Windows: the UB Mannheim build, with its folder on PATH or
TESSDATA_PREFIX set to its tessdata folder). OCR text is cached under data/cache/ocr, so a rerun
only OCRs documents it hasn't read.

Usage:
    uv run python -m finra_nlp.ocr --limit 3     # check Tesseract and the timing first
    uv run python -m finra_nlp.ocr
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import duckdb
import pandas as pd
import pymupdf

from finra_nlp.dao import SEARCH_URL, document_links, labels_from_document
from finra_nlp.http import PoliteClient
from finra_nlp.labels import case_labels

CACHE = Path("data/cache/ocr")


def ocr_text(data: bytes, dpi: int = 300) -> tuple[str, int]:
    """Text of every page from Tesseract, ignoring any text layer; returns (text, pages)."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [page.get_text(textpage=page.get_textpage_ocr(dpi=dpi, full=True)) for page in doc]
    return "\n".join(pages), len(pages)


def documents(con, client: PoliteClient, cases: list[str]) -> dict[str, list[str]]:
    """Each case's document URLs: the DAO index, then a cached search page, then the one the labels kept."""
    index = con.execute("SELECT case_no, doc_url FROM raw.dao_index ORDER BY action_date DESC").fetchall()
    kept = dict(con.execute("SELECT case_no, doc_url FROM raw.case_document_labels").fetchall())
    out: dict[str, list[str]] = {c: [] for c in cases}
    for c, u in index:
        if c in out:
            out[c].append(u)
    for c in cases:
        if not out[c]:
            url = SEARCH_URL.format(case=c)
            if client.cached(url):
                out[c] = document_links(client.get(url)[1], url, c)
        if not out[c] and kept.get(c):
            out[c] = [kept[c]]
    return out


def build(db: str, client: PoliteClient, limit: int | None = None, dpi: int = 300, cache: Path = CACHE) -> pd.DataFrame:
    con = duckdb.connect(db)
    labels = case_labels(con, ocr=False)
    with_doc = set(r[0] for r in con.execute(
        "SELECT case_no FROM raw.case_document_labels WHERE n_docs > 0").fetchall())
    cases = sorted(c for c in labels[labels.rules == ""].case_no if c in with_doc)
    if limit:
        cases = cases[:limit]
    docs = documents(con, client, cases)
    print(f"{len(cases)} cases have a document but no rule from it or from the summary; OCR at {dpi} dpi", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    rows, start, found, pages_done = [], time.monotonic(), 0, 0
    for i, case_no in enumerate(cases, 1):
        keys, used, pages = [], None, 0
        for url in docs[case_no]:
            path = cache / (hashlib.sha256(f"{url}|{dpi}".encode()).hexdigest() + ".txt")
            if path.exists():
                text = path.read_text(encoding="utf-8")
            else:
                status, pdf = client.get(url)
                if status != 200 or not pdf.startswith(b"%PDF"):
                    continue
                text, n = ocr_text(pdf, dpi)
                pages += n
                path.write_text(text, encoding="utf-8")
            keys, used = labels_from_document(text), url
            if keys:
                break
        found += bool(keys)
        pages_done += pages
        rows.append({"case_no": case_no, "doc_url": used, "ocr_rules": "|".join(keys)})
        took = time.monotonic() - start
        print(f"  {i}/{len(cases)} {case_no}: {'|'.join(keys) or 'no rule'} | {pages} pages OCR'd, "
              f"{took / 60:.1f} min so far, about {(len(cases) - i) * took / i / 60:.0f} min left", flush=True)
    df = pd.DataFrame(rows, columns=["case_no", "doc_url", "ocr_rules"])
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE raw.case_ocr_labels AS SELECT * FROM df")
    left = (case_labels(con).rules == "").sum()
    con.close()
    print(f"OCR labels for {found}/{len(cases)} cases ({pages_done} pages OCR'd this run); "
          f"{left} monthly-report cases still have no rule from any source")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    try:
        pymupdf.get_tessdata()
    except RuntimeError as e:
        raise SystemExit(f"Tesseract not found ({e}). Install it (Windows: UB Mannheim build), then put its "
                         "folder on PATH or set TESSDATA_PREFIX to its tessdata folder.") from None
    client = PoliteClient(cache_dir="data/cache/finra", max_per_second=1.0, max_retries=6, retry_wait=30.0,
                          max_interval=4.0)
    build(a.db, client, a.limit, a.dpi)


if __name__ == "__main__":
    main()
