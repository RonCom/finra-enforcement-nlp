"""Hand-check of document labels (spec, "Labels" step 3) and the label profile (step 4).

Usage:
    uv run python -m finra_nlp.labels sample            # writes data/label_handcheck.csv
    uv run python -m finra_nlp.labels score             # precision/recall vs. the 0.98 gate
    uv run python -m finra_nlp.labels profile           # writes reports/label_profile.md
    uv run python -m finra_nlp.labels export            # writes data/label_handcheck_docs.json

Hand-check: open each doc_url, correct true_rules (prefilled with the extracted labels, keys
joined by "|", e.g. FINRA:3110|FINRA:2010), and set checked to Y.

`export` reads each sampled document from the HTTP cache (fetching it if missing) and writes,
per case, every sentence before the waiver section that cites a rule or states a violation.
That file is short enough to read in one pass when a reviewer proposes true_rules.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from finra_nlp.citations import extract_citations
from finra_nlp.dao import VIOLATION_RE, WAIVER_SECTION_RE
from finra_nlp.http import PoliteClient
from finra_nlp.monthly import pdf_text

GATE = 0.98
TRAIN_YEARS = (2016, 2022)
MIN_SERIES_CASES = 30
CATCH_ALL = "FINRA:2010"


def _set(s: str) -> set[str]:
    return {x.strip() for x in (s or "").split("|") if x.strip()}


def series_label(key: str) -> str:
    """FINRA:3110 -> FINRA:3000; NASD:3010 -> NASD:3000; other families keep only the family."""
    family, _, rule = key.partition(":")
    if family in ("FINRA", "NASD"):
        m = re.match(r"(?:IM-)?(\d+)", rule)
        return f"{family}:{int(m.group(1)) // 1000 * 1000}" if m else family
    return family


def _cases(con) -> pd.DataFrame:
    return con.execute(
        """SELECT m.case_no, m.report_month, m.section, m.action_date, m.summary_rules,
                  d.doc_url, d.doc_rules
           FROM raw.monthly_cases m LEFT JOIN raw.case_document_labels d USING (case_no)"""
    ).df()


def case_labels(con, ocr: bool = True) -> pd.DataFrame:
    """One row per monthly-report case: the rules used as labels and where they came from.

    Order: the case document's violation sentences; the monthly summary's citations when the document
    gave no rule (old scans with garbled text, scrambled fonts, no text layer); then rules from an OCR
    pass over the document (raw.case_ocr_labels, from `finra_nlp.ocr`). label_source is None when
    no source has a rule.
    """
    has_ocr = ocr and con.execute(
        """SELECT count(*) FROM information_schema.tables
           WHERE table_schema = 'raw' AND table_name = 'case_ocr_labels'""").fetchone()[0]
    ocr_rules = "o.ocr_rules" if has_ocr else "NULL"
    ocr_join = "LEFT JOIN raw.case_ocr_labels o USING (case_no)" if has_ocr else ""
    return con.execute(
        f"""SELECT m.case_no,
                   CASE WHEN coalesce(d.doc_rules, '') <> '' THEN d.doc_rules
                        WHEN coalesce(m.summary_rules, '') <> '' THEN m.summary_rules
                        WHEN coalesce({ocr_rules}, '') <> '' THEN {ocr_rules} ELSE '' END AS rules,
                   CASE WHEN coalesce(d.doc_rules, '') <> '' THEN 'document'
                        WHEN coalesce(m.summary_rules, '') <> '' THEN 'summary'
                        WHEN coalesce({ocr_rules}, '') <> '' THEN 'ocr' END AS label_source
            FROM raw.monthly_cases m LEFT JOIN raw.case_document_labels d USING (case_no) {ocr_join}"""
    ).df()


def sample(db: str, out: str, n: int = 100, seed: int = 42) -> None:
    con = duckdb.connect(db, read_only=True)
    df = _cases(con)
    con.close()
    df = df[df.doc_rules.fillna("") != ""]
    picked = df.sample(n=min(n, len(df)), random_state=seed)[["case_no", "doc_url", "doc_rules"]].copy()
    picked["true_rules"] = picked.doc_rules
    picked["checked"] = ""
    picked["notes"] = ""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    picked.to_csv(out, index=False)
    print(f"Wrote {len(picked)} cases to {out}")


def score(csv: str) -> tuple[float, float]:
    df = pd.read_csv(csv, dtype=str).fillna("")
    df = df[df.checked.str.upper().str.strip() == "Y"]
    tp = fp = fn = 0
    for pred, true in zip(df.doc_rules, df.true_rules):
        p, t = _set(pred), _set(true)
        tp, fp, fn = tp + len(p & t), fp + len(p - t), fn + len(t - p)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    verdict = "PASS" if precision >= GATE and recall >= GATE else "FAIL"
    print(f"{len(df)} cases checked: precision {precision:.3f}, recall {recall:.3f} (gate {GATE}) {verdict}")
    return precision, recall


SENTENCE_RE = re.compile(r"[^.]*(?:\.(?=\s|$)|$)")  # periods inside "Rule 2010." are fine; "Sec." splits early


def document_sentences(text: str) -> tuple[list[str], int, bool]:
    """Sentences before the waiver heading that cite a rule or state a violation.
    Returns (sentences, characters before the waiver, whether a waiver heading was found)."""
    text = re.sub(r"\s+", " ", text)
    cut = WAIVER_SECTION_RE.search(text)
    head = text[: cut.start()] if cut else text
    out = []
    for m in SENTENCE_RE.finditer(head):
        s = m.group(0).strip()
        if s and (extract_citations(s) or VIOLATION_RE.search(s)):
            out.append(s)
    return out, len(head), cut is not None


def export(csv: str, out: str, client=None) -> list[dict]:
    """Write the hand-check sample with each document's citing and violation sentences."""
    client = client or PoliteClient(cache_dir="data/cache/finra")
    df = pd.read_csv(csv, dtype=str).fillna("")
    rows = []
    for r in df.itertuples(index=False):
        status, body = client.get(r.doc_url) if r.doc_url else (0, b"")
        if status != 200 or not body.startswith(b"%PDF"):
            rows.append({"case_no": r.case_no, "doc_url": r.doc_url, "doc_rules": r.doc_rules,
                         "error": f"document not read (status {status})", "sentences": []})
            continue
        sents, chars, waiver = document_sentences(pdf_text(body))
        rows.append({"case_no": r.case_no, "doc_url": r.doc_url, "doc_rules": r.doc_rules,
                     "chars_before_waiver": chars, "waiver_found": waiver, "sentences": sents})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    bad = sum(1 for x in rows if "error" in x)
    print(f"Wrote {len(rows)} cases to {out} ({bad} documents not read)")
    return rows


def profile(db: str, out: str) -> str:
    con = duckdb.connect(db, read_only=True)
    df = _cases(con)
    con.close()
    df["year"] = pd.to_numeric(df.report_month.str[:4], errors="coerce")
    df["doc_set"] = df.doc_rules.fillna("").map(_set)
    df["sum_set"] = df.summary_rules.fillna("").map(_set)
    labeled = df[df.doc_set.map(bool)]

    lines = ["# Label profile", ""]
    by_year = df.groupby("year").agg(
        cases=("case_no", "size"),
        with_doc_labels=("doc_set", lambda s: int(s.map(bool).sum())),
        summary_cites_any=("sum_set", lambda s: round(s.map(bool).mean(), 3)),
    )
    lines += ["## Cases by report year", "", by_year.to_markdown(), ""]

    con = duckdb.connect(db, read_only=True)
    sources = case_labels(con).label_source.fillna("none").value_counts()
    con.close()
    lines += ["## Label source", "",
              "Document violation sentences first, then the monthly summary's citations, then OCR of the document.",
              "", sources.rename("cases").to_frame().to_markdown(), ""]

    share_2010 = labeled.doc_set.map(lambda s: CATCH_ALL in s).mean()
    lines += [f"Rule 2010 appears in {share_2010:.1%} of labeled cases and is dropped as a label.", ""]

    train = labeled[labeled.year.between(*TRAIN_YEARS)]
    counts: dict[str, int] = {}
    for s in train.doc_set:
        for lab in {series_label(k) for k in s if k != CATCH_ALL}:
            counts[lab] = counts.get(lab, 0) + 1
    sc = pd.Series(counts, name="train_cases").sort_values(ascending=False).to_frame()
    sc["merged_to_other"] = sc.train_cases < MIN_SERIES_CASES
    lines += [f"## Series labels, train years {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}", "",
              f"Series under {MIN_SERIES_CASES} training cases merge into 'other'.", "",
              sc.to_markdown(), ""]

    both = labeled[labeled.sum_set.map(bool)]
    if len(both):
        jac = [len(a & b) / len(a | b) for a, b in zip(both.doc_set, both.sum_set)]
        exact = sum(a == b for a, b in zip(both.doc_set, both.sum_set)) / len(both)
        lines += ["## Summary citations vs. document labels", "",
                  f"{len(both)} cases have both. Identical sets: {exact:.1%}. Mean Jaccard: {sum(jac) / len(jac):.3f}.",
                  ""]
    text = "\n".join(lines)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--db", default="data/finra.duckdb")
    s.add_argument("--out", default="data/label_handcheck.csv")
    s.add_argument("--seed", type=int, default=42, help="a new seed draws a fresh sample for a re-check")
    c = sub.add_parser("score")
    c.add_argument("--csv", default="data/label_handcheck.csv")
    p = sub.add_parser("profile")
    p.add_argument("--db", default="data/finra.duckdb")
    p.add_argument("--out", default="reports/label_profile.md")
    e = sub.add_parser("export")
    e.add_argument("--csv", default="data/label_handcheck.csv")
    e.add_argument("--out", default="data/label_handcheck_docs.json")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.db, a.out, seed=a.seed)
    elif a.cmd == "score":
        score(a.csv)
    elif a.cmd == "export":
        export(a.csv, a.out)
    else:
        profile(a.db, a.out)


if __name__ == "__main__":
    main()
