"""Classifier dataset (spec, "Classifier"): masked summaries with series labels and date splits.

Labels come from the case documents, or the monthly summary's citations or an OCR pass where the
document gave no rule (labels.case_labels; label_source says which). Each rule maps to its
series (FINRA:3110 -> FINRA:3000; an NASD rule takes its FINRA successor's series, NASD 3010 -> FINRA 3110),
Rule 2010 and procedural rules are dropped, and series with fewer than
MIN_SERIES_CASES training cases merge into "other". Cases with no rule from any source are left out.
A case that cites only Rule 2010 keeps an empty label set.

Splits use the action date (the report month when the date is missing):
train 2016-2022, validation 2023, test 2024-2026.

Usage:
    uv run python -m finra_nlp.dataset --db data/finra.duckdb    # writes model.dataset
"""

from __future__ import annotations

import argparse

import duckdb
import pandas as pd

from finra_nlp.labels import MIN_SERIES_CASES, TRAIN_YEARS, _set, case_labels, label_series

VALIDATION_YEAR = 2023
TEST_YEARS = (2024, 2026)
OTHER = "other"


def split_for(year: int | None) -> str | None:
    if year is None or pd.isna(year):
        return None
    if TRAIN_YEARS[0] <= year <= TRAIN_YEARS[1]:
        return "train"
    if year == VALIDATION_YEAR:
        return "validation"
    if TEST_YEARS[0] <= year <= TEST_YEARS[1]:
        return "test"
    return None


def build(db: str, write: bool = True) -> pd.DataFrame:
    con = duckdb.connect(db, read_only=not write)
    df = con.execute("SELECT case_no, action_date, report_month, masked_text FROM raw.monthly_cases").df()
    df = df.merge(case_labels(con), on="case_no")
    df = df[df.rules != ""].copy()
    year = pd.to_datetime(df.action_date, errors="coerce").dt.year
    df["year"] = year.fillna(pd.to_numeric(df.report_month.str[:4], errors="coerce")).astype("Int64")
    df["split"] = df.year.map(split_for)
    df = df[df.split.notna()].copy()
    df["series"] = df.rules.map(lambda s: sorted(label_series(_set(s))))

    counts = df[df.split == "train"].series.explode().value_counts()
    keep = set(counts[counts >= MIN_SERIES_CASES].index)
    df["labels"] = df.series.map(lambda ss: "|".join(sorted({s if s in keep else OTHER for s in ss})))
    out = df[["case_no", "year", "split", "masked_text", "rules", "label_source", "labels"]].reset_index(drop=True)

    if write:
        con.execute("CREATE SCHEMA IF NOT EXISTS model")
        con.register("out", out)
        con.execute("CREATE OR REPLACE TABLE model.dataset AS SELECT * FROM out")
    con.close()

    print(out.groupby("split").agg(cases=("case_no", "size"),
                                   no_label=("labels", lambda s: int((s == "").sum())),
                                   from_summary=("label_source", lambda s: int((s == "summary").sum())),
                                   from_ocr=("label_source", lambda s: int((s == "ocr").sum()))).to_string())
    print(f"series kept: {sorted(keep)}")
    merged = sorted(set(counts.index) - keep)
    print(f"merged into '{OTHER}': {merged or 'none'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    build(ap.parse_args().db)


if __name__ == "__main__":
    main()
