"""Classifier dataset (spec, "Classifier"): masked summaries with series labels and date splits.

Labels come from the case documents (raw.case_document_labels.doc_rules). Each rule maps to its
series (FINRA:3110 -> FINRA:3000), Rule 2010 is dropped, and series with fewer than
MIN_SERIES_CASES training cases merge into "other". Cases without document labels are left out.
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

from finra_nlp.labels import CATCH_ALL, MIN_SERIES_CASES, TRAIN_YEARS, _set, series_label

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
    df = con.execute(
        """SELECT m.case_no, m.action_date, m.report_month, m.masked_text, d.doc_rules
           FROM raw.monthly_cases m JOIN raw.case_document_labels d USING (case_no)
           WHERE coalesce(d.doc_rules, '') <> ''"""
    ).df()
    year = pd.to_datetime(df.action_date, errors="coerce").dt.year
    df["year"] = year.fillna(pd.to_numeric(df.report_month.str[:4], errors="coerce")).astype("Int64")
    df["split"] = df.year.map(split_for)
    df = df[df.split.notna()].copy()
    df["series"] = df.doc_rules.map(lambda s: sorted({series_label(k) for k in _set(s) if k != CATCH_ALL}))

    counts = df[df.split == "train"].series.explode().value_counts()
    keep = set(counts[counts >= MIN_SERIES_CASES].index)
    df["labels"] = df.series.map(lambda ss: "|".join(sorted({s if s in keep else OTHER for s in ss})))
    out = df[["case_no", "year", "split", "masked_text", "doc_rules", "labels"]].reset_index(drop=True)

    if write:
        con.execute("CREATE SCHEMA IF NOT EXISTS model")
        con.register("out", out)
        con.execute("CREATE OR REPLACE TABLE model.dataset AS SELECT * FROM out")
    con.close()

    print(out.groupby("split").agg(cases=("case_no", "size"),
                                   no_label=("labels", lambda s: int((s == "").sum()))).to_string())
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
