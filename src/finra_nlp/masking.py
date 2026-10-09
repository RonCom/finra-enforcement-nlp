"""Masking check (spec, "Labels" step 5): no numbered rule reference may remain in a masked summary.

Exits with status 1 and lists the leaks if any summary in raw.monthly_cases.masked_text still
contains one. Rule titles stay in by design (change log, 2026-10-08), so only numbers are checked.

Usage:
    uv run python -m finra_nlp.masking --db data/finra.duckdb
"""

from __future__ import annotations

import argparse
import sys

import duckdb

from finra_nlp.citations import leaks


def check(db: str, show: int = 20) -> list[tuple[str, list[str]]]:
    con = duckdb.connect(db, read_only=True)
    rows = con.execute("SELECT case_no, masked_text FROM raw.monthly_cases ORDER BY case_no").fetchall()
    con.close()
    bad = [(case_no, hits) for case_no, text in rows if (hits := leaks(text or ""))]
    print(f"{len(rows)} masked summaries checked, {len(bad)} with a numbered rule reference left")
    for case_no, hits in bad[:show]:
        print(f"  {case_no}: {hits}")
    if len(bad) > show:
        print(f"  ... {len(bad) - show} more")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    a = ap.parse_args()
    sys.exit(1 if check(a.db) else 0)


if __name__ == "__main__":
    main()
