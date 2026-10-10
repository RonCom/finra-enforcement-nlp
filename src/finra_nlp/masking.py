"""Masking check (spec, "Labels" step 5): no numbered rule reference may remain in a masked summary.

Exits with status 1 and lists the leaks if any summary in raw.monthly_cases.masked_text still
contains one. Rule titles stay in by design (change log, 2026-10-08), so only numbers are checked.

Usage:
    uv run python -m finra_nlp.masking --db data/finra.duckdb
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

import duckdb

from finra_nlp.citations import leaks


def check(db: str, show: int = 20, out: str | None = None) -> list[tuple[str, list[str]]]:
    con = duckdb.connect(db, read_only=True)
    rows = con.execute("SELECT case_no, masked_text FROM raw.monthly_cases ORDER BY case_no").fetchall()
    con.close()
    bad = [(case_no, hits) for case_no, text in rows if (hits := leaks(text or ""))]
    if out:  # each leak with 80 characters either side, to see what the masking missed
        texts = dict(rows)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_no", "leak", "context"])
            for case_no, hits in bad:
                text = texts[case_no]
                for h in dict.fromkeys(hits):
                    for m in re.finditer(re.escape(h), text):
                        w.writerow([case_no, h, text[max(0, m.start() - 80): m.end() + 80]])
        print(f"Wrote every leak with its context to {out}")
    print(f"{len(rows)} masked summaries checked, {len(bad)} with a numbered rule reference left")
    for case_no, hits in bad[:show]:
        print(f"  {case_no}: {hits}")
    if len(bad) > show:
        print(f"  ... {len(bad) - show} more")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--out", help="CSV of every leak with its context")
    a = ap.parse_args()
    sys.exit(1 if check(a.db, out=a.out) else 0)


if __name__ == "__main__":
    main()
