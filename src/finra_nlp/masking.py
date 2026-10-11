"""Masking check (spec, "Labels" step 5): no numbered rule reference may remain in a masked summary.

Exits with status 1 and lists the leaks if any summary in raw.monthly_cases.masked_text still
contains one. Rule titles stay in by design (change log, 2026-10-08), so only numbers are checked.

`--titles` reports the other half of that step: the share of test cases whose masked summary contains the title
of a FINRA rule they were charged with (titles from the downloaded Rulebook, data/rulebook.duckdb), overall, by
series and by rule. Each rule's title is also counted in the test summaries of cases not charged with it, which
shows whether a title gives the charge away or is ordinary conduct vocabulary. Writes reports/title_share.md.

Usage:
    uv run python -m finra_nlp.masking --db data/finra.duckdb
    uv run python -m finra_nlp.masking --titles
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

from pathlib import Path

import duckdb
import pandas as pd

from finra_nlp.citations import leaks
from finra_nlp.labels import series_label

TITLE_REPORT = "reports/title_share.md"
MIN_RULE_CASES = 5  # rules charged in fewer test cases are left out of the per-rule table


def clean_title(title: str) -> str:
    """'3110. Supervision' -> 'supervision'."""
    t = re.sub(r"^\s*(?:IM-)?[\d.\-]+\.?\s*", "", str(title or ""))
    return re.sub(r"\s+", " ", t).strip().rstrip(".").lower()


def titles(db: str, rulebook_db: str = "data/rulebook.duckdb", split: str = "test", out: str = TITLE_REPORT) -> dict:
    from finra_nlp.rag import base_rule, finra_rules
    con = duckdb.connect(rulebook_db, read_only=True)
    pages = con.execute("SELECT rule_no, title FROM raw.rulebook_pages ORDER BY rule_no").fetchall()
    con.close()
    title_of = {}
    for rule, title in pages:
        t = clean_title(title)
        if t and base_rule(rule) not in title_of:
            title_of[base_rule(rule)] = t
    con = duckdb.connect(db, read_only=True)
    cases = con.execute("SELECT case_no, masked_text, rules FROM model.dataset WHERE split = ? ORDER BY case_no",
                        [split]).df()
    con.close()
    text = cases.masked_text.fillna("").str.lower().str.replace(r"\s+", " ", regex=True)
    charged = cases.rules.map(lambda r: {x for x in finra_rules(r) if x in title_of})
    has = [sorted(r for r in c if title_of[r] in t) for c, t in zip(charged, text)]
    scored = charged.map(bool)
    overall = sum(bool(h) for h, s in zip(has, scored) if s) / max(1, int(scored.sum()))

    rows = []
    for rule in sorted({r for c in charged for r in c}):
        mask = charged.map(lambda c: rule in c)
        if mask.sum() < MIN_RULE_CASES:
            continue
        found = text.str.contains(title_of[rule], regex=False)
        rows.append({"rule": rule, "title": title_of[rule], "charged cases": int(mask.sum()),
                     "title in summary, charged": found[mask].mean(),
                     "title in summary, not charged": found[~mask].mean()})
    per_rule = pd.DataFrame(rows).sort_values("charged cases", ascending=False) if rows else pd.DataFrame()
    ser = pd.DataFrame({"series": [sorted({series_label("FINRA:" + r) for r in c}) for c in charged],
                        "title": [bool(h) for h in has]})[scored.values].explode("series")
    by_series = ser.groupby("series").agg(cases=("title", "size"), with_title=("title", "mean"))

    lines = [f"# Rule titles in the masked summaries, {split} split", "",
             f"Titles from {len(title_of)} Rulebook rules, matched as phrases anywhere in the text, ignoring case. "
             f"{int(scored.sum())} of {len(cases)} {split} cases were charged with a FINRA rule in the Rulebook "
             "(NASD rules as their FINRA successors; Rule 2010 and procedural rules left out).", "",
             f"**{overall:.1%} of them have the title of a charged rule in their masked summary.**", "",
             "## By series", "", by_series.round(3).to_markdown(), "",
             f"## By rule (charged in {MIN_RULE_CASES}+ {split} cases)", "",
             "A title found as often in cases not charged with the rule is conduct vocabulary; one found mostly in "
             "charged cases points to the charge.", "",
             per_rule.round(3).to_markdown(index=False) if len(per_rule) else "(none)", ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return {"overall": overall, "per_rule": per_rule, "by_series": by_series}


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
    ap.add_argument("--titles", action="store_true", help="report rule titles in the masked test summaries")
    ap.add_argument("--rulebook-db", default="data/rulebook.duckdb")
    a = ap.parse_args()
    if a.titles:
        titles(a.db, a.rulebook_db)
        return
    sys.exit(1 if check(a.db, out=a.out) else 0)


if __name__ == "__main__":
    main()
