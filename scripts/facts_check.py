"""Sentences in model.facts that mention sanctions or a fine, or still carry a rule number, to see where they come from.

Usage:
    uv run python scripts/facts_check.py            # 25 random sentences of each kind
"""

import argparse
import re

import duckdb

from finra_nlp.citations import leaks
from finra_nlp.facts import SENTENCE_RE

SANCTION_RE = re.compile(r"sanction|\bfine of\b|statutory disqualification", re.IGNORECASE)

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/finra.duckdb")
ap.add_argument("-n", type=int, default=25)
a = ap.parse_args()
con = duckdb.connect(a.db, read_only=True)
df = con.execute("SELECT case_no, start, facts_text FROM model.facts").df()
con.close()

sanction, numbered = [], []
for r in df.itertuples(index=False):
    sents = SENTENCE_RE.split(r.facts_text)
    for i, x in enumerate(sents):
        if SANCTION_RE.search(x):
            # how far into the text, and how much follows: a sanctions section runs to the end
            sanction.append((r.case_no, r.start, i, len(sents), x))
        if leaks(x):
            numbered.append((r.case_no, r.start, x))

cases = {c for c, *_ in sanction}
last = {c for c, _, i, n, _ in sanction if i >= n - 3}
print(f"{len(sanction)} sentences in {len(cases)} cases mention sanctions or a fine; in {len(last)} cases one of the "
      f"last three sentences does")
for c, start, i, n, x in sorted(sanction, key=lambda t: hash(t[0]))[: a.n]:
    print(f"  [{c}, {start}, sentence {i + 1} of {n}] {x[:250]}")
print(f"\n{len(numbered)} sentences still carry a rule number")
for c, start, x in numbered[: a.n]:
    print(f"  [{c}, {start}] {x[:250]}")
