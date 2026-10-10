"""NASD rules without a FINRA successor in labels.NASD_SUCCESSOR, with case counts by split.

    uv run python scripts/nasd_rules.py
"""

from __future__ import annotations

import argparse

import duckdb
import pandas as pd

from finra_nlp.labels import NASD_SUCCESSOR, _set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    a = ap.parse_args()
    con = duckdb.connect(a.db, read_only=True)
    df = con.execute("SELECT split, rules FROM model.dataset").df()
    con.close()
    rows = [(r.split, k.partition(":")[2]) for r in df.itertuples() for k in _set(r.rules)
            if k.startswith("NASD:") and k.partition(":")[2] not in NASD_SUCCESSOR]
    out = pd.DataFrame(rows, columns=["split", "rule"]).value_counts().unstack(fill_value=0)
    out["total"] = out.sum(axis=1)
    print(out.sort_values("total", ascending=False).head(40).to_string())


if __name__ == "__main__":
    main()
