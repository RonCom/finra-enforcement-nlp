"""Paired bootstrap of the test-split macro-F1 differences, from the stored test predictions
(model.test_predictions, written once by `testrun score`). Descriptive, added after the test run: it
resamples cases with replacement and reports 95% intervals for each model's macro-F1 and for the H1 and H2
differences. Nothing is refitted or re-predicted.

    uv run python scripts/test_bootstrap.py              # 2,000 resamples
"""

from __future__ import annotations

import argparse

import duckdb
import numpy as np
from sklearn.metrics import f1_score

from finra_nlp.baseline import THRESHOLD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    con = duckdb.connect(a.db, read_only=True)
    df = con.execute("SELECT * FROM model.test_predictions").df()
    con.close()
    classes = [c.split(":", 1)[1] for c in df.columns if c.startswith("baseline:")]
    y = np.array([[int(c in (s or "").split("|")) for c in classes] for s in df["labels"]])
    pred = {m: (df[[f"{m}:{c}" for c in classes]].to_numpy() >= THRESHOLD).astype(int)
            for m in ("baseline", "finetune", "zeroshot")}
    support = y.sum(axis=0)

    def macros(idx):
        out = {}
        for m, p in pred.items():
            yy, pp = y[idx], p[idx]
            out[m] = f1_score(yy, pp, average="macro", zero_division=0)
            keep = support >= 10
            out[m + "_10"] = f1_score(yy[:, keep], pp[:, keep], average="macro", zero_division=0)
        return out

    rng = np.random.default_rng(a.seed)
    full = macros(np.arange(len(y)))
    draws = [macros(rng.integers(0, len(y), len(y))) for _ in range(a.n)]

    def ci(f):
        v = np.array([f(d) for d in draws])
        return f(full), np.percentile(v, 2.5), np.percentile(v, 97.5), float((v > 0).mean())

    rows = [
        ("Baseline", lambda d: d["baseline"]), ("Fine-tuned", lambda d: d["finetune"]),
        ("Zero-shot", lambda d: d["zeroshot"]),
        ("H1: fine-tuned minus baseline", lambda d: d["finetune"] - d["baseline"]),
        ("H2: fine-tuned minus zero-shot", lambda d: d["finetune"] - d["zeroshot"]),
        ("H1, 10+ series", lambda d: d["finetune_10"] - d["baseline_10"]),
        ("H2, 10+ series", lambda d: d["finetune_10"] - d["zeroshot_10"]),
    ]
    print(f"{len(y)} test cases, {a.n} paired resamples (macro-F1 over all series unless noted)\n")
    print(f"{'':34} {'value':>7} {'95% interval':>18} {'share > 0':>10}")
    for name, f in rows:
        v, lo, hi, pos = ci(f)
        print(f"{name:34} {v:7.3f} {lo:8.3f} to {hi:6.3f} {pos:10.2f}")


if __name__ == "__main__":
    main()
