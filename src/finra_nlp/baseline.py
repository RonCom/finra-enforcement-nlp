"""Baseline classifier (spec, "Classifier"): TF-IDF and one-vs-rest logistic regression.

Trains on the train split of model.dataset and scores the validation split. The test split is
not read. Writes reports/baseline_validation.md with macro-F1 (all series, and series with 10+
validation cases), per-series F1 and the Brier score
per series, and model.baseline_validation with each validation case's predicted probabilities.

Usage:
    uv run python -m finra_nlp.baseline --db data/finra.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

THRESHOLD = 0.5


def _labels(s: str) -> list[str]:
    return [x for x in (s or "").split("|") if x]


def fit(train: pd.DataFrame, seed: int = 0):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(train["labels"].map(_labels))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    x = vec.fit_transform(train.masked_text.fillna(""))
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, random_state=seed))
    clf.fit(x, y)
    return vec, clf, mlb


MIN_SUPPORT = 10  # macro-F1 is also reported over series with at least this many validation cases


def macro_supported(per: pd.DataFrame, min_support: int = MIN_SUPPORT) -> tuple[float, int]:
    """Macro-F1 over series with enough validation cases that one case doesn't swing it; and their count."""
    kept = per[per.support >= min_support]
    return (float(kept.f1.mean()) if len(kept) else float("nan")), len(kept)


def evaluate(vec, clf, mlb, df: pd.DataFrame) -> tuple[pd.DataFrame, float, np.ndarray]:
    y = mlb.transform(df["labels"].map(_labels))
    prob = clf.predict_proba(vec.transform(df.masked_text.fillna("")))
    pred = (prob >= THRESHOLD).astype(int)
    rows = []
    for j, name in enumerate(mlb.classes_):
        rows.append({"series": name, "support": int(y[:, j].sum()), "predicted": int(pred[:, j].sum()),
                     "f1": f1_score(y[:, j], pred[:, j], zero_division=0),
                     "brier": brier_score_loss(y[:, j], prob[:, j]) if len(df) else float("nan")})
    per = pd.DataFrame(rows).set_index("series")
    macro = f1_score(y, pred, average="macro", zero_division=0)
    return per, macro, prob


def run(db: str, out: str) -> tuple[pd.DataFrame, float]:
    con = duckdb.connect(db)
    data = con.execute("SELECT * FROM model.dataset WHERE split IN ('train', 'validation')").df()
    train, val = data[data.split == "train"], data[data.split == "validation"]
    vec, clf, mlb = fit(train)
    per, macro, prob = evaluate(vec, clf, mlb, val)

    probs = pd.DataFrame(prob, columns=mlb.classes_)
    probs.insert(0, "case_no", val.case_no.values)
    probs.insert(1, "labels", val["labels"].values)
    con.register("probs", probs)
    con.execute("CREATE OR REPLACE TABLE model.baseline_validation AS SELECT * FROM probs")
    con.close()

    text = "\n".join([
        "# Baseline: TF-IDF + one-vs-rest logistic regression, validation split", "",
        f"Train cases: {len(train)}. Validation cases: {len(val)}. Threshold {THRESHOLD}.", "",
        f"Macro-F1: {macro:.3f} (all {len(per)} series)", "",
        f"Macro-F1, series with {MIN_SUPPORT}+ validation cases: {macro_supported(per)[0]:.3f} "
        f"({macro_supported(per)[1]} series)", "",
        per.round(3).to_markdown(), "",
    ])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    return per, macro


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--out", default="reports/baseline_validation.md")
    a = ap.parse_args()
    run(a.db, a.out)


if __name__ == "__main__":
    main()
