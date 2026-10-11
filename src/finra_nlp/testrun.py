"""Frozen test run for the classifiers (spec, H1 and H2): the test split (2024-2026) is scored once.

Two steps, both refused unless HEAD carries a freeze-* tag and tracked files are unchanged:

    uv run python -m finra_nlp.testrun zeroshot --workers 2   # the local model answers the test cases (resumes)
    uv run python -m finra_nlp.testrun score                  # baseline, fine-tuned average, zero-shot; once
    uv run python -m finra_nlp.testrun h6                     # facts-section model vs. summary model (H6); once

The zero-shot prompt used is the one with the higher validation macro-F1 over series with 10+ cases; both
prompts must have answered every validation case. Progress lines don't show test labels. `score` fits the
baseline on the train split, averages the three saved fine-tuned seeds, reads the zero-shot answers, and writes
reports/test_results.md with H1 and H2 judged on macro-F1 over all series (as pre-registered), macro-F1 over
series with 10+ test cases next to it, and per-series tables. It refuses to run if that report exists.

Stop Ollama (ollama stop <model>) before `score`; the fine-tuned models need the GPU.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pandas as pd

from finra_nlp import baseline, finetune, zeroshot
from finra_nlp.baseline import _labels, macro_supported, score

REPORT = "reports/test_results.md"
H1_GAIN = 0.05   # fine-tuned minus baseline, test macro-F1
H2_GAP = 0.10    # fine-tuned minus zero-shot, test macro-F1
MIN_TEST = 10    # series with at least this many test cases enter the second macro-F1


def frozen(report: str = REPORT) -> str:
    """The freeze tag at HEAD; exits if there is none, tracked files changed, or the test has been scored."""
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(f"Uncommitted changes to tracked files; the test split is scored only on frozen code:\n{dirty}")
    tags = [t for t in git("tag", "--points-at", "HEAD").split() if t.startswith("freeze")]
    if not tags:
        raise SystemExit("HEAD has no freeze-* tag. Freeze first: git tag freeze-1, then git push origin freeze-1")
    if Path(report).exists():
        raise SystemExit(f"{report} exists: the test split has been scored. It is scored once.")
    return f"{tags[0]} at {git('rev-parse', '--short', 'HEAD')}"


def _data(con):
    data = con.execute("SELECT case_no, split, masked_text, labels FROM model.dataset").df()
    classes = sorted({lab for s in data[data.split == "train"]["labels"] for lab in _labels(s)})
    return data, classes


def chosen_prompt(con, classes, model: str = zeroshot.MODEL) -> tuple[str, dict]:
    """The zero-shot prompt with the higher validation macro-F1 over 10+ series, and each prompt's score."""
    data, _ = _data(con)
    val = data[data.split == "validation"].sort_values("case_no")
    y = np.array([[int(c in _labels(s)) for c in classes] for s in val["labels"]])
    scores = {}
    for p in sorted(zeroshot.PROMPTS):
        prob = zeroshot.stored(con, val.case_no.tolist(), classes, p, model)
        if prob is None:
            raise SystemExit(f"Prompt {p} hasn't answered every validation case; run "
                             f"uv run python -m finra_nlp.zeroshot --prompt {p} first")
        scores[p] = macro_supported(score(classes, y, prob)[0])[0]
    return max(scores, key=scores.get), scores


def answer_test(db: str, workers: int = 1, client: httpx.Client | None = None, model: str = zeroshot.MODEL,
                check=frozen) -> None:
    check()
    con = duckdb.connect(db)
    zeroshot.ensure_table(con)
    data, classes = _data(con)
    prompt, scores = chosen_prompt(con, classes, model)
    zeroshot.PROMPT_VERSION = prompt
    test = data[data.split == "test"].sort_values("case_no")
    done = {r[0] for r in con.execute("SELECT case_no FROM model.zeroshot_runs WHERE model = ? AND prompt_version = ? "
                                      "AND think = ?", [model, prompt, zeroshot.THINK]).fetchall()}
    todo = [r for r in test.itertuples(index=False) if r.case_no not in done]
    print(f"Prompt {prompt} (validation macro-F1 over 10+ series: "
          f"{', '.join(f'{k} {v:.3f}' for k, v in scores.items())}); {len(test)} test cases, {len(todo)} to run",
          flush=True)
    zeroshot.answer_cases(con, todo, classes, client or httpx.Client(timeout=600), model, workers, show_labels=False)
    con.close()


def score_test(db: str, out: str = REPORT, model: str = zeroshot.MODEL, check=frozen, ensemble=None) -> dict:
    freeze = check(out)
    con = duckdb.connect(db)
    data, classes = _data(con)
    prompt, zs_scores = chosen_prompt(con, classes, model)
    test = data[data.split == "test"].sort_values("case_no").reset_index(drop=True)
    zs = zeroshot.stored(con, test.case_no.tolist(), classes, prompt, model)
    if zs is None:
        raise SystemExit("The zero-shot model hasn't answered every test case; run "
                         "uv run python -m finra_nlp.testrun zeroshot first")
    con.close()  # the fine-tuned step opens the database read-only; DuckDB won't mix that with this connection
    train = data[data.split == "train"]
    vec, clf, mlb = baseline.fit(train)
    if list(mlb.classes_) != classes:
        raise SystemExit("Baseline series differ from the dataset's training series")
    base = clf.predict_proba(vec.transform(test.masked_text.fillna("")))
    ft_classes, ft_cases, _, ft = (ensemble or finetune.ensemble_probs)(db, "test")
    if ft_classes != classes or ft_cases.case_no.tolist() != test.case_no.tolist():
        raise SystemExit("The saved fine-tuned models were trained on different series or cases")
    y = np.array([[int(c in _labels(s)) for c in classes] for s in test["labels"]])

    results, tables = {}, {}
    for name, prob in [("TF-IDF baseline", base), ("Fine-tuned (average of 3 seeds)", ft),
                       (f"Zero-shot ({model}, prompt {prompt})", zs)]:
        per, macro = score(classes, y, prob)
        m10, n10 = macro_supported(per, MIN_TEST)
        results[name] = {"macro_f1": macro, "macro_f1_10plus": m10}
        tables[name] = per
    names = list(results)
    b, f, z = (results[n]["macro_f1"] for n in names)
    h1 = f - b
    h2 = f - z
    summary = pd.DataFrame(results).T
    probs = pd.DataFrame({"case_no": test.case_no, "labels": test["labels"]})
    for name, prob in [("baseline", base), ("finetune", ft), ("zeroshot", zs)]:
        for j, c in enumerate(classes):
            probs[f"{name}:{c}"] = prob[:, j]
    con = duckdb.connect(db)
    con.register("probs", probs)
    con.execute("CREATE OR REPLACE TABLE model.test_predictions AS SELECT * FROM probs")
    con.close()

    lines = [
        "# Test split results (2024-2026)", "", f"Code: {freeze}. Test cases: {len(test)}. Threshold 0.5.", "",
        "Labels: the hand-checks put 3-5% of cases with a wrong series (spec change log); read differences under "
        "about 0.02 against that.", "",
        summary.round(3).rename(columns={"macro_f1": "macro-F1, all series",
                                         "macro_f1_10plus": f"macro-F1, series with {MIN_TEST}+ test cases"}).to_markdown(), "",
        "| ID | Test | Expectation | Result | Verdict |", "| --- | --- | --- | --- | --- |",
        f"| H1 | Fine-tuned vs. baseline, test macro-F1 | Gain >= {H1_GAIN} | {h1:+.3f} ({f:.3f} vs {b:.3f}) | "
        f"{'Met' if h1 >= H1_GAIN else 'Failed'} |",
        f"| H2 | Zero-shot vs. fine-tuned, test macro-F1 | Zero-shot lower by >= {H2_GAP} | {h2:+.3f} ({z:.3f} vs {f:.3f}) | "
        f"{'Met' if h2 >= H2_GAP else 'Failed: within 0.10, labeled data adds little'} |", "",
        f"Zero-shot prompt chosen on validation (macro-F1 over 10+ series): "
        f"{', '.join(f'{k} {v:.3f}' for k, v in zs_scores.items())}.", "",
    ]
    for name in names:
        lines += [f"## {name}", "", tables[name].round(3).to_markdown(), ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return {"h1": h1, "h2": h2, **{n: r for n, r in results.items()}}


H6_REPORT = "reports/h6_results.md"
H6_GAIN = 0.03   # facts-section model minus summary model, macro-F1 on the sample
H6_N = 300
FACTS_DIRS = ["data/models/facts", "data/models/facts_seed1", "data/models/facts_seed2"]


def _bootstrap_diff(y, a, b, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of macro-F1(b) - macro-F1(a) over cases: 2.5th and 97.5th percentiles, share above 0."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    pa, pb = (a >= baseline.THRESHOLD).astype(int), (b >= baseline.THRESHOLD).astype(int)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], pb[i], average="macro", zero_division=0)
                 - f1_score(y[i], pa[i], average="macro", zero_division=0))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def score_h6(db: str, out: str = H6_REPORT, check=frozen, facts_probs=None) -> dict:
    """H6: the fine-tuned model reading facts sections vs. the one reading summaries, on 300 test cases with a
    facts section (fixed random sample). Summary predictions are the stored ones from `score`."""
    freeze = check(out)
    con = duckdb.connect(db, read_only=True)
    pool = [r[0] for r in con.execute("SELECT case_no FROM model.facts WHERE split = 'test' ORDER BY case_no").fetchall()]
    stored = con.execute("SELECT * FROM model.test_predictions").df()
    con.close()
    rng = np.random.default_rng(6)
    sample = sorted(rng.choice(pool, size=min(H6_N, len(pool)), replace=False).tolist())
    classes = [c.split(":", 1)[1] for c in stored.columns if c.startswith("finetune:")]
    stored = stored.set_index("case_no").loc[sample]
    summ = stored[[f"finetune:{c}" for c in classes]].to_numpy()
    f_classes, f_cases, _, facts = (facts_probs or finetune.ensemble_probs)(
        db, "test", FACTS_DIRS, 1024, 8, source="facts", case_nos=sample)
    if f_classes != classes or f_cases.case_no.tolist() != sample:
        raise SystemExit("The facts models were trained on different series, or the sample cases don't line up")
    y = np.array([[int(c in _labels(s)) for c in classes] for s in stored["labels"]])
    per_s, macro_s = score(classes, y, summ)
    per_f, macro_f = score(classes, y, facts)
    gain = macro_f - macro_s
    lo, hi, pos = _bootstrap_diff(y, summ, facts)
    lines = [
        "# H6: facts section vs. monthly summary as input (test split)", "",
        f"Code: {freeze}. {len(sample)} test cases with a facts section, drawn at random (seed 6) from {len(pool)}. "
        "Both models are ModernBERT-base, fine-tuned the same way, average of 3 seeds; one read summaries "
        "(stored predictions from the test run), one read facts sections (max 1,024 tokens). Threshold 0.5.", "",
        "| Input | Macro-F1, all series | Macro-F1, series with 10+ cases in the sample |", "| --- | --- | --- |",
        f"| Monthly summary | {macro_s:.3f} | {macro_supported(per_s, MIN_TEST)[0]:.3f} |",
        f"| Facts section | {macro_f:.3f} | {macro_supported(per_f, MIN_TEST)[0]:.3f} |", "",
        "| ID | Test | Expectation | Result | Verdict |", "| --- | --- | --- | --- | --- |",
        f"| H6 | Facts section vs. summary, macro-F1 | Gain >= {H6_GAIN} | {gain:+.3f} (95% interval {lo:+.3f} to "
        f"{hi:+.3f}, {pos:.0%} of resamples above 0) | {'Met' if gain >= H6_GAIN else 'Failed: summaries are enough for triage'} |",
        "", "## Summary input", "", per_s.round(3).to_markdown(), "", "## Facts input", "", per_f.round(3).to_markdown(), "",
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return {"summary": macro_s, "facts": macro_f, "gain": gain}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zeroshot")
    z.add_argument("--db", default="data/finra.duckdb")
    z.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("--db", default="data/finra.duckdb")
    h = sub.add_parser("h6")
    h.add_argument("--db", default="data/finra.duckdb")
    a = ap.parse_args()
    if a.cmd == "zeroshot":
        answer_test(a.db, a.workers)
    elif a.cmd == "h6":
        score_h6(a.db)
    else:
        score_test(a.db)


if __name__ == "__main__":
    main()
