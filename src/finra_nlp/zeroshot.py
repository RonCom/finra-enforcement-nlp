"""Zero-shot classifier (spec, "Classifier", H2): a local model through Ollama reads each masked summary
and gives, for every rule series, the probability that FINRA charged a rule in it. No training labels are used.

Each series is described by its Rulebook title and the rules most often charged in it, written by hand from
the Rulebook's table of contents (the Rulebook download is still pending). NASD rules are described under
their FINRA successors, as in the labels. Rule 2010 is left out, as in the labels.

Answers are stored per case, model, prompt version and reasoning setting in model.zeroshot_runs, so a stopped
run resumes where it left off. Scored with the baseline's metrics; writes reports/zeroshot_validation.md and
model.zeroshot_validation. Only the validation split is read.

Settings (shell): OLLAMA_MODEL (default gemma4:26b), OLLAMA_NUM_CTX (default 4096), OLLAMA_THINK (default
false: reasoning off), OLLAMA_URL.

Speed: each progress line shows prompt and answer tokens and their rates, so you can see which dominates.
--workers 2 sends two cases at once (start Ollama with OLLAMA_NUM_PARALLEL=2 or more); --recheck 20 asks 20
already-answered cases again under the current settings and reports how far the probabilities moved, without
storing anything, so a speed setting can be checked before it's used for the rest of the run.

Usage:
    uv run python -m finra_nlp.zeroshot --limit 5     # check the answers and the time per case first
    uv run python -m finra_nlp.zeroshot
    uv run python -m finra_nlp.zeroshot --recheck 20  # after changing a speed setting
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pandas as pd

from finra_nlp.baseline import MIN_SUPPORT, _labels, macro_supported, score

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
THINK = os.environ.get("OLLAMA_THINK", "false").lower() in ("1", "true", "yes")
PROMPT_VERSION = "v1"
WAIT_FOR_OLLAMA = 600  # seconds to keep retrying while Ollama starts or loads the model

SERIES = {
    "FINRA:1000": "Member application and associated person registration: registration and qualification "
                  "(1210, 1220, 1240 continuing education), Form U4/U5 filing and disclosure (1122), "
                  "membership changes (1017).",
    "FINRA:2000": "Duties and conflicts: suitability (2111), fraud and manipulation (2020), communications with "
                  "the public (2210), fair prices and commissions (2121), customer funds and securities (2150), "
                  "annuities (2330), options (2360), research analysts (2241), confirmations (2232).",
    "FINRA:3000": "Supervision and responsibilities of associated persons: supervision (3110, 3120, 3130), "
                  "outside business activities (3270), private securities transactions (3280), accounts at "
                  "other firms (3210), borrowing from or lending to customers (3240), discretionary accounts "
                  "(3260), anti-money laundering (3310), do-not-call (3230).",
    "FINRA:4000": "Financial and operational rules: books and records (4511), net capital (4110), margin "
                  "(4210), reporting requirements (4530), short interest reporting (4560).",
    "FINRA:5000": "Securities offering and trading standards: best execution (5310), new issue allocations "
                  "(5130, 5131), private placement filings (5122, 5123), publication of transactions (5210), "
                  "trading ahead of customer orders (5320).",
    "FINRA:6000": "Quotation, order and transaction reporting facilities: trade reporting to TRACE (6730), "
                  "OTC and ADF reporting (6380, 6622), Consolidated Audit Trail reporting (6800 series).",
    "FINRA:7000": "Clearing, transaction and order data: OATS order reporting (7450), trade reporting facility "
                  "reporting (7230, 7330).",
    "FINRA:8000": "Investigations and sanctions: failing to provide information, documents or testimony to "
                  "FINRA (8210).",
    "MSRB": "Municipal Securities Rulemaking Board rules for municipal securities and advisors: fair dealing "
            "(G-17), supervision (G-27), records (G-8, G-9), transaction reporting (G-14), pricing (G-30), "
            "political contributions (G-37).",
    "REG_NMS": "SEC Regulation NMS: order protection and trade-throughs (Rule 611), order routing and execution "
               "disclosure reports (Rules 605, 606).",
    "REG_SHO": "SEC Regulation SHO for short sales: order marking (Rule 200), locate requirement (Rule 203), "
               "close-out of fails to deliver (Rule 204), short sale price test (Rule 201).",
    "SA_SECTION": "Sections of the Securities Act of 1933: selling unregistered securities (Section 5), "
                  "fraud in the offer or sale of securities (Section 17(a)).",
    "SEC_RULE": "SEC rules under the Exchange Act: Rule 10b-5 (fraud), Rules 17a-3 and 17a-4 (making and "
                "preserving records), Rule 15c3-1 (net capital), Rule 15c3-3 (customer protection), Rule 15c3-5 "
                "(market access controls), Rule 15l-1 (Regulation Best Interest), Rule 17a-14 (Form CRS).",
    "SEC_SECTION": "Sections of the Securities Exchange Act of 1934: Section 10(b) (fraud), Section 15(c) "
                   "(broker-dealer conduct, net capital and customer protection), Section 17(a) (records).",
    "other": "Any rule outside the series above: exchange rules, FINRA arbitration codes, Regulation M, "
             "Regulation Crowdfunding, funding portal rules.",
}

SYSTEM = """You classify FINRA disciplinary actions by the rules FINRA charged. You read the monthly-report \
summary of one case; rule numbers in it are replaced by [RULE]. For every rule series listed, give the \
probability, from 0 to 1, that FINRA charged at least one rule in that series in this case. FINRA Rule 2010 \
(standards of commercial honor) is charged in nearly every case and is not one of the series. A case usually \
charges rules in one to three series. Answer with JSON only.

Rule series:
{series}"""


def _series_block(classes: list[str]) -> str:
    return "\n".join(f"- {c}: {SERIES.get(c, 'Rules in ' + c + '.')}" for c in classes)


def schema(classes: list[str]) -> dict:
    return {"type": "object", "properties": {c: {"type": "number", "minimum": 0, "maximum": 1} for c in classes},
            "required": classes}


def _post_waiting(client: httpx.Client, payload: dict) -> httpx.Response:
    start = time.monotonic()
    while True:
        try:
            resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            if resp.status_code < 500:
                return resp
        except httpx.TransportError:
            pass
        if time.monotonic() - start > WAIT_FOR_OLLAMA:
            raise SystemExit(f"Ollama at {OLLAMA_URL} didn't answer for {WAIT_FOR_OLLAMA} s; is it running?")
        time.sleep(5)


def classify(text: str, classes: list[str], client: httpx.Client, model: str = MODEL,
             stats: dict | None = None) -> dict[str, float]:
    """Probabilities per series. If stats is given, it gets Ollama's token counts and durations."""
    payload = {
        "model": model, "stream": False, "think": THINK, "format": schema(classes), "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
        "messages": [{"role": "system", "content": SYSTEM.format(series=_series_block(classes))},
                     {"role": "user", "content": text}],
    }
    resp = _post_waiting(client, payload)
    resp.raise_for_status()
    body = resp.json()
    if stats is not None:
        stats.update({k: body.get(k) for k in ("prompt_eval_count", "prompt_eval_duration", "eval_count",
                                               "eval_duration", "load_duration")})
    answer = json.loads(body["message"]["content"])
    return {c: min(1.0, max(0.0, float(answer.get(c, 0.0)))) for c in classes}


def _rate(n, ns) -> str:
    return f"{n} tok at {n / (ns / 1e9):.0f}/s" if n and ns else f"{n or 0} tok"


def recheck(db: str, n: int, client: httpx.Client, model: str = MODEL) -> None:
    """Ask n answered cases again under the current settings; report how far the probabilities moved."""
    con = duckdb.connect(db, read_only=True)
    rows = con.execute("""SELECT r.case_no, r.probs, d.masked_text FROM model.zeroshot_runs r
                          JOIN model.dataset d USING (case_no)
                          WHERE r.model = ? AND r.prompt_version = ? AND r.think = ? ORDER BY r.case_no LIMIT ?""",
                       [model, PROMPT_VERSION, THINK, n]).fetchall()
    con.close()
    diffs, flips, secs = [], 0, []
    for case_no, old, text in rows:
        old = json.loads(old)
        t0 = time.monotonic()
        new = classify(text or "", list(old), client, model)
        secs.append(time.monotonic() - t0)
        d = [abs(new[k] - old[k]) for k in old]
        diffs += d
        flips += sum((new[k] >= 0.5) != (old[k] >= 0.5) for k in old)
        print(f"  {case_no}: largest change {max(d):.2f}, {secs[-1]:.1f} s", flush=True)
    if rows:
        print(f"{len(rows)} cases asked again: mean change {np.mean(diffs):.3f}, largest {max(diffs):.2f}, "
              f"{flips} series flipped across 0.5 (of {len(diffs)}); median {np.median(secs):.1f} s per case")


def run(db: str, out: str, limit: int | None = None, client: httpx.Client | None = None,
        model: str = MODEL, workers: int = 1) -> tuple[pd.DataFrame, float] | None:
    client = client or httpx.Client(timeout=600)
    con = duckdb.connect(db)
    data = con.execute("SELECT case_no, split, masked_text, labels FROM model.dataset "
                       "WHERE split IN ('train', 'validation')").df()
    classes = sorted({lab for s in data[data.split == "train"]["labels"] for lab in _labels(s)})
    val = data[data.split == "validation"].sort_values("case_no").reset_index(drop=True)
    con.execute("CREATE SCHEMA IF NOT EXISTS model")
    con.execute("""CREATE TABLE IF NOT EXISTS model.zeroshot_runs (case_no VARCHAR, model VARCHAR,
                   prompt_version VARCHAR, think BOOLEAN, probs VARCHAR, seconds DOUBLE)""")
    key = (model, PROMPT_VERSION, THINK)
    done = {r[0] for r in con.execute("SELECT case_no FROM model.zeroshot_runs WHERE model = ? AND "
                                      "prompt_version = ? AND think = ?", list(key)).fetchall()}
    todo = [r for r in val.itertuples(index=False) if r.case_no not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(val)} validation cases, {len(done)} already answered by {model} (prompt {PROMPT_VERSION}, "
          f"reasoning {'on' if THINK else 'off'}); {len(todo)} to run", flush=True)
    def one(r):
        t0, stats = time.monotonic(), {}
        probs = classify(r.masked_text or "", classes, client, model, stats)
        return r, probs, time.monotonic() - t0, stats

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:  # results come back in order; DuckDB writes here
        for i, (r, probs, took, st) in enumerate(pool.map(one, todo), 1):
            con.execute("INSERT INTO model.zeroshot_runs VALUES (?, ?, ?, ?, ?, ?)",
                        [r.case_no, *key, json.dumps(probs), took])
            top = ", ".join(f"{c} {p:.2f}" for c, p in sorted(probs.items(), key=lambda kv: -kv[1])[:3])
            elapsed = time.monotonic() - start
            print(f"  {i}/{len(todo)} {r.case_no}: {took:.1f} s (prompt "
                  f"{_rate(st.get('prompt_eval_count'), st.get('prompt_eval_duration'))}, answer "
                  f"{_rate(st.get('eval_count'), st.get('eval_duration'))}) | true {r.labels or '-'} | top {top} | "
                  f"about {(len(todo) - i) * elapsed / i / 60:.0f} min left", flush=True)

    rows = dict(con.execute("SELECT case_no, probs FROM model.zeroshot_runs WHERE model = ? AND "
                            "prompt_version = ? AND think = ?", list(key)).fetchall())
    scored = val[val.case_no.isin(rows)]
    if len(scored) < len(val):
        print(f"{len(scored)} of {len(val)} validation cases answered; the report needs all of them")
        con.close()
        return None
    prob = np.array([[json.loads(rows[c]).get(k, 0.0) for k in classes] for c in scored.case_no])
    y = np.array([[int(k in _labels(s)) for k in classes] for s in scored["labels"]])
    per, macro = score(classes, y, prob)
    probs = pd.DataFrame(prob, columns=classes)
    probs.insert(0, "case_no", scored.case_no.values)
    probs.insert(1, "labels", scored["labels"].values)
    con.register("probs", probs)
    con.execute("CREATE OR REPLACE TABLE model.zeroshot_validation AS SELECT * FROM probs")
    secs = con.execute("SELECT median(seconds) FROM model.zeroshot_runs WHERE model = ? AND prompt_version = ? "
                       "AND think = ?", list(key)).fetchone()[0]
    con.close()
    m10, n10 = macro_supported(per)
    text = "\n".join([
        "# Zero-shot: local model through Ollama, validation split", "",
        f"Model {model}, prompt {PROMPT_VERSION}, reasoning {'on' if THINK else 'off'}, context {NUM_CTX}. "
        f"Validation cases: {len(scored)}. Median {secs:.1f} s per case. Threshold 0.5.", "",
        f"Macro-F1: {macro:.3f} (all {len(per)} series)", "",
        f"Macro-F1, series with {MIN_SUPPORT}+ validation cases: {m10:.3f} ({n10} series)", "",
        per.round(3).to_markdown(), "",
    ])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    return per, macro


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--out", default="reports/zeroshot_validation.md")
    ap.add_argument("--limit", type=int, help="answer at most this many more cases (to check the timing)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=1, help="cases sent at once; needs OLLAMA_NUM_PARALLEL >= this")
    ap.add_argument("--recheck", type=int, help="ask this many answered cases again and report the change")
    a = ap.parse_args()
    if a.recheck:
        recheck(a.db, a.recheck, httpx.Client(timeout=600), a.model)
    else:
        run(a.db, a.out, a.limit, model=a.model, workers=a.workers)


if __name__ == "__main__":
    main()
