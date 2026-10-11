"""Answers from the Rulebook index, with citations or a decline (spec, "RAG", H4 and H5).

The local model (Ollama, the zero-shot classifier's model) gets the top 5 retrieved rules' chunks (BM25, H3's retriever) and either
answers, citing each rule it relies on by number, or declines when the question isn't governed by FINRA rules or
the excerpts don't answer it. Citations of rules that aren't among the excerpts are dropped and counted.

    cases     answer case summaries ("which FINRA rules does this conduct violate?") on a split; the test split
              answers the 100-case H5 sample, once, under a freeze tag
    judge     the LLM judge labels each stored case answer correct or incorrect against FINRA's citations
    report    validation: decline rate, rule-level precision and recall against FINRA's citations, judge's rate
    h4        the 20 out-of-scope questions (docs/out_of_scope_questions.csv, keep = Y), once, under a freeze tag
    export    the H5 sample's answers as a CSV to label by hand, without the judge's labels
    kappa     Cohen's kappa between the judge and the hand labels (H5)
    smoke     one in-scope and one out-of-scope question over three stand-in excerpts, to check the local model

Answers and judge labels are stored in data/rag.duckdb (tables answers, judgments) and runs resume. Prompts are
versioned; changing one means a new version, developed on validation only.

Usage:
    uv run python -m finra_nlp.answer smoke                        # try the prompt on the local model, no index needed
    uv run python -m finra_nlp.answer cases --n 50                 # validation, 50 random cases
    uv run python -m finra_nlp.answer judge
    uv run python -m finra_nlp.answer report
    uv run python -m finra_nlp.answer h4                           # frozen code only
    uv run python -m finra_nlp.answer cases --split test           # frozen code only: the H5 sample
    uv run python -m finra_nlp.answer judge --split test
    uv run python -m finra_nlp.answer export                       # reports/h5_handlabels.csv: fill `label`
    uv run python -m finra_nlp.answer kappa --labels <filled csv>  # reports/h5_results.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import httpx
import pandas as pd

from finra_nlp import rag
from finra_nlp.zeroshot import MODEL, NUM_CTX, THINK, _post_waiting

ANSWER_PROMPT = "v2"  # --prompt v1 reruns or rescores the first version
JUDGE_PROMPT = "v1"
H5_N, H5_SEED = 100, 5
H4_QUESTIONS = "docs/out_of_scope_questions.csv"
H4_MIN_DECLINED = 18
H5_KAPPA = 0.60
H4_REPORT, H5_REPORT = "reports/h4_results.md", "reports/h5_results.md"
HANDLABELS = "reports/h5_handlabels.csv"

SYSTEM = {"v1": """You answer questions about the FINRA Rulebook, the rules FINRA sets for its member broker-dealers \
and their associated persons. You get numbered excerpts from the Rulebook; each begins with its rule number and \
title. Use only the excerpts.

- Cite each rule you rely on by the number shown in its excerpt header, such as "3110". Cite only rules that appear \
in the excerpts.
- When the user describes conduct from a disciplinary case, cite the rules the conduct violated.
- Decline when the question is governed by something other than FINRA's rules (SEC rules for investment advisers \
or public companies, federal statutes, state law, court decisions) or when the excerpts don't answer it. To \
decline, set "decline" to true, cite no rules, and give the reason in one sentence.

Reply in JSON.""",
          "v2": """You answer questions about the FINRA Rulebook, the rules FINRA sets for its member broker-dealers \
and their associated persons. You get numbered excerpts from the Rulebook; each begins with its rule number and \
title. Use only the excerpts.

- Cite each rule you rely on by the number shown in its excerpt header, such as "3110". Cite only rules that appear \
in the excerpts.
- When the user describes conduct from a FINRA disciplinary case, name the rules among the excerpts that the \
conduct violated. The same conduct may also break SEC rules or federal statutes; that is no reason to decline, \
cite the FINRA rules that apply. Decline only if no excerpt covers the conduct.
- For any other question, decline when it is governed by something other than FINRA's rules (SEC rules for \
investment advisers or public companies, federal statutes, state law, court decisions) or when the excerpts don't \
answer it.
- To decline, set "decline" to true, cite no rules, and give the reason in one sentence.

Reply in JSON."""}

JUDGE = {"v1": """You check an answer that names the FINRA rules a described conduct violated, against the rules \
FINRA actually charged in that case.

Label the answer "correct" when both hold:
1. At least one rule it cites is a rule FINRA charged. A NASD rule FINRA charged counts as its FINRA successor \
(NASD 3010 = FINRA 3110, NASD 2310 = FINRA 2111, NASD 3030 = FINRA 3270, NASD 3040 = FINRA 3280). Rule 2010 \
doesn't count: FINRA charges it in almost every case.
2. No rule it cites is unrelated to the conduct described. A rule FINRA didn't charge but that plainly covers the \
same conduct is not unrelated.
Otherwise label it "incorrect". An answer that declined is "incorrect".

Reply in JSON."""}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "decline": {"type": "boolean"},
        "rules": {"type": "array", "items": {"type": "object", "properties": {
            "rule": {"type": "string"}, "reason": {"type": "string"}}, "required": ["rule", "reason"]}},
        "answer": {"type": "string"},
    },
    "required": ["decline", "rules", "answer"],
}
JUDGE_SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["correct", "incorrect"]},
                                                 "reason": {"type": "string"}}, "required": ["label", "reason"]}
RULE_NO_RE = re.compile(r"IM-\d{4}-\d+|\d{4,5}")


def rule_no(text: str) -> str:
    """'FINRA Rule 3110(b)(2)' -> '3110'; '' if no rule number."""
    m = RULE_NO_RE.search(str(text).upper())
    return m.group(0) if m else ""


def excerpts(chunks: pd.DataFrame) -> str:
    return "\n\n".join(f"[{i}] Rule {c.rule_no}: {c.title}\n{c.text}"
                       for i, c in enumerate(chunks.itertuples(index=False), 1))


def case_question(summary: str) -> str:
    return ("Conduct from a FINRA disciplinary case (rule numbers removed):\n" + summary.strip()
            + "\n\nWhich FINRA rules did this conduct violate?")


def _chat(client: httpx.Client, system: str, user: str, schema: dict, model: str) -> dict:
    payload = {"model": model, "stream": False, "think": THINK, "format": schema, "keep_alive": "30m",
               "options": {"temperature": 0, "num_ctx": NUM_CTX},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    resp = _post_waiting(client, payload)
    resp.raise_for_status()
    return json.loads(resp.json()["message"]["content"])


def answer(question: str, chunks: pd.DataFrame, client: httpx.Client, model: str = MODEL,
           prompt: str | None = None) -> dict:
    """The model's answer, with cited rules kept only if they're among the excerpts' rules."""
    out = _chat(client, SYSTEM[prompt or ANSWER_PROMPT], f"Excerpts:\n\n{excerpts(chunks)}\n\nQuestion:\n{question}",
                ANSWER_SCHEMA, model)
    shown = {rag.base_rule(r) for r in chunks.rule_no}
    cited = [rule_no(r.get("rule", "")) for r in out.get("rules", [])]
    kept = list(dict.fromkeys(c for c in cited if c and rag.base_rule(c) in shown))
    decline = bool(out.get("decline")) or not kept
    return {"decline": decline, "rules": "" if decline else "|".join(kept),
            "outside": sum(1 for c in cited if c and rag.base_rule(c) not in shown),
            "reasons": json.dumps(out.get("rules", []), ensure_ascii=False), "answer": out.get("answer", ""),
            "raw": out}


# ---- storage ------------------------------------------------------------------------------------------------

def ensure_tables(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS answers (kind VARCHAR, item_id VARCHAR, split VARCHAR,
                   question VARCHAR, chunk_rules VARCHAR, decline BOOLEAN, rules VARCHAR, outside INTEGER,
                   reasons VARCHAR, answer VARCHAR, model VARCHAR, prompt VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS judgments (item_id VARCHAR, split VARCHAR, label VARCHAR,
                   reason VARCHAR, model VARCHAR, prompt VARCHAR, answer_prompt VARCHAR)""")
    # judgments made before answer prompts were versioned judged v1 answers
    con.execute("ALTER TABLE judgments ADD COLUMN IF NOT EXISTS answer_prompt VARCHAR DEFAULT 'v1'")


def _done(con, kind: str, model: str, prompt: str) -> set[str]:
    return {r[0] for r in con.execute("SELECT item_id FROM answers WHERE kind = ? AND model = ? AND prompt = ?",
                                      [kind, model, prompt]).fetchall()}


def _store(con, row: dict) -> None:
    cols = ["kind", "item_id", "split", "question", "chunk_rules", "decline", "rules", "outside", "reasons",
            "answer", "model", "prompt"]
    con.execute(f"INSERT INTO answers VALUES ({', '.join('?' * len(cols))})", [row[c] for c in cols])


def _answer_all(items: list[tuple[str, str, str, str]], kind: str, index, client, model: str, rag_db: str) -> None:
    """items: (item_id, split, question, retrieval query: the case summary or the question itself)."""
    con = duckdb.connect(rag_db)
    ensure_tables(con)
    done = _done(con, kind, model, ANSWER_PROMPT)
    todo = [it for it in items if it[0] not in done]
    print(f"{len(items) - len(todo)} already answered; answering {len(todo)}", flush=True)
    for i, (item_id, split, question, query) in enumerate(todo, 1):
        # a case asks which rules the conduct broke, so it searches the conduct rules; an H4 question searches all
        chunks = index.rank(query, rag.H3_RETRIEVER, rag.TOP_K, conduct_only=kind == "case")
        a = answer(question, chunks, client, model)
        _store(con, {"kind": kind, "item_id": item_id, "split": split, "question": question,
                     "chunk_rules": "|".join(chunks.rule_no), "model": model, "prompt": ANSWER_PROMPT, **a})
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
    con.close()


# ---- cases --------------------------------------------------------------------------------------------------

def scored_cases(db: str, split: str) -> pd.DataFrame:
    """Cases of a split charged with at least one FINRA Rulebook rule (H3's scored set), with gold rules."""
    con = duckdb.connect(db, read_only=True)
    cases = con.execute("SELECT case_no, masked_text, rules FROM model.dataset WHERE split = ? ORDER BY case_no",
                        [split]).df()
    con.close()
    cases["gold"] = cases.rules.map(rag.finra_rules)
    return cases[cases.gold.map(bool)].reset_index(drop=True)


def sample(cases: pd.DataFrame, split: str, n: int | None) -> pd.DataFrame:
    if split == "test":
        return cases.sample(min(H5_N, len(cases)), random_state=H5_SEED).sort_values("case_no")
    return cases.sample(min(n, len(cases)), random_state=0).sort_values("case_no") if n else cases


def run_cases(db: str, split: str = "validation", n: int | None = 50, rag_db: str = rag.RAG_DB, index=None,
              client: httpx.Client | None = None, model: str = MODEL, check=None) -> None:
    if split == "test":
        from finra_nlp.testrun import frozen
        (check or frozen)(H5_REPORT)
    cases = sample(scored_cases(db, split), split, n)
    items = [(r.case_no, split, case_question(rag.query_text(r.masked_text)), rag.query_text(r.masked_text))
             for r in cases.itertuples(index=False)]
    _answer_all(items, "case", index or rag.Index(rag_db), client or httpx.Client(timeout=600), model, rag_db)


def judge(db: str, split: str = "validation", rag_db: str = rag.RAG_DB, client: httpx.Client | None = None,
          model: str = MODEL, prompt: str = JUDGE_PROMPT) -> None:
    client = client or httpx.Client(timeout=600)
    gold = scored_cases(db, split).set_index("case_no")
    con = duckdb.connect(rag_db)
    ensure_tables(con)
    todo = con.execute("""SELECT a.item_id, a.question, a.decline, a.reasons, a.rules FROM answers a
                          WHERE a.kind = 'case' AND a.split = ? AND a.prompt = ? AND a.item_id NOT IN
                          (SELECT item_id FROM judgments WHERE split = ? AND model = ? AND prompt = ?
                           AND answer_prompt = ?)""",
                       [split, ANSWER_PROMPT, split, model, prompt, ANSWER_PROMPT]).df()
    print(f"judging {len(todo)} answers", flush=True)
    for r in todo.itertuples(index=False):
        if r.decline:
            label, reason = "incorrect", "declined"
        else:
            charged = gold.loc[r.item_id, "rules"].replace("|", ", ")
            cited = "\n".join(f"- Rule {x.get('rule')}: {x.get('reason')}" for x in json.loads(r.reasons)
                              if rule_no(x.get("rule", "")) in r.rules.split("|"))
            out = _chat(client, JUDGE[prompt], f"Conduct: {gold.loc[r.item_id, 'masked_text']}\n\n"
                        f"Rules FINRA charged: {charged}\n\nThe answer's cited rules:\n{cited}", JUDGE_SCHEMA, model)
            label, reason = out.get("label", "incorrect"), out.get("reason", "")
        con.execute("INSERT INTO judgments (item_id, split, label, reason, model, prompt, answer_prompt) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", [r.item_id, split, label, reason, model, prompt, ANSWER_PROMPT])
    con.close()


def _case_answers(db: str, split: str, rag_db: str, model: str) -> pd.DataFrame:
    ensure = duckdb.connect(rag_db)
    ensure_tables(ensure)
    ensure.close()
    con = duckdb.connect(rag_db, read_only=True)
    ans = con.execute("""SELECT a.item_id AS case_no, a.decline, a.rules AS cited, a.outside, a.chunk_rules,
                         a.answer, a.reasons, j.label AS judge FROM answers a LEFT JOIN judgments j
                         ON j.item_id = a.item_id AND j.split = a.split AND j.model = a.model AND j.prompt = ?
                         AND j.answer_prompt = a.prompt
                         WHERE a.kind = 'case' AND a.split = ? AND a.model = ? AND a.prompt = ?""",
                      [JUDGE_PROMPT, split, model, ANSWER_PROMPT]).df()
    con.close()
    return ans.merge(scored_cases(db, split)[["case_no", "masked_text", "rules", "gold"]], on="case_no")


def report(db: str, split: str = "validation", rag_db: str = rag.RAG_DB, model: str = MODEL,
           out: str | None = None) -> pd.DataFrame:
    df = _case_answers(db, split, rag_db, model)
    cited = df.cited.map(lambda s: {rag.base_rule(x) for x in (s or "").split("|") if x} - {"2010"})
    hits = [len(c & g) for c, g in zip(cited, df.gold)]
    n_cited, n_gold = sum(map(len, cited)), sum(map(len, df.gold))
    retrieved = [bool({rag.base_rule(x) for x in r.split("|")} & g) for r, g in zip(df.chunk_rules, df.gold)]
    lines = [f"# Answers, {split} split", "",
             f"{len(df)} cases charged with a FINRA Rulebook rule; model {model}, answer prompt {ANSWER_PROMPT}, "
             f"judge prompt {JUDGE_PROMPT}. Rule 2010 is left out of the counts.", "",
             "| Measure | Value |", "| --- | --- |",
             f"| Declined | {df.decline.mean():.3f} |",
             f"| A charged rule among the 5 retrieved chunks | {pd.Series(retrieved).mean():.3f} |",
             f"| At least one charged rule cited | {pd.Series(hits).gt(0).mean():.3f} |",
             f"| Cited rules that FINRA charged (precision) | {sum(hits) / n_cited if n_cited else 0:.3f} |",
             f"| Charged rules cited (recall) | {sum(hits) / n_gold if n_gold else 0:.3f} |",
             f"| Citations dropped as not in the excerpts | {int(df.outside.sum())} |",
             f"| Judge: correct | {(df.judge == 'correct').mean():.3f} ({int(df.judge.notna().sum())} judged) |", ""]
    df["retrieved"] = retrieved
    df["hits"] = hits
    gave_up = df[df.decline & df.retrieved]
    lines += [f"Declined although a charged rule was among the excerpts: {len(gave_up)} of {int(df.decline.sum())} "
              "declines.", "", "## Declines with a charged rule in the excerpts", ""]
    for r in gave_up.head(8).itertuples(index=False):
        lines.append(f"- **{r.case_no}** (charged {r.rules.replace('|', ', ')}; excerpts "
                     f"{r.chunk_rules.replace('|', ', ')}): {r.answer[:300]}")
    wrong = df[~df.decline & (df.hits == 0)]
    lines += ["", f"## Answers citing no charged rule ({len(wrong)})", ""]
    for r in wrong.head(8).itertuples(index=False):
        lines.append(f"- **{r.case_no}** (charged {r.rules.replace('|', ', ')}; cited {r.cited.replace('|', ', ')}; "
                     f"excerpts {r.chunk_rules.replace('|', ', ')})")
    lines.append("")
    path = out or f"reports/answers_{split}_{ANSWER_PROMPT}.md"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return df


# ---- H4 -----------------------------------------------------------------------------------------------------

def h4(rag_db: str = rag.RAG_DB, questions: str = H4_QUESTIONS, index=None, client: httpx.Client | None = None,
       model: str = MODEL, check=None, out: str = H4_REPORT) -> int:
    from finra_nlp.testrun import frozen
    freeze = (check or frozen)(out)
    qs = pd.read_csv(questions)
    qs = qs[qs.keep.astype(str).str.upper().eq("Y")]
    _answer_all([(r.id, "h4", r.question, r.question) for r in qs.itertuples(index=False)], "h4", index or rag.Index(rag_db),
                client or httpx.Client(timeout=600), model, rag_db)
    con = duckdb.connect(rag_db, read_only=True)
    ans = con.execute("SELECT * FROM answers WHERE kind = 'h4' AND model = ? AND prompt = ?",
                      [model, ANSWER_PROMPT]).df()
    con.close()
    df = qs.merge(ans.drop(columns="question"), left_on="id", right_on="item_id")
    declined = int(df.decline.sum())
    lines = ["# H4: out-of-scope questions", "",
             f"Code: {freeze}. Model {model}, answer prompt {ANSWER_PROMPT}. {len(df)} questions, reviewed by hand "
             "before the run (spec change log).", "",
             "| ID | Test | Expectation | Result | Verdict |", "| --- | --- | --- | --- | --- |",
             f"| H4 | Out-of-scope questions declined | >= {H4_MIN_DECLINED} of {len(df)} | {declined} of {len(df)} | "
             f"{'Met' if declined >= H4_MIN_DECLINED else 'Failed'} |", ""]
    answered = df[~df.decline]
    if len(answered):
        lines += ["## Answered instead of declined", ""]
        for r in answered.itertuples(index=False):
            lines += [f"### {r.id}: {r.question}", "", f"Governed by: {r.governing_source}. Cited: {r.rules}. "
                      f"Retrieved chunks' rules: {r.chunk_rules.replace('|', ', ')}.", "", f"> {r.answer}", ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return declined


# ---- H5 -----------------------------------------------------------------------------------------------------

def export(db: str, rag_db: str = rag.RAG_DB, model: str = MODEL, out: str = HANDLABELS) -> pd.DataFrame:
    """The test sample's answers to label by hand, blind to the judge (labels: correct / incorrect)."""
    df = _case_answers(db, "test", rag_db, model)
    con = duckdb.connect(rag_db, read_only=True)
    reasons = dict(con.execute("SELECT item_id, reasons FROM answers WHERE kind = 'case' AND split = 'test' "
                               "AND model = ? AND prompt = ?", [model, ANSWER_PROMPT]).fetchall())
    con.close()
    rows = pd.DataFrame({
        "case_no": df.case_no, "summary": df.masked_text, "finra_charged": df.rules.str.replace("|", ", "),
        "declined": df.decline,
        "answer_rules": [" | ".join(f"{x.get('rule')}: {x.get('reason')}" for x in json.loads(reasons[c]))
                         if not d else "" for c, d in zip(df.case_no, df.decline)],
        "label": "", "notes": ""})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, index=False)
    print(f"{len(rows)} answers written to {out}. Fill `label` with correct or incorrect, by the judge's rule: "
          "at least one cited rule FINRA charged (NASD rules as their FINRA successors, 2010 not counting) and no "
          "cited rule unrelated to the conduct; a declined answer is incorrect.")
    return rows


def kappa(db: str, labels: str, rag_db: str = rag.RAG_DB, model: str = MODEL, out: str = H5_REPORT) -> float:
    from sklearn.metrics import cohen_kappa_score
    hand = pd.read_csv(labels, dtype={"case_no": str})
    hand["label"] = hand.label.astype(str).str.strip().str.lower()
    bad = hand[~hand.label.isin(["correct", "incorrect"])]
    if len(bad):
        raise SystemExit(f"{len(bad)} rows have no label of correct or incorrect: {bad.case_no.tolist()[:10]}")
    df = hand[["case_no", "label"]].merge(_case_answers(db, "test", rag_db, model)[["case_no", "judge"]], on="case_no")
    if df.judge.isna().any() or len(df) != len(hand):
        raise SystemExit("The judge hasn't labeled every hand-labeled answer; run `answer judge --split test`")
    k = cohen_kappa_score(df.label, df.judge)
    agree = (df.label == df.judge).mean()
    lines = ["# H5: LLM judge vs. hand labels", "",
             f"{len(df)} test answers (random, seed {H5_SEED}), labeled by hand blind to the judge. Judge: {model}, "
             f"prompt {JUDGE_PROMPT}.", "",
             "| ID | Test | Expectation | Result | Verdict |", "| --- | --- | --- | --- | --- |",
             f"| H5 | LLM judge vs. hand labels, Cohen's kappa | >= {H5_KAPPA} | {k:.3f} (agreement {agree:.3f}) | "
             f"{'Met' if k >= H5_KAPPA else 'Failed: judge scores unreliable; hand labels used'} |", "",
             f"Hand labels: {(df.label == 'correct').mean():.3f} correct. Judge: {(df.judge == 'correct').mean():.3f} "
             "correct.", "",
             pd.crosstab(df.label.rename("hand"), df.judge.rename("judge")).to_markdown(), ""]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return k


# ---- smoke test --------------------------------------------------------------------------------------------

# Paraphrased stand-ins for three Rulebook rules (not FINRA's text), so the prompt and schema can be tried on the
# local model before the Rulebook is downloaded
SMOKE_CHUNKS = pd.DataFrame([
    {"rule_no": "3110", "title": "Supervision", "text": "Each member firm must have a system to supervise the "
     "activities of its registered representatives and associated persons, with written procedures, designated "
     "principals and reviews of correspondence and transactions, reasonably designed to achieve compliance."},
    {"rule_no": "8210", "title": "Provision of Information and Testimony and Inspection and Copying of Books",
     "text": "FINRA staff may require a member or associated person to provide information and documents and to "
     "testify under oath in an investigation or examination. A person may not fail to comply with such a request."},
    {"rule_no": "3270", "title": "Outside Business Activities of Registered Persons", "text": "A registered "
     "person may not be employed by or receive compensation from any outside business activity without first "
     "giving the member firm written notice."},
])
SMOKE_QUESTIONS = [
    ("in scope", case_question("FINRA sent the respondent two written requests for documents and for on-the-record "
                               "testimony in an investigation of his termination. He acknowledged receiving them "
                               "and refused to provide the documents or appear.")),
    ("out of scope", "How soon after a client signs an advisory contract must a registered investment adviser "
                     "deliver its Form ADV Part 2A brochure?"),
]


def smoke(client: httpx.Client | None = None, model: str = MODEL) -> list[dict]:
    """Ask the local model one in-scope and one out-of-scope question over three stand-in excerpts; print its raw
    JSON and what `answer` keeps. Expected: 8210 cited for the first, a decline for the second."""
    import time
    client = client or httpx.Client(timeout=600)
    results = []
    for kind, q in SMOKE_QUESTIONS:
        start = time.monotonic()
        a = answer(q, SMOKE_CHUNKS, client, model)
        print(f"--- {kind} ({time.monotonic() - start:.0f} s) ---")
        print("raw:", json.dumps(a["raw"], indent=2, ensure_ascii=False))
        print(f"kept: decline={a['decline']} rules={a['rules'] or '-'} dropped={a['outside']}\n")
        results.append(a)
    ok = results[0]["rules"] == "8210" and results[1]["decline"]
    print("as expected" if ok else "NOT as expected: in scope should cite 8210 only, out of scope should decline")
    return results


def main() -> None:
    global ANSWER_PROMPT
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--split", default="validation", choices=["validation", "test"])
    c.add_argument("--n", type=int, default=50, help="validation cases to answer, at random (test: the H5 sample)")
    j = sub.add_parser("judge")
    j.add_argument("--split", default="validation", choices=["validation", "test"])
    sub.add_parser("report")
    sub.add_parser("h4")
    sub.add_parser("smoke")
    sub.add_parser("export")
    k = sub.add_parser("kappa")
    k.add_argument("--labels", required=True)
    for p in (c, j, sub.choices["report"], sub.choices["export"], k):
        p.add_argument("--db", default="data/finra.duckdb")
    for p in sub.choices.values():
        p.add_argument("--prompt", choices=sorted(SYSTEM), default=ANSWER_PROMPT, help="answer prompt version")
    a = ap.parse_args()
    ANSWER_PROMPT = a.prompt
    if a.cmd == "cases":
        run_cases(a.db, a.split, a.n)
    elif a.cmd == "judge":
        judge(a.db, a.split)
    elif a.cmd == "report":
        report(a.db)
    elif a.cmd == "smoke":
        smoke()
    elif a.cmd == "h4":
        h4()
    elif a.cmd == "export":
        export(a.db)
    else:
        kappa(a.db, a.labels)


if __name__ == "__main__":
    main()
