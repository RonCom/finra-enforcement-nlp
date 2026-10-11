"""Facts sections of the case documents (spec H6): the classifier input for the full-document comparison.

The labels come from the documents' violation sentences, so a model reading the whole document would read the
answer. This keeps only the narrative of what happened: the text after the FACTS AND VIOLATIVE CONDUCT heading
(or another facts heading) up to the sanctions or conclusions of law, with "(Violation of ...)" sub-headings and
every sentence that states a violation removed. Rule numbers are masked as in the summaries, then every sentence that quotes or explains a
rule is removed (narrative()), so the model reads what happened. Cases whose facts come to under MIN_CHARS characters
are left out and counted. OCR text is used where the OCR pass read the document.

Writes model.facts (case_no, split, facts_text, chars, start). Reads cached documents only.

Usage:
    uv run python -m finra_nlp.facts
    uv run python -m finra_nlp.facts --show 5      # print five extracted sections to read
    uv run python -m finra_nlp.facts --dropped 8   # and what the rule filter removed from eight cases it left out
"""

from __future__ import annotations

import argparse
import re

import duckdb
import pandas as pd

from finra_nlp.citations import leaks, mask
from finra_nlp.dao import WAIVER_SECTION_RE
from finra_nlp.http import PoliteClient
from finra_nlp.monthly import pdf_text
from finra_nlp.ocr import cache_path

MIN_CHARS = 300
MAX_CHARS = 20000
HEADINGS = [  # tried in order; the first found starts the facts
    ("facts and violative conduct", re.compile(r"FACTS\s+AND\s+VIOLATIVE\s+CONDUCT")),
    ("findings of fact", re.compile(r"FINDINGS\s+OF\s+FACTS?|Findings\s+of\s+Fact\b")),
    ("facts", re.compile(r"\b(?:[IVX]+\.\s*|[A-H]\.\s*)?(?:FACTS|Facts)\b(?=\s+[A-Z])")),
]
# Where the facts end: the sanctions an AWC's respondent consents to, or a decision's conclusions and sanctions
# (OCR text drops spaces: "imposition ofthe following sanctions")
# OCR misreads letters too ("conscnt to thc imposition", "iinposition"). A decision's sanctions part opens with a
# "Sanctions" heading and cites the Sanction Guidelines.
END_RE = re.compile(r"\bto\s*th[ec]\s*i\w{0,2}position\b|\bf\w{2}lowing\s*sanctions|agrees?\s*to\s*pay\s*the\s*\w*onetary|"
                    r"CONCLUSIONS?\s+OF\s+LAW|Sanction\s+Guidelines|"
                    r"\b(?:[IVX]+\.\s*)?SANCTIONS\b|\b(?:[IVX]+\.\s*)?Sanctions\s+(?:[A-Z]\.|FINRA|The|In|Enforcement)")
# with OCR misspellings: "viola?d", "iolated"
STATED_RE = re.compile(r"\bviolat\w*|\bviola\S{0,3}d\b|\biolat\w*|\bin contravention of\b|"
                       r"\bby reason of th\w forcgoing|\bby reason of the foregoing", re.IGNORECASE)
SUBHEAD_RE = re.compile(r"\(\s*(?:Alleged\s+)?Violations?\s+of\b.{0,300}?\)", re.IGNORECASE)
# After masking, a sentence that quotes or explains a rule ("[RULE] prohibits registered persons from ...") says
# which rule was charged as plainly as its number would; a complaint or tip has nothing like it. Such a sentence names
# a rule ([RULE], "FINRA rules") and either uses a verb of obligation or quotes. A sentence that only mentions a rule
# ("FINRA sent a request under [RULE]") stays: with the number masked it tells the model no more than the conduct.
# Guidance (Regulatory Notices, Supplementary Material) goes either way, as does the BrokerCheck page footer.
# statute citations the masking missed in OCR text ("Exchange Act § I 7(a)") count too
RULE_REF_RE = re.compile(r"\[RULE\]|\b(?:FINRA|NASD|MSRB|SEC) rules?\b|\b(?:Exchange|Securities) Act\s*(?:Rule|§|Section)|"
                         r"§|C\.\s*F\.\s*R\b", re.IGNORECASE)
EXPLAIN_RE = re.compile(r"\b(?:requires?|required|requiring|provides?|prohibit(?:s|ed)?|states?|mandates?|obligates?|"
                        r"defines?|imposes?|appl(?:y|ies)|establish(?:es)?|governs?|must|shall|in pertinent part|"
                        r"is (?:designed|intended)|means|violat\w*|constitut\w*|inconsistent with)\b|[\"“”]", re.IGNORECASE)
GUIDANCE_RE = re.compile(r"Regulatory Notice|Notice to Members|Supplementary Material|BrokerCheck", re.IGNORECASE)
# PDF text often drops the space after a period ("procedures.The firm"), so a lowercase letter, a period and a
# capital also end a sentence. A closing quote can follow the period.
SENTENCE_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”]))\s+(?=[A-Z\"“(\[])|(?<=[a-z][.!?])(?=[A-Z])")


def rule_talk(sentence: str) -> bool:
    return bool(GUIDANCE_RE.search(sentence) or (RULE_REF_RE.search(sentence) and EXPLAIN_RE.search(sentence)))


def narrative(masked: str) -> str:
    """The masked facts with every sentence that quotes or explains a rule removed."""
    return " ".join(x for x in SENTENCE_RE.split(masked) if x and not rule_talk(x)).strip()


def facts_section(text: str) -> tuple[str, str]:
    """(facts text, start heading or '' if none found). The text is unmasked; sentences stating a violation are
    removed, and reading continues past them, since an AWC with several charges states each after its own facts."""
    text = re.sub(r"\s+", " ", text)
    cut = WAIVER_SECTION_RE.search(text)
    if cut:
        text = text[: cut.start()]
    start, kind = None, ""
    for name, pat in HEADINGS:
        m = pat.search(text)
        if m:
            start, kind = m.end(), name
            break
    if start is None:
        return "", ""
    body = text[start:]
    end = END_RE.search(body, 40)  # skip the "and Conclusions of Law" of a "Findings of Fact and ..." heading
    if end:
        body = body[: end.start()]
        # drop the "B. Respondent also" before the cut
        body = re.sub(r"\s+(?:[A-Z]|[IVX]+)\.$", "", body[: body.rfind(".") + 1] or body)
    body = SUBHEAD_RE.sub(" ", body)
    body = " ".join(x for x in SENTENCE_RE.split(body) if x and not STATED_RE.search(x))
    return body.strip()[:MAX_CHARS], kind


def document_text(client: PoliteClient, url) -> str:
    if not isinstance(url, str) or not url:  # no document link comes through pandas as NaN
        return ""
    ocr = cache_path(url)
    if ocr.exists():
        return ocr.read_text(encoding="utf-8")
    if not client.cached(url):
        return ""
    status, body = client.get(url)
    return pdf_text(body) if status == 200 and body.startswith(b"%PDF") else ""


def build(db: str, client: PoliteClient | None = None, show: int = 0, dropped_show: int = 0) -> pd.DataFrame:
    client = client or PoliteClient(cache_dir="data/cache/finra")
    con = duckdb.connect(db)
    cases = con.execute("""SELECT d.case_no, d.split, l.doc_url FROM model.dataset d
                           LEFT JOIN raw.case_document_labels l USING (case_no) ORDER BY d.case_no""").df()
    try:
        ocr_urls = dict(con.execute("SELECT case_no, doc_url FROM raw.case_ocr_labels").fetchall())
    except duckdb.CatalogException:
        ocr_urls = {}
    filtered_out = []
    rows, dropped = [], {"no document": 0, "no facts heading": 0, f"under {MIN_CHARS} characters": 0}
    for i, r in enumerate(cases.itertuples(index=False), 1):
        text = document_text(client, ocr_urls.get(r.case_no) or r.doc_url)
        if not text:
            dropped["no document"] += 1
            continue
        facts, kind = facts_section(text)
        if not kind:
            dropped["no facts heading"] += 1
            continue
        before = mask(facts)
        masked = narrative(before)
        if len(masked) < MIN_CHARS:
            dropped[f"under {MIN_CHARS} characters"] += 1
            if len(before) >= MIN_CHARS:
                filtered_out.append({"case_no": r.case_no, "before": len(before), "after": len(masked),
                                     "removed": [x for x in SENTENCE_RE.split(before) if x and rule_talk(x)]})
            continue
        rows.append({"case_no": r.case_no, "split": r.split, "facts_text": masked, "chars": len(masked), "start": kind})
        if i % 500 == 0:
            print(f"  {i}/{len(cases)} cases read, {len(rows)} with facts", flush=True)
    df = pd.DataFrame(rows, columns=["case_no", "split", "facts_text", "chars", "start"])
    con.execute("CREATE SCHEMA IF NOT EXISTS model")
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE model.facts AS SELECT * FROM df")
    con.close()

    print(f"\n{len(df)} of {len(cases)} dataset cases have a facts section; left out: "
          + ", ".join(f"{v} {k}" for k, v in dropped.items())
          + f" ({len(filtered_out)} of those were long enough before rule sentences were removed)")
    if len(df):
        print(df.groupby("split").agg(cases=("case_no", "size"), median_chars=("chars", "median")).to_string())
        print(df.start.value_counts().to_string())
        leaky = df.facts_text.map(lambda t: bool(leaks(t)))
        stated = df.facts_text.str.contains(r"\bviolat", case=False)
        sanctions = df.facts_text.str.contains(r"sanction|\bfine of\b|statutory disqualification", case=False)
        print(f"numbered rule references left: {int(leaky.sum())}; texts still saying 'violated'/'violation': "
              f"{int(stated.sum())}; texts mentioning sanctions or a fine: {int(sanctions.sum())}")
    if filtered_out:
        fo = pd.DataFrame(filtered_out)
        print(f"\nCases the rule filter took under {MIN_CHARS} characters: characters before, quartiles "
              f"{fo.before.quantile([.25, .5, .75]).astype(int).tolist()}; after, "
              f"{fo.after.quantile([.25, .5, .75]).astype(int).tolist()}; sentences removed, median "
              f"{int(fo.removed.str.len().median())}")
        for r in fo.sample(min(dropped_show, len(fo)), random_state=0).itertuples(index=False):
            print(f"\n--- removed from {r.case_no} ({r.before} -> {r.after} characters) ---")
            for x in r.removed[:6]:
                print(f"  * {x[:300]}")
    for r in df.sample(min(show, len(df)), random_state=0).itertuples(index=False) if show else []:
        print(f"\n--- {r.case_no} ({r.split}, {r.chars} characters, from '{r.start}') ---\n{r.facts_text[:1500]}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/finra.duckdb")
    ap.add_argument("--show", type=int, default=0, help="print this many extracted sections")
    ap.add_argument("--dropped", type=int, default=0,
                    help="print the removed sentences of this many cases the rule filter took under the minimum")
    a = ap.parse_args()
    build(a.db, show=a.show, dropped_show=a.dropped)


if __name__ == "__main__":
    main()
