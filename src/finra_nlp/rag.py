"""Retrieval over the FINRA Rulebook (spec, "RAG", H3), adapted from the policy-rag pipeline.

`index` splits each downloaded Rulebook page (data/rulebook.duckdb, raw.rulebook_pages) into ~250-word chunks
that keep their rule number, embeds them with a local Ollama embedding model, and stores them in data/rag.duckdb.
Three retrievers rank chunks for a query: BM25 (exact words), dense (embeddings) and hybrid (reciprocal rank
fusion of the two), as in policy-rag.

`retrieval` scores H3 on a split: the query is the masked case summary, and a case is a hit if any FINRA rule the
case was charged with (NASD rules counted as their FINRA successors; Rule 2010 and procedural rules left out)
is the rule of one of the top 5 chunks. Cases whose charges include no FINRA rule (only SEC rules, MSRB rules and
the like, which aren't in the Rulebook) are left out and counted. Validation is for development; the test split
is scored once under a freeze tag (`retrieval --split test`).

Settings (shell): OLLAMA_URL, RAG_EMBED_MODEL (default nomic-embed-text).

Usage:
    uv run python -m finra_nlp.rag index --show 5             # print five rules' first chunks to check the text
    uv run python -m finra_nlp.rag retrieval                  # validation split
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import duckdb
import httpx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from finra_nlp.dao import procedural
from finra_nlp.labels import CATCH_ALL, _set, series_label, successor

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
RULEBOOK_DB = "data/rulebook.duckdb"
RAG_DB = "data/rag.duckdb"
MAX_WORDS, OVERLAP = 250, 40
# Section headings ("2120. Commissions, Mark Ups and Charges") have no text of their own, and reserved or deleted
# rules a line ("Deleted by SR-FINRA-2010-060 eff. Dec. 15, 2010."); the shortest real rules run 8 words ("All
# witnesses must testify under oath or affirmation.")
MIN_WORDS = 8
TOP_K = 5
RETRIEVERS = ["bm25", "dense", "hybrid"]
# H3 is judged on BM25, the best of the three on every validation run (spec change log); the answers use it too
H3_RETRIEVER = "bm25"
H3_LINE = 0.70
STOP = set(("a an and are as at be by for from has have if in into is it its of on or that the their this to was "
            "were which with not no may can will shall should must any such other").split())


# ---- chunks -------------------------------------------------------------------------------------------------

# The rule page's tab labels come first, then the rule, then its amendment history ("Amended by SR-FINRA-2008-021
# eff. ...", "Selected Notice: 07-07") and the empty tabs ("No Results Found")
TABS_RE = re.compile(r"^(?:The Rule\s+)?(?:Notices\s+)?(?:Guidance\s+)?(?:News Releases\s+)?(?:FAQs\s+)?")
HISTORY_RE = re.compile(r"\s(?:(?:Amended|Adopted|Redesignated|Renumbered|Approved)(?: and \w+)? by SR-|"
                        r"Selected Notices?:|No Results Found)")


def rule_body(text: str, title: str) -> str:
    """The rule text of a Rulebook page: from the title's last appearance up to the amendment history or footer."""
    t = re.sub(r"\s+", " ", text)
    i = t.rfind(title) if title else -1
    if i >= 0:
        t = t[i + len(title):]
    t = TABS_RE.sub("", t.strip())
    # without the title to cut at, the page's navigation is still in front; don't mistake it for the footer
    first = 0 if i >= 0 else 200
    for end in ("Rulebook Search", "Related Topics", "Sign Up for Newsletters", "©"):
        j = t.find(end)
        if j >= first:
            t = t[:j]
    h = HISTORY_RE.search(t)
    if h and h.start() > 0:
        t = t[:h.start()]
    return t.strip()


def split_words(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP) -> list[str]:
    w = text.split()
    if len(w) <= max_words:
        return [text]
    step = max_words - overlap
    return [" ".join(w[i:i + max_words]) for i in range(0, max(1, len(w) - overlap), step)]


# Series pages (0100, 1000, ..., 9000) list the titles of the rules in the series; no case is charged under them
SERIES_PAGE_RE = re.compile(r"^(?:0100|\d000)$")


def chunk_pages(pages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in pages.itertuples(index=False):
        if SERIES_PAGE_RE.match(str(p.rule_no)):
            continue
        body = rule_body(p.text or "", p.title or "")
        if len(body.split()) < MIN_WORDS:
            continue
        for j, piece in enumerate(split_words(body)):
            rows.append({"rule_no": str(p.rule_no).upper(), "title": p.title, "url": p.url, "part": j,
                         "text": piece})
    df = pd.DataFrame(rows, columns=["rule_no", "title", "url", "part", "text"])
    df.insert(0, "chunk_id", range(len(df)))
    return df


# ---- models -------------------------------------------------------------------------------------------------

class Ollama:
    def __init__(self, url: str = OLLAMA_URL, model: str = EMBED_MODEL, client: httpx.Client | None = None):
        self.url, self.model, self.client = url.rstrip("/"), model, client or httpx.Client(timeout=600)

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        # nomic-embed-text expects task prefixes; other embedding models ignore them
        prefix = {"query": "search_query: ", "document": "search_document: "}.get(kind, "") \
            if "nomic" in self.model else ""
        out = []
        for i in range(0, len(texts), 32):
            r = self.client.post(f"{self.url}/api/embed", json={"model": self.model,
                                                                "input": [prefix + t for t in texts[i:i + 32]]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_index(rulebook_db: str = RULEBOOK_DB, rag_db: str = RAG_DB, embedder=None, show: int = 0,
                model: str | None = None) -> pd.DataFrame:
    con = duckdb.connect(rulebook_db, read_only=True)
    pages = con.execute("SELECT rule_no, title, url, text FROM raw.rulebook_pages").df()
    con.close()
    df = chunk_pages(pages)
    embedder = embedder or Ollama(model=model or EMBED_MODEL)
    emb = embedder.embed((df["title"] + ". " + df["text"]).tolist(), "document")
    df["embedding"] = list(emb)
    Path(rag_db).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(rag_db)
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE chunks AS SELECT * FROM df")
    # queries must be embedded by the model that embedded the chunks; Index reads it from here
    con.execute("CREATE OR REPLACE TABLE meta AS SELECT 'embed_model' AS key, ? AS value",
                [getattr(embedder, "model", "unknown")])
    con.close()
    print(f"Embeddings: {getattr(embedder, 'model', 'unknown')}. {len(df)} chunks from {df.rule_no.nunique()} rules ({len(pages)} pages) in {rag_db}; median "
          f"{int(df.text.str.split().str.len().median())} words per chunk")
    for r in df[df.part == 0].sample(min(show, len(df)), random_state=0).itertuples(index=False) if show else []:
        print(f"\n--- {r.rule_no}: {r.title} ---\n{r.text[:800]}")
    return df


# ---- retrieval ----------------------------------------------------------------------------------------------

class BM25:
    """Okapi BM25 as one sparse matrix, so a query scores every chunk in one product."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.vec = CountVectorizer(token_pattern=r"[a-z0-9]+(?:[.-][a-z0-9]+)*", lowercase=True,
                                   stop_words=list(STOP))
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.w = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: str) -> np.ndarray:
        q = (self.vec.transform([query]) > 0).astype(np.float32).T
        return (self.w @ q).toarray().ravel()


class Index:
    def __init__(self, rag_db: str = RAG_DB, embedder=None):
        con = duckdb.connect(rag_db, read_only=True)
        self.chunks = con.execute("SELECT chunk_id, rule_no, title, url, part, text FROM chunks ORDER BY chunk_id").df()
        emb = con.execute("SELECT embedding FROM chunks ORDER BY chunk_id").fetchnumpy()["embedding"]
        try:
            self.model = con.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()[0]
        except duckdb.CatalogException:  # an index built before the model was recorded
            self.model = EMBED_MODEL
        con.close()
        self.emb = np.stack([np.asarray(e, np.float32) for e in emb])
        self.bm25 = BM25((self.chunks["title"] + " " + self.chunks["text"]).tolist())
        self.embedder = embedder or Ollama(model=self.model)

    def scores(self, query: str, method: str, qvec: np.ndarray | None = None) -> np.ndarray:
        if method == "bm25":
            return self.bm25.scores(query)
        qv = qvec if qvec is not None else self.embedder.embed([query], "query")[0]
        if method == "dense":
            return self.emb @ qv
        if method == "hybrid":  # reciprocal rank fusion, k = 60
            s = np.zeros(len(self.chunks))
            for part in (self.bm25.scores(query), self.emb @ qv):
                order = np.argsort(-part, kind="stable")
                s[order] += 1.0 / (60 + np.arange(1, len(order) + 1))
            return s
        raise ValueError(method)

    def rank(self, query: str, method: str = "hybrid", k: int = TOP_K, qvec=None,
             conduct_only: bool = False) -> pd.DataFrame:
        """The top k chunks from k different rules: each rule is represented by its best-scoring chunk, so one long
        rule can't fill the list. conduct_only keeps to the conduct rules (0100-8000), the only part of the
        Rulebook cases are charged under."""
        s = self.scores(query, method, qvec)
        order = np.argsort(-s, kind="stable")
        rules = self.chunks.rule_no.map(base_rule).to_numpy()
        conduct = self.chunks.rule_no.map(rulebook_part).eq(PARTS[0]).to_numpy()
        top, seen = [], set()
        for j in order:
            if conduct_only and not conduct[j]:
                continue
            if rules[j] not in seen:
                seen.add(rules[j])
                top.append(j)
                if len(top) == k:
                    break
        top = np.array(top, dtype=int)
        return self.chunks.iloc[top].assign(score=s[top], rank=np.arange(1, len(top) + 1))


def base_rule(rule_no: str) -> str:
    """'3110.05' (a Supplementary Material page) -> '3110'."""
    return str(rule_no).upper().split(".")[0]


# A monthly summary opens with the sanction ("An AWC was issued in which the firm was censured and fined $50,000")
# and ends with when it takes effect; those sentences match FINRA's rules on sanctions (8310, 8320, 9552), not the
# rules broken. The conduct follows "the entry of findings that".
SANCTION_Q_RE = re.compile(
    r"\b(?:censur\w*|fined|fines?|suspen(?:ded|sions?)|barred|bar|sanction\w*|restitution|disgorg\w*|AWC|"
    r"Acceptance, Waiver|in effect|appeal\w*|Hearing Panel|National Adjudicatory|Hearing Officers?|FINRA Case|"
    r"default decision|requalify|ordered to pay)\b", re.IGNORECASE)
FINDINGS_RE = re.compile(r"\bfindings that\s+(.*)", re.IGNORECASE | re.DOTALL)
QUERY_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def query_text(summary: str) -> str:
    """The conduct in a masked summary: sanction and procedure sentences dropped, "findings that X" cut to X."""
    out = []
    for sent in QUERY_SENTENCE_RE.split(re.sub(r"\s+", " ", summary or "").strip()):
        m = FINDINGS_RE.search(sent)
        if m:
            out.append(m.group(1))
        elif not SANCTION_Q_RE.search(sent):
            out.append(sent)
    q = " ".join(out).strip()
    return q or (summary or "")


PARTS = ["conduct rules (0100-8000)", "Code of Procedure (9000)", "Uniform Practice Code (11000)",
         "arbitration and mediation codes (12000-14000)"]


def rulebook_part(rule_no: str) -> str:
    m = re.match(r"(?:IM-)?(\d+)", str(rule_no))
    n = int(m.group(1)) if m else 0
    return PARTS[0] if n < 9000 else PARTS[1] if n < 10000 else PARTS[2] if n < 12000 else PARTS[3]


def finra_rules(rules: str) -> set[str]:
    """Rule numbers a case was charged with that the Rulebook holds: FINRA rules and NASD rules' successors,
    without Rule 2010 and procedural rules."""
    out = set()
    for k in _set(rules):
        k = successor(k)
        if k.startswith("FINRA:") and k != CATCH_ALL and not procedural(k):
            out.add(k.split(":", 1)[1].upper())
    return out


def retrieval(db: str, split: str = "validation", rag_db: str = RAG_DB, out: str | None = None, index=None,
              check=None) -> pd.DataFrame:
    if split == "test":
        from finra_nlp.testrun import frozen
        freeze = (check or frozen)(out or "reports/retrieval_test.md")
    con = duckdb.connect(db, read_only=True)
    cases = con.execute("SELECT case_no, masked_text, rules FROM model.dataset WHERE split = ? ORDER BY case_no",
                        [split]).df()
    con.close()
    cases["gold"] = cases.rules.map(finra_rules)
    scored = cases[cases.gold.map(bool)].reset_index(drop=True)
    idx = index or Index(rag_db)
    in_book = set(idx.chunks.rule_no.map(base_rule))
    queries = scored.masked_text.fillna("").map(query_text)
    qvecs = idx.embedder.embed(queries.tolist(), "query")
    rows = []
    for i, r in enumerate(scored.itertuples(index=False)):
        row = {"case_no": r.case_no, "words": len((r.masked_text or "").split()),
               "series": sorted({series_label("FINRA:" + g) for g in r.gold}),
               "gold_in_book": bool(r.gold & in_book)}
        for m in RETRIEVERS:
            hits = idx.rank(queries[i], m, TOP_K, qvecs[i], conduct_only=True)
            ranks = [k + 1 for k, rule in enumerate(hits.rule_no) if base_rule(rule) in r.gold]
            row[f"{m}_hit"] = bool(ranks)
            row[f"{m}_rr"] = 1.0 / ranks[0] if ranks else 0.0
            row[f"{m}_top"] = [base_rule(x) for x in hits.rule_no]
        rows.append(row)
    res = pd.DataFrame(rows)
    gold_parts = scored.gold.explode().map(rulebook_part).value_counts()
    lines = [f"# Retrieval (H3), {split} split", "",
             f"{len(cases)} cases; {len(scored)} charged at least one FINRA Rulebook rule other than 2010 and are scored "
             f"({len(cases) - len(scored)} charged only SEC, MSRB or other rules and are left out). "
             f"{int((~res.gold_in_book).sum()) if len(res) else 0} scored cases have no charged rule among the "
             f"{len(in_book)} rules indexed. Embeddings: {idx.model}. Query: the masked summary without its sanction and procedure sentences. Searched: the conduct rules (0100-8000). Hit: a charged rule is among the top {TOP_K} rules (each rule by its best chunk).",
             "", "Charged rules by part of the Rulebook: " + ", ".join(f"{p} {int(gold_parts.get(p, 0))}" for p in PARTS)
             + ".",
             "", "| Retriever | Hit rate, top 5 | MRR |", "| --- | --- | --- |"]
    for m in RETRIEVERS:
        lines.append(f"| {m} | {res[f'{m}_hit'].mean():.3f} | {res[f'{m}_rr'].mean():.3f} |")
    if split == "test":
        hit = res[f"{H3_RETRIEVER}_hit"].mean()
        lines += ["", f"Code: {freeze}.", "", "| ID | Test | Expectation | Result | Verdict |", "| --- | --- | --- | --- | --- |",
                  f"| H3 | A charged rule among the top {TOP_K} rules, {H3_RETRIEVER} | >= {H3_LINE:.0%} | {hit:.1%} | "
                  f"{'Met' if hit >= H3_LINE else 'Failed'} |"]
    if len(res):
        # thirds by summary length; ranking first keeps three bins when many summaries share a length
        res["length"] = pd.qcut(res.words.rank(method="first"), min(3, len(res)),
                                labels=["short", "medium", "long"][:min(3, len(res))])
        by_len = res.groupby("length", observed=True).agg(cases=("case_no", "size"),
                                                          hybrid_hit=("hybrid_hit", "mean"))
        ser = res.explode("series").groupby("series").agg(cases=("case_no", "size"), hybrid_hit=("hybrid_hit", "mean"))
        lines += ["", "## Hybrid hit rate by summary length", "", by_len.round(3).to_markdown(), "",
                  "## Hybrid hit rate by charged series", "", ser.round(3).to_markdown(), ""]
        lines += ["## What fills the top 5", "", "Share of all top-5 slots by part of the Rulebook:", "",
                  "| Part | " + " | ".join(RETRIEVERS) + " |", "| --- |" + " --- |" * len(RETRIEVERS)]
        parts = {m: res[f"{m}_top"].explode().map(rulebook_part).value_counts(normalize=True) for m in RETRIEVERS}
        for part in PARTS:
            lines.append(f"| {part} | " + " | ".join(f"{parts[m].get(part, 0):.3f}" for m in RETRIEVERS) + " |")
        lines += ["", "Rules retrieved most often (share of scored cases with the rule in their top 5):", ""]
        for m in RETRIEVERS:
            top = res[f"{m}_top"].map(set).explode().value_counts().head(10) / len(res)
            lines.append(f"- **{m}:** " + ", ".join(f"{r} {v:.0%}" for r, v in top.items()))
        lines.append("")
    path = out or f"reports/retrieval_{split}.md"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return res


def unindexed(rulebook_db: str = RULEBOOK_DB, show: int = 100) -> pd.DataFrame:
    """Rules with a page but no chunk (rule text under MIN_WORDS words, or a series page), with what their text says."""
    con = duckdb.connect(rulebook_db, read_only=True)
    pages = con.execute("SELECT rule_no, title, url, text FROM raw.rulebook_pages").df()
    con.close()
    kept = set(chunk_pages(pages).rule_no)
    left = pages[~pages.rule_no.astype(str).str.upper().isin(kept)].copy()
    left["body"] = [rule_body(t or "", ti or "") for t, ti in zip(left.text, left.title)]
    print(f"{len(left)} of {len(pages)} pages have no chunk")
    for r in left.sort_values("rule_no").head(show).itertuples(index=False):
        print(f"  {r.rule_no}: {r.title[:60]} | {r.body[:100]!r}")
    return left


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    i.add_argument("--rulebook-db", default=RULEBOOK_DB)
    i.add_argument("--show", type=int, default=0, help="print the first chunk of this many rules")
    i.add_argument("--model", default=EMBED_MODEL, help="Ollama embedding model (default from RAG_EMBED_MODEL)")
    u = sub.add_parser("unindexed", help="list the Rulebook pages that produced no chunk")
    u.add_argument("--rulebook-db", default=RULEBOOK_DB)
    r = sub.add_parser("retrieval")
    r.add_argument("--db", default="data/finra.duckdb")
    r.add_argument("--split", default="validation", choices=["validation", "test"])
    a = ap.parse_args()
    if a.cmd == "index":
        build_index(a.rulebook_db, show=a.show, model=a.model)
    elif a.cmd == "unindexed":
        unindexed(a.rulebook_db)
    else:
        retrieval(a.db, a.split)


if __name__ == "__main__":
    main()
