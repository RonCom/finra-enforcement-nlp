import duckdb
import numpy as np
import pandas as pd

from finra_nlp import rag

WORDS = {"supervise": 0, "supervisory": 0, "branch": 0, "cooperate": 1, "testimony": 1, "information": 1,
         "books": 2, "records": 2}


class FakeEmbedder:
    """Bag of topic words: supervision, cooperation, records."""

    def embed(self, texts, kind):
        v = np.full((len(texts), 3), 0.01, dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(".", " ").split():
                if w in WORDS:
                    v[i, WORDS[w]] += 1
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _page(rule_no, title, body):
    return {"rule_no": rule_no, "title": title, "url": f"https://www.finra.org/rules/{rule_no}",
            "text": f"Menu Home Rules {title} {body} Rulebook Search footer links ©"}


PAGES = [
    _page("3110", "3110. Supervision", "Each member shall establish and maintain a system to supervise the activities "
          "of each registered representative and branch office, with written supervisory procedures. " * 3),
    _page("8210", "8210. Provision of Information and Testimony", "Persons shall provide information and testimony "
          "and cooperate with requests made under this rule, in writing or electronically. " * 3),
    _page("4511", "4511. General Requirements", "Members shall make and preserve books and records as required "
          "under the FINRA rules and the Exchange Act. " * 3),
    _page("0110", "0110. Short", "Too short to index."),
    _page("3000", "3000. SUPERVISION AND RESPONSIBILITIES", "3110. Supervision 3120. Supervisory Control System "
          "3130. Annual Certification of Compliance and Supervisory Processes " * 3),
]


def _rulebook(tmp_path):
    db = str(tmp_path / "rb.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.register("df", pd.DataFrame(PAGES))
    con.execute("CREATE TABLE raw.rulebook_pages AS SELECT * FROM df")
    con.close()
    return db


def _dataset(tmp_path):
    db = str(tmp_path / "f.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA model")
    df = pd.DataFrame([
        ("a1", "validation", "failed to supervise a branch office", "NASD:3010|FINRA:2010"),
        ("a2", "validation", "refused to provide testimony and information", "FINRA:8210|FINRA:2010"),
        ("a3", "validation", "did not preserve books and records", "FINRA:4511"),
        ("a4", "validation", "short sales without a locate", "SEC_RULE:203|FINRA:2010"),
        ("a5", "test", "failed to supervise", "FINRA:3110"),
    ], columns=["case_no", "split", "masked_text", "rules"])
    con.register("df", df)
    con.execute("CREATE TABLE model.dataset AS SELECT * FROM df")
    con.close()
    return db


def test_rule_body_drops_navigation_and_footer():
    p = PAGES[0]
    body = rag.rule_body(p["text"], p["title"])
    assert body.startswith("Each member") and "Rulebook Search" not in body and "Menu" not in body


def test_split_words_overlaps():
    text = " ".join(str(i) for i in range(600))
    parts = rag.split_words(text, 250, 40)
    assert [len(p.split()) for p in parts] == [250, 250, 180]
    assert parts[1].split()[0] == "210"
    assert rag.split_words("a b c") == ["a b c"]


def test_finra_rules_maps_nasd_and_drops_2010_and_procedural():
    assert rag.finra_rules("NASD:3010|FINRA:2010|FINRA:9552|FINRA:0140|SEC_RULE:17a-4") == {"3110"}
    assert rag.finra_rules("SEC_RULE:203|MSRB:G-27") == set()


def test_bm25_ranks_matching_chunk_first():
    bm = rag.BM25(["supervise the branch office", "books and records", "testimony and information"])
    assert int(np.argmax(bm.scores("preserve records"))) == 1


def test_index_and_retrieval(tmp_path):
    rb, rdb = _rulebook(tmp_path), str(tmp_path / "rag.duckdb")
    chunks = rag.build_index(rb, rdb, FakeEmbedder())
    assert sorted(chunks.rule_no.unique()) == ["3110", "4511", "8210"]  # 0110 too short; 3000 is a series page
    idx = rag.Index(rdb, FakeEmbedder())
    for m in rag.RETRIEVERS:
        assert idx.rank("refused to give testimony", m, 1).rule_no.iloc[0] == "8210"
    out = tmp_path / "r.md"
    res = rag.retrieval(_dataset(tmp_path), "validation", rdb, str(out), idx)
    assert list(res.case_no) == ["a1", "a2", "a3"]  # a4 charged no FINRA rule
    assert res.hybrid_hit.all() and res.dense_rr.eq(1).all()
    assert "| hybrid | 1.000 |" in out.read_text()


def test_test_split_requires_freeze(tmp_path):
    called = []
    rb, rdb = _rulebook(tmp_path), str(tmp_path / "rag.duckdb")
    rag.build_index(rb, rdb, FakeEmbedder())
    rag.retrieval(_dataset(tmp_path), "test", rdb, str(tmp_path / "t.md"), rag.Index(rdb, FakeEmbedder()),
                  check=called.append)
    assert called == [str(tmp_path / "t.md")]


def test_rule_body_drops_tabs_and_history():
    text = ("Menu 12505. Cooperation of Parties in Discovery The Rule Notices Guidance News Releases FAQs The parties "
            "must cooperate to the fullest extent practicable in the exchange of documents. Amended by "
            "SR-FINRA-2008-021 eff. Dec. 15, 2008. Adopted by SR-NASD-2003-158 eff. April 16, 2007. Selected Notice: "
            "07-07 , 08-57 . No Results Found No Results Found")
    assert rag.rule_body(text, "12505. Cooperation of Parties in Discovery") == \
        "The parties must cooperate to the fullest extent practicable in the exchange of documents."
