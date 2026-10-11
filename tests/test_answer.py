import json

import duckdb
import httpx
import pandas as pd
import pytest

from finra_nlp import answer, rag
from test_rag import FakeEmbedder, _dataset, _rulebook


def _ollama(log):
    """Answers like the local model: supervision -> 3110 (and an invented 9999), testimony -> 8210,
    records -> 4511 and 2010, an adviser question -> decline. The judge calls 3110 and 8210 correct."""
    def handle(req):
        body = json.loads(req.content)
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        log.append(system[:20])
        q = user.rsplit("Question:", 1)[-1].lower()
        if system == answer.JUDGE["v1"]:
            ok = "Rule 3110" in user or "Rule 8210" in user
            out = {"label": "correct" if ok else "incorrect", "reason": "x"}
        elif "adviser" in q:
            out = {"decline": True, "rules": [], "answer": "Advisers Act, not FINRA."}
        elif "supervise" in q:
            out = {"decline": False, "rules": [{"rule": "FINRA Rule 3110(a)", "reason": "supervision"},
                                               {"rule": "9999", "reason": "made up"}], "answer": "3110"}
        elif "testimony" in q:
            out = {"decline": False, "rules": [{"rule": "8210", "reason": "testimony"}], "answer": "8210"}
        else:
            out = {"decline": False, "rules": [{"rule": "4511", "reason": "records"}], "answer": "4511"}
        return httpx.Response(200, json={"message": {"content": json.dumps(out)}})
    return httpx.Client(transport=httpx.MockTransport(handle))


@pytest.fixture
def env(tmp_path):
    rb, rdb = _rulebook(tmp_path), str(tmp_path / "rag.duckdb")
    rag.build_index(rb, rdb, FakeEmbedder())
    return {"db": _dataset(tmp_path), "rag_db": rdb, "index": rag.Index(rdb, FakeEmbedder()), "log": [],
            "tmp": tmp_path}


def test_rule_no():
    assert answer.rule_no("FINRA Rule 3110(b)(2)") == "3110" and answer.rule_no("IM-2110-2") == "IM-2110-2"
    assert answer.rule_no("the conduct rule") == ""


def test_answer_drops_rules_not_in_excerpts(env):
    chunks = env["index"].rank("failed to supervise", "hybrid", 3)
    a = answer.answer(answer.case_question("failed to supervise the branch"), chunks, _ollama(env["log"]))
    assert a == {**a, "decline": False, "rules": "3110", "outside": 1}


def test_cases_judge_report(env):
    client = _ollama(env["log"])
    answer.run_cases(env["db"], "validation", None, env["rag_db"], env["index"], client)
    answer.run_cases(env["db"], "validation", None, env["rag_db"], env["index"], client)  # resumes: nothing new
    answer.judge(env["db"], "validation", env["rag_db"], client)
    df = answer.report(env["db"], "validation", env["rag_db"], out=str(env["tmp"] / "a.md"))
    assert sorted(df.case_no) == ["a1", "a2", "a3"]  # a4 charged no FINRA rule
    assert dict(zip(df.case_no, df.judge)) == {"a1": "correct", "a2": "correct", "a3": "incorrect"}
    text = (env["tmp"] / "a.md").read_text()
    assert "| At least one charged rule cited | 1.000 |" in text and "| Citations dropped as not in the excerpts | 1 |" in text
    assert sum(1 for s in env["log"] if s.startswith("You answer")) == 3


def test_h4_counts_declines_and_lists_answered(env):
    qs = env["tmp"] / "q.csv"
    pd.DataFrame({"id": ["Q1", "Q2", "Q3"], "question": ["When must an adviser deliver a brochure?",
                                                         "Who must supervise a branch?", "dropped"],
                  "governing_source": ["Advisers Act", "x", "x"], "why_finra_rules_dont_answer": "x",
                  "keep": ["Y", "Y", "N"]}).to_csv(qs, index=False)
    out = env["tmp"] / "h4.md"
    n = answer.h4(env["rag_db"], str(qs), env["index"], _ollama(env["log"]), check=lambda r: "freeze-x",
                  out=str(out))
    assert n == 1
    text = out.read_text()
    assert "| 1 of 2 | Failed |" in text and "### Q2" in text and "Q3" not in text


def test_test_split_sample_export_and_kappa(env, monkeypatch):
    monkeypatch.setattr(answer, "H5_N", 1)
    checked = []
    client = _ollama(env["log"])
    answer.run_cases(env["db"], "test", None, env["rag_db"], env["index"], client, check=checked.append)
    assert checked == [answer.H5_REPORT]
    answer.judge(env["db"], "test", env["rag_db"], client)
    labels = env["tmp"] / "h.csv"
    rows = answer.export(env["db"], env["rag_db"], out=str(labels))
    assert list(rows.case_no) == ["a5"] and "judge" not in rows.columns
    rows.assign(label="correct").to_csv(labels, index=False)
    k = answer.kappa(env["db"], str(labels), env["rag_db"], out=str(env["tmp"] / "h5.md"))
    assert "agreement 1.000" in (env["tmp"] / "h5.md").read_text()
    rows.assign(label="").to_csv(labels, index=False)
    with pytest.raises(SystemExit, match="no label"):
        answer.kappa(env["db"], str(labels), env["rag_db"], out=str(env["tmp"] / "h5b.md"))


def test_smoke(capsys):
    log = []
    res = answer.smoke(_ollama(log))
    assert res[0]["decline"] is False and res[0]["rules"] == "8210" and res[1]["decline"] is True
    assert "as expected" in capsys.readouterr().out


def test_old_judgments_table_gets_answer_prompt(tmp_path):
    con = duckdb.connect(str(tmp_path / "r.duckdb"))
    con.execute("""CREATE TABLE judgments (item_id VARCHAR, split VARCHAR, label VARCHAR, reason VARCHAR,
                   model VARCHAR, prompt VARCHAR)""")
    con.execute("INSERT INTO judgments VALUES ('a1', 'validation', 'correct', 'x', 'm', 'v1')")
    answer.ensure_tables(con)
    assert con.execute("SELECT answer_prompt FROM judgments").fetchone()[0] == "v1"
    con.close()
