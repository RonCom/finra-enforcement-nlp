import duckdb
import pandas as pd
import pytest

from finra_nlp import baseline, dataset, labels, masking, rulebook


def _db(tmp_path, rows):
    db = str(tmp_path / "f.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    df = pd.DataFrame(rows, columns=["case_no", "action_date", "report_month", "masked_text", "doc_rules"])
    df["summary_rules"] = df.case_no.map({"c5": "FINRA:3110|FINRA:2010"}).fillna("")
    con.register("df", df)
    con.execute("""CREATE TABLE raw.monthly_cases AS
                   SELECT case_no, action_date, report_month, masked_text, summary_rules FROM df""")
    con.execute("CREATE TABLE raw.case_document_labels AS SELECT case_no, doc_rules FROM df")
    con.close()
    return db


def _cases():
    rows = []
    for i in range(80):
        year = 2018 if i < 60 else 2023
        if i % 2:
            rows.append((f"a{i}", f"{year}-03-01", f"{year}-04", "failed to supervise the [RULE] branch office",
                         "FINRA:3110|FINRA:2010"))
        else:
            rows.append((f"b{i}", f"{year}-03-01", f"{year}-04", "unsuitable recommendations of leveraged funds",
                         "FINRA:2111|FINRA:2010"))
    rows += [("c1", None, "2019-06", "late filing of a Form [RULE] amendment", "NASD:1000|FINRA:2010"),
             ("c2", "2024-02-01", "2024-03", "conduct inconsistent with just and equitable principles", "FINRA:2010"),
             ("c3", "2015-02-01", "2015-03", "too early", "FINRA:3110"),
             ("c4", "2018-02-01", "2018-03", "no document labels", ""),
             ("c5", "2017-02-01", "2017-03", "document unreadable; the summary cites the rules", "")]
    return rows


def test_dataset(tmp_path):
    df = dataset.build(_db(tmp_path, _cases()))
    by = df.set_index("case_no")
    assert "c3" not in by.index and "c4" not in by.index
    assert by.loc["c1", "split"] == "train" and by.loc["c1", "labels"] == "other"  # one NASD:1000 case
    assert by.loc["c2", "split"] == "test" and by.loc["c2", "labels"] == ""  # 2010 only
    assert by.loc["a1", "labels"] == "FINRA:3000" and by.loc["a61", "split"] == "validation"
    assert by.loc["a1", "label_source"] == "document"
    assert by.loc["c5", "label_source"] == "summary" and by.loc["c5", "labels"] == "FINRA:3000"


def test_baseline(tmp_path):
    db = _db(tmp_path, _cases())
    dataset.build(db)
    per, macro = baseline.run(db, str(tmp_path / "b.md"))
    assert per.loc["FINRA:3000", "f1"] == 1.0 and per.loc["FINRA:2000", "f1"] == 1.0
    assert per.loc["FINRA:3000", "brier"] < 0.1
    m, n = baseline.macro_supported(per, min_support=5)
    assert n == len(per[per.support >= 5]) and 0 <= m <= 1
    con = duckdb.connect(db, read_only=True)
    assert con.execute("SELECT count(*) FROM model.baseline_validation").fetchone()[0] == 20


def test_masking_check(tmp_path):
    db = _db(tmp_path, [("x1", None, "2020-01", "violated [RULE] and [RULE]", ""),
                        ("x2", None, "2020-01", "violated FINRA Rule 3110 and [RULE]", "")])
    bad = masking.check(db)
    assert [c for c, _ in bad] == ["x2"]
    with pytest.raises(SystemExit) as e:
        import sys
        sys.argv = ["masking", "--db", db]
        masking.main()
    assert e.value.code == 1


AWC = ("Firm X violated FINRA Rules 3110 and 2010. It also failed to keep records in violation of "
       "Exchange Act Section 17(a) and Rule 17a-4 thereunder. The firm has 20 branches. "
       "WAIVER OF PROCEDURAL RIGHTS The firm waives its right under FINRA Rules 9143 and 9144.")


def test_document_sentences():
    sents, chars, waiver = labels.document_sentences(AWC)
    assert waiver and chars == AWC.index("WAIVER")
    assert len(sents) == 2 and "9143" not in " ".join(sents)


class _Client:
    def __init__(self, bodies):
        self.bodies = bodies

    def get(self, url, use_cache=True):
        return (200, self.bodies[url]) if url in self.bodies else (404, b"")


def test_export(tmp_path, monkeypatch):
    csv = tmp_path / "h.csv"
    pd.DataFrame({"case_no": ["1", "2"], "doc_url": ["u1", "u2"], "doc_rules": ["FINRA:3110", "FINRA:2111"],
                  "true_rules": ["FINRA:3110", "FINRA:2111"], "checked": "", "notes": ""}).to_csv(csv, index=False)
    monkeypatch.setattr(labels, "pdf_text", lambda b: AWC)
    rows = labels.export(str(csv), str(tmp_path / "o.json"), client=_Client({"u1": b"%PDF-1.4"}))
    assert rows[0]["waiver_found"] and len(rows[0]["sentences"]) == 2
    assert rows[1]["error"].startswith("document not read")


def test_rulebook_helpers():
    home = b'<footer><a href="/terms-use">Terms of Use</a><a href="/privacy">Privacy</a></footer>'
    assert rulebook.terms_link(home, "https://www.finra.org/") == "https://www.finra.org/terms-use"
    text = "Intro\n\nYou may copy materials for personal, non-commercial use.\n\nPrivacy matters."
    assert rulebook.terms_passages(text) == ["You may copy materials for personal, non-commercial use."]
    allowed, _ = rulebook.robots_rules("User-agent: *\nDisallow: /search\n")
    assert allowed
    page = (b'<a href="/rules-guidance/rulebooks/finra-rules/3110">3110</a>'
            b'<a href="/rules-guidance/rulebooks/finra-rules/2010#x">2010</a>'
            b'<a href="/rules-guidance/guidance">x</a>')
    assert rulebook.rule_links(page, rulebook.INDEX_URL) == [
        "https://www.finra.org/rules-guidance/rulebooks/finra-rules/3110",
        "https://www.finra.org/rules-guidance/rulebooks/finra-rules/2010"]
    row = rulebook.parse_rule(b"<html><body><main><h1>3110. Supervision</h1><p>Each member shall</p></main></body></html>",
                              "https://www.finra.org/rules-guidance/rulebooks/finra-rules/3110")
    assert row["rule_no"] == "3110" and row["title"] == "3110. Supervision"


def test_download_needs_terms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="rulebook_terms_accepted"):
        rulebook.download(_Client({}), "f.duckdb", 1)


class _Ollama:
    """Answers like Ollama: supervision summaries get FINRA:3000, the rest FINRA:2000."""

    def __init__(self):
        self.calls = 0

    def post(self, url, json):
        import json as js
        self.calls += 1
        text = json["messages"][1]["content"]
        assert "FINRA:3000" in json["format"]["properties"] and "Rule 2010" in json["messages"][0]["content"]
        ans = {"FINRA:3000": 0.9, "FINRA:2000": 0.1} if "supervise" in text else {"FINRA:3000": 0.1, "FINRA:2000": 0.8}

        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"content": js.dumps(ans)}}
        return R()


def test_zeroshot_scores_and_resumes(tmp_path):
    from finra_nlp import zeroshot
    db = _db(tmp_path, _cases())
    dataset.build(db)
    c = _Ollama()
    assert zeroshot.run(db, str(tmp_path / "z.md"), limit=5, client=c) is None  # report waits for all cases
    per, macro = zeroshot.run(db, str(tmp_path / "z.md"), client=c)
    assert c.calls == 20  # 5 + the remaining 15, none twice
    assert per.loc["FINRA:3000", "f1"] == 1.0 and per.loc["FINRA:2000", "f1"] == 1.0
    con = duckdb.connect(db, read_only=True)
    assert con.execute("SELECT count(*) FROM model.zeroshot_validation").fetchone()[0] == 20
