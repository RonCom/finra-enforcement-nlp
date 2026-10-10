import duckdb

from finra_nlp import ocr

AWC = ("III. FINDINGS. Respondent violated FINRA Rules 4511 and 2010 by failing to preserve records. "
       "WAIVER OF PROCEDURAL RIGHTS Respondent waives any right to claim a violation of FINRA Rule 9143.")


def _db(tmp_path):
    db = str(tmp_path / "o.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.execute("""CREATE TABLE raw.monthly_cases AS SELECT * FROM (VALUES
                   ('1', ''), ('2', 'FINRA:2111'), ('3', ''), ('4', '')) t(case_no, summary_rules)""")
    con.execute("""CREATE TABLE raw.case_document_labels AS SELECT * FROM (VALUES
                   ('1', 'FINRA:3110', 'd1', 1), ('2', '', 'd2', 1), ('3', '', 'd3', 1), ('4', '', NULL, 0))
                   t(case_no, doc_rules, doc_url, n_docs)""")
    con.execute("""CREATE TABLE raw.dao_index AS SELECT * FROM (VALUES
                   ('3', 'i3', DATE '2017-01-01')) t(case_no, doc_url, action_date)""")
    con.close()
    return db


class Client:
    def __init__(self):
        self.urls = []

    def cached(self, url):
        return False

    def get(self, url, use_cache=True):
        self.urls.append(url)
        return 200, b"%PDF"


def test_ocr_only_cases_without_document_or_summary_rules(tmp_path, monkeypatch):
    db, calls = _db(tmp_path), []
    monkeypatch.setattr(ocr, "ocr_text", lambda pdf, dpi: (calls.append(dpi), (AWC, 2))[1])
    c = Client()
    ocr.build(db, c, cache=tmp_path / "cache")
    assert c.urls == ["i3"]  # case 1 has document rules, 2 summary rules, 4 no document
    con = duckdb.connect(db)
    assert con.execute("SELECT case_no, ocr_rules FROM raw.case_ocr_labels").fetchall() == [
        ("3", "FINRA:4511|FINRA:2010")]
    con.close()
    ocr.build(db, c, cache=tmp_path / "cache")  # a rerun reads the cached OCR text
    assert len(calls) == 1
