import duckdb
import pandas as pd

from finra_nlp import labels


def _db(tmp_path):
    db = str(tmp_path / "f.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    rows = []
    for i in range(40):
        rows.append((f"2018{i:09d}", "2018-05", "Firms Fined", None,
                     "FINRA:3110|FINRA:2010" if i % 2 else "", f"u{i}", "FINRA:3110|FINRA:2010"))
    rows.append(("2025000000001", "2025-07", "Firms Fined", None, "", "u", "MSRB:G-27|NASD:3010"))
    rows.append(("2025000000002", "2025-07", "Firms Fined", None, "", None, None))
    df = pd.DataFrame(rows, columns=["case_no", "report_month", "section", "action_date", "summary_rules",
                                     "doc_url", "doc_rules"])
    con.register("df", df)
    con.execute("CREATE TABLE raw.monthly_cases AS SELECT case_no, report_month, section, action_date, summary_rules FROM df")
    con.execute("CREATE TABLE raw.case_document_labels AS SELECT case_no, doc_url, doc_rules FROM df")
    con.close()
    return db


def test_series_label():
    assert labels.series_label("FINRA:3110") == "FINRA:3000"
    assert labels.series_label("FINRA:11870") == "FINRA:11000"
    assert labels.series_label("NASD:IM-2210-1") == "NASD:2000"
    assert labels.series_label("SEC_RULE:15l-1") == "SEC_RULE"


def test_sample_and_score(tmp_path):
    db, csv = _db(tmp_path), str(tmp_path / "h.csv")
    labels.sample(db, csv, n=5)
    df = pd.read_csv(csv, dtype=str).fillna("")
    assert len(df) == 5 and (df.true_rules == df.doc_rules).all()
    df.loc[:, "checked"] = "Y"
    df.loc[0, "true_rules"] = df.loc[0, "doc_rules"] + "|FINRA:4511"  # one missed rule
    df.to_csv(csv, index=False)
    p, r = labels.score(csv)
    assert p == 1.0 and r == 10 / 11


def test_profile(tmp_path):
    text = labels.profile(_db(tmp_path), str(tmp_path / "p.md"))
    assert "Rule 2010 appears in 97.6% of labeled cases" in text
    assert "| FINRA:3000 |            40 | False" in text
    assert "20 cases have both. Identical sets: 100.0%" in text


def test_case_labels_order(tmp_path):
    db = str(tmp_path / "c.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.execute("""CREATE TABLE raw.monthly_cases AS SELECT * FROM (VALUES
                   ('1', 'FINRA:2111'), ('2', 'FINRA:2111'), ('3', ''), ('4', '')) t(case_no, summary_rules)""")
    con.execute("""CREATE TABLE raw.case_document_labels AS SELECT * FROM (VALUES
                   ('1', 'FINRA:3110'), ('2', ''), ('3', ''), ('4', '')) t(case_no, doc_rules)""")
    got = labels.case_labels(con).set_index("case_no")
    assert got.loc["1", "rules"] == "FINRA:3110" and got.loc["1", "label_source"] == "document"
    assert got.loc["2", "rules"] == "FINRA:2111" and got.loc["2", "label_source"] == "summary"
    assert got.loc["3", "rules"] == "" and pd.isna(got.loc["3", "label_source"])
    con.execute("""CREATE TABLE raw.case_ocr_labels AS SELECT * FROM (VALUES
                   ('2', 'u', 'FINRA:4511'), ('3', 'u', 'FINRA:4511')) t(case_no, doc_url, ocr_rules)""")
    got = labels.case_labels(con).set_index("case_no")
    assert got.loc["2", "label_source"] == "summary"  # the summary still comes before OCR
    assert got.loc["3", "rules"] == "FINRA:4511" and got.loc["3", "label_source"] == "ocr"
    con.close()


def test_nasd_rules_take_their_finra_successor_series():
    assert labels.successor("NASD:3010") == "FINRA:3110"
    assert labels.successor("NASD:9999") == "NASD:9999"
    assert labels.successor("FINRA:3110") == "FINRA:3110"
    # NASD 2110 becomes FINRA 2010 and is dropped like it; 0140 and the Code of Procedure are never labels
    assert labels.label_series(["NASD:3010", "FINRA:3110", "NASD:2110", "FINRA:0140", "FINRA:9216",
                                "NASD:2510", "SEC_RULE:17a-3"]) == {"FINRA:3000", "SEC_RULE"}
