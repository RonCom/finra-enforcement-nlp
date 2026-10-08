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
