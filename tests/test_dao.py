from finra_nlp.dao import document_links, labels_from_document

AWC = """I. ACCEPTANCE AND CONSENT ... Respondent was registered from 2010 through 2023.
FACTS AND VIOLATIVE CONDUCT
The firm received 71 complaints but did not report them. FINRA Rule 4530(d) requires firms to report
statistical information. Therefore, the firm violated FINRA Rules 4530 and 2010.
The firm's procedures did not address escalation. Therefore, the firm violated FINRA Rules 3110(a) and 2010.
B. Respondent also consents to a censure and a fine of $200,000."""


def test_labels_only_from_violation_sentences():
    assert labels_from_document(AWC) == ["FINRA:4530", "FINRA:2010", "FINRA:3110"]


def test_document_links_prefers_case_number():
    html = b"""<html><body>
      <a href="/sites/default/files/other.pdf">Guide</a>
      <a href="/fda_documents/2020068991101_tastytrade_AWC.pdf">AWC</a></body></html>"""
    links = document_links(html, "https://www.finra.org/x?search=2020068991101", "2020068991101")
    assert links == ["https://www.finra.org/fda_documents/2020068991101_tastytrade_AWC.pdf"]


def test_waiver_boilerplate_ignored():
    text = AWC + """
II. WAIVER OF PROCEDURAL RIGHTS
Respondent further specifically and voluntarily waives any right to claim that a person violated
the ex parte prohibitions of FINRA Rule 9143 or the separation of functions prohibitions of FINRA Rule 9144."""
    assert labels_from_document(text) == ["FINRA:4530", "FINRA:2010", "FINRA:3110"]


def test_waive_sentence_skipped_without_heading():
    text = AWC + " Respondent waives any claim that staff violated FINRA Rule 9144."
    assert "FINRA:9144" not in labels_from_document(text)


def test_document_links_ignores_other_cases():
    """DAO's search is full-text: a case not in DAO returns other cases' documents."""
    html = b"""<a href="/sites/default/files/fda_documents/2021069373001%20David%20Wong%20OHO%20Decision.pdf">2021069373001</a>
      <a href="/rules-guidance/oversight-enforcement/finra-disciplinary-actions?search=2021069373001">Related</a>"""
    assert document_links(html, "https://www.finra.org/x?search=2009019837302", "2009019837302") == []


LISTING = b"""<table><tr><th>Case ID</th></tr>
<tr><td class="views-field views-field-field-fda-attachment-file-media">
  <a href="/sites/default/files/fda_documents/2022073772701%20Keith%20C.%20Baron%20CRD%203231494%20OHO%20Decision.pdf">202207377270</a></td>
  <td class="views-field views-field-views-conditional-field">summary</td>
  <td class="views-field views-field-field-fda-document-type-tax">OHO Decisions</td>
  <td class="views-field views-field-views-conditional-field-3">Keith Baron</td>
  <td class="views-field views-field-field-core-official-dt is-active">02/11/2025</td></tr>
<tr><td class="views-field views-field-field-fda-attachment-file-media">
  <a href="/sites/default/files/fda_documents/Acme%20AWC.pdf">2016049565901</a></td>
  <td class="views-field views-field-field-fda-document-type-tax">AWC</td>
  <td class="views-field views-field-field-core-official-dt is-active">03/01/2016</td></tr></table>"""


def test_parse_listing_takes_case_number_from_file_name():
    from datetime import date

    from finra_nlp.dao import parse_listing
    rows = parse_listing(LISTING, "https://www.finra.org/x")
    # the link text has a typo (12 digits); the file name has the right number
    assert rows[0]["case_no"] == "2022073772701" and rows[0]["doc_type"] == "OHO Decisions"
    assert rows[0]["action_date"] == date(2025, 2, 11) and rows[0]["respondent"] == "Keith Baron"
    assert rows[1]["case_no"] == "2016049565901"  # no number in the file name: the link text


def test_labels_use_the_index(tmp_path, monkeypatch):
    import duckdb
    import pandas as pd

    from finra_nlp import dao
    db = str(tmp_path / "f.duckdb")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA raw")
    con.execute("CREATE TABLE raw.monthly_cases AS SELECT * FROM (VALUES ('111'), ('222')) t(case_no)")
    con.register("ix", pd.DataFrame({"case_no": ["111"], "doc_url": ["u1"], "doc_type": ["AWC"],
                                     "respondent": ["x"], "action_date": [pd.Timestamp("2020-01-01").date()]}))
    con.execute("CREATE TABLE raw.dao_index AS SELECT * FROM ix")
    con.close()

    class C:
        min_interval, throttled, urls = 1.0, 0, []

        def cached(self, url):
            return False

        def get(self, url, use_cache=True):
            self.urls.append(url)
            return 200, b"%PDF"

    monkeypatch.setattr(dao, "pdf_text", lambda b: AWC)
    c = C()
    dao.build_labels(c, db, None, index_only=True)
    assert c.urls == ["u1"]  # no search page requested
    con = duckdb.connect(db)
    got = dict(con.execute("SELECT case_no, doc_rules FROM raw.case_document_labels").fetchall())
    assert got == {"111": "FINRA:4530|FINRA:2010|FINRA:3110", "222": ""}


def test_disciplinary_history_and_prior_awcs_dropped():
    text = ("RELEVANT DISCIPLINARY HISTORY On October 10, 2013, the firm was fined for violations of FINRA Rules "
            "6622, 7330 and 2010. On May 1, 2012, FINRA accepted an AWC finding violations of FINRA Rule 5123. "
            "In 2011, the firm consented to a censure for violations of NASD Rule 3010. The SEC filed a complaint "
            "against the issuer alleging violations of Exchange Act Rule 10b-5. OVERVIEW The conduct described "
            "in this paragraph constitutes separate and distinct violations of FINRA Rule 7330.")
    assert labels_from_document(text) == ["FINRA:7330"]


def test_sales_charge_waivers_are_not_the_waiver_of_rights():
    text = ("By failing to reasonably supervise mutual fund sales to ensure that customers received sales charge "
            "waivers, Summit violated NASD Conduct Rule 3010, FINRA Rule 3110 and FINRA Rule 2010. Respondent "
            "waives any right to claim a violation of FINRA Rule 9143.")
    assert labels_from_document(text) == ["NASD:3010", "FINRA:3110", "FINRA:2010"]


def test_procedural_rules_and_decimal_amounts():
    text = ("Pursuant to FINRA Rule 9216, Respondent submits this AWC to settle the alleged rule violations "
            "described below. The firm was fined $2.5 million for violations of FINRA Rules 3310 and 2010.")
    assert labels_from_document(text) == ["FINRA:3310", "FINRA:2010"]


def test_procedural_rules_and_other_cases_are_not_labels():
    text = ("Respondent is notified that he may move to set aside this Default Decision under FINRA Rule 9269(c), "
            "and the SEC reviews violations under Exchange Act Section 19(e). In the Prior Matter, FINRA found a "
            "violation of FINRA Rule 3240. DS consented to violations of FINRA Rule 2111. The records constituted "
            "records required under Exchange Act Rule 17a-3. Respondent violated FINRA Rules 8210 and 2010.")
    assert labels_from_document(text) == ["FINRA:8210", "FINRA:2010"]


def test_more_prior_case_phrasings():
    text = ("In 2016, Torino was censured and fined $2,000 for violations of FINRA Rule 6730. The AWC included fines "
            "of $30,000 for violations of FINRA Rule 6760. The findings in that proceeding stated violations of "
            "FINRA Rule 2210. The firm violated FINRA Rules 3110 and 2010.")
    assert labels_from_document(text) == ["FINRA:3110", "FINRA:2010"]
