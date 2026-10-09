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
