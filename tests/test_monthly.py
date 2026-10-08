from datetime import date

import pymupdf

from finra_nlp.monthly import pdf_text, report_month_from, segment

REPORT = """Disciplinary and Other FINRA Actions | July 2026 1
Disciplinary
and Other
FINRA Actions
Search for FINRA Disciplinary Actions
Firms Fined
Alpha Securities, Inc. (CRD #111, New York, New York)
May 4, 2026 – An AWC was issued in which the firm was censured and fined $250,000. The firm
failed to supervise account statements, in violation of FINRA Rules 3110 and 2010. (FINRA Case #2023077018401)
Beta Capital LLC (CRD #222, Tampa, Florida)
May 7, 2026 – An AWC was issued in which the firm was censured and fined $100,000. The firm
did not maintain a surveil-
lance system for exchanges.
Disciplinary and Other FINRA Actions | July 2026 2
The system was not functional until 2025. (FINRA Case
#2023077036901)
Individuals Barred
Carl Delta (CRD #333, Jackson, New Jersey)
May 4, 2026 – An AWC was issued in which Delta was barred. He refused to provide information
requested under FINRA Rule 8210. (FINRA Case #2026088818101)
Individuals Barred for Failure to Provide Information Pursuant to FINRA Rule 9552(h)
Erin Echo (CRD #444)
Trinity, North Carolina
(May 15, 2026)
FINRA Case #2025086456101
"""


def test_segment():
    cases = segment(REPORT, "u", "2026-07")
    assert [c.case_no for c in cases] == ["2023077018401", "2023077036901", "2026088818101"]
    a, b, c = cases
    assert a.section == "Firms Fined" and c.section == "Individuals Barred"
    assert a.respondent == "Alpha Securities, Inc." and c.respondent == "Carl Delta"
    assert a.action_date == date(2026, 5, 4)
    assert a.summary_rules == "FINRA:3110|FINRA:2010"
    assert "[RULE]" in a.masked_text and "3110" not in a.masked_text
    assert b.summary_rules == ""
    assert "surveillance system" in b.text  # hyphenated line break rejoined
    assert "Disciplinary and Other FINRA Actions" not in b.text  # page header removed
    assert c.summary_rules == "FINRA:8210"


def test_report_month():
    assert report_month_from("https://x/2026-07/disciplinary-actions-july-2026.pdf", "") == "2026-07"
    assert report_month_from("https://x/Disciplinary%20Actions_July_2024.pdf", "") == "2024-07"
    assert report_month_from("https://x/p122797.pdf", "Disciplinary and Other FINRA Actions\nJanuary 2011") == "2011-01"


def test_pdf_roundtrip():
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in REPORT.splitlines()[:12]:
        page.insert_text((36, y), line, fontsize=8)
        y += 11
    data = doc.tobytes()
    cases = segment(pdf_text(data))
    assert cases[0].case_no == "2023077018401"


def test_wrapped_line_starting_with_firm_is_not_a_heading():
    text = REPORT.replace("The firm\nfailed to supervise", "The\nFirm personnel failed to supervise")
    a = segment(text)[0]
    assert a.section == "Firms Fined"
    assert "Firm personnel failed" in a.text


class FakeClient:
    """Index lists two recent PDFs; older months exist only as HTML pages under one of two patterns."""
    def get(self, url, use_cache=True):
        if url.startswith("https://www.finra.org/rules-guidance/oversight-enforcement/disciplinary-actions"):
            if "?page=" in url:
                return 200, b"<html></html>"
            return 200, b"""<a href="/sites/default/files/2026-07/disciplinary-actions-july-2026.pdf">Jul</a>
                <a href="/sites/default/files/2026-08/disciplinary-actions-august-2026.pdf">Aug</a>
                <a href="/about">About</a>"""
        if url.endswith("monthly-disciplinary-actions-june-2026"):
            return 200, b'<a href="/sites/default/files/2026-06/Disciplinary_Actions_June_2026.pdf">PDF</a>'
        if url.endswith("disciplinary-actions/may-2026"):
            return 200, b"<main>Monthly Disciplinary Actions May 2026 text only</main>"
        return 404, b""


def test_discover_one_url_per_month():
    from finra_nlp.monthly import discover
    found = discover(FakeClient(), start="2026-04", end="2026-08")
    assert list(found) == ["2026-05", "2026-06", "2026-07", "2026-08"]
    assert found["2026-06"].endswith("Disciplinary_Actions_June_2026.pdf")
    assert found["2026-05"].endswith("disciplinary-actions/may-2026")
    assert found["2026-07"] == "https://www.finra.org/sites/default/files/2026-07/disciplinary-actions-july-2026.pdf"
