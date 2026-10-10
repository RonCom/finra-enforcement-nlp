import pytest

from finra_nlp.citations import extract_citations, fix_ocr_numbers, leaks, mask


def keys(text):
    return [c.key for c in extract_citations(text)]


@pytest.mark.parametrize("text,expected", [
    ("violated FINRA Rules 3110 and 2010.", ["FINRA:3110", "FINRA:2010"]),
    ("violated FINRA Rules 2010, 3110 and 4511.", ["FINRA:2010", "FINRA:3110", "FINRA:4511"]),
    ("violated FINRA Rules 2010, 3110, and 4511.", ["FINRA:2010", "FINRA:3110", "FINRA:4511"]),
    ("obligations under FINRA Rule 2330(c).", ["FINRA:2330"]),
    ("the factors in FINRA Rule 3270.01 for each", ["FINRA:3270"]),
    ("violated FINRA Rules 2010 (ethical standards), 1122 (filing misleading registration information)",
     ["FINRA:2010", "FINRA:1122"]),
    ("violated NASD Rule 3010 and FINRA Rule 2010", ["NASD:3010", "FINRA:2010"]),
    ("Exchange Act Rule 15l-1(a)(1) (Reg BI)", ["SEC_RULE:15l-1"]),
    ("Section 10(b) of the Securities Exchange Act of 1934 and Rule 10b-5 thereunder",
     ["SEC_SECTION:10", "SEC_RULE:10b-5"]),
    ("Section 5 of the Securities Act", ["SA_SECTION:5"]),
    ("Exchange Act Section 10(b), Exchange Act Rule 10b-5, and FINRA Rules 2020 and 2010",
     ["SEC_SECTION:10", "SEC_RULE:10b-5", "FINRA:2020", "FINRA:2010"]),
    ("Regulation NMS Rule 605 of the Securities Exchange Act", ["REG_NMS:605"]),
    ("MSRB Rules G-27 and G-17", ["MSRB:G-27", "MSRB:G-17"]),
    ("Pursuant to FINRA Rule Series 9554", ["FINRA:9554"]),
    ("FINRA Rule 4530(d). ... reportable under Rule 4530(d).", ["FINRA:4530", "FINRA:4530"]),
    ("NASD Rule 2110 applied. Later, Rule 3010 was breached.", ["NASD:2110", "NASD:3010"]),
])
def test_extract(text, expected):
    assert keys(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("violated NASD Rule 30401 and FINRA Rule 2010.", ["NASD:3040", "FINRA:2010"]),
    ("violated FINRA Rules 8210 and 20101 by failing", ["FINRA:8210", "FINRA:2010"]),
    ("FINRA Rules 2020 and 201028 by marking the close", ["FINRA:2020", "FINRA:2010"]),
    ("violated NASD Rules 3010 and 30122 and FINRA Rules 3110 and 2010", ["NASD:3010", "NASD:3012", "FINRA:3110",
                                                                         "FINRA:2010"]),
    ("FINRA Rules 45112 and 2010", ["FINRA:4511", "FINRA:2010"]),
    # five-digit rules that exist are kept
    ("FINRA Rule 12904 and FINRA Rule 11870 and NASD Rule 10330", ["FINRA:12904", "FINRA:11870", "NASD:10330"]),
])
def test_footnote_against_rule_number(text, expected):
    assert keys(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("violated NASD Rule 3040 and FlNRA Rules 3280 and 2010", ["NASD:3040", "FINRA:3280", "FINRA:2010"]),
    ("FINRA Rule 2010 and violation ofNASD Rule 3010(a)", ["FINRA:2010", "NASD:3010"]),
    ("FINRA Rule 2010 andNASD Rule 3010", ["FINRA:2010", "NASD:3010"]),
    ("FINRA Rule 4511 and NASO Rule 3010", ["FINRA:4511", "NASD:3010"]),
    ("NASD Rule 3010(b) and F?NRA Rule 2010", ["NASD:3010", "FINRA:2010"]),
    ("NASD Rule 3010 and FIN RA Rule 2010 and FIRNA Rule 4511", ["NASD:3010", "FINRA:2010", "FINRA:4511"]),
    ("FINRA Rule 2010 and NASD Condi?Ct Rule 2420", ["FINRA:2010", "NASD:2420"]),
    ("FINRA Rule 2010 and NASD Membership and Registration Rule 1031(a)", ["FINRA:2010", "NASD:1031"]),
    ("violated Exchange Act Rule 606 and FINRA Rule 2010", ["REG_NMS:606", "FINRA:2010"]),
    ("violated Rule 606 under the Securities Exchange Act of 1934", ["REG_NMS:606"]),
    ("the SEC Rule 101 violations", ["REG_M:101"]),
    ("Rule 102 of Regulation M", ["REG_M:102"]),
    ("violated Nasdaq Rule 4613 and FINRA Rule 2010", ["EXCHANGE:4613", "FINRA:2010"]),
    ("violations of FINRA Rules 6380A, 6622, 7230A and 7330", ["FINRA:6380", "FINRA:6622", "FINRA:7230", "FINRA:7330"]),
    ("violated FINRA Rules 3110(a) and (b) and 2010", ["FINRA:3110", "FINRA:2010"]),
    ("NASD Conduct Rules 3010(a) and (b) and FINRA Rule 2010", ["NASD:3010", "FINRA:2010"]),
])
def test_ocr_names_and_other_regulations(text, expected):
    assert keys(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("violated Exchange Act § 17(a) and Exchange Act§ 15(c)(3)", ["SEC_SECTION:17", "SEC_SECTION:15"]),
    ("Section 10(b) of the Securities and Exchange Act of 1934", ["SEC_SECTION:10"]),
    ("Municipal Securities Rulemaking Board (MSRB) Rule G-17 and Rule G-27", ["MSRB:G-17", "MSRB:G-27"]),
    ("compliance with Rule 606(a) and Rules 204(a) and (c)", ["REG_NMS:606", "REG_SHO:204"]),
    ("in violation of IM-12000 of FINRA's Code of Arbitration Procedure", ["FINRA:IM-12000"]),
    ("violated NASD Rule 2510(b) and 2010", ["NASD:2510", "FINRA:2010"]),
    ("violated FINRA Rule 3010 and NASD Rule 3280", ["NASD:3010", "FINRA:3280"]),
    ("violated Rules 301(a) and 301(c)(2) of SEC Regulation Crowdfunding and FINRA Funding Portal Rules 200(a) "
     "and 300(a)", ["REG_CF:301", "REG_CF:301", "FUNDING_PORTAL:200", "FUNDING_PORTAL:300"]),
])
def test_statute_signs_bare_regulations_and_rulebook_fixes(text, expected):
    assert keys(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("violated FINRA Rule 451 1 and FINRA Rule 2010", "violated FINRA Rule 4511 and FINRA Rule 2010"),
    ("FINRA Rules 1122and2010", "FINRA Rules 1122 and 2010"),
    ("FIN RA Rule 20 I 0, which", "FIN RA Rule 2010, which"),
    ("FINRA Rules 831 1 and 2010", "FINRA Rules 8311 and 2010"),
    ("Exchange Act Rule lOb-9", "Exchange Act Rule 10b-9"),
    ("the SEC Rule 101 5 violations", "the SEC Rule 101 5 violations"),  # Regulation M Rule 101, footnote 5
    ("Rules 3110 and 2010 by failing", "Rules 3110 and 2010 by failing"),
])
def test_fix_ocr_numbers(text, expected):
    assert fix_ocr_numbers(text) == expected


def test_years_and_amounts_are_not_rules():
    assert keys("In 2010, the firm paid $2,010 across 3110 accounts from May 2022.") == []


def test_series():
    c = extract_citations("FINRA Rules 3110 and 11870 and NASD IM-2210-1")
    assert [x.series for x in c] == ["FINRA:3000", "FINRA:11000", "NASD:2000"]


def test_mask_and_leak_check():
    text = ("The firm violated FINRA Rules 3110 and 2010 and Exchange Act Rule 17a-4(b). "
            "In 2010 it opened 3110 accounts.")
    m = mask(text)
    assert m == "The firm violated [RULE] and [RULE]. In 2010 it opened 3110 accounts."
    assert leaks(m) == []
    assert leaks("failed to supervise under Rule 3110") == ["Rule 3"]
    assert leaks(m, rule_titles=["Outside Business Activities"]) == []
    assert leaks("its outside business activities were unapproved", ["Outside Business Activities"])


def test_mask_catches_what_the_citation_patterns_leave():
    text = ("The firm lacked procedures for Section 5, Section 206(3) of the Investment Advisers Act, its Rule 606 "
            "reports, Form G-37 filings, Exchange Act Rule17a-3 notices and 12b-1 fees. He passed the Series 7.")
    m = mask(text)
    assert leaks(m) == []
    assert "Series 7" in m and "206" not in m and "606" not in m and "12b-1" not in m
