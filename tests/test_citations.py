import pytest

from finra_nlp.citations import extract_citations, leaks, mask


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
