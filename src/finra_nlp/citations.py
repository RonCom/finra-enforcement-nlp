"""Find, normalize and mask rule citations in FINRA disciplinary text.

Handles lists ("FINRA Rules 2020, 3110 and 2010"), subsections ("3110(a)", "15l-1(a)(1)"),
supplementary material ("3270.01"), parenthetical descriptions ("2010 (ethical standards)"),
reversed statute order ("Section 10(b) of the Exchange Act"), and bare "Rule 4530(d)" that
inherits the family of the nearest earlier citation in the same text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUB = r"(?:\([a-zA-Z0-9]{1,4}\))*(?:(?:\s*,\s*|\s+and\s+|\s+or\s+)\([a-zA-Z0-9]{1,4}\)(?:\([a-zA-Z0-9]{1,4}\))*)*"
DESC = r"(?:\s*\((?=[^)]*\s)[^()]{2,80}\))?"  # "(ethical standards)", not "(a)"
NUM_FINRA = rf"(?:IM-\d{{4,5}}(?:-\d+)?|\d{{4,5}}[A-C]?(?:\.\d{{2}})?){SUB}"  # "6380A", "7230A"
NUM_IM = rf"IM-\d{{4,5}}(?:-\d+)?{SUB}"
NUM_SEC = rf"\d{{1,2}}[a-z]{{1,2}}\d?-\d{{1,2}}[a-z]?{SUB}"  # 10b-5, 17a-3, 15c3-1, 15l-1
NUM_MSRB = rf"[A-G]-\d{{1,2}}{SUB}"
NUM_SECTION = rf"\d{{1,2}}[A-Z]?{SUB}"
NUM_NMS = rf"\d{{3}}{SUB}"
NUM_NMS_6 = rf"6\d\d{SUB}"  # Regulation NMS rules are 600-613
NUM_REG_M = rf"10[0-5]{SUB}"  # Regulation M rules are 100-105
NUM_REG_SHO = rf"20[0-4]{SUB}"  # Regulation SHO rules are 200-204
NUM_REG_CF = rf"\d{{3}}{SUB}"
NUM_EXCHANGE = rf"\d{{1,4}}(?:\.\d{{1,2}})?[A-Z]?{SUB}"  # "Nasdaq Rule 4613", "Cboe Rule 4.24"
EXCHANGES = r"(?:NASDAQ|Nasdaq|NYSE(?:\s+(?:Arca|American|MKT))?|Cboe|CBOE|BATS|Phlx|PHLX|ISE|MIAX|BOX|IEX)"
SEP = r"(?:\s*,\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+)"


def _list(num: str) -> str:
    return rf"(?P<nums>{num}{DESC}(?:{SEP}{num}{DESC})*)"


ACT = (r"(?:the\s+)?(?:Securities\s+(?:and\s+)?Exchange\s+Act(?:\s+of\s+1934)?|Exchange\s+Act"
       r"|Securities\s+Act(?:\s+of\s+1933)?)")
EXCHANGE_ACT = r"(?:Securities\s+(?:and\s+)?Exchange\s+Act(?:\s+of\s+1934)?|Exchange\s+Act|SEA)"
SECTION = r"(?:\bSections?\s+|§§?\s*)"  # "Section 17(a)", "§ 17(a)", "Exchange Act§ 17(a)"

# OCR spells FINRA as "FlNRA", "F1NRA", "F?NRA", "FIN RA", "FIRNA" and glues it to the word before
# ("ofFINRA", "andNASD"), so the names carry no leading word boundary. A misread name would otherwise
# leave its rules to the bare pattern, which gives them the family of the citation before.
FINRA_NAME = r"(?:FIN\s?RA|F[A-Za-z0-9?!|\[\]]{1,4}(?:RA|NA))"
NASD_NAME = r"NAS[DO](?:\s*Cond\S*)?(?:\s+Membership\s+(?:and\s+)?Registration)?"

PATTERNS: list[tuple[str, re.Pattern]] = [
    # exchange rules come first so the bare pattern doesn't give them the family of a FINRA citation
    ("EXCHANGE", re.compile(rf"\b{EXCHANGES}\s+Rules?\s+{_list(NUM_EXCHANGE)}")),
    ("FINRA", re.compile(rf"{FINRA_NAME}\s*Rules?\s+(?:Series\s+)?{_list(NUM_FINRA)}")),
    ("NASD", re.compile(rf"{NASD_NAME}\s*(?:Rules?\s+|(?=IM-)){_list(NUM_FINRA)}")),
    ("FINRA", re.compile(rf"\b{_list(NUM_IM)}\s+of\s+(?:the\s+)?FINRA")),  # "IM-12000 of FINRA's Code of Arbitration"
    ("MSRB", re.compile(rf"\bMSRB\)?\s+Rules?\s+{_list(NUM_MSRB)}")),  # "(MSRB) Rule G-17"
    ("SEC_RULE", re.compile(rf"\b{EXCHANGE_ACT}\s+Rules?\s+{_list(NUM_SEC)}")),
    ("SEC_SECTION", re.compile(rf"\b{EXCHANGE_ACT}\s*{SECTION}{_list(NUM_SECTION)}")),
    ("SECTION_OF_ACT", re.compile(rf"{SECTION}{_list(NUM_SECTION)}\s+of\s+{ACT}")),
    ("REG_NMS", re.compile(rf"\bRegulation\s+NMS\s+Rules?\s+{_list(NUM_NMS)}")),
    ("REG_NMS", re.compile(rf"\bRules?\s+{_list(NUM_NMS)}\s+of\s+Regulation\s+NMS")),
    ("REG_SHO", re.compile(rf"\bRegulation\s+SHO\s+Rules?\s+{_list(NUM_NMS)}")),
    ("REG_SHO", re.compile(rf"\bRules?\s+{_list(NUM_NMS)}\s+of\s+Regulation\s+SHO")),
    ("REG_NMS", re.compile(rf"\b(?:Exchange\s+Act|SEC)\s+Rules?\s+{_list(NUM_NMS_6)}(?![-\d])")),
    ("REG_NMS", re.compile(rf"\bRules?\s+{_list(NUM_NMS_6)}\s+(?:under|of)\s+{ACT}")),
    ("REG_M", re.compile(rf"\bRegulation\s+M\s+Rules?\s+{_list(NUM_REG_M)}")),
    ("REG_M", re.compile(rf"\bRules?\s+{_list(NUM_REG_M)}\s+of\s+Regulation\s+M\b")),
    ("REG_M", re.compile(rf"\bSEC\s+Rules?\s+{_list(NUM_REG_M)}(?![-\d])")),
    ("FUNDING_PORTAL", re.compile(rf"\bFundin\S?\s+Portal\s+Rules?\s+{_list(NUM_REG_CF)}")),  # FINRA Funding Portal Rules
    ("REG_CF", re.compile(rf"\bRules?\s+{_list(NUM_REG_CF)}\s+of\s+(?:SEC\s+)?Regulation\s+Crowdfunding")),
    ("REG_CF", re.compile(rf"\bRegulation\s+Crowdfunding\s+Rules?\s+{_list(NUM_REG_CF)}")),
    ("SEC_RULE", re.compile(rf"\bRules?\s+{_list(NUM_SEC)}")),  # bare "Rule 10b-5" is always an SEC rule
    # bare three-digit rules: 600-613 are Regulation NMS, 200-204 Regulation SHO; FINRA's are four digits
    ("REG_NMS", re.compile(rf"\bRules?\s+{_list(NUM_NMS_6)}(?![-\d])")),
    ("REG_SHO", re.compile(rf"\bRules?\s+{_list(NUM_REG_SHO)}(?![-\d])")),
    ("MSRB", re.compile(rf"\bRules?\s+{_list(NUM_MSRB)}")),  # "Rule G-27" is always MSRB
    ("BARE", re.compile(rf"\bRules?\s+(?:Series\s+)?{_list(NUM_FINRA)}")),
]

ONE = {
    "FINRA": re.compile(NUM_FINRA), "NASD": re.compile(NUM_FINRA), "BARE": re.compile(NUM_FINRA),
    "MSRB": re.compile(NUM_MSRB), "SEC_RULE": re.compile(NUM_SEC),
    "SEC_SECTION": re.compile(NUM_SECTION), "SECTION_OF_ACT": re.compile(NUM_SECTION),
    "REG_NMS": re.compile(NUM_NMS), "REG_SHO": re.compile(NUM_NMS), "REG_M": re.compile(NUM_REG_M),
    "EXCHANGE": re.compile(NUM_EXCHANGE), "REG_CF": re.compile(NUM_REG_CF),
    "FUNDING_PORTAL": re.compile(NUM_REG_CF),
}

# Rule numbers that exist in only one of the two rulebooks. A bare rule that inherited the wrong family
# ("violated NASD Rule 2510(b) and 2010"), or a misread name, is moved to the rulebook that has it.
NASD_ONLY = {"2110", "2510", "3010", "3011", "3012", "3013", "3030", "3040", "3050"}
FINRA_ONLY = {"2010", "2111", "3270", "3280", "4511"}


def _rulebook(family: str, base: str) -> str:
    if family == "FINRA" and base in NASD_ONLY:
        return "NASD"
    if family == "NASD" and base in FINRA_ONLY:
        return "FINRA"
    return family


@dataclass(frozen=True)
class Citation:
    family: str     # FINRA, NASD, MSRB, SEC_RULE, SEC_SECTION, SA_SECTION, REG_NMS, REG_SHO, REG_M, REG_CF, FUNDING_PORTAL, EXCHANGE
    rule: str       # as written, with subsections: "3110(a)", "15l-1(a)(1)"
    base: str       # without subsections or supplementary material: "3110", "15l-1"
    start: int
    end: int

    @property
    def key(self) -> str:
        return f"{self.family}:{self.base}"

    @property
    def series(self) -> str | None:
        """FINRA/NASD rulebook series, e.g. 3110 -> 'FINRA:3000', IM-2210-1 -> 'FINRA:2000'."""
        if self.family not in ("FINRA", "NASD"):
            return None
        digits = re.match(r"(?:IM-)?(\d+)", self.base).group(1)
        return f"{self.family}:{int(digits) // 1000 * 1000}"


# Five-digit rule numbers that exist: FINRA's Uniform Practice, arbitration and mediation codes
# (11000-14999), NASD's arbitration code and Uniform Practice Code (10000-11999).
FIVE_DIGIT = {"FINRA": (11000, 14999), "NASD": (10000, 11999)}


def _drop_footnote(family: str, base: str) -> str:
    """A footnote number printed against a rule number reads as one longer number: "NASD Rule 30401"
    is Rule 3040 with footnote 1, "FINRA Rule 201028" is Rule 2010 with footnote 28. A five-digit
    FINRA or NASD number outside the ranges that exist keeps its first four digits."""
    lo, hi = FIVE_DIGIT.get(family, (0, 0))
    if family in FIVE_DIGIT and re.fullmatch(r"\d{5}", base) and not lo <= int(base) <= hi:
        return base[:4]
    return base


def _base(rule: str) -> str:
    rule = re.sub(r"\(.*$", "", rule)
    rule = re.sub(r"^(\d{4,5})[A-C]", r"\1", rule)  # 6380A is in the 6380 family
    if re.fullmatch(r"\d{4,5}\.\d{2}", rule):
        rule = rule.split(".")[0]
    return rule


def extract_citations(text: str) -> list[Citation]:
    spans: list[tuple[int, int, str, re.Match]] = []
    taken = [False] * (len(text) + 1)
    for family, pat in PATTERNS:
        for m in pat.finditer(text):
            if any(taken[m.start(): m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            spans.append((m.start(), m.end(), family, m))
    spans.sort(key=lambda s: s[0])

    out: list[Citation] = []
    last_rulebook = "FINRA"
    for start, end, family, m in spans:
        nums_text, nums_start = m.group("nums"), m.start("nums")
        parser = ONE[family]
        if family == "BARE":
            family = last_rulebook
        elif family == "SECTION_OF_ACT":
            family = "SEC_SECTION" if "Exchange" in m.group(0) else "SA_SECTION"
        if family in ("FINRA", "NASD"):
            last_rulebook = family
        # blank out descriptions like "(ethical standards)" so numbers aren't read from them
        cleaned = re.sub(r"\((?=[^)]*\s)[^()]{2,80}\)", lambda d: " " * len(d.group(0)), nums_text)
        for n in parser.finditer(cleaned):
            rule = n.group(0)
            base = _drop_footnote(family, _base(rule))
            out.append(Citation(_rulebook(family, base), rule, base, nums_start + n.start(), nums_start + n.end()))
    return out


_OCR_DIGITS = str.maketrans({"I": "1", "l": "1", "O": "0"})
_SPLIT_NUM_RE = re.compile(r"((?:Rules?|and|or|,)\s+)((?:[\dIlO]\s?){3,6}?)(?=[\s,.;:)(]|$)")


def fix_ocr_numbers(text: str) -> str:
    """Repair rule numbers OCR split or misread in case documents: "451 1", "45 l l", "20 I 0" -> 4511,
    4511, 2010 when they follow "Rule(s)", "and", "or" or a comma; "lOb-9", "1 0b-9" -> 10b-9;
    "1122and2010" -> "1122 and 2010". Changes lengths, so it's for label extraction, not masking."""
    text = re.sub(r"(\d{4})(and|or)(\d{4})", r"\1 \2 \3", text)
    text = re.sub(r"\b[l1I]\s?[O0o]b-(\d)", r"10b-\1", text)

    def join(m: re.Match) -> str:
        raw = m.group(2)
        # after "Rule(s)", only a FINRA or NASD citation: "SEC Rule 101 5" is Regulation M Rule 101 and a footnote
        if m.group(1)[0] == "R" and not re.search(r"RA|NAS", text[max(0, m.start() - 14): m.start()]):
            return m.group(0)
        cand = raw.replace(" ", "").translate(_OCR_DIGITS)
        if " " in raw.strip() or not raw.strip().isdigit():
            if len(cand) == 4 and cand.isdigit() and sum(ch.isdigit() for ch in raw) >= 2:
                return m.group(1) + cand + (" " if raw.endswith(" ") else "")
        return m.group(0)
    return _SPLIT_NUM_RE.sub(join, text)


def citation_spans(text: str) -> list[tuple[int, int]]:
    """Full spans to mask, including the 'FINRA Rules' prefix and descriptions."""
    spans, taken = [], [False] * (len(text) + 1)
    for _, pat in PATTERNS:
        for m in pat.finditer(text):
            if not any(taken[m.start(): m.end()]):
                for i in range(m.start(), m.end()):
                    taken[i] = True
                spans.append((m.start(), m.end()))
    return sorted(spans)


# Anything the citation patterns leave that still reads as a numbered rule or statute section: "Section 5",
# "Section 206(3) of the Investment Advisers Act", "Rule 606", "Rule17a-3", "12b-1", "Form G-37".
_ITEM = r"\(?\d[0-9A-Za-z().\-]*"
GENERIC_RE = re.compile(
    rf"\b(?:Rules?|Sections?)\s*{_ITEM}(?:(?:\s*,\s*|\s+and\s+|\s+or\s+){_ITEM})*|§§?\s*{_ITEM}"
    r"|\bIM-\d[\d\-]*|\bG-\d{1,2}\b(?:\([a-z0-9]{1,3}\))*|\b\d{1,2}[a-z]{1,2}\d?-\d{1,2}[a-z]?\b"
)


def mask(text: str, token: str = "[RULE]") -> str:
    out, pos = [], 0
    for s, e in citation_spans(text):
        out.append(text[pos:s])
        out.append(token)
        pos = e
    out.append(text[pos:])
    return GENERIC_RE.sub(token, "".join(out))


# "Series 7" is an exam, not a rule; "FINRA Rule Series 9554" is masked by its FINRA prefix
LEAK_RE = re.compile(r"\b(?:Rules?|Sections?|IM-)\s*\(?\d|§|\bG-\d{1,2}\b|\b\d{1,2}[a-z]{1,2}\d?-\d\b")


def leaks(masked: str, rule_titles: list[str] | None = None) -> list[str]:
    """Return anything in masked text that still identifies a rule: numbered references, or titles."""
    hits = [m.group(0) for m in LEAK_RE.finditer(masked)]
    for title in rule_titles or []:
        if len(title) >= 12 and re.search(re.escape(title), masked, re.IGNORECASE):
            hits.append(title)
    return hits
