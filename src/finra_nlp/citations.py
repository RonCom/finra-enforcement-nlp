"""Find, normalize and mask rule citations in FINRA disciplinary text.

Handles lists ("FINRA Rules 2020, 3110 and 2010"), subsections ("3110(a)", "15l-1(a)(1)"),
supplementary material ("3270.01"), parenthetical descriptions ("2010 (ethical standards)"),
reversed statute order ("Section 10(b) of the Exchange Act"), and bare "Rule 4530(d)" that
inherits the family of the nearest earlier citation in the same text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUB = r"(?:\([a-zA-Z0-9]{1,4}\))*"
DESC = r"(?:\s*\((?=[^)]*\s)[^()]{2,80}\))?"  # "(ethical standards)", not "(a)"
NUM_FINRA = rf"(?:IM-\d{{4}}(?:-\d+)?|\d{{4,5}}(?:\.\d{{2}})?){SUB}"
NUM_SEC = rf"\d{{1,2}}[a-z]{{1,2}}\d?-\d{{1,2}}[a-z]?{SUB}"  # 10b-5, 17a-3, 15c3-1, 15l-1
NUM_MSRB = rf"[A-G]-\d{{1,2}}{SUB}"
NUM_SECTION = rf"\d{{1,2}}[A-Z]?{SUB}"
NUM_NMS = rf"\d{{3}}{SUB}"
SEP = r"(?:\s*,\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+)"


def _list(num: str) -> str:
    return rf"(?P<nums>{num}{DESC}(?:{SEP}{num}{DESC})*)"


ACT = r"(?:the\s+)?(?:Securities\s+Exchange\s+Act(?:\s+of\s+1934)?|Exchange\s+Act|Securities\s+Act(?:\s+of\s+1933)?)"

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("FINRA", re.compile(rf"\bFINRA\s+Rules?\s+(?:Series\s+)?{_list(NUM_FINRA)}")),
    ("NASD", re.compile(rf"\bNASD\s+(?:Conduct\s+)?(?:Rules?\s+|(?=IM-)){_list(NUM_FINRA)}")),
    ("MSRB", re.compile(rf"\bMSRB\s+Rules?\s+{_list(NUM_MSRB)}")),
    ("SEC_RULE", re.compile(rf"\b(?:Securities\s+Exchange\s+Act(?:\s+of\s+1934)?|Exchange\s+Act)\s+Rules?\s+{_list(NUM_SEC)}")),
    ("SEC_SECTION", re.compile(rf"\b(?:Exchange\s+Act|Securities\s+Exchange\s+Act(?:\s+of\s+1934)?)\s+Sections?\s+{_list(NUM_SECTION)}")),
    ("SECTION_OF_ACT", re.compile(rf"\bSections?\s+{_list(NUM_SECTION)}\s+of\s+{ACT}")),
    ("REG_NMS", re.compile(rf"\bRegulation\s+NMS\s+Rules?\s+{_list(NUM_NMS)}")),
    ("REG_NMS", re.compile(rf"\bRules?\s+{_list(NUM_NMS)}\s+of\s+Regulation\s+NMS")),
    ("REG_SHO", re.compile(rf"\bRegulation\s+SHO\s+Rules?\s+{_list(NUM_NMS)}")),
    ("REG_SHO", re.compile(rf"\bRules?\s+{_list(NUM_NMS)}\s+of\s+Regulation\s+SHO")),
    ("SEC_RULE", re.compile(rf"\bRules?\s+{_list(NUM_SEC)}")),  # bare "Rule 10b-5" is always an SEC rule
    ("BARE", re.compile(rf"\bRules?\s+(?:Series\s+)?{_list(NUM_FINRA)}")),
]

ONE = {
    "FINRA": re.compile(NUM_FINRA), "NASD": re.compile(NUM_FINRA), "BARE": re.compile(NUM_FINRA),
    "MSRB": re.compile(NUM_MSRB), "SEC_RULE": re.compile(NUM_SEC),
    "SEC_SECTION": re.compile(NUM_SECTION), "SECTION_OF_ACT": re.compile(NUM_SECTION),
    "REG_NMS": re.compile(NUM_NMS), "REG_SHO": re.compile(NUM_NMS),
}


@dataclass(frozen=True)
class Citation:
    family: str     # FINRA, NASD, MSRB, SEC_RULE, SEC_SECTION, SA_SECTION, REG_NMS, REG_SHO
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


def _base(rule: str) -> str:
    rule = re.sub(r"\(.*$", "", rule)
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
            out.append(Citation(family, rule, _base(rule), nums_start + n.start(), nums_start + n.end()))
    return out


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


def mask(text: str, token: str = "[RULE]") -> str:
    out, pos = [], 0
    for s, e in citation_spans(text):
        out.append(text[pos:s])
        out.append(token)
        pos = e
    out.append(text[pos:])
    return "".join(out)


LEAK_RE = re.compile(r"\b(?:Rules?|Sections?|IM-|Series)\s*\(?\d|\bG-\d{1,2}\b|\b\d{1,2}[a-z]{1,2}\d?-\d\b")


def leaks(masked: str, rule_titles: list[str] | None = None) -> list[str]:
    """Return anything in masked text that still identifies a rule: numbered references, or titles."""
    hits = [m.group(0) for m in LEAK_RE.finditer(masked)]
    for title in rule_titles or []:
        if len(title) >= 12 and re.search(re.escape(title), masked, re.IGNORECASE):
            hits.append(title)
    return hits
