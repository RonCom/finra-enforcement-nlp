"""Parse FINRA's monthly "Disciplinary and Other FINRA Actions" reports into one row per case.

Reports exist as PDFs (/sites/default/files/YYYY-MM/...pdf) and, for older months, HTML pages
(/rules-guidance/disciplinary-actions/<month>-<year>). Each case write-up ends with
"(FINRA Case #<number>)", which is the split point.

Usage:
    uv run python -m finra_nlp.monthly discover --out data/monthly_urls.txt
    uv run python -m finra_nlp.monthly parse --urls data/monthly_urls.txt --db data/finra.duckdb
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from urllib.parse import urljoin

import duckdb
import pandas as pd
import pymupdf
from bs4 import BeautifulSoup

from finra_nlp.citations import extract_citations, mask
from finra_nlp.http import PoliteClient

INDEX_URL = "https://www.finra.org/rules-guidance/oversight-enforcement/disciplinary-actions"
LINK_RE = re.compile(
    r"disc?iplinary[-_ %20]*actions?.*\.pdf$"  # May-Sep 2017 PDFs are spelled "Disiplinary"
    r"|/monthly-disciplinary-actions-[a-z]+-\d{4}/?$"
    r"|/disciplinary-actions/[a-z]+-\d{4}/?$",
    re.I,
)
MONTH_PAGE_PATTERNS = [
    "https://www.finra.org/rules-guidance/rulebooks/monthly-disciplinary-actions-{month}-{year}",
    "https://www.finra.org/rules-guidance/disciplinary-actions/{month}-{year}",
]
PDF_NAME_PATTERNS = [  # names seen on finra.org for monthly reports
    "{Month}_{year}_Disciplinary_Actions.pdf",
    "{Month}_{year}_Disciplinary_Actions_0.pdf",
    "{Month}_{year}_Disiplinary_Actions.pdf",  # FINRA's spelling for May-Sep 2017
    "Disciplinary_Actions_{Month}_{year}.pdf",
    "Disciplinary_Actions_{Month}_{year}_0.pdf",
    "Disciplinary%20Actions_{Month}_{year}.pdf",
    "disciplinary-actions-{month}-{year}.pdf",
]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]

CASE_END_RE = re.compile(r"\(\s*FINRA\s+Case\s*#\s*(\d{8,14})\s*\)")
PAGE_HEADER_RE = re.compile(
    r"^\s*(?:\d+\s+)?Disciplinary\s+(?:and|&)\s+Other\s+FINRA\s+Actions(?:\s*\|\s*[A-Z][a-z]+\s+\d{4})?\s*\d*\s*$",
    re.M,
)
SECTION_RE = re.compile(
    r"^(?:Firms?|Individuals?|Firms?\s+and\s+Individuals?|Complaints?\s+Filed|Decisions?\s+Issued)\b[^.\n]{0,200}$"
)
CRD_RE = re.compile(r"^(?P<name>.+?)\s*\(CRD\s*#")
ACTION_DATE_RE = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4})\s*[–—-]"
)
REPORT_MONTH_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)[\s_-]+(\d{4})", re.I
)
# file names like Oct_2024_..., 11_Nov_..., 12_2024_December_..., August_Disciplinary Actions_2022
NAME_MONTH_RE = re.compile(
    r"(?<![a-z])(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)(?![a-z])", re.I)
NAME_YEAR_RE = re.compile(r"(?<!\d)(20\d\d)(?!\d)")
FOLDER_RE = re.compile(r"/files/(\d{4})-(\d{2})/")
# the Quarterly Disciplinary Review summarizes actions without the case write-ups
SKIP_RE = re.compile(r"quarterly", re.I)


@dataclass
class Case:
    case_no: str
    report_url: str
    report_month: str | None
    section: str | None
    respondent: str | None
    action_date: date | None
    text: str
    masked_text: str
    summary_rules: str  # "|"-joined citation keys found in the summary itself (may be empty)


def clean(text: str) -> str:
    text = PAGE_HEADER_RE.sub("", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # rejoin words hyphenated across lines
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln and not ln.isdigit())


def pdf_text(data: bytes) -> str:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def html_text(data: bytes) -> str:
    soup = BeautifulSoup(data, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return (soup.find("main") or soup.body or soup).get_text("\n", strip=True)


def _parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(re.sub(r"\s+", " ", s), "%B %d, %Y").date()
    except ValueError:
        return None


def segment(text: str, report_url: str = "", report_month: str | None = None) -> list[Case]:
    text = clean(text)
    cases, prev_end, section = [], 0, None
    for m in CASE_END_RE.finditer(text):
        block = text[prev_end:m.start()]
        prev_end = m.end()
        lines = block.splitlines()
        # a section heading can sit between cases; keep the last one seen
        body_start = 0
        for i, ln in enumerate(lines):
            # a heading is followed within a few lines by a respondent line; this keeps
            # wrapped body lines that start with "Firm..." from being read as headings
            if SECTION_RE.match(ln) and "(CRD" not in ln and len(ln) <= 160 and any(
                "(CRD" in nxt for nxt in lines[i + 1: i + 6]
            ):
                section, body_start = ln.strip(), i + 1
        lines = lines[body_start:]
        respondent = None
        for ln in lines[:4]:
            cm = CRD_RE.match(ln)
            if cm:
                respondent = cm.group("name").strip()
                break
        body = " ".join(lines).strip()
        body = re.sub(r"\s+", " ", body)
        dm = ACTION_DATE_RE.search(body)
        cases.append(Case(
            case_no=m.group(1),
            report_url=report_url,
            report_month=report_month,
            section=section,
            respondent=respondent,
            action_date=_parse_date(dm.group(1)) if dm else None,
            text=body,
            masked_text=mask(body),
            summary_rules="|".join(dict.fromkeys(c.key for c in extract_citations(body))),
        ))
    return cases


def report_month_from(url: str, text: str) -> str | None:
    """The report's month from its URL, then its first lines, then the upload folder (YYYY-MM)."""
    url = url.replace("%20", " ")
    for src in (url, text[:400]):
        m = REPORT_MONTH_RE.search(src)
        if m:
            return datetime.strptime(f"{m.group(1).title()} {m.group(2)}", "%B %Y").strftime("%Y-%m")
    name = url.rsplit("/", 1)[-1]
    mon, yr = NAME_MONTH_RE.search(name), NAME_YEAR_RE.search(name)
    folder = FOLDER_RE.search(url)
    if mon and (yr or folder):
        month = [m[:3] for m in MONTHS].index(mon.group(1).lower()[:3]) + 1
        if yr:
            return f"{yr.group(1)}-{month:02d}"
        fy, fm = int(folder.group(1)), int(folder.group(2))
        return f"{fy - (month > fm)}-{month:02d}"  # a December report uploaded in January
    return None


def _links(html: bytes, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        if LINK_RE.search(href.split("?")[0]) and not SKIP_RE.search(href):
            out.append(href.split("#")[0])
    return list(dict.fromkeys(out))


def _pdf_on_page(html: bytes, page_url: str) -> str | None:
    """A month's HTML page usually links its PDF; return it if present."""
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        if (href.lower().endswith(".pdf") and re.search(r"disc?iplin", href + a.get_text(), re.I)
                and not SKIP_RE.search(href)):
            return href
    return None


def _month_range(start: str, end: str) -> list[tuple[int, int]]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def discover(client: PoliteClient, index_url: str = INDEX_URL, start: str = "2016-01",
             end: str | None = None, max_pages: int = 40) -> dict[str, str]:
    """One report URL per month (YYYY-MM -> URL), preferring PDFs.

    Sources: links on the index page and its ?page=N pages, plus month pages built from the
    two URL patterns FINRA has used. Each month page is opened to find its PDF.
    """
    end = end or date.today().strftime("%Y-%m")
    candidates: list[str] = []
    for page in range(max_pages):
        url = index_url if page == 0 else f"{index_url}?page={page}"
        status, body = client.get(url)
        if status != 200:
            break
        found = [u for u in _links(body, url) if u not in candidates]
        if not found and page > 0:
            break
        candidates += found
    for y, m in _month_range(start, end):
        for pat in MONTH_PAGE_PATTERNS:
            candidates.append(pat.format(month=MONTHS[m - 1], year=y))

    by_month: dict[str, str] = {}
    statuses: dict[int, int] = {}
    for url in dict.fromkeys(candidates):
        month = report_month_from(url, "")
        if month is None or not (start <= month <= end):
            continue
        if month in by_month and by_month[month].lower().endswith(".pdf"):
            continue
        if url.lower().endswith(".pdf"):
            by_month[month] = url
            continue
        status, body = client.get(url)
        statuses[status] = statuses.get(status, 0) + 1
        if status != 200:
            continue
        by_month[month] = _pdf_on_page(body, url) or url

    # months still missing or found only as an HTML page: guess the PDF under the report month's and
    # the prior month's upload folders, and the publication_file folders used through 2018
    for y, m in _month_range(start, end):
        month = f"{y}-{m:02d}"
        if month in by_month and by_month[month].lower().endswith(".pdf"):
            continue
        py, pm = (y - 1, 12) if m == 1 else (y, m - 1)
        for folder in (f"{y}-{m:02d}", f"{py}-{pm:02d}", "publication_file", ""):
            for pat in PDF_NAME_PATTERNS:
                name = pat.format(Month=MONTHS[m - 1].title(), month=MONTHS[m - 1], year=y)
                url = f"https://www.finra.org/sites/default/files/{folder}/{name}".replace("files//", "files/")
                status, body = client.get(url)
                statuses[status] = statuses.get(status, 0) + 1
                if status == 200 and body[:4] == b"%PDF":
                    by_month[month] = url
                    break
            if by_month.get(month, "").lower().endswith(".pdf"):
                break
    other = {k: v for k, v in statuses.items() if k not in (200, 404)}
    if other:
        print(f"responses other than 200/404 (possible blocking): {other}")
    return dict(sorted(by_month.items()))


def parse_all(client: PoliteClient, urls: list[str]) -> list[Case]:
    out = []
    for url in urls:
        status, body = client.get(url)
        if status != 200:
            print(f"  skip {status}: {url}")
            continue
        text = pdf_text(body) if url.lower().endswith(".pdf") or body[:4] == b"%PDF" else html_text(body)
        month = report_month_from(url, text)
        cases = segment(text, url, month)
        print(f"  {month or '?'}: {len(cases)} cases  {url}{'  <- NO CASES' if not cases else ''}")
        out.extend(cases)
    return out


def save(cases: list[Case], db: str) -> int:
    """One row per case number; a case summarized in more than one report keeps the latest report's row."""
    df = pd.DataFrame([asdict(c) for c in cases]).drop_duplicates("case_no", keep="last")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE raw.monthly_cases AS SELECT * FROM df")
    con.close()
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--index", default=INDEX_URL)
    d.add_argument("--out", default="data/monthly_urls.txt")
    d.add_argument("--start", default="2016-01")
    d.add_argument("--end", default=None)
    u = sub.add_parser("dump", help="save a downloaded report as a file, to inspect one that yields no cases")
    u.add_argument("--url", required=True)
    u.add_argument("--out", default="data/dump")
    p = sub.add_parser("parse")
    p.add_argument("--urls", default="data/monthly_urls.txt")
    p.add_argument("--db", default="data/finra.duckdb")
    a = ap.parse_args()
    client = PoliteClient(cache_dir="data/cache/finra")
    if a.cmd == "discover":
        found = discover(client, a.index, a.start, a.end)
        with open(a.out, "w") as f:
            f.write("\n".join(found.values()))
        expected = [f"{y}-{m:02d}" for y, m in _month_range(a.start, a.end or date.today().strftime("%Y-%m"))]
        missing = [mo for mo in expected if mo not in found]
        n_pdf = sum(u.lower().endswith(".pdf") for u in found.values())
        print(f"{len(found)} of {len(expected)} months found ({n_pdf} PDF, {len(found) - n_pdf} HTML) -> {a.out}")
        if missing:
            print("missing:", ", ".join(missing))
    elif a.cmd == "dump":
        status, body = client.get(a.url)
        ext = ".pdf" if body[:4] == b"%PDF" else ".html"
        with open(a.out + ext, "wb") as f:
            f.write(body)
        print(f"{status}: {len(body)} bytes -> {a.out}{ext}")
    else:
        urls = [u.strip() for u in open(a.urls) if u.strip() and not u.startswith("#")]
        cases = parse_all(client, urls)
        n = save(cases, a.db)
        print(f"Saved {n} cases to {a.db} ({len(cases)} write-ups; {len(cases) - n} repeat a case number "
              f"from another report, the latest is kept)")


if __name__ == "__main__":
    main()
