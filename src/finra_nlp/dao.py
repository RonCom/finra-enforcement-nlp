"""Labels from full case documents in FINRA Disciplinary Actions Online (DAO).

Recent monthly summaries often omit rule numbers (the July 2026 report cites them in a
minority of cases), so labels come from the AWC, complaint or decision, where the
violations section names each rule.

The DAO page layout is not verified yet. Run `probe` on one case first and confirm it
finds the document link before running `labels` on everything.

The search page is rate-limited hard (429s even at one request per 4 s), so `index` first reads DAO's
full listing, newest first, 15 or more documents per page, into raw.dao_index (case number, document
URL, type, action date). `labels` then takes documents from the index and searches only for cases the
index doesn't have (none with --index-only).

Usage:
    uv run python -m finra_nlp.dao probe --case 2023077018401
    uv run python -m finra_nlp.dao index --pages 2        # check the parse first
    uv run python -m finra_nlp.dao index
    uv run python -m finra_nlp.dao labels --db data/finra.duckdb
"""

from __future__ import annotations

import argparse
import re
import time
from urllib.parse import unquote, urljoin

import duckdb
import pandas as pd
from bs4 import BeautifulSoup

from finra_nlp.citations import extract_citations, fix_ocr_numbers
from finra_nlp.http import PoliteClient
from finra_nlp.monthly import pdf_text

SEARCH_URL = "https://www.finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions?search={case}"
# a sentence runs to the next period; a period between digits ("$2.5 million") doesn't end it
VIOLATION_RE = re.compile(
    r"(?:[^.]|\.(?=\d))*\b(?:violated|violation|violations|in contravention of)\b(?:[^.]|\.(?=\d))*\.",
    re.IGNORECASE,
)


def document_links(html: bytes, page_url: str, case_no: str) -> list[str]:
    """PDFs whose file name or link text carries the case number. The search is full-text: a case
    that isn't in DAO still returns a page of other cases' documents, so nothing else is taken."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        if href.lower().endswith(".pdf") and (case_no in href or case_no in a.get_text()):
            links.append(href)
    return list(dict.fromkeys(links))


LIST_URL = ("https://www.finra.org/rules-guidance/oversight-enforcement/finra-disciplinary-actions"
            "?search=&order=field_core_official_dt&sort=desc&items_per_page={per_page}&page={page}")
CASE_IN_NAME_RE = re.compile(r"^(\d{11,14})\b")


def parse_listing(html: bytes, page_url: str) -> list[dict]:
    """One dict per document row of a DAO listing page: case_no, doc_url, doc_type, respondent, action_date."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for tr in soup.select("table tr"):
        cell = tr.select_one("td.views-field-field-fda-attachment-file-media")
        if cell is None:
            continue
        a = cell.find("a", href=True)
        if a is None:
            continue
        url = urljoin(page_url, a["href"])
        m = CASE_IN_NAME_RE.match(unquote(url.rsplit("/", 1)[-1]))
        text = re.sub(r"\D", "", cell.get_text())
        case_no = m.group(1) if m else (text or None)

        def col(cls):
            td = tr.select_one(f"td.{cls}")
            return td.get_text(" ", strip=True) if td else None
        date = pd.to_datetime(col("views-field-field-core-official-dt"), format="%m/%d/%Y", errors="coerce")
        out.append({"case_no": case_no, "doc_url": url, "doc_type": col("views-field-field-fda-document-type-tax"),
                    "respondent": col("views-field-views-conditional-field-3"),
                    "action_date": None if pd.isna(date) else date.date()})
    return out


def build_index(client: PoliteClient, db: str, since: str = "2015-01-01", pages: int | None = None,
                per_page: int = 100) -> pd.DataFrame:
    """Read DAO's listing newest first until every row on a page is older than `since`."""
    rows, page, stop = [], 0, pd.Timestamp(since).date()
    while pages is None or page < pages:
        url = LIST_URL.format(per_page=per_page, page=page)
        status, html = client.get(url)
        if status != 200:
            print(f"  page {page}: status {status}; stopping (rerun to continue; pages read are cached)", flush=True)
            break
        got = parse_listing(html, url)
        if not got:
            print(f"  page {page}: no rows; end of the listing", flush=True)
            break
        rows += got
        dates = [r["action_date"] for r in got if r["action_date"]]
        if page % 10 == 0 or (pages is not None and page < pages):
            print(f"  page {page}: {len(got)} rows, action dates {min(dates, default=None)} to "
                  f"{max(dates, default=None)}; {len(rows)} documents so far | 1 request per "
                  f"{client.min_interval:.1f} s, {client.throttled} 429s", flush=True)
        if dates and max(dates) < stop:
            break
        page += 1
    df = pd.DataFrame(rows, columns=["case_no", "doc_url", "doc_type", "respondent", "action_date"])
    df = df.drop_duplicates("doc_url")
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE raw.dao_index AS SELECT * FROM df")
    covered = con.execute("""SELECT count(*), count(i.case_no) FROM raw.monthly_cases m
                             LEFT JOIN (SELECT DISTINCT case_no FROM raw.dao_index) i USING (case_no)""").fetchone()
    con.close()
    print(f"{len(df)} documents for {df.case_no.nunique()} case numbers in raw.dao_index; "
          f"{covered[1]} of {covered[0]} monthly-report cases have a document there")
    return df


WAIVER_SECTION_RE = re.compile(r"WAIVER\s+OF\s+PROCEDURAL\s+RIGHTS", re.IGNORECASE)
WAIVE_RE = re.compile(r"\bwaive[sd]?\b", re.IGNORECASE)  # the verb; "sales charge waivers" is conduct
# Sentences about earlier cases (a "Relevant Disciplinary History" section, prior AWCs, actions against
# others) cite rules this case didn't charge.
PRIOR_RE = re.compile(
    r"DISCIPLINARY\s+HISTORY|accepted an AWC|In an AWC|entry of an AWC|consented to (?:a|the) (?:censure|fine)"
    r"|in which (?:it|the firm|he|she|Respondent) was (?:censured|fined|suspended|barred)|filed a complaint against"
    r"|entered into an AWC|issued a Letter of Acceptance|Prior Matter|consented to violations of",
    re.IGNORECASE,
)
# The Code of Procedure (FINRA 9000-9999: AWCs, waivers of rights, hearings, defaults, appeals), the SEC's
# standard for reviewing FINRA actions (Exchange Act Section 19) and FINRA's statutory mandate (Section 15A)
# are cited in decisions and AWCs, never charged.
PROCEDURAL_SECTIONS = {"SEC_SECTION:19", "SEC_SECTION:15A"}


def procedural(key: str) -> bool:
    family, _, rule = key.partition(":")
    return key in PROCEDURAL_SECTIONS or (family == "FINRA" and re.fullmatch(r"9\d{3}", rule) is not None)


def labels_from_document(text: str) -> list[str]:
    """Citation keys from sentences that state violations, in order of first appearance.

    AWCs end with a procedural-rights waiver that mentions Rules 9143 and 9144 next to
    the word "violated"; text from that heading on is dropped, as is any sentence that waives.
    Sentences about earlier cases (PRIOR_RE) are dropped too: they cite rules from other cases.
    """
    text = fix_ocr_numbers(re.sub(r"\s+", " ", text))
    cut = WAIVER_SECTION_RE.search(text)
    if cut:
        text = text[: cut.start()]
    keys: dict[str, None] = {}
    for sent in VIOLATION_RE.finditer(text):
        if WAIVE_RE.search(sent.group(0)) or PRIOR_RE.search(sent.group(0)):
            continue
        for c in extract_citations(sent.group(0)):
            if not procedural(c.key):
                keys[c.key] = None
    return list(keys)


def probe(client: PoliteClient, case_no: str) -> None:
    url = SEARCH_URL.format(case=case_no)
    status, html = client.get(url, use_cache=False)
    print(f"{status} {url} ({len(html)} bytes)")
    links = document_links(html, url, case_no)
    print("document links:", links or "NONE FOUND - page may be rendered by JavaScript")
    if links:
        s, pdf = client.get(links[0])
        text = pdf_text(pdf)
        print(f"{s} {links[0]} ({len(text)} chars)")
        print("labels:", labels_from_document(text))


def build_labels(client: PoliteClient, db: str, limit: int | None, pdf_client: PoliteClient | None = None,
                 every: int = 10, index_only: bool = False) -> None:
    """pdf_client fetches the documents; they're static files, so they can go faster than the search
    page, which is what draws 429s. Prints progress every `every` cases."""
    pdf_client = pdf_client or client
    con = duckdb.connect(db)
    case_nos = [r[0] for r in con.execute("SELECT case_no FROM raw.monthly_cases ORDER BY case_no").fetchall()]
    if limit:
        case_nos = case_nos[:limit]
    indexed: dict[str, list[str]] = {}
    try:
        for c, u in con.execute("SELECT case_no, doc_url FROM raw.dao_index ORDER BY action_date DESC").fetchall():
            indexed.setdefault(c, []).append(u)
    except duckdb.CatalogException:
        print("No raw.dao_index; every case is searched (run `dao index` first to skip most searches)")
    need = [c for c in case_nos if c not in indexed]
    cached = sum(client.cached(SEARCH_URL.format(case=c)) for c in need)
    print(f"{len(case_nos)} cases: {len(case_nos) - len(need)} from the DAO index, {len(need)} to search "
          f"({cached} already cached){'; searches skipped (--index-only)' if index_only else ''}", flush=True)
    rows, start, found = [], time.monotonic(), 0
    for i, case_no in enumerate(case_nos, 1):
        if case_no in indexed:
            status, links = 200, indexed[case_no]
        elif index_only:
            status, links = 0, []
        else:
            url = SEARCH_URL.format(case=case_no)
            status, html = client.get(url)
            if status != 200:  # retried by the client and not cached; a rerun tries again
                print(f"  {case_no}: search returned {status}", flush=True)
            links = document_links(html, url, case_no) if status == 200 else []
        keys, doc_url = [], None
        for link in links:
            s, pdf = pdf_client.get(link)
            if s == 200:
                keys, doc_url = labels_from_document(pdf_text(pdf)), link
                if keys:
                    break
        found += bool(keys)
        rows.append({"case_no": case_no, "doc_url": doc_url, "doc_rules": "|".join(keys), "n_docs": len(links),
                     "search_status": status})
        if i % every == 0 or i == len(case_nos):
            took = time.monotonic() - start
            left = (len(case_nos) - i) * took / i
            print(f"  {i}/{len(case_nos)} cases, {found} labeled | {took / 60:.0f} min so far, about "
                  f"{left / 3600:.1f} h left at this pace | search 1 per {client.min_interval:.1f} s, "
                  f"{client.throttled} 429s", flush=True)
    con.register("df", pd.DataFrame(rows))
    con.execute("CREATE OR REPLACE TABLE raw.case_document_labels AS SELECT * FROM df")
    found = sum(1 for r in rows if r["doc_rules"])
    no_doc = sum(1 for r in rows if r["search_status"] == 200 and not r["n_docs"])
    failed = sum(1 for r in rows if r["search_status"] not in (0, 200))
    skipped = sum(1 for r in rows if r["search_status"] == 0)
    unlabeled = len(rows) - found - no_doc - failed - skipped
    print(f"Labels for {found}/{len(rows)} cases; {no_doc} not in DAO (no document with the case number), "
          f"{unlabeled} with a document but no rule found, {failed} searches failed (rerun to retry), "
          f"{skipped} not in the index and not searched")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--case", required=True)
    lb = sub.add_parser("labels")
    lb.add_argument("--db", default="data/finra.duckdb")
    lb.add_argument("--limit", type=int)
    lb.add_argument("--index-only", action="store_true", help="don't search for cases missing from the index")
    ix = sub.add_parser("index")
    ix.add_argument("--db", default="data/finra.duckdb")
    ix.add_argument("--pages", type=int, help="stop after this many pages (to check the parse)")
    ix.add_argument("--since", default="2015-01-01")
    a = ap.parse_args()
    # 2/s and then 1/s drew 429s on the search page; back off up to 1 request per 4 s, and wait
    # 30, 60, 90 ... s between retries
    def log(msg):
        print(f"  search: {msg}", flush=True)

    # the search page answers 429 for minutes once tripped; speeding back up after 25 successes tripped it
    # again within a minute, so the search pace only recovers after 300 successes in a row
    client = PoliteClient(cache_dir="data/cache/finra", max_per_second=1.0, max_retries=6,
                          retry_wait=30.0, max_interval=4.0, recover_after=300, log=log)
    # documents drew 429s too, after about 400 cases at 2/s; recover slowly, as for the search page
    pdf_client = PoliteClient(cache_dir="data/cache/finra", max_per_second=2.0, max_retries=6, retry_wait=30.0,
                              max_interval=4.0, recover_after=300,
                              log=lambda m: print(f"  document: {m}", flush=True))
    if a.cmd == "probe":
        probe(client, a.case)
    elif a.cmd == "index":
        build_index(client, a.db, a.since, a.pages)
    else:
        build_labels(client, a.db, a.limit, pdf_client, index_only=a.index_only)


if __name__ == "__main__":
    main()
