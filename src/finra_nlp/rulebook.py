"""FINRA Rulebook download for the RAG corpus (spec, "Data sources").

The spec requires a terms-of-use check before any bulk download. `terms` finds the terms page
from the finra.org footer, saves it, and writes reports/rulebook_terms.md with every passage on
copying, downloading, automated access or commercial use, plus robots.txt's rules for the
Rulebook path. Read that file. If the terms allow the download, record the decision with
`terms --accept`, which writes data/rulebook_terms_accepted.txt; `download` refuses to run
without it.

The Rulebook's page layout is not verified yet: run `download --limit 5` and check the rows
before the full run.

Usage:
    uv run python -m finra_nlp.rulebook terms
    uv run python -m finra_nlp.rulebook terms --accept "read 2026-10-09; non-commercial research allowed"
    uv run python -m finra_nlp.rulebook download --limit 5
    uv run python -m finra_nlp.rulebook download
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import duckdb
import pandas as pd
from bs4 import BeautifulSoup

from finra_nlp.http import PoliteClient
from finra_nlp.monthly import html_text

HOME = "https://www.finra.org/"
ROBOTS = "https://www.finra.org/robots.txt"
INDEX_URL = "https://www.finra.org/rules-guidance/rulebooks/finra-rules"
TERMS_CANDIDATES = ["https://www.finra.org/terms-use", "https://www.finra.org/terms-of-use"]
ACCEPTED = "data/rulebook_terms_accepted.txt"
TERMS_WORDS = re.compile(
    r"cop(?:y|ies|ying)|download|reproduc|redistribut|automat|robot|spider|scrap|crawl|"
    r"commercial|personal use|permission|license|framing|data mining", re.I)
RULE_LINK_RE = re.compile(r"/rules-guidance/rulebooks/finra-rules/(\d{4,5}(?:\.\d{2})?|IM-\d{4}-\d+)/?$", re.I)


def terms_link(html: bytes, page_url: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        if re.search(r"terms\s+(?:of\s+)?use", a.get_text(" ", strip=True), re.I):
            return urljoin(page_url, a["href"])
    return None


def terms_passages(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z0-9])", text) if p.strip()]
    return [p for p in paras if TERMS_WORDS.search(p)]


def robots_rules(robots_txt: str, url: str = INDEX_URL) -> tuple[bool, list[str]]:
    rp = RobotFileParser()
    rp.parse(robots_txt.splitlines())
    lines = [ln for ln in robots_txt.splitlines()
             if re.match(r"\s*(user-agent|disallow|allow|crawl-delay)\s*:", ln, re.I)
             and ("rule" in ln.lower() or ln.lower().startswith(("user-agent", "crawl")))]
    return rp.can_fetch("*", url), lines


def terms(client: PoliteClient, out: str) -> str:
    lines = ["# FINRA terms of use and robots.txt, checked before the Rulebook download", "",
             f"Checked {date.today()}.", ""]
    status, home = client.get(HOME, use_cache=False)
    candidates = ([terms_link(home, HOME)] if status == 200 else []) + TERMS_CANDIDATES
    for url in [u for u in dict.fromkeys(candidates) if u]:
        s, body = client.get(url, use_cache=False)
        if s == 200:
            text = html_text(body)
            Path("data").mkdir(exist_ok=True)
            Path("data/finra_terms.txt").write_text(text, encoding="utf-8")
            lines += [f"## Terms page: {url}", "", "Full text saved to data/finra_terms.txt.", "",
                      "Passages on copying, downloading, automated access or commercial use:", ""]
            lines += [f"> {p}\n" for p in terms_passages(text)] or ["(none matched; read the full text)", ""]
            break
        lines += [f"- {url}: status {s}"]
    else:
        lines += ["", "No terms page found. Find it in the finra.org footer and read it by hand.", ""]
    s, body = client.get(ROBOTS, use_cache=False)
    if s == 200:
        allowed, rules = robots_rules(body.decode("utf-8", "replace"))
        lines += ["## robots.txt", "", f"Rulebook index allowed for '*': {allowed}", "", "```", *rules, "```", ""]
    else:
        lines += [f"## robots.txt: status {s}", ""]
    lines += ["Record the decision with `uv run python -m finra_nlp.rulebook terms --accept \"<what you read>\"`."]
    text = "\n".join(lines)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    return text


def rule_links(html: bytes, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = [urljoin(page_url, a["href"]).split("#")[0].split("?")[0] for a in soup.find_all("a", href=True)]
    return list(dict.fromkeys(u for u in links if RULE_LINK_RE.search(urlparse(u).path)))


def parse_rule(html: bytes, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "")
    rule_no = RULE_LINK_RE.search(urlparse(url).path).group(1).upper()
    return {"rule_no": rule_no, "title": title, "url": url, "text": html_text(html)}


def download(client: PoliteClient, db: str, limit: int | None) -> pd.DataFrame:
    if not Path(ACCEPTED).exists():
        raise SystemExit(f"{ACCEPTED} not found. Run `rulebook terms`, read reports/rulebook_terms.md, "
                         "then record the decision with `rulebook terms --accept`.")
    status, index = client.get(INDEX_URL)
    if status != 200:
        raise SystemExit(f"Rulebook index returned {status}")
    queue, seen, rows = rule_links(index, INDEX_URL), set(), []
    while queue and (limit is None or len(rows) < limit):
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        s, body = client.get(url)
        if s != 200:
            print(f"  skip {s}: {url}")
            continue
        rows.append(parse_rule(body, url))
        queue += [u for u in rule_links(body, url) if u not in seen]  # series pages list their rules
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} pages")
    df = pd.DataFrame(rows, columns=["rule_no", "title", "url", "text"])
    con = duckdb.connect(db)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE raw.rulebook_pages AS SELECT * FROM df")
    con.close()
    print(f"Saved {len(df)} Rulebook pages to raw.rulebook_pages")
    print(df[["rule_no", "title"]].head(10).to_string(index=False))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("terms")
    t.add_argument("--out", default="reports/rulebook_terms.md")
    t.add_argument("--accept", metavar="NOTE", help="record that the terms allow the download, with a note")
    d = sub.add_parser("download")
    d.add_argument("--db", default="data/finra.duckdb")
    d.add_argument("--limit", type=int)
    a = ap.parse_args()
    client = PoliteClient(cache_dir="data/cache/finra")
    if a.cmd == "terms" and a.accept:
        Path(ACCEPTED).parent.mkdir(parents=True, exist_ok=True)
        Path(ACCEPTED).write_text(f"{date.today()}: {a.accept}\n", encoding="utf-8")
        print(f"Recorded in {ACCEPTED}")
    elif a.cmd == "terms":
        terms(client, a.out)
    else:
        download(client, a.db, a.limit)


if __name__ == "__main__":
    main()
