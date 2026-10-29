"""
Stage 3 / step 1 — collect GovHK press-release HEADLINES (not article bodies).

WHY HEADLINES AND NOT FULL TEXT
-------------------------------
The daily archive page already carries one line per press release, and that line
is the event in the government's own words ("Retail sales up 3.2 per cent in
December").  Fetching 100k+ article bodies would multiply the request count by
roughly 30x, take hours, and add almost no signal beyond the headline.  So the
pipeline keeps the headline list only, which is a deliberate, documented
scope decision rather than a shortcut.

MARKUP HAS TWO ERAS (verified)
    legacy  <li><a href="http://www.info.gov.hk/gia/general/...">Title</a></li>
    modern  <li><a class="NEW" href="/gia/general/YYYYMM/DD/P....htm">Title</a></li>
Both are matched by one regex.

SAMPLING
--------
Fetching every day from 2004-10 to 2024-12 is ~7,300 requests.  The pipeline
samples 7 fixed days per month (1, 5, 10, 15, 20, 25, 28) which keeps the count
COMPARABLE ACROSS MONTHS (every month gets the same 7 chances) and cuts the run
to ~1,700 requests.  A month's `news_count` is therefore "headlines on those 7
days", not "headlines in the month"; the feature is only ever used as a
cross-month relative measure, which is what the model sees anyway.

CACHE
-----
One JSON per month under `stage3/data/news_cache/YYYYMM.json`, so a re-run costs
nothing and a partial run can resume.  `--days all` switches to every day.

Usage
    .venv\\Scripts\\python.exe ../stage3/news_fetch.py
    .venv\\Scripts\\python.exe ../stage3/news_fetch.py --days all --workers 6
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "news_cache"
BASE = "https://www.info.gov.hk/gia/general/{ym}/{dd}.htm"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SAMPLE_DAYS = [1, 5, 10, 15, 20, 25, 28]
START, END = (2004, 10), (2024, 12)

_TAG = re.compile(r"<[^>]+>")

# Three markup eras, all verified live:
#   2004-2007  <a href="05/1105198.htm">Title</a>                      (day/seq)
#   2008-2016  <a href="http://www.info.gov.hk/gia/general/...">T</a>  (absolute)
#   2017-2024  <a class="NEW" href="/gia/general/YYYYMM/DD/P...">T</a> (relative)
# The href test is what separates a press release from the surrounding chrome
# (javascript: font-size toggles, gov.hk sitemap, language switch), so no extra
# block-list is needed.
_HREF_OK = re.compile(
    r"^(?:https?://www\.info\.gov\.hk)?"
    r"(?:/gia/general/\d{6}/\d{2}/(?:P?\d+|[A-Z]\w*)\.htm|\d{2}/\d{4,}\.htm)$")
_ANCHOR = re.compile(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)


def months():
    y, m = START
    while (y, m) <= END:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def parse_titles(text: str) -> list[str]:
    out, seen = [], set()
    for href, raw in _ANCHOR.findall(text):
        if not _HREF_OK.match(href.strip()):
            continue
        t = htmlmod.unescape(_TAG.sub("", raw))
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > 8 and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def fetch_day(ym: str, dd: int) -> tuple[str, list[str], str]:
    """returns (date_key, titles, status)"""
    key = f"{ym}{dd:02d}"
    url = BASE.format(ym=ym, dd=f"{dd:02d}")
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                return key, parse_titles(r.content.decode("utf-8", "ignore")), "200"
            if r.status_code == 404:
                return key, [], "404"
        except Exception:                                   # noqa: BLE001
            time.sleep(1.0 + attempt)
    return key, [], "ERR"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="sample", help="'sample' (7 fixed days) or 'all'")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    days = SAMPLE_DAYS if args.days == "sample" else list(range(1, 32))
    todo = []
    for y, m in months():
        ym = f"{y}{m:02d}"
        f = CACHE / f"{ym}.json"
        if f.exists():
            continue
        todo.append(ym)
    print(f"months total {sum(1 for _ in months())}, cached "
          f"{sum(1 for y, m in months() if (CACHE / f'{y}{m:02d}.json').exists())}, "
          f"to fetch {len(todo)}  (days={len(days)})")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ym in todo:
            futs = [ex.submit(fetch_day, ym, d) for d in days]
            rows, codes = [], {}
            for fu in as_completed(futs):
                key, titles, code = fu.result()
                codes[code] = codes.get(code, 0) + 1
                if titles:
                    rows.append({"date": key, "titles": titles})
            rows.sort(key=lambda r: r["date"])
            (CACHE / f"{ym}.json").write_text(
                json.dumps({"month": ym, "days": len(days), "codes": codes,
                            "n_days_with_news": len(rows), "rows": rows},
                           ensure_ascii=False), encoding="utf-8")
            done += 1
            if done % 12 == 0:
                print(f"  {done}/{len(todo)} months  ({time.time()-t0:.0f}s)  "
                      f"last={ym} days_with_news={len(rows)}", flush=True)

    # ---- flatten to a single csv -----------------------------------------
    import csv
    out = DATA / "news_headlines.csv"
    n = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "month", "title"])
        for y, m in months():
            ym = f"{y}{m:02d}"
            p = CACHE / f"{ym}.json"
            if not p.exists():
                continue
            blob = json.loads(p.read_text(encoding="utf-8"))
            for row in blob["rows"]:
                for t in row["titles"]:
                    w.writerow([row["date"], ym, t])
                    n += 1
    print(f"\nwrote {out.name}: {n:,} headlines over {len(list(months()))} months "
          f"in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
