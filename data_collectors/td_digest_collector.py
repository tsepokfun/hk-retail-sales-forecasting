#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
td_digest_collector.py
======================
Collect Hong Kong monthly **vehicle first registration** statistics from the
Transport Department's "Monthly Traffic and Transport Digest".

Target tables
-------------
  4.1(a)  Registration and Licensing of Vehicles by Class of Vehicles
  4.1(c)  Net First Registration of Private Cars (Gross / Deregistration / Net)
  4.1(e)  First Registration of Private Cars by Make

Source eras (verified by probing, not assumed)
----------------------------------------------
  2003-01 .. 2003-12   table41.pdf                 (single Table 4.1, vector PDF)
  2004-01 .. 2017-12   table41a/b[/c][/d..h].pdf   (per-table vector PDFs)
  2018-01 .. 2026-07   table41a..h.xls             (real OLE2/BIFF .xls)

Every monthly issue is a ROLLING WINDOW (~5 years of annual rows + ~19 months of
monthly rows), so the full monthly history is assembled by scraping every issue
and merging vintages.  Both the raw per-vintage rows and a de-duplicated
"latest vintage" view are written.

Output columns
--------------
  date                  YYYY-MM (monthly) or YYYY (annual)
  granularity           monthly | annual
  issue                 digest vintage, YYYY-MM
  table                 4.1(a) | 4.1(c) | 4.1(e)
  vehicle_class_or_make entity (e.g. "Private Cars > Petrol", "TOYOTA")
  measure               e.g. "First Registration", "Total Licensed"
  value                 integer count
  source_url            exact URL the value came from
  fetched_at            ISO-8601 UTC timestamp

Usage
-----
  python td_digest_collector.py --listing-only
  python td_digest_collector.py --limit-issues 4          # smoke test
  python td_digest_collector.py                           # full crawl
  python td_digest_collector.py --analyze-only            # re-run analysis
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

BASE = "https://www.td.gov.hk"
DIGEST = "/en/transport_in_hong_kong/transport_figures/monthly_traffic_and_transport_digest"
INDEX_URL = BASE + DIGEST + "/index.html"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

FIRST_YEAR, LAST_YEAR = 2003, 2026
TARGET_TABLES = ("4.1(a)", "4.1(c)", "4.1(e)")

# asset basenames we care about, in preference order
ASSET_PATTERNS = [
    ("4.1(a)", r"table41a?\.(xlsx?|pdf)$"),
    ("4.1(c)", r"table41c\.(xlsx?|pdf)$"),
    ("4.1(e)", r"table41e\.(xlsx?|pdf)$"),
]

LONG_COLS = ["date", "granularity", "issue", "table", "vehicle_class_or_make",
             "measure", "value", "source_url", "fetched_at"]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _safe(s: str) -> str:
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(s))


# --------------------------------------------------------------------------
# fetching (polite, cached, retrying)
# --------------------------------------------------------------------------

class Blocked(Exception):
    """Site returned 403/429 repeatedly - stop instead of hammering."""


class Fetcher:
    def __init__(self, cache_dir: Path, sleep: float = 1.5, retries: int = 3,
                 timeout: int = 90, refresh: bool = False):
        import requests  # local import so --analyze-only works without it
        self.requests = requests
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep
        self.retries = retries
        self.timeout = timeout
        self.refresh = refresh
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/pdf,"
                      "application/vnd.ms-excel,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.meta_path = self.cache_dir / "_cache_index.json"
        self.meta = {}
        if self.meta_path.exists():
            try:
                self.meta = json.loads(self.meta_path.read_text("utf-8"))
            except Exception:
                self.meta = {}
        self._last_net = 0.0
        self.stats = {"cache_hits": 0, "net_200": 0, "net_404": 0, "errors": 0,
                      "consecutive_block": 0}
        self.http_log = []

    # -- cache ------------------------------------------------------------
    def _paths(self, url: str):
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        name = url.rsplit("/", 1)[-1] or "index.html"
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:60]
        stem = Path(name).stem
        ext = Path(name).suffix or ".bin"
        if len(stem) > 40:
            stem = stem[:40]
        return self.cache_dir / f"{h}_{stem}{ext}"

    def _polite(self):
        dt_ = time.time() - self._last_net
        if dt_ < self.sleep:
            time.sleep(self.sleep - dt_)
        self._last_net = time.time()

    def _save_meta(self):
        try:
            self.meta_path.write_text(json.dumps(self.meta, indent=1), "utf-8")
        except Exception:
            pass

    def fetch(self, url: str, expect_binary: bool = True):
        """Return (status, bytes|None). 404 -> (404, None). Raises Blocked/Exception."""
        p = self._paths(url)
        if p.exists() and not self.refresh:
            self.stats["cache_hits"] += 1
            rec = self.meta.get(url, {})
            return rec.get("status", 200), p.read_bytes()

        last = None
        for attempt in range(self.retries):
            self._polite()
            try:
                r = self.sess.get(url, timeout=self.timeout)
            except Exception as e:                      # network error
                last = f"exception {e!r}"
                self.stats["errors"] += 1
                time.sleep(2.0 * (2 ** attempt))
                continue

            self.http_log.append((url, r.status_code))
            if r.status_code == 200:
                p.write_bytes(r.content)
                self.meta[url] = {
                    "status": 200, "bytes": len(r.content),
                    "content_type": r.headers.get("Content-Type", ""),
                    "fetched_at": dt.datetime.now(dt.timezone.utc)
                                    .replace(microsecond=0).isoformat(),
                }
                self._save_meta()
                self.stats["net_200"] += 1
                self.stats["consecutive_block"] = 0
                return 200, r.content

            if r.status_code == 404:
                self.meta[url] = {"status": 404,
                                  "fetched_at": dt.datetime.now(dt.timezone.utc)
                                    .replace(microsecond=0).isoformat()}
                self._save_meta()
                self.stats["net_404"] += 1
                self.stats["consecutive_block"] = 0
                return 404, None

            if r.status_code in (403, 429, 503):
                self.stats["consecutive_block"] += 1
                if self.stats["consecutive_block"] >= 3:
                    raise Blocked(
                        f"site is blocking us: HTTP {r.status_code} on {url} "
                        f"(3rd consecutive 403/429/503) - stopping as instructed")
                time.sleep(5.0 * (2 ** attempt))
                last = f"HTTP {r.status_code}"
                continue

            last = f"HTTP {r.status_code}"
            time.sleep(2.0 * (2 ** attempt))

        raise RuntimeError(f"failed after {self.retries} attempts: {url} ({last})")

    def fetched_at(self, url: str) -> str:
        return self.meta.get(url, {}).get("fetched_at", "")


# --------------------------------------------------------------------------
# issue discovery
# --------------------------------------------------------------------------

MONTH_LINK = re.compile(r"/monthly_traffic_and_transport_digest/(\d{4})/(\d{6})/index\.html")
YEAR_LINK = re.compile(r"/monthly_traffic_and_transport_digest/(\d{4})/index\.html")
ASSET_LINK = re.compile(r'href\s*=\s*"([^"]+\.(?:xlsx?|pdf))"', re.I)

# Prefer the native spreadsheet over the printed PDF: from 2018 onwards TD
# publishes BOTH for the same table, and the HTML lists the .pdf first, so a
# naive first-match would silently parse the PDF for the whole modern era.
EXT_RANK = {".xls": 0, ".xlsx": 0, ".pdf": 1}


def enumerate_issues(f: Fetcher):
    """Return sorted list of (issue 'YYYY-MM', issue_page_url)."""
    found = {}

    def harvest(html: str, label: str):
        n = 0
        for m in MONTH_LINK.finditer(html):
            y, ym = m.group(1), m.group(2)
            issue = f"{ym[:4]}-{ym[4:]}"
            url = f"{BASE}{DIGEST}/{y}/{ym}/index.html"
            if issue not in found:
                found[issue] = url
                n += 1
        return n

    st, body = f.fetch(INDEX_URL)
    if st != 200:
        raise RuntimeError(f"index page HTTP {st}")
    html = body.decode("utf-8", "replace")
    log(f"[enum] main index {INDEX_URL} -> HTTP 200, {len(html)} chars, "
        f"{harvest(html, 'main')} month links")

    # old years are not linked as month pages from the main index -> per-year index
    year_ok = {}
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        yurl = f"{BASE}{DIGEST}/{y}/index.html"
        try:
            st, body = f.fetch(yurl)
        except Blocked:
            raise
        except Exception as e:
            log(f"[enum] {y}/index.html -> {e!r}")
            year_ok[y] = False
            continue
        if st != 200 or not body:
            log(f"[enum] {y}/index.html -> HTTP {st}")
            year_ok[y] = False
            continue
        n = harvest(body.decode("utf-8", "replace"), str(y))
        year_ok[y] = True
        log(f"[enum] {y}/index.html -> HTTP 200, {n} new month link(s)")

    # Only if a year index could not be read do we guess that year's month URLs
    # (the pages 404 if the issue does not exist, so this stays honest).
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        if year_ok.get(y, False):
            continue
        log(f"[enum] {y}: year index unavailable -> probing all 12 months by URL")
        for m in range(1, 13):
            issue = f"{y}-{m:02d}"
            found.setdefault(issue, f"{BASE}{DIGEST}/{y}/{y}{m:02d}/index.html")

    return sorted(found.items())


def find_assets(html: str, page_url: str):
    """Map table id -> absolute asset URL for one issue page (best format wins)."""
    best = {}
    for href in ASSET_LINK.findall(html):
        fn = href.rsplit("/", 1)[-1].lower()
        ext = "." + fn.rsplit(".", 1)[-1]
        for tid, pat in ASSET_PATTERNS:
            if re.search(pat, fn):
                rank = EXT_RANK.get(ext, 9)
                if tid not in best or rank < best[tid][0]:
                    url = href if href.startswith("http") else BASE + href
                    best[tid] = (rank, url)
    return {tid: v[1] for tid, v in best.items()}


def whole_issue_pdf(html: str, page_url: str):
    """The full-digest PDF linked from an issue page (fallback source)."""
    best = None
    for href in ASSET_LINK.findall(html):
        fn = href.rsplit("/", 1)[-1].lower()
        if not fn.endswith(".pdf"):
            continue
        if re.match(r"^\d{4}\.pdf$", fn):
            return href if href.startswith("http") else BASE + href
        if fn not in ("symbols.pdf",):
            best = best or (href if href.startswith("http") else BASE + href)
    return best


# --------------------------------------------------------------------------
# table 4.1(a) / 4.1(c) / 4.1(e)  --  XLSX/XLS era (2018+)
# --------------------------------------------------------------------------

def load_xls(content: bytes):
    """Open a real .xls (OLE2/BIFF) or .xlsx workbook; return list of (name, rows)."""
    if content[:4] == b"PK\x03\x04":
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        return [(ws.title, [list(r) for r in ws.iter_rows(values_only=True)])
                for ws in wb.worksheets]
    import xlrd
    bk = xlrd.open_workbook(file_contents=content)
    out = []
    for sh in bk.sheets():
        rows = []
        for r in range(sh.nrows):
            rows.append([sh.cell_value(r, c) for c in range(sh.ncols)])
        out.append((sh.name, rows))
    return out


def cell_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else str(v)
    return str(v).strip()


def as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return int(round(f)) if abs(f - round(f)) < 1e-9 else f
    s = cell_str(v).replace(",", "").replace(" ", "")
    if s in ("", "-", "--", "N.A.", "NA", "n.a.", "*", "#"):
        return None
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def ascii_part(s: str) -> str:
    """Keep only the ASCII portion of a cell (TD puts the Chinese label first)."""
    s = cell_str(s)
    s = "".join(ch if ord(ch) < 128 else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def ascii_label(s: str):
    """English (pure-ASCII, alphabetic) header label, or None."""
    s = ascii_part(s)
    if not s or not any(c.isalpha() for c in s):
        return None
    return s


MEASURE_VOCAB = (
    "first registration", "new registration", "total", "registration",
    "licensed", "brand new", "others", "sub-total", "sub total", "subtotal",
    "gross first registration", "net first registration",
    "cumulative deregistration", "year/month",
)


def is_measure_label(lab: str) -> bool:
    low = lab.lower().strip()
    return any(low.startswith(v) for v in MEASURE_VOCAB)


def find_header_rows(rows):
    """Locate (title_row, ym_row, first_data_row).

    The header block's shape changed over the years: in the 2018-2019 files a
    'First Registration' row sits ABOVE the 'Year/Month' row, while the 2020s
    files put the measure names ON that row.  So rather than assuming a fixed
    layout, the header rows are split by CONTENT (see header_col_map).
    """
    title_r, ym_r = None, None
    for i, r in enumerate(rows):
        joined = " ".join(cell_str(v) for v in r)
        if title_r is None and re.search(r"Table\s*4\.1", joined):
            title_r = i
        if ym_r is None and "Year/Month" in cell_str(r[0] if r else ""):
            ym_r = i
    if ym_r is None:
        return title_r, None, None

    first_data = None
    for r in range(ym_r + 1, len(rows)):
        v = as_number(rows[r][0]) if rows[r] else None
        if v is not None and 1900 <= int(v) <= 2100:
            first_data = r
            break
    if first_data is None:
        first_data = len(rows)
    return title_r, ym_r, first_data


def header_col_map(rows, title_r, first_data_r):
    """Build (entity_path, measure) for every column >= 2.

    Header rows are classified: a row whose English labels are ALL measure words
    is a measure row; any other row carries entity (class / group) labels.
    Entity rows are forward-filled because Excel writes a merged label once, at
    the first column of its group.
    """
    ncol = max((len(r) for r in rows), default=0)
    top_r = 0 if title_r is None else title_r + 1
    meas_rows, ent_rows = [], []
    for r in range(top_r, first_data_r):
        labs = [ascii_label(rows[r][c]) for c in range(len(rows[r]))]
        labs = [l for l in labs if l]
        if labs and all(is_measure_label(l) for l in labs):
            meas_rows.append(r)
        else:
            ent_rows.append(r)

    paths, measures = {}, {}
    for c in range(2, ncol):
        parts = []
        for r in ent_rows:
            row = rows[r]
            cur = None
            for cc in range(2, c + 1):          # forward-fill to column c
                lab = ascii_label(row[cc]) if cc < len(row) else None
                if lab:
                    cur = lab
            if cur and (not parts or parts[-1] != cur):
                parts.append(cur)
        raw = []
        for r in meas_rows:
            lab = ascii_label(rows[r][c]) if c < len(rows[r]) else None
            if lab:
                raw.append(lab)
        paths[c] = " > ".join(parts)
        measures[c] = canon_measure(" > ".join(raw))
    return paths, measures


FOOTNOTE = re.compile(r"\s*\((?:[0-9]{1,2}|[a-z])\)\s*$")
FOOTNOTE_LEAD = re.compile(r"^\s*\((?:[0-9]{1,2}|[a-z])\)\s*")


def clean_measure_label(s: str):
    """Strip footnote citations and collapse the verbose 4.1(c) column names."""
    if not s:
        return s
    s = re.sub(r"\s+", " ", s).strip()
    prev = None
    while prev != s:                      # "Net First Registration (1) (2)"
        prev = s
        s = FOOTNOTE.sub("", s).strip()
        s = FOOTNOTE_LEAD.sub("", s).strip()
    low = s.lower()
    if low.startswith("cumulative deregistration"):
        return "Cumulative Deregistration"
    if low.startswith("gross first registration") or low.startswith("gross new registration"):
        return "Gross First Registration"
    if low.startswith("net first registration") or low.startswith("net new registration"):
        return "Net First Registration"
    return s


def iter_dated_rows(rows, measure_r):
    """Yield (date, granularity, row) for annual then monthly data rows.

    Annual rows carry only a year in col 0; monthly rows carry a month in col 1,
    with the year repeated on the first monthly row of each year.
    """
    cur_year = None
    for r in range(measure_r + 1, len(rows)):
        row = rows[r]
        y = as_number(row[0]) if len(row) > 0 else None
        mo = as_number(row[1]) if len(row) > 1 else None
        if y is not None and 1900 <= int(y) <= 2100:
            cur_year = int(y)
        if cur_year is None:
            continue
        if mo is not None and 1 <= int(mo) <= 12:
            yield f"{cur_year}-{int(mo):02d}", "monthly", row
        elif y is not None and 1900 <= int(y) <= 2100:
            yield f"{cur_year}", "annual", row


def parse_xls_series(content: bytes, url: str, issue: str, table: str,
                     default_entity: str):
    """Shared parser for the Year/Month x measure tables 4.1(a) and 4.1(c)."""
    rows_out, warnings = [], []
    for name, rows in load_xls(content):
        if not rows:
            continue
        title_r, ym_r, first_data_r = find_header_rows(rows)
        if ym_r is None:
            warnings.append(f"{name}: no 'Year/Month' header cell found")
            continue
        paths, measures = header_col_map(rows, title_r, first_data_r)
        for date, gran, row in iter_dated_rows(rows, ym_r):
            for c in range(2, len(row)):
                meas = measures.get(c)
                if not meas:
                    continue
                val = as_number(row[c])
                if val is None:
                    continue
                ent = paths.get(c) or ascii_part(name) or default_entity
                rows_out.append((date, gran, issue, table, ent, meas, val, url))
    return rows_out, warnings


def parse_xls_41a(content: bytes, url: str, issue: str):
    return parse_xls_series(content, url, issue, "4.1(a)", "Total")


def parse_xls_41c(content: bytes, url: str, issue: str):
    return parse_xls_series(content, url, issue, "4.1(c)", "Private Cars")


def parse_xls_41e(content: bytes, url: str, issue: str):
    """4.1(e) is a single-month table: rows = (make, condition), last col = row total.

    Aggregated across condition rows (and across continuation sheets) to one
    'first registration' figure per make for the issue month.
    """
    rows_out, warnings = [], []
    per_make = {}
    for name, rows in load_xls(content):
        if not rows:
            continue
        ncol = max(len(r) for r in rows)
        hdr_r = None
        for i, r in enumerate(rows[:15]):
            if not r:
                continue
            if re.search(r"\bMake\b", ascii_part(r[0])):
                labels = sum(1 for c in range(1, len(r)) if ascii_label(r[c]))
                if labels >= 3:
                    hdr_r = i
                    break
        if hdr_r is None:
            warnings.append(f"{name}: no 'Make' header row")
            continue
        cur_make = None
        for r in range(hdr_r + 1, len(rows)):
            row = rows[r]
            lab = ascii_part(row[0]) if len(row) > 0 else ""
            if lab and lab.lower() != "make":
                cur_make = lab
            if not cur_make:
                continue
            total = as_number(row[ncol - 1]) if len(row) >= ncol else None
            if total is None:
                continue
            per_make[cur_make] = per_make.get(cur_make, 0) + total
    for make, total in per_make.items():
        # The grand-total row is labelled "Total" in some vintages and
        # "(2) Total (2)" (with table footnote markers) in others.  It is an
        # aggregate, not a make: leaving it in double-counts the whole table
        # (verified - 2018-01 then summed to exactly 2x the private-car value
        # that Table 4.1(c) reports for the same month).
        if clean_measure_label(make).lower().startswith(("total", "grand total")):
            continue
        rows_out.append((issue, "monthly", issue, "4.1(e)", make,
                         "First Registration", total, url))
    return rows_out, warnings


# --------------------------------------------------------------------------
# PDF positional extraction (2004-2017 era, identical Excel-print template)
# --------------------------------------------------------------------------

def pdf_page_bands(page, y_tol=2.0, x_gap=6.0):
    """Cluster words into text bands (rows) and within each band into cells."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return []
    words.sort(key=lambda w: (w["bottom"], w["x0"]))
    bands = []
    for w in words:
        if bands and abs(w["bottom"] - bands[-1][0]) <= y_tol:
            bands[-1][1].append(w)
        else:
            bands.append([w["bottom"], [w]])
    out = []
    for y, ws in bands:
        ws.sort(key=lambda z: z["x0"])
        cells, cur = [], []
        for w in ws:
            if cur and (w["x0"] - cur[-1]["x1"]) > x_gap:
                cells.append(cur)
                cur = []
            cur.append(w)
        if cur:
            cells.append(cur)
        out.append({
            "y": y,
            "cells": [{
                "x0": min(z["x0"] for z in g),
                "x1": max(z["x1"] for z in g),
                "tight": "".join(z["text"] for z in g),
                "spaced": " ".join(z["text"] for z in g),
            } for g in cells],
        })
    out.sort(key=lambda b: b["y"])
    return out


def _canon_component(p):
    p = clean_measure_label(p) or p     # drop footnote refs, shorten 4.1(c) heads
    low = p.lower().strip()
    if low == "brand new":
        return ["First Registration", "Brand New"]
    if low == "others":
        return ["First Registration", "Others"]
    if low in ("sub-total", "sub total", "subtotal"):
        return ["First Registration", "Sub-total"]
    if low.startswith("gross first regi") or low.startswith("gross new regi"):
        return ["Gross First Registration"]
    if low.startswith("net first regi") or low.startswith("net new regi"):
        return ["Net First Registration"]
    if low.startswith("cumulative dereg"):
        return ["Cumulative Deregistration"]
    if low.startswith("first regi") or low.startswith("new regi"):
        return ["First Registration"]
    if low.startswith("total regi"):
        return ["Total Registration"]
    if low.startswith("total lic"):
        return ["Total Licensed"]
    if low == "total":
        return []                      # merged with the component that follows
    return [p]


def canon_measure(s: str) -> str:
    """Normalise a PDF-era measure label.

    The template splits several measures across two header lines ("Total" sits
    one band above "Registration"), renames 'New Registration' to 'First
    Registration' over the years, and clips long labels ('Total Regi').  This
    rebuilds the canonical name from the observed vocabulary so the result does
    not depend on merged-title geometry.
    """
    parts = [p.strip() for p in re.split(r"\s*>\s*", re.sub(r"\s+", " ", s).strip())
             if p.strip()]
    comps, pending = [], False
    for p in parts:
        if p.lower().strip() == "total":
            pending = True
            continue
        c = _canon_component(p)
        if pending and c and c[0] in ("Total Registration", "Total Licensed"):
            comps.extend(c)
        elif pending and c and c[0] in ("Registration", "Licensed"):
            comps.append("Total " + c[0])
        else:
            comps.extend(c)
        pending = False
    if pending:
        comps.append("Total")
    out = []
    for c in comps:
        if not out or out[-1] != c:
            out.append(c)
    return " > ".join(out)


def cluster(vals, tol):
    vals = sorted(vals)
    cl = []
    for v in vals:
        if cl and v - cl[-1][-1] <= tol:
            cl[-1].append(v)
        else:
            cl.append([v])
    return [sum(c) / len(c) for c in cl]


def pdf_grid(path: Path):
    """Yield (page_index, page_text, bands, col_positions, pitch) for each page."""
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        for pi, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            bands = pdf_page_bands(page)
            yield pi, text, bands


def pdf_measure_and_entity(bands, hdr_i, cols):
    """Return per-column (measure, entity) labels from a page's header bands."""
    pitch = 30.0
    if len(cols) > 1:
        pitch = (cols[-1] - cols[0]) / (len(cols) - 1)

    def assign(band_cells, per_col):
        for cell in band_cells:
            lab = ascii_label(cell["spaced"])
            if not lab:
                continue
            c0 = (cell["x0"] - pitch * 0.5)
            c1 = (cell["x1"] + pitch * 0.5)
            for i, cx in enumerate(cols):
                if c0 <= cx <= c1:
                    per_col[i].append(lab)

    measures = [[] for _ in cols]
    entities = [[] for _ in cols]
    # measure window = the 'Year/Month' band +- 1 band
    for j in (hdr_i - 1, hdr_i, hdr_i + 1):
        if 0 <= j < len(bands):
            assign(bands[j]["cells"], measures)
    # entity window = everything above the measure window
    for j in range(0, hdr_i - 1):
        assign(bands[j]["cells"], entities)

    ent_out, meas_out = [], []
    for i in range(len(cols)):
        seen, parts = set(), []
        for x in entities[i]:
            if x not in seen:
                seen.add(x)
                parts.append(x)
        ent_out.append(" > ".join(parts))
        seen, parts = set(), []
        for x in measures[i]:
            if x not in seen:
                seen.add(x)
                parts.append(x)
        meas_out.append(" > ".join(parts))
    return ent_out, meas_out


def _col_spans(cols):
    """Each numeric column is right-aligned; its cell runs to the previous
    column's right edge.  Return (spans, centres, pitch)."""
    pitch = (cols[-1] - cols[0]) / (len(cols) - 1) if len(cols) > 1 else 30.0
    spans = []
    for i, r in enumerate(cols):
        lo = cols[i - 1] if i > 0 else r - pitch
        spans.append((lo, r))
    centres = [(lo + hi) / 2.0 for lo, hi in spans]
    return spans, centres, pitch


_BARE_DATE = re.compile(r"\d{4}(/\d{1,2})?|\d{1,2}")


def _is_data_like(band) -> bool:
    """True if this band looks like a data row (starts with a year or month)."""
    if not band["cells"]:
        return False
    return bool(re.fullmatch(_BARE_DATE,
                             band["cells"][0]["tight"].replace(" ", "")))


def _is_ym_band(band) -> bool:
    return bool(band["cells"]) and \
        band["cells"][0]["tight"].replace(" ", "").startswith("Year/Month")


def pdf_header_model(bands, hdr_i, cols, ctx_floor=0):
    """Return (class_path, group_names, measures).

    Header levels in these Excel-printed PDFs:
        sheet title / vehicle class / sub-classes / GROUP LABELS / measure names
    Group labels are centred over their column group, so segmentation is done
    by nearest-group-centre (Voronoi), which is exact for this template.

    Measure labels come in two shapes and need two different rules:
      * narrow labels ('Others', 'Registration') are drawn over ONE column and
        are matched to the single nearest column centre;
      * merged titles ('First Registration', 'Total') are centred over a span of
        sub-columns and are matched to every column within w/2 + pitch/2.
    Using one rule for both smears narrow labels onto their neighbours.
    """
    spans, centres, pitch = _col_spans(cols)

    def labels_of(band):
        """ASCII labels on this band that sit over the numeric columns."""
        out = []
        floor_x = (min(cols) - 5) if cols else 0
        for cell in band["cells"]:
            lab = ascii_label(cell["spaced"])
            if lab and cell["x1"] >= floor_x:        # drops the Year/Month cell
                out.append((cell["x0"], cell["x1"], lab))
        return out

    if not cols:
        return [], [], []

    measures = [[] for _ in cols]
    for j in (hdr_i - 1, hdr_i, hdr_i + 1):
        if not (0 <= j < len(bands)):
            continue
        labs = labels_of(bands[j])
        narrow = [(x0, x1, l) for x0, x1, l in labs if (x1 - x0) < 1.25 * pitch]
        wide = [(x0, x1, l) for x0, x1, l in labs if (x1 - x0) >= 1.25 * pitch]
        # narrow labels are drawn over the number they head, so their RIGHT edge
        # lines up with the right-aligned number's right edge (verified: max
        # error 9pt on the Motor Cycles page, 6pt on Private Cars).  Matching on
        # the label's centre instead is off by up to 50pt where the template's
        # merge ranges do not line up with the column pitch.
        for x0, x1, lab in narrow:
            i = min(range(len(cols)), key=lambda k: abs(cols[k] - x1))
            measures[i].append(lab)
        # merged titles ('First Registration', 'Gross First Registration') only
        # fill columns that no narrow label claimed.
        for x0, x1, lab in wide:
            lc = (x0 + x1) / 2.0
            half = (x1 - x0) / 2.0 + pitch / 2.0
            hit = [i for i, c in enumerate(centres) if abs(c - lc) <= half + 0.5]
            for i in hit:
                if not measures[i]:
                    measures[i].append(lab)

    # Context bands above the measure window, nearest-first.  The walk MUST stop
    # at the previous sub-table's data rows: a single PDF page can stack several
    # sub-tables (e.g. Public Light Buses, then Goods Vehicles, then the grand
    # Total), and running to the top of the page merges their class labels into
    # one bogus path like 'Goods Vehicles > Government Vehicles > Total'.
    cands = []
    for j in range(hdr_i - 2, ctx_floor - 1, -1):
        if _is_data_like(bands[j]):
            break
        labs = labels_of(bands[j])
        if labs:
            cands.append((j, labs))

    group_names, ctx = [], cands
    if cands and len(cands[0][1]) >= 2:
        group_names = [lab for _, _, lab in sorted(cands[0][1], key=lambda z: z[0])]
        ctx = cands[1:]

    class_parts = []
    for _j, labs in reversed(ctx):                      # top-down reading order
        blob = " ".join(lab for _, _, lab in labs)
        if re.search(r"\bTable\b|\bSection\b|\bChart\b|\bPage\b", blob):
            continue
        for _x0, _x1, lab in sorted(labs, key=lambda z: z[0]):
            if re.fullmatch(r"\d+(\.\d+)*\s*\(?[a-z]?\)?", lab):
                continue                                # bare table number "4.1(a)"
            if re.fullmatch(r"\(?cont'?d\.?\)?", lab, re.I):
                continue                                # "(cont'd)" page marker
            if len(lab) > 45:
                continue        # a table description, not a class/group name
            if lab not in class_parts:
                class_parts.append(lab)

    meas_out = []
    for i in range(len(cols)):
        seen, parts = set(), []
        for x in measures[i]:
            if x not in seen:
                seen.add(x)
                parts.append(x)
        meas_out.append(canon_measure(" > ".join(parts)))
    return class_parts, group_names, meas_out


def pdf_extract_rows(path: Path, table: str, url: str, issue: str):
    """Extract rows from a per-table PDF using word geometry.

    A page may stack SEVERAL sub-tables (each with its own 'Year/Month' header
    and its own vehicle class / group labels), so the page is split at every
    'Year/Month' band and each segment is parsed independently with its own
    context bounds.  Without that split the class labels of neighbouring
    sub-tables merge into one bogus entity path and the grand-total block gets
    attributed to whichever class happened to precede it.
    """
    rows_out, warnings = [], []
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        for pi, page in enumerate(pdf.pages):
            bands = pdf_page_bands(page)
            if not bands:
                continue
            ym_idx = [i for i, b in enumerate(bands) if _is_ym_band(b)]
            if not ym_idx:
                warnings.append(f"p{pi}: no Year/Month header band")
                continue

            for si, hdr_i in enumerate(ym_idx):
                seg_end = ym_idx[si + 1] if si + 1 < len(ym_idx) else len(bands)

                # A data row starts with the date cell, which sits directly
                # under the 'Year/Month' header cell.  Its absolute x varies a
                # lot between pages (x=57 on Motor Cycles, x=138 on Private
                # Cars), so anchor to the header cell, not a fixed coordinate.
                ym_cell = bands[hdr_i]["cells"][0]
                date_re = r"(\d{4}(/\d{1,2})?|\d{1,2})"

                def is_date_band(b):
                    if not b["cells"]:
                        return False
                    c0 = b["cells"][0]
                    if not re.fullmatch(date_re, c0["tight"].replace(" ", "")):
                        return False
                    return (abs(c0["x0"] - ym_cell["x0"]) <= 35
                            and c0["x1"] <= ym_cell["x1"] + 35)

                data_bands = [b for b in bands[hdr_i + 1:seg_end] if is_date_band(b)]
                if not data_bands:
                    warnings.append(f"p{pi} seg{si}: no data rows")
                    continue

                col_vals = [c["x1"] for b in data_bands for c in b["cells"][1:]]
                cols = cluster(col_vals, 8.0)
                if not cols:
                    warnings.append(f"p{pi} seg{si}: no numeric columns")
                    continue

                # stop the context walk at the previous sub-table's data rows
                floor = 0
                for j in range(hdr_i - 1, -1, -1):
                    if _is_ym_band(bands[j]) or _is_data_like(bands[j]):
                        floor = j + 1
                        break
                class_parts, group_names, meas_lab = pdf_header_model(
                    bands, hdr_i, cols, ctx_floor=floor)
                cls = " > ".join(class_parts) if class_parts else f"p{pi + 1}"

                # Segment columns into groups by nearest group-label centre
                # (Voronoi).  Exact for the fixed Excel print template and does
                # not depend on measure labels being perfectly recovered.
                _spans, centres, _pitch = _col_spans(cols)
                gname_of = {}
                if group_names:
                    gc = []
                    for b in bands[floor:hdr_i]:
                        for cell in b["cells"]:
                            lab = ascii_label(cell["spaced"])
                            mid = (cell["x0"] + cell["x1"]) / 2.0
                            if (lab in group_names
                                    and lab not in [g[0] for g in gc]
                                    and mid not in [g[1] for g in gc]):
                                gc.append((lab, mid))
                    if gc:
                        mids = [(gc[k][1] + gc[k + 1][1]) / 2.0
                                for k in range(len(gc) - 1)]
                        for i, c in enumerate(centres):
                            k = sum(1 for m in mids if c >= m)
                            gname_of[i] = gc[min(k, len(gc) - 1)][0]

                cur_year = None
                for b in data_bands:
                    head = b["cells"][0]["tight"].replace(" ", "")
                    m = re.fullmatch(r"(\d{4})/(\d{1,2})", head)
                    if m:
                        cur_year, month = int(m.group(1)), int(m.group(2))
                    elif re.fullmatch(r"\d{4}", head):
                        cur_year, month = int(head), None
                    elif re.fullmatch(r"\d{1,2}", head):
                        month = int(head)
                    else:
                        continue
                    if cur_year is None:
                        continue
                    if month is None:
                        date, gran = f"{cur_year}", "annual"
                    else:
                        date, gran = f"{cur_year}-{month:02d}", "monthly"

                    # map cells to columns (skip anything left of column 0)
                    bycol = {}
                    for c in b["cells"][1:]:
                        if c["x1"] < cols[0] - 5:
                            continue
                        idx = min(range(len(cols)),
                                  key=lambda i: abs(cols[i] - c["x1"]))
                        bycol.setdefault(idx, c["tight"])
                    for idx, raw in bycol.items():
                        if idx >= len(cols):
                            continue
                        val = as_number(raw)
                        if val is None:
                            continue
                        gname = gname_of.get(idx, "")
                        ent = f"{cls} > {gname}" if gname else cls
                        meas = meas_lab[idx] if idx < len(meas_lab) else ""
                        if not meas:
                            continue
                        rows_out.append((date, gran, issue, table, ent, meas,
                                         val, url))
    return rows_out, warnings


_E41_HEADER = re.compile(
    r"^(make|petrol|diesel|electric|others|brand\s?new|saloon|station\s?waggon|"
    r"waggon|convertible|total|grand\s?total)$", re.I)
_TABLE_NUM = re.compile(r"^\d+(\.\d+)*[()a-z]*$", re.I)


def _is_e41_make_label(lab: str) -> bool:
    """True if this ASCII label can anchor a make block.

    The make column sits at x~72 on one page but x~66 on the next, and the
    'Others' / 'Brand new' row labels sit close enough to be mistaken for makes,
    so they are excluded by vocabulary.  The grand-total row IS kept as an anchor
    (otherwise its number is absorbed by the last make on the page) and is
    dropped again when rows are emitted.
    """
    if not lab or _TABLE_NUM.match(lab):
        return False
    if lab.lower() in ("total", "grand total"):
        return True
    if _E41_HEADER.match(lab) or is_measure_label(lab):
        return False
    return True


def pdf_extract_rows_41e(path: Path, url: str, issue: str):
    """Table 4.1(e) in the PDF era is a single-month table with no Year/Month
    axis: one block per make, each block holding a 'Brand new' row and an
    'Others' row whose last cell is that row's total.  Numeric rows are attached
    to the nearest make name by band index, then summed per make.
    """
    rows_out, warnings = [], []
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        for pi, page in enumerate(pdf.pages):
            bands = pdf_page_bands(page)
            makes, nums = [], []
            for i, b in enumerate(bands):
                cells = b["cells"]
                if not cells:
                    continue
                c0 = cells[0]
                lab = ascii_label(c0["spaced"])
                if (lab and c0["x0"] < 125 and (c0["x1"] - c0["x0"]) < 70
                        and _is_e41_make_label(lab)):
                    makes.append((i, lab))
                    continue
                if len(cells) >= 8:
                    val = as_number(cells[-1]["tight"])
                    if val is not None:
                        nums.append((i, val))
            if not makes or not nums:
                warnings.append(f"p{pi}: 4.1(e) found {len(makes)} make(s), "
                                f"{len(nums)} numeric row(s)")
                continue
            per_make = {}
            for i, val in nums:
                mk = min(makes, key=lambda m: abs(m[0] - i))[1]
                per_make[mk] = per_make.get(mk, 0) + val
            for mk, tot in per_make.items():
                # The grand-total row is labelled "Total" in some vintages and
                # "(2) Total (2)" (with table footnote markers) in others.  It is
                # an aggregate, not a make, and leaving it in double-counts the
                # whole table (verified: 2018-01 summed to exactly 2x the value
                # that Table 4.1(c) reports for the same month).
                if clean_measure_label(mk).lower().startswith(("total", "grand total")):
                    continue
                rows_out.append((issue, "monthly", issue, "4.1(e)", mk,
                                 "First Registration", tot, url))
    return rows_out, warnings


CONDITION_WORDS = ("brand new", "others", "sub-total")


def normalize_entity_measure(ent: str, meas: str):
    """Unify two equivalent ways the digest encodes a first-registration split.

    2004-2019 files put the sub-columns in the MEASURE:
        entity 'Private Cars'           measure 'First Registration > Sub-total'
    2020s files put them in the ENTITY:
        entity 'Private Cars > Sub-total'  measure 'First Registration'
    Both describe the same number, so they are folded onto the second form;
    otherwise the same month would appear as two different series and the
    vintage analysis would be wrong.
    """
    pre = "First Registration > "
    if meas.startswith(pre):
        tail = meas[len(pre):]
        if tail.lower() in CONDITION_WORDS:
            if ent.split(" > ")[-1].lower() == tail.lower():
                return ent, "First Registration"     # 'Sub-total > Sub-total'
            return f"{ent} > {tail}", "First Registration"
    return ent, meas


def normalize_rows(rows):
    out = []
    for date, gran, issue, table, ent, meas, val, url in rows:
        e, m = normalize_entity_measure(ent, meas)
        out.append((date, gran, issue, table, e, m, val, url))
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def collect(args):
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    f = Fetcher(cache, sleep=args.sleep, retries=args.retries, refresh=args.refresh)

    log("=" * 78)
    log("TD Monthly Traffic and Transport Digest - collector")
    log(f"started {dt.datetime.now().isoformat(timespec='seconds')}")
    log("=" * 78)

    issues = enumerate_issues(f)
    log(f"[enum] TOTAL monthly issues discovered: {len(issues)} "
        f"({issues[0][0]} .. {issues[-1][0]})")
    (out_dir / "issues_found.txt").write_text(
        "\n".join(f"{i}\t{u}" for i, u in issues), "utf-8")

    if args.listing_only:
        log("[enum] --listing-only: stopping here")
        return

    if args.min_issue:
        before = len(issues)
        issues = [p for p in issues if p[0] >= args.min_issue]
        log(f"[enum] --min-issue {args.min_issue}: skipped {before - len(issues)} "
            f"older issue(s); collecting {len(issues)}"
            + (f" ({issues[0][0]} .. {issues[-1][0]})" if issues else ""))
    if args.limit_issues:
        issues = issues[-args.limit_issues:]
        log(f"[enum] limited to last {len(issues)} issue(s): "
            f"{issues[0][0]} .. {issues[-1][0]}")
    if args.issues:
        want = {s.strip() for s in args.issues.split(",") if s.strip()}
        issues = [p for p in issues if p[0] in want]
        log(f"[enum] --issues selected {len(issues)}: "
            f"{', '.join(p[0] for p in issues)}")

    long_rows = []
    per_issue = []
    asset_index = []

    for n, (issue, page_url) in enumerate(issues, 1):
        t0 = time.time()
        try:
            st, body = f.fetch(page_url)
        except Blocked as e:
            log(f"[FATAL] {e}")
            break
        except Exception as e:
            log(f"[{n}/{len(issues)}] {issue} page FAILED {e!r}")
            per_issue.append((issue, "page_error", str(e)))
            continue

        if st != 200 or not body:
            log(f"[{n}/{len(issues)}] {issue} page HTTP {st} - skipped")
            per_issue.append((issue, "page_404", ""))
            continue

        html = body.decode("utf-8", "replace")
        assets = find_assets(html, page_url)
        asset_index.append((issue, page_url, json.dumps(assets)))

        got = {}
        for tid, aurl in assets.items():
            try:
                ast, abody = f.fetch(aurl)
            except Blocked as e:
                log(f"[FATAL] {e}")
                return finish(out_dir, long_rows, per_issue, asset_index, f, args)
            except Exception as e:
                log(f"    {issue} {tid} download FAILED {e!r}")
                continue
            if ast != 200 or not abody:
                continue
            rows, warns = [], []
            try:
                if aurl.lower().endswith(".pdf"):
                    tmp = f._paths(aurl)
                    if tid == "4.1(e)":
                        rows, warns = pdf_extract_rows_41e(tmp, aurl, issue)
                    else:
                        rows, warns = pdf_extract_rows(tmp, tid, aurl, issue)
                elif tid == "4.1(a)":
                    rows, warns = parse_xls_41a(abody, aurl, issue)
                elif tid == "4.1(c)":
                    rows, warns = parse_xls_41c(abody, aurl, issue)
                elif tid == "4.1(e)":
                    rows, warns = parse_xls_41e(abody, aurl, issue)
            except Exception as e:
                warns = [f"{tid}: parse exception {e!r}"]
                rows = []
            got[tid] = len(rows)
            long_rows.extend(rows)
            for w in warns[:3]:
                log(f"    WARN {issue} {tid}: {w}")

        missing = [t for t in TARGET_TABLES if t not in assets]
        log(f"[{n}/{len(issues)}] {issue}  "
            f"{' '.join(f'{k}={v}' for k, v in sorted(got.items())) or 'no tables'}"
            f"  missing={','.join(missing) or '-'}"
            f"  cache={f.stats['cache_hits']} net={f.stats['net_200']}"
            f"  {time.time() - t0:.1f}s")
        per_issue.append((issue, ",".join(sorted(got)), json.dumps(missing)))

    return finish(out_dir, long_rows, per_issue, asset_index, f, args)


def finish(out_dir, long_rows, per_issue, asset_index, f, args):
    long_rows = normalize_rows(long_rows)
    log("")
    log(f"[write] long rows: {len(long_rows)}")
    long_path = out_dir / "td_first_registration_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(LONG_COLS)
        for r in long_rows:
            date, gran, issue, table, ent, meas, val, url = r
            if isinstance(val, float) and abs(val - round(val)) < 1e-9:
                val = int(round(val))       # all counts are integers
            w.writerow([date, gran, issue, table, ent, meas, val, url,
                        f.fetched_at(url)])
    log(f"[write] {long_path}")

    with (out_dir / "issues_parsed.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["issue", "tables_with_rows", "missing_tables"])
        w.writerows(per_issue)
    with (out_dir / "asset_index.tsv").open("w", encoding="utf-8") as fh:
        for issue, page, js in asset_index:
            fh.write(f"{issue}\t{page}\t{js}\n")
    with (out_dir / "http_log.txt").open("w", encoding="utf-8") as fh:
        for u, s in f.http_log:
            fh.write(f"{s}\t{u}\n")

    log(f"[stats] cache_hits={f.stats['cache_hits']} net_200={f.stats['net_200']} "
        f"net_404={f.stats['net_404']} errors={f.stats['errors']}")
    return long_rows


# --------------------------------------------------------------------------
# merge / vintage analysis
# --------------------------------------------------------------------------

def analyze(out_dir: Path):
    src = out_dir / "td_first_registration_long.csv"
    if not src.exists():
        log(f"[analyze] {src} not found")
        return
    with src.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    log(f"[analyze] read {len(rows)} raw occurrence rows")

    key = lambda r: (r["date"], r["table"], r["vehicle_class_or_make"],
                     r["measure"], r["granularity"])
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    latest = []
    revisions = []
    for k, g in groups.items():
        g.sort(key=lambda r: r["issue"])
        first, last = g[0], g[-1]
        latest.append(last)
        if len(g) > 1:
            vals = {int(float(r["value"])) for r in g}
            revisions.append({
                "date": k[0], "table": k[1], "entity": k[2], "measure": k[3],
                "n_vintages": len(g),
                "first_issue": first["issue"], "last_issue": last["issue"],
                "first_value": int(float(first["value"])),
                "last_value": int(float(last["value"])),
                "n_distinct_values": len(vals),
                "changed": int(len(vals) > 1),
                "abs_change": abs(int(float(last["value"])) - int(float(first["value"]))),
            })

    latest.sort(key=lambda r: (r["date"], r["table"], r["vehicle_class_or_make"],
                               r["measure"]))
    lat_path = out_dir / "td_first_registration_latest.csv"
    with lat_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LONG_COLS)
        w.writeheader()
        for r in latest:
            w.writerow(r)
    log(f"[analyze] {lat_path}  ({len(latest)} de-duplicated series points)")

    rev_path = out_dir / "revision_stats.csv"
    with rev_path.open("w", newline="", encoding="utf-8") as fh:
        if revisions:
            w = csv.DictWriter(fh, fieldnames=list(revisions[0].keys()))
            w.writeheader()
            w.writerows(revisions)
    log(f"[analyze] {rev_path}  ({len(revisions)} series seen in >1 vintage)")

    n_changed = sum(r["changed"] for r in revisions)
    log("")
    log("-" * 78)
    log("REVISION SUMMARY")
    log(f"  distinct series points                : {len(groups)}")
    log(f"  series appearing in >1 vintage        : {len(revisions)}")
    log(f"  ...of those, value changed at least once: {n_changed} "
        f"({100.0 * n_changed / len(revisions):.1f}%)" if revisions else "  n/a")
    if revisions:
        import statistics
        ch = [r for r in revisions if r["changed"]]
        if ch:
            log(f"  mean |first-last| change (changed only) : "
                f"{statistics.mean(r['abs_change'] for r in ch):.1f}")
            log(f"  max  |first-last| change                : "
                f"{max(r['abs_change'] for r in ch)}")
            ch.sort(key=lambda r: -r["abs_change"])
            for r in ch[:8]:
                log(f"    {r['date']:9s} {r['table']:7s} {r['entity'][:34]:34s} "
                    f"{r['measure'][:20]:20s} {r['first_value']:>8} -> "
                    f"{r['last_value']:>8} ({r['n_vintages']} vintages)")

    # coverage
    monthly = sorted({r["date"] for r in latest if r["granularity"] == "monthly"})
    annual = sorted({r["date"] for r in latest if r["granularity"] == "annual"})
    log("")
    log("COVERAGE (latest-vintage view)")
    log(f"  monthly points: {len(monthly)}  range {monthly[0] if monthly else '-'}"
        f" .. {monthly[-1] if monthly else '-'}")
    log(f"  annual  points: {len(annual)}  range {annual[0] if annual else '-'}"
        f" .. {annual[-1] if annual else '-'}")
    for t in TARGET_TABLES:
        sub = [r for r in latest if r["table"] == t]
        if not sub:
            log(f"  {t}: NO DATA")
            continue
        m = sorted({r["date"] for r in sub if r["granularity"] == "monthly"})
        a = sorted({r["date"] for r in sub if r["granularity"] == "annual"})
        log(f"  {t}: {len(sub):7d} points | monthly {len(m):4d} "
            f"({m[0] if m else '-'}..{m[-1] if m else '-'}) | annual {len(a):3d} "
            f"({a[0] if a else '-'}..{a[-1] if a else '-'}) | "
            f"{len({r['vehicle_class_or_make'] for r in sub})} entities")

    # private-car headline series
    pc = [r for r in latest
          if r["table"] == "4.1(c)" and r["measure"] == "Gross First Registration"
          and r["granularity"] == "monthly"]
    if pc:
        log("")
        log("  Private-car gross first registration (latest vintage), last 12 months:")
        for r in sorted(pc, key=lambda r: r["date"])[-12:]:
            log(f"    {r['date']}  {r['value']:>8}   [{r['issue']} vintage]")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--outdir", default=str(here / "out"))
    ap.add_argument("--cache", default=str(here / "cache"))
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--refresh", action="store_true", help="ignore cache")
    ap.add_argument("--limit-issues", type=int, default=0)
    ap.add_argument("--issues", default="", help="comma-separated issue ids e.g. 2017-01,2004-01")
    ap.add_argument("--min-issue", default="2004-01",
                    help="ignore issues older than this. The 2003 issues use a "
                         "single combined table41.pdf whose header text encodes "
                         "differently, so their class labels are not reliable; "
                         "they are outside the project's 2004-01 target anyway.")
    ap.add_argument("--listing-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    if args.analyze_only:
        analyze(Path(args.outdir))
        return
    collect(args)
    analyze(Path(args.outdir))


if __name__ == "__main__":
    main()
