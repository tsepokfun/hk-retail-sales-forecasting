"""
Stage 0 — exogenous data fetch / cache layer.

Downloads the raw payloads used by build_exog.py into data/exog_raw/ with a
date suffix.  An existing dated snapshot is NEVER overwritten: if any snapshot
for the same source already exists, the newest one is reused so that re-runs do
not hit the servers again.

Sources
  censtatd JSON API : https://www.censtatd.gov.hk/api/get.php?id=<TABLE>&lang=en&full_series=1
  HKO monthly extract: https://www.hko.gov.hk/cis/monthlyExtract/monthlyExtract.xml
                       (JSON body served with an XML-ish content type)
  MTR patronage      : local file, already downloaded, copied into the snapshot
                       dir for provenance (read-only use)

Usage (fetch everything that is not cached yet):
    .venv\\Scripts\\python.exe data\\exog_fetch.py
"""
from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "exog_raw"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TIMEOUT = 180
RETRIES = 3
BACKOFF = (2, 4, 8)          # seconds, applied between retries
POLITE_SLEEP = 2.0           # seconds between censtatd calls

CENSTATD_TABLES = ("510-60001A", "340-45021", "340-46002", "650-80001", "410-55111A")
CENSTATD_URL = "https://www.censtatd.gov.hk/api/get.php?id={table}&lang=en&full_series=1"
HKO_URL = "https://www.hko.gov.hk/cis/monthlyExtract/monthlyExtract.xml"
MTR_SRC = ROOT.parent / "_mtr_patronage.csv"


def today() -> str:
    return date.today().strftime("%Y%m%d")


def _newest(prefix: str, suffix: str) -> Path | None:
    """newest existing dated snapshot for a source, or None"""
    hits = sorted(RAW.glob(f"{prefix}_*{suffix}"))
    return hits[-1] if hits else None


def fetch_bytes(url: str, label: str) -> tuple[bytes, int]:
    """GET with RETRIES attempts and exponential backoff; raises on failure."""
    last_exc: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                status, body = r.status, r.read()
            if status != 200:
                raise RuntimeError(f"HTTP {status}")
            print(f"  GET {url}\n    HTTP {status} | {len(body):,} bytes  ({label})")
            return body, status
        except Exception as exc:  # noqa: BLE001 - retry everything transport-level
            last_exc = exc
            code = getattr(exc, "code", None)
            print(f"  !! attempt {attempt}/{RETRIES} failed for {label}: "
                  f"{type(exc).__name__}{f' [HTTP {code}]' if code else ''}: {exc}")
            if attempt < RETRIES:
                time.sleep(BACKOFF[attempt - 1])
    raise RuntimeError(f"{label}: gave up after {RETRIES} attempts ({last_exc})") from last_exc


def get_censtatd(table: str, force: bool = False) -> Path:
    """cached snapshot path for one censtatd table (downloads if absent)"""
    RAW.mkdir(parents=True, exist_ok=True)
    existing = None if force else _newest(f"censtatd_{table}", ".json")
    if existing is not None:
        print(f"  cache hit: {existing.name} ({existing.stat().st_size:,} B)")
        return existing
    body, _ = fetch_bytes(CENSTATD_URL.format(table=table), f"censtatd {table}")
    out = RAW / f"censtatd_{table}_{today()}.json"
    n = 1
    while out.exists():                      # never overwrite a dated snapshot
        out = RAW / f"censtatd_{table}_{today()}_v{n}.json"
        n += 1
    out.write_bytes(body)
    print(f"    saved -> {out.relative_to(ROOT)}")
    time.sleep(POLITE_SLEEP)
    return out


def get_hko(force: bool = False) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    existing = None if force else _newest("hko_monthlyExtract", ".xml")
    if existing is not None:
        print(f"  cache hit: {existing.name} ({existing.stat().st_size:,} B)")
        return existing
    body, _ = fetch_bytes(HKO_URL, "HKO monthlyExtract")
    out = RAW / f"hko_monthlyExtract_{today()}.xml"
    n = 1
    while out.exists():
        out = RAW / f"hko_monthlyExtract_{today()}_v{n}.xml"
        n += 1
    out.write_bytes(body)
    print(f"    saved -> {out.relative_to(ROOT)}")
    return out


def get_mtr() -> Path:
    """the MTR file is already downloaded; copy it in for provenance"""
    RAW.mkdir(parents=True, exist_ok=True)
    existing = _newest("mtr_patronage", ".csv")
    if existing is not None and not (MTR_SRC.exists() and
                                     existing.stat().st_size != MTR_SRC.stat().st_size):
        print(f"  cache hit: {existing.name} ({existing.stat().st_size:,} B)")
        return existing
    out = RAW / f"mtr_patronage_{today()}.csv"
    n = 1
    while out.exists():
        out = RAW / f"mtr_patronage_{today()}_v{n}.csv"
        n += 1
    shutil.copyfile(MTR_SRC, out)
    print(f"  copied {MTR_SRC} -> {out.relative_to(ROOT)} ({out.stat().st_size:,} B)")
    return out


def fetch_all(force: bool = False) -> dict[str, Path]:
    print("=" * 78)
    print("EXOG / STEP A — fetch + cache raw payloads")
    print("=" * 78)
    out = {}
    for t in CENSTATD_TABLES:
        print(f"\n[{t}]")
        out[f"censtatd_{t}"] = get_censtatd(t, force)
    print("\n[hko]")
    out["hko"] = get_hko(force)
    print("\n[mtr]")
    out["mtr"] = get_mtr()
    return out


if __name__ == "__main__":
    import sys

    paths = fetch_all(force="--force" in sys.argv)
    print("\n--- snapshot files ---")
    for k, p in paths.items():
        print(f"  {k:22s} {p}")
