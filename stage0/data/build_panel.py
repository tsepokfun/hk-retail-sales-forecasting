"""
Stage 0 — Step 1: build the retail panel.

Source : censtatd table 620-67002  (retail sales value by type of retail outlet)
Endpoint: https://www.censtatd.gov.hk/api/get.php?id=620-67002&lang=en&full_series=1

Rules (verified against the live API, see 02_DATASET.md):
  - keep only sv == "VAL_RS"   (HK$ million, monthly sale value)
  - keep only freq == "M"      (payload mixes M and Y)
  - period must be a 6-digit YYYYMM string
  - drop OUTLET_TYPE == ""  (that row is the TOTAL aggregate -> leakage)
  - drop OUTLET_TYPE == 49  (49 is a subset of 8 and 30 -> duplicate counting)
  - figure may be "***" or "" -> coerce to NaN, then assert zero missing

Outputs
  raw/censtatd_620-67002_<download-date>.json   (immutable snapshot)
  data/panel_wide.csv                            period x 23 outlet types
  data/panel_long.csv                            tidy long form with split label
  data/panel_meta.json                           shape / split / series metadata
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
DATA = ROOT / "data"

TABLE = "620-67002"
URL = f"https://www.censtatd.gov.hk/api/get.php?id={TABLE}&lang=en&full_series=1"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

TRAIN_END = "202312"
TEST_START, TEST_END = "202401", "202412"

DROP_KEYS = {"", "TOTAL", "49"}

# hierarchy: parent -> children (verified numerically, sum == 271)
HIERARCHY = {
    "2": ["3", "4", "5", "6", "7"],
    "12": ["13", "14"],
    "18": ["19", "23", "51"],
    "35": ["36", "37", "38", "39", "40"],
}
TOP_LEVEL = ["2", "8", "9", "12", "18", "30", "32", "35"]


def fetch_snapshot() -> tuple[dict, Path]:
    """Download the API payload once and keep a dated snapshot."""
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"censtatd_{TABLE}_{date.today():%Y%m%d}.json"
    if out.exists():
        print(f"  snapshot already present: {out.name} ({out.stat().st_size:,} B)")
        return json.loads(out.read_text(encoding="utf-8")), out

    print(f"  GET {URL}")
    req = urllib.request.Request(URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=180) as r:
        status = r.status
        body = r.read()
    print(f"  HTTP {status} | {len(body):,} bytes")
    out.write_bytes(body)
    print(f"  saved snapshot -> {out.relative_to(ROOT)}")
    return json.loads(body.decode("utf-8")), out


def to_float(v):
    """figures may be '***' (suppressed) or '' -> NaN."""
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s in {"", "***", "N.A.", "-", "nan"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def clean_label(s: str) -> str:
    for tag in ("<strong>", "</strong>", "<i>", "</i>", '<font color="blue">', "</font>"):
        s = s.replace(tag, "")
    return " ".join(s.split())


def main() -> int:
    print("=" * 78)
    print("STAGE 0 / STEP 1 — build retail panel")
    print("=" * 78)

    payload, snap = fetch_snapshot()
    hdr = payload.get("header", {})
    print(f"  title   : {hdr.get('title')}")
    print(f"  source  : {hdr.get('source')}")
    cols = (hdr.get("count") or {}).get("noOfRecords")
    print(f"  noOfRecords: {cols}")

    ds = payload["dataSet"]
    print(f"  raw dataSet rows: {len(ds):,}")

    # ---- filter to monthly sale value --------------------------------
    n_before = len(ds)
    recs = []
    names: dict[str, str] = {}
    for r in ds:
        if r.get("sv") != "VAL_RS":
            continue
        if r.get("freq") != "M":
            continue
        p = str(r.get("period", ""))
        if not (p.isdigit() and len(p) == 6):
            continue
        key = "" if r.get("OUTLET_TYPE") in (None, "") else str(r.get("OUTLET_TYPE")).strip()
        if key in DROP_KEYS:
            continue
        names.setdefault(key, clean_label(r.get("OUTLET_TYPEDesc") or ""))
        recs.append((p, key, to_float(r.get("figure"))))

    print(f"  VAL_RS + freq=M rows     : {len(recs):,}  (from {n_before:,})")
    print(f"  distinct outlet types    : {len(names)}")

    wide = (
        pd.DataFrame(recs, columns=["period", "outlet_type", "value_hkd_m"])
        .pivot_table(index="period", columns="outlet_type", values="value_hkd_m", aggfunc="first")
        .sort_index()
    )
    wide.index.name = "period"
    wide = wide[sorted(wide.columns, key=lambda c: (len(c), c))]

    # ---- sanity checks -----------------------------------------------
    print("\n  --- sanity ---")
    print(f"  shape                 : {wide.shape[0]} periods x {wide.shape[1]} series")
    print(f"  period range          : {wide.index[0]} .. {wide.index[-1]}")
    n_missing = int(wide.isna().sum().sum())
    print(f"  missing cells         : {n_missing}")
    if n_missing > 0:
        bad = wide.isna().sum()
        print("  MISSING per series:")
        for k, v in bad[bad > 0].items():
            print(f"      {k:4s} {names.get(k,'')[:48]:50s} {v}")
        print("  -> ABORT: panel must be complete for stage 0")
        return 2

    # verify the accounting identities on the latest complete month
    last = wide.index[-1]
    print(f"\n  --- hierarchy identity check at {last} ---")
    total_check = sum(wide.loc[last, k] for k in TOP_LEVEL)
    print(f"  sum(8 top-level)      = {total_check:,.0f}")
    for parent, kids in HIERARCHY.items():
        s_kids = sum(wide.loc[last, k] for k in kids)
        s_par = wide.loc[last, parent]
        ok = abs(s_kids - s_par) <= max(1.0, 0.001 * abs(s_par))
        print(f"  {parent:>4s} = {'+'.join(kids):16s}: kids={s_kids:12,.0f}  parent={s_par:12,.0f}  {'OK' if ok else 'MISMATCH'}")

    # ---- split --------------------------------------------------------
    train_idx = [p for p in wide.index if p <= TRAIN_END]
    test_idx = [p for p in wide.index if TEST_START <= p <= TEST_END]
    print("\n  --- split ---")
    print(f"  train  {len(train_idx):3d} months  ({train_idx[0]} .. {train_idx[-1]})")
    print(f"  test   {len(test_idx):3d} months  ({test_idx[0]} .. {test_idx[-1]})")
    print(f"  train cells {len(train_idx) * wide.shape[1]:,}   test cells {len(test_idx) * wide.shape[1]:,}")

    # ---- write ---------------------------------------------------------
    DATA.mkdir(parents=True, exist_ok=True)
    wide.to_csv(DATA / "panel_wide.csv", encoding="utf-8")

    split_map = {p: ("train" if p <= TRAIN_END else "test" if TEST_START <= p <= TEST_END else "post") for p in wide.index}
    long = (
        wide.reset_index()
        .melt(id_vars="period", var_name="outlet_type", value_name="value_hkd_m")
        .assign(
            outlet_name=lambda d: d["outlet_type"].map(names),
            split=lambda d: d["period"].map(split_map),
            level=lambda d: d["outlet_type"].map(lambda k: 0 if k in TOP_LEVEL else 1),
            parent=lambda d: d["outlet_type"].map(
                {kid: par for par, kids in HIERARCHY.items() for kid in kids}
            ),
        )
    )
    long = long.sort_values(["period", "outlet_type"]).reset_index(drop=True)
    long.to_csv(DATA / "panel_long.csv", index=False, encoding="utf-8")

    meta = {
        "source_table": TABLE,
        "source_url": URL,
        "snapshot": str(snap.name),
        "downloaded": date.today().isoformat(),
        "n_periods": int(wide.shape[0]),
        "n_series": int(wide.shape[1]),
        "period_first": wide.index[0],
        "period_last": wide.index[-1],
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "n_train_months": len(train_idx),
        "n_test_months": len(test_idx),
        "missing_cells": n_missing,
        "series": {k: {"name": names[k], "level": 0 if k in TOP_LEVEL else 1,
                       "parent": {kid: par for par, kids in HIERARCHY.items() for kid in kids}.get(k)}
                   for k in wide.columns},
    }
    (DATA / "panel_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n  --- wrote ---")
    for f in ("panel_wide.csv", "panel_long.csv", "panel_meta.json"):
        p = DATA / f
        print(f"    {f:20s} {p.stat().st_size:>9,} B")
    print("\nSTEP 1 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
