"""
Stage 0 — Step 2: feature engineering.

Input : data/panel_wide.csv  (263 periods x 23 series, from build_panel.py)
Output: data/features_long.csv   one row per (period, series)
        data/scaling.csv         per-series train mu/sigma used for the z-score
        data/lny.csv             Lunar New Year dates actually used

Feature groups (see 01_ARCHITECTURE.md 决策 3a / 决策 4):
  TARGET (transformed)   y_log = log(value)         ; y_z = (y_log - mu_s) / sigma_s
                         mu_s / sigma_s are computed on TRAIN ONLY
  LAGS (on y_log)        lag1, lag2, lag3, lag12, lag13
  ROLLING (on y_log)     mean/std over 3, 6, 12 months, ending at t (inclusive)
  MONTH                  month_1 .. month_12 one-hot
  LUNAR NEW YEAR         lny_pre14  = |[LNY-14, LNY-1] ∩ month| / 14
                         lny_post7  = |[LNY,    LNY+6] ∩ month| /  7
  REGIME                 covid_border_closed  (2020-02 .. 2023-01 = 1)
                         consumption_voucher  (verified issue months only)
  IDENTITY               series_id, level_id (0=top / 1=child), parent_id

No macro exogenous variables in stage 0 (those arrive in stage 1).
Everything here is derived from the target itself + the calendar.
"""
from __future__ import annotations

import calendar
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TRAIN_END = "202312"

# ---- Lunar New Year, day 1 of the lunar year -------------------------------
# Cross-checked 2004-2026 against the `lunardate` package: 0 mismatches.
LNY = {
    2004: (1, 22), 2005: (2, 9), 2006: (1, 29), 2007: (2, 18), 2008: (2, 7),
    2009: (1, 26), 2010: (2, 14), 2011: (2, 3), 2012: (1, 23), 2013: (2, 10),
    2014: (1, 31), 2015: (2, 19), 2016: (2, 8), 2017: (1, 28), 2018: (2, 16),
    2019: (2, 5), 2020: (1, 25), 2021: (2, 12), 2022: (2, 1), 2023: (1, 22),
    2024: (2, 10), 2025: (1, 29), 2026: (2, 17),
}

# ---- Consumption voucher issue months --------------------------------------
# Only months verified against primary sources are set to 1.
#   2021-08 : first tranche, 2021-08-01        (gov press release)
#   2022-04 : first stage,   2022-04-07        (gov press release 2022-03-07)
#   2022-08 : second stage,  2022-08-07        (wikipedia + press release)
# 2021-10, 2023-04, 2023-07 are NOT verified -> deliberately excluded.
VOUCHER_MONTHS = {"202108", "202204", "202208"}

# ---- COVID border regime ---------------------------------------------------
# Land ports suspended Feb 2020; partial reopening 2023-01-08;
# full reopening 2023-02-06.  So closed = 2020-02 .. 2023-01.
COVID_FROM, COVID_TO = "202002", "202301"

HIERARCHY = {
    "2": ["3", "4", "5", "6", "7"],
    "12": ["13", "14"],
    "18": ["19", "23", "51"],
    "35": ["36", "37", "38", "39", "40"],
}
TOP_LEVEL = ["2", "8", "9", "12", "18", "30", "32", "35"]
PARENT_OF = {kid: par for par, kids in HIERARCHY.items() for kid in kids}

LAGS = (1, 2, 3, 12, 13)
ROLL = (3, 6, 12)


def month_bounds(period: str) -> tuple[date, date]:
    y, m = int(period[:4]), int(period[4:])
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def overlap_days(window_start: date, window_end: date, period: str) -> int:
    """number of days the [start, end] window shares with the given month"""
    m_start, m_end = month_bounds(period)
    lo = max(window_start, m_start)
    hi = min(window_end, m_end)
    return (hi - lo).days + 1 if hi >= lo else 0


def lny_features(period: str) -> tuple[float, float]:
    """(pre14 share, post7 share) for the month containing/before LNY"""
    y = int(period[:4])
    m = int(period[4:])
    # the LNY that can affect Jan/Feb of year y is LNY(y)
    if y not in LNY:
        return 0.0, 0.0
    lm, ld = LNY[y]
    if m not in (1, 2, lm):
        return 0.0, 0.0
    lny = date(y, lm, ld)
    pre_share = overlap_days(lny - timedelta(days=14), lny - timedelta(days=1), period) / 14.0
    post_share = overlap_days(lny, lny + timedelta(days=6), period) / 7.0
    return round(pre_share, 6), round(post_share, 6)


def main() -> int:
    print("=" * 78)
    print("STAGE 0 / STEP 2 — features")
    print("=" * 78)

    wide = pd.read_csv(DATA / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    print(f"  loaded panel: {wide.shape[0]} periods x {wide.shape[1]} series")

    periods = list(wide.index)
    series = list(wide.columns)
    train_periods = [p for p in periods if p <= TRAIN_END]
    print(f"  train periods: {len(train_periods)}  ({train_periods[0]} .. {train_periods[-1]})")

    # ---- per-series scaling, TRAIN ONLY ---------------------------------
    logs = np.log(wide.astype(float))
    mu = logs.loc[train_periods].mean()
    sd = logs.loc[train_periods].std(ddof=0)
    scaling = pd.DataFrame({"series": mu.index, "mu_log": mu.values, "sigma_log": sd.values})
    scaling["level_id"] = scaling["series"].map(lambda s: 0 if s in TOP_LEVEL else 1)
    scaling["parent_id"] = scaling["series"].map(lambda s: PARENT_OF.get(s, "none"))
    scaling.to_csv(DATA / "scaling.csv", index=False, encoding="utf-8")
    print(f"  scaling table: {len(scaling)} series  "
          f"(sigma range {sd.min():.4f} .. {sd.max():.4f})")

    y_z = (logs - mu) / sd
    y_z = y_z.dropna(how="all")

    # ---- build the long feature table -----------------------------------
    rows = []
    for sid, s in enumerate(series):
        s_log = logs[s]
        s_z = y_z[s]
        for p in periods:
            rec = {
                "period": p,
                "series_id": sid,
                "series": s,
                "level_id": 0 if s in TOP_LEVEL else 1,
                "parent_id": PARENT_OF.get(s, "none"),
                "value_hkd_m": float(wide.loc[p, s]),
                "y_log": float(s_log.loc[p]),
                "y_z": float(s_z.loc[p]) if pd.notna(s_z.loc[p]) else np.nan,
            }
            # lags on y_log
            for L in LAGS:
                j = periods.index(p) - L
                rec[f"lag{L}"] = float(s_log.iloc[j]) if j >= 0 else np.nan
            # rolling on y_log, ending at t inclusive
            i = periods.index(p)
            for w in ROLL:
                lo = i - w + 1
                if lo >= 0:
                    seg = s_log.iloc[lo:i + 1]
                    rec[f"roll{w}_mean"] = float(seg.mean())
                    rec[f"roll{w}_std"] = float(seg.std(ddof=0))
                else:
                    rec[f"roll{w}_mean"] = np.nan
                    rec[f"roll{w}_std"] = np.nan
            # month one-hot
            mm = int(p[4:])
            for k in range(1, 13):
                rec[f"month_{k}"] = 1.0 if k == mm else 0.0
            # LNY
            pre, post = lny_features(p)
            rec["lny_pre14"] = pre
            rec["lny_post7"] = post
            # regime
            rec["covid_border_closed"] = 1.0 if COVID_FROM <= p <= COVID_TO else 0.0
            rec["consumption_voucher"] = 1.0 if p in VOUCHER_MONTHS else 0.0
            rows.append(rec)

    feat = pd.DataFrame(rows)
    feat = feat.sort_values(["period", "series_id"]).reset_index(drop=True)
    feat["split"] = np.where(feat["period"] <= TRAIN_END, "train", "post")

    # ---- report ---------------------------------------------------------
    feature_cols = [c for c in feat.columns if c.startswith(("lag", "roll", "month_"))] + [
        "lny_pre14", "lny_post7", "covid_border_closed", "consumption_voucher",
        "level_id", "series_id",
    ]
    print(f"\n  feature columns : {len(feature_cols)}")
    print(f"  rows            : {len(feat):,}  ({feat['split'].value_counts().to_dict()})")

    # warm-up rows will contain NaN lags -> report how many usable
    usable = feat.dropna(subset=[f"lag{L}" for L in LAGS] + [f"roll{w}_mean" for w in ROLL])
    usable_train = usable[usable["split"] == "train"]
    print(f"  rows with complete lags/rolling : {len(usable):,}")
    print(f"     of which train              : {len(usable_train):,}")

    # known leading NaNs: first 13 months have no lag13
    print(f"  NaN lag13 rows                  : {int(feat['lag13'].isna().sum()):,} "
          f"(first 13 months x 23 series = 299 expected)")

    # ---- LNY spot check: the 2024 vs 2023 anomaly ------------------------
    print("\n  --- LNY shares, Jan/Feb of the key years ---")
    for y in (2023, 2024):
        for m in (1, 2):
            p = f"{y}{m:02d}"
            r = feat[(feat["period"] == p)].iloc[0]
            print(f"    {p}: lny_pre14={r['lny_pre14']:.3f}  lny_post7={r['lny_post7']:.3f}")

    print("\n  --- regime dummies ---")
    for name in ("covid_border_closed", "consumption_voucher"):
        on = sorted(feat.loc[feat[name] == 1, "period"].unique())
        print(f"    {name:22s}: {len(on):3d} months  {on[:3]}{' ...' if len(on) > 3 else ''}{on[-2:] if len(on) > 3 else ''}")

    # ---- write ----------------------------------------------------------
    feat.to_csv(DATA / "features_long.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [{"year": y, "month": m, "day": d, "iso": date(y, m, d).isoformat()} for y, (m, d) in sorted(LNY.items())]
    ).to_csv(DATA / "lny.csv", index=False, encoding="utf-8")

    print("\n  --- wrote ---")
    for f in ("features_long.csv", "scaling.csv", "lny.csv"):
        p = DATA / f
        print(f"    {f:22s} {p.stat().st_size:>10,} B")
    print("\nSTEP 2 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
