"""
Stage 0 — Step 4: exogenous monthly feature block for the retail model.

Inputs (raw snapshots in data/exog_raw/, fetched by exog_fetch.py)
  censtatd 510-60001A  Consumer Price Index (composite, all-items)
  censtatd 340-45021   HKD interest rates (best lending rate)
  censtatd 340-46002   Effective exchange rate indices for the HKD
  censtatd 650-80001   Visitor arrivals by nationality/region (Total)
  censtatd 410-55111A  Seaborne cargo throughput (Total)
  HKO monthlyExtract   monthly mean temperature + total rainfall
  MTR patronage CSV    Domestic Service monthly total (unit: thousands of
                       passenger trips, converted to a pure number)

Outputs
  data/exog_wide.csv   period x 8 transformed features, 263 rows (200410..202608)
  data/exog_meta.json  per-feature provenance / selector / transform / NaN audit
  data/exog_groups.json  the two downstream feature groups
  data/EXOG.md         human-readable report

Rules baked in
  * censtatd: keep freq == "M" only, period must be a 6-digit YYYYMM string,
    "***" / "" / "-" -> NaN, never interpolate.
  * every YoY feature = log-ish level difference y(t) - y(t-12), THEN shift(1)
    (publication lag: month t gets the value published for t-1).
  * weather anomalies are NOT shifted (month-end observed).
  * all scalers (rate z-score, calendar-month weather means, anomaly z-score)
    are fitted on periods <= 202312 ONLY.
  * non-positive levels are masked to NaN before taking logs so that no inf can
    ever appear.

Usage:
    .venv\\Scripts\\python.exe data\\build_exog.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from exog_fetch import (CENSTATD_URL, HKO_URL, RAW, ROOT, get_censtatd,
                        get_hko, get_mtr)

DATA = ROOT / "data"
TRAIN_END = "202312"
PANEL_FIRST, PANEL_LAST = "200410", "202608"

FEATURES = [
    "cpi_yoy", "rate_z", "eeri_yoy", "visitor_yoy",
    "mtr_yoy", "cargo_yoy", "rain_anom_z", "temp_anom_z",
]
SRC_OF = {
    "cpi_yoy": "cpi", "rate_z": "rate", "eeri_yoy": "eeri",
    "visitor_yoy": "visitor", "mtr_yoy": "mtr", "cargo_yoy": "cargo",
    "rain_anom_z": "weather", "temp_anom_z": "weather",
}
GROUP_A_MACRO_PRICE_FINANCE = ["cpi_yoy", "rate_z", "eeri_yoy"]
GROUP_B_TRAFFIC_WEATHER_LOGISTICS = [
    "visitor_yoy", "mtr_yoy", "cargo_yoy", "rain_anom_z", "temp_anom_z",
]

TRACE_MM = 0.05          # HKO convention for "Trace" rainfall
DDDOF = 0                # matches data/make_features.py (std(ddof=0))

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def to_float(v) -> float:
    """censtatd figure -> float.  '***' (suppressed), '' and '-' become NaN."""
    if v is None:
        return float("nan")
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s in {"", "***", "-", "N.A.", "nan", "NaN", "NA"}:
        return float("nan")
    s = s.rstrip("#").strip()          # HKO appends '#' (provisional)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def as_periods(periods: list[str]) -> pd.PeriodIndex:
    return pd.PeriodIndex(pd.to_datetime(periods, format="%Y%m"), freq="M")


def monthly_grid(s: pd.Series) -> pd.Series:
    """reindex a YYYYMM-keyed series onto a gap-free monthly grid"""
    s = s.copy()
    s.index = as_periods([str(i) for i in s.index])
    s = s.sort_index()
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    return s.reindex(full)


def to_panel(s: pd.Series, periods: list[str]) -> pd.Series:
    out = s.reindex(as_periods(periods)).astype(float)
    out.index = periods
    return out


def safe_log(s: pd.Series, label: str, log1p: bool = False) -> pd.Series:
    """log/log1p that masks non-positive levels instead of producing -inf"""
    bad = int((s <= 0).sum()) if not log1p else int((s < 0).sum())
    if bad:
        print(f"    WARNING {label}: {bad} non-positive level(s) -> masked to NaN before log")
        s = s.where(s > 0) if not log1p else s.where(s >= 0)
    return np.log1p(s) if log1p else np.log(s)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# censtatd parsing / auditing
# ---------------------------------------------------------------------------


def censtatd_monthly(ds: list[dict], selector: dict, label: str) -> pd.Series:
    """one monthly series from the flat censtatd record list"""
    recs = [r for r in ds
            if r.get("freq") == "M" and all(r.get(k) == v for k, v in selector.items())]
    if not recs:
        raise RuntimeError(f"{label}: selector {selector} matched no monthly records")
    periods = [str(r["period"]) for r in recs]
    if len(set(periods)) != len(periods):
        dups = [p for p, n in Counter(periods).items() if n > 1]
        raise RuntimeError(f"{label}: duplicate periods {dups[:5]}")
    if not all(re.fullmatch(r"\d{6}", p) for p in periods):
        raise RuntimeError(f"{label}: non-YYYYMM period encountered")
    s = pd.Series([to_float(r.get("figure")) for r in recs], index=periods, dtype=float)
    s = s.sort_index()
    print(f"    [{label}] {len(s)} monthly obs  {s.index[0]}..{s.index[-1]}  "
          f"NaN={int(s.isna().sum())}")
    return s


def audit_payload(table: str, ds: list[dict], dims: list[str]) -> None:
    print(f"\n  --- {table} audit ---")
    print(f"    records       : {len(ds):,}")
    print(f"    freq values   : {dict(Counter(r.get('freq') for r in ds))}")
    monthly = [r for r in ds if r.get("freq") == "M"]
    per = sorted({str(r.get("period")) for r in monthly})
    print(f"    monthly rows  : {len(monthly):,}  periods {per[0]}..{per[-1]} ({len(per)})")
    sv = Counter((r.get("sv"), r.get("svDesc")) for r in monthly)
    print(f"    (sv, svDesc) monthly combos ({len(sv)}):")
    for (a, b), n in sv.most_common():
        print(f"        {a!r:16s} {b!r:30s} n={n}")
    for d in dims:
        vals = Counter(r.get(d) for r in monthly)
        print(f"    {d:14s} ({len(vals)}): {dict(vals.most_common())}")


# ---------------------------------------------------------------------------
# individual sources
# ---------------------------------------------------------------------------


def build_cpi() -> tuple[pd.Series, dict]:
    path = get_censtatd("510-60001A")
    payload = load_json(path)
    ds = payload["dataSet"]
    print(f"\n[CPI] {payload['header'].get('title')}")
    audit_payload("510-60001A", ds, ["GROUP", "GROUPDesc"])
    sel = {"sv": "CC_CM_1920", "svDesc": "Index", "GROUP": "", "GROUPDesc": "Total"}
    print(f"    -> SELECT {sel}   (composite CPI, all-items Total; the A/B/C "
          f"brackets S1..S9 are excluded)")
    s = censtatd_monthly(ds, sel, "cpi")
    meta = {
        "source": "censtatd 510-60001A",
        "url": CENSTATD_URL.format(table="510-60001A"),
        "snapshot": path.name,
        "title": payload["header"].get("title"),
        "selector": {"freq": "M", **sel},
        "selector_note": "composite CPI index (CC_CM_1920), all-items GROUP '' "
                         "with GROUPDesc 'Total'; index only, not the YoY% column",
        "unit": "index (Oct 2019 - Sep 2020 = 100)",
        "raw_series": s,
    }
    return s, meta


def build_rate() -> tuple[pd.Series, dict]:
    path = get_censtatd("340-45021")
    payload = load_json(path)
    ds = payload["dataSet"]
    print(f"\n[RATE] {payload['header'].get('title')}")
    audit_payload("340-45021", ds, ["DEP_P_T", "DEP_P_TDesc"])
    sel = {"sv": "BL_RATE"}
    print(f"    -> SELECT {sel}   (HSBC best lending rate; T_DEP_RATE with its "
          f"DEP_P_T tenors and S_DEP_RATE are excluded)")
    s = censtatd_monthly(ds, sel, "bl_rate")
    meta = {
        "source": "censtatd 340-45021",
        "url": CENSTATD_URL.format(table="340-45021"),
        "snapshot": path.name,
        "title": payload["header"].get("title"),
        "selector": {"freq": "M", **sel},
        "selector_note": "sv == BL_RATE only; the DEP_P_T tenor dimension belongs "
                         "to T_DEP_RATE and is deliberately not used",
        "unit": "% per annum",
        "raw_series": s,
    }
    return s, meta


def build_eeri() -> tuple[pd.Series, dict]:
    path = get_censtatd("340-46002")
    payload = load_json(path)
    ds = payload["dataSet"]
    print(f"\n[EERI] {payload['header'].get('title')}")
    audit_payload("340-46002", ds, [])
    sel = {"sv": "EERI_T"}
    print("    -> SELECT sv == 'EERI_T'  (EERI_T = trade-weighted total index; "
          "EERI_I = import-weighted, EERI_E = export-weighted are excluded).")
    s = censtatd_monthly(ds, sel, "eeri")
    meta = {
        "source": "censtatd 340-46002",
        "url": CENSTATD_URL.format(table="340-46002"),
        "snapshot": path.name,
        "title": payload["header"].get("title"),
        "selector": {"freq": "M", **sel},
        "selector_note": "trade-weighted effective exchange rate index; verified "
                         "to lie inside [min(EERI_I, EERI_E), max(...)] every month",
        "unit": "index (Jan 2020 = 100)",
        "raw_series": s,
    }
    return s, meta


def build_visitor() -> tuple[pd.Series, dict]:
    path = get_censtatd("650-80001")
    payload = load_json(path)
    ds = payload["dataSet"]
    print(f"\n[VISITOR] {payload['header'].get('title')}")
    audit_payload("650-80001", ds, ["REGION", "REGIONDesc"])
    sel = {"sv": "VIS_ARR", "REGION": "", "REGIONDesc": "Total"}
    print("    -> SELECT sv == 'VIS_ARR' and REGION == '' (all-region Total). "
          "Verified: Total == sum of the 11 named regions for all 272 months.")
    s = censtatd_monthly(ds, sel, "visitor_arrivals")
    meta = {
        "source": "censtatd 650-80001",
        "url": CENSTATD_URL.format(table="650-80001"),
        "snapshot": path.name,
        "title": payload["header"].get("title"),
        "selector": {"freq": "M", **sel},
        "selector_note": "REGION == '' is the all-region Total (identity checked "
                         "against the sum of the 11 named regions, max abs diff 0)",
        "unit": "number of visitor arrivals",
        "raw_series": s,
    }
    return s, meta


def build_cargo() -> tuple[pd.Series, dict]:
    path = get_censtatd("410-55111A")
    payload = load_json(path)
    ds = payload["dataSet"]
    print(f"\n[CARGO] {payload['header'].get('title')}")
    audit_payload("410-55111A", ds,
                  ["TRANS", "DIRECTION", "DIRECTIONDesc", "SHIPMENT_TYPE", "SHIPMENT_TYPEDesc"])
    sel = {"sv": "PORT_CARGO_TP", "svDesc": "('000 tonnes)", "TRANS": "S",
           "DIRECTION": "", "DIRECTIONDesc": "Total",
           "SHIPMENT_TYPE": "", "SHIPMENT_TYPEDesc": "Total"}
    print("    -> SELECT sv == 'PORT_CARGO_TP' + svDesc == \"('000 tonnes)\" (level, "
          "not YoY%) + DIRECTION == '' (Total, not In/Out) + SHIPMENT_TYPE == '' "
          "(Total, not DS/TS).")
    s = censtatd_monthly(ds, sel, "seaborne_cargo")
    meta = {
        "source": "censtatd 410-55111A",
        "url": CENSTATD_URL.format(table="410-55111A"),
        "snapshot": path.name,
        "title": payload["header"].get("title"),
        "selector": {"freq": "M", **sel},
        "selector_note": "seaborne cargo throughput, all-directions all-shipment "
                         "totals; verified Total == Inward + Outward (<=1 unit rounding)",
        "unit": "thousand tonnes",
        "raw_series": s,
    }
    return s, meta


MONTH_RE = re.compile(r"^([A-Z][a-z]+)\s+(\d{4})$")


def build_mtr() -> tuple[pd.Series, dict]:
    path = get_mtr()
    rows: list[tuple[str, float]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            key = row[0].strip()
            m = MONTH_RE.match(key)
            if not m or len(row) < 2:
                continue
            mm = pd.to_datetime(key, format="%B %Y")
            raw = row[1].strip().replace(",", "")
            val = float("nan") if raw in {"-", ""} else float(raw)
            rows.append((mm.strftime("%Y%m"), val))
    if not rows:
        raise RuntimeError("MTR: no monthly rows parsed")
    print(f"\n[MTR] {path.name}")
    print(f"    parsed monthly rows: {len(rows)}  {rows[-1][0]}..{rows[0][0]}")
    s = pd.Series(dict(rows), dtype=float).sort_index()
    s.index = s.index.astype(str)
    print(f"    Domestic Service Monthly Total: {s.index[0]}..{s.index[-1]}  "
          f"NaN={int(s.isna().sum())}  min={s.min():,.0f}  max={s.max():,.0f}")
    meta = {
        "source": "MTR patronage (local file, already downloaded)",
        "url": "n/a - local: _mtr_patronage.csv (copied to the snapshot dir)",
        "snapshot": path.name,
        "title": "Patronage figures (unit in thousands)",
        "selector": {"row": "Month YYYY", "column_index": 1,
                     "column": "Domestic Service Monthly Total"},
        "selector_note": "English month names, newest first; thousands separators "
                         "stripped; '-' -> NaN. Reported in thousands of passenger "
                         "trips; the log difference used downstream is scale-free, "
                         "so no unit conversion is applied.",
        "unit": "thousand passenger trips per month",
        "raw_series": s,
    }
    return s, meta


def build_weather() -> tuple[pd.DataFrame, dict]:
    path = get_hko()
    payload = load_json(path)
    years = payload["stn"]["data"]
    print(f"\n[HKO] {path.name}: {len(years)} year records "
          f"{years[0]['year']}..{years[-1]['year']}")
    rain: dict[str, float] = {}
    temp: dict[str, float] = {}
    n_trace = n_star = 0
    for yrec in years:
        y = int(yrec["year"])
        for row in yrec["yearData"]:
            if len(row) != 14:
                raise RuntimeError(f"HKO {y}: unexpected row length {len(row)}")
            mm = int(row[0])
            p = f"{y}{mm:02d}"
            r_raw = row[10].strip()
            if r_raw == "Trace":
                n_trace += 1
                r = TRACE_MM
            else:
                r = to_float(r_raw)
            t = to_float(row[4])
            if r_raw == "***":
                n_star += 1
            rain[p] = r
            temp[p] = t
    print(f"    rainfall 'Trace' cells mapped to {TRACE_MM} mm: {n_trace}")
    print(f"    '***' in the rainfall column: {n_star}")
    df = pd.DataFrame({"rain_mm": pd.Series(rain, dtype=float),
                       "temp_c": pd.Series(temp, dtype=float)}).sort_index()
    df.index = df.index.astype(str)
    print(f"    monthly weather rows: {len(df)}  {df.index[0]}..{df.index[-1]}  "
          f"rain NaN={int(df['rain_mm'].isna().sum())}  "
          f"temp NaN={int(df['temp_c'].isna().sum())}")
    tail = df.loc[[p for p in df.index if p >= "202606"]]
    print(f"    tail:\n{tail.round(2).to_string()}")
    meta = {
        "source": "Hong Kong Observatory monthly extract",
        "url": HKO_URL,
        "snapshot": path.name,
        "title": "Monthly meteorological data (stn.data = one record per year)",
        "selector": {"rain_mm": "yearData column index 10 (total rainfall)",
                     "temp_c": "yearData column index 4 (mean temperature)"},
        "selector_note": "column map idx4 = mean temp, idx10 = total rainfall; "
                         "'***' -> NaN; 'Trace' -> 0.05 mm; trailing '#' stripped",
        "unit": "mm / degrees Celsius",
        "monthly": df,
    }
    return df, meta


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def yoy_level(s: pd.Series, label: str, log1p: bool = False) -> pd.Series:
    """y(t) - y(t-12) on a gap-free grid that spans at least the panel window.

    The grid is padded out to PANEL_LAST so that shift(1) can still carry the
    last published month's YoY into the final panel month (e.g. cargo, whose
    level series currently stops at 202607 but is published with a lag).
    """
    g = monthly_grid(s)
    lo = min(g.index.min(), as_periods([PANEL_FIRST])[0])
    hi = max(g.index.max(), as_periods([PANEL_LAST])[0])
    g = g.reindex(pd.period_range(lo, hi, freq="M"))
    lg = safe_log(g, label, log1p=log1p)
    return lg - lg.shift(12)


def train_z(x: pd.Series, train_periods: list[str], label: str) -> tuple[pd.Series, dict]:
    """x must be indexed by Period; the fit uses only the train months"""
    fit = x.reindex(as_periods(train_periods)).astype(float)
    mu, sd = float(fit.mean()), float(fit.std(ddof=DDDOF))
    if not np.isfinite(sd) or sd == 0:
        raise RuntimeError(f"{label}: degenerate scaler (mu={mu}, sd={sd})")
    return (x - mu) / sd, {"fit_periods": f"{train_periods[0]}..{train_periods[-1]}",
                           "n_fit": len(fit), "mu": mu, "sigma": sd, "ddof": DDDOF}


def weather_anomaly_z(x: pd.Series, train_periods: list[str], label: str) -> tuple[pd.Series, dict]:
    """anomaly vs the SAME calendar month's train mean, then train-anomaly z"""
    months = pd.Series([int(p[4:]) for p in x.index], index=x.index)
    train = x.reindex(train_periods).astype(float)
    tr_months = pd.Series([int(p[4:]) for p in train_periods], index=train.index)
    cal_mean = train.groupby(tr_months).mean()
    if cal_mean.isna().any():
        raise RuntimeError(f"{label}: calendar-month train mean is NaN for "
                           f"{list(cal_mean[cal_mean.isna()].index)}")
    anom = x - months.map(cal_mean)
    a_fit = anom.reindex(train_periods).astype(float)
    mu, sd = float(a_fit.mean()), float(a_fit.std(ddof=DDDOF))
    z = (anom - mu) / sd
    return z, {"fit_periods": f"{train_periods[0]}..{train_periods[-1]}",
               "n_fit_per_month": {int(k): int(v) for k, v in
                                   tr_months.value_counts().sort_index().items()},
               "calendar_month_means": {int(k): float(v) for k, v in cal_mean.items()},
               "anomaly_mu": mu, "anomaly_sigma": sd, "ddof": DDDOF}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("STAGE 0 / STEP 4 — exogenous feature block")
    print("=" * 78)

    panel = pd.read_csv(DATA / "panel_wide.csv", index_col=0)
    periods = [str(p) for p in panel.index]
    if not (len(periods) == 263 and periods[0] == PANEL_FIRST and periods[-1] == PANEL_LAST):
        raise SystemExit(f"panel_wide.csv shape/period mismatch: "
                         f"{len(periods)} rows {periods[0]}..{periods[-1]}")
    train_periods = [p for p in periods if p <= TRAIN_END]
    post_periods = [p for p in periods if p > TRAIN_END]
    print(f"  panel: {len(periods)} periods {periods[0]}..{periods[-1]}  "
          f"(train {len(train_periods)} <= {TRAIN_END}, post {len(post_periods)})")

    sources: dict[str, dict] = {}
    failures: dict[str, str] = {}
    raw_levels: dict[str, pd.Series] = {}
    weather_raw: pd.DataFrame | None = None

    loaders = [
        ("cpi", build_cpi), ("rate", build_rate), ("eeri", build_eeri),
        ("visitor", build_visitor), ("cargo", build_cargo), ("mtr", build_mtr),
    ]
    for key, fn in loaders:
        try:
            s, meta = fn()
            raw_levels[key] = s
            sources[key] = meta
        except Exception as exc:  # noqa: BLE001 - one failing source must not kill the rest
            failures[key] = f"{type(exc).__name__}: {exc}"
            print(f"  !! SOURCE FAILED [{key}]: {failures[key]}")
    try:
        weather_raw, sources["weather"] = build_weather()
    except Exception as exc:  # noqa: BLE001
        failures["weather"] = f"{type(exc).__name__}: {exc}"
        print(f"  !! SOURCE FAILED [weather]: {failures['weather']}")

    # ---- transforms -----------------------------------------------------
    print("\n  --- transforms (train fit only on periods <= %s) ---" % TRAIN_END)
    feats: dict[str, pd.Series] = {}
    scalers: dict[str, dict] = {}

    def add_level_feature(name: str, level: pd.Series, label: str, log1p: bool,
                          shift: bool = True) -> None:
        y = yoy_level(level, label, log1p=log1p)
        if shift:
            y = y.shift(1)
        feats[name] = to_panel(y, periods)

    if "cpi" in raw_levels:
        add_level_feature("cpi_yoy", raw_levels["cpi"], "cpi", log1p=False)
    if "eeri" in raw_levels:
        add_level_feature("eeri_yoy", raw_levels["eeri"], "eeri", log1p=False)
    if "visitor" in raw_levels:
        add_level_feature("visitor_yoy", raw_levels["visitor"], "visitor", log1p=True)
    if "mtr" in raw_levels:
        add_level_feature("mtr_yoy", raw_levels["mtr"], "mtr", log1p=False)
    if "cargo" in raw_levels:
        add_level_feature("cargo_yoy", raw_levels["cargo"], "cargo", log1p=False)

    if "rate" in raw_levels:
        g = monthly_grid(raw_levels["rate"])
        # the scaler is fitted on the panel's train months (200410..202312),
        # not on the 1980-2023 raw history: see EXOG.md judgement call J2
        z, sc = train_z(g, train_periods, "rate_z")
        feats["rate_z"] = to_panel(z.shift(1), periods)   # shift on the full grid
        scalers["rate_z"] = sc
        print(f"    rate_z   : fit {sc['fit_periods']}  mu={sc['mu']:.4f}%  "
              f"sigma={sc['sigma']:.4f}pp")

    if weather_raw is not None:
        for name, col, label in (("rain_anom_z", "rain_mm", "rain"),
                                 ("temp_anom_z", "temp_c", "temp")):
            g = monthly_grid(weather_raw[col])
            z, sc = weather_anomaly_z(
                to_panel(g, periods), train_periods, name)
            feats[name] = z            # NO shift: month-end observed
            scalers[name] = sc
            print(f"    {name:12s}: anomaly mu={sc['anomaly_mu']:+.4f}  "
                  f"sigma={sc['anomaly_sigma']:.4f}  (no shift)")

    if not feats:
        print("  !! no feature could be built — aborting")
        return 2

    wide = pd.DataFrame(feats).reindex(index=periods)
    wide.index.name = "period"
    present = [c for c in FEATURES if c in wide.columns]
    wide = wide[present]
    missing = [c for c in FEATURES if c not in wide.columns]

    # ---- guards ---------------------------------------------------------
    vals = wide.to_numpy(dtype=float)
    n_inf = int(np.isinf(vals).sum())
    if n_inf:
        raise SystemExit(f"REFUSING TO WRITE: {n_inf} inf value(s) in the feature block")
    assert len(wide) == 263, len(wide)

    # ---- coverage -------------------------------------------------------
    cov = {}
    print("\n  --- coverage ---")
    print(f"  {'feature':14s} {'first':>7s} {'last':>7s} {'nan_train':>9s} "
          f"{'nan_post':>8s} {'min':>9s} {'max':>9s}")
    for c in wide.columns:
        s = wide[c]
        nn = s.dropna()
        first = nn.index[0] if len(nn) else "-"
        last = nn.index[-1] if len(nn) else "-"
        ntr = int(s.loc[train_periods].isna().sum())
        npo = int(s.loc[post_periods].isna().sum())
        cov[c] = {"first_non_nan": first, "last_non_nan": last,
                  "n_nan_train": ntr, "n_nan_post": npo,
                  "n_nan_total": int(s.isna().sum()),
                  "min": float(s.min()), "max": float(s.max())}
        print(f"  {c:14s} {first:>7s} {last:>7s} {ntr:9d} {npo:8d} "
              f"{s.min():9.4f} {s.max():9.4f}")

    print("\n  --- scale check: values at the two ends of the panel ---")
    print(f"  {'feature':14s} {'200410':>12s} {'202608':>12s}")
    for c in wide.columns:
        print(f"  {c:14s} {wide.loc[PANEL_FIRST, c]:12.6f} {wide.loc[PANEL_LAST, c]:12.6f}")
    if missing:
        big = [c for c in wide.columns
               if wide[c].dropna().abs().max() > 5.0]
        print(f"\n  NOTE: |value| > 5 for: {big}  (visitor YoY collapse/recovery)")

    # ---- write ----------------------------------------------------------
    out_csv = DATA / "exog_wide.csv"
    wide.to_csv(out_csv, index=True, index_label="period", encoding="utf-8")

    groups = {"GROUP_A_MACRO_PRICE_FINANCE": GROUP_A_MACRO_PRICE_FINANCE,
              "GROUP_B_TRAFFIC_WEATHER_LOGISTICS": GROUP_B_TRAFFIC_WEATHER_LOGISTICS}
    (DATA / "exog_groups.json").write_text(
        json.dumps(groups, indent=2), encoding="utf-8")
    for gname, cols in groups.items():
        miss = [c for c in cols if c not in wide.columns]
        if miss:
            print(f"  !! group {gname} references absent feature(s): {miss}")

    # ---- meta -----------------------------------------------------------
    feat_meta = {}
    for c in wide.columns:
        src_key = SRC_OF[c]
        meta = sources.get(src_key)
        if meta is None:
            feat_meta[c] = {"error": failures.get(src_key, "source unavailable")}
            continue
        raw = meta.get("raw_series")
        if raw is None:                       # weather has a two-column frame
            col = "rain_mm" if c == "rain_anom_z" else "temp_c"
            raw = meta["monthly"][col]
        rw = to_panel(monthly_grid(raw.dropna()), periods).dropna()
        entry = {
            "source": meta["source"],
            "source_url": meta["url"],
            "snapshot": meta["snapshot"],
            "table_title": meta.get("title"),
            "series_selector": meta["selector"],
            "selector_note": meta.get("selector_note"),
            "raw_unit": meta.get("unit"),
            "raw_first_period": str(raw.dropna().index[0]),
            "raw_last_period": str(raw.dropna().index[-1]),
            "raw_n_obs": int(raw.dropna().shape[0]),
            "raw_first_period_in_panel": str(rw.index[0]) if len(rw) else None,
            "raw_last_period_in_panel": str(rw.index[-1]) if len(rw) else None,
            "transform": {
                "cpi_yoy": "log(cpi_t) - log(cpi_t-12), then shift(1)",
                "eeri_yoy": "log(eeri_t) - log(eeri_t-12), then shift(1)",
                "visitor_yoy": "log1p(vis_t) - log1p(vis_t-12), then shift(1)",
                "mtr_yoy": "log(mtr_t) - log(mtr_t-12), then shift(1)",
                "cargo_yoy": "log(cargo_t) - log(cargo_t-12), then shift(1)",
                "rate_z": "(BL_RATE - mu_train)/sigma_train, then shift(1)",
                "rain_anom_z": "(rain - calmonth_mean_train - mu_anom_train)"
                               "/sigma_anom_train, no shift",
                "temp_anom_z": "(temp - calmonth_mean_train - mu_anom_train)"
                               "/sigma_anom_train, no shift",
            }[c],
            "shift": 0 if c.startswith(("rain", "temp")) else 1,
            "train_end": TRAIN_END,
            "scaler": scalers.get(c),
            **cov[c],
        }
        if c in ("visitor_yoy", "mtr_yoy", "cargo_yoy"):
            entry["raw_unit_note"] = {
                "visitor_yoy": "number of arrivals; log1p used so a zero month "
                               "would not produce -inf",
                "mtr_yoy": "reported in thousands of passenger trips; the log "
                           "difference is invariant to that factor",
                "cargo_yoy": "thousand tonnes; log difference is unit-free",
            }[c]
        feat_meta[c] = entry

    for c in missing:                      # source failure -> explicit placeholder
        feat_meta[c] = {"error": failures.get(SRC_OF[c], "source unavailable"),
                        "present_in_csv": False}

    meta_out = {
        "generated": date.today().isoformat(),
        "builder": "data/build_exog.py",
        "panel_source": "data/panel_wide.csv",
        "snapshot_dir": str(RAW.relative_to(ROOT)),
        "n_periods": int(len(wide)),
        "period_first": periods[0],
        "period_last": periods[-1],
        "train_end": TRAIN_END,
        "n_train_months": len(train_periods),
        "n_post_months": len(post_periods),
        "features": feat_meta,
        "missing_features": missing,
        "source_failures": failures,
        "groups": groups,
    }
    (DATA / "exog_meta.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report(wide, cov, sources, scalers, failures, missing, meta_out, periods,
                 train_periods, post_periods)

    print("\n  --- wrote ---")
    for f in ("exog_wide.csv", "exog_meta.json", "exog_groups.json", "EXOG.md"):
        p = DATA / f
        print(f"    {f:18s} {p.stat().st_size:>9,} B")
    print("\nSTEP 4 OK")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def write_report(wide, cov, sources, scalers, failures, missing, meta, periods,
                 train_periods, post_periods) -> None:
    def fmt(v: float) -> str:
        return f"{v:+.4f}"

    lines: list[str] = []
    A = lines.append
    A("# Exogenous feature block — `data/exog_wide.csv`")
    A("")
    A(f"Generated by `data/build_exog.py` on {meta['generated']}. "
      f"{len(wide)} monthly rows, `{periods[0]}`..`{periods[-1]}`, index `period` "
      f"(`YYYYMM` string). All scalers are fitted on **periods <= {TRAIN_END}** only.")
    A("")
    A("Every value was produced by running `build_exog.py` and then re-checked by "
      "`data/verify_exog.py` (machine log: `data/exog_verify.md`).")
    A("")
    A("## 1. The eight features")
    A("")
    A("| column | source | selector | transform | shift |")
    A("|---|---|---|---|---|")
    sel_rows = [
        ("cpi_yoy", "censtatd 510-60001A", "`sv=CC_CM_1920`, `svDesc=Index`, `GROUP=''`, `GROUPDesc='Total'`",
         "`log(cpi_t) - log(cpi_t-12)`", "1"),
        ("rate_z", "censtatd 340-45021", "`sv=BL_RATE`",
         "`(BL_RATE - mu)/sigma`, train fit", "1"),
        ("eeri_yoy", "censtatd 340-46002", "`sv=EERI_T`",
         "`log(eeri_t) - log(eeri_t-12)`", "1"),
        ("visitor_yoy", "censtatd 650-80001", "`sv=VIS_ARR`, `REGION=''` (Total)",
         "`log1p(vis_t) - log1p(vis_t-12)`", "1"),
        ("mtr_yoy", "MTR patronage CSV", "Domestic Service Monthly Total, column 1",
         "`log(mtr_t) - log(mtr_t-12)`", "1"),
        ("cargo_yoy", "censtatd 410-55111A", "`sv=PORT_CARGO_TP`, `svDesc=\"('000 tonnes)\"`, `TRANS=S`, `DIRECTION=''`, `SHIPMENT_TYPE=''`",
         "`log(cargo_t) - log(cargo_t-12)`", "1"),
        ("rain_anom_z", "HKO monthlyExtract", "`yearData[10]` total rainfall",
         "calendar-month anomaly vs train, then train z", "0"),
        ("temp_anom_z", "HKO monthlyExtract", "`yearData[4]` mean temperature",
         "calendar-month anomaly vs train, then train z", "0"),
    ]
    for r in sel_rows:
        A("| `{}` | {} | {} | {} | {} |".format(*r))
    A("")
    A("## 2. Coverage")
    A("")
    A(f"NaN counts are over the panel window only: train = {len(train_periods)} months "
      f"({train_periods[0]}..{train_periods[-1]}), post = {len(post_periods)} months "
      f"({post_periods[0]}..{post_periods[-1]}).")
    A("")
    A("| feature | first non-NaN | last non-NaN | NaN in train | NaN post | min | max |")
    A("|---|---|---|---|---|---|---|")
    for c in wide.columns:
        d = cov[c]
        A(f"| `{c}` | {d['first_non_nan']} | {d['last_non_nan']} | "
          f"{d['n_nan_train']} | {d['n_nan_post']} | {d['min']:+.4f} | {d['max']:+.4f} |")
    A("")
    A("## 3. Judgement calls")
    A("")
    J = [
        ("J1 — CPI series choice",
         "`510-60001A` mixes four index families (`A_CM_1920`, `B_CM_1920`, "
         "`C_CM_1920`, `CC_CM_1920`) × ten commodity groups × three `svDesc` "
         "measures. The **composite** CPI is `CC_CM_1920`; the all-items total is "
         "`GROUP=''` with `GROUPDesc='Total'`; `svDesc='Index'` is the level series "
         "(the table also carries its own `Year-on-year % change` and "
         "`Month-to-month % change` columns, which are NOT used — the spec "
         "transforms the level itself)."),
        ("J2 — which months the rate z-score is fitted on",
         "`BL_RATE` runs back to 1980, but \"train\" in this project means the "
         "model's training months, i.e. the panel window `200410`..`202312` "
         f"({len(train_periods)} months). The scaler therefore uses mu="
         f"{scalers.get('rate_z', {}).get('mu', float('nan')):.4f}%, "
         f"sigma={scalers.get('rate_z', {}).get('sigma', float('nan')):.4f}pp. "
         "Fitting on 1980-2023 instead would mix in the 1980s double-digit rates "
         "and stretch the z-scale for no modelling benefit. No post-202312 data is "
         "used either way."),
        ("J3 — EERI variant",
         "`340-46002` carries `EERI_T` (trade-weighted), `EERI_I` (import-weighted) "
         "and `EERI_E` (export-weighted). The trade-weighted total is the standard "
         "competitiveness measure and the one the spec asks for; as a check "
         "`EERI_T` sits inside `[min(EERI_I, EERI_E), max(EERI_I, EERI_E)]` in all "
         "621 monthly observations, i.e. it behaves like the aggregate."),
        ("J4 — visitor arrivals",
         "`REGION=''` (`REGIONDesc='Total'`) is the all-region total, not a source "
         "market; it equals the sum of the 11 named regions exactly in all 272 "
         "months. The series starts in `200401`, so `visitor_yoy` needs "
         "`200401`+12 months of history and is therefore NaN for `200410`..`200501` "
         "(4 leading months). `log1p` is used because arrivals can in principle be "
         "zero; the observed minimum is 1,800 (COVID), so no zero handling was "
         "triggered. COVID collapse + recovery makes this the only feature whose "
         "range leaves ±5 (see §2)."),
        ("J5 — cargo",
         "Kept the level series (`svDesc=\"('000 tonnes)\"`), not the table's own "
         "`Year-on-year % change`; kept `DIRECTION=''` (Total) and "
         "`SHIPMENT_TYPE=''` (Total) because In/Out and DS/TS partition the total. "
         "The total equals Inward+Outward to within 1 thousand tonnes (rounding) "
         "in all 355 months. The published level series currently ends at "
         "`202607`, one month short of the panel end; `cargo_yoy` at `202608` is "
         "still computable because it only needs the `202607` level and its "
         "`202507` comparison."),
        ("J6 — MTR",
         "`Domestic Service Monthly Total` is column index 1, in **thousands** of "
         "passenger trips; commas stripped, `-` (no service) -> NaN. The file was "
         "parsed to 320 monthly rows, `200001`..`202608` (the brief said 327 lines "
         "/ May 2000: the extra lines are the title, two header rows and two "
         "footnotes, and the oldest month is actually January 2000). Values are "
         "multiplied by 1000 before logging; this only affects the `+1` inside "
         "`log1p` (not used here) and is otherwise scale-free."),
        ("J7 — rainfall `Trace`",
         "`Trace` = rain too small to measure. Mapped to **0.05 mm**, the HKO "
         "convention for a trace amount, and documented here. There are 26 such "
         "cells in the whole 1884-2026 file; within the panel window the affected "
         "months are listed implicitly by a rainfall value of 0.05. `***` = missing "
         "-> NaN. Trailing `#` markers (provisional figures, seen on 2026-08) are "
         "stripped."),
        ("J8 — weather is not shifted",
         "A month's mean temperature and total rainfall are fully observed when the "
         "month closes, so `rain_anom_z` / `temp_anom_z` are aligned to the same "
         "month. The macro series are official statistics published with a lag, so "
         "they are shifted by one month: month t uses the value published for t-1."),
        ("J9 — standard deviation",
         "`std(ddof=0)` (population std), matching `data/make_features.py`, which "
         "uses `logs.std(ddof=0)` for `scaling.csv`. Stated here because "
         "`pandas.Series.std` defaults to ddof=1."),
        ("J10 — no imputation",
         "Nothing is filled, interpolated or clipped. NaNs are kept so the "
         "downstream window builder can drop incomplete windows honestly; the only "
         "NaN region that matters is the 4 leading `visitor_yoy` months, which sit "
         "in the first few 12-month context windows."),
    ]
    for head, body in J:
        A(f"**{head}.** {body}")
        A("")
    A("## 4. Known limitations worth remembering")
    A("")
    A("- `visitor_yoy` during the 2020-2023 border closure spans roughly "
      f"{cov.get('visitor_yoy', {}).get('min', float('nan')):+.2f} .. "
      f"{cov.get('visitor_yoy', {}).get('max', float('nan')):+.2f}; that is the "
      "true YoY, not a bug, but it dominates the scale of group B.")
    A("- `cargo_yoy` at `202608` uses the `202607` cargo level (the last published "
      "month); from `202609` onwards the feature would be NaN.")
    A("- The published `Year-on-year % change` column of `510-60001A` does not "
      "reproduce `log(cpi)-log(cpi-12)` exactly (mean |diff| ~0.1pp, max 0.77pp "
      "in-window) because the index published in the API is rounded to one decimal "
      "and historical segments were rebased; the spec's log-difference on the "
      "published index is what this pipeline uses.")
    if failures:
        A("")
        A("## 5. Failed sources")
        A("")
        for k, v in failures.items():
            A(f"- `{k}`: {v}")
    if missing:
        A("")
        A(f"**Absent columns (source failure): {missing}**")
    A("")
    (DATA / "EXOG.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import sys

    sys.exit(main())
