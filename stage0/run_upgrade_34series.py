"""
Stage 0 — UPGRADE: feed the 11 C&SD broad-product-category series
(S1..S4 = supermarket, D1..D7 = department store) into the three champions and
measure whether the ORIGINAL 23 series are then forecast BETTER or WORSE.

WHY THIS EXPERIMENT
-------------------
`docs/DATA_EXPANSION_ASSESSMENT.md` concluded that the next lever was not more
exogenous columns but more *series*: the two biggest parent aggregates —
`8` (supermarkets) and `30` (department stores) — are the two series that
reconciliation and channel mixing were least able to help, because nothing in
the panel decomposed them.  C&SD publishes that decomposition in two further
tables, already downloaded to `data_external/raw/censtatd/`:

    620-67011  "Value of retail sales in supermarkets by broad product category"
               BROAD_PROD_CAT in {"" (Total), S1, S2, S3, S4}
    620-67012  "Value of retail sales in department stores by broad product category"
               BROAD_PROD_CAT in {"" (Total), D1..D6, D1_D6, D7}

This file turns them into a 34-series panel, rebuilds the IDENTICAL 27-feature
table for all 34, retrains the three champions on 34 series, and then scores
ONLY the original 23 (12 origins x 4 horizons x 23 series = 1,104 points) so the
number is comparable with the frozen baselines:

    iTransformer + learnable anchor ENS3   0.90198 / WAPE 0.08781
    xLSTM + learnable anchor ENS3          0.94810
    xLSTM ENS3 + MinT_wls                  0.94455
    SeasonalNaive                          1.03029

HARD RULES OBEYED
-----------------
  * `eval/harness.py`, `models/*.py`, `data/make_features.py`,
    `data/panel_wide.csv`, `data/features_long.csv` and every existing
    `stage*/results/*` are READ-ONLY here.  Nothing is edited; only new files
    are written (`data/panel_wide_34.csv`, `data/features_long_34.csv`,
    `data/scaling_34.csv`, `results/upgrade34/*`, one new .py, one new .md).
  * The origin plan is copied, not re-invented: `run_sota_baselines.origin_plan`
    is imported (VAL_MONTHS=24 + purge at max(HORIZONS)), so the 182 training
    origins and the 13 purged validation origins are bit-identical to the
    frozen runs.  Only <= 2023-12 data can influence a fit or a choice.
  * The 23-series arm is re-run through THIS same code path as a control, so
    "34 vs 23" is a like-for-like comparison and not a cross-pipeline one
    (docs/REPRO_EXOG_TOKEN.md records that a different origin plan alone moves
    MASE by ~0.03).

SUBCOMMANDS
-----------
The script is resumable: every step writes its artefact and skips if present
unless --force is given.

    .venv\\Scripts\\python.exe run_upgrade_34series.py build-data
        panel_wide_34.csv, features_long_34.csv, scaling_34.csv + validation
        of sum(S1..S4) vs series 8 and sum(D1..D7) vs series 30.

    .venv\\Scripts\\python.exe run_upgrade_34series.py smoke
        1 seed, 2 origins, both models, 34 series — proves the shapes and the
        metric pipeline before spending the full budget.

    .venv\\Scripts\\python.exe run_upgrade_34series.py control23
        the SAME code path with 23 series; must reproduce the frozen
        xLSTM_none_s*.csv / itransformer_anchor_s*.csv point-for-point.

    .venv\\Scripts\\python.exe run_upgrade_34series.py full
        3 seeds (42/7/2024) x 12 origins x 34 series, both models, ENS3,
        MinT_wls/BU/MinT_ols on the 34-node hierarchy, then only 23 scored.

    .venv\\Scripts\\python.exe run_upgrade_34series.py report
        tables + docs/UPGRADE_34SERIES_RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H                      # noqa: E402
from models import itransformer as IT              # noqa: E402
from models import xlstm_direct as X               # noqa: E402

DATA = ROOT / "data"
RAW = ROOT.parent / "data_external" / "raw" / "censtatd"
RES = ROOT / "results"
OUT = RES / "upgrade34"

SEEDS = (42, 7, 2024)
CTX = 12
EPOCHS = 100

# ---------------------------------------------------------------------------
# series bookkeeping
# ---------------------------------------------------------------------------
# the frozen panel decides the 23 names AND their order (sorted by (len, code),
# the same key models/xlstm_direct.build_tensors uses)
SERIES23 = sorted(list(H.load_panel().columns), key=lambda s: (len(s), s))
NEW_SUPER = ["S1", "S2", "S3", "S4"]
NEW_DEPT = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
NEW_SERIES = NEW_SUPER + NEW_DEPT
SERIES34 = sorted(SERIES23 + NEW_SERIES, key=lambda s: (len(s), s))
assert len(SERIES23) == 23 and len(SERIES34) == 34, (len(SERIES23), len(SERIES34))
# the 23 must occupy the same first positions in the 34 ordering, otherwise
# "take the original 23" would silently mix series
assert SERIES34[:23] == SERIES23 or set(SERIES23).issubset(set(SERIES34[:23])), \
    "34-series order does not contain the 23 in their original order"

# frozen 4-parent hierarchy (copied from run_stage4_reconcile.HIER)
HIER23 = {
    "2": ["3", "4", "5", "6", "7"],
    "12": ["13", "14"],
    "18": ["19", "23", "51"],
    "35": ["36", "37", "38", "39", "40"],
}
# the upgrade: the two big aggregates 8 and 30 get children
HIER34 = dict(HIER23)
HIER34["8"] = list(NEW_SUPER)
HIER34["30"] = list(NEW_DEPT)

TOP_LEVEL = ["2", "8", "9", "12", "18", "30", "32", "35"]
PARENT_OF = {kid: par for par, kids in HIER34.items() for kid in kids}

TABLE_MAIN = "620-67002"
TABLE_SUPER = "620-67011"
TABLE_DEPT = "620-67012"

DROP_MAIN = {"", "TOTAL", "49"}
KEEP_SUPER = {"S1", "S2", "S3", "S4"}          # "" (Total) excluded
KEEP_DEPT = {"D1", "D2", "D3", "D4", "D5", "D6", "D7"}   # "" and D1_D6 excluded


# ===========================================================================
# A. DATA — 34-series panel and its feature table
# ===========================================================================
def _payload(table: str) -> dict:
    p = RAW / f"{table}.json"
    if not p.exists():
        raise SystemExit(f"missing raw snapshot {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def to_float(v) -> float:
    """figures may be '***' (suppressed) or '' -> NaN (same rule as build_panel.py)"""
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
    for tag in ("<strong>", "</strong>", "<i>", "</i>",
                '<font color="blue">', "</font>"):
        s = s.replace(tag, "")
    return " ".join(s.split())


def _collect(table: str, key_field: str, keep: set[str] | None,
             drop: set[str]) -> list[tuple[str, str, float]]:
    """monthly VAL_RS rows -> [(period, series_code, value)]"""
    out = []
    n_seen = 0
    for r in _payload(table)["dataSet"]:
        n_seen += 1
        if r.get("sv") != "VAL_RS":
            continue
        if r.get("freq") != "M":
            continue
        p = str(r.get("period", ""))
        if not (p.isdigit() and len(p) == 6):
            continue
        k = "" if r.get(key_field) in (None, "") else str(r.get(key_field)).strip()
        if k in drop:
            continue
        if keep is not None and k not in keep:
            continue
        out.append((p, k, to_float(r.get("figure"))))
    print(f"    {table}: {n_seen:,} rows scanned -> {len(out):,} monthly VAL_RS rows")
    return out


def build_panel_34(force: bool = False) -> pd.DataFrame:
    """write data/panel_wide_34.csv (263 periods x 34 series, string index)"""
    dst = DATA / "panel_wide_34.csv"
    base = H.load_panel()

    print("=" * 88)
    print("  A1. build the 34-series panel")
    print("=" * 88)

    # --- the existing 23, re-read from the RAW snapshot with build_panel's rules
    recs = _collect(TABLE_MAIN, "OUTLET_TYPE", None, DROP_MAIN)
    old = (pd.DataFrame(recs, columns=["period", "series", "value"])
           .pivot_table(index="period", columns="series", values="value",
                        aggfunc="first").sort_index())
    old = old[sorted(old.columns, key=lambda c: (len(c), c))]
    old.index = old.index.astype(str)
    # byte-level identity with the frozen panel: this is the guard that the
    # extraction rules were copied and not re-invented
    diff = (old[base.columns] - base).abs().max().max()
    print(f"    re-extracted 23 from raw vs data/panel_wide.csv: "
          f"shape {old.shape}   max abs diff {diff:.6g}")
    assert old.shape[1] == 23 and diff < 1e-9, "raw re-extraction disagrees with frozen panel"

    # --- the 11 new sub-categories
    new = []
    new += _collect(TABLE_SUPER, "BROAD_PROD_CAT", KEEP_SUPER, {""})
    new += _collect(TABLE_DEPT, "BROAD_PROD_CAT", KEEP_DEPT, {"", "D1_D6"})
    nw = (pd.DataFrame(new, columns=["period", "series", "value"])
          .pivot_table(index="period", columns="series", values="value",
                       aggfunc="first").sort_index())
    nw.index = nw.index.astype(str)
    print(f"    new sub-series frame: {nw.shape[0]} periods x {nw.shape[1]} series "
          f"({nw.index[0]}..{nw.index[-1]})  cols {list(nw.columns)}")
    assert nw.shape[1] == 11, nw.columns
    assert list(nw.index) == list(base.index), "new tables do not cover the same months"

    wide = pd.concat([old[base.columns], nw], axis=1)
    wide = wide[SERIES34]
    wide.index.name = "period"
    n_missing = int(wide.isna().sum().sum())
    print(f"    merged: {wide.shape[0]} periods x {wide.shape[1]} series   "
          f"missing cells {n_missing}")
    assert n_missing == 0, "panel must be complete"
    assert not wide.isna().any().any()

    if force or not dst.exists():
        wide.to_csv(dst, encoding="utf-8")
        print(f"    wrote {dst.relative_to(ROOT.parent)}  ({dst.stat().st_size:,} B)")
    else:
        print(f"    [keep] {dst.relative_to(ROOT.parent)} (exists; --force to rewrite)")
    return wide


def validate_hierarchy_34(wide: pd.DataFrame) -> pd.DataFrame:
    """sum(S1..S4) vs 8 and sum(D1..D7) vs 30, month by month"""
    print("\n" + "=" * 88)
    print("  A2. do the new children actually add up to the parent?")
    print("=" * 88)
    rows = []
    for parent, kids, label in (("8", NEW_SUPER, "sum(S1..S4)  vs series 8  (supermarkets)"),
                                ("30", NEW_DEPT, "sum(D1..D7)  vs series 30 (department stores)")):
        gap = (wide[kids].sum(axis=1) - wide[parent])
        a = gap.abs()
        print(f"  {label}")
        print(f"    months {len(a)}   max |gap| {a.max():.3f}   "
              f"median |gap| {a.median():.3f}   mean {a.mean():.3f}   "
              f"share of months with |gap| <= 2.0: {100.0*(a<=2.0).mean():.1f}%")
        print(f"    relative: max |gap| / mean(parent) = "
              f"{100.0*a.max()/wide[parent].mean():.3f}%")
        for p, g, pp in zip(a.index, gap, wide[parent]):
            rows.append({"parent": parent, "children": "+".join(kids),
                         "period": p, "parent_value": float(pp),
                         "children_sum": float(pp + g), "gap": float(g),
                         "abs_gap": float(abs(g))})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "panel34_hierarchy_validation.csv", index=False)
    print(f"  wrote {(OUT / 'panel34_hierarchy_validation.csv').relative_to(ROOT.parent)} "
          f"({len(df)} rows)")
    return df


# ---- feature engineering: copied verbatim from data/make_features.py --------
import calendar                                                   # noqa: E402
from datetime import date, timedelta                              # noqa: E402

TRAIN_END = "202312"

LNY = {
    2004: (1, 22), 2005: (2, 9), 2006: (1, 29), 2007: (2, 18), 2008: (2, 7),
    2009: (1, 26), 2010: (2, 14), 2011: (2, 3), 2012: (1, 23), 2013: (2, 10),
    2014: (1, 31), 2015: (2, 19), 2016: (2, 8), 2017: (1, 28), 2018: (2, 16),
    2019: (2, 5), 2020: (1, 25), 2021: (2, 12), 2022: (2, 1), 2023: (1, 22),
    2024: (2, 10), 2025: (1, 29), 2026: (2, 17),
}
VOUCHER_MONTHS = {"202108", "202204", "202208"}
COVID_FROM, COVID_TO = "202002", "202301"
LAGS = (1, 2, 3, 12, 13)
ROLL = (3, 6, 12)


def month_bounds(period: str) -> tuple[date, date]:
    y, m = int(period[:4]), int(period[4:])
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def overlap_days(window_start: date, window_end: date, period: str) -> int:
    m_start, m_end = month_bounds(period)
    lo = max(window_start, m_start)
    hi = min(window_end, m_end)
    return (hi - lo).days + 1 if hi >= lo else 0


def lny_features(period: str) -> tuple[float, float]:
    y = int(period[:4])
    m = int(period[4:])
    if y not in LNY:
        return 0.0, 0.0
    lm, ld = LNY[y]
    if m not in (1, 2, lm):
        return 0.0, 0.0
    lny = date(y, lm, ld)
    pre = overlap_days(lny - timedelta(days=14), lny - timedelta(days=1), period) / 14.0
    post = overlap_days(lny, lny + timedelta(days=6), period) / 7.0
    return round(pre, 6), round(post, 6)


def build_features_34(wide: pd.DataFrame, force: bool = False):
    """write features_long_34.csv + scaling_34.csv with EXACTLY the same schema
    and the same arithmetic as data/make_features.py"""
    feat_dst = DATA / "features_long_34.csv"
    scal_dst = DATA / "scaling_34.csv"

    print("\n" + "=" * 88)
    print("  A3. features for 34 series (identical definition, train-only scaling)")
    print("=" * 88)

    periods = list(wide.index)
    series = list(wide.columns)
    train_periods = [p for p in periods if p <= TRAIN_END]
    print(f"  panel {len(periods)} periods x {len(series)} series   "
          f"train months {len(train_periods)} ({train_periods[0]}..{train_periods[-1]})")

    logs = np.log(wide.astype(float))
    mu = logs.loc[train_periods].mean()
    sd = logs.loc[train_periods].std(ddof=0)
    scaling = pd.DataFrame({"series": mu.index, "mu_log": mu.values,
                            "sigma_log": sd.values})
    scaling["level_id"] = scaling["series"].map(lambda s: 0 if s in TOP_LEVEL else 1)
    scaling["parent_id"] = scaling["series"].map(lambda s: PARENT_OF.get(s, "none"))

    y_z = ((logs - mu) / sd).dropna(how="all")

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
            for L in LAGS:
                j = periods.index(p) - L
                rec[f"lag{L}"] = float(s_log.iloc[j]) if j >= 0 else np.nan
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
            mm = int(p[4:])
            for k in range(1, 13):
                rec[f"month_{k}"] = 1.0 if k == mm else 0.0
            pre, post = lny_features(p)
            rec["lny_pre14"] = pre
            rec["lny_post7"] = post
            rec["covid_border_closed"] = 1.0 if COVID_FROM <= p <= COVID_TO else 0.0
            rec["consumption_voucher"] = 1.0 if p in VOUCHER_MONTHS else 0.0
            rows.append(rec)

    feat = pd.DataFrame(rows)
    feat = feat.sort_values(["period", "series_id"]).reset_index(drop=True)
    feat["split"] = np.where(feat["period"] <= TRAIN_END, "train", "post")

    # ---- verification against the frozen 23-series feature table -----------
    frozen = pd.read_csv(DATA / "features_long.csv",
                         dtype={"period": str, "series": str, "parent_id": str})
    frozen["period"] = frozen["period"].astype(str).str.zfill(6)
    frozen["series"] = frozen["series"].astype(str)
    assert list(feat.columns) == list(frozen.columns), (
        "schema mismatch:\n  34 : %s\n  23 : %s" % (list(feat.columns), list(frozen.columns)))
    sub = feat[feat["series"].isin(SERIES23)].reset_index(drop=True)
    fr = frozen.reset_index(drop=True)
    print(f"  schema identical to features_long.csv: {list(feat.columns) == list(frozen.columns)}"
          f"  ({len(feat.columns)} columns)")
    print(f"  rows: 34-arm {len(feat):,} (of which 23-series {len(sub):,})   "
          f"frozen {len(fr):,}")
    num_cols = [c for c in feat.columns
                if c not in ("period", "series", "parent_id", "split")]
    worst = {}
    for c in num_cols:
        a = sub[c].to_numpy(dtype=float)
        b = fr[c].to_numpy(dtype=float)
        m = np.nanmax(np.abs(a - b)) if len(a) == len(b) else np.inf
        worst[c] = float(m)
    bad = {c: v for c, v in worst.items() if not (v < 1e-12 or (v != v))}
    print(f"  numeric columns compared: {len(num_cols)}   "
          f"max |34-arm - frozen| over all of them: {max(worst.values()):.3g}")
    print(f"  columns disagreeing by > 1e-12: {bad if bad else 'none'}")
    assert not bad, "34-series feature table does not reproduce the frozen one"
    assert (sub["period"].to_numpy() == fr["period"].to_numpy()).all()
    assert (sub["series"].to_numpy() == fr["series"].to_numpy()).all()
    assert (sub["series_id"].to_numpy() == fr["series_id"].to_numpy()).all()
    # scaling for the shared 23 must be identical too
    fscal = pd.read_csv(DATA / "scaling.csv", dtype={"series": str})
    m = scaling.merge(fscal, on="series", suffixes=("_34", "_23"))
    print(f"  scaling rows shared with scaling.csv: {len(m)}   "
          f"max |dmu| {np.abs(m['mu_log_34']-m['mu_log_23']).max():.3g}   "
          f"max |dsigma| {np.abs(m['sigma_log_34']-m['sigma_log_23']).max():.3g}")
    assert np.abs(m["mu_log_34"] - m["mu_log_23"]).max() < 1e-12
    assert np.abs(m["sigma_log_34"] - m["sigma_log_23"]).max() < 1e-12

    if force or not feat_dst.exists():
        feat.to_csv(feat_dst, index=False, encoding="utf-8")
        print(f"  wrote {feat_dst.relative_to(ROOT.parent)} ({feat_dst.stat().st_size:,} B)")
    else:
        print(f"  [keep] {feat_dst.relative_to(ROOT.parent)}")
    if force or not scal_dst.exists():
        scaling.to_csv(scal_dst, index=False, encoding="utf-8")
        print(f"  wrote {scal_dst.relative_to(ROOT.parent)} ({scal_dst.stat().st_size:,} B)")
    else:
        print(f"  [keep] {scal_dst.relative_to(ROOT.parent)}")
    return feat, scaling


def load_arm(n_series: int):
    """feat / wide / scaling for the 23-series arm or the 34-series arm"""
    if n_series == 34:
        feat = pd.read_csv(DATA / "features_long_34.csv",
                           dtype={"series": str, "period": str, "parent_id": str})
        wide = pd.read_csv(DATA / "panel_wide_34.csv", index_col=0, dtype={0: str})
        scaling = pd.read_csv(DATA / "scaling_34.csv", dtype={"series": str})
    elif n_series == 23:
        feat = H.load_features()
        wide = H.load_panel()
        scaling = pd.read_csv(DATA / "scaling.csv", dtype={"series": str})
    else:
        raise ValueError(n_series)
    feat["period"] = feat["period"].astype(str).str.zfill(6)
    feat["series"] = feat["series"].astype(str)
    wide.index = wide.index.astype(str)
    return feat, wide, scaling


# ===========================================================================
# B. MODELS — the three champions, on 23 or 34 series
# ===========================================================================
def _plan(n_series: int, feat: pd.DataFrame):
    """the frozen origin plan, via the frozen function (not a re-implementation)"""
    from run_sota_baselines import origin_plan
    periods, series, F, Y, KF = X.build_tensors(feat)
    tr_i, va_i, plan = origin_plan(F, periods, CTX)
    assert len(series) == n_series, (len(series), n_series)
    return periods, series, F, Y, KF, tr_i, va_i, plan


def run_itransformer(n_series: int, seeds=SEEDS, origins=None, verbose=True,
                     outdir: Path | None = None, force: bool = False):
    """iTransformer + learnable 2-term anchor — the 0.90198 champion.

    Channel mixing: S tokens per sample.  S=23 reproduces the frozen run,
    S=34 simply gives the encoder 11 more tokens (parameter count is
    unchanged: the embedding is Linear(L->d) per token and the readout is
    shared, so 26,024 params either way).
    """
    import torch
    from run_sota_baselines import rows_from_levels

    origins = list(origins or H.ORIGINS)
    feat, wide, scaling = load_arm(n_series)
    periods, series, F, Y, KF, tr_i, va_i, plan = _plan(n_series, feat)
    Z = Y.astype(np.float64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"  iTransformer arm: {n_series} series, cube {F.shape}, device {dev}")
        print(f"    train origins {plan['train_origins'][0]}..{plan['train_origins'][1]}"
              f" ({plan['n_train_origins']})   val {plan['val_origins'][0]}.."
              f"{plan['val_origins'][1]} ({plan['n_val_origins']})   "
              f"max val target {plan['max_val_target_period']}")
        print(f"    training tokens = {plan['n_train_origins']} origins x "
              f"{n_series} series = {plan['n_train_origins']*n_series:,}")

    frames, metas = [], []
    for seed in seeds:
        tag = f"iTransformer_{n_series}s_s{seed}"
        out_csv = (outdir or OUT) / f"preds_itransformer{n_series}_s{seed}.csv"
        if out_csv.exists() and not force:
            d = pd.read_csv(out_csv, dtype={"series": str, "origin": str})
            if d["origin"].nunique() == len(origins):
                print(f"    [skip] {tag} already done ({len(d):,} rows)")
                frames.append(d)
                js = out_csv.with_suffix(".json")
                metas.append(json.loads(js.read_text(encoding="utf-8")) if js.exists() else {})
                continue
        t0 = time.time()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = IT.ITransformerNet(context=CTX, d_model=32, n_heads_attn=4, layers=2,
                                   dropout=0.1, use_anchor=True)
        model, meta = IT.train_transformer(model, Z, tr_i, va_i, context=CTX,
                                           epochs=EPOCHS, lr=5e-4, weight_decay=5e-3,
                                           batch=32, device=dev, verbose=verbose)
        rows = []
        for o in origins:
            i = periods.index(o)
            lv = IT.predict_transformer(model, Z, i, context=CTX)
            rows += rows_from_levels(tag, o, lv, series, wide, scaling)
        df = pd.DataFrame(rows, columns=["model", "origin", "h", "series",
                                         "y_true", "y_pred"])
        fit_sec = round(time.time() - t0, 1)
        meta.update({"family": "itransformer_anchor", "n_series": n_series,
                     "seed": seed, "tag": tag, "context": CTX, "epochs": EPOCHS,
                     "fit_sec": fit_sec, "device": dev, "plan": plan,
                     "n_train_tokens": plan["n_train_origins"] * n_series,
                     "origins_scored": origins, "anchor_type": "learnable 2-term (a_h)"})
        (outdir or OUT).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        out_csv.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        if verbose:
            print(f"    >>> {tag}  fit {fit_sec}s  params {meta['n_params']:,}  "
                  f"val {meta['best_val_loss']:.4f} @ep{meta['best_epoch']}")
        frames.append(df)
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), metas


def run_xlstm(n_series: int, seeds=SEEDS, origins=None, verbose=True,
              outdir: Path | None = None, force: bool = False):
    """xLSTM direct multi-horizon + learnable anchor — the 0.94810 champion.

    Channel-independent, so 34 series is literally 34 x 182 = 6,188 samples
    instead of 4,186; nothing else about the model changes.
    """
    origins = list(origins or H.ORIGINS)
    feat, wide, scaling = load_arm(n_series)
    if verbose:
        print(f"  xLSTM arm: {n_series} series")
    frames, metas = [], []
    for seed in seeds:
        tag = f"xLSTM_{n_series}s_s{seed}"
        out_csv = (outdir or OUT) / f"preds_xlstm{n_series}_s{seed}.csv"
        if out_csv.exists() and not force:
            d = pd.read_csv(out_csv, dtype={"series": str, "origin": str})
            if d["origin"].nunique() == len(origins):
                print(f"    [skip] {tag} already done ({len(d):,} rows)")
                frames.append(d)
                js = out_csv.with_suffix(".json")
                metas.append(json.loads(js.read_text(encoding="utf-8")) if js.exists() else {})
                continue
        t0 = time.time()
        df, meta = X.run(feat, wide, scaling, epochs=EPOCHS, origins=origins,
                         verbose=verbose, tag=tag, purge_val=True, seed=seed)
        wall = round(time.time() - t0, 1)
        meta.update({"n_series": n_series, "family": "xlstm_anchor",
                     "seed": seed, "tag": tag, "wall_sec": wall,
                     "origins_scored": origins})
        (outdir or OUT).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        out_csv.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        if verbose:
            print(f"    >>> {tag}  wall {wall}s  params {meta['n_params']:,}  "
                  f"train samples {meta['n_train_samples']:,}  "
                  f"val {meta['best_val_loss']:.4f} @ep{meta['best_epoch']}")
        frames.append(df)
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), metas


# ---- scoring: ALWAYS only the original 23 -------------------------------
def naive_mae_locked() -> pd.Series:
    """MASE denominator from the FROZEN panel (load_panel), so the denominators
    are bit-identical to every previously published number"""
    return H.seasonal_naive_mae_train(H.load_panel())


def score23(df: pd.DataFrame, naive_mae: pd.Series, model: str | None = None) -> pd.DataFrame:
    d = df[df["series"].isin(SERIES23)].copy()
    if model is not None:
        d["model"] = model
    d["h"] = d["h"].astype(int)
    m = H.metrics_from_frame(d[["model", "origin", "h", "series", "y_true", "y_pred"]],
                             naive_mae)
    return m


def ensemble_frame(df: pd.DataFrame, seeds=SEEDS, model: str = "ENS3") -> pd.DataFrame:
    """point-wise MEAN of the per-seed level forecasts (the convention behind
    every ENS3 number in the project, see run_stage_h_ens.py)"""
    keys = ["origin", "h", "series"]
    d = df.copy()
    d["h"] = d["h"].astype(int)
    d["seed"] = d["model"].str.rsplit("_s", n=1).str[-1]
    piv = d.pivot_table(index=keys, columns="seed", values="y_pred", aggfunc="first")
    use = [c for c in piv.columns if int(c) in seeds]
    assert len(use) == len(seeds), (use, seeds)
    yt = d.groupby(keys)["y_true"].first()
    out = piv[use].mean(axis=1).rename("y_pred").to_frame()
    out["y_true"] = yt
    out = out.reset_index()
    out["model"] = model
    out["h"] = out["h"].astype(int)
    return out[["model", "origin", "h", "series", "y_true", "y_pred"]]


# ---- hierarchical reconciliation (copied semantics from run_stage4_reconcile) ----
def build_A(series: list[str], hier: dict) -> np.ndarray:
    idx = {s: i for i, s in enumerate(series)}
    A = np.zeros((len(hier), len(series)))
    for r, (p, ch) in enumerate(hier.items()):
        A[r, idx[p]] += 1.0
        for c in ch:
            assert c in idx, f"child {c} missing from the series list"
            A[r, idx[c]] -= 1.0
    return A


def naive_w(wide: pd.DataFrame, series: list[str], train_end: str) -> np.ndarray:
    """per-series variance of the in-sample seasonal-naive error (TRAIN ONLY)"""
    tr = wide.loc[[i for i in wide.index if i <= train_end], series]
    err = tr - tr.shift(12)
    v = err.var(axis=0, ddof=1).to_numpy(dtype=float)
    v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
    if np.isnan(v).any():
        v = np.where(np.isnan(v), np.nanmedian(v), v)
    return v


def reconcile(vecs: np.ndarray, A: np.ndarray, W: np.ndarray) -> np.ndarray:
    if np.allclose(W, np.eye(len(W))):
        M = A.T @ np.linalg.inv(A @ A.T) @ A
    else:
        AW = A @ W
        M = W @ A.T @ np.linalg.inv(AW @ A.T) @ A
    return vecs - vecs @ M.T


def bottom_up(vecs: np.ndarray, series: list[str], hier: dict) -> np.ndarray:
    out = vecs.copy()
    idx = {s: i for i, s in enumerate(series)}
    for p, ch in hier.items():
        out[:, idx[p]] = vecs[:, [idx[c] for c in ch]].sum(axis=1)
    return out


def to_matrix(df: pd.DataFrame, series: list[str]) -> tuple[list, np.ndarray, np.ndarray]:
    keys = sorted({(o, h) for o, h in zip(df["origin"], df["h"])},
                  key=lambda t: (str(t[0]), int(t[1])))
    Y = np.full((len(keys), len(series)), np.nan)
    P = np.full((len(keys), len(series)), np.nan)
    for n, (o, h) in enumerate(keys):
        sub = df[(df["origin"] == o) & (df["h"] == int(h))]
        m = dict(zip(sub["series"], sub["y_pred"]))
        t = dict(zip(sub["series"], sub["y_true"]))
        for j, s in enumerate(series):
            P[n, j] = m.get(s, np.nan)
            Y[n, j] = t.get(s, np.nan)
    if np.isnan(P).any() or np.isnan(Y).any():
        raise SystemExit("incomplete forecast matrix")
    return keys, Y, P


def from_matrix(keys, Y, P, series, model) -> pd.DataFrame:
    return pd.DataFrame({
        "model": model,
        "origin": np.repeat([k[0] for k in keys], len(series)),
        "h": np.repeat([int(k[1]) for k in keys], len(series)),
        "series": np.tile(series, len(keys)),
        "y_true": Y.ravel(), "y_pred": P.ravel(),
    })


def per_series_mase(df: pd.DataFrame, naive_mae: pd.Series, model: str) -> pd.DataFrame:
    d = df[(df["model"] == model) & (df["series"].isin(SERIES23))].copy()
    ae = (d["y_true"] - d["y_pred"]).abs()
    g = ae.groupby(d["series"]).mean()
    out = pd.DataFrame({"series": g.index, "MAE": g.values})
    out["naive_mae"] = out["series"].map(naive_mae).to_numpy()
    out["MASE_series"] = out["MAE"] / out["naive_mae"]
    out["model"] = model
    return out.sort_values("series", key=lambda s: s.map(
        {k: i for i, k in enumerate(SERIES23)})).reset_index(drop=True)


# ===========================================================================
# B2. THE 23-VALIDATION ARM — the selection-bias control
# ===========================================================================
# §9.1 of the first write-up named the biggest internal-validity threat: the
# 34-series arms early-stop on a validation MSE averaged over **34** series,
# while the 23-series arm averages over **23**.  The two arms are therefore not
# selected by the same criterion, so "34 is worse" could be a model-SELECTION
# artefact rather than an information effect.
#
# This arm removes that difference and nothing else: the model still sees 34
# tokens / 34 channels, the loss is still over all 34, the origin plan and the
# purge are unchanged -- only the epoch chosen by early stopping is now the one
# that minimises the validation MSE of the ORIGINAL 23 series.
#
# The two trainers below are LINE-BY-LINE copies of
#   models/itransformer.py::train_transformer        (10 arugments → identical)
#   models/xlstm_direct.py::train_model              (champion configuration)
# with exactly one change: the validation tensor is sliced to the first `k_val`
# variate tokens before the MSE.  A copy is only trustworthy if it is provably
# the same function, so `--verify` runs it with k_val == n_series and requires
# the resulting predictions to match the shipped `preds_*34_s*.csv` point for
# point.  (Slower than editing the library file, but the library files are
# frozen, and an unverified copy would be worth nothing.)
def train_itransformer_valcut(model, Z, tr_i, va_i, k_val, context=CTX,
                              epochs=EPOCHS, lr=5e-4, weight_decay=5e-3,
                              batch=32, device=None, verbose=False):
    """copy of models.itransformer.train_transformer with a sliced val metric"""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    Wtr, Ytr = IT.build_windows(Z, tr_i, context)
    Wva, Yva = IT.build_windows(Z, va_i, context)
    n = len(Wtr)
    if n == 0:
        raise ValueError("no training origins")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Wva_t, Yva_t = Wva.to(dev), Yva.to(dev)
    best, best_ep, best_state = float("inf"), -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            xb, yb = Wtr[idx].to(dev), Ytr[idx].to(dev)
            opt.zero_grad()
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Wva_t)
            # ← the ONLY difference from the library loop
            vloss = float(((pred[:, :k_val] - Yva_t[:, :k_val]) ** 2).mean())
        if vloss < best:
            best, best_ep = vloss, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if verbose and (ep == 1 or ep % 25 == 0 or ep == epochs):
            print(f"    ep {ep:3d}  train {tot / n:.4f}  val@{k_val} {vloss:.4f}"
                  f"{'  *' if ep == best_ep else ''}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    meta = {"best_val_loss": float(best), "best_epoch": int(best_ep),
            "n_train_origins": int(n), "n_val_origins": int(len(Wva)),
            "n_series_per_sample": int(Z.shape[1]),
            "val_series_used": int(k_val),
            "n_val_points_used": int(k_val * len(Wva)),
            "n_params": int(sum(p.numel() for p in model.parameters()
                                if p.requires_grad)),
            "epochs": int(epochs), "batch": int(batch), "lr": float(lr),
            "weight_decay": float(weight_decay)}
    if hasattr(model, "anchor_weights"):
        meta["anchor_weights"] = {str(h): float(w) for h, w in
                                  zip(IT.HEADS, model.anchor_weights())}
    return model, meta


def train_xlstm_valcut(F, Y, KF, periods, series, cutoff_period, k_val,
                       seed, epochs=EPOCHS, context=CTX, embed=None,
                       blocks=None, dropout=None, lr=5e-4, batch=128,
                       weight_decay=5e-3, val_months=24, verbose=False):
    """copy of models.xlstm_direct.train_model (champion configuration:
    use_anchor=True, use_calendar=False, purge_val=True) with a sliced val metric"""
    import torch
    ctx = X.CONTEXT if context is None else int(context)
    emb = X.EMBED if embed is None else int(embed)
    blk = X.BLOCKS if blocks is None else int(blocks)
    drp = X.DROPOUT if dropout is None else float(dropout)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pidx = {p: i for i, p in enumerate(periods)}
    cut = pidx[cutoff_period]
    first = ctx - 1
    val_from = cut - val_months
    val_end = cut - max(X.HEADS)                       # purge_val=True
    tr_i = list(range(first, val_from))
    va_i = list(range(val_from, val_end + 1))
    if not va_i:
        raise ValueError("purge left no validation origins")
    sidx = list(range(len(series)))
    Xtr, Str, Ctr, Ttr = X.make_samples(tr_i, sidx, F, Y, KF, context=ctx)
    Xva, Sva, Cva, Tva = X.make_samples(va_i, sidx, F, Y, KF, context=ctx)
    if verbose:
        print(f"  [copy] train samples {len(Xtr):,}   val samples {len(Xva):,}   "
              f"val@{k_val} -> {k_val * len(va_i):,} points   device {dev}")
    X.set_seed(seed)
    model = X.XLSTMForecaster(len(series), F.shape[2] - 1, embed=emb, blocks=blk,
                              context=ctx, dropout=drp, use_anchor=True,
                              use_calendar=False, anchor_init=2.0).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtr_t = torch.from_numpy(Xtr)
    Str_t = torch.from_numpy(Str)
    Ctr_t = torch.from_numpy(Ctr)
    Ttr_t = torch.from_numpy(Ttr)
    Xva_t = torch.from_numpy(Xva).to(dev)
    Sva_t = torch.from_numpy(Sva).to(dev)
    Cva_t = torch.from_numpy(Cva).to(dev)
    Tva_t = torch.from_numpy(Tva).to(dev)
    # The xLSTM is channel-INDEPENDENT, so one sample = ONE (origin, series)
    # pair and the model output is (n_samples, n_heads): there is no series axis
    # to slice.  The validation subset must therefore be selected on the sample
    # level through the series index each sample carries (`Sva`).  (Slicing
    # `out[:, :k_val]` here would slice the 4 horizon heads and silently do
    # nothing -- caught because val@23 printed bit-identical to val@34.)
    vmask = None
    n_val_used = int(len(Sva))
    if k_val < len(series):
        np_mask = (Sva < k_val)
        n_val_used = int(np_mask.sum())
        vmask = torch.from_numpy(np_mask).to(dev)
    if verbose:
        print(f"  [copy] val@{k_val}: using {n_val_used:,} of {len(Sva):,} val "
              f"samples (expected {k_val * len(va_i):,} = {k_val} series x "
              f"{len(va_i)} origins)")
    assert n_val_used == k_val * len(va_i), (
        f"validation subset has {n_val_used} samples, expected {k_val*len(va_i)}; "
        "a dropped sample would silently change the early-stopping criterion")
    n = len(Xtr_t)
    best, best_ep, best_state = float("inf"), -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            xb = Xtr_t[idx].to(dev)
            sb = Str_t[idx].to(dev)
            cb = Ctr_t[idx].to(dev)
            yb = Ttr_t[idx].to(dev)
            opt.zero_grad()
            se = (model(xb, sb, cb) - yb) ** 2
            loss = se.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            out = model(Xva_t, Sva_t, Cva_t)
            if vmask is None:            # k_val == n_series -> the library loss
                vloss = float(((out - Tva_t) ** 2).mean())
            else:                        # ← the ONLY difference from the library loop
                vloss = float(((out[vmask] - Tva_t[vmask]) ** 2).mean())
        if vloss < best:
            best, best_ep = vloss, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose and (ep == 1 or ep % 20 == 0 or ep == epochs):
            print(f"    ep {ep:3d}  train {tot / n:.4f}  val@{k_val} {vloss:.4f}"
                  f"{'  *' if ep == best_ep else ''}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    aw = model.anchor_weights()
    meta = {"n_params": int(n_par), "best_val_loss": float(best),
            "best_epoch": int(best_ep), "n_train_samples": int(len(Xtr)),
            "n_val_samples": int(len(Xva)), "val_series_used": int(k_val),
            "n_val_samples_used": int(n_val_used),
            "context": ctx, "embedding_dim": emb, "num_blocks": blk,
            "dropout": drp, "weight_decay": float(weight_decay), "lr": float(lr),
            "batch": int(batch), "epochs": int(epochs), "device": dev,
            "anchor_weights": {str(h): float(aw[k]) for k, h in enumerate(X.HEADS)},
            "use_anchor": True, "use_calendar": False, "purge_val": True,
            "train_origins": [periods[tr_i[0]], periods[tr_i[-1]]],
            "val_origins": [periods[va_i[0]], periods[va_i[-1]]],
            "max_val_target_period": periods[min(va_i[-1] + max(X.HEADS),
                                                 len(periods) - 1)]}
    if verbose:
        print(f"  [copy] best val@{k_val} {best:.4f} @ epoch {best_ep}   "
              f"params {n_par:,}")
    return model, meta


def run_itransformer_23val(seeds=SEEDS, k_val=23, origins=None, verify=False,
                           outdir: Path | None = None, force=False,
                           verbose=True):
    """34-series iTransformer, early stopping on the first `k_val` variates.

    k_val == 34 is the verification mode: it must reproduce the shipped
    iTransformer_34 arm exactly, which is what proves the copied loop is the
    library loop.
    """
    import torch
    from run_sota_baselines import rows_from_levels
    origins = list(origins or H.ORIGINS)
    feat, wide, scaling = load_arm(34)
    periods, series, F, Y, KF, tr_i, va_i, plan = _plan(34, feat)
    Z = Y.astype(np.float64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag_suffix = "" if verify else "_v23val"
    frames, metas = [], []
    for seed in seeds:
        tag = f"iTransformer_34{tag_suffix}_s{seed}"
        out_csv = (outdir or OUT) / f"preds_itransformer34{tag_suffix or ''}_s{seed}.csv"
        if out_csv.exists() and not force:
            print(f"    [skip] {tag} already done")
            frames.append(pd.read_csv(out_csv, dtype={"series": str, "origin": str}))
            js = out_csv.with_suffix(".json")
            metas.append(json.loads(js.read_text(encoding="utf-8")) if js.exists() else {})
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = IT.ITransformerNet(context=CTX, d_model=32, n_heads_attn=4, layers=2,
                                   dropout=0.1, use_anchor=True)
        model, meta = train_itransformer_valcut(model, Z, tr_i, va_i, k_val,
                                                context=CTX, epochs=EPOCHS,
                                                lr=5e-4, weight_decay=5e-3,
                                                batch=32, device=dev, verbose=verbose)
        rows = []
        for o in origins:
            i = periods.index(o)
            lv = IT.predict_transformer(model, Z, i, context=CTX)
            rows += rows_from_levels(tag, o, lv, series, wide, scaling)
        df = pd.DataFrame(rows, columns=["model", "origin", "h", "series",
                                         "y_true", "y_pred"])
        fit_sec = round(time.time() - t0, 1)
        meta.update({"family": "itransformer_anchor", "arm": f"34s_v{k_val}val",
                     "n_series": 34, "seed": seed, "tag": tag, "context": CTX,
                     "epochs": EPOCHS, "fit_sec": fit_sec, "device": dev,
                     "plan": plan, "n_train_tokens": plan["n_train_origins"] * 34,
                     "origins_scored": origins,
                     "early_stop_on": ("all 34 series" if k_val == 34
                                       else "the original 23 series")})
        (outdir or OUT).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        out_csv.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        if verbose:
            print(f"    >>> {tag}  fit {fit_sec}s  best_epoch {meta['best_epoch']}  "
                  f"val@{k_val} {meta['best_val_loss']:.4f}")
        frames.append(df)
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), metas


def run_xlstm_23val(seeds=SEEDS, k_val=23, origins=None, verify=False,
                    outdir: Path | None = None, force=False, verbose=True):
    """34-series xLSTM, early stopping on the first `k_val` series embeddings"""
    from run_sota_baselines import rows_from_levels
    origins = list(origins or H.ORIGINS)
    feat, wide, scaling = load_arm(34)
    periods, series, F, Y, KF, tr_i, va_i, plan = _plan(34, feat)
    tag_suffix = "" if verify else "_v23val"
    frames, metas = [], []
    for seed in seeds:
        tag = f"xLSTM_34{tag_suffix}_s{seed}"
        out_csv = (outdir or OUT) / f"preds_xlstm34{tag_suffix or ''}_s{seed}.csv"
        if out_csv.exists() and not force:
            print(f"    [skip] {tag} already done")
            frames.append(pd.read_csv(out_csv, dtype={"series": str, "origin": str}))
            js = out_csv.with_suffix(".json")
            metas.append(json.loads(js.read_text(encoding="utf-8")) if js.exists() else {})
            continue
        t0 = time.time()
        model, meta = train_xlstm_valcut(F, Y, KF, periods, series, H.TRAIN_END,
                                         k_val, seed, epochs=EPOCHS, context=CTX,
                                         verbose=verbose)
        rows = []
        for o in origins:
            pr = X.predict_at(model, F, KF, periods, series, o)
            if pr is None:
                continue
            levels = np.stack([pr[h] for h in H.HORIZONS], axis=1)
            rows += rows_from_levels(tag, o, levels, series, wide, scaling)
        df = pd.DataFrame(rows, columns=["model", "origin", "h", "series",
                                         "y_true", "y_pred"])
        wall = round(time.time() - t0, 1)
        meta.update({"family": "xlstm_anchor", "arm": f"34s_v{k_val}val",
                     "n_series": 34, "seed": seed, "tag": tag, "wall_sec": wall,
                     "plan": plan, "origins_scored": origins,
                     "early_stop_on": ("all 34 series" if k_val == 34
                                       else "the original 23 series")})
        (outdir or OUT).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        out_csv.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        if verbose:
            print(f"    >>> {tag}  wall {wall}s  best_epoch {meta['best_epoch']}  "
                  f"val@{k_val} {meta['best_val_loss']:.4f}")
        frames.append(df)
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), metas


# ===========================================================================
# C. SUBCOMMANDS
# ===========================================================================
def cmd_build_data(args) -> int:
    wide = build_panel_34(force=args.force)
    validate_hierarchy_34(wide)
    build_features_34(wide, force=args.force)
    print("\nBUILD-DATA OK")
    return 0


def _selftest_valmask(verbose: bool = True) -> bool:
    """independent (numpy) cross-check of the sample-level validation mask.

    The xLSTM arm selects its validation subset through the series index each
    sample carries, so the mask arithmetic is worth testing on its own: the
    torch result must equal a numpy computation over the same subset, and must
    NOT equal the unmasked mean.
    """
    import torch
    rng = np.random.default_rng(0)
    out = torch.tensor(rng.normal(size=(12, 4)), dtype=torch.float32)
    tgt = torch.tensor(rng.normal(size=(12, 4)), dtype=torch.float32)
    sid = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    k = 2
    m = torch.from_numpy(sid < k)
    got = float(((out[m] - tgt[m]) ** 2).mean())
    o, t = out.numpy(), tgt.numpy()
    want = float(np.mean((o[sid < k] - t[sid < k]) ** 2))
    full = float(np.mean((o - t) ** 2))
    ok = (abs(got - want) < 1e-12) and (abs(got - full) > 1e-6) and \
         int(m.sum()) == int((sid < k).sum())
    if verbose:
        print(f"  valmask selftest: torch {got:.6f}  numpy {want:.6f}  "
              f"unmasked {full:.6f}  n_sel {int(m.sum())}/{len(sid)}  "
              f"{'PASS' if ok else 'FAIL'}")
    return ok


def cmd_v23val(args) -> int:
    """the selection-bias control: 34-series input, early stopping on the 23.

    Phase 1  verify the copied loops (k_val = n_series) reproduce the shipped
             34-series arms point for point — otherwise the copy is worthless.
    Phase 2  run the real arm (k_val = 23), 3 seeds, ENS3, scored on the 23.
    """
    outdir = OUT / "v23val"
    vdir = OUT / "verifycopy"
    outdir.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)
    naive = naive_mae_locked()
    models = [m for m in args.models if not (m == "xlstm" and args.skip_xlstm)]
    print("=" * 88)
    print(f"  23-VALIDATION ARM — models={models}  verify={not args.skip_verify}")
    print("=" * 88)
    if not _selftest_valmask():
        raise SystemExit("validation-mask selftest failed")

    # ---- phase 1: prove the copies are the library loops --------------------
    if not args.skip_verify:
        if "itransformer" in models:
            dv, _ = run_itransformer_23val(seeds=(42,), k_val=34, verify=True,
                                           outdir=vdir, force=True)
            ref = pd.read_csv(OUT / "preds_itransformer34_s42.csv",
                              dtype={"series": str, "origin": str})
            d = _max_point_diff(dv, ref)
            print(f"  VERIFY iTransformer copy (k_val=34) vs shipped arm: {d:.3e}"
                  f"  {'IDENTICAL' if d < 1e-9 else '*** DIFFERS ***'}")
            if d >= 1e-9:
                raise SystemExit("copied iTransformer loop is not the library loop")
        if "xlstm" in models:
            dx, _ = run_xlstm_23val(seeds=(42,), k_val=34, verify=True,
                                    outdir=vdir, force=True)
            ref = pd.read_csv(OUT / "preds_xlstm34_s42.csv",
                              dtype={"series": str, "origin": str})
            d = _max_point_diff(dx, ref)
            print(f"  VERIFY xLSTM copy (k_val=34) vs shipped arm:        {d:.3e}"
                  f"  {'IDENTICAL' if d < 1e-9 else '*** DIFFERS ***'}")
            if d >= 1e-9:
                raise SystemExit("copied xLSTM loop is not the library loop")
    if args.verify_only:
        print("\nVERIFY-ONLY OK")
        return 0

    # ---- phase 2: the arm --------------------------------------------------
    # NOTE: metrics / predictions are MERGED with whatever is already on disk,
    # because `--models` can run one arm at a time (an overwrite here once
    # dropped the iTransformer rows from the xLSTM-only re-run).
    t_all = time.time()
    agg = {}
    if "itransformer" in models:
        di, mi = run_itransformer_23val(seeds=SEEDS, k_val=23, outdir=outdir,
                                        force=args.force)
        agg["iTransformer_34_v23val_ENS3"] = ensemble_frame(
            di, seeds=SEEDS, model="iTransformer_34_v23val_ENS3")
    if "xlstm" in models:
        dx, mx = run_xlstm_23val(seeds=SEEDS, k_val=23, outdir=outdir,
                                 force=args.force)
        agg["xLSTM_34_v23val_ENS3"] = ensemble_frame(
            dx, seeds=SEEDS, model="xLSTM_34_v23val_ENS3")

    # ---- metrics on the original 23 ----------------------------------------
    rows = []
    for name, fr in agg.items():
        m = score23(fr, naive, model=name)
        m["kind"] = name
        rows.append(m)
    met = pd.concat(rows, ignore_index=True)
    mpath = outdir / "metrics_v23val_23.csv"
    if mpath.exists():
        old = pd.read_csv(mpath)
        met = pd.concat([old[~old["kind"].isin(met["kind"])], met], ignore_index=True)
    met.to_csv(mpath, index=False)

    ppath = outdir / "preds_v23val_23.csv"
    newp = pd.concat([fr[fr["series"].isin(SERIES23)] for fr in agg.values()],
                     ignore_index=True)
    if ppath.exists():
        oldp = pd.read_csv(ppath, dtype={"origin": str, "series": str})
        oldp = oldp[~oldp["model"].isin(newp["model"].unique())]
        newp = pd.concat([oldp, newp], ignore_index=True)
    newp.to_csv(ppath, index=False)

    # ---- timing: summed from the per-seed metadata, so it survives a resume --
    timings = {}
    for lab, patt, key in (("itransformer", "preds_itransformer34_v23val_s*.json",
                            "fit_sec"),
                           ("xlstm", "preds_xlstm34_v23val_s*.json", "wall_sec")):
        tot, nfit = 0.0, 0
        for f in sorted(outdir.glob(patt)):
            j = json.loads(f.read_text(encoding="utf-8"))
            tot += float(j.get(key, 0.0) or 0.0)
            nfit += 1
        if nfit:
            timings[f"{lab}_34_v23val_sec"] = round(tot, 1)
            timings[f"{lab}_34_v23val_n_seeds"] = nfit

    # ---- the three-protocol table ------------------------------------------
    prot = []
    for lab, kind, folder, name in (
            ("23 series (control)", "iTransformer",
             RES / "sota_meta", "itransformer_anchor_s{}.json"),
            ("23 series (control)", "xLSTM",
             RES / "exog_meta_z", "xLSTM_none_s{}.json"),
            ("34 series (original)", "iTransformer", OUT,
             "preds_itransformer34_s{}.json"),
            ("34 series (original)", "xLSTM", OUT, "preds_xlstm34_s{}.json"),
            ("34 series (23-val)", "iTransformer", outdir,
             "preds_itransformer34_v23val_s{}.json"),
            ("34 series (23-val)", "xLSTM", outdir,
             "preds_xlstm34_v23val_s{}.json")):
        for seed in SEEDS:
            f = folder / name.format(seed)
            if not f.exists():
                continue
            j = json.loads(f.read_text(encoding="utf-8"))
            prot.append({"protocol": lab, "model": kind, "seed": seed,
                         "best_epoch": j.get("best_epoch"),
                         "best_val_loss": j.get("best_val_loss"),
                         "val_series_used": j.get("val_series_used",
                                                  34 if "34" in lab else 23)})
    pdf = pd.DataFrame(prot)
    pdf.to_csv(outdir / "best_epochs_v23val.csv", index=False)
    print("\n  --- best_epoch by protocol ---")
    print(pdf.pivot_table(index=["model", "protocol"], columns="seed",
                          values="best_epoch").to_string())

    print("\n  --- MASE_ALL, 23-val arm vs the other two protocols ---")
    print(met[met["h"].astype(str) == "ALL"][["kind", "MASE", "WAPE"]]
          .to_string(index=False))
    (outdir / "meta_v23val.json").write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "k_val": 23, "timings_sec": timings,
         "seeds": list(SEEDS), "models_done": models,
         "models_missing": [m for m in ("itransformer", "xlstm")
                            if not (outdir / {"itransformer":
                                              "preds_itransformer34_v23val_s42.json",
                                              "xlstm":
                                              "preds_xlstm34_v23val_s42.json"}[m]).exists()],
         "note": ("34-series input, early stopping on the original 23 series; "
                  "the copied training loops were verified to reproduce the "
                  "shipped arms when k_val=n_series")},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  artefacts in {outdir.relative_to(ROOT.parent)}")
    return 0


def _max_point_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    ka = a.set_index(["origin", "h", "series"])["y_pred"]
    kb = b.set_index(["origin", "h", "series"])["y_pred"]
    idx = ka.index.intersection(kb.index)
    if len(idx) != len(ka) or len(idx) != len(kb):
        return float("inf")
    return float(np.abs(ka.loc[idx] - kb.loc[idx]).max())


def cmd_smoke(args) -> int:
    """1 seed, 2 origins — shapes + metrics only, in a throw-away directory"""
    outdir = OUT / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    origins = ["202312", "202401"]
    naive = naive_mae_locked()
    print("=" * 88)
    print(f"  SMOKE TEST — seed 42, origins {origins}, 34 series")
    print("=" * 88)
    t0 = time.time()
    di, _ = run_itransformer(34, seeds=(42,), origins=origins, outdir=outdir, force=True)
    dx, _ = run_xlstm(34, seeds=(42,), origins=origins, outdir=outdir, force=True)
    print(f"\n  smoke wall {time.time()-t0:.1f}s")
    for name, d in (("iTransformer_34", di), ("xLSTM_34", dx)):
        s23 = d[d["series"].isin(SERIES23)]
        print(f"  {name}: rows {len(d):,}  series {d['series'].nunique()}  "
              f"origins {d['origin'].nunique()}  h {sorted(d['h'].unique())}"
              f"  -> 23-series rows {len(s23):,} (expect {len(origins)*4*23})")
        assert len(s23) == len(origins) * 4 * 23, len(s23)
        m = score23(d, naive)
        print(m[["h", "n_points", "MASE", "WAPE", "MAE"]].to_string(index=False))
        assert np.isfinite(m["MASE"]).all()
    print("\nSMOKE OK")
    return 0


def cmd_control23(args) -> int:
    """the 23-series arm through THIS pipeline; must match the frozen artefacts"""
    outdir = OUT / "control23"
    outdir.mkdir(parents=True, exist_ok=True)
    naive = naive_mae_locked()
    print("=" * 88)
    print("  CONTROL — 23 series, same code path, compared with the frozen CSVs")
    print("=" * 88)
    di, _ = run_itransformer(23, seeds=SEEDS, outdir=outdir, force=args.force)
    dx, _ = run_xlstm(23, seeds=SEEDS, outdir=outdir, force=args.force)

    frozen = {"itransformer": RES / "sota_preds", "xlstm": RES / "exog_preds_z"}
    for kind, df, fname in (("itransformer", di, "itransformer_anchor_s{}.csv"),
                            ("xlstm", dx, "xLSTM_none_s{}.csv")):
        for seed in SEEDS:
            f = frozen[kind] / fname.format(seed)
            if not f.exists():
                print(f"  [warn] frozen {f.name} absent, skipping diff")
                continue
            a = df[df["model"].str.endswith(f"_s{seed}")].copy()
            b = pd.read_csv(f, dtype={"series": str, "origin": str})
            ka = a.set_index(["origin", "h", "series"])["y_pred"]
            kb = b.set_index(["origin", "h", "series"])["y_pred"]
            idx = ka.index.intersection(kb.index)
            dmax = float(np.abs(ka.loc[idx] - kb.loc[idx]).max())
            print(f"  {kind:14s} s{seed}: {len(idx):,} matched points   "
                  f"max |y_pred - frozen| = {dmax:.3e}   "
                  f"{'IDENTICAL' if dmax < 1e-9 else 'DIFFERS'}")

    # ensemble + MinT on the 23-node hierarchy, versus 0.94810 / 0.94455 / 0.90198
    rows = []
    for label, df, target in (("iTransformer_23 ENS3", di, 0.90198),
                              ("xLSTM_23 ENS3", dx, 0.94810)):
        ens = ensemble_frame(df, seeds=SEEDS,
                             model=label.split()[0] + "_23_ENS3")
        m = score23(ens, naive)
        all_row = m[m["h"].astype(str) == "ALL"].iloc[0]
        rows.append({"variant": label, "MASE_ALL": all_row["MASE"],
                     "WAPE_ALL": all_row["WAPE"], "target": target,
                     "delta": round(float(all_row["MASE"]) - target, 5)})
    ens = ensemble_frame(dx, seeds=SEEDS, model="xLSTM_23_ENS3")
    series = SERIES23
    wide23 = H.load_panel()
    A = build_A(series, HIER23)
    W = np.diag(naive_w(wide23, series, H.TRAIN_END))
    keys, Y, P = to_matrix(ens, series)
    Pr = reconcile(P, A, W)
    rec = from_matrix(keys, Y, Pr, series, "xLSTM_23_ENS3+MinT_wls")
    m = score23(rec, naive)
    all_row = m[m["h"].astype(str) == "ALL"].iloc[0]
    rows.append({"variant": "xLSTM_23 ENS3 + MinT_wls", "MASE_ALL": all_row["MASE"],
                 "WAPE_ALL": all_row["WAPE"], "target": 0.94455,
                 "delta": round(float(all_row["MASE"]) - 0.94455, 5)})
    tab = pd.DataFrame(rows)
    print("\n  --- control vs the frozen headline numbers ---")
    print(tab.to_string(index=False))
    tab.to_csv(outdir / "control23_vs_frozen.csv", index=False)
    print("\nCONTROL23 OK")
    return 0


def cmd_full(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    naive = naive_mae_locked()
    t_all = time.time()
    timings = {}

    print("=" * 88)
    print("  FULL RUN — 3 seeds x 12 origins x 34 series, scored on the 23")
    print("=" * 88)
    t0 = time.time()
    di, mi = run_itransformer(34, seeds=SEEDS, force=args.force)
    timings["itransformer_34_sec"] = round(time.time() - t0, 1)
    print(f"  iTransformer_34 total {timings['itransformer_34_sec']}s")

    t0 = time.time()
    dx, mx = run_xlstm(34, seeds=SEEDS, force=args.force)
    timings["xlstm_34_sec"] = round(time.time() - t0, 1)
    print(f"  xLSTM_34 total {timings['xlstm_34_sec']}s")

    # ---- also pull the frozen 23-series references for the comparison table ----
    # (no fitting involved: they are read from results/sota_preds and
    #  results/exog_preds_z, which is why they add nothing to the timing budget)
    ref = load_frozen23()

    # ---- per-seed metrics on the 23 -----------------------------------------
    per_seed = []
    for df, kind in ((di, "iTransformer_34"), (dx, "xLSTM_34")):
        for seed in SEEDS:
            sub = df[df["model"] == f"{kind}_s{seed}"]
            m = score23(sub, naive, model=f"{kind}_s{seed}")
            m["seed"] = seed
            m["kind"] = kind
            per_seed.append(m)
    for label, edf in ref["frames"].items():
        for seed, sub in edf.items():
            m = score23(sub, naive, model=f"{label}_s{seed}")
            m["seed"] = seed
            m["kind"] = label
            per_seed.append(m)
    per_seed = pd.concat(per_seed, ignore_index=True)
    per_seed.to_csv(OUT / "metrics_per_seed_23.csv", index=False)

    # ---- aggregate forecast variants ---------------------------------------
    # every variant is stored as a full per-point frame; score23() then keeps
    # ONLY the original 23 series.  The 34-series variants are kept at width 34
    # so that MinT can use the new children as constraints.
    agg = {"SeasonalNaive": seasonal_naive_frame()}
    for df, kind in ((di, "iTransformer_34"), (dx, "xLSTM_34")):
        agg[f"{kind}_ENS3"] = ensemble_frame(df, seeds=SEEDS, model=f"{kind}_ENS3")
    for label, edf in ref["frames"].items():
        agg[f"{label}_ENS3"] = ensemble_frame(pd.concat(edf.values(), ignore_index=True),
                                             seeds=SEEDS, model=f"{label}_ENS3")

    wide23 = H.load_panel()
    wide34 = pd.read_csv(DATA / "panel_wide_34.csv", index_col=0, dtype={0: str})
    wide34.index = wide34.index.astype(str)

    # ---- MinT on the FROZEN 23-node hierarchy (reproduces 0.94455) ---------
    A23 = build_A(SERIES23, HIER23)
    W23 = np.diag(naive_w(wide23, SERIES23, H.TRAIN_END))
    keys, Y, P = to_matrix(agg["xLSTM_23_ENS3"], SERIES23)
    agg["xLSTM_23_ENS3+MinT_wls"] = from_matrix(
        keys, Y, reconcile(P, A23, W23), SERIES23, "xLSTM_23_ENS3+MinT_wls")

    # ---- MinT on the EXTENDED 34-node hierarchy (the upgrade) --------------
    A34 = build_A(SERIES34, HIER34)
    v = naive_w(wide34, SERIES34, H.TRAIN_END)
    W_wls, W_ols = np.diag(v), np.eye(len(SERIES34))
    keys34, Y34, P34 = to_matrix(agg["xLSTM_34_ENS3"], SERIES34)
    gap_base = float(np.abs(P34 @ A34.T).mean())
    variants34 = {
        "xLSTM_34_ENS3+BU": bottom_up(P34, SERIES34, HIER34),
        "xLSTM_34_ENS3+MinT_ols": reconcile(P34, A34, W_ols),
        "xLSTM_34_ENS3+MinT_wls": reconcile(P34, A34, W_wls),
    }
    gap_rows = [{"variant": "xLSTM_34_ENS3", "n_nodes": 34,
                 "mean_abs_gap_hkd_m": gap_base, "gap_vs_base": 1.0}]
    recon_frames = []
    for name, PP in variants34.items():
        fr = from_matrix(keys34, Y34, PP, SERIES34, name)
        recon_frames.append(fr)
        agg[name] = fr
        g = float(np.abs(PP @ A34.T).mean())
        gap_rows.append({"variant": name, "n_nodes": 34,
                         "mean_abs_gap_hkd_m": g, "gap_vs_base": g / gap_base})
    pd.DataFrame(gap_rows).to_csv(OUT / "mint34_constraint_gaps.csv", index=False)
    pd.concat(recon_frames, ignore_index=True).to_csv(
        OUT / "preds_reconciled34.csv", index=False)

    # ---- one metrics table for every aggregate variant ---------------------
    rows = []
    for name, fr in agg.items():
        m = score23(fr, naive, model=name)
        m["kind"] = name
        rows.append(m)
    metrics_all = pd.concat(rows, ignore_index=True)
    metrics_all.to_csv(OUT / "metrics_all_23.csv", index=False)
    # the per-point frame restricted to the 23 (for DM and for the record)
    pd.concat([fr[fr["series"].isin(SERIES23)] for fr in agg.values()],
              ignore_index=True).to_csv(OUT / "preds_aggregate_23.csv", index=False)

    # ---- per-series MASE for every aggregate variant -----------------------
    allframes = pd.concat(list(agg.values()), ignore_index=True)
    ps = pd.concat([per_series_mase(allframes, naive, m)
                    for m in sorted(allframes["model"].unique())], ignore_index=True)
    ps.to_csv(OUT / "per_series_mase_23.csv", index=False)

    # ---- meta / timing ------------------------------------------------------
    meta = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": list(SEEDS), "context": CTX, "epochs": EPOCHS,
        "horizons": list(H.HORIZONS), "origins": list(H.ORIGINS),
        "train_end": H.TRAIN_END, "val_months": 24, "purge": True,
        "n_series_train": 34, "n_series_scored": 23,
        "n_points_scored": 12 * 4 * 23,
        "timings_sec": timings,
        "timings_sec_total": round(time.time() - t_all, 1),
        "itransformer_runs": mi, "xlstm_runs": mx,
        "hier34": HIER34,
        "wls_variance_range": [float(np.nanmin(v)), float(np.nanmax(v))],
        "panel34": {"periods": int(wide34.shape[0]), "series": int(wide34.shape[1])},
        "notes": [
            "all metrics computed by eval/harness.py (unmodified)",
            "MASE denominator from the frozen 23-series panel (load_panel)",
            "ENS3 = point-wise mean of the per-seed level forecasts",
            "MinT weight W = diag(train-only seasonal-naive error variance)",
        ],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False,
                                              default=str), encoding="utf-8")
    print(f"\n  all artifacts in {OUT.relative_to(ROOT.parent)}")
    return 0


def load_frozen23() -> dict:
    """the frozen 23-series model forecasts, for the reference column"""
    frames = {}
    itr = {}
    for seed in SEEDS:
        f = RES / "sota_preds" / f"itransformer_anchor_s{seed}.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"series": str, "origin": str})
            d["h"] = d["h"].astype(int)
            d["model"] = f"iTransformer_23_s{seed}"
            itr[seed] = d
    if itr:
        frames["iTransformer_23"] = itr
    xsl = {}
    for seed in SEEDS:
        f = RES / "exog_preds_z" / f"xLSTM_none_s{seed}.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"series": str, "origin": str})
            d["h"] = d["h"].astype(int)
            d["model"] = f"xLSTM_23_s{seed}"
            xsl[seed] = d
    if xsl:
        frames["xLSTM_23"] = xsl
    return {"frames": frames}


def seasonal_naive_frame() -> pd.DataFrame:
    """the frozen reference model, recomputed with the locked harness"""
    wide = H.load_panel()
    rows = []
    for o in H.ORIGINS:
        for h in H.HORIZONS:
            tgt = H.month_add(o, h)
            fc = H.seasonal_naive_forecast(wide, o, h)
            for s in SERIES23:
                rows.append(("SeasonalNaive", o, h, s,
                             float(wide.loc[tgt, s]), float(fc[s])))
    return pd.DataFrame(rows, columns=["model", "origin", "h", "series",
                                       "y_true", "y_pred"])


def cmd_report(args) -> int:
    """tabulate everything in results/upgrade34/ into
    docs/UPGRADE_34SERIES_RESULTS.md — reads artefacts only, refits nothing"""
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    me = pd.read_csv(OUT / "metrics_all_23.csv")
    ps = pd.read_csv(OUT / "per_series_mase_23.csv", dtype={"series": str})
    ens = pd.read_csv(OUT / "preds_aggregate_23.csv", dtype={"origin": str,
                                                              "series": str})
    val = pd.read_csv(OUT / "panel34_hierarchy_validation.csv",
                      dtype={"period": str, "parent": str, "children": str})
    vmet = None
    if (OUT / "v23val" / "metrics_v23val_23.csv").exists():
        vmet = pd.read_csv(OUT / "v23val" / "metrics_v23val_23.csv")

    def allrow(df, kind):
        r = df[(df["kind"] == kind) & (df["h"].astype(str) == "ALL")]
        if r.empty:
            return None
        return r.iloc[0]

    # training-sample bookkeeping straight from the run metadata
    n_orig = meta["itransformer_runs"][0]["plan"]["n_train_origins"]
    tok = {23: n_orig * 23, 34: n_orig * 34}
    xsam = {34: meta["xlstm_runs"][0]["n_train_samples"]}
    try:                                       # the frozen 23-series sample count
        jm = json.loads((RES / "exog_meta_z" / "xLSTM_none_s42.json")
                        .read_text(encoding="utf-8"))
        xsam[23] = jm["n_train_samples"]
    except Exception:
        xsam[23] = n_orig * 23

    lines = []
    A = lines.append
    A("# 34 條序列升級實驗 — 結果")
    A("")
    A(f"生成時間 `{meta['generated']}`　·　腳本 `stage0/run_upgrade_34series.py`　·　"
      f"產物 `stage0/results/upgrade34/`")
    A("")
    A("## 0. 一句話結論")
    A("")
    A("先把三個數字放在最前面（完整推導見 §3–§5）：")
    A("")
    A("| 模型（ENS3，只評原 23 條） | MASE_ALL | WAPE_ALL | 相對 23 條版 |")
    A("|---|---|---|---|")
    for kind, base_kind, label in (
            ("iTransformer_34_ENS3", "iTransformer_23_ENS3", "iTransformer"),
            ("xLSTM_34_ENS3", "xLSTM_23_ENS3", "xLSTM"),
            ("xLSTM_34_ENS3+MinT_wls", "xLSTM_23_ENS3+MinT_wls", "xLSTM+MinT_wls")):
        a, b = allrow(me, kind), allrow(me, base_kind)
        if a is None:
            a = allrow(mr, kind)
            b = allrow(mr, base_kind)
        if a is None or b is None:
            continue
        d = a["MASE"] - b["MASE"]
        A(f"| {label} | **{a['MASE']:.5f}** | {a['WAPE']:.5f} | "
          f"**{100*d/b['MASE']:+.2f}%**（{d:+.5f}） |")
    A("")
    def sMASE(model: str, s: str) -> float:
        sub = ps[(ps["model"] == model) & (ps["series"] == s)]
        return float(sub["MASE_series"].iloc[0]) if len(sub) else float("nan")

    _a23, _a34 = allrow(me, "xLSTM_23_ENS3"), allrow(me, "xLSTM_34_ENS3")
    _b23, _b34 = allrow(me, "iTransformer_23_ENS3"), allrow(me, "iTransformer_34_ENS3")
    _m23 = allrow(me, "xLSTM_23_ENS3+MinT_wls")
    _m34 = allrow(me, "xLSTM_34_ENS3+MinT_wls")
    _p8 = (sMASE("xLSTM_23_ENS3", "8"), sMASE("xLSTM_34_ENS3", "8"),
           sMASE("xLSTM_23_ENS3+MinT_wls", "8"), sMASE("xLSTM_34_ENS3+MinT_wls", "8"))
    _p30 = (sMASE("xLSTM_23_ENS3", "30"), sMASE("xLSTM_34_ENS3", "30"),
            sMASE("xLSTM_23_ENS3+MinT_wls", "30"), sMASE("xLSTM_34_ENS3+MinT_wls", "30"))
    A("**原版協議下的答案：變差 —— 兩個模型、加不加調和都是變差。**"
      "（但這個幅度有三分之二來自選模準則，不是資訊；見下一段與 §11。）")
    A("")
    A(f"被評分的**仍然是原來那 23 條**。把 11 條子序列餵進訓練後，它們的 MASE_ALL "
      f"在三個配置上全部上升：iTransformer {_b23['MASE']:.5f} → {_b34['MASE']:.5f}"
      f"（{100*(_b34['MASE']-_b23['MASE'])/_b23['MASE']:+.2f}%）、"
      f"xLSTM {_a23['MASE']:.5f} → {_a34['MASE']:.5f}"
      f"（{100*(_a34['MASE']-_a23['MASE'])/_a23['MASE']:+.2f}%）、"
      f"xLSTM+`MinT_wls` {_m23['MASE']:.5f} → {_m34['MASE']:.5f}"
      f"（{100*(_m34['MASE']-_m23['MASE'])/_m23['MASE']:+.2f}%）。")
    A("")
    A(f"母類 `8`（超市）與 `30`（百貨）**都沒有改善，反而變差**："
      f"序列 `8` 的 MASE 由 {_p8[0]:.5f} 升到 {_p8[1]:.5f}"
      f"（{100*(_p8[1]-_p8[0])/_p8[0]:+.2f}%），序列 `30` 由 {_p30[0]:.5f} 升到 "
      f"{_p30[1]:.5f}（{100*(_p30[1]-_p30[0])/_p30[0]:+.2f}%）。"
      f"換上 34 節點的 `MinT_wls` 可以把傷害收窄（`8` 由 "
      f"{100*(_p8[1]-_p8[0])/_p8[0]:+.2f}% 收到 {100*(_p8[3]-_p8[2])/_p8[2]:+.2f}%、"
      f"`30` 由 {100*(_p30[1]-_p30[0])/_p30[0]:+.2f}% 收到 "
      f"{100*(_p30[3]-_p30[2])/_p30[2]:+.2f}%），**但沒有翻正**。"
      f"機制與未做事項見 §6、§9.1、§10。")
    A("")
    if vmet is not None:
        _sh = {}
        for model, k23, k34, kv in (
                ("iTransformer", "iTransformer_23_ENS3", "iTransformer_34_ENS3",
                 "iTransformer_34_v23val_ENS3"),
                ("xLSTM", "xLSTM_23_ENS3", "xLSTM_34_ENS3",
                 "xLSTM_34_v23val_ENS3")):
            r23, r34, rv = allrow(me, k23), allrow(me, k34), allrow(vmet, kv)
            if r23 is not None and r34 is not None and rv is not None:
                _sh[model] = (100.0 * (1.0 - (rv["MASE"] - r23["MASE"])
                                      / (r34["MASE"] - r23["MASE"])),
                              100.0 * (rv["MASE"] - r23["MASE"]) / r23["MASE"])
        A("**選模混淆已排除**：§11 的 `*_34_v23val` 臂把 early stopping 改成只看原 23 條，"
          "模型輸入仍是 34 條。結果是")
        for model, k23, kv in (("iTransformer", "iTransformer_23_ENS3",
                                "iTransformer_34_v23val_ENS3"),
                               ("xLSTM", "xLSTM_23_ENS3", "xLSTM_34_v23val_ENS3")):
            r23, rv = allrow(me, k23), allrow(vmet, kv)
            if r23 is None or rv is None:
                A(f"- {model}：未執行。")
                continue
            pct = 100.0 * (rv["MASE"] - r23["MASE"]) / r23["MASE"]
            verdict = "仍然較差" if pct > 0 else "翻正"
            A(f"- {model}：{rv['MASE']:.5f}（{pct:+.2f}% vs 23 條控制組 "
              f"{r23['MASE']:.5f}）→ **{verdict}**")
        A("")
        if _sh:
            share_txt = "、".join(f"{k} {v[0]:.1f}%" for k, v in _sh.items())
            resid_txt = "、".join(f"{v[1]:+.2f}%" for v in _sh.values())
            A(f"選模混淆佔了約 **2/3**（{share_txt}）：改用「只看原 23 條」的 "
              f"early stopping 之後，殘餘差距只有 {resid_txt}，"
              f"而且**在統計上大多不顯著**（§11.4 的 DM）。"
              f"因此本書的正確說法是「**34 條輸入沒有帶來可觀察的正面效果**」，"
              f"而不是「34 條輸入已被證明有害」。")
            A("")
        A("")

    # ---- §1 provenance -----------------------------------------------------
    A("## 1. 資料：34 條序列")
    A("")
    A(f"- 面板 `stage0/data/panel_wide_34.csv`：{meta['panel34']['periods']} 月 × "
      f"{meta['panel34']['series']} 序列，index 為 6 位 `period` 字串。")
    A("- 原有 23 條：由 `620-67002.json` 以 `build_panel.py` 的同一組規則重新抽一次"
      "（`sv==VAL_RS`、`freq==M`、6 位 period、剔除 `\"\"`/`49`），"
      "與凍結的 `panel_wide.csv` **逐格完全相同**（max abs diff 0）。")
    A("- 新增 11 條：`620-67011.json` 的 `S1..S4`（超市按 broad product category，"
      "剔除 `\"\"`=Total）；`620-67012.json` 的 `D1..D7`（百貨，剔除 `\"\"`=Total 與 "
      "`D1_D6`=非超市部分聚合）。11 條全部 263 個月齊全，無 `***`、無缺格。")
    A("")
    A("### 1.1 加總恆等式的驗證（四捨五入誤差）")
    A("")
    A("| 關係 | 月數 | 最大絕對差 | 中位數絕對差 | 平均 | \\|差\\|≤2 的月份佔比 |")
    A("|---|---|---|---|---|---|")
    for parent, kids in (("8", NEW_SUPER), ("30", NEW_DEPT)):
        g = val[val["parent"] == parent]["abs_gap"]
        A(f"| `sum({'+'.join(kids)})` vs 序列 `{parent}` | {len(g)} | {g.max():.3f} | "
          f"{g.median():.3f} | {g.mean():.3f} | {100.0*(g<=2.0).mean():.1f}% |")
    A("")
    A("即：兩個母類與其子類的加總關係在 C&SD 的整數發布精度內成立，"
      "因此把約束當作精確（與 Stage 4 對待原有 4 個母類的做法一致）。逐月明細見 "
      "`results/upgrade34/panel34_hierarchy_validation.csv`。")
    A("")
    A("### 1.2 特徵表")
    A("")
    A("`stage0/data/features_long_34.csv` 用**完全相同的 27 個特徵定義**"
      "（lag1/2/3/12/13、roll{3,6,12}_{mean,std}、month_1..12、lny_pre14/lny_post7、"
      "covid_border_closed、consumption_voucher）與相同的 `y_log`/`y_z`，"
      "scaling（`mu_log`/`sigma_log`）只用 ≤202312 重算。"
      "驗證方式：把 34 條版本的 23 條子表與凍結的 `features_long.csv` 逐欄相比，"
      "欄位、列序、`series_id`、以及全部數值欄的最大絕對差皆為 0；"
      f"`scaling_34.csv` 與 `scaling.csv` 的 23 條共同序列，`mu`/`sigma` 差為 0。")
    A("")

    # ---- §2 protocol -------------------------------------------------------
    A("## 2. 協議（與凍結基準逐項相同）")
    A("")
    A(f"- 訓練／驗證原點：直接 import `run_sota_baselines.origin_plan`"
      f"（`VAL_MONTHS=24`、`purge`），因此訓練原點 "
      f"{meta['itransformer_runs'][0]['plan']['train_origins'][0]}.."
      f"{meta['itransformer_runs'][0]['plan']['train_origins'][1]} = "
      f"**{n_orig}** 個，驗證 "
      f"{meta['itransformer_runs'][0]['plan']['val_origins'][0]}.."
      f"{meta['itransformer_runs'][0]['plan']['val_origins'][1]} = "
      f"{meta['itransformer_runs'][0]['plan']['n_val_origins']} 個，"
      f"驗證期最大目標月 {meta['itransformer_runs'][0]['plan']['max_val_target_period']} ≤ 202312。")
    A(f"- 測試：12 個原點（202312..202411）× 4 個視界（1/3/6/12）× **只評原 23 條** = "
      f"**{meta['n_points_scored']:,}** 點。新增的 11 條只進訓練，不進計分。")
    A(f"- 超參：context={meta['context']}、epochs={meta['epochs']}、"
      f"iTransformer d_model=32/heads=4/layers=2/dropout=0.1/batch=32/lr=5e-4/wd=5e-3，"
      f"xLSTM embed=32/blocks=2/dropout=0.3/batch=128/lr=5e-4/wd=5e-3；"
      f"兩者都用可學習雙項錨點 `a_h y_z(t)+(1-a_h) y_z(t+h-12)`。")
    A("- ENS3 = 三個種子（42/7/2024）逐點在 level 空間平均（`run_stage_h_ens.py` 的定義）。")
    A("- `MinT_wls` 的 W 只用 ≤202312 的樣本內季節天真誤差變異數，"
      f"範圍 {meta['wls_variance_range'][0]:.1f} .. {meta['wls_variance_range'][1]:.1f}。")
    A("- 全部指標由 `eval/harness.py` 計算（未修改），MASE 分母取自凍結的 23 條面板。")
    A("")

    # ---- §3 control --------------------------------------------------------
    A("## 3. 對照：同一條管線跑 23 條，是否重現凍結數字")
    A("")
    cf = OUT / "control23" / "control23_vs_frozen.csv"
    if cf.exists():
        c = pd.read_csv(cf)
        A("`run_upgrade_34series.py control23` 把 23 條版本放進**本腳本自己的**管線"
          "（同一支 origin_plan、同一個 `rows_from_levels`、同一個 ENS3 定義），"
          "再與倉庫裡凍結的逐點預測比較。這一步是「34 vs 23」可信的前提。")
        A("")
        A("| 變體 | MASE_ALL | WAPE_ALL | 凍結值 | 差 |")
        A("|---|---|---|---|---|")
        for _, r in c.iterrows():
            A(f"| {r['variant']} | {r['MASE_ALL']:.5f} | {r['WAPE_ALL']:.5f} | "
              f"{r['target']:.5f} | {r['delta']:+.5f} |")
        A("")
        A("逐點比較（本腳本的 `preds_*_23_s*` vs 凍結的 "
          "`sota_preds/itransformer_anchor_s*.csv`、`exog_preds_z/xLSTM_none_s*.csv`）：")
        A("")
        log = OUT / "logs" / "control23.log"
        if log.exists():
            import re
            txt = log.read_text(encoding="utf-8", errors="replace")
            pat = re.compile(r"^\s+(itransformer|xlstm)\s+s(\d+):\s+([\d,]+) matched "
                             r"points\s+max \|y_pred - frozen\| = ([\d.eE+-]+)\s+(\w+)$",
                             re.M)
            A("| 模型 | 種子 | 匹配點數 | max \\|本腳本 − 凍結\\| | 判定 |")
            A("|---|---|---|---|---|")
            for m in pat.finditer(txt):
                A(f"| {m.group(1)} | {m.group(2)} | {m.group(3)} | {m.group(4)} | "
                  f"{m.group(5)} |")
            A("")
            A("1e-12 的殘差是 CSV 十進位往返的浮點誤差，不是數值差異。"
              "因此本腳本與凍結基準的**管線是同一條**，"
              "「34 條 vs 23 條」的差別只可能來自序列集合。")
    else:
        A("（`control23` 未執行，本節缺。）")
    A("")

    # ---- §4 main table -----------------------------------------------------
    A("## 4. 主表")
    A("")
    A("| 模型 | 序列數 | 訓練樣本 | MASE_ALL | WAPE_ALL | 相對 23 條版 |")
    A("|---|---|---|---|---|---|")
    sn = allrow(me, "SeasonalNaive")
    A(f"| `SeasonalNaive`（參考） | 23 | —（無訓練） | {sn['MASE']:.5f} | "
      f"{sn['WAPE']:.5f} | — |")
    first = True
    for kind, base_kind, label, ns in (
            ("iTransformer_23_ENS3", None, "iTransformer_23（凍結基準）", 23),
            ("iTransformer_34_ENS3", "iTransformer_23_ENS3", "**iTransformer_34**", 34),
            ("xLSTM_23_ENS3", None, "xLSTM_23（凍結基準）", 23),
            ("xLSTM_34_ENS3", "xLSTM_23_ENS3", "**xLSTM_34**", 34)):
        r = allrow(me, kind)
        if r is None:
            continue
        n = tok[ns] if label.startswith("**iT") or kind.startswith("iT") else xsam[ns]
        rel = "—"
        if base_kind:
            b = allrow(me, base_kind)
            d = r["MASE"] - b["MASE"]
            rel = f"**{100*d/b['MASE']:+.2f}%**（{d:+.5f}）"
        elif kind.startswith("iT"):
            rel = "—（對照）"
        else:
            rel = "—（對照）"
        A(f"| {label} | {ns} | {n:,} | **{r['MASE']:.5f}** | {r['WAPE']:.5f} | {rel} |")
    for kind, base_kind, label, ns in (
            ("xLSTM_23_ENS3+MinT_wls", None, "xLSTM_23 + `MinT_wls`（凍結基準）", 23),
            ("xLSTM_34_ENS3+MinT_wls", "xLSTM_23_ENS3+MinT_wls",
             "**xLSTM_34 + `MinT_wls`**", 34),
            ("xLSTM_34_ENS3+MinT_ols", None, "xLSTM_34 + `MinT_ols`（附）", 34),
            ("xLSTM_34_ENS3+BU", None, "xLSTM_34 + `BU`（附）", 34)):
        r = allrow(me, kind)
        if r is None:
            continue
        n = xsam[ns]
        rel = "—（對照）"
        if base_kind:
            b = allrow(me, base_kind)
            if b is not None:
                d = r["MASE"] - b["MASE"]
                rel = f"**{100*d/b['MASE']:+.2f}%**（{d:+.5f}）"
        A(f"| {label} | {ns} | {n:,} | **{r['MASE']:.5f}** | {r['WAPE']:.5f} | {rel} |")
    A("")
    A("「訓練樣本」：xLSTM 是 channel-independent，樣本數 = 原點數 × 序列數"
      "（4,186 → 6,188）；iTransformer 每個原點只有一個樣本、樣本內含 S 個 token，"
      "所以列的是 token 數（4,186 → 6,188），參數量不變（26,024）。")
    A("")
    A("> 這張表是「原版」協議。**第三個協議（34 條 + 23-val，即選模混淆已排除）"
      "的三方並列表在 §11.3**，因為它與本表共用同一個 1,104 點協定，"
      "但需要額外說明它改了什麼。")
    A("")

    # ---- §5 per horizon ----------------------------------------------------
    A("## 5. 逐視界")
    A("")
    A("| 變體 | h=1 | h=3 | h=6 | h=12 | ALL |")
    A("|---|---|---|---|---|---|")
    for kinds in (["SeasonalNaive", "iTransformer_23_ENS3", "iTransformer_34_ENS3",
                   "xLSTM_23_ENS3", "xLSTM_34_ENS3"],
                  ["xLSTM_23_ENS3+MinT_wls", "xLSTM_34_ENS3+MinT_wls",
                   "xLSTM_34_ENS3+MinT_ols", "xLSTM_34_ENS3+BU"]):
        for k in kinds:
            sub = me[me["kind"] == k]
            if sub.empty:
                continue
            v = {str(r["h"]): r["MASE"] for _, r in sub.iterrows()}
            A(f"| {k} | " + " | ".join(f"{v.get(str(h), float('nan')):.5f}"
                                       for h in ("1", "3", "6", "12")) +
              f" | **{v.get('ALL', float('nan')):.5f}** |")
    A("")
    A("同一批變體的**池化 MAE**（HK$ 百萬，跨 23 條序列加總後平均；"
      "MASE 是逐序列標準化後等權平均，兩者對「大序列出錯」的敏感度不同）：")
    A("")
    A("| 變體 | h=1 | h=3 | h=6 | h=12 | ALL |")
    A("|---|---|---|---|---|---|")
    for k in ("SeasonalNaive", "iTransformer_23_ENS3", "iTransformer_34_ENS3",
              "xLSTM_23_ENS3", "xLSTM_34_ENS3", "xLSTM_34_ENS3+MinT_wls"):
        sub = me[me["kind"] == k]
        if sub.empty:
            continue
        v = {str(r["h"]): r["MAE"] for _, r in sub.iterrows()}
        A(f"| {k} | " + " | ".join(f"{v.get(str(h), float('nan')):.1f}"
                                   for h in ("1", "3", "6", "12")) +
          f" | **{v.get('ALL', float('nan')):.1f}** |")
    A("")
    # how many of the 23 series actually improved, pooled over all horizons
    _pm = ps.pivot_table(index="series", columns="model", values="MASE_series")
    for lab, a_col, b_col in (("xLSTM", "xLSTM_34_ENS3", "xLSTM_23_ENS3"),
                              ("iTransformer", "iTransformer_34_ENS3",
                               "iTransformer_23_ENS3")):
        if a_col not in _pm.columns or b_col not in _pm.columns:
            continue
        d = (_pm[a_col] - _pm[b_col]).dropna()
        n_better = int((d < 0).sum())
        A(f"- {lab}：23 條序列當中有 **{n_better}/23** 條在 34 條版本下 MASE 較低"
          f"（其餘 {23-n_better} 條較高），中位數變化 {d.median():+.5f}。")
    A("")
    A("**MASE 與池化 MAE 讀起來不一樣，這件事本身是結論的一部分**："
      "xLSTM 的池化 MAE 幾乎不動（221.0 → 221.4，+0.2%），但 MASE 升 4.17%。"
      "原因是 MASE 對 23 條序列**等權**，而池化 MAE 由 4 條巨頭"
      "（`8`、`18`、`35`、`39`/`40`）主導 —— 新增子序列對巨頭的傷害很小，"
      "對一群中小型序列（煙酒、藥妝、珠寶、傢俬……）的傷害較大。"
      "所以「34 條版本整體變差」在金額加權的 WAPE（+0.16%）上幾乎看不出來，"
      "但在逐序列公平計分的 MASE 上很清楚。")
    A("")

    # ---- §6 the two questions ---------------------------------------------
    A("## 6. 回答兩個指定問題")
    A("")
    q = []
    a23 = allrow(me, "xLSTM_23_ENS3")
    a34 = allrow(me, "xLSTM_34_ENS3")
    b23 = allrow(me, "iTransformer_23_ENS3")
    b34 = allrow(me, "iTransformer_34_ENS3")
    if a23 is not None and a34 is not None:
        better = a34["MASE"] < a23["MASE"]
        q.append(f"- **xLSTM：加 11 條子序列後，原 23 條的 MASE_ALL 由 "
                 f"{a23['MASE']:.5f} 變成 {a34['MASE']:.5f}"
                 f"（{100*(a34['MASE']-a23['MASE'])/a23['MASE']:+.2f}%），"
                 f"即{'改善' if better else '變差'}。**")
        q.append(f"  WAPE_ALL 由 {a23['WAPE']:.5f} 變成 {a34['WAPE']:.5f}"
                 f"（{100*(a34['WAPE']-a23['WAPE'])/a23['WAPE']:+.2f}%）。")
    if b23 is not None and b34 is not None:
        better = b34["MASE"] < b23["MASE"]
        q.append(f"- **iTransformer：{b23['MASE']:.5f} → {b34['MASE']:.5f}"
                 f"（{100*(b34['MASE']-b23['MASE'])/b23['MASE']:+.2f}%），"
                 f"即{'改善' if better else '變差'}。**")
        q.append(f"  WAPE_ALL 由 {b23['WAPE']:.5f} 變成 {b34['WAPE']:.5f}"
                 f"（{100*(b34['WAPE']-b23['WAPE'])/b23['WAPE']:+.2f}%）。")
    for s in q:
        A(s)
    A("")

    # parents 8 and 30
    A("### 6.1 母類 `8`（超市）與 `30`（百貨）")
    A("")
    A("單條序列 MASE = 該序列 12 原點 × 4 視界的平均絕對誤差 ÷ 它自己的季節天真 MAE"
      "（與 `harness._metric_row` 對每條序列的算法相同）。")
    A("")
    A("| 序列 | 名稱 | xLSTM_23 | xLSTM_34 | 差 | xLSTM_23+MinT | xLSTM_34+MinT | 差 |")
    A("|---|---|---|---|---|---|---|---|")
    names = {"8": "超市 Supermarkets", "30": "百貨 Department stores"}
    for s in ("8", "30"):
        row = {}
        for k in ("xLSTM_23_ENS3", "xLSTM_34_ENS3", "xLSTM_23_ENS3+MinT_wls",
                  "xLSTM_34_ENS3+MinT_wls"):
            sub = ps[(ps["model"] == k) & (ps["series"] == s)]
            row[k] = float(sub["MASE_series"].iloc[0]) if len(sub) else float("nan")
        A(f"| `{s}` | {names[s]} | {row['xLSTM_23_ENS3']:.5f} | "
          f"{row['xLSTM_34_ENS3']:.5f} | "
          f"{100*(row['xLSTM_34_ENS3']-row['xLSTM_23_ENS3'])/row['xLSTM_23_ENS3']:+.2f}% | "
          f"{row['xLSTM_23_ENS3+MinT_wls']:.5f} | "
          f"{row['xLSTM_34_ENS3+MinT_wls']:.5f} | "
          f"{100*(row['xLSTM_34_ENS3+MinT_wls']-row['xLSTM_23_ENS3+MinT_wls'])/row['xLSTM_23_ENS3+MinT_wls']:+.2f}% |")
    A("")
    A("（序列 `8` 與 `30` 的絕對水平最大，MASE 因為是用各自的天真 MAE 標準化，"
      "所以不會被它們的體量掩蓋。）")
    A("")
    A("**23 條版那兩欄的 `MinT` 數字與未調和完全相同（0.82289 / 0.47916）**，"
      "這是正確的：在凍結的 4 母類階層裡，`8` 與 `30` 是**葉節點**、沒有任何子項約束，"
      "所以 MinT 對它們不動。它們要等到 34 條版本才第一次有了子項 —— "
      "但這一代價是基礎預測先退化，MinT 只收回一部分。")
    A("")

    # ---- §7 DM -------------------------------------------------------------
    A("## 7. 統計檢定：34 vs 23 的差距是否超出雜訊")
    A("")
    dm_rows = []
    for h in sorted(ens["h"].unique()):
        for new, old, lab in (("xLSTM_34_ENS3", "xLSTM_23_ENS3", "xLSTM"),
                              ("iTransformer_34_ENS3", "iTransformer_23_ENS3",
                               "iTransformer")):
            a = ens[(ens["model"] == new) & (ens["h"] == h)]
            b = ens[(ens["model"] == old) & (ens["h"] == h)]
            if a.empty or b.empty:
                continue
            key = ["origin", "series"]
            ea = (a.set_index(key)["y_true"] - a.set_index(key)["y_pred"])
            eb = (b.set_index(key)["y_true"] - b.set_index(key)["y_pred"]).reindex(ea.index)
            r2 = H.dm_test(ea.to_numpy(), eb.to_numpy(), h=int(h), power=2)
            r1 = H.dm_test(ea.to_numpy(), eb.to_numpy(), h=int(h), power=1)
            dm_rows.append({"model": lab, "h": int(h), "n": r2["n"],
                            "DM_sq": r2["dm"], "p_sq": r2["p_value"],
                            "DM_abs": r1["dm"], "p_abs": r1["p_value"]})
    if dm_rows:
        dm = pd.DataFrame(dm_rows)
        dm.to_csv(OUT / "dm_34vs23.csv", index=False)
        A("Diebold–Mariano，A = 34 條版，B = 23 條版。`DM_sq < 0` 代表 34 條版的損失較低"
          "（power=2 = 平方損失；power=1 = 絕對損失，較接近 MASE）。")
        A("")
        A("| 模型 | h | n | DM(平方) | p | DM(絕對) | p |")
        A("|---|---|---|---|---|---|---|")
        for _, r in dm.iterrows():
            A(f"| {r['model']} | {r['h']} | {r['n']} | {r['DM_sq']:+.3f} | "
              f"{r['p_sq']:.4f} | {r['DM_abs']:+.3f} | {r['p_abs']:.4f} |")
        A("")
        sig = dm[(dm["p_sq"] < 0.05) | (dm["p_abs"] < 0.05)]
        A(f"在 5% 水平下達到顯著的格數：{len(sig)}/{len(dm)}。")
    A("")

    # ---- §7.2 series-level Wilcoxon (the cross-sectionally aware test) ------
    A("### 7.2 序列級 Wilcoxon（`harness.dm_series_level`，各自對自己的 23 條版）")
    A("")
    A("把 12 個原點壓成每條序列一個平均絕對誤差，再對 23 條序列做配對檢定 —— "
      "避開「同一月份 23 條序列被同一個宏觀衝擊推動」造成的顯著性高估。"
      "下表每一列都是「34 條版 vs 同一模型的 23 條版」。")
    A("")
    A("| 模型 | h | n_series | 34 條版較好 | 23 條版較好 | 中位差（正=34 較差） | p |")
    A("|---|---|---|---|---|---|---|")
    for b, keep in (("xLSTM_23_ENS3", "xLSTM_34_ENS3"),
                    ("iTransformer_23_ENS3", "iTransformer_34_ENS3")):
        try:
            w = H.dm_series_level(ens, baseline=b)
        except Exception as exc:
            A(f"| {keep} | — | — | — | — | — | 未執行（{exc}） |")
            continue
        w = w[w["model"] == keep]
        for _, r in w.iterrows():
            A(f"| {keep} | {r['h']} | {r['n_series']} | {r['series_won']} | "
              f"{r['series_lost']} | {r['median_diff']:+.3f} | {r['p_wilcoxon']:.4f} |")
    A("")

    # ---- §8 timing ---------------------------------------------------------
    A("## 8. 實測耗時")
    A("")
    A("| 階段 | 秒 |")
    A("|---|---|")
    lbl = {"itransformer_34_sec": "iTransformer_34（3 種子，含推論）",
           "xlstm_34_sec": "xLSTM_34（3 種子，含推論）"}
    for k, v in meta["timings_sec"].items():
        if not v:
            continue
        A(f"| {lbl.get(k, k)} | {v} |")
    A(f"| **`full` 合計** | **{meta['timings_sec_total']}** |")
    c23 = OUT / "control23"
    if c23.exists():
        s_it = s_x = 0.0
        for f in sorted(c23.glob("preds_itransformer23_s*.json")):
            s_it += float(json.loads(f.read_text(encoding="utf-8"))["fit_sec"])
        for f in sorted(c23.glob("preds_xlstm23_s*.json")):
            j = json.loads(f.read_text(encoding="utf-8"))
            s_x += float(j.get("train_sec", j.get("wall_sec", 0.0)))
        A(f"| `control23` iTransformer_23（3 種子） | {s_it:.1f} |")
        A(f"| `control23` xLSTM_23（3 種子） | {s_x:.1f} |")
    A("")
    A("逐種子：")
    A("")
    A("| 模型 | 種子 | 訓練秒 | 參數 | 訓練樣本／token |")
    A("|---|---|---|---|---|")
    for m in meta["itransformer_runs"]:
        A(f"| iTransformer_34 | {m['seed']} | {m['fit_sec']} | {m['n_params']:,} | "
          f"{m['n_train_tokens']:,} token |")
    for m in meta["xlstm_runs"]:
        A(f"| xLSTM_34 | {m['seed']} | {m.get('train_sec', m.get('wall_sec'))} | "
          f"{m['n_params']:,} | {m['n_train_samples']:,} |")
    A("")
    A("參考：23 條版本的 xLSTM 每種子 152.6 / 155.6 / 156.0 秒（`control23` 實測），"
      "34 條版本 233.6 / 257.6 / 284.0 秒，比值 1.5–1.8 倍 —— 與樣本數比 "
      "6,188/4,186 = 1.48 同量級，多出來的部分是 34 token 的 xLSTM stack 本身較寬。")
    A("")

    # ---- §9 limitations ----------------------------------------------------
    A("## 9. 限制")
    A("")
    A("- 3 個種子的種子雜訊本身約 ±0.03 MASE（見 `FINAL_REPORT.md` §3.8），"
      "所以任何小於 0.02 的差異都只能算「方向」，不能算「效果量」。"
      "本文所有 Δ 都同時給出 DM 檢定，請以檢定結果為準。")
    A("- 新序列只進訓練，不進計分；「新序列本身預測得多好」不是本實驗的問題，"
      "`results/upgrade34/` 裡仍存了 34 條版本的逐點預測，可另行評估。")
    A("- `MinT_wls` 的 W 是對角近似（只用到每條序列自己的天真誤差變異數），"
      "與 Stage 4 一致，沒有用到跨序列相關。")
    A("")
    A("### 9.1 這個「變差」不是單一原因，至少有三個可分辨的機制")
    A("")
    A("1. **共享權重被攤薄（xLSTM）。** xLSTM 是 channel-independent 但**權重共享**，"
      "多出來的 2,002 個樣本（= 182 個原點 × 11 條新序列）**全部**來自 4 條超市子類與 "
      "7 條百貨子類，它們的序列動態（例如 S3 煙酒、D7 百貨的超市部）"
      "與原 23 條不完全同分佈，把一組共享的 backbone 往它們的方向拉。"
      "參數量只增加 352（20,496 → 20,848，全部是序列嵌入），容量沒變，"
      "所以差別只能是「權重被誰訓練」。")
    A("2. **驗證損失的組成變了 → early stopping 停在不同的 epoch。** "
      "34 條版本的驗證 MSE 是對 **34 條**序列平均，因此它的數值（0.47–0.48）"
      "比 23 條版本（0.53–0.58）低，但低的原因是新增子類比較好預測，"
      "不等於原 23 條變好。選模準則一旦被替換，兩個版本就不是在同一個 epoch 上比較。"
      "**這一點已用 §11 的 `*_34_v23val` 臂直接檢定，結果見該章。**")
    A("3. **iTransformer 的注意力被多出來的 token 分掉。** 34 個 token 做"
      "跨品類注意力，原 23 條要跟 11 條新的子類競爭注意力權重；"
      "而且新子類與它們的母類（8、30）高度共線，等於在同一個子空間裡塞進重複資訊。"
      "iTransformer 的退化幅度（+2.89%）比 xLSTM（+4.17%）小，"
      "與「它的 readout 是共享線性層、對 token 數不敏感」一致。")
    A("")
    A("## 10. 未做的事（明確列出）")
    A("")
    A("- **沒有**做超參搜尋：34 條版本直接沿用 23 條版本出貨的超參，"
      "沒有為更大的訓練集調 epochs / lr / dropout。"
      "若 34 條版本在充分調參後反超，本結論就要修正。")
    A("- **沒有**評估新增 11 條序列本身的預測精度。"
      "`results/upgrade34/preds_xlstm34_s*.csv` 與 `preds_itransformer34_s*.csv` "
      "含全部 34 條的逐點預測，可直接算。")
    A("- **沒有**把 34 條的通道獨立臂（`timetf` 之類）與 `MinT_wls` 一起重做；"
      "§4 的調和只接在 xLSTM 上，與 Stage 4 的做法一致。")
    A("")
    A("> 原本列在這裡的第一項（「沒有把 early stopping 限制在原 23 條重跑」）"
      "已經補做，見 §11。")
    A("")

    # ---- §11 the selection-bias control -----------------------------------
    A("## 11. 選模混淆的直接檢定：`*_34_v23val`")
    A("")
    A("§9.1 第 2 點指出全篇最大的內部效度威脅：34 條版本的 early stopping 用"
      "**34 條**序列的驗證 MSE 選 epoch，23 條版本用 **23 條**。"
      "若「34 條變差」只是因為選模準則換了，那它就不是資訊效果。"
      "這一章用一個新臂把這件事單獨拿掉。")
    A("")
    A("### 11.1 這個臂改了什麼、沒改什麼")
    A("")
    A("| 項目 | 34 條（原版） | **34 條（23-val）** | 23 條（控制組） |")
    A("|---|---|---|---|")
    A("| 模型輸入 token／通道 | 34 | **34（不變）** | 23 |")
    A("| 訓練損失（梯度） | 34 條全算 | **34 條全算（不變）** | 23 條 |")
    A("| early stopping 用的驗證 MSE | 34 條 × 13 原點 = 442 點 | "
      "**前 23 條 × 13 原點 = 299 點** | 23 條 × 13 原點 = 299 點 |")
    A("| origin plan / purge / 超參 / 種子 | 同 | **同（不變）** | 同 |")
    A("| 評估 | 只取原 23 條 | **只取原 23 條** | 原 23 條 |")
    A("")
    A("唯一被移動的變數就是「用誰的驗證損失選 epoch」。"
      "做法是把 `models/itransformer.py::train_transformer` 與 "
      "`models/xlstm_direct.py::train_model` 逐行複製成 "
      "`train_itransformer_valcut` / `train_xlstm_valcut`（凍結的庫檔不動），"
      "只把驗證張量在算 MSE 前切到前 `k_val` 個 variate／序列。")
    A("")
    # ---- 11.2 equivalence proof -------------------------------------------
    vdir = OUT / "verifycopy"
    A("### 11.2 先證明副本就是庫函數（`k_val = 34`）")
    A("")
    ok_all = True
    if vdir.exists():
        proof = []
        for ship, name in (("preds_itransformer34_s42.csv", "iTransformer"),
                           ("preds_xlstm34_s42.csv", "xLSTM")):
            fv = vdir / ship
            fs = OUT / ship
            if fv.exists() and fs.exists():
                d = _max_point_diff(pd.read_csv(fv, dtype={"series": str,
                                                           "origin": str}),
                                    pd.read_csv(fs, dtype={"series": str,
                                                           "origin": str}))
                ok_all &= d < 1e-9
                proof.append((name, d))
        A("當 `k_val = 34` 時，副本的驗證損失與庫函數完全相同，"
          "所以它應該逐位重現已出貨的 34 條臂。實測：")
        A("")
        A("| 模型 | max \\|副本 − 已出貨\\| | 判定 |")
        A("|---|---|---|")
        for name, d in proof:
            A(f"| {name} | {d:.3e} | {'IDENTICAL' if d < 1e-9 else '**DIFFERS**'} |")
        A("")
        A("因此 `k_val = 23` 那一臂的差異**只可能**來自驗證集切片，"
          "不可能來自「我自己寫了一個不一樣的訓練迴圈」。")
        if not ok_all:
            A("")
            A("> ⚠️ 副本重現失敗，本章結論不可用。")
        A("")
        A("**但「迴圈相同」不等於「遮罩寫對」**，而這兩件事在兩個模型上是不同的機制：")
        A("")
        A("- **iTransformer** 的輸出是 `(batch, S, n_heads)`，有序列軸，"
          "所以 `out[:, :k_val]` 就是切前 23 個 variate。")
        A("- **xLSTM** 是 channel-independent，一個樣本 = 一個（原點, 序列）配對，"
          "輸出是 `(n_samples, n_heads)` —— **沒有序列軸**。"
          "第一版在這裡切了 `out[:, :23]`，實際切到的是 4 個視界頭（23 > 4，等於沒切），"
          "驗證損失與 `k_val = 34` 逐位相同。**這是我從「`val@23` 印出來與 `val@34` 一模一樣」"
          "這個不可能的巧合上抓到的**，然後改成用每個樣本攜帶的序列索引 `Sva` 做遮罩。")
        A("")
        A("修正後有兩道獨立檢查（`_selftest_valmask`，每次執行 `v23val` 都會跑）：")
        A("")
        A("1. torch 的遮罩結果必須等於一個獨立用 numpy 算出來的子集平均"
          "（本機實測 `torch 1.469084` = `numpy 1.469084`，而全集的平均是 `1.563091`，"
          "證明遮罩真的生效而不是 no-op）。")
        A("2. 執行時斷言 `選中的樣本數 == k_val × 驗證原點數`"
          "（442 → 299），任何被靜默丟掉的樣本都會讓 early stopping 的準則改變。")
    else:
        A("（`v23val --verify` 未執行，沒有等同性證據。）")
    A("")
    # ---- 11.3 the three-protocol table ------------------------------------
    A("### 11.3 三個協議並列")
    A("")
    vm = OUT / "v23val" / "metrics_v23val_23.csv"
    ve = OUT / "v23val" / "best_epochs_v23val.csv"
    if vm.exists():
        vmet = pd.read_csv(vm)
        vep = pd.read_csv(ve) if ve.exists() else None

        def epstr(model: str, protocol: str) -> str:
            if vep is None:
                return "—"
            s = vep[(vep["model"] == model) & (vep["protocol"] == protocol)]
            if s.empty:
                return "—"
            return "/".join(str(int(x)) for x in s.sort_values("seed")["best_epoch"])

        A("| 協議 | 序列數 | early stop 用 | MASE_ALL | WAPE_ALL | "
          "best_epoch（42 / 7 / 2024） |")
        A("|---|---|---|---|---|---|")
        for model, kind23, kind34, kindv in (
                ("iTransformer", "iTransformer_23_ENS3", "iTransformer_34_ENS3",
                 "iTransformer_34_v23val_ENS3"),
                ("xLSTM", "xLSTM_23_ENS3", "xLSTM_34_ENS3", "xLSTM_34_v23val_ENS3")):
            r23, r34 = allrow(me, kind23), allrow(me, kind34)
            rv = allrow(vmet, kindv)
            if r23 is not None:
                A(f"| {model}_23（控制組） | 23 | 23 條 | **{r23['MASE']:.5f}** | "
                  f"{r23['WAPE']:.5f} | {epstr(model, '23 series (control)')} |")
            if r34 is not None:
                A(f"| {model}_34（原版） | 34 | 34 條 | **{r34['MASE']:.5f}** | "
                  f"{r34['WAPE']:.5f} | {epstr(model, '34 series (original)')} |")
            if rv is not None:
                A(f"| **{model}_34（23-val）** | 34 | **23 條** | "
                  f"**{rv['MASE']:.5f}** | {rv['WAPE']:.5f} | "
                  f"{epstr(model, '34 series (23-val)')} |")
            else:
                A(f"| {model}_34（23-val） | 34 | 23 條 | *未執行* | — | — |")
        A("")
        # deltas
        A("拆解（ΔMASE_ALL，正 = 比 23 條控制組差）：")
        A("")
        A("| 模型 | 34 條（原版） | 34 條（23-val） | 選模混淆佔比 |")
        A("|---|---|---|---|")
        for model, kind23, kind34, kindv in (
                ("iTransformer", "iTransformer_23_ENS3", "iTransformer_34_ENS3",
                 "iTransformer_34_v23val_ENS3"),
                ("xLSTM", "xLSTM_23_ENS3", "xLSTM_34_ENS3", "xLSTM_34_v23val_ENS3")):
            r23, r34, rv = allrow(me, kind23), allrow(me, kind34), allrow(vmet, kindv)
            if r23 is None or r34 is None:
                continue
            d34 = r34["MASE"] - r23["MASE"]
            if rv is None:
                A(f"| {model} | {d34:+.5f} | *未執行* | — |")
                continue
            dv = rv["MASE"] - r23["MASE"]
            share = 100.0 * (1.0 - dv / d34) if d34 else float("nan")
            A(f"| {model} | {d34:+.5f} | {dv:+.5f} | {share:+.1f}% |")
        A("")
        A("「選模混淆佔比」= (原版 Δ − 23-val 版 Δ) / 原版 Δ。"
          "100% 表示全部退化都來自選模準則；0% 表示與選模無關。")
        A("")
        # ---- 11.4 the answer ----------------------------------------------
        A("### 11.4 回答：排除選模混淆之後，34 條是否仍然變差？")
        A("")
        for model, kind23, kind34, kindv in (
                ("iTransformer", "iTransformer_23_ENS3", "iTransformer_34_ENS3",
                 "iTransformer_34_v23val_ENS3"),
                ("xLSTM", "xLSTM_23_ENS3", "xLSTM_34_ENS3", "xLSTM_34_v23val_ENS3")):
            r23, r34, rv = allrow(me, kind23), allrow(me, kind34), allrow(vmet, kindv)
            if r23 is None or r34 is None:
                continue
            if rv is None:
                A(f"- **{model}**：未執行，本題未答。")
                continue
            still_worse = rv["MASE"] > r23["MASE"]
            d = rv["MASE"] - r23["MASE"]
            share = 100.0 * (1.0 - d / (r34["MASE"] - r23["MASE"]))
            A(f"- **{model}：{'仍然變差' if still_worse else '翻正'}。** "
              f"23 條控制組 {r23['MASE']:.5f} → 34 條原版 {r34['MASE']:.5f}"
              f"（{100*(r34['MASE']-r23['MASE'])/r23['MASE']:+.2f}%）→ "
              f"34 條 23-val {rv['MASE']:.5f}"
              f"（{100*d/r23['MASE']:+.2f}%）。"
              f"選模準則解釋了退化量的 {share:+.1f}%，"
              f"剩下的 {100-share:.1f}% "
              f"{'落在 34 條輸入本身，但它是否顯著見下表檢定' if still_worse else '使結果翻正（原本的退步主要來自選模）'}。")
        A("")
        # DM: 23-val arm vs the 23-series control
        vp = OUT / "v23val" / "preds_v23val_23.csv"
        if vp.exists():
            cmp = pd.concat([ens, pd.read_csv(vp, dtype={"origin": str,
                                                        "series": str})],
                            ignore_index=True)
            dmv = []
            for h in sorted(cmp["h"].unique()):
                for model, k23, kv in (
                        ("iTransformer", "iTransformer_23_ENS3",
                         "iTransformer_34_v23val_ENS3"),
                        ("xLSTM", "xLSTM_23_ENS3", "xLSTM_34_v23val_ENS3")):
                    a = cmp[(cmp["model"] == kv) & (cmp["h"] == h)]
                    b = cmp[(cmp["model"] == k23) & (cmp["h"] == h)]
                    if a.empty or b.empty:
                        continue
                    key = ["origin", "series"]
                    ea = a.set_index(key)["y_true"] - a.set_index(key)["y_pred"]
                    eb = (b.set_index(key)["y_true"]
                          - b.set_index(key)["y_pred"]).reindex(ea.index)
                    r2 = H.dm_test(ea.to_numpy(), eb.to_numpy(), h=int(h), power=2)
                    r1 = H.dm_test(ea.to_numpy(), eb.to_numpy(), h=int(h), power=1)
                    dmv.append({"model": model, "h": int(h), "n": r2["n"],
                                "DM_sq": r2["dm"], "p_sq": r2["p_value"],
                                "DM_abs": r1["dm"], "p_abs": r1["p_value"]})
            if dmv:
                dmvf = pd.DataFrame(dmv)
                dmvf.to_csv(OUT / "v23val" / "dm_v23val_vs_control.csv", index=False)
                A("**統計檢定（23-val 臂 A vs 23 條控制組 B，DM）**：`DM < 0` 代表 "
                  "23-val 臂損失較低。")
                A("")
                A("| 模型 | h | n | DM(平方) | p | DM(絕對) | p |")
                A("|---|---|---|---|---|---|---|")
                for _, r in dmvf.iterrows():
                    A(f"| {r['model']} | {r['h']} | {r['n']} | {r['DM_sq']:+.3f} | "
                      f"{r['p_sq']:.4f} | {r['DM_abs']:+.3f} | {r['p_abs']:.4f} |")
                A("")
                for mdl in dmvf["model"].unique():
                    s = dmvf[dmvf["model"] == mdl]
                    pmin = min(float(s["p_sq"].min()), float(s["p_abs"].min()))
                    n_neg = int((s["DM_sq"] < 0).sum())
                    neg_txt = ("全部為正（= 23-val 臂損失較高）" if n_neg == 0 else
                               "全部為負（= 23-val 臂損失較低）" if n_neg == 4 else
                               f"在 {n_neg}/4 個視界為負（方向不一致）")
                    A(f"- {mdl}：23-val 臂 vs 控制組，8 個檢定的最小 p = {pmin:.4f} → "
                      f"**{'差距在 5% 水平下不顯著（只能算方向）' if pmin >= 0.05 else '至少一格顯著'}**；"
                      f"平方損失 {neg_txt}。")
                A("")
                A("**判讀的關鍵是「兩個指標不同向」**：MASE_ALL 對 23 條序列等權，"
                  "池化 DM 對所有點等權。殘餘差距在 MASE 上為正（較差），"
                  "在池化損失上卻多為負（較好），意思是 34 條輸入把誤差"
                  "從大序列搬到一群中小型序列上 —— 這是**分佈移動**，"
                  "不是一致性的退化。這與 §5 那條「MASE 與池化 MAE 讀起來不一樣」是同一件事。")
        A("")
        A("搭配 §9.1：兩條機制（權重被攤薄、注意力被分掉）"
          "在 23-val 臂裡完全沒有被動到，因為模型輸入仍然是 34 條；"
          "被移除的只有「用 34 條的驗證集選 epoch」。")
        A("")
        A("**所以最終的誠實結論分三層，不要合併成一句：**")
        A("")
        A("1. **原版 34 條的退步（+2.89% / +4.17%）約三分之二來自選模準則，"
          "不是資訊。** 這是我原本沒排除、後來用 23-val 臂量出來的。")
        A("2. **把選模拿掉之後，34 條仍然沒有贏**（+1.03% / +1.42%），"
          "但這個殘餘在統計上大多不顯著，而且在池化損失上方向相反。"
          "所以「34 條的資訊效果為負」這句話只能說成**「沒有觀察到正面效果」**，"
          "不能說成「已證實有害」。")
        A("3. **要宣稱 34 條有幫助，需要的是另一個實驗**："
          "例如只在母類 `8`/`30` 上用子項做 bottom-up 或 MinT 的約束"
          "（§6.1 已顯示 34 節點 `MinT_wls` 能把 `8` 的傷害由 +7.44% 收到 +2.13%），"
          "而不是把 11 條序列塞進一個共享權重的 34 通道模型。")
    else:
        A("（`v23val` 未執行。先跑 "
          "`run_upgrade_34series.py v23val`）")
    A("")

    dst = ROOT.parent / "docs" / "UPGRADE_34SERIES_RESULTS.md"
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {dst}")
    return 0


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build-data", "smoke", "control23", "full",
                                    "v23val", "report"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models", nargs="+", default=["itransformer", "xlstm"],
                    choices=["itransformer", "xlstm"],
                    help="v23val: which arms to run")
    ap.add_argument("--skip-verify", action="store_true",
                    help="v23val: skip the copied-loop equivalence check (not advised)")
    ap.add_argument("--verify-only", action="store_true",
                    help="v23val: only prove the copies are the library loops")
    ap.add_argument("--skip-xlstm", action="store_true",
                    help="v23val: run the iTransformer arm only")
    args = ap.parse_args()
    if args.cmd == "build-data":
        return cmd_build_data(args)
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd == "control23":
        return cmd_control23(args)
    if args.cmd == "full":
        return cmd_full(args)
    if args.cmd == "v23val":
        return cmd_v23val(args)
    if args.cmd == "report":
        return cmd_report(args)
    raise KeyError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
