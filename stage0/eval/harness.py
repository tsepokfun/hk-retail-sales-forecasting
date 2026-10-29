"""
Stage 0 — Step 3: the LOCKED evaluation harness.

Everything downstream goes through this file. Do not change the metric
definitions once results exist, or the numbers stop being comparable.

Design (01_ARCHITECTURE.md 6.2 / 决策 5):
  * Forecast origins   : 2023-12 .. 2024-11   (12 origins)
  * Horizons           : h = 1, 3, 6, 12      (Direct Multi-Horizon, no recursion)
  * Target period      : month(origin) + h
  * Ground truth       : the ACTUAL value at origin+h.
                         Data runs to 2026-08, so every t+h has a real value.
                         The origin never sees it -> this is evaluation, not leakage.
  * Scale              : all metrics on the ORIGINAL HK$ million scale.
  * Metrics            : MASE, WAPE, RMSE, MAE
  * Jan+Feb special    : a pooled 2024 Jan+Feb WAPE is reported separately,
                         because Lunar New Year moved 19 days between 2023 and 2024.

MASE denominator = in-sample seasonal-naive MAE at lag 12, computed on train only
(Hyndman & Athanasopoulos, FPP3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TRAIN_END = "202312"
TEST_START, TEST_END = "202401", "202412"
HORIZONS = (1, 3, 6, 12)
SEASON = 12

# origins: 2023-12 .. 2024-11  -> 12 origins
ORIGINS = [
    "202312", "202401", "202402", "202403", "202404", "202405",
    "202406", "202407", "202408", "202409", "202410", "202411",
]

FEATURE_COLS = (
    [f"lag{L}" for L in (1, 2, 3, 12, 13)]
    + [f"roll{w}_{k}" for w in (3, 6, 12) for k in ("mean", "std")]
    + [f"month_{m}" for m in range(1, 13)]
    + ["lny_pre14", "lny_post7", "covid_border_closed", "consumption_voucher"]
)
ID_COLS = ["series_id", "level_id"]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    wide = pd.read_csv(DATA / "panel_wide.csv", index_col=0)
    wide.index = wide.index.astype(str)
    return wide


def load_features() -> pd.DataFrame:
    # series codes are "2","3",... -> must stay strings, not ints
    f = pd.read_csv(DATA / "features_long.csv", dtype={"period": str, "series": str, "parent_id": str})
    f["period"] = f["period"].astype(str).str.zfill(6)
    f["series"] = f["series"].astype(str)
    return f


def month_add(period: str, h: int) -> str:
    y, m = int(period[:4]), int(period[4:])
    m2 = m - 1 + h
    return f"{y + m2 // 12}{(m2 % 12) + 1:02d}"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def seasonal_naive_mae_train(wide: pd.DataFrame) -> pd.Series:
    """in-sample seasonal-naive MAE at lag 12, per series, TRAIN ONLY"""
    train = wide.loc[[p for p in wide.index if p <= TRAIN_END]]
    d = (train - train.shift(SEASON)).abs().iloc[SEASON:]
    return d.mean(axis=0)


def seasonal_naive_forecast(wide: pd.DataFrame, origin: str, h: int) -> pd.Series:
    """yhat(t+h) = y(t+h-12).  Falls back to y(origin) if that month is absent."""
    target = month_add(origin, h)
    back = month_add(target, -SEASON)
    if back in wide.index:
        return wide.loc[back]
    return wide.loc[origin]


@dataclass
class Acc:
    """accumulates per-series absolute errors so MASE can be pooled correctly"""
    rows: list = field(default_factory=list)

    def add(self, model, origin, h, series, y_true, y_pred):
        for s, yt, yp in zip(series, y_true, y_pred):
            self.rows.append((model, origin, h, s, float(yt), float(yp)))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])


def metrics_from_frame(df: pd.DataFrame, naive_mae: pd.Series) -> pd.DataFrame:
    """
    df: columns model, origin, h, series, y_true, y_pred   (original HK$m scale)
    returns one row per (model, h) plus an 'ALL' horizon row.
    """
    out = []
    for (model, h), g in df.groupby(["model", "h"], sort=False):
        out.append(_metric_row(g, naive_mae, model, h))
    for model, g in df.groupby("model", sort=False):
        out.append(_metric_row(g, naive_mae, model, "ALL"))
    return pd.DataFrame(out)


def _metric_row(g: pd.DataFrame, naive_mae: pd.Series, model, h) -> dict:
    e = g["y_true"] - g["y_pred"]
    ae = e.abs()
    # MASE: per-series MAE divided by that series' naive MAE, then averaged
    per_series = (
        ae.groupby(g["series"]).mean()
        / naive_mae.reindex(sorted(g["series"].unique())).replace(0, np.nan)
    )
    mase = float(np.nanmean(per_series.values))
    denom = float(g["y_true"].abs().sum())
    return {
        "model": model,
        "h": h,
        "n_points": int(len(g)),
        "MASE": round(mase, 5),
        "WAPE": round(float(ae.sum() / denom) if denom else np.nan, 5),
        "RMSE": round(float(np.sqrt((e ** 2).mean())), 3),
        "MAE": round(float(ae.mean()), 3),
        "bias": round(float(e.mean()), 3),
        "bias_pct": round(float(100 * e.sum() / denom), 3) if denom else np.nan,
    }


def janfeb_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """
    2024 Jan+Feb pooled WAPE per (model, h).
    Because LNY moved 2023-01-22 -> 2024-02-10, single-month Jan or Feb
    comparisons are meaningless; the pooled figure is the honest one.
    """
    sub = df[df["origin"].isin(["202401", "202402"])].copy()
    out = []
    for (model, h), g in sub.groupby(["model", "h"], sort=False):
        ae = (g["y_true"] - g["y_pred"]).abs().sum()
        den = g["y_true"].abs().sum()
        out.append({"model": model, "h": h,
                    "janfeb_WAPE": round(float(ae / den), 5) if den else np.nan,
                    "n_points": int(len(g))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# supervised sample construction (merge-based, avoids index gymnastics)
# ---------------------------------------------------------------------------
def make_supervised(feat: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    One row per (t, series) with:
      - FEATURE_COLS / ID_COLS as observed at t
      - target_period = t + h
      - target_yz     = y_z at t + h   (merged in; NaN if beyond data)

    Training at origin O uses rows with target_period <= O   (target known at O)
    Prediction at origin O uses rows with period == O
    """
    src = feat[["period", "series", "y_z"] + FEATURE_COLS + ID_COLS].copy()
    src["target_period"] = src["period"].map(lambda p: month_add(p, h))

    tgt = feat[["period", "series", "y_z"]].rename(
        columns={"period": "target_period", "y_z": "target_yz"}
    )
    out = src.merge(tgt, on=["target_period", "series"], how="inner")
    out = out.dropna(subset=FEATURE_COLS + ["target_yz"]).reset_index(drop=True)
    return out


def matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS + ID_COLS].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# window construction for supervised models
# ---------------------------------------------------------------------------
def build_windows(feat: pd.DataFrame, usable_origins: list[str], horizons=HORIZONS,
                  return_meta: bool = False):
    """
    Build direct multi-horizon samples from the feature table.

    A sample is (features_at_t, origin=t) with targets y_z at t+h for each h.
    Only rows where every h has a real target inside the feature table are kept.
    """
    feat = feat.set_index(["period", "series"]).sort_index()
    periods = sorted(feat.index.get_level_values("period").unique())
    pidx = {p: i for i, p in enumerate(periods)}

    X, Y, META = [], {h: [] for h in horizons}, []
    for (p, s), row in feat.iterrows():
        i = pidx[p]
        # need all horizons inside the data
        tgt_idx = {}
        ok = True
        for h in horizons:
            j = i + h
            if j >= len(periods):
                ok = False
                break
            tgt_idx[h] = periods[j]
        if not ok:
            continue
        vals = {h: feat.loc[(tgt_idx[h], s), "y_z"] for h in horizons}
        if any(pd.isna(v) for v in vals.values()):
            continue
        if any(pd.isna(row[c]) for c in FEATURE_COLS):
            continue
        X.append(row[FEATURE_COLS + ID_COLS].to_numpy(dtype=np.float64))
        for h in horizons:
            Y[h].append(vals[h])
        META.append({"period": p, "series": s, "origin_index": i})

    X = np.asarray(X, dtype=np.float64)
    Y = {h: np.asarray(v, dtype=np.float64) for h, v in Y.items()}
    meta = pd.DataFrame(META)
    if return_meta:
        return X, Y, meta
    return X, Y


# ---------------------------------------------------------------------------
# back-transform  y_z -> HK$ million
# ---------------------------------------------------------------------------
def inverse_transform(z: np.ndarray, series: np.ndarray, scaling: pd.DataFrame) -> np.ndarray:
    mu = scaling.set_index("series").loc[series, "mu_log"].to_numpy()
    sd = scaling.set_index("series").loc[series, "sigma_log"].to_numpy()
    return np.exp(z * sd + mu)


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# KNOWN-FUTURE calendar features for the TARGET month t+h
# ---------------------------------------------------------------------------
# At origin t we do not know y(t+h), but we DO know the calendar of t+h.
# Feeding the target month's own LNY shares and month-of-year directly to each
# head is free information and costs nothing in parametrisation.
LNY_BY_YEAR = {
    2004: (1, 22), 2005: (2, 9), 2006: (1, 29), 2007: (2, 18), 2008: (2, 7),
    2009: (1, 26), 2010: (2, 14), 2011: (2, 3), 2012: (1, 23), 2013: (2, 10),
    2014: (1, 31), 2015: (2, 19), 2016: (2, 8), 2017: (1, 28), 2018: (2, 16),
    2019: (2, 5), 2020: (1, 25), 2021: (2, 12), 2022: (2, 1), 2023: (1, 22),
    2024: (2, 10), 2025: (1, 29), 2026: (2, 17), 2027: (2, 6),
}
KNOWNFUT_COLS = ["kf_lny_pre14", "kf_lny_post7", "kf_month_sin", "kf_month_cos"]


def _lny_pair(period: str) -> tuple[float, float]:
    import calendar as _cal
    from datetime import date as _date, timedelta as _td

    y, m = int(period[:4]), int(period[4:])
    if y not in LNY_BY_YEAR:
        return 0.0, 0.0
    if m not in (1, 2):
        return 0.0, 0.0
    lm, ld = LNY_BY_YEAR[y]
    lny = _date(y, lm, ld)
    m0 = _date(y, m, 1)
    m1 = _date(y, m, _cal.monthrange(y, m)[1])
    lo, hi = lny - _td(days=14), lny - _td(days=1)
    pre = (min(hi, m1) - max(lo, m0)).days + 1
    pre = max(0, pre) / 14.0
    lo2, hi2 = lny, lny + _td(days=6)
    post = (min(hi2, m1) - max(lo2, m0)).days + 1
    post = max(0, post) / 7.0
    return round(pre, 6), round(post, 6)


def known_future(period: str) -> list[float]:
    """calendar facts about a month that are known in advance"""
    pre, post = _lny_pair(period)
    m = int(period[4:])
    ang = 2 * np.pi * (m - 1) / 12.0
    return [pre, post, float(np.sin(ang)), float(np.cos(ang))]


def build_known_future(periods: list[str], horizons=HORIZONS) -> np.ndarray:
    """
    (n_periods, n_horizons, 4) tensor of target-month calendar facts.
    Row i, head k corresponds to the month at origin index i + h_k.
    Beyond the data range we tile the last available value (never used anyway).
    """
    n, H_ = len(periods), len(horizons)
    out = np.zeros((n, H_, len(KNOWNFUT_COLS)), dtype=np.float32)
    for i in range(n):
        for k, h in enumerate(horizons):
            j = min(i + h, n - 1)
            out[i, k, :] = known_future(periods[j])
    return out


# ---------------------------------------------------------------------------
# Diebold-Mariano test (with Newey-West for h-step autocorrelation)
# ---------------------------------------------------------------------------
def dm_test(err_a: np.ndarray, err_b: np.ndarray, h: int = 1,
            power: int = 2) -> dict:
    """
    H0: model A and model B have equal expected loss.
    err_a, err_b: aligned error arrays (ytrue - ypred).
    loss = |e|^power.  DM = mean(d) / sqrt(LRV(d)/n), two-sided p from normal.

    Newey-West with lag = h-1 corrects for the overlap induced by an
    h-step-ahead forecast.  Returns the DM statistic and p-value, plus the sign
    convention: DM > 0 means model A has HIGHER loss (B is better).
    """
    a = np.asarray(err_a, dtype=float)
    b = np.asarray(err_b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 10:
        return {"n": n, "dm": np.nan, "p_value": np.nan, "better": None}
    d = np.abs(a) ** power - np.abs(b) ** power
    dbar = d.mean()
    dc = d - dbar
    lag = max(0, int(h) - 1)
    lrv = float(np.dot(dc, dc) / n)
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        gamma = float(np.dot(dc[L:], dc[:-L]) / n)
        lrv += 2.0 * w * gamma
    if lrv <= 0:
        return {"n": n, "dm": np.nan, "p_value": np.nan, "better": None}
    dm = dbar / np.sqrt(lrv / n)
    from math import erf, sqrt as _sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm) / _sqrt(2.0))))
    # d = loss_A - loss_B, so dm > 0 means model A has the HIGHER loss.
    # (this sign was inverted in the first version and silently flipped every
    #  conclusion; cross-checked against MASE/WAPE and corrected)
    better = "B" if dm > 0 else "A"
    return {"n": n, "dm": round(float(dm), 4), "p_value": round(float(p), 6),
            "better": better}


def dm_table(preds: pd.DataFrame, baseline: str = "SeasonalNaive") -> pd.DataFrame:
    """
    Pairwise DM of every model against `baseline`, per horizon, for BOTH loss
    functions:
      power=2  squared errors  -> value-weighted, dominated by the big series
      power=1  absolute errors -> closer to what MASE measures

    Reporting only one of them can flip the conclusion, so both are given.
    'better' names which model has the lower loss.
    """
    rows = []
    for h in sorted(preds["h"].unique()):
        sub = preds[preds["h"] == h]
        base = sub[sub["model"] == baseline]
        if len(base) == 0:
            continue
        key = ["origin", "series"]
        b = base.set_index(key)["y_true"] - base.set_index(key)["y_pred"]
        for m in sub["model"].unique():
            if m == baseline:
                continue
            o = sub[sub["model"] == m]
            a = (o.set_index(key)["y_true"] - o.set_index(key)["y_pred"]).reindex(b.index)
            r2 = dm_test(a.to_numpy(), b.to_numpy(), h=h, power=2)
            r1 = dm_test(a.to_numpy(), b.to_numpy(), h=h, power=1)
            rows.append({
                "h": h, "model": m, "vs": baseline, "n": r2["n"],
                "DM_sq": r2["dm"], "p_sq": r2["p_value"],
                "better_sq": ("model" if r2["better"] == "A" else "baseline") if r2["better"] else None,
                "DM_abs": r1["dm"], "p_abs": r1["p_value"],
                "better_abs": ("model" if r1["better"] == "A" else "baseline") if r1["better"] else None,
                "sig_sq": (r2["p_value"] is not None and r2["p_value"] < 0.05),
                "sig_abs": (r1["p_value"] is not None and r1["p_value"] < 0.05),
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# cross-sectionally aware tests for PANEL forecasts
# ---------------------------------------------------------------------------
# Pooling all (origin x series) points into one vector of length 276 treats the
# 23 retail categories as independent observations.  They are not: in any given
# month the whole retail sector moves with the same macro shock, so the pooled
# DM UNDERSTATES the variance and OVERSTATES significance.
# Two aggregation levels are therefore provided alongside the pooled test.
# ---------------------------------------------------------------------------
def dm_series_level(preds: pd.DataFrame, baseline: str = "SeasonalNaive") -> pd.DataFrame:
    """
    Collapse to one number per SERIES (N=23), then run a paired test.
      - metric: mean absolute error across the 12 origins
      - test  : Wilcoxon signed-rank (non-parametric, N=23, no normality assumed)
    Answers: "is the model better on most retail categories?"
    """
    from scipy import stats as st

    rows = []
    for h in sorted(preds["h"].unique()):
        sub = preds[preds["h"] == h]
        base = sub[sub["model"] == baseline]
        if len(base) == 0:
            continue
        b = base.assign(ae=(base["y_true"] - base["y_pred"]).abs()) \
                .groupby("series")["ae"].mean()
        for m in sub["model"].unique():
            if m == baseline:
                continue
            a = sub[sub["model"] == m]
            a = a.assign(ae=(a["y_true"] - a["y_pred"]).abs()) \
                 .groupby("series")["ae"].mean().reindex(b.index)
            x = a.to_numpy(float)
            y = b.to_numpy(float)
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            n = len(x)
            wins = int((x < y).sum())
            try:
                stat, p = st.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                stat, p = float(stat), float(p)
            except Exception:
                stat, p = np.nan, np.nan
            rows.append({"h": h, "model": m, "n_series": n,
                         "series_won": wins, "series_lost": int((x > y).sum()),
                         "median_diff": round(float(np.median(x - y)), 3),
                         "p_wilcoxon": round(p, 6) if p == p else np.nan,
                         "sig": bool(p == p and p < 0.05),
                         "better": m if np.median(x - y) < 0 else baseline})
    return pd.DataFrame(rows)


def dm_time_aggregated(preds: pd.DataFrame, baseline: str = "SeasonalNaive") -> pd.DataFrame:
    """
    Collapse across SERIES first, keeping the time axis: one value per ORIGIN
    (T=12).  Removes cross-sectional correlation by averaging it away, and
    answers: "is the model better on the average month?"
    Because T is small, a paired Wilcoxon is used (or a sign test if all
    differences are non-zero and N is tiny).
    """
    from scipy import stats as st

    rows = []
    for h in sorted(preds["h"].unique()):
        sub = preds[preds["h"] == h]
        base = sub[sub["model"] == baseline]
        if len(base) == 0:
            continue
        b = base.assign(ae=(base["y_true"] - base["y_pred"]).abs()) \
                .groupby("origin")["ae"].mean()
        for m in sub["model"].unique():
            if m == baseline:
                continue
            a = sub[sub["model"] == m]
            a = a.assign(ae=(a["y_true"] - a["y_pred"]).abs()) \
                 .groupby("origin")["ae"].mean().reindex(b.index)
            x = a.to_numpy(float); y = b.to_numpy(float)
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            n = len(x)
            wins = int((x < y).sum())
            try:
                stat, p = st.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                p = float(p)
            except Exception:
                p = np.nan
            rows.append({"h": h, "model": m, "n_origins": n,
                         "origins_won": wins, "origins_lost": int((x > y).sum()),
                         "median_diff": round(float(np.median(x - y)), 3),
                         "p_wilcoxon": round(p, 6) if p == p else np.nan,
                         "sig": bool(p == p and p < 0.05),
                         "better": m if np.median(x - y) < 0 else baseline})
    return pd.DataFrame(rows)
