"""Reproduce the classic statistical baselines that this project NEVER tested:
ETS / Holt-Winters, Theta, and SARIMAX.

Why this exists
---------------
FINAL_REPORT.md can claim "we beat the industry's workhorse (LightGBM)" but
cannot claim "we beat industry practice", because ETS/SARIMAX/Theta were never
run.  This script closes that gap under the project's LOCKED evaluation
protocol, so the numbers are directly comparable to:

    SeasonalNaive         MASE_ALL 1.03029   WAPE 0.10444
    LightGBMTuned         MASE_ALL 1.04108
    xLSTM ENS3            MASE_ALL 0.94810
    xLSTM + MinT_wls      MASE_ALL 0.94455
    iTransformer+anchor   MASE_ALL 0.90198   WAPE 0.08781

Protocol (identical to stage0/run_stage0.py)
--------------------------------------------
* train window ends at the origin (inclusive); no shuffling
* rolling origins  202312 .. 202411   (12 origins)
* horizons         h = 1, 3, 6, 12
* 23 series        -> 23 x 12 x 4 = 1104 test points
* metric           harness.metrics_from_frame (MASE denominator = TRAIN-period
                   seasonal-naive MAE), plus MinT_wls reconciliation reusing
                   run_stage4_reconcile.reconcile / build_A / naive_w

Each model is fitted ONCE per (origin, series) and forecast 12 steps ahead;
h = 1/3/6/12 are then sliced out.  That is a rolling-origin *direct* evaluation
and it is stated as such in the report.

Usage
-----
    stage0\\.venv\\Scripts\\python.exe stage0\\repro_stats.py ets
    stage0\\.venv\\Scripts\\python.exe stage0\\repro_stats.py theta
    stage0\\.venv\\Scripts\\python.exe stage0\\repro_stats.py sarimax
    stage0\\.venv\\Scripts\\python.exe stage0\\repro_stats.py all
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "eval"))
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import run_stage4_reconcile as R4  # noqa: E402

OUT = HERE / "results" / "repro"
ORIGINS = ["202312", "202401", "202402", "202403", "202404", "202405",
           "202406", "202407", "202408", "202409", "202410", "202411"]
HORIZONS = list(H.HORIZONS)          # (1, 3, 6, 12)
MAX_H = max(HORIZONS)                 # forecast 12 steps, then slice


# ---------------------------------------------------------------------------
# model wrappers: take a 1-D training series, return a length-12 forecast
# ---------------------------------------------------------------------------
def fit_ets(y: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    m = ExponentialSmoothing(
        np.asarray(y, dtype=float),
        trend="add", damped_trend=False,
        seasonal="add", seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=False)
    return np.asarray(m.forecast(MAX_H), dtype=float)


def fit_theta(y: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    m = ThetaModel(np.asarray(y, dtype=float), period=12, deseasonalize=True)
    res = m.fit()
    return np.asarray(res.forecast(MAX_H), dtype=float)


def fit_sarimax(y: np.ndarray) -> np.ndarray:
    """Airline-model-style fixed order.  NOT tuned on any test label."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    m = SARIMAX(
        np.asarray(y, dtype=float),
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = m.fit(disp=False, maxiter=200)
    return np.asarray(res.forecast(MAX_H), dtype=float)


FITTERS = {"ets": fit_ets, "theta": fit_theta, "sarimax": fit_sarimax,
           "ets_frozen": None}       # frozen is handled by run_model_frozen
LABELS = {
    "ets": "ETS_HW_add",
    "theta": "Theta_seas",
    "sarimax": "SARIMAX_111_0112",
    "ets_frozen": "ETS_HW_add_FROZEN",
}


# ---------------------------------------------------------------------------
# STRICT protocol: match the project's own setup exactly.
# The deep models were trained ONCE on <= 2023-12 and then frozen; at each
# origin they only see the actual history up to that origin through their input
# window.  The analogue for ETS is: estimate parameters ONCE on <= 2023-12, then
# at each origin re-filter the state on the extended series (res.apply) WITHOUT
# re-estimating any parameter.  Rolling refit (run_model) is the business-
# realistic alternative and is reported separately.
# ---------------------------------------------------------------------------
def run_model_frozen(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    label = LABELS["ets_frozen"]
    series = list(wide.columns)
    train_idx = [p for p in wide.index if p <= H.TRAIN_END]
    frozen = {}
    fails = []
    for s in series:
        y0 = wide.loc[train_idx, s].to_numpy(dtype=float)
        try:
            res = ExponentialSmoothing(
                y0, trend="add", seasonal="add", seasonal_periods=12,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
            # keep ONLY the three smoothing parameters; states are re-estimated
            # on the extended history at every origin (the analogues of the deep
            # models keeping their weights while their input window rolls).
            frozen[s] = {
                "smoothing_level": float(res.params["smoothing_level"]),
                "smoothing_trend": float(res.params["smoothing_trend"]),
                "smoothing_seasonal": float(res.params["smoothing_seasonal"]),
            }
        except Exception as exc:                          # noqa: BLE001
            fails.append((H.TRAIN_END, s, type(exc).__name__, str(exc)[:120]))

    rows = []
    t0 = time.perf_counter()
    for origin in ORIGINS:
        hist_idx = [p for p in wide.index if p <= origin]
        for s in series:
            if s not in frozen:
                continue
            y_now = wide.loc[hist_idx, s].to_numpy(dtype=float)
            try:
                m = ExponentialSmoothing(
                    y_now, trend="add", seasonal="add", seasonal_periods=12,
                    initialization_method="heuristic",
                ).fit(optimized=False, **frozen[s])
                fc = np.asarray(m.forecast(MAX_H), dtype=float)
            except Exception as exc:                      # noqa: BLE001
                fails.append((origin, s, "refilter:" + type(exc).__name__, str(exc)[:120]))
                fc = np.full(MAX_H, np.nan)
            if not np.isfinite(fc).all():
                last = float(y_now[-1])
                fc = np.where(np.isfinite(fc), fc, last)
            for h in HORIZONS:
                target = H.month_add(origin, h)
                if target not in wide.index:
                    continue
                rows.append((label, origin, h, s, float(wide.loc[target, s]), float(fc[h - 1])))

    df = pd.DataFrame(rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])
    meta = {"model": label, "seconds": round(time.perf_counter() - t0, 2),
            "n_points": int(len(df)), "n_fail": len(fails),
            "fail_examples": fails[:10], "protocol": "params estimated <=2023-12, state re-filtered per origin"}
    return df, meta


# ---------------------------------------------------------------------------
def run_model(key: str, wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fit = FITTERS[key]
    label = LABELS[key]
    series = list(wide.columns)
    rows, fails = [], []
    t0 = time.perf_counter()

    for oi, origin in enumerate(ORIGINS, 1):
        train_idx = [p for p in wide.index if p <= origin]
        for s in series:
            y = wide.loc[train_idx, s].to_numpy(dtype=float)
            try:
                fc = fit(y)
            except Exception as exc:                      # noqa: BLE001
                fails.append((origin, s, type(exc).__name__, str(exc)[:120]))
                fc = np.full(MAX_H, np.nan)
            if not np.isfinite(fc).all():
                fails.append((origin, s, "nonfinite", "nan in forecast"))
                last = float(y[-1])
                fc = np.where(np.isfinite(fc), fc, last)
            for h in HORIZONS:
                target = H.month_add(origin, h)
                if target not in wide.index:
                    continue
                rows.append((label, origin, h, s, float(wide.loc[target, s]), float(fc[h - 1])))
        print(f"  [{label}] origin {origin} done ({oi}/{len(ORIGINS)})  "
              f"elapsed {time.perf_counter() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])
    meta = {"model": label, "seconds": round(time.perf_counter() - t0, 2),
            "n_points": int(len(df)), "n_fail": len(fails),
            "fail_examples": fails[:10]}
    return df, meta


def add_mint(df: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """MinT_wls on the 1104-point panel, reusing the project's own functions."""
    series = list(wide.columns)
    A = R4.build_A(series)
    W = np.diag(R4.naive_w(wide, series, H.TRAIN_END))

    piv = df.pivot_table(index=["origin", "h"], columns="series",
                         values="y_pred", aggfunc="first").reindex(columns=series)
    vecs = piv.to_numpy(dtype=float)
    rec = R4.reconcile(vecs, A, W)
    out = df.copy()
    recdf = pd.DataFrame(rec, index=piv.index,
                         columns=pd.Index(series, name="series"))
    lookup = recdf.stack().rename("y_pred_rec").reset_index()
    lookup.columns = ["origin", "h", "series", "y_pred_rec"]
    out = out.merge(lookup, on=["origin", "h", "series"], how="left")
    out["y_pred"] = out["y_pred_rec"]
    out = out.drop(columns=["y_pred_rec"])
    out["model"] = out["model"] + "+MinT_wls"
    return out


def main(argv: list[str]) -> int:
    keys = argv[1:] or ["all"]
    if keys == ["all"]:
        keys = ["ets", "theta", "ets_frozen"]
    keys = [k for k in keys if k in LABELS]
    if not keys:
        print(f"usage: repro_stats.py [{'|'.join(LABELS)}|all]")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    wide = H.load_panel()
    naive_mae = H.seasonal_naive_mae_train(wide)
    print(f"panel {wide.shape[0]} x {wide.shape[1]}   origins {ORIGINS[0]}..{ORIGINS[-1]}")

    metas = []
    for key in keys:
        print(f"\n=== {LABELS[key]} ===", flush=True)
        if key == "ets_frozen":
            df, meta = run_model_frozen(wide)
        else:
            df, meta = run_model(key, wide)
        df.to_csv(OUT / f"repro_preds_{key}.csv", index=False)
        drec = add_mint(df, wide)
        drec.to_csv(OUT / f"repro_preds_{key}_mint.csv", index=False)
        allp = pd.concat([df, drec], ignore_index=True)
        m = H.metrics_from_frame(allp, naive_mae)
        m.to_csv(OUT / f"repro_metrics_{key}.csv", index=False)
        print(m[m["h"] == "ALL"].to_string(index=False))
        meta["mase_all"] = float(m.loc[(m["model"] == meta["model"]) & (m["h"] == "ALL"), "MASE"].iloc[0])
        meta["mase_all_mint"] = float(m.loc[(m["model"] == meta["model"] + "+MinT_wls") & (m["h"] == "ALL"), "MASE"].iloc[0])
        meta["wape_all"] = float(m.loc[(m["model"] == meta["model"]) & (m["h"] == "ALL"), "WAPE"].iloc[0])
        metas.append(meta)
        print(f"  seconds={meta['seconds']}  fails={meta['n_fail']}", flush=True)

    pd.DataFrame(metas).to_csv(OUT / f"repro_meta_{'_'.join(keys)}.csv", index=False)
    print("\nmeta:", metas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
