"""
AUDIT step 2 — materialise per-point predictions for the five addendum
baselines that until now only had aggregate metrics on disk.

Every call goes through `models/baselines.py` unchanged, with
target_mode="residual" (the form Stage 0 establishes as the defensible one).
The learner hyper-parameters for LightGBMTuned are read from the ALREADY
PERSISTED train-only tuning result (results/extra_lgb_tuning.json), so nothing
is re-tuned here and no 2024 label enters any choice.

Output: results/extra_preds_<model>.csv   (same 6 columns as the xLSTM files)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H           # noqa: E402
from models import baselines as B       # noqa: E402

RES = ROOT / "results"
COLS = ["model", "origin", "h", "series", "y_true", "y_pred"]


def main() -> int:
    feat = H.load_features()
    wide = H.load_panel()
    scaling = pd.read_csv(ROOT / "data" / "scaling.csv", dtype={"series": str})

    tuning = json.loads((RES / "extra_lgb_tuning.json").read_text())
    params_by_h = {int(k): v for k, v in tuning.items()}
    print("LightGBMTuned params (persisted, train-only tuning):")
    for h, p in sorted(params_by_h.items()):
        print(f"  h={h:>2}  {p}")

    jobs = [
        ("kNN_residual", lambda: B.run_knn(
            feat, wide, scaling, tag="kNN_residual", target_mode="residual")),
        ("KMeansRidge_residual", lambda: B.run_kmeans_ridge(
            feat, wide, scaling, tag="KMeansRidge_residual", target_mode="residual")),
        ("NLinear_residual", lambda: B.run_nlinear(
            feat, wide, scaling, tag="NLinear_residual", target_mode="residual")),
        ("DLinear_residual", lambda: B.run_dlinear(
            feat, wide, scaling, tag="DLinear_residual", target_mode="residual")),
        ("LightGBMTuned_residual", lambda: B.run_lightgbm_tuned(
            feat, wide, scaling, tag="LightGBMTuned_residual",
            target_mode="residual", params_by_h=params_by_h, seed=42)),
    ]

    naive_mae = H.seasonal_naive_mae_train(wide)
    summary = []
    for tag, fn in jobs:
        t0 = time.time()
        df = fn()
        dt = time.time() - t0
        df = df[COLS].copy()
        df["series"] = df["series"].astype(str)
        df["origin"] = df["origin"].astype(str)
        out = RES / f"extra_preds_{tag}.csv"
        df.to_csv(out, index=False)
        m = H.metrics_from_frame(df, naive_mae)
        m["sec"] = round(dt, 1)
        m["file"] = out.name
        summary.append(m)
        print(f"  {tag:24s} {len(df)} rows  {dt:6.1f}s  -> {out.name}")

    summ = pd.concat(summary, ignore_index=True)
    summ.to_csv(RES / "extra_preds_metrics.csv", index=False)
    print("\nrecomputed metrics (should match results/stage0_baselines_extra.csv):")
    piv = summ.pivot_table(index="model", columns="h", values="MASE", aggfunc="first")
    print(piv[[1, 3, 6, 12, "ALL"]].round(5).to_string() if set([1, 3, 6, 12, "ALL"]) <= set(piv.columns)
          else piv.round(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
