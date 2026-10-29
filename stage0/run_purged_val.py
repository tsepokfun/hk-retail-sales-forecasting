"""
Stage 0 / v2.3 — validation-set LEAK repair + honest baseline.

WHY THIS EXISTS
---------------
The Stage-0 baseline trained with `val_end = cutoff` (2023-12), so validation
origins ran 2021-12 .. 2023-12.  Because the target array `Y` covers all 263
months (2004-10 .. 2026-08), a validation origin at 2023-12 with h=12 pulls its
target from **2024-12**, i.e. a TEST-period month.  Early stopping therefore
selected the best epoch using (partly) the test set.

Measured leak in the shipped baseline (verified by direct cube inspection):
    validation origins scanned            25   (2021-12 .. 2023-12)
    origins with >=1 target in 2024       12
    validation samples with a 2024 target 276 / 575 = 48.0%
    origin 2023-12 leaks ALL FOUR heads
    per head: h=12 leaks in 12 origins, h=6 in 5, h=3 in 3, h=1 in 1

This script re-runs the champion configuration under the CORRECT protocol
(`purge_val=True`: a validation origin is legal only when *every* horizon's
target still lies inside the training period, i.e. origin <= 2023-12 - 12, so
val origins 2021-12 .. 2022-12 = 13 origins / 299 samples) and reports leaky vs
purged side by side, including a Diebold-Mariano test.

RESUMABLE
---------
Each (protocol, seed) run is written to its own file the moment it finishes:
    results/pv_preds/<tag>.csv      results/pv_meta/<tag>.json
A finished tag is skipped on re-run, so a kill or reboot does not throw away
completed fits.  Delete a per-tag file to force that one to re-run.
`--combine-only` skips fitting entirely and just rebuilds the summary tables.

Outputs
    results/purged_val_predictions.csv     combined predictions
    results/stage0_purged_val.csv          metrics per (model, h)
    results/stage0_purged_val_perseed.csv  metrics per (tag, ALL horizon)
    results/stage0_purged_val_agg.csv      mean +/- std across seeds
    results/stage0_purged_val_dm.csv       DM: leaky (A) vs purged (B)
    results/stage0_purged_val_meta.json    per-run training metadata

Usage
    .venv\\Scripts\\python.exe run_purged_val.py
    .venv\\Scripts\\python.exe run_purged_val.py --seeds 42 7 2024 --epochs 100
    .venv\\Scripts\\python.exe run_purged_val.py --combine-only
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

from eval import harness as H          # noqa: E402
from models import xlstm_direct as X   # noqa: E402

RES = ROOT / "results"
PRED_DIR = RES / "pv_preds"
META_DIR = RES / "pv_meta"


def load_all():
    feat = pd.read_csv(ROOT / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(ROOT / "data" / "scaling.csv", dtype={"series": str})
    return feat, wide, scaling


def fit_one(feat, wide, scaling, proto: str, seed: int, epochs: int):
    pv = (proto == "purged")
    tag = f"xLSTM_{proto}_s{seed}"
    print("\n" + "=" * 78, flush=True)
    print(f"  tag={tag}   purge_val={pv}   seed={seed}   epochs={epochs}", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    # NOTE: passing `seed=` is REQUIRED.  `X.set_seed(seed)` alone is useless,
    # because train_model() calls set_seed() again with the module default and
    # silently overwrites it -- which is how the first version of this script
    # produced three "different" seeds with byte-identical results.
    df, meta = X.run(feat, wide, scaling, epochs=epochs, tag=tag,
                     purge_val=pv, verbose=True, seed=seed)
    df["seed"] = seed
    df["protocol"] = proto
    meta = dict(meta)
    meta.update({"protocol": proto, "seed": seed, "tag": tag,
                 "train_sec": round(time.time() - t0, 1)})
    df.to_csv(PRED_DIR / f"{tag}.csv", index=False)
    (META_DIR / f"{tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved {tag}  ({len(df):,} rows, {meta['train_sec']}s)", flush=True)
    return df, meta


def combine(naive_mae) -> int:
    files = sorted(PRED_DIR.glob("*.csv"))
    if not files:
        print("no per-tag predictions found -- nothing to combine")
        return 1
    pred = pd.concat([pd.read_csv(f, dtype={"series": str, "period": str})
                      for f in files], ignore_index=True)
    pred.to_csv(RES / "purged_val_predictions.csv", index=False)

    res = H.metrics_from_frame(pred, naive_mae)
    res.to_csv(RES / "stage0_purged_val.csv", index=False)

    rows = []
    for tag, g in pred.groupby("model", sort=False):
        one = H.metrics_from_frame(g, naive_mae)
        allrow = one[one["h"].astype(str) == "ALL"].iloc[0]
        rows.append({"tag": tag,
                     "protocol": g["protocol"].iloc[0],
                     "seed": int(g["seed"].iloc[0]),
                     "MASE_ALL": float(allrow["MASE"]),
                     "WAPE_ALL": float(allrow["WAPE"]),
                     "bias_pct": float(allrow["bias_pct"]),
                     "n_points": int(allrow["n_points"])})
    per_seed = pd.DataFrame(rows).sort_values(["protocol", "seed"])
    per_seed.to_csv(RES / "stage0_purged_val_perseed.csv", index=False)

    print("\n" + "=" * 78)
    print("  PER-RUN  (ALL-horizon row)")
    print("=" * 78)
    print(per_seed.to_string(index=False))

    agg = []
    for proto, g in per_seed.groupby("protocol"):
        for col in ("MASE_ALL", "WAPE_ALL"):
            v = g[col].to_numpy(dtype=float)
            agg.append({"protocol": proto, "metric": col, "n_seeds": len(v),
                        "mean": round(float(np.mean(v)), 5),
                        "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, 5),
                        "min": round(float(np.min(v)), 5),
                        "max": round(float(np.max(v)), 5)})
    agg = pd.DataFrame(agg)
    print("\n  --- mean +/- std across seeds ---")
    print(agg.to_string(index=False))
    agg.to_csv(RES / "stage0_purged_val_agg.csv", index=False)

    print("\n" + "=" * 78)
    print("  DIEBOLD-MARIANO   A = leaky-val   B = purged-val   (power=2)")
    print("  dm > 0  =>  A has the higher loss  =>  PURGED is better")
    print("=" * 78)
    dm_rows = []
    for h in H.HORIZONS:
        a = pred[(pred["protocol"] == "leaky") & (pred["h"] == h)]
        b = pred[(pred["protocol"] == "purged") & (pred["h"] == h)]
        if a.empty or b.empty:
            print(f"  h={h}: missing a side, skipped")
            continue
        ea = (a["y_true"] - a["y_pred"]).groupby([a["origin"], a["series"]]).mean()
        eb = (b["y_true"] - b["y_pred"]).groupby([b["origin"], b["series"]]).mean()
        idx = ea.index.intersection(eb.index)
        if len(idx) < 10:
            print(f"  h={h}: only {len(idx)} overlapping points, skipped")
            continue
        r = H.dm_test(ea.loc[idx].to_numpy(), eb.loc[idx].to_numpy(), h=h, power=2)
        r.update({"h": h})
        dm_rows.append(r)
    dm = pd.DataFrame(dm_rows)
    print(dm.to_string(index=False))
    dm.to_csv(RES / "stage0_purged_val_dm.csv", index=False)

    print("\n" + "=" * 78)
    print("  HEADLINE")
    print("=" * 78)
    for proto, g in per_seed.groupby("protocol"):
        v = g["MASE_ALL"].to_numpy(dtype=float)
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        print(f"    {proto:7s}  MASE_ALL  mean {v.mean():.5f}  std {sd:.5f}  "
              f"range {v.min():.4f}..{v.max():.4f}  (n={len(v)})")

    metas = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(META_DIR.glob("*.json"))]
    (RES / "stage0_purged_val_meta.json").write_text(
        json.dumps(metas, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--only", default=None,
                    help="fit exactly one tag, e.g. --only purged:7 "
                         "(used by the PowerShell driver so every fit gets a "
                         "fresh CUDA context; this box has shown transient "
                         "CUDA OOM / 'unknown error' when several fits share one)")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    feat, wide, scaling = load_all()
    naive_mae = H.seasonal_naive_mae_train(wide)

    if args.only:
        proto, seed_s = args.only.split(":")
        tag = f"xLSTM_{proto}_s{seed_s}"
        if (PRED_DIR / f"{tag}.csv").exists():
            print(f"  [skip] {tag} already done", flush=True)
            return 0
        fit_one(feat, wide, scaling, proto, int(seed_s), args.epochs)
        return 0

    if not args.combine_only:
        t_all = time.time()
        for proto in ("leaky", "purged"):
            for seed in args.seeds:
                tag = f"xLSTM_{proto}_s{seed}"
                if (PRED_DIR / f"{tag}.csv").exists():
                    print(f"  [skip] {tag} already done", flush=True)
                    continue
                fit_one(feat, wide, scaling, proto, seed, args.epochs)
        print(f"\n  fitting wall time {time.time() - t_all:.0f}s", flush=True)

    return combine(naive_mae)


if __name__ == "__main__":
    raise SystemExit(main())
