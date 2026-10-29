"""
Stage 0 — orchestrator.

  python run_stage0.py --models naive,drift,ridge,lgbm
  python run_stage0.py --models all

Writes:
  results/predictions.csv        every (model, origin, h, series) prediction
  results/stage0_results.csv     metrics table (MASE / WAPE / RMSE / MAE)
  results/stage0_janfeb.csv      pooled 2024 Jan+Feb WAPE
  results/stage0_summary.txt     human-readable report
  results/stage0_meta.json       run metadata (params, param counts, timings)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "eval"))
sys.path.insert(0, str(HERE / "models"))

import harness as H  # noqa: E402
import baselines as B  # noqa: E402

RES = HERE / "results"
RES.mkdir(parents=True, exist_ok=True)


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                    help="comma list of naive,drift,ridge,lgbm,xlstm or 'all'")
    ap.add_argument("--epochs", type=int, default=60, help="xLSTM epochs per fit")
    ap.add_argument("--refit", action="store_true",
                    help="refit supervised models at every origin (default: refit per origin)")
    args = ap.parse_args()

    wanted = ([m.strip() for m in args.models.split(",")] if args.models != "all"
              else ["naive", "drift", "ridge", "lgbm", "xlstm"])

    t_start = time.time()
    banner("STAGE 0 — rolling-origin evaluation")
    print(f"  models requested : {wanted}")
    print(f"  origins          : {len(H.ORIGINS)}  ({H.ORIGINS[0]} .. {H.ORIGINS[-1]})")
    print(f"  horizons         : {H.HORIZONS}")
    print(f"  train end        : {H.TRAIN_END}")

    wide = H.load_panel()
    feat = H.load_features()
    scaling = pd.read_csv(HERE / "data" / "scaling.csv", dtype={"series": str})
    naive_mae = H.seasonal_naive_mae_train(wide)

    print(f"  panel            : {wide.shape[0]} periods x {wide.shape[1]} series")
    print(f"  feature rows     : {len(feat):,}")
    print(f"  naive MAE (train): median {naive_mae.median():,.1f} HK$m")

    frames = []
    meta = {"horizons": list(H.HORIZONS), "origins": H.ORIGINS,
            "train_end": H.TRAIN_END, "models": {}, "epochs": args.epochs}

    # ---------------- traditional baselines -------------------------------
    if "naive" in wanted:
        banner("Seasonal Naive")
        t = time.time()
        df = B.run_seasonal_naive(wide)
        frames.append(df)
        print(f"  {len(df):,} prediction rows  ({time.time() - t:.1f}s)")

    if "drift" in wanted:
        banner("Drift")
        t = time.time()
        df = B.run_drift(wide)
        frames.append(df)
        print(f"  {len(df):,} prediction rows  ({time.time() - t:.1f}s)")

    if "ridge" in wanted:
        banner("Ridge  (both target formulations)")
        t = time.time()
        for mode, tag in (("level", "Ridge_level"), ("residual", "Ridge_residual")):
            df = B.run_ridge(feat, wide, scaling, target_mode=mode, tag=tag)
            frames.append(df)
            print(f"  {tag:18s} {len(df):,} rows   ({time.time() - t:.1f}s)")

    if "lgbm" in wanted:
        banner("LightGBM  (both target formulations)")
        t = time.time()
        for mode, tag in (("level", "LightGBM_level"), ("residual", "LightGBM_residual")):
            df = B.run_lightgbm(feat, wide, scaling, target_mode=mode, tag=tag)
            frames.append(df)
            print(f"  {tag:18s} {len(df):,} rows   ({time.time() - t:.1f}s)")

    # ---------------- xLSTM ------------------------------------------------
    if "xlstm" in wanted:
        banner("xLSTM (Direct Multi-Horizon)")
        try:
            import xlstm_direct as X
            t = time.time()
            df, xmeta = X.run(feat, wide, scaling, epochs=args.epochs)
            frames.append(df)
            meta["models"]["xlstm"] = xmeta
            print(f"  {len(df):,} prediction rows  ({time.time() - t:.1f}s)")
        except Exception as e:
            import traceback
            print("  xLSTM FAILED:")
            traceback.print_exc()
            print("  -> continuing with the baselines that succeeded")

    # ---------------- metrics ---------------------------------------------
    banner("Metrics")
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(RES / "predictions.csv", index=False, encoding="utf-8")
    print(f"  predictions.csv : {len(preds):,} rows")

    tbl = H.metrics_from_frame(preds, naive_mae)
    tbl.to_csv(RES / "stage0_results.csv", index=False, encoding="utf-8")

    jf = H.janfeb_pooled(preds)
    jf.to_csv(RES / "stage0_janfeb.csv", index=False, encoding="utf-8")

    dm = H.dm_table(preds, baseline="SeasonalNaive")
    dm.to_csv(RES / "stage0_dm.csv", index=False, encoding="utf-8")

    # ---------------- console report ---------------------------------------
    banner("MASE by model x horizon  (lower is better; <1 beats seasonal naive)")
    piv = tbl.pivot(index="model", columns="h", values="MASE")
    order_h = list(H.HORIZONS) + ["ALL"]
    piv = piv[[c for c in order_h if c in piv.columns]]
    print(piv.round(4).to_string())

    banner("WAPE by model x horizon")
    piv2 = tbl.pivot(index="model", columns="h", values="WAPE")
    piv2 = piv2[[c for c in order_h if c in piv2.columns]]
    print((piv2 * 100).round(2).to_string())

    banner("2024 Jan+Feb pooled WAPE (%)  — the honest Jan/Feb comparison")
    if len(jf):
        pj = jf.pivot(index="model", columns="h", values="janfeb_WAPE")
        pj = pj[[c for c in order_h if c in pj.columns]]
        print((pj * 100).round(2).to_string())
    else:
        print("  (none)")

    banner("Diebold-Mariano vs SeasonalNaive  (p<0.05 = significant)")
    if len(dm):
        print(f"  {'model':22s} {'h':>3s} | {'DM_sq':>8s} {'p_sq':>8s} {'win':>9s} | "
              f"{'DM_abs':>8s} {'p_abs':>8s} {'win':>9s}")
        for r in dm.itertuples(index=False):
            s1 = " *" if r.sig_sq else "  "
            s2 = " *" if r.sig_abs else "  "
            print(f"  {r.model:22s} {r.h:>3} | {r.DM_sq:>8.4f} {r.p_sq:>8.5f} {str(r.better_sq):>9s}{s1} | "
                  f"{r.DM_abs:>8.4f} {r.p_abs:>8.5f} {str(r.better_abs):>9s}{s2}")
    else:
        print("  (none)")

    # ---------------- write summary ----------------------------------------
    lines = []
    lines.append("STAGE 0 — rolling-origin evaluation summary")
    lines.append(f"models   : {sorted(preds['model'].unique())}")
    lines.append(f"origins  : {H.ORIGINS[0]} .. {H.ORIGINS[-1]}  ({len(H.ORIGINS)})")
    lines.append(f"horizons : {H.HORIZONS}")
    lines.append(f"train end: {H.TRAIN_END}")
    lines.append("")
    lines.append("MASE (lower better, <1 beats seasonal naive)")
    lines.append(piv.round(4).to_string())
    lines.append("")
    lines.append("WAPE %")
    lines.append((piv2 * 100).round(2).to_string())
    lines.append("")
    lines.append("2024 Jan+Feb pooled WAPE %")
    if len(jf):
        lines.append((pj * 100).round(2).to_string())
    lines.append("")
    lines.append("")
    lines.append("Diebold-Mariano vs SeasonalNaive (p<0.05 = significant)")
    if len(dm):
        lines.append(dm.to_string(index=False))
    lines.append("")
    lines.append(tbl.to_string(index=False))
    (RES / "stage0_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    meta["elapsed_sec"] = round(time.time() - t_start, 1)
    meta["n_predictions"] = int(len(preds))
    (RES / "stage0_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    banner("WROTE")
    for f in ("predictions.csv", "stage0_results.csv", "stage0_janfeb.csv", "stage0_dm.csv",
              "stage0_summary.txt", "stage0_meta.json"):
        p = RES / f
        print(f"  {f:24s} {p.stat().st_size:>10,} B")
    print(f"\n  total elapsed: {meta['elapsed_sec']}s")
    print("\nSTAGE 0 OK — first model trained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
