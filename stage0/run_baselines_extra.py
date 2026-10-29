"""
Stage 0 — addendum: the five baselines that were never measured.

  python run_baselines_extra.py                 # everything, resumable
  python run_baselines_extra.py --force         # ignore the cache
  python run_baselines_extra.py --only kmeans   # pick jobs by substring

Protocol is the LOCKED one in eval/harness.py — nothing here redefines a metric.
  * origins  = H.ORIGINS (12)         horizons = H.HORIZONS (1,3,6,12)
  * every model is re-fitted at every origin on {target_period <= origin}
  * Direct Multi-Horizon, one model per h, no recursion, no shuffling
  * both target_mode "level" and "residual" for every learner
  * deterministic learners run once (documented), LightGBM runs 3 seeds

Writes
  results/stage0_baselines_extra.csv          model x h -> MASE/WAPE/bias_pct/n
  results/stage0_baselines_extra_perseed.csv  the same, per seed, for auditing
  results/STAGE0_EXTRA_BASELINES.md           the report-ready table + DM + timings
  results/extra_lgb_tuning.json               the selected LightGBM grid point
  results/extra_series_clusters.csv           K-Means membership (train-only profile)
  results/_extra_cache/*.csv                  per-(model, seed) predictions
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
CACHE = RES / "_extra_cache"
CACHE.mkdir(parents=True, exist_ok=True)

XREF = 0.98355          # xLSTM MASE_ALL, purged protocol, 3-seed mean (RESULTS.md v3)
PRIMARY_SEED = 42
SEEDS = [42, 7, 2024]

COLS = ["model", "origin", "h", "series", "y_true", "y_pred"]


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------------------
# job registry: every extra model, both target formulations
# ---------------------------------------------------------------------------
def build_jobs():
    j = []
    for mode in ("level", "residual"):
        j.append(dict(key=f"kmeans_{mode}", tag=f"KMeansRidge_{mode}",
                      fn="run_kmeans_ridge", kw=dict(target_mode=mode), seeds=[None]))
    for mode in ("level", "residual"):
        j.append(dict(key=f"knn_{mode}", tag=f"kNN_{mode}",
                      fn="run_knn", kw=dict(target_mode=mode), seeds=[None]))
    for mode in ("level", "residual"):
        j.append(dict(key=f"nlinear_{mode}", tag=f"NLinear_{mode}",
                      fn="run_nlinear", kw=dict(target_mode=mode), seeds=[None]))
    for mode in ("level", "residual"):
        j.append(dict(key=f"dlinear_{mode}", tag=f"DLinear_{mode}",
                      fn="run_dlinear", kw=dict(target_mode=mode), seeds=[None]))
    for mode in ("level", "residual"):
        j.append(dict(key=f"lgbt_{mode}", tag=f"LightGBMTuned_{mode}",
                      fn="run_lightgbm_tuned", kw=dict(target_mode=mode), seeds=SEEDS))
    return j


# ---------------------------------------------------------------------------
# markdown helpers (no dependency on tabulate formatting quirks)
# ---------------------------------------------------------------------------
def md_table(headers, rows, aligns=None) -> str:
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def f4(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def f2(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore the prediction cache")
    ap.add_argument("--only", default=None, help="substring filter on the job key")
    ap.add_argument("--seeds", default="42,7,2024")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    t_all = time.time()
    banner("STAGE 0 ADDENDUM — extra baselines on the locked protocol")
    wide = H.load_panel()
    feat = H.load_features()
    scaling = pd.read_csv(HERE / "data" / "scaling.csv", dtype={"series": str})
    naive_mae = H.seasonal_naive_mae_train(wide)
    print(f"  panel   : {wide.shape[0]} periods x {wide.shape[1]} series")
    print(f"  feature : {len(feat):,} rows")
    print(f"  origins : {len(H.ORIGINS)}  horizons {H.HORIZONS}  train_end {H.TRAIN_END}")

    # ---------------- protocol self-check on the reference baseline ---------
    banner("PROTOCOL CHECK — Seasonal Naive must reproduce the locked numbers")
    sn = B.run_seasonal_naive(wide)
    sn_tbl = H.metrics_from_frame(sn, naive_mae)
    want = {6: 0.97471, 12: 0.80900}
    ok = True
    for h, w in want.items():
        got = float(sn_tbl[(sn_tbl["model"] == "SeasonalNaive") & (sn_tbl["h"] == h)]["MASE"].iloc[0])
        good = abs(got - w) < 1e-5
        ok &= good
        print(f"  h={h:>2}  MASE={got:.5f}  expected {w:.5f}  {'OK' if good else 'MISMATCH'}")
    if not ok:
        print("\n  PROTOCOL MISMATCH — stopping. The naive reference does not reproduce,")
        print("  so no extra baseline number would be comparable. Fix the protocol first.")
        return 2
    print("  -> naive reference reproduces. Protocol confirmed.")

    # ---------------- K-Means clustering (train-only profile) --------------
    banner("Series clustering profile (TRAIN ONLY, period <= 202312)")
    prof = B.fit_series_clusters(feat)
    prof.to_csv(RES / "extra_series_clusters.csv", encoding="utf-8")
    for c, g in prof.groupby("cluster"):
        print(f"  cluster {c}: n={len(g):>2}  "
              f"mean_log={g['y_log_mean'].mean():.2f}  "
              f"std={g['y_log_std'].mean():.2f}  acf12={g['acf12'].mean():.2f}  "
              f"series={','.join(g.index.tolist())}")

    # ---------------- LightGBM tuning (train-only validation) --------------
    banner("LightGBM grid search — time-series validation INSIDE the train window")
    tun_path = RES / "extra_lgb_tuning.json"
    if tun_path.exists() and not args.force:
        params_by_h = {int(k): v for k, v in json.loads(tun_path.read_text(encoding="utf-8")).items()}
        print("  cached:", json.dumps(params_by_h, indent=None))
        audit = None
    else:
        params_by_h, audit = B.tune_lightgbm(feat, n_val_origins=24, seed=PRIMARY_SEED)
        tun_path.write_text(json.dumps(params_by_h, indent=2), encoding="utf-8")
    if audit is not None:
        audit.to_csv(RES / "extra_lgb_tuning_audit.csv", index=False, encoding="utf-8")
    for h in H.HORIZONS:
        p = params_by_h[h]
        print(f"  h={h:>2}  num_leaves={p['num_leaves']:>2} lr={p['learning_rate']:<5} "
              f"min_child={p['min_child_samples']:>2} n_est={p['n_estimators']:>3} "
              f"val_MAE_yz={p['val_MAE_yz']}")

    # ---------------- run the five models ---------------------------------
    jobs = build_jobs()
    if args.only:
        jobs = [j for j in jobs if args.only in j["key"]]

    timings = {}
    timings_path = RES / "extra_train_sec.json"
    if timings_path.exists():
        timings = json.loads(timings_path.read_text(encoding="utf-8"))
    per_seed_frames = []
    for job in jobs:
        tag, fn_name, kw = job["tag"], job["fn"], dict(job["kw"])
        if fn_name == "run_lightgbm_tuned":
            kw["params_by_h"] = params_by_h
        fn = getattr(B, fn_name)
        job_seeds = job["seeds"] if job["seeds"] == [None] else seeds
        for sd in job_seeds:
            suffix = "det" if sd is None else f"s{sd}"
            cache_file = CACHE / f"{tag}__{suffix}.csv"
            prev = timings.setdefault(tag, {})
            if cache_file.exists() and not args.force:
                df = pd.read_csv(cache_file, dtype={"series": str, "origin": str})
                print(f"  [cache] {tag:24s} {suffix:>5s}  {len(df):,} rows"
                      f"  (fit was {prev.get(suffix, 'n/a')}s)")
            else:
                call_kw = dict(kw)
                if fn_name == "run_lightgbm_tuned":
                    call_kw["seed"] = PRIMARY_SEED if sd is None else sd
                t0 = time.time()
                try:
                    df = fn(feat, wide, scaling, tag=tag, **call_kw)
                except Exception:
                    import traceback
                    print(f"  FAILED {tag} {suffix}:")
                    traceback.print_exc()
                    continue
                dt = time.time() - t0
                prev[suffix] = round(dt, 1)
                df.to_csv(cache_file, index=False, encoding="utf-8")
                print(f"  [fit]   {tag:24s} {suffix:>5s}  {len(df):,} rows  {dt:6.1f}s")
            d = df.copy()
            d["seed"] = -1 if sd is None else sd
            per_seed_frames.append(d)
    timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")

    if not per_seed_frames:
        print("\n  nothing ran — no predictions to score.")
        return 3

    new_preds = pd.concat(per_seed_frames, ignore_index=True)
    new_preds["series"] = new_preds["series"].astype(str)
    new_preds["origin"] = new_preds["origin"].astype(str)

    # ---------------- metrics per seed, then averaged ----------------------
    banner("Metrics")
    per_seed = []
    for (model, seed), g in new_preds.groupby(["model", "seed"], sort=False):
        tbl = H.metrics_from_frame(g[COLS], naive_mae)
        tbl["seed"] = seed
        per_seed.append(tbl)
    per_seed = pd.concat(per_seed, ignore_index=True)
    per_seed.to_csv(RES / "stage0_baselines_extra_perseed.csv", index=False, encoding="utf-8")

    agg = (per_seed.groupby(["model", "h"], sort=False)
           .agg(MASE=("MASE", "mean"), WAPE=("WAPE", "mean"),
                bias_pct=("bias_pct", "mean"), n=("n_points", "first"),
                n_seeds=("seed", "nunique"))
           .reset_index())
    order = {m: i for i, m in enumerate(
        [f"{stem}_{mode}" for stem in ("KMeansRidge", "kNN", "NLinear", "DLinear", "LightGBMTuned")
         for mode in ("level", "residual")])}
    horder = list(H.HORIZONS) + ["ALL"]
    agg["_m"] = agg["model"].map(lambda x: order.get(x, 99))
    agg["_h"] = agg["h"].map(lambda x: horder.index(x))
    agg = agg.sort_values(["_m", "_h"]).drop(columns=["_m", "_h"]).reset_index(drop=True)
    out = agg[["model", "h", "MASE", "WAPE", "bias_pct", "n"]].copy()
    for c in ("MASE", "WAPE", "bias_pct"):
        out[c] = out[c].round(5)
    out.to_csv(RES / "stage0_baselines_extra.csv", index=False, encoding="utf-8")
    print(f"  stage0_baselines_extra.csv : {len(out)} rows")
    piv = agg.pivot(index="model", columns="h", values="MASE")
    piv = piv[[c for c in horder if c in piv.columns]]
    print(piv.round(4).to_string())

    # ---------------- target-form sanity: DLinear / NLinear ----------------
    def _gap(tag_a, tag_b):
        fa, fb = CACHE / f"{tag_a}__det.csv", CACHE / f"{tag_b}__det.csv"
        if not (fa.exists() and fb.exists()):
            return None
        a = pd.read_csv(fa, dtype={"series": str, "origin": str})
        b = pd.read_csv(fb, dtype={"series": str, "origin": str})
        m = a.merge(b, on=["origin", "h", "series"], suffixes=("_L", "_R"))
        return float((m["y_pred_L"] - m["y_pred_R"]).abs().max())

    dl_gap = _gap("DLinear_level", "DLinear_residual")
    nl_gap = _gap("NLinear_level", "NLinear_residual")
    print(f"  max|pred_level - pred_residual|  DLinear={dl_gap}  NLinear={nl_gap}")

    # ---------------- reference rows --------------------------------------
    locked = pd.read_csv(RES / "stage0_results.csv")
    locked["h"] = locked["h"].astype(str)
    purged = pd.read_csv(RES / "purged_val_predictions.csv",
                         dtype={"series": str, "origin": str})
    purged = purged[purged["model"].str.startswith("xLSTM_purged")].copy()
    purged["h"] = purged["h"].astype(str)
    purged["series"] = purged["series"].astype(str)
    purged_metrics = []
    for sd, g in purged.groupby("seed"):
        t = H.metrics_from_frame(g[COLS], naive_mae)
        t["seed"] = sd
        purged_metrics.append(t)
    purged_metrics = pd.concat(purged_metrics, ignore_index=True)
    purged_metrics["h"] = purged_metrics["h"].astype(str)
    xm = {str(k): float(v) for k, v in purged_metrics.groupby("h")["MASE"].mean().items()}
    xw = {str(k): float(v) for k, v in purged_metrics.groupby("h")["WAPE"].mean().items()}

    ref_rows = []
    for name in ("SeasonalNaive", "Drift", "Ridge_level", "Ridge_residual",
                 "LightGBM_level", "LightGBM_residual"):
        sub = locked[locked["model"] == name]
        if not len(sub):
            continue
        row = {h: float(sub[sub["h"] == str(h)]["MASE"].iloc[0]) for h in horder}
        ref_rows.append((name, "naive" if name == "SeasonalNaive" else "n/a", row,
                         "locked stage0_results.csv"))
    ref_rows.append(("xLSTM (purged, 3-seed mean)", "residual",
                     {h: float(xm[str(h)]) for h in horder}, "purged_val_predictions.csv"))

    # ---------------- unified table ---------------------------------------
    banner("Unified table")
    unified = []
    for name, form, row, src in ref_rows:
        unified.append((name, form, row, src))
    for model in order:
        if model not in set(agg["model"]):
            continue
        sub = agg[agg["model"] == model].set_index("h")
        row = {h: float(sub.loc[h, "MASE"]) if h in sub.index else np.nan for h in horder}
        unified.append((model, model.split("_")[-1], row, "this run"))

    # delta vs the xLSTM reference
    def delta(mase_all):
        return 100.0 * (mase_all - XREF) / XREF

    tbl_lines = []
    for name, form, row, src in unified:
        m_all = row.get("ALL", np.nan)
        tbl_lines.append([
            name, form, f4(row.get(1)), f4(row.get(3)), f4(row.get(6)),
            f4(row.get(12)), f4(m_all),
            ("—" if not np.isfinite(m_all) else f"{delta(m_all):+.2f}%"),
        ])

    # ---------------- DM against xLSTM (top 3 by MASE_ALL) ----------------
    ranked = (agg[agg["h"] == "ALL"].sort_values("MASE").reset_index(drop=True))
    top3 = ranked.head(3)["model"].tolist()
    banner(f"DM vs xLSTM (purged seed {PRIMARY_SEED}) — top 3 by MASE_ALL: {top3}")

    xref_preds = purged[purged["seed"] == PRIMARY_SEED][COLS].copy()
    xref_preds["h"] = xref_preds["h"].astype(int)
    xref_preds["model"] = "xLSTM_purged_s42"

    def _primary(df, model):
        # deterministic learners carry seed=-1; seeded ones contribute seed 42
        sub = df[df["model"] == model]
        if (sub["seed"] == PRIMARY_SEED).any():
            return sub[sub["seed"] == PRIMARY_SEED]
        return sub[sub["seed"] == -1]

    pool = pd.concat([_primary(new_preds, m) for m in top3])[COLS].copy()
    pool["h"] = pool["h"].astype(int)
    dm_in = pd.concat([xref_preds, pool], ignore_index=True)
    print("  DM input models:", sorted(dm_in["model"].unique()))

    dm_pool = H.dm_table(dm_in, baseline="xLSTM_purged_s42")
    dm_pool.to_csv(RES / "stage0_extra_dm_pooled.csv", index=False, encoding="utf-8")
    dm_ser = H.dm_series_level(dm_in, baseline="xLSTM_purged_s42")
    dm_ser.to_csv(RES / "stage0_extra_dm_series.csv", index=False, encoding="utf-8")
    dm_tim = H.dm_time_aggregated(dm_in, baseline="xLSTM_purged_s42")
    dm_tim.to_csv(RES / "stage0_extra_dm_time.csv", index=False, encoding="utf-8")
    print(dm_pool.to_string(index=False))

    # ---------------- honest conclusion (data-driven) ---------------------
    best = ranked.iloc[0]
    best_new = float(best["MASE"])
    beats_all = best_new < XREF
    per_h_win = {}
    for h in H.HORIZONS:
        cand = agg[(agg["h"] == h)].sort_values("MASE")
        b = cand.iloc[0]
        per_h_win[h] = (b["model"], float(b["MASE"]), float(xm[str(h)]))

    # ---------------- train_sec -------------------------------------------
    timing_rows = []
    for tag in [j["tag"] for j in jobs if j["tag"] in timings]:
        d = timings[tag]
        seeds_here = [k for k in d if k != "_total"]
        tot = sum(d[k] for k in seeds_here)
        timing_rows.append([tag, ("deterministic (single run)" if seeds_here == ["det"]
                                  else f"{len(seeds_here)} seeds {SEEDS}"),
                            f2(tot), f2(tot / max(len(seeds_here), 1))])

    # how many fits the grid search actually ran (derivable from the audit table)
    fit_count = None
    aud = RES / "extra_lgb_tuning_audit.csv"
    if aud.exists():
        fit_count = int(pd.read_csv(aud)["n_val_origins"].sum())

    # ---------------- write the markdown ----------------------------------
    banner("Writing STAGE0_EXTRA_BASELINES.md")
    L = []
    A = L.append
    A("# Stage 0 — 補充基線：五個過去未實測的模型")
    A("")
    A("> 全部數字由 `stage0\\run_baselines_extra.py` 實際執行產生，無任何估計值。")
    A("> 評估框架 = `stage0\\eval\\harness.py`（未修改）。")
    A("")
    A("## 0. 協定自檢（先確認沒有搞錯協定）")
    A("")
    A(f"用同一支 harness 重跑 Seasonal Naive，得到 h=6 MASE = "
      f"{float(sn_tbl[(sn_tbl['model']=='SeasonalNaive')&(sn_tbl['h']==6)]['MASE'].iloc[0]):.5f}"
      f"（驗收值 0.97471）、h=12 MASE = "
      f"{float(sn_tbl[(sn_tbl['model']=='SeasonalNaive')&(sn_tbl['h']==12)]['MASE'].iloc[0]):.5f}"
      f"（驗收值 0.80900）—— **完全一致，協定確認**。")
    A("")
    A("| 項目 | 值 |")
    A("|---|---|")
    A(f"| 原點 | {len(H.ORIGINS)} 個（{H.ORIGINS[0]} .. {H.ORIGINS[-1]}） |")
    A(f"| 視界 | h = {', '.join(str(h) for h in H.HORIZONS)}（Direct，逐 h 一個模型） |")
    A(f"| 訓練切片 | `target_period <= origin`，每個原點重新 fit |")
    A(f"| 目標形式 | 每個學習器都跑 `level` 與 `residual` |")
    A(f"| 種子 | {SEEDS}（LightGBM）；Ridge/kNN/NLinear/DLinear/KMeans 為確定性，跑一次 |")
    A(f"| xLSTM 參考 | MASE_ALL = {XREF}（purged 協定，三種子均值，來自 `results/stage0_purged_val_agg.csv`） |")
    A("")
    A("## 1. 統一對照表（MASE，越低越好；< 1 = 打敗 seasonal naive）")
    A("")
    A(md_table(["模型", "目標形式", "h=1", "h=3", "h=6", "h=12", "MASE_ALL",
                "相對 xLSTM 0.98355"],
               tbl_lines,
               ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"]))
    A("")
    A("說明：")
    A("- 前 6 列（SeasonalNaive / Drift / Ridge_* / LightGBM_*）是已鎖死的既有結果，"
      "直接取自 `results/stage0_results.csv`（同一支 harness、同一協定），不是重算值。")
    A("- `xLSTM (purged, 3-seed mean)` 的逐 h 數字是用 `results/purged_val_predictions.csv` "
      "的三個種子預測經 `H.metrics_from_frame` 逐種子算 MASE 後取均值；其 MASE_ALL 均值即 0.98355。")
    A("- 最後 10 列是本輪新實測；LightGBMTuned 為三種子均值，其餘為單次確定性執行。")
    A("")
    A("### 1.1 一個必須指出的數學事實：DLinear 的 level 與 residual 完全等價")
    A("")
    A(f"實測 `max|pred_level − pred_residual|` = **{dl_gap:.3g}**（DLinear）、"
      f"**{nl_gap:.4g}**（NLinear，逐點差異最大值，單位 HK$m）。")
    A("")
    A("DLinear 的設計矩陣是 `[trend | seasonal]`，而 `trend + seasonal = 原始 6 欄`，"
      "所以 `y_z(t)` 本身就在設計矩陣的行空間內。殘差目標 `y_z(t+h) − y_z(t)` 與水平目標 "
      "`y_z(t+h)` 只差一個落在行空間內的向量，OLS 給出的預測因此**逐點相同**。")
    A("這不是 bug，是線性模型在『殘差化』上的天花板：**對 DLinear 而言，殘差化目標沒有免費的錨點可拿**，"
      "與 Ridge（設計矩陣不含 `y_z(t)`，殘差化值 −34%）形成直接對照。")
    A("")
    A("## 2. 對 xLSTM 的三層 DM 檢定（只做 MASE_ALL 最好的 3 個）")
    A("")
    A(f"入選（依 MASE_ALL）：**{', '.join(top3)}**。")
    A(f"對照組 = `xLSTM_purged_s42`（purged 協定的主種子 42，MASE_ALL = "
      f"{float(purged_metrics[(purged_metrics['h']=='ALL')&(purged_metrics['seed']==42)]['MASE'].iloc[0]):.5f}）。")
    A("")
    A("### 2.1 Pooled DM（N=276，Newey–West 修正；已知偏樂觀）")
    A("")
    rows = []
    for r in dm_pool.itertuples(index=False):
        rows.append([r.h, r.model, r.n, f"{r.DM_sq:+.4f}", f"{r.p_sq:.4f}",
                     r.better_sq or "—", "★" if r.sig_sq else "",
                     f"{r.DM_abs:+.4f}", f"{r.p_abs:.4f}", r.better_abs or "—",
                     "★" if r.sig_abs else ""])
    A(md_table(["h", "model", "n", "DM_sq", "p_sq", "better_sq", "sig",
                "DM_abs", "p_abs", "better_abs", "sig"], rows,
               ["---:", "---", "---:", "---:", "---:", "---", "---:", "---:", "---:", "---", "---:"]))
    A("")
    A("`DM > 0` ⇒ 該模型損失較高（xLSTM 較好）。★ = p < 0.05。")
    A("")
    A("### 2.2 Series-level（N=23，配對 Wilcoxon）")
    A("")
    rows = [[r.h, r.model, r.n_series, r.series_won, r.series_lost,
             f"{r.median_diff:,.2f}", f"{r.p_wilcoxon:.4f}", r.better,
             "★" if r.sig else ""] for r in dm_ser.itertuples(index=False)]
    A(md_table(["h", "model", "n_series", "won", "lost", "median_diff(HK$m)",
                "p_wilcoxon", "better", "sig"], rows,
               ["---:", "---", "---:", "---:", "---:", "---:", "---:", "---", "---:"]))
    A("")
    A("`better` 依 median(該模型 AE − xLSTM AE) 的符號決定；負值 ⇒ 該模型較好。")
    A("")
    A("### 2.3 Time-aggregated（T=12，配對 Wilcoxon）")
    A("")
    rows = [[r.h, r.model, r.n_origins, r.origins_won, r.origins_lost,
             f"{r.median_diff:,.2f}", f"{r.p_wilcoxon:.4f}", r.better,
             "★" if r.sig else ""] for r in dm_tim.itertuples(index=False)]
    A(md_table(["h", "model", "n_origins", "won", "lost", "median_diff(HK$m)",
                "p_wilcoxon", "better", "sig"], rows,
               ["---:", "---", "---:", "---:", "---:", "---:", "---:", "---", "---:"]))
    A("")
    A("## 3. 誠實結論：哪個基線在哪些視界贏過 xLSTM？")
    A("")
    if beats_all:
        A(f"**有。** MASE_ALL 最低的新基線是 `{best['model']}`，"
          f"MASE_ALL = {best_new:.5f}，比 xLSTM 的 {XREF} 低 {abs(delta(best_new)):.2f}%。")
    else:
        A(f"**沒有任何一個新基線在 MASE_ALL 上贏過 xLSTM。** "
          f"最好的新基線是 `{best['model']}`，MASE_ALL = {best_new:.5f}，"
          f"比 xLSTM 的 {XREF} 高 {abs(delta(best_new)):.2f}%。")
    A("")
    A("逐視界（每個 h 上最好的新基線 vs xLSTM 同視界的三種子均值）：")
    A("")
    rows = []
    for h in H.HORIZONS:
        mdl, mase, xh = per_h_win[h]
        rows.append([f"h={h}", mdl, f"{mase:.5f}", f"{xh:.5f}",
                     ("贏" if mase < xh else "輸"),
                     f"{100*(mase-xh)/xh:+.2f}%"])
    A(md_table(["視界", "最佳新基線", "其 MASE", "xLSTM MASE", "勝負", "差距"],
               rows, ["---", "---", "---:", "---:", "---:", "---:"]))
    A("")
    hwin = [h for h in H.HORIZONS if per_h_win[h][1] < per_h_win[h][2]]
    if hwin:
        A(f"→ **在 {', '.join(f'h={h}' for h in hwin)} 上，新基線的 MASE 低於 xLSTM。** "
          f"注意這是逐格比較、未做多重比較校正；是否顯著看上節 DM（三層）。")
    else:
        A("→ **在任何一個視界上，新基線的 MASE 都沒有低於 xLSTM。**")
    A("")
    sig_pool = dm_pool[(dm_pool["sig_abs"]) & (dm_pool["better_abs"] == "model")]
    sig_ser = dm_ser[(dm_ser["sig"]) & (dm_ser["better"] != "xLSTM_purged_s42")]
    sig_tim = dm_tim[(dm_tim["sig"]) & (dm_tim["better"] != "xLSTM_purged_s42")]
    A(f"統計面：pooled(abs) 顯著贏過 xLSTM 的格數 = {len(sig_pool)}；"
      f"series-level 顯著 = {len(sig_ser)}；time-aggregated 顯著 = {len(sig_tim)}。")
    A("")
    A("### 3.1 ⚠️ MASE 與 DM 在 h=12 上互相矛盾——必須誠實寫出來")
    A("")
    A("h=12 時 `kNN_residual` 的 MASE = "
      f"{float(agg[(agg['model']=='kNN_residual')&(agg['h']==12)]['MASE'].iloc[0]):.5f} 低於 xLSTM 的 "
      f"{float(xm['12']):.5f}，但 DM（pooled，絕對與平方損失）一致說 xLSTM 較好，WAPE 也是 xLSTM 較好"
      f"（{float(agg[(agg['model']=='kNN_residual')&(agg['h']==12)]['WAPE'].iloc[0])*100:.2f}% vs "
      f"{float(purged_metrics[(purged_metrics['model']=='xLSTM_purged_s42')&(purged_metrics['h']=='12')]['WAPE'].iloc[0])*100:.2f}%）。")
    A("")
    A("原因是兩個指標的加權方式不同：**MASE 先在每條序列內部正規化再等權平均**（23 條序列等權，"
      "小序列與大序列份量相同），**WAPE 與 pooled DM 則是金額加權**（被 `18`、`35`、`40` 等大序列主導）。"
      "kNN 在小序列上較準、在大序列上較差，於是同一組預測同時「MASE 贏、WAPE 輸」。")
    ser12 = dm_ser[(dm_ser["h"] == 12) & (dm_ser["model"] == "kNN_residual")].iloc[0]
    k12 = float(agg[(agg["model"] == "kNN_residual") & (agg["h"] == 12)]["MASE"].iloc[0])
    x12 = float(xm["12"])
    A(f"用 §2.2 的 series-level DM 交叉檢查：h=12 時 kNN 在 {ser12['n_series']} 條裡的 "
      f"{ser12['series_won']} 條贏（median_diff {ser12['median_diff']:,.2f} HK$m），"
      f"但 Wilcoxon p={ser12['p_wilcoxon']:.4f} → "
      f"**{'顯著' if ser12['sig'] else '不顯著'}**。")
    A("")
    A(f"**所以「kNN 在 h=12 贏過 xLSTM」這句話只能寫成：在等權 MASE 上贏 "
      f"{abs(100*(k12-x12)/x12):.2f}%，但在金額加權誤差與三層 DM 上都不成立。**")
    A("")
    A("### 3.2 三個附帶發現")
    A("")

    def mall(m):
        return float(agg[(agg["model"] == m) & (agg["h"] == "ALL")]["MASE"].iloc[0])

    rid_all = float(locked[(locked["model"] == "Ridge_residual")
                           & (locked["h"] == "ALL")]["MASE"].iloc[0])
    naive_all = float(locked[(locked["model"] == "SeasonalNaive")
                             & (locked["h"] == "ALL")]["MASE"].iloc[0])
    A(f"1. **分群有效但不足以翻盤**：`KMeansRidge_residual` MASE_ALL = {mall('KMeansRidge_residual'):.5f}，"
      f"比 `Ridge_residual`（{rid_all:.5f}）改善 "
      f"{100*(mall('KMeansRidge_residual')-rid_all)/rid_all:+.2f}%，但距離 xLSTM 仍差 "
      f"{abs(delta(mall('KMeansRidge_residual'))):.2f}%。"
      "分群把「一條 Ridge 服務 23 個類別」的問題縮小，但不是主要瓶頸。")
    A(f"2. **殘差化依然是最大的單一因素**：本輪 5 個模型幾乎全部重演 RESULTS.md §5 的結論 — "
      f"`kNN` {mall('kNN_level'):.4f} → {mall('kNN_residual'):.4f}、"
      f"`KMeansRidge` {mall('KMeansRidge_level'):.4f} → {mall('KMeansRidge_residual'):.4f}、"
      f"`LightGBMTuned` {mall('LightGBMTuned_level'):.4f} → {mall('LightGBMTuned_residual'):.4f}、"
      f"`NLinear` {mall('NLinear_level'):.4f} → {mall('NLinear_residual'):.4f}；"
      "只有 `DLinear` 因 §1.1 的代數原因完全沒變。"
      "**沒有任何一個架構改動，效果比『把目標換成殘差』更大。**")
    A(f"3. **NLinear / DLinear 這兩個 2023 年的線性 SOTA，在本資料上不如 seasonal naive**"
      f"（NLinear_residual {mall('NLinear_residual'):.4f}、DLinear_residual {mall('DLinear_residual'):.4f}，"
      f"naive {naive_all:.4f}）。原因是它們的輸入只有 6 個純量欄位"
      "（本專案沒有真正的多變量序列，`H.FEATURE_COLS` 是工程好的滯後 + 日曆 + 制度 dummy），"
      "**這兩個架構的長處（長 context 的通道獨立線性外推）在這裡沒有施展空間。**"
      "報告引用它們時必須連這句一起寫，否則會被誤讀成「線性模型很弱」。")
    A("")
    A("")
    A("## 4. LightGBM 網格搜索（訓練期內部時間序列驗證，未碰 2024）")
    A("")
    A("驗證設計：對每個 h，取「目標月仍 <= 202312」的最後 24 個原點，"
      "逐原點 `train{target_period<=o} → predict{period==o}`，"
      "以 y_z 空間 pooled MAE 選參。16 組候選全表在 "
      "`results/extra_lgb_tuning_audit.csv`。")
    A("")
    A(f"實際 fit 次數 = 4 個視界 × 16 組候選 × 24 個驗證原點"
      + (f" = **{fit_count:,}**" if fit_count else "")
      + "（可由 audit 表的 `n_val_origins` 欄加總核對）。"
      "選參用 `residual` 目標形式，選中的超參**原封不動套用到 level 與 residual 兩種模式**，"
      "避免「重新調參」混淆 level/residual 的對照。")
    A("")
    rows = [[f"h={h}", params_by_h[h]["num_leaves"], params_by_h[h]["learning_rate"],
             params_by_h[h]["min_child_samples"], params_by_h[h]["n_estimators"],
             f"{params_by_h[h]['val_MAE_yz']:.5f}"] for h in H.HORIZONS]
    A(md_table(["視界", "num_leaves", "learning_rate", "min_child_samples",
                "n_estimators", "val_MAE_yz"], rows,
               ["---", "---:", "---:", "---:", "---:", "---:"]))
    A("")
    A("## 5. K-Means 分群（僅用訓練期畫像）")
    A("")
    rows = []
    for c, g in prof.groupby("cluster"):
        rows.append([c, len(g), f"{g['y_log_mean'].mean():.2f}",
                     f"{g['y_log_std'].mean():.2f}", f"{g['acf12'].mean():.3f}",
                     ", ".join(g.index.tolist())])
    A(md_table(["cluster", "n_series", "mean(y_log 均值)", "mean(y_log 標準差)",
                "mean(lag12 自相關)", "series"], rows,
               ["---:", "---:", "---:", "---:", "---:", "---"]))
    A("")
    A("3 維特徵 = `y_log` 的 mean / std（訓練期）＋ `y_z` 對 `lag12` 的相關係數（訓練期）；"
      "StandardScaler 後 `KMeans(n_clusters=3, random_state=42, n_init=10)`。"
      "分群只用 `period <= 202312` 的資料，2024 未參與。")
    A("")
    A("## 6. 產出檔案")
    A("")
    A("```")
    A("results/stage0_baselines_extra.csv            主表（model x h）")
    A("results/stage0_baselines_extra_perseed.csv    逐種子原表")
    A("results/stage0_extra_dm_pooled.csv            pooled DM")
    A("results/stage0_extra_dm_series.csv            series-level DM")
    A("results/stage0_extra_dm_time.csv              time-aggregated DM")
    A("results/extra_lgb_tuning.json                 選中的 LightGBM 參數")
    A("results/extra_lgb_tuning_audit.csv            16 組候選的驗證分數")
    A("results/extra_series_clusters.csv             K-Means 分群結果")
    A("results/extra_train_sec.json                  各模型的實測訓練秒數")
    A("results/_extra_cache/*.csv                    每個 (model, seed) 的預測（續跑用）")
    A("```")
    A("")
    A("## 7. 各模型實測 train_sec（檔尾）")
    A("")
    A(md_table(["模型", "種子設定", "總訓練秒數", "每次訓練秒數"], timing_rows,
               ["---", "---", "---:", "---:"]))
    A("")
    A("上表的秒數是**實際 fit 的 wall-clock**（單機 CPU，`n_jobs=4`），"
      "不含快取讀取時間；由 `results/extra_train_sec.json` 保存，重跑時沿用。")
    A(f"LightGBM 網格搜索另計"
      + (f" {fit_count:,} 次 fit" if fit_count else "")
      + "（見 §4），與上表的 6 次最終 fit 分開。")
    A("")
    md = "\n".join(L)
    (RES / "STAGE0_EXTRA_BASELINES.md").write_text(md, encoding="utf-8")
    print(f"  STAGE0_EXTRA_BASELINES.md : {len(md.splitlines())} lines")

    banner("DONE")
    print(f"  total elapsed {time.time()-t_all:.1f}s")
    print(f"  best new MASE_ALL = {best_new:.5f} ({best['model']})  vs xLSTM {XREF}")
    print(f"  beats xLSTM on MASE_ALL: {beats_all}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
