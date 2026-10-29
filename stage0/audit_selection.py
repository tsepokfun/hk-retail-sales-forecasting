"""
AUDIT steps 3-5 — per-horizon model selection.

  step 3  ORACLE upper bound: per horizon, pick the model with the lowest
          MASE computed ON THE 2024 TEST SET.  This uses 2024 labels, so it is
          NOT deployable; it only quantifies how much a perfect per-horizon
          switch could be worth.
  step 4  DEPLOYABLE per-horizon selection: the only admissible evidence is
          performance at forecast origins <= 2022-12.
            V1  frozen protocol: every candidate trained on target_period
                <= 2017-12, evaluated on origins 2019-01 .. 2021-11.
                This is the ONLY block that can contain the xLSTM, because the
                repo stores xLSTM out-of-fold predictions exactly for that block
                (results/stage2_oof_s*.csv) and no xLSTM predictions exist for
                any other pre-2023 origin.  All CPU candidates are put through
                the identical frozen protocol so the comparison is fair.
            V2  rolling protocol: every candidate re-fit at each origin
                2022-01 .. 2022-12 (targets <= 2023-12).  Post-COVID and close
                to the test regime, but the xLSTM is ABSENT here.
  step 5  per-series weakness of xLSTM_ensemble against SeasonalNaive.

No 2024 label is read by anything in step 4 or step 5.
Writes: audit_model_by_h.csv, audit_oracle.csv, audit_val_v1.csv,
        audit_val_v2.csv, audit_series_weakness.csv, audit_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H                       # noqa: E402
from models import baselines as B                   # noqa: E402
from run_stage4_reconcile import (                  # noqa: E402
    build_A, naive_w, reconcile,
)

RES = ROOT / "results"
SEEDS = (42, 7, 2024)
COLS = ["model", "origin", "h", "series", "y_true", "y_pred"]
V1_LO, V1_HI = "201901", "202111"
V2_ORIGINS = [f"2022{m:02d}" for m in range(1, 13)]
FROZEN_CUTOFF = "201712"

PARTICIPANTS = [
    "SeasonalNaive", "Drift", "Ridge_level", "Ridge_residual",
    "LightGBM_level", "LightGBM_residual", "kNN_residual",
    "KMeansRidge_residual", "NLinear_residual", "DLinear_residual",
    "LightGBMTuned_residual", "LSTM_none", "xLSTM_ensemble",
    "xLSTM_ensemble_MinT_wls",
]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def read(fname, dtype=("series", "origin")) -> pd.DataFrame:
    d = pd.read_csv(fname, dtype={c: str for c in dtype})
    d["h"] = d["h"].astype(int)
    return d


def mean_preds(frames, model="ensemble") -> pd.DataFrame:
    """average y_pred over a list of frames that share (origin, h, series)"""
    base = frames[0].copy()
    if "model" not in base.columns:
        base["model"] = model
    base = base[COLS].copy()
    stack = []
    for f in frames:
        g = f.set_index(["origin", "h", "series"]).sort_index()
        stack.append(g["y_pred"])
    P = pd.concat(stack, axis=1).mean(axis=1)
    idx = base.set_index(["origin", "h", "series"]).index
    base["y_pred"] = P.reindex(idx).to_numpy()
    return base


def load_test_grid() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    p = read(RES / "predictions.csv")
    for m, g in p.groupby("model"):
        out[m] = g[COLS].reset_index(drop=True)

    for tag in ("kNN_residual", "KMeansRidge_residual", "NLinear_residual",
                "DLinear_residual"):
        out[tag] = read(RES / f"extra_preds_{tag}.csv")

    # LightGBMTuned: the file on disk is the 3-seed MEAN of per-seed MASE
    # (1.04108).  The prediction analogue of that is the 3-seed average, so both
    # the single s42 run and the 3-seed average are kept.
    lg_s42 = read(RES / "extra_preds_LightGBMTuned_residual.csv")
    lg_s7 = read(RES / "extra_preds_LightGBMTuned_residual_s7.csv")
    lg_2024 = read(RES / "extra_preds_LightGBMTuned_residual_s2024.csv")
    out["LightGBMTuned_residual_s42"] = lg_s42
    out["LightGBMTuned_residual"] = mean_preds([lg_s42, lg_s7, lg_2024])

    lstm = [read(ROOT.parent / "stage3" / "results" / "preds" / f"LSTM_none_s{s}.csv")
            for s in SEEDS]
    out["LSTM_none"] = mean_preds(lstm)
    out["LSTM_none"]["model"] = "LSTM_none"

    xl = [read(RES / "exog_preds_z" / f"xLSTM_none_s{s}.csv") for s in SEEDS]
    out["xLSTM_ensemble"] = mean_preds(xl)
    out["xLSTM_ensemble"]["model"] = "xLSTM_ensemble"
    # the 3-seed average of the tuned GBDT, same treatment as the xLSTM
    out["LightGBMTuned_residual"]["model"] = "LightGBMTuned_residual"
    return out


def check_grid(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """every source must sit on the same (origin, h, series) grid and y_true"""
    ref = frames["SeasonalNaive"].sort_values(["origin", "h", "series"]) \
                              .reset_index(drop=True)
    key = ref[["origin", "h", "series"]].astype(str).agg("|".join, axis=1)
    print("  grid check: reference cells =", len(ref),
          " unique =", key.nunique())
    for name, f in frames.items():
        g = f.sort_values(["origin", "h", "series"]).reset_index(drop=True)
        k = g[["origin", "h", "series"]].astype(str).agg("|".join, axis=1)
        same_keys = k.equals(key)
        same_truth = np.allclose(g["y_true"].to_numpy(), ref["y_true"].to_numpy())
        finite = np.isfinite(g["y_pred"].to_numpy()).all()
        flag = "OK " if (same_keys and same_truth and finite) else "**"
        print(f"   {flag} {name:32s} n={len(g):5d} keys_match={same_keys} "
              f"y_true_match={same_truth} finite={finite}")
    return ref


# ---------------------------------------------------------------------------
# step 3 — oracle
# ---------------------------------------------------------------------------
def step3_oracle(frames, naive_mae):
    met = []
    for name, f in frames.items():
        m = H.metrics_from_frame(f, naive_mae)
        m["model"] = name
        met.append(m)
    met = pd.concat(met, ignore_index=True)
    met["h"] = met["h"].astype(str)
    met.to_csv(RES / "audit_model_by_h.csv", index=False)

    lookup = met.set_index(["model", "h"])
    picks, rows = {}, []
    for h in ("1", "3", "6", "12"):
        cand = lookup.xs(h, level="h").reset_index()
        cand = cand[cand["model"].isin(PARTICIPANTS)]
        best = cand.sort_values("MASE").iloc[0]
        picks[h] = best["model"]
        rows.append({"h": h, "best_model": best["model"], "MASE": best["MASE"],
                     "WAPE": best["WAPE"], "second": cand.sort_values("MASE")
                     .iloc[1]["model"],
                     "second_MASE": cand.sort_values("MASE").iloc[1]["MASE"]})
    oracle = pd.DataFrame(rows)
    oracle.to_csv(RES / "audit_oracle_picks.csv", index=False)

    hybrid = pd.concat([frames[m][frames[m]["h"] == int(h)]
                        for h, m in picks.items()], ignore_index=True)
    hm = H.metrics_from_frame(hybrid.assign(model="ORACLE_per_h"), naive_mae)
    hm["h"] = hm["h"].astype(str)
    hm.to_csv(RES / "audit_oracle.csv", index=False)
    return met, oracle, hm


# ---------------------------------------------------------------------------
# step 4 — validation blocks
# ---------------------------------------------------------------------------
def mase_block(df, denom, model=None) -> pd.DataFrame:
    d = df if model is None else df.assign(model=model)
    return H.metrics_from_frame(d, denom)


def rolling_cpu(feat, wide, scaling, origins, tag, fn, **kw) -> pd.DataFrame:
    """the repo's own rolling runners, restricted to `origins`"""
    df = fn(feat, wide, scaling, origins=origins, tag=tag, target_mode="residual", **kw)
    df = df[COLS].copy()
    df["h"] = df["h"].astype(int)
    df["origin"] = df["origin"].astype(str)
    df["series"] = df["series"].astype(str)
    return df


def frozen_cpu(feat, wide, scaling, cutoff, origins, kind, tag,
               params_by_h=None, seed=42) -> pd.DataFrame:
    """one training window (target_period <= cutoff), predict `origins`.

    Mirrors models/baselines.py exactly, but freezes the training window so the
    row budget matches what the xLSTM out-of-fold block had.
    """
    import lightgbm as lgb

    rows = []
    for h in H.HORIZONS:
        sup = H.make_supervised(feat, h)
        tr = sup[sup["target_period"] <= cutoff]
        Xtr = H.matrix(tr)
        ytr = B._targets(tr, "residual")
        if kind == "knn":
            sc = StandardScaler().fit(Xtr)
            mdl = KNeighborsRegressor(n_neighbors=15, weights="distance") \
                .fit(sc.transform(Xtr), ytr)
            prep = sc.transform
        elif kind == "ridge":
            sc = StandardScaler().fit(Xtr)
            mdl = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
            prep = sc.transform
        elif kind == "lgb":
            p = (dict(B.LGB_FIXED, seed=seed,
                      **{k: params_by_h[h][k] for k in B.LGB_GRID})
                 if params_by_h else dict(B.LGB_FIXED, seed=seed))
            mdl = lgb.LGBMRegressor(**p).fit(Xtr, ytr)
            prep = lambda x: x                              # noqa: E731
        else:
            raise ValueError(kind)
        print(f"    frozen {tag:22s} h={h:>2}  n_train_rows={len(tr)}")
        for o in origins:
            te = sup[sup["period"] == o]
            if len(te) == 0:
                continue
            pz = B._reconstruct(mdl.predict(prep(H.matrix(te))), te, "residual")
            pred = H.inverse_transform(pz, te["series"].to_numpy(), scaling)
            for s, yp in zip(te["series"], pred):
                rows.append((tag, o, h, s,
                             float(wide.loc[H.month_add(o, h), s]), float(yp)))
    df = pd.DataFrame(rows, columns=COLS)
    df["origin"] = df["origin"].astype(str)
    df["series"] = df["series"].astype(str)
    return df


def denom_for(wide, train_end, series) -> pd.Series:
    """per-series seasonal-naive MAE on months <= train_end (scale for MASE)"""
    tr = wide.loc[[p for p in wide.index if p <= train_end], series]
    d = (tr - tr.shift(12)).abs().iloc[12:]
    return d.mean(axis=0)


def pair_test(a: pd.DataFrame, b: pd.DataFrame, label_a, label_b) -> dict:
    """series-level paired Wilcoxon on per-series mean absolute error (N=23)"""
    from scipy import stats as st

    ka = a.assign(ae=(a["y_true"] - a["y_pred"]).abs()).groupby("series")["ae"].mean()
    kb = b.assign(ae=(b["y_true"] - b["y_pred"]).abs()).groupby("series")["ae"].mean()
    idx = ka.index.intersection(kb.index)
    x, y = ka.loc[idx].to_numpy(float), kb.loc[idx].to_numpy(float)
    try:
        _, p = st.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        p = float(p)
    except Exception:                                       # noqa: BLE001
        p = float("nan")
    return {"A": label_a, "B": label_b, "n_series": len(idx),
            "series_A_better": int((x < y).sum()),
            "series_B_better": int((x > y).sum()),
            "median_rel_diff_pct": round(float(np.median((x - y) / np.where(y, y, np.nan)) * 100), 2),
            "p_wilcoxon": round(p, 6), "sig": bool(p == p and p < 0.05)}


def main() -> int:
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    naive_mae = H.seasonal_naive_mae_train(wide)
    feat = H.load_features()
    scaling = pd.read_csv(ROOT / "data" / "scaling.csv", dtype={"series": str})
    params_by_h = {int(k): v for k, v in
                   json.loads((RES / "extra_lgb_tuning.json").read_text()).items()}

    print("=" * 92)
    print("STEP 3 — all participants on the 2024 test grid")
    print("=" * 92)
    frames = load_test_grid()
    ref = check_grid({k: v for k, v in frames.items() if k in PARTICIPANTS})
    series = sorted(ref["series"].unique(), key=lambda s: (len(s), s))

    # xLSTM_ensemble + MinT_wls
    A = build_A(series)
    W = np.diag(naive_w(wide, series, H.TRAIN_END))
    xe = frames["xLSTM_ensemble"].sort_values(["origin", "h", "series"])
    keys = sorted({(o, h) for o, h in zip(xe["origin"], xe["h"])},
                  key=lambda t: (str(t[0]), int(t[1])))
    P = np.full((len(keys), len(series)), np.nan)
    for n, (o, h) in enumerate(keys):
        sub = xe[(xe["origin"] == o) & (xe["h"] == h)]
        m = dict(zip(sub["series"], sub["y_pred"]))
        for j, s in enumerate(series):
            P[n, j] = m[s]
    Pr = reconcile(P, A, W)
    ytrue = np.array([float(wide.loc[H.month_add(o, int(h)), s])
                      for o, h in keys for s in series])
    frames["xLSTM_ensemble_MinT_wls"] = pd.DataFrame({
        "model": "xLSTM_ensemble_MinT_wls",
        "origin": np.repeat([k[0] for k in keys], len(series)),
        "h": np.repeat([int(k[1]) for k in keys], len(series)),
        "series": np.tile(series, len(keys)),
        "y_true": ytrue,
        "y_pred": Pr.ravel(),
    })
    chk = frames["xLSTM_ensemble"].sort_values(["origin", "h"]).reset_index(drop=True)
    print("  MinT_wls frame built on the same grid; y_true agrees with "
          f"xLSTM_ensemble: {np.allclose(np.sort(ytrue), np.sort(chk['y_true']))}")

    met, oracle_picks, oracle_metrics = step3_oracle(frames, naive_mae)
    piv = met.pivot_table(index="model", columns="h", values="MASE", aggfunc="first")
    piv.columns = [str(c) for c in piv.columns]
    order = [c for c in ("1", "3", "6", "12", "ALL") if c in piv.columns]
    print("\n  MASE per model / horizon (2024 TEST):")
    print(piv[order].round(5).to_string())
    print("\n  ORACLE picks (uses 2024 labels — NOT deployable):")
    print(oracle_picks.to_string(index=False))
    print("\n  ORACLE hybrid MASE:")
    print(oracle_metrics[["h", "MASE", "WAPE", "bias_pct", "n_points"]]
          .to_string(index=False))

    # ---------------------------------------------------------------- step 4
    print("\n" + "=" * 92)
    print("STEP 4a — validation block V1 (frozen at target_period <= %s, "
          "origins %s..%s)" % (FROZEN_CUTOFF, V1_LO, V1_HI))
    print("=" * 92)
    v1_origins = [p for p in wide.index if V1_LO <= p <= V1_HI]
    print(f"  n_origins = {len(v1_origins)}  ({v1_origins[0]}..{v1_origins[-1]})")
    denom_v1 = denom_for(wide, FROZEN_CUTOFF, series)

    oof = [read(RES / f"stage2_oof_s{s}.csv") for s in SEEDS]
    for f in oof:
        assert f["origin"].between(V1_LO, V1_HI).all()
    v1 = {}
    v1["xLSTM_ensemble"] = mean_preds(oof)
    v1["SeasonalNaive"] = B.run_seasonal_naive(wide, origins=v1_origins)[COLS]
    v1["kNN_residual"] = frozen_cpu(feat, wide, scaling, FROZEN_CUTOFF,
                                    v1_origins, "knn", "kNN_residual")
    v1["Ridge_residual"] = frozen_cpu(feat, wide, scaling, FROZEN_CUTOFF,
                                      v1_origins, "ridge", "Ridge_residual")
    v1["LightGBM_residual"] = frozen_cpu(feat, wide, scaling, FROZEN_CUTOFF,
                                         v1_origins, "lgb", "LightGBM_residual")
    v1["LightGBMTuned_residual"] = frozen_cpu(
        feat, wide, scaling, FROZEN_CUTOFF, v1_origins, "lgb",
        "LightGBMTuned_residual", params_by_h=params_by_h)

    v1met = []
    for name, f in v1.items():
        f = f.copy()
        f["origin"] = f["origin"].astype(str)
        f["series"] = f["series"].astype(str)
        f["h"] = f["h"].astype(int)
        m = H.metrics_from_frame(f, denom_v1)
        m["model"] = name
        v1met.append(m)
        f.to_csv(RES / f"audit_v1_preds_{name}.csv", index=False)
    v1met = pd.concat(v1met, ignore_index=True)
    v1met["h"] = v1met["h"].astype(str)
    v1met.to_csv(RES / "audit_val_v1.csv", index=False)
    p1 = v1met.pivot_table(index="model", columns="h", values="MASE", aggfunc="first")
    p1 = p1[[c for c in ("1", "3", "6", "12", "ALL") if c in p1.columns]]
    print("\n  V1 MASE (denominator = seasonal-naive MAE on <=%s):" % FROZEN_CUTOFF)
    print(p1.round(5).to_string())

    print("\n  V1 paired Wilcoxon on per-series MAE (N=23), candidate vs xLSTM_ensemble:")
    tests = []
    for name in v1:
        if name == "xLSTM_ensemble":
            continue
        for h in H.HORIZONS:
            r = pair_test(v1[name][v1[name]["h"] == h],
                          v1["xLSTM_ensemble"][v1["xLSTM_ensemble"]["h"] == h],
                          name, "xLSTM_ensemble")
            r["h"] = h
            r["block"] = "V1"
            tests.append(r)
    t1 = pd.DataFrame(tests)
    print(t1[["h", "A", "n_series", "series_A_better", "series_B_better",
              "median_rel_diff_pct", "p_wilcoxon", "sig"]].to_string(index=False))

    print("\n" + "=" * 92)
    print("STEP 4b — validation block V2 (rolling, origins %s..%s, "
          "targets <= 2023-12)" % (V2_ORIGINS[0], V2_ORIGINS[-1]))
    print("=" * 92)
    denom_v2 = denom_for(wide, H.TRAIN_END, series)
    v2 = {}
    v2["SeasonalNaive"] = B.run_seasonal_naive(wide, origins=V2_ORIGINS)[COLS]
    v2["kNN_residual"] = rolling_cpu(feat, wide, scaling, V2_ORIGINS,
                                     "kNN_residual", B.run_knn)
    v2["KMeansRidge_residual"] = rolling_cpu(feat, wide, scaling, V2_ORIGINS,
                                             "KMeansRidge_residual",
                                             B.run_kmeans_ridge)
    v2["NLinear_residual"] = rolling_cpu(feat, wide, scaling, V2_ORIGINS,
                                         "NLinear_residual", B.run_nlinear)
    v2["DLinear_residual"] = rolling_cpu(feat, wide, scaling, V2_ORIGINS,
                                         "DLinear_residual", B.run_dlinear)
    v2["Ridge_residual"] = rolling_cpu(feat, wide, scaling, V2_ORIGINS,
                                       "Ridge_residual", B.run_ridge)
    v2["LightGBM_residual"] = rolling_cpu(feat, wide, scaling, V2_ORIGINS,
                                          "LightGBM_residual", B.run_lightgbm)
    v2["LightGBMTuned_residual"] = rolling_cpu(
        feat, wide, scaling, V2_ORIGINS, "LightGBMTuned_residual",
        B.run_lightgbm_tuned, params_by_h=params_by_h, seed=42)

    v2met = []
    for name, f in v2.items():
        f["origin"] = f["origin"].astype(str)
        f["series"] = f["series"].astype(str)
        f["h"] = f["h"].astype(int)
        m = H.metrics_from_frame(f, denom_v2)
        m["model"] = name
        v2met.append(m)
        f.to_csv(RES / f"audit_v2_preds_{name}.csv", index=False)
    v2met = pd.concat(v2met, ignore_index=True)
    v2met["h"] = v2met["h"].astype(str)
    v2met.to_csv(RES / "audit_val_v2.csv", index=False)
    p2 = v2met.pivot_table(index="model", columns="h", values="MASE", aggfunc="first")
    p2 = p2[[c for c in ("1", "3", "6", "12", "ALL") if c in p2.columns]]
    print("\n  V2 MASE (rolling refit at every origin):")
    print(p2.round(5).to_string())

    print("\n  V2 paired Wilcoxon on per-series MAE, candidate vs SeasonalNaive:")
    tests2 = []
    for name in v2:
        if name == "SeasonalNaive":
            continue
        for h in H.HORIZONS:
            r = pair_test(v2[name][v2[name]["h"] == h],
                          v2["SeasonalNaive"][v2["SeasonalNaive"]["h"] == h],
                          name, "SeasonalNaive")
            r["h"] = h
            r["block"] = "V2"
            tests2.append(r)
    t2 = pd.DataFrame(tests2)
    print(t2[["h", "A", "n_series", "series_A_better", "series_B_better",
              "median_rel_diff_pct", "p_wilcoxon", "sig"]].to_string(index=False))
    pd.concat([t1, t2], ignore_index=True).to_csv(
        RES / "audit_validation_tests.csv", index=False)

    # ---------------------------------------------------------------- step 5
    print("\n" + "=" * 92)
    print("STEP 5 — per-series weakness: xLSTM_ensemble vs SeasonalNaive (2024)")
    print("=" * 92)
    rows = []
    for s in series:
        a = frames["xLSTM_ensemble"]
        b = frames["SeasonalNaive"]
        a = a[a["series"] == s]
        b = b[b["series"] == s]
        mae_x = float((a["y_true"] - a["y_pred"]).abs().mean())
        mae_n = float((b["y_true"] - b["y_pred"]).abs().mean())
        d = float(denom_v2.loc[s])
        size = float(a["y_true"].abs().mean())
        rows.append({"series": s, "is_parent": s in ("2", "12", "18", "35"),
                     "mean_abs_y_2024": round(size, 1),
                     "naive_mae_scale": round(d, 1),
                     "MASE_xLSTM": round(mae_x / d, 4),
                     "MASE_naive": round(mae_n / d, 4),
                     "rel_vs_naive_pct": round(100 * (mae_x / mae_n - 1), 1),
                     "xLSTM_share_of_total_abs_err_pct": None})
    sw = pd.DataFrame(rows)
    tot = float((frames["xLSTM_ensemble"]["y_true"]
                 - frames["xLSTM_ensemble"]["y_pred"]).abs().sum())
    for i, s in enumerate(sw["series"]):
        a = frames["xLSTM_ensemble"]
        sw.loc[i, "xLSTM_share_of_total_abs_err_pct"] = round(
            100 * float((a[a["series"] == s]["y_true"]
                         - a[a["series"] == s]["y_pred"]).abs().sum()) / tot, 2)
    sw = sw.sort_values("rel_vs_naive_pct", ascending=False)
    sw.to_csv(RES / "audit_series_weakness.csv", index=False)
    print(sw.to_string(index=False))

    worse10 = sw[sw["rel_vs_naive_pct"] > 10]
    from scipy import stats as st
    rho = st.spearmanr(sw["mean_abs_y_2024"], sw["rel_vs_naive_pct"])
    print(f"\n  series where xLSTM is >10% worse than SeasonalNaive: {len(worse10)}/23")
    print(worse10[["series", "is_parent", "mean_abs_y_2024",
                   "rel_vs_naive_pct"]].to_string(index=False))
    print(f"  parents among them: {list(worse10['is_parent'])}"
          f"  ({int(worse10['is_parent'].sum())}/4 parents)")
    print(f"  Spearman(series size, rel_vs_naive_pct) rho={rho.statistic:+.3f} "
          f"p={rho.pvalue:.4f}")

    json.dump({
        "oracle_picks": oracle_picks.to_dict(orient="records"),
        "oracle_all": float(oracle_metrics[oracle_metrics["h"] == "ALL"]["MASE"].iloc[0]),
        "v1_mase": p1.round(5).to_dict(),
        "v2_mase": p2.round(5).to_dict(),
        "worse10": worse10[["series", "is_parent", "mean_abs_y_2024",
                            "rel_vs_naive_pct"]].to_dict(orient="records"),
        "spearman_size_vs_rel": [round(float(rho.statistic), 4),
                                 round(float(rho.pvalue), 4)],
    }, open(RES / "audit_summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
