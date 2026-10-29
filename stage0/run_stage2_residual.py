"""
Stage 2 (lightweight) — two-stage residual correction with Ridge.

WHY THIS EXISTS
---------------
Stage 1 concluded that putting exogenous variables INSIDE the xLSTM hurts,
because every extra input column costs more than it contributes (width-matched
noise controls, `stage1/RESULTS.md` §7).  But that leaves one hypothesis
untested: maybe the exogenous block is fine and the *architecture* is the wrong
place to put it.  Stage 2 tests the cheap version of that hypothesis:

    base model unchanged  ->  model the RESIDUAL with a linear model instead

Crucially the residual model does NOT touch the xLSTM, so it costs **zero**
extra width in the neural network.  If the exogenous variables carry any usable
information at all, this is the setting where it should show up.

NO-LEAKAGE DESIGN
-----------------
The base model must not have seen the residual-training origins.  So:

    cutoff 201912, purge_val=True
      base-model training origins   2006-10 .. 2017-12
      base-model validation origins 2018-01 .. 2018-12   (purged: <= 2019-12 - 12)
      out-of-fold origins           2019-01 .. 2021-11   <- residuals fitted here
      test origins                  2023-12 .. 2024-11   <- correction applied here

2019-01 .. 2021-11 is completely unseen by the base model, so those residual
targets are honest out-of-fold values.  There is a deliberate 13-month gap
(2022-01 .. 2023-11) between the OOF block and the test block; it is not used for
fitting anything, which keeps the residual model free of any test-period label.

WHAT IS MODELLED
----------------
Residuals are put on the MASE scale, r = (y - yhat) / naive_mae_series, so one
Ridge can pool 23 series of wildly different size (22 .. 800 HK$m).  Correction
is `yhat + rhat * naive_mae_series`.

    r_{i,h} = b0 + sum_k b_k * exog_k(origin_i) + series fixed effect + e

One Ridge per horizon, alpha chosen by leave-one-origin-out CV among the 35 OOF
origins.  Alpha selection uses only OOF residuals -- no test information.

Also reports the ZERO-CORRECTION control (rhat = 0), so a failure is a real
failure rather than a scaling artefact.

Usage
    .venv\\Scripts\\python.exe run_stage2_residual.py
    .venv\\Scripts\\python.exe run_stage2_residual.py --seeds 42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H            # noqa: E402
from models import xlstm_direct as X     # noqa: E402
from models import xlstm_exog as XE      # noqa: E402

RES = ROOT / "results"
CUTOFF = "201912"
OOF_LO, OOF_HI = "201901", "202111"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]


def oof_predictions(seed: int, epochs: int):
    """train the base model on <=2017-12 and predict the unseen OOF block"""
    feat = pd.read_csv(ROOT / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(ROOT / "data" / "scaling.csv", dtype={"series": str})

    periods, series, F, Y, KF = X.build_tensors(feat)
    t0 = time.time()
    model, meta = X.train_model(F, Y, KF, periods, series, CUTOFF,
                                epochs=epochs, verbose=True, purge_val=True, seed=seed)
    meta["train_sec"] = round(time.time() - t0, 1)

    rows = []
    for o in [p for p in periods if OOF_LO <= p <= OOF_HI]:
        pr = X.predict_at(model, F, KF, periods, series, o)
        if pr is None:
            continue
        for h in X.HEADS:
            tgt = H.month_add(o, h)
            if tgt not in wide.index:
                continue
            yhat = H.inverse_transform(pr[h], np.asarray(series), scaling)
            for s, yp in zip(series, yhat):
                rows.append((o, h, s, float(wide.loc[tgt, s]), float(yp)))
    df = pd.DataFrame(rows, columns=["origin", "h", "series", "y_true", "y_pred"])
    df["seed"] = seed
    return df, meta


def design(origin_periods, series, exog_z, exog_cols):
    """exog z-scores + series one-hot (drop first) for every (origin, series) cell"""
    n_o, n_s = len(origin_periods), len(series)
    E = exog_z.loc[origin_periods, exog_cols].to_numpy(dtype=float)   # (n_o, k)
    k = E.shape[1]
    Xd = np.zeros((n_o * n_s, k + n_s - 1))
    Yd = np.tile(np.arange(n_s), n_o)
    for i in range(n_o):
        blk = slice(i * n_s, (i + 1) * n_s)
        Xd[blk, :k] = E[i]
        for j in range(1, n_s):
            Xd[blk, k + j - 1] = (Yd[blk] == j).astype(float)
    return Xd, Yd


def ridge_fit(Xd, y, alpha):
    mu = Xd.mean(axis=0)
    sd = Xd.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (Xd - mu) / sd
    ym = y.mean()
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    b = Xs.T @ (y - ym)
    w = np.linalg.solve(A, b)
    return mu, sd, ym, w


def ridge_pred(theta, Xd):
    mu, sd, ym, w = theta
    return ((Xd - mu) / sd) @ w + ym


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse cached OOF predictions if present")
    ap.add_argument("--demean", action="store_true",
                    help="subtract the OOF mean residual per horizon before fitting, "
                         "so the correction models only the EXOGENOUS-EXPLAINED part "
                         "and not a regime-specific level shift")
    args = ap.parse_args()

    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    naive_mae = H.seasonal_naive_mae_train(wide)
    exog = XE.load_exog()
    exog_cols = list(exog.columns)
    tr_end = "202312"
    tr_mask = exog.index <= tr_end
    exog_z = (exog - exog[tr_mask].mean()) / exog[tr_mask].std(ddof=0)

    print("=" * 86)
    print("STAGE 2 — two-stage residual correction (Ridge on out-of-fold residuals)")
    print("=" * 86)
    print(f"  base-model training origins  <= 2017-12 (purged)")
    print(f"  out-of-fold origins          {OOF_LO} .. {OOF_HI}")
    print(f"  test origins                 {', '.join(H.ORIGINS[:1])} .. {H.ORIGINS[-1:]}")
    print(f"  exog used                    {exog_cols}")

    all_eval, all_alpha, metas = [], [], []
    for seed in args.seeds:
        cache = RES / f"stage2_oof_s{seed}.csv"
        if args.reuse and cache.exists():
            oof = pd.read_csv(cache, dtype={"series": str, "origin": str})
            print(f"\n  [reuse] {cache.name}")
        else:
            print("\n" + "=" * 86)
            print(f"  fitting base model for OOF, seed {seed}")
            print("=" * 86)
            oof, meta = oof_predictions(seed, args.epochs)
            oof.to_csv(cache, index=False)
            metas.append(meta)
            print(f"  OOF rows {len(oof):,}  origins {oof['origin'].nunique()}")

        series = sorted(oof["series"].unique(), key=lambda s: (len(s), s))
        oof_origins = sorted(oof["origin"].unique())

        # ---- base test predictions (Stage-1 baseline, same seed) ----------
        base_file = RES / "exog_preds_z" / f"xLSTM_none_s{seed}.csv"
        if not base_file.exists():
            raise SystemExit(f"missing {base_file} -- run the Stage-1 baseline first")
        base = pd.read_csv(base_file, dtype={"series": str, "origin": str})
        test_origins = sorted(base["origin"].unique())

        # ---- per-horizon Ridge on the MASE-scale residual -----------------
        corr = base.copy()
        corr["y_pred"] = base["y_pred"]
        chosen = []
        for h in X.HEADS:
            o = oof[oof["h"] == h]
            piv_r = np.full((len(oof_origins), len(series)), np.nan)
            for _, row in o.iterrows():
                i = oof_origins.index(row["origin"])
                j = series.index(row["series"])
                piv_r[i, j] = (row["y_true"] - row["y_pred"]) / naive_mae[row["series"]]
            y = piv_r.ravel()
            keep = np.isfinite(y)
            Xd, _ = design(oof_origins, series, exog_z, exog_cols)

            # A mandatory diagnostic: the raw OOF residual has a large NEGATIVE
            # mean that grows with horizon (-0.035 / -0.236 / -0.475 / -0.939 in
            # MASE units) because 2019-2021 is the COVID collapse and the model
            # over-predicted throughout it.  Fitting that mean and re-applying it
            # to 2024 shifts every forecast down by 66-122% and is catastrophic.
            # `--demean` removes that level shift so the Ridge can only exploit
            # the part of the residual that the exogenous block actually explains,
            # which is the question we meant to ask.
            y_mean = float(np.nanmean(y[keep])) if args.demean else 0.0
            y_fit = y - y_mean

            # leave-one-origin-out CV for alpha
            best = (np.inf, ALPHAS[0])
            n_s = len(series)
            for a in ALPHAS:
                err = 0.0
                cnt = 0
                for i in range(len(oof_origins)):
                    tr = np.ones(len(y), bool)
                    tr[i * n_s:(i + 1) * n_s] = False
                    tr &= keep
                    va = keep & ~tr
                    if tr.sum() < 50 or va.sum() == 0:
                        continue
                    th = ridge_fit(Xd[tr], y_fit[tr], a)
                    err += float(((ridge_pred(th, Xd[va]) - y_fit[va]) ** 2).sum())
                    cnt += int(va.sum())
                if cnt and err / cnt < best[0]:
                    best = (err / cnt, a)
            alpha = best[1]
            chosen.append({"seed": seed, "h": h, "alpha": alpha,
                           "cv_mse": round(float(best[0]), 6), "n_oof": int(keep.sum()),
                           "oof_resid_mean": round(y_mean, 4)})
            th = ridge_fit(Xd[keep], y_fit[keep], alpha)

            # apply to the test origins
            Xt, _ = design(test_origins, series, exog_z, exog_cols)
            rhat = ridge_pred(th, Xt).reshape(len(test_origins), len(series))
            for i, o_ in enumerate(test_origins):
                m = (corr["origin"] == o_) & (corr["h"] == h)
                add = np.array([rhat[i, series.index(s)] * naive_mae[s]
                                for s in corr.loc[m, "series"]])
                corr.loc[m, "y_pred"] = corr.loc[m, "y_pred"].to_numpy() + add
            print(f"    h={h:2d}  alpha={alpha:<7g} cv_mse={best[0]:.6f}  n_oof={int(keep.sum())}")

        all_alpha.append(pd.DataFrame(chosen))
        base = base.assign(model="base", seed=seed)
        corr = corr.assign(model="ridge_residual", seed=seed)
        ev = pd.concat([base, corr], ignore_index=True)
        ev.to_csv(RES / f"stage2_predictions_s{seed}{'_demean' if args.demean else ''}.csv", index=False)
        m = H.metrics_from_frame(ev, naive_mae)
        m["seed"] = seed
        all_eval.append(m)

    ev = pd.concat(all_eval, ignore_index=True)
    ev.to_csv(RES / f"stage2_metrics_all{'_demean' if args.demean else ''}.csv", index=False)
    al = pd.concat(all_alpha, ignore_index=True)
    al.to_csv(RES / f"stage2_alpha{'_demean' if args.demean else ''}.csv", index=False)

    print("\n" + "=" * 86)
    print("  MASE by method and horizon (mean over seeds)")
    print("=" * 86)
    piv = ev.pivot_table(index="model", columns="h", values="MASE", aggfunc="mean")
    cols = [c for c in piv.columns if str(c) in {"1", "3", "6", "12", "ALL"}]
    cols = sorted(cols, key=lambda c: 99 if str(c) == "ALL" else int(c))
    print(piv[cols].round(5).to_string())

    print("\n  --- ALL horizon, per seed ---")
    aw = ev[ev["h"].astype(str) == "ALL"].pivot(index="model", columns="seed", values="MASE")
    print(aw.round(5).to_string())
    if {"base", "ridge_residual"} <= set(aw.index):
        d = aw.loc["ridge_residual"] - aw.loc["base"]
        print(f"\n  delta vs base: mean {d.mean():+.5f} ({100*d.mean()/aw.loc['base'].mean():+.2f}%)"
              f"   per-seed {np.round(d.to_numpy(),5)}")

    print("\n" + "=" * 86)
    print("  DIEBOLD-MARIANO  A = base   B = ridge_residual   (dm > 0 => correction helps)")
    print("=" * 86)
    sfx = "_demean" if args.demean else ""
    allpred = pd.concat(
        [pd.read_csv(RES / f"stage2_predictions_s{s}{sfx}.csv",
                     dtype={"series": str, "origin": str}) for s in args.seeds],
        ignore_index=True)
    dm_rows = []
    for h in H.HORIZONS:
        a = allpred[(allpred["model"] == "base") & (allpred["h"] == h)]
        b = allpred[(allpred["model"] == "ridge_residual") & (allpred["h"] == h)]
        ea = (a["y_true"] - a["y_pred"]).groupby([a["origin"], a["series"]]).mean()
        eb = (b["y_true"] - b["y_pred"]).groupby([b["origin"], b["series"]]).mean()
        idx = ea.index.intersection(eb.index)
        if len(idx) < 10:
            continue
        r = H.dm_test(ea.loc[idx].to_numpy(), eb.loc[idx].to_numpy(), h=h, power=2)
        r.update({"h": h})
        dm_rows.append(r)
    dm = pd.DataFrame(dm_rows)
    print(dm[["h", "n", "dm", "p_value", "better"]].to_string(index=False))
    dm.to_csv(RES / f"stage2_dm{'_demean' if args.demean else ''}.csv", index=False)

    if metas:
        import json
        (RES / "stage2_oof_meta.json").write_text(
            json.dumps(metas, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
