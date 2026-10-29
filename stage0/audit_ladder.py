"""
AUDIT (read-only w.r.t. existing artefacts) — the ensemble ladder, recomputed
independently from the stored per-point xLSTM predictions.

Purpose: confirm or refute the two numbers the report quotes
    ensemble            MASE_ALL = 0.94810
    ensemble+MinT_wls   MASE_ALL = 0.94455
and publish ONE table with MASE / WAPE / bias_pct per horizon, because MASE is
equal-weighted over series while WAPE is HK$-weighted and the two can disagree.

Writes  results/ensemble_ladder.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H                       # noqa: E402
from run_stage4_reconcile import (                  # noqa: E402
    HIER, build_A, bottom_up, naive_w, reconcile,
)

RES = ROOT / "results"
SEEDS = (42, 7, 2024)


def load_xlstm(seeds=SEEDS) -> pd.DataFrame:
    frames = []
    for s in seeds:
        f = RES / "exog_preds_z" / f"xLSTM_none_s{s}.csv"
        d = pd.read_csv(f, dtype={"series": str, "origin": str})
        d["seed"] = s
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["h"] = d["h"].astype(int)
    return d


def order_series(series) -> list[str]:
    return sorted(series, key=lambda s: (len(s), s))


def wide_matrix(g: pd.DataFrame, series: list[str]) -> tuple[list, np.ndarray, np.ndarray]:
    """(keys, Y, P) with one row per (origin, h) and one column per series"""
    keys = sorted({(o, h) for o, h in zip(g["origin"], g["h"])},
                  key=lambda t: (str(t[0]), int(t[1])))
    Y = np.full((len(keys), len(series)), np.nan)
    P = np.full((len(keys), len(series)), np.nan)
    for n, (o, h) in enumerate(keys):
        sub = g[(g["origin"] == o) & (g["h"] == h)]
        m = dict(zip(sub["series"], sub["y_pred"]))
        t = dict(zip(sub["series"], sub["y_true"]))
        for j, s in enumerate(series):
            P[n, j] = m.get(s, np.nan)
            Y[n, j] = t.get(s, np.nan)
    if np.isnan(P).any() or np.isnan(Y).any():
        raise SystemExit("incomplete forecast matrix")
    return keys, Y, P


def frame_from(keys, Y, P, series, model) -> pd.DataFrame:
    return pd.DataFrame({
        "model": model,
        "origin": np.repeat([k[0] for k in keys], len(series)),
        "h": np.repeat([int(k[1]) for k in keys], len(series)),
        "series": np.tile(series, len(keys)),
        "y_true": Y.ravel(),
        "y_pred": P.ravel(),
    })


def main() -> int:
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    naive_mae = H.seasonal_naive_mae_train(wide)

    base = load_xlstm()
    series = order_series(base["series"].unique())
    A = build_A(series)
    W_wls = np.diag(naive_w(wide, series, H.TRAIN_END))
    W_ols = np.eye(len(series))

    # ---- integrity: y_true must not depend on the seed ----------------------
    chk = base.groupby(["origin", "h", "series"])["y_true"].nunique()
    print("=" * 88)
    print("AUDIT — ensemble ladder")
    print("=" * 88)
    print(f"  xLSTM rows {len(base)}   seeds {sorted(base['seed'].unique())}")
    print(f"  y_true identical across seeds for every cell: {bool((chk == 1).all())}")
    # and must equal the panel value
    pv = np.array([
        float(wide.loc[H.month_add(o, int(h)), s])
        for o, h, s in zip(base["origin"], base["h"], base["series"])
    ])
    print(f"  y_true == panel_wide[origin+h]:  max abs diff "
          f"{np.abs(pv - base['y_true'].to_numpy()).max():.6g}")

    # ---- per-seed metrics ---------------------------------------------------
    per_seed = []
    for s in SEEDS:
        m = H.metrics_from_frame(base[base["seed"] == s][
            ["model", "origin", "h", "series", "y_true", "y_pred"]], naive_mae)
        m["seed"] = s
        per_seed.append(m)
    per_seed = pd.concat(per_seed, ignore_index=True)
    per_seed.to_csv(RES / "audit_seed_metrics.csv", index=False)

    allrow = per_seed[per_seed["h"].astype(str) == "ALL"]
    print("\n  per-seed MASE_ALL (recomputed):")
    print(allrow[["seed", "MASE", "WAPE", "bias_pct", "n_points"]]
          .sort_values("seed").to_string(index=False))
    print(f"  mean of per-seed MASE_ALL = {allrow['MASE'].mean():.5f}")

    # ---- build the ladder ---------------------------------------------------
    keys, Y, _ = wide_matrix(base[base["seed"] == SEEDS[0]], series)
    Pseed = {}
    for s in SEEDS:
        k2, Y2, P2 = wide_matrix(base[base["seed"] == s], series)
        assert k2 == keys and np.allclose(Y2, Y)
        Pseed[s] = P2
    P_ens = np.mean(np.stack([Pseed[s] for s in SEEDS]), axis=0)

    variants = {
        "ensemble": P_ens,
        "ensemble+BU": bottom_up(P_ens, series),
        "ensemble+MinT_ols": reconcile(P_ens, A, W_ols),
        "ensemble+MinT_wls": reconcile(P_ens, A, W_wls),
    }
    for s in SEEDS:                       # single seeds, for the ladder's left end
        variants[f"xLSTM_none_s{s}"] = Pseed[s]

    frames = [frame_from(keys, Y, P, series, name) for name, P in variants.items()]
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(RES / "audit_ladder_predictions.csv", index=False)

    met = H.metrics_from_frame(pred, naive_mae)

    # ---- single_seed_mean: average of the per-seed metrics, NOT an ensemble --
    ss = met[met["model"].str.startswith("xLSTM_none_s")].copy()
    ss_mean = ss.groupby("h", as_index=False).agg(
        MASE=("MASE", "mean"), WAPE=("WAPE", "mean"),
        bias_pct=("bias_pct", "mean"), n=("n_points", "mean"))
    ss_mean["variant"] = "single_seed_mean"
    ss_mean["n"] = ss_mean["n"].astype(int)

    agg = (met.groupby(["model", "h"], as_index=False)
              .agg(MASE=("MASE", "first"), WAPE=("WAPE", "first"),
                   bias_pct=("bias_pct", "first"), n=("n_points", "first")))
    agg = agg.rename(columns={"model": "variant"})
    agg = agg[~agg["variant"].str.startswith("xLSTM_none_s")]
    # keep single-seed rows too, they are the ladder's evidence
    ladder = pd.concat([ss_mean, agg], ignore_index=True)
    ladder["h"] = ladder["h"].astype(str)
    ladder = ladder[["variant", "h", "MASE", "WAPE", "bias_pct", "n"]]

    order = ["single_seed_mean", "xLSTM_none_s42", "xLSTM_none_s7",
             "xLSTM_none_s2024", "ensemble", "ensemble+BU",
             "ensemble+MinT_ols", "ensemble+MinT_wls"]
    ladder["__o"] = ladder["variant"].map({v: i for i, v in enumerate(order)})
    ladder["__h"] = ladder["h"].map(lambda x: 99 if x == "ALL" else int(x))
    ladder = ladder.sort_values(["__o", "__h"]).drop(columns=["__o", "__h"])
    ladder.to_csv(RES / "ensemble_ladder.csv", index=False)

    print("\n" + "-" * 88)
    print("  ENSEMBLE LADDER  (MASE = series-equal-weighted, WAPE = HK$-weighted)")
    print("-" * 88)
    print(ladder.round(5).to_string(index=False))

    print("\n" + "-" * 88)
    print("  CLAIM CHECK")
    print("-" * 88)
    def get(v, hh, col="MASE"):
        r = ladder[(ladder["variant"] == v) & (ladder["h"] == hh)]
        return float(r[col].iloc[0])

    for v, claim in [("ensemble", 0.94810), ("ensemble+MinT_wls", 0.94455)]:
        got = get(v, "ALL")
        print(f"  {v:20s} recomputed {got:.5f}   claimed {claim:.5f}   "
              f"delta {got - claim:+.5f}  {'MATCH' if abs(got-claim) < 5e-6 else 'MISMATCH'}")
    print("  ensemble+MinT_wls by horizon: " + "  ".join(
        f"h{h}={get('ensemble+MinT_wls', h):.4f}" for h in ("1", "3", "6", "12")))
    print("  ensemble           by horizon: " + "  ".join(
        f"h{h}={get('ensemble', h):.4f}" for h in ("1", "3", "6", "12")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
