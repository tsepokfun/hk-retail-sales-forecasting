"""
Stage 4 — hierarchical reconciliation of the 23 retail series.

WHY
---
`01_ARCHITECTURE.md` 決策 6 requires post-hoc reconciliation, and Stage 0 already
produced the motive: the channel-independent model under-performs on the two big
PARENT aggregates (series `18` MASE 804 -> 931, `35` 372 -> 529) while beating the
seasonal naive on their CHILDREN (`19`, `23`, `37`, `38`).  A model that gets the
parts right and the total wrong is exactly the case reconciliation is built for.

HIERARCHY (verified on data/panel_wide.csv)
    2  = 3 + 4 + 5 + 6 + 7
    12 = 13 + 14
    18 = 19 + 23 + 51
    35 = 36 + 37 + 38 + 39 + 40

    23 nodes = 4 parents + 19 leaves.  The identities hold to publication
    rounding (max |parent - sum(children)| = 1-2 HK$m, because censtatd publishes
    integers), so the constraint is treated as exact, as is standard practice.

METHODS
    base     the forecasts as they are
    BU       bottom-up: parent := sum(children).  Leaves untouched.
    MinT_ols MinT with W = I.  Orthogonal projection onto the constraint set.
    MinT_wls MinT with W = diag(per-series in-sample seasonal-naive error
             variance).  NO test-period information is used to build W -- the
             variances come from the TRAINING window only, which is the whole
             point: an oracle-W version would leak and is deliberately absent.

    With only parent-vs-children constraints and a block-disjoint A, A W A' is
    diagonal, so the closed form is cheap and exact.

Usage
    .venv\\Scripts\\python.exe run_stage4_reconcile.py
    .venv\\Scripts\\python.exe run_stage4_reconcile.py --seeds 42 7 2024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H          # noqa: E402

RES = ROOT / "results"
HIER = {
    "2": ["3", "4", "5", "6", "7"],
    "12": ["13", "14"],
    "18": ["19", "23", "51"],
    "35": ["36", "37", "38", "39", "40"],
}


def load_base(seeds) -> pd.DataFrame:
    frames = []
    for s in seeds:
        f = RES / "exog_preds_z" / f"xLSTM_none_s{s}.csv"
        if not f.exists():
            raise SystemExit(f"missing {f} -- Stage-1 baseline must be run first")
        d = pd.read_csv(f, dtype={"series": str, "origin": str})
        d["seed"] = s
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def build_A(series: list[str]) -> np.ndarray:
    """constraint matrix: one row per parent, +1 on the parent, -1 on each child"""
    idx = {s: i for i, s in enumerate(series)}
    A = np.zeros((len(HIER), len(series)))
    for r, (p, ch) in enumerate(HIER.items()):
        A[r, idx[p]] += 1.0
        for c in ch:
            A[r, idx[c]] -= 1.0
    return A


def naive_w(wide: pd.DataFrame, series: list[str], train_end: str) -> np.ndarray:
    """per-series variance of the in-sample seasonal-naive error (train only).

    Uses ONLY months <= train_end, so W carries no test-period information.
    """
    tr = wide.loc[[i for i in wide.index if i <= train_end], series]
    err = tr - tr.shift(12)
    v = err.var(axis=0, ddof=1).to_numpy(dtype=float)
    v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
    if np.isnan(v).any():
        fill = np.nanmedian(v)
        v = np.where(np.isnan(v), fill, v)
    return v


def reconcile(vecs: np.ndarray, A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """vecs: (N, 23) forecasts in ORIGINAL units.  Returns reconciled (N, 23).

    MinT:  y~ = y - W A' (A W A')^-1 A y
    """
    if np.allclose(W, np.eye(len(W))):
        M = A.T @ np.linalg.inv(A @ A.T) @ A
    else:
        AW = A @ W
        M = W @ A.T @ np.linalg.inv(AW @ A.T) @ A
    return vecs - vecs @ M.T


def bottom_up(vecs: np.ndarray, series: list[str]) -> np.ndarray:
    out = vecs.copy()
    idx = {s: i for i, s in enumerate(series)}
    for p, ch in HIER.items():
        out[:, idx[p]] = vecs[:, [idx[c] for c in ch]].sum(axis=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    ap.add_argument("--train-end", default=H.TRAIN_END)
    args = ap.parse_args()

    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    naive_mae = H.seasonal_naive_mae_train(wide)

    base = load_base(args.seeds)
    series = sorted(base["series"].unique(), key=lambda s: (len(s), s))
    A = build_A(series)
    v = naive_w(wide, series, args.train_end)
    W_ols = np.eye(len(series))
    W_wls = np.diag(v)

    # ---- sanity: the hierarchy must actually bind on the forecasts ----------
    print("=" * 84)
    print("STAGE 4 — hierarchical reconciliation")
    print("=" * 84)
    print(f"  nodes {len(series)}   parents {len(HIER)}   leaves {len(series)-len(HIER)}")
    print(f"  W_wls per-series naive error variance (train-only), "
          f"range {v.min():.1f} .. {v.max():.1f}")

    rows = []
    for seed, g in base.groupby("seed"):
        # vecs[n, j] for one (origin, h)
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

        variants = {
            "base": P,
            "BU": bottom_up(P, series),
            "MinT_ols": reconcile(P, A, W_ols),
            "MinT_wls": reconcile(P, A, W_wls),
        }
        # how big is the constraint violation before/after, relative to scale
        gap_base = np.abs(P @ A.T).mean()
        for name, PP in variants.items():
            gap = np.abs(PP @ A.T).mean()
            rows.append({"seed": seed, "method": name, "n": len(keys),
                         "mean_abs_gap_hkd_m": float(gap),
                         "gap_vs_base": float(gap / gap_base if gap_base else np.nan)})
        pd.DataFrame(rows).to_csv(RES / "stage4_gaps.csv", index=False)

        # build long prediction frames per variant
        frames = []
        for name, PP in variants.items():
            frames.append(pd.DataFrame({
                "model": name, "seed": seed,
                "origin": np.repeat([k[0] for k in keys], len(series)),
                "h": np.repeat([int(k[1]) for k in keys], len(series)),
                "series": np.tile(series, len(keys)),
                "y_true": Y.ravel(), "y_pred": PP.ravel()}))
        pred = pd.concat(frames, ignore_index=True)
        pred.to_csv(RES / f"stage4_predictions_s{seed}.csv", index=False)

        m = H.metrics_from_frame(pred, naive_mae)
        m["seed"] = seed
        m.to_csv(RES / f"stage4_metrics_s{seed}.csv", index=False)

    # ---- aggregate over seeds ---------------------------------------------
    allm = []
    for s in args.seeds:
        allm.append(pd.read_csv(RES / f"stage4_metrics_s{s}.csv",
                                dtype={"series": str}))
    allm = pd.concat(allm, ignore_index=True)

    print("\n" + "=" * 84)
    print("  MASE by method and horizon (mean over seeds)")
    print("=" * 84)
    piv = allm.pivot_table(index="model", columns="h", values="MASE", aggfunc="mean")
    order = ["base", "BU", "MinT_ols", "MinT_wls"]
    piv = piv.reindex([o for o in order if o in piv.index])
    # `h` round-trips through CSV as object, so match on the string form
    cols = [c for c in piv.columns
            if str(c) in {"1", "3", "6", "12", "ALL"}]
    cols = sorted(cols, key=lambda c: 99 if str(c) == "ALL" else int(c))
    print(piv[cols].round(5).to_string())

    print("\n  --- ALL-horizon, per seed ---")
    aw = allm[allm["h"].astype(str) == "ALL"].pivot(
        index="model", columns="seed", values="MASE").reindex(order)
    print(aw.round(5).to_string())

    print("\n  --- delta vs base, ALL horizon (negative = better) ---")
    b = aw["base"] if "base" in aw else aw.iloc[0]
    for meth in order:
        if meth not in aw.index:
            continue
        d = aw.loc[meth] - b
        print(f"    {meth:9s} mean {d.mean():+.5f}  "
              f"({100*d.mean()/b.mean():+.2f}%)   per-seed {np.round(d.to_numpy(),5)}")

    # ---- parent-only view: the whole point of reconciliation --------------
    print("\n" + "=" * 84)
    print("  PARENT SERIES ONLY (2, 12, 18, 35) — this is what reconciliation targets")
    print("=" * 84)
    prow = []
    for s in args.seeds:
        f = RES / f"stage4_predictions_s{s}.csv"
        d = pd.read_csv(f, dtype={"series": str, "origin": str})
        d = d[d["series"].isin(list(HIER))]
        mm = H.metrics_from_frame(d, naive_mae)
        mm = mm[mm["h"].astype(str) == "ALL"]
        for _, r in mm.iterrows():
            prow.append({"seed": s, "model": r["model"], "MASE_parents": r["MASE"]})
    pdf = pd.DataFrame(prow).pivot(index="model", columns="seed",
                                   values="MASE_parents").reindex(order)
    print(pdf.round(5).to_string())
    pb = pdf["base"] if "base" in pdf else pdf.iloc[0]
    print("\n  --- delta vs base on parents (negative = better) ---")
    for meth in order:
        if meth not in pdf.index:
            continue
        d = pdf.loc[meth] - pb
        print(f"    {meth:9s} mean {d.mean():+.5f}  "
              f"({100*d.mean()/pb.mean():+.2f}%)")
    pdf.to_csv(RES / "stage4_parents.csv")

    gaps = pd.read_csv(RES / "stage4_gaps.csv")
    print("\n  --- constraint violation |A y| in HK$m, mean over cells ---")
    print(gaps.groupby("method")["mean_abs_gap_hkd_m"].mean().round(4).to_string())

    # ---- DM: each method vs base, pooled per horizon ----------------------
    print("\n" + "=" * 84)
    print("  DIEBOLD-MARIANO   A = base   B = method   (power=2, pooled)")
    print("  dm > 0 => base has higher loss => the METHOD helps")
    print("=" * 84)
    dm_rows = []
    allpred = pd.concat(
        [pd.read_csv(RES / f"stage4_predictions_s{s}.csv",
                     dtype={"series": str, "origin": str}) for s in args.seeds],
        ignore_index=True)
    for meth in order:
        if meth == "base":
            continue
        for h in H.HORIZONS:
            a = allpred[(allpred["model"] == "base") & (allpred["h"] == h)]
            b = allpred[(allpred["model"] == meth) & (allpred["h"] == h)]
            if a.empty or b.empty:
                continue
            ea = (a["y_true"] - a["y_pred"]).groupby([a["origin"], a["series"]]).mean()
            eb = (b["y_true"] - b["y_pred"]).groupby([b["origin"], b["series"]]).mean()
            idx = ea.index.intersection(eb.index)
            if len(idx) < 10:
                continue
            r = H.dm_test(ea.loc[idx].to_numpy(), eb.loc[idx].to_numpy(), h=h, power=2)
            r.update({"method": meth, "h": h})
            dm_rows.append(r)
    dm = pd.DataFrame(dm_rows)
    if not dm.empty:
        print(dm[["method", "h", "n", "dm", "p_value", "better"]].to_string(index=False))
        dm.to_csv(RES / "stage4_dm.csv", index=False)
        sig = dm[dm["p_value"] < 0.05]
        print(f"\n  significant at 5%: {len(sig)}/{len(dm)}")
        if len(sig):
            print(sig[["method", "h", "dm", "p_value", "better"]].to_string(index=False))

    allm.to_csv(RES / "stage4_metrics_all.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
