"""
Post-hoc analysis of the Stage-1 exogenous ablation.

`run_exog_ablation.py --combine-only` already prints the per-cell means and a
DM test against the baseline.  This script adds the three things that the
write-up actually needs and that the combine step does not do:

1. PER-HORIZON breakdown, because Stage 0 established that the model only beats
   the seasonal naive at h=1 (01_ARCHITECTURE.md §7.2).  A single pooled MASE
   hides the only place where an effect could plausibly show up.

2. DM agains the `random` control, not just against `none`.  `all` carries
   +1,344 parameters that `none` does not, and the random cell carries exactly
   the same +1,344.  So `all` vs `random` is the ONLY contrast that isolates
   exogenous INFORMATION from added CAPACITY.  An `all`-vs-`none` result alone
   is not interpretable.

3. The capacity-corrected effect size:
       info_effect = MASE(all) - MASE(random)
   negative  -> the real columns carry information the noise columns do not
   ~0        -> the model is fitting capacity, not signal
   positive  -> the exogenous columns actively mislead

Also reports the `none` vs `random` gap as the pure cost of widening the input,
and the per-seed spread so a single diverged run cannot masquerade as a result.

Usage
    .venv\\Scripts\\python.exe analyze_exog.py            # standardised pass
    .venv\\Scripts\\python.exe analyze_exog.py --raw      # first (flawed) pass
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
CELL_ORDER = ["none", "cpi", "random1", "visitor", "visitor_mtr", "random2",
              "rain_temp", "macro_fin", "random3", "no_visitor",
              "people_weather_logistics", "all", "random"]
LABEL = {
    "none": "baseline (no exogenous, 0 cols)",
    "cpi": "CPI YoY only (1)",
    "random1": "1 random column (width-1 control)",
    "visitor": "visitor arrivals YoY only (1)",
    "visitor_mtr": "visitor + MTR YoY (2)",
    "macro_fin": "macro price + finance (3)",
    "random3": "3 random columns (width-3 control)",
    "people_weather_logistics": "people + weather + logistics (5)",
    "all": "all 8 exogenous (8)",
    "random": "8 random columns (width-8 control)",
}
# The ONLY fair reference for a cell of width w is a noise block of width w.
# Comparing a 3-column real block against an 8-column noise block conflates the
# information effect with the width effect, which is exactly the mistake the
# width-1 and width-3 controls exist to prevent.
WIDTH_MATCH = {
    "cpi": "random1",
    "visitor": "random1",
    "visitor_mtr": "random2",
    "rain_temp": "random2",
    "no_visitor": "random3",              # width 4 -> nearest control is 3
    "macro_fin": "random3",
    "people_weather_logistics": "random",  # width 5 -> nearest is 8
    "all": "random",
}
NOISE_CELLS = {"random", "random1", "random2", "random3"}


def load(suffix: str):
    f = RES / f"exog_ablation_predictions{suffix}.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found -- run run_exog_ablation.py --combine-only first")
    d = pd.read_csv(f, dtype={"series": str, "origin": str})
    if "cell" not in d.columns:
        raise SystemExit("predictions file has no 'cell' column")
    return d


def mase_per_cell_h(pred, naive_mae) -> pd.DataFrame:
    rows = []
    for (cell, h), g in pred.groupby(["cell", "h"], sort=False):
        e = (g["y_true"] - g["y_pred"]).abs()
        per = e.groupby(g["series"]).mean() / naive_mae.reindex(
            sorted(g["series"].unique())).replace(0, np.nan)
        den = float(g["y_true"].abs().sum())
        rows.append({"cell": cell, "h": h, "n": len(g),
                     "MASE": float(np.nanmean(per.values)),
                     "WAPE": float(e.sum() / den) if den else np.nan,
                     "bias_pct": float(100 * (g["y_true"] - g["y_pred"]).sum() / den)})
    for cell, g in pred.groupby("cell", sort=False):
        e = (g["y_true"] - g["y_pred"]).abs()
        per = e.groupby(g["series"]).mean() / naive_mae.reindex(
            sorted(g["series"].unique())).replace(0, np.nan)
        den = float(g["y_true"].abs().sum())
        rows.append({"cell": cell, "h": "ALL", "n": len(g),
                     "MASE": float(np.nanmean(per.values)),
                     "WAPE": float(e.sum() / den) if den else np.nan,
                     "bias_pct": float(100 * (g["y_true"] - g["y_pred"]).sum() / den)})
    return pd.DataFrame(rows)


def dm_between(pred, a_cell, b_cell, h, power=2):
    a = pred[(pred["cell"] == a_cell) & (pred["h"] == h)]
    b = pred[(pred["cell"] == b_cell) & (pred["h"] == h)]
    if a.empty or b.empty:
        return None
    ea = (a["y_true"] - a["y_pred"]).groupby([a["origin"], a["series"]]).mean()
    eb = (b["y_true"] - b["y_pred"]).groupby([b["origin"], b["series"]]).mean()
    idx = ea.index.intersection(eb.index)
    if len(idx) < 10:
        return None
    r = H.dm_test(ea.loc[idx].to_numpy(), eb.loc[idx].to_numpy(), h=h, power=power)
    r.update({"A": a_cell, "B": b_cell, "h": h})
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    suffix = "_raw" if args.raw else ""
    mode = "RAW (first pass, flawed)" if args.raw else "STANDARDISED (train-only z)"

    pred = load(suffix)
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    naive_mae = H.seasonal_naive_mae_train(wide)

    print("=" * 88)
    print(f"STAGE-1 EXOGENOUS ANALYSIS   mode = {mode}")
    print(f"  predictions: {len(pred):,} rows, cells = {sorted(pred['cell'].unique())}")
    print("=" * 88)

    # ---------------- per-horizon table ----------------
    tb = mase_per_cell_h(pred, naive_mae)
    tb["cell"] = pd.Categorical(tb["cell"], CELL_ORDER, ordered=True)
    tb = tb.sort_values(["cell", "h"])
    print("\n--- MASE by cell x horizon (pooled over seeds; lower is better) ---")
    piv = tb.pivot(index="cell", columns="h", values="MASE")
    piv = piv[[c for c in [1, 3, 6, 12, "ALL"] if c in piv.columns]]
    print(piv.round(5).to_string())

    print("\n--- WAPE by cell x horizon ---")
    pivw = tb.pivot(index="cell", columns="h", values="WAPE")
    pivw = pivw[[c for c in [1, 3, 6, 12, "ALL"] if c in pivw.columns]]
    print((pivw * 100).round(2).to_string())

    print("\n--- bias_pct by cell x horizon (negative = over-prediction) ---")
    pivb = tb.pivot(index="cell", columns="h", values="bias_pct")
    pivb = pivb[[c for c in [1, 3, 6, 12, "ALL"] if c in pivb.columns]]
    print(pivb.round(2).to_string())

    tb.to_csv(RES / f"stage1_exog_by_h{suffix}.csv", index=False)

    # ---------------- the contrast that matters ----------------
    print("\n" + "=" * 88)
    print("CAPACITY-CORRECTED EFFECT")
    print("  none   = no exogenous block            (20,496 params)")
    print("  random = 8 NOISE columns, same width   (21,840 params)  <- capacity control")
    print("  all    = 8 REAL columns, same width    (21,840 params)")
    print("  info_effect = MASE(all) - MASE(random):  <0 real information, >0 misleading")
    print("=" * 88)

    base = tb[tb["h"] == "ALL"].set_index("cell")["MASE"]
    rows = []
    for cell in CELL_ORDER:
        if cell not in base.index:
            continue
        rows.append({"cell": cell, "MASE_ALL": round(float(base[cell]), 5),
                     "vs_none_%": round(100 * (base[cell] / base["none"] - 1), 2)
                     if "none" in base.index else np.nan,
                     "vs_random_%": round(100 * (base[cell] / base["random"] - 1), 2)
                     if "random" in base.index else np.nan})
    eff = pd.DataFrame(rows)
    print(eff.to_string(index=False))

    if "all" in base.index and "random" in base.index:
        d = float(base["all"] - base["random"])
        print(f"\n  information effect  MASE(all) - MASE(random) = {d:+.5f} "
              f"({100*d/base['random']:+.2f}%)")
        print("  -> " + ("REAL COLUMNS CARRY INFORMATION" if d < -0.005 else
                         "NO EVIDENCE of information beyond extra capacity" if d < 0.005 else
                         "REAL COLUMNS ARE WORSE THAN NOISE"))
    if "none" in base.index and "random" in base.index:
        d = float(base["random"] - base["none"])
        print(f"  widening cost       MASE(random) - MASE(none) = {d:+.5f} "
              f"({100*d/base['none']:+.2f}%)")
    eff.to_csv(RES / f"stage1_exog_effect{suffix}.csv", index=False)

    # ---------------- DM: each cell vs baseline AND vs its WIDTH-MATCHED control
    print("\n" + "=" * 88)
    print("DIEBOLD-MARIANO (power=2, pooled N=276)")
    print("  vs none   : dm > 0 => baseline worse  => the CELL helps")
    print("  vs control: dm > 0 => the same-width NOISE block worse")
    print("              => the real columns carry information beyond capacity")
    print("=" * 88)
    dm_rows = []
    for cell in CELL_ORDER:
        if cell == "none":
            continue
        refs = ["none"]
        wm = WIDTH_MATCH.get(cell)
        if wm and wm in set(pred["cell"].unique()):
            refs.append(wm)
        for ref in refs:
            if ref == cell:
                continue
            for h in H.HORIZONS:
                r = dm_between(pred, ref, cell, h)
                if r:
                    r["ref"] = ref
                    dm_rows.append(r)
    dm = pd.DataFrame(dm_rows)
    if not dm.empty:
        dm = dm[["A", "B", "ref", "h", "n", "dm", "p_value", "better"]]
        print(dm.to_string(index=False))
        dm.to_csv(RES / f"stage1_exog_dm2{suffix}.csv", index=False)
        print("\n  --- cell vs WIDTH-MATCHED control, significant at 5% ---")
        wmdm = dm[dm["ref"].isin(NOISE_CELLS) & (dm["p_value"] < 0.05)]
        if len(wmdm):
            print(wmdm.to_string(index=False))
        else:
            print("  none -- no real block is distinguishable from same-width noise")

    # ---------------- per-seed spread (divergence check) ----------------
    print("\n" + "=" * 88)
    print("PER-SEED SPREAD  (a cell whose worst seed is far from its best is unstable)")
    print("=" * 88)
    rows = []
    for cell in CELL_ORDER:
        g = pred[pred["cell"] == cell]
        if g.empty:
            continue
        vals = []
        for seed, gg in g.groupby("seed"):
            e = (gg["y_true"] - gg["y_pred"]).abs()
            per = e.groupby(gg["series"]).mean() / naive_mae.reindex(
                sorted(gg["series"].unique())).replace(0, np.nan)
            vals.append(float(np.nanmean(per.values)))
        v = np.array(vals)
        rows.append({"cell": cell, "n_seeds": len(v),
                     "mean": round(float(v.mean()), 5),
                     "std": round(float(v.std(ddof=1)), 5) if len(v) > 1 else np.nan,
                     "min": round(float(v.min()), 5), "max": round(float(v.max()), 5),
                     "spread": round(float(v.max() - v.min()), 5)})
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
