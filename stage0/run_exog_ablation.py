"""
Stage 1 ablation — do exogenous variables actually help?

PROTOCOL
--------
Everything runs under the CORRECTED validation protocol (`purge_val=True`, see
run_purged_val.py / 01_ARCHITECTURE.md §7.3).  Comparing an exog run against the
old leaky baseline would mix two changes at once and prove nothing.

CELLS (each x 3 seeds)
    none     plain XLSTMForecaster, no exogenous block   <- the baseline
    macro_fin     cpi_yoy, rate_z, eeri_yoy
    people_weather_logistics
                  visitor_yoy, mtr_yoy, rain_anom_z, temp_anom_z, cargo_yoy
    all           all 8
    random        all 8 replaced by fixed N(0,1) draws     <- PARAMETER CONTROL

Why the `random` cell matters
-----------------------------
The FiLM variant adds ~1.3k parameters (inp_endo + inp_exog + gate) on top of
20,496.  If `all` beats `none` by less than `random` does, the gain is the extra
capacity, not the exogenous information.  Without this cell the headline claim
"exogenous data helps" is unfalsifiable, which is exactly the kind of result the
course brief penalises.

RESUMABLE
---------
    results/exog_preds/<tag>.csv   results/exog_meta/<tag>.json
A finished tag is skipped.  Use --only <cell>:<seed> from the PowerShell driver
so each fit gets a fresh interpreter (this box has 32 GB RAM and already ran out
of paging space once when two torch processes coexisted).

Usage
    .venv\\Scripts\\python.exe run_exog_ablation.py --only macro_fin:42
    .venv\\Scripts\\python.exe run_exog_ablation.py --combine-only
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

from eval import harness as H            # noqa: E402
from models import xlstm_direct as X     # noqa: E402
from models import xlstm_exog as XE      # noqa: E402

RES = ROOT / "results"

# Two output trees so the flawed first pass is preserved for the write-up:
#   exog_preds/    the UN-standardised exog block (first pass, kept for the record)
#   exog_preds_z/  the train-only z-scored block (the run that counts)
# `--raw` selects the first tree (reproduction only); the default is the z tree.
RAW_MODE = False


def set_mode(raw: bool) -> None:
    global RAW_MODE, PRED_DIR, META_DIR, SUFFIX
    RAW_MODE = raw
    sub = "exog_preds" if raw else "exog_preds_z"
    PRED_DIR = RES / sub
    META_DIR = RES / ("exog_meta" if raw else "exog_meta_z")
    SUFFIX = "_raw" if raw else ""


PRED_DIR = RES / "exog_preds_z"
META_DIR = RES / "exog_meta_z"
SUFFIX = ""

CELLS = {
    "none": [],
    "macro_fin": ["cpi_yoy", "rate_z", "eeri_yoy"],
    "people_weather_logistics": ["visitor_yoy", "mtr_yoy",
                                 "rain_anom_z", "temp_anom_z", "cargo_yoy"],
    "all": XE.ALL_COLS,
    # ---- width-matched noise controls --------------------------------
    # The 8-column ablation is confounded: every cell that adds columns also adds
    # 168 params/column, and `none` vs `random` was already significant at
    # h=3/6/12.  So a cell can only be credited with INFORMATION if it beats a
    # noise block of the SAME width.  `random` is the width-8 control.
    "random": XE.ALL_COLS,
    # ---- narrow blocks (v2.4) --------------------------------------
    # Width costs accuracy, so the fairest test of whether these variables carry
    # any signal is a minimal block.  Width-1 and width-3 real blocks are each
    # paired with a noise block of identical width.
    "cpi": ["cpi_yoy"],
    "visitor": ["visitor_yoy"],
    "visitor_mtr": ["visitor_yoy", "mtr_yoy"],
    "random1": ["cpi_yoy"],
    "random3": ["cpi_yoy", "rate_z", "eeri_yoy"],
    # ---- isolation cells (v2.5) ------------------------------------
    # `visitor` alone scored h=12 MASE 0.87114, i.e. it did NOT reproduce the
    # 5-column collapse to 2.10141.  So the earlier "visitor_yoy is the culprit"
    # explanation was wrong.  These three cells exist only to find out what
    # actually breaks: weather on its own, and everything EXCEPT visitor (which,
    # if it also collapses, exonerates visitor completely).
    "rain_temp": ["rain_anom_z", "temp_anom_z"],
    "random2": ["visitor_yoy", "mtr_yoy"],
    "no_visitor": ["mtr_yoy", "rain_anom_z", "temp_anom_z", "cargo_yoy"],
}
CELL_ORDER = ["none", "cpi", "random1", "visitor", "visitor_mtr", "random2",
              "rain_temp", "macro_fin", "random3", "no_visitor",
              "people_weather_logistics", "all", "random"]
LABEL = {
    "none": "baseline (no exogenous)",
    "cpi": "CPI YoY only (1)",
    "random1": "1 random column (width-1 control)",
    "visitor": "visitor arrivals YoY only (1)",
    "visitor_mtr": "visitor + MTR YoY (2)",
    "random2": "2 random columns (width-2 control)",
    "rain_temp": "rain + temperature anomaly (2)",
    "macro_fin": "macro price + finance (3)",
    "random3": "3 random columns (width-3 control)",
    "no_visitor": "everything EXCEPT visitor (4)",
    "people_weather_logistics": "people + weather + logistics (5)",
    "all": "all 8 exogenous",
    "random": "8 random columns (width-8 control)",
}
# cells whose contents are replaced by noise
NOISE_CELLS = {"random", "random1", "random2", "random3"}


def load_all():
    feat = pd.read_csv(ROOT / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(ROOT / "data" / "scaling.csv", dtype={"series": str})
    exog = XE.load_exog()
    return feat, wide, scaling, exog


def load_exog_random(periods: list[str], cols: list[str],
                     seed: int = 12345) -> pd.DataFrame:
    """pure N(0,1) noise with exactly the requested column names and width.

    Fixed across training seeds on purpose: the control must be identical for all
    three seeds, otherwise the noise itself becomes a source of variance.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((len(periods), len(cols))).astype(np.float32)
    return pd.DataFrame(a, index=periods, columns=list(cols))


def fit_one(feat, wide, scaling, exog, cell: str, seed: int, epochs: int):
    tag = f"xLSTM_{cell}_s{seed}{SUFFIX}"
    cols = CELLS[cell]
    print("\n" + "=" * 78, flush=True)
    print(f"  tag={tag}   cell={cell} ({LABEL[cell]})   seed={seed}   "
          f"epochs={epochs}   n_exog={len(cols)}", flush=True)
    print("=" * 78, flush=True)

    t0 = time.time()
    if not cols:                                   # the baseline cell
        # seed MUST be forwarded: train_model() re-seeds internally, so calling
        # set_seed(seed) alone would silently collapse all three seeds to 42.
        df, meta = X.run(feat, wide, scaling, epochs=epochs, tag=tag,
                         purge_val=True, verbose=True, seed=seed)
        meta = dict(meta)
    else:
        block = exog
        if cell in NOISE_CELLS:
            # width-matched noise control: identical shape, identical column
            # names, identical parameter count -- only the values are noise
            block = load_exog_random(list(exog.index), cols)
        df, meta = XE.run_exog(feat, wide, scaling, block, cols, tag=tag,
                               epochs=epochs, seed=seed, purge_val=True,
                               verbose=True, standardize=(not RAW_MODE))
        meta = dict(meta)
    df["cell"] = cell
    df["seed"] = seed
    meta.update({"cell": cell, "seed": seed, "tag": tag, "label": LABEL[cell],
                 "n_exog": len(cols), "exog_cols": list(cols),
                 "train_sec": round(time.time() - t0, 1)})
    df.to_csv(PRED_DIR / f"{tag}.csv", index=False)
    (META_DIR / f"{tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  saved {tag}  ({len(df):,} rows, {meta['train_sec']}s)", flush=True)
    return df, meta


def combine(naive_mae) -> int:
    files = sorted(PRED_DIR.glob("*.csv"))
    if not files:
        print("no per-tag predictions found")
        return 1
    pred = pd.concat([pd.read_csv(f, dtype={"series": str, "origin": str})
                      for f in files], ignore_index=True)
    pred.to_csv(RES / f"exog_ablation_predictions{SUFFIX}.csv", index=False)

    res = H.metrics_from_frame(pred, naive_mae)
    res.to_csv(RES / f"stage1_exog_by_h{SUFFIX}.csv", index=False)

    rows = []
    for tag, g in pred.groupby("model", sort=False):
        one = H.metrics_from_frame(g, naive_mae)
        allrow = one[one["h"].astype(str) == "ALL"].iloc[0]
        rows.append({"tag": tag, "cell": g["cell"].iloc[0],
                     "seed": int(g["seed"].iloc[0]),
                     "MASE_ALL": float(allrow["MASE"]),
                     "WAPE_ALL": float(allrow["WAPE"]),
                     "bias_pct": float(allrow["bias_pct"]),
                     "n_points": int(allrow["n_points"])})
    per = pd.DataFrame(rows)
    per["cell"] = pd.Categorical(per["cell"], CELL_ORDER, ordered=True)
    per = per.sort_values(["cell", "seed"])
    per.to_csv(RES / f"stage1_exog_perseed{SUFFIX}.csv", index=False)

    print("\n" + "=" * 78)
    print("  PER-RUN (ALL horizon)")
    print("=" * 78)
    print(per.to_string(index=False))

    print("\n" + "=" * 78)
    print("  CELL SUMMARY  (mean +/- std over seeds)   lower MASE is better")
    print("=" * 78)
    agg = []
    for cell in CELL_ORDER:
        g = per[per["cell"] == cell]
        if g.empty:
            continue
        v = g["MASE_ALL"].to_numpy(dtype=float)
        w = g["WAPE_ALL"].to_numpy(dtype=float)
        agg.append({"cell": cell, "n_seeds": len(v), "label": LABEL[cell],
                    "MASE_mean": round(float(v.mean()), 5),
                    "MASE_std": round(float(v.std(ddof=1)) if len(v) > 1 else np.nan, 5),
                    "WAPE_mean": round(float(w.mean()), 5),
                    "WAPE_std": round(float(w.std(ddof=1)) if len(v) > 1 else np.nan, 5)})
    agg = pd.DataFrame(agg)
    print(agg.to_string(index=False))
    agg.to_csv(RES / f"stage1_exog_cells{SUFFIX}.csv", index=False)

    base = per[per["cell"] == "none"]["MASE_ALL"]
    if len(base):
        b = float(base.mean())
        print(f"\n  deltas vs baseline ({b:.5f}):")
        for _, r in agg.iterrows():
            if r["cell"] == "none":
                continue
            d = r["MASE_mean"] - b
            print(f"    {r['cell']:26s} {r['MASE_mean']:.5f}  "
                  f"delta {d:+.5f}  ({(100*d/b):+.2f}%)")

    # ---- DM: each cell vs baseline, pooled per horizon -------------------
    print("\n" + "=" * 78)
    print("  DIEBOLD-MARIANO  baseline (A) vs cell (B), power=2, pooled")
    print("  dm > 0 => baseline has higher loss => the CELL is better")
    print("=" * 78)
    dm_rows = []
    for cell in CELL_ORDER:
        if cell == "none":
            continue
        for h in H.HORIZONS:
            a = pred[(pred["cell"] == "none") & (pred["h"] == h)]
            b = pred[(pred["cell"] == cell) & (pred["h"] == h)]
            if a.empty or b.empty:
                continue
            ea = (a["y_true"] - a["y_pred"]).groupby([a["origin"], a["series"]]).mean()
            eb = (b["y_true"] - b["y_pred"]).groupby([b["origin"], b["series"]]).mean()
            idx = ea.index.intersection(eb.index)
            if len(idx) < 10:
                continue
            r = H.dm_test(ea.loc[idx].to_numpy(), eb.loc[idx].to_numpy(), h=h, power=2)
            r.update({"cell": cell, "h": h, "n": len(idx)})
            dm_rows.append(r)
    dm = pd.DataFrame(dm_rows)
    if not dm.empty:
        dm = dm[["cell", "h", "n", "dm", "p_value", "better"]]
        print(dm.to_string(index=False))
        dm.to_csv(RES / f"stage1_exog_dm{SUFFIX}.csv", index=False)
        sig = dm[dm["p_value"] < 0.05]
        print(f"\n  cells/horizons significant at 5%: {len(sig)} / {len(dm)}")
        if len(sig):
            print(sig.to_string(index=False))

    metas = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(META_DIR.glob("*.json"))]
    (RES / f"stage1_exog_meta{SUFFIX}.json").write_text(
        json.dumps(metas, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    ap.add_argument("--cells", nargs="+", default=CELL_ORDER)
    ap.add_argument("--only", default=None, help="e.g. --only all:42")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="reproduce the first pass: UN-standardised exog block")
    args = ap.parse_args()
    set_mode(args.raw)

    for d in (PRED_DIR, META_DIR):
        d.mkdir(parents=True, exist_ok=True)

    feat, wide, scaling, exog = load_all()
    naive_mae = H.seasonal_naive_mae_train(wide)

    if args.only:
        cell, seed_s = args.only.split(":")
        tag = f"xLSTM_{cell}_s{seed_s}{SUFFIX}"
        if (PRED_DIR / f"{tag}.csv").exists():
            print(f"  [skip] {tag} already done", flush=True)
            return 0
        fit_one(feat, wide, scaling, exog, cell, int(seed_s), args.epochs)
        return 0

    if not args.combine_only:
        t_all = time.time()
        for cell in args.cells:
            for seed in args.seeds:
                tag = f"xLSTM_{cell}_s{seed}{SUFFIX}"
                if (PRED_DIR / f"{tag}.csv").exists():
                    print(f"  [skip] {tag} already done", flush=True)
                    continue
                fit_one(feat, wide, scaling, exog, cell, seed, args.epochs)
        print(f"\n  fitting wall time {time.time() - t_all:.0f}s", flush=True)

    return combine(naive_mae)


if __name__ == "__main__":
    raise SystemExit(main())
