"""
Stage H, task C -- horizon-specific gate arms.  One fit per process.

ARMS
----
    visitor_mtr_full : visitor_yoy + mtr_yoy, gates OPEN at h=1,3,6,12
    visitor_mtr_h12  : visitor_yoy + mtr_yoy, gates FROZEN-ZERO at h=1,3, open at h=6,12
    (optional, time permitting)
    all_full / all_h12     : the same two arms with all 8 exogenous columns
    visitor_mtr_h1h3       : mirror control, gates open ONLY at h=1,3

Reference points that this script does NOT recompute:
    `none` 3-seed mean-of-per-seed MASE_ALL = 0.98355   (results/stage_h_meta)
    Stage-1 shared-projection `visitor_mtr`  = 1.01204

WHY A BASELINE `none` IS STILL RUN HERE
---------------------------------------
`visitor_mtr_h12` must be compared against the SAME 3 seeds as the other arms, so
`--cell none` reproduces the plain model into the same directory tree.  It is
skipped automatically if results/stage_h_preds/xLSTM_base_s<seed>.csv exists
(the sweep runner already made it for seed 42 / 1 / 1234).

Usage
    .venv\\Scripts\\python.exe run_stage_h_horizon.py --cell visitor_mtr_h12 --seed 42
    .venv\\Scripts\\python.exe run_stage_h_horizon.py --selftest
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
PRED_DIR = RES / "stage_h_preds"
META_DIR = RES / "stage_h_meta"

VM = ["visitor_yoy", "mtr_yoy"]

CELLS: dict[str, dict] = {
    "none":            dict(cols=[], open=(1, 3, 6, 12)),        # plain baseline
    "visitor_mtr_full": dict(cols=VM, open=(1, 3, 6, 12)),
    "visitor_mtr_h12":  dict(cols=VM, open=(6, 12)),
    "visitor_mtr_h1h3": dict(cols=VM, open=(1, 3)),              # mirror control
    # NOT pre-registered.  Added AFTER the two registered arms came back:
    # they are significantly WORSE at h=12 (DM p=0.0023 / 0.0038) and better at
    # h=6 (p=0.040 for the full arm) -- the opposite of the premise that these
    # variables are a long-horizon signal.  These two arms are post-hoc and
    # hypothesis-generating only: their test numbers are NOT confirmatory.
    "visitor_mtr_h6only":  dict(cols=VM, open=(6,)),
    "visitor_mtr_h12only": dict(cols=VM, open=(12,)),
    "all_full":         dict(cols=XE.ALL_COLS, open=(1, 3, 6, 12)),
    "all_h12":          dict(cols=XE.ALL_COLS, open=(6, 12)),
}
ORDER = ["none", "visitor_mtr_full", "visitor_mtr_h12", "visitor_mtr_h1h3",
         "visitor_mtr_h6only", "visitor_mtr_h12only", "all_full", "all_h12"]


def tag_of(cell: str, seed: int) -> str:
    return f"xLSTM_hz-{cell}_s{seed}"


def load_all():
    feat = pd.read_csv(ROOT / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(ROOT / "data" / "scaling.csv", dtype={"series": str})
    return feat, wide, scaling, XE.load_exog()


def fit_one(cell: str, seed: int, epochs: int = 100) -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    tag = tag_of(cell, seed)
    if (PRED_DIR / f"{tag}.csv").exists():
        print(f"  [skip] {tag} already done", flush=True)
        return 0
    spec = CELLS[cell]
    feat, wide, scaling, exog = load_all()
    naive = H.seasonal_naive_mae_train(wide)
    print("=" * 78, flush=True)
    print(f"  tag={tag}  cell={cell}  seed={seed}  cols={spec['cols']}  "
          f"open_heads={spec['open']}", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    if cell == "none":
        df, meta = X.run(feat, wide, scaling, epochs=epochs, tag=tag,
                         purge_val=True, verbose=True, seed=seed)
    else:
        df, meta = XE.run_exog_horizon(
            feat, wide, scaling, exog, spec["cols"], open_heads=spec["open"],
            tag=tag, epochs=epochs, seed=seed, purge_val=True, verbose=True)
    wall = round(time.time() - t0, 1)
    meta = dict(meta)
    meta.update({"tag": tag, "config": f"hz-{cell}", "cell": cell, "seed": int(seed),
                 "hz_open_heads": list(spec["open"]),
                 "hz_closed_heads": [h for h in X.HEADS if h not in spec["open"]],
                 "exog_cols": list(spec["cols"]), "n_exog": len(spec["cols"]),
                 "wall_sec": wall})
    assert meta["purge_val"] is True
    assert meta["max_val_target_period"] <= H.TRAIN_END, meta["max_val_target_period"]
    df.to_csv(PRED_DIR / f"{tag}.csv", index=False)
    (META_DIR / f"{tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    m = H.metrics_from_frame(df, naive)
    print(m[["h", "n_points", "MASE", "WAPE", "bias_pct"]].to_string(index=False))
    allrow = m[m.h.astype(str) == "ALL"].iloc[0]
    print(f"  >>> {tag}  MASE_ALL={allrow['MASE']:.5f}  WAPE_ALL={allrow['WAPE']:.5f}  "
          f"n_train={meta['n_train_samples']}  n_params={meta['n_params']}  "
          f"val={meta['best_val_loss']:.4f}  wall={wall}s", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="visitor_mtr_h12", choices=list(CELLS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        r = XE.selftest()
        print()
        rh = XE.selftest_horizon()
        return 0 if (r["PASS"] and rh["PASS"]) else 1
    return fit_one(args.cell, args.seed, args.epochs)


if __name__ == "__main__":
    raise SystemExit(main())
