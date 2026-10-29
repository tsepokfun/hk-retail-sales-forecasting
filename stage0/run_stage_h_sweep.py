"""
Stage H -- one-fit-per-process runner for the hyper-parameter grid (task B) and
the two extra seeds of task A.

ONE FIT PER PROCESS
-------------------
This box has already produced a phantom `CUDA error: out of memory` when two
torch processes coexisted, so every fit here is a separate interpreter launched
by `run_stage_h.ps1`.  Nothing in this file loops over fits.

WHAT IS RECORDED
----------------
results/stage_h_preds/<tag>.csv   per-(origin,h,series) predictions, HK$m scale
results/stage_h_meta/<tag>.json   full hyper-parameters + protocol provenance:
                                  n_train_samples, val_origins, max_val_target_period,
                                  n_params, best_val_loss, train_sec

`tag` = xLSTM_<config>_s<seed>, so every later aggregation (3/5-seed ensemble,
DM tests) only has to glob a directory.

Usage
    .venv\\Scripts\\python.exe run_stage_h_sweep.py --list
    .venv\\Scripts\\python.exe run_stage_h_sweep.py --config ctx24 --seed 42
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

RES = ROOT / "results"
PRED_DIR = RES / "stage_h_preds"
META_DIR = RES / "stage_h_meta"

# ---- the shipped configuration (everything not mentioned stays here) --------
SHIPPED = dict(context=12, embed=32, blocks=2, dropout=0.3, weight_decay=5e-3)

# ---- one-at-a-time grid.  Exactly ONE key differs from SHIPPED per config. --
CONFIGS: dict[str, dict] = {
    "base":  {},
    "ctx24": dict(context=24),
    "emb64": dict(embed=64),
    "blk3":  dict(blocks=3),
    "dp02":  dict(dropout=0.2),
    "dp05":  dict(dropout=0.5),
    "wd5e2": dict(weight_decay=5e-2),
    # --- beyond the brief's list, only if GPU time is left over -------------
    "emb16": dict(embed=16),
    "ctx6":  dict(context=6),
    "lr1e3": dict(lr=1e-3),
    "b64":   dict(batch=64),
}

# The brief's stated priority order (it names 6 non-baseline cells; the text also
# says "8", which is inconsistent with its own table -- see the report).
PRIORITY = ["ctx24", "emb64", "dp02", "dp05", "blk3", "wd5e2"]
EXTRA_PRIORITY = ["emb16", "ctx6", "lr1e3", "b64"]

ORDER = ["base"] + PRIORITY + EXTRA_PRIORITY


def tag_of(cfg: str, seed: int) -> str:
    return f"xLSTM_{cfg}_s{seed}"


def load_all():
    feat = pd.read_csv(ROOT / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(ROOT / "data" / "scaling.csv", dtype={"series": str})
    return feat, wide, scaling


def hp_of(cfg: str) -> dict:
    hp = dict(SHIPPED)
    unknown = set(CONFIGS[cfg]) - set(SHIPPED) - {"lr", "batch"}
    if unknown:
        raise KeyError(f"{cfg}: unknown hyper-parameter(s) {unknown}")
    hp.update(CONFIGS[cfg])
    return hp


def fit_one(cfg: str, seed: int, epochs: int = 100) -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    tag = tag_of(cfg, seed)
    if (PRED_DIR / f"{tag}.csv").exists():
        print(f"  [skip] {tag} already done", flush=True)
        return 0
    hp = hp_of(cfg)
    feat, wide, scaling = load_all()
    naive = H.seasonal_naive_mae_train(wide)

    print("=" * 78, flush=True)
    print(f"  tag={tag}  config={cfg}  seed={seed}  epochs={epochs}", flush=True)
    print(f"  hp={hp}", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    df, meta = X.run(feat, wide, scaling, epochs=epochs, tag=tag,
                     purge_val=True, verbose=True, seed=seed, **hp)
    wall = round(time.time() - t0, 1)
    meta = dict(meta)
    meta.update({"tag": tag, "config": cfg, "seed": int(seed),
                 "hp": hp, "wall_sec": wall, "shipped": dict(SHIPPED),
                 "n_diff_from_shipped": len(CONFIGS[cfg]),
                 "diff_from_shipped": dict(CONFIGS[cfg])})
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
    ap.add_argument("--config", default="base", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for c in ORDER:
            print(f"  {c:8s} {hp_of(c)}")
        return 0
    return fit_one(args.config, args.seed, args.epochs)


if __name__ == "__main__":
    raise SystemExit(main())
