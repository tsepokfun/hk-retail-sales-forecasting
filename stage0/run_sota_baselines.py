"""
Stage 0 -- modern-baseline campaign: ONE FIT PER PROCESS.

WHY ONE FIT PER PROCESS
-----------------------
This box has produced a phantom `CUDA error: out of memory` when two torch
processes coexisted (see models/xlstm_exog.py and run_stage_h_sweep.py).  So this
file trains exactly ONE tagged fit and exits.  `run_sota.ps1` drives it in a
strictly sequential loop; nothing here loops over seeds.

The per-series DLinear/NLinear families do fit 23 small models (one per series)
inside a single process -- that is 23 SEQUENTIAL fits owned by one interpreter,
not two coexisting processes, and it is the only way to express "逐序列訓練"
without spawning 23 interpreters.  The report states this.

PROTOCOL (identical for every family, so nothing but the model can differ)
-------------------------------------------------------------------------
origins : train = period indices [ctx-1, cut-24) that pass the window check
          val   = [cut-24, cut-12]      <- purged: max val target = 202312
          test  = H.ORIGINS (202312..202411)
          `cut` = index of 202312.  Both bounds and the window check are copied
          from xlstm_direct.train_model so every family is fitted on exactly the
          same 182 origins x 23 series as the xLSTM champion.  Nothing about
          2024 can reach a fit or a model-selection decision.
loss    : MSE in y_z space on the ANCHORED forecast (the champion's convention)
target  : residual, anchor = y_z(t) (use_anchor False) unless the tag says
          `_anchor`, in which case the champion's 2-term learnable anchor is
          used.  Either way the head output is a residual on top of an anchor.
metric  : never computed here.  results/SOTA_BASELINES.md is produced by
          run_sota_report.py, which calls harness.metrics_from_frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H                    # noqa: E402
from models import itransformer as IT            # noqa: E402
from models import neural_linear as NL           # noqa: E402
from models import xlstm_direct as X             # noqa: E402

RES = ROOT / "results"
PRED_DIR = RES / "sota_preds"
META_DIR = RES / "sota_meta"

VAL_MONTHS = 24            # the champion's purge window, kept identical

# --- families -------------------------------------------------------------
# shared-weights  : channel-independent, ONE weight set, trained by the xLSTM
#                   champion's own loop (train_model) -> strict like-for-like
# per-series      : channel-independent, 23 independently fitted models
# transformer*    : channel-MIXING (iTransformer) and its channel-independent
#                   twin (timetf), same capacity, same loop
FAMILIES = {
    "dlinear_shared":     "DLinear (NN, shared weights, xLSTM loop)",
    "dlinear_kt":         "DLinear (NN, shared weights, TRAINABLE decomposition kernel)",
    "dlinear_perseries":  "DLinear (NN, 23 independent fits)",
    "nlinear_shared":     "NLinear (NN, shared weights, xLSTM loop)",
    "nlinear_perseries":  "NLinear (NN, 23 independent fits)",
    "itransformer":       "iTransformer (channel-mixing, residual anchor)",
    "itransformer_anchor": "iTransformer (channel-mixing, xLSTM learnable anchor)",
    "itransformer_noattn": "iTransformer with the cross-series ATTENTION bypassed (control)",
    "timetf":             "Transformer over time tokens (channel-INDEPENDENT twin)",
    "timetf_anchor":      "Transformer over time tokens (channel-independent, learnable anchor)",
    "xlstm_persist":      "xLSTM with FIXED persistence anchor (control for #9)",
}


# ---------------------------------------------------------------------------
def load_all():
    feat = pd.read_csv(ROOT / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(ROOT / "data" / "scaling.csv", dtype={"series": str})
    return feat, wide, scaling


def origin_plan(F, periods, context: int):
    """train / val period indices, purged.  Mirrors xlstm_direct.train_model."""
    pidx = {p: i for i, p in enumerate(periods)}
    cut = pidx[H.TRAIN_END]
    val_end = cut - max(H.HORIZONS)
    val_from = cut - VAL_MONTHS
    usable = {}
    for i in range(len(periods)):
        lo = i - context + 1
        if lo < 0:
            continue
        usable[i] = not bool(np.isnan(F[lo:i + 1]).any())
    tr = [i for i in range(context - 1, val_from) if usable.get(i, False)]
    va = [i for i in range(val_from, val_end + 1) if usable.get(i, False)]
    if not va:
        raise ValueError("purge left no validation origins")
    max_val_target = periods[min(va[-1] + max(H.HORIZONS), len(periods) - 1)]
    assert max_val_target <= H.TRAIN_END, max_val_target
    return tr, va, {"train_origins": [periods[tr[0]], periods[tr[-1]]],
                    "val_origins": [periods[va[0]], periods[va[-1]]],
                    "max_val_target_period": max_val_target,
                    "n_train_origins": len(tr), "n_val_origins": len(va)}


def rows_from_levels(tag, origin, levels, series, wide, scaling):
    """levels: (n_series, n_heads) in y_z units -> prediction rows in HK$m"""
    rows = []
    for k, h in enumerate(H.HORIZONS):
        tgt = H.month_add(origin, h)
        if tgt not in wide.index:
            continue
        yhat = H.inverse_transform(levels[:, k], np.asarray(series), scaling)
        for s, yp in zip(series, yhat):
            rows.append((tag, origin, h, s, float(wide.loc[tgt, s]), float(yp)))
    return rows


# ---------------------------------------------------------------------------
def fit_family(family: str, seed: int, kernel: int = 3, epochs: int = 100,
               context: int = 12, verbose: bool = True):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feat, wide, scaling = load_all()
    periods, series, F, Y, KF = X.build_tensors(feat)
    tr_i, va_i, plan = origin_plan(F, periods, context)
    Z = Y.astype(np.float64)                      # (P, S) y_z, no NaNs
    n_series = len(series)
    if verbose:
        print(f"  family={family} seed={seed} device={dev}", flush=True)
        print(f"  cube {F.shape}  series {n_series}  ctx {context}", flush=True)
        print(f"  train origins {plan['train_origins'][0]}..{plan['train_origins'][1]} "
              f"({plan['n_train_origins']})  val {plan['val_origins'][0]}.."
              f"{plan['val_origins'][1]} ({plan['n_val_origins']})  "
              f"max val target {plan['max_val_target_period']}", flush=True)

    t0 = time.time()
    predict_levels = None
    extra: dict = {}

    # ---- A. channel-independent, shared weights, trained by the xLSTM loop ----
    if family in ("dlinear_shared", "nlinear_shared", "dlinear_kt"):
        X.set_seed(seed)

        def factory(ns, n_feat):
            if family == "dlinear_shared":
                return NL.DLinearNet(context=context, kernel=kernel, use_anchor=False,
                                     dropout=0.0)
            if family == "dlinear_kt":
                # brief's wording: "learnable decomposition kernel".  The paper's
                # kernel is fixed, so this arm exists only to measure the
                # difference -- see the report's §9.
                return NL.DLinearNet(context=context, kernel=kernel, use_anchor=False,
                                     dropout=0.0, kernel_trainable=True)
            return NL.NLinearNet(context=context, use_anchor=False, dropout=0.0)

        # dropout must be 0.0 because these models have no dropout layers at
        # all; train_model asserts that the factory agrees with the loop.
        model, meta = X.train_model(F, Y, KF, periods, series, H.TRAIN_END,
                                    epochs=epochs, verbose=verbose, purge_val=True,
                                    model_factory=factory, seed=seed,
                                    context=context, dropout=0.0, batch=128,
                                    lr=5e-4, weight_decay=5e-3)
        meta["anchor_type"] = "fixed persistence"

        @torch.no_grad()
        def predict_levels(i):
            pr = X.predict_at(model, F, KF, periods, series, periods[i], context=context)
            return None if pr is None else np.stack([pr[h] for h in H.HORIZONS], axis=1)

    # ---- B. channel-independent, 23 independent fits -------------------------
    elif family in ("dlinear_perseries", "nlinear_perseries"):
        X.set_seed(seed)
        kind = "dlinear" if family.startswith("dlinear") else "nlinear"
        predict_delta, meta = NL.fit_per_series(
            kind, Z, tr_i, va_i, context=context, kernel=kernel, epochs=epochs,
            lr=5e-4, batch=128, weight_decay=5e-3, seed=seed, device=dev,
            verbose=verbose)
        meta["anchor_type"] = "fixed persistence"

        def predict_levels(i):
            # predict_delta already returns anchored FORECAST LEVELS (the model's
            # forward adds the persistence anchor).  Adding y_z(t) here would
            # double-count the anchor -- that bug produced MASE 2.886 before it
            # was caught by comparing against the shared-weights arm.
            return predict_delta(i)

    # ---- C/D. transformers --------------------------------------------------
    elif family in ("itransformer", "itransformer_anchor", "timetf", "timetf_anchor",
                    "itransformer_noattn"):
        use_anchor = family.endswith("_anchor")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cls = IT.ITransformerNet if family.startswith("itransformer") else IT.TimeTokenCINet
        kw = dict(context=context, d_model=32, n_heads_attn=4, layers=2,
                  dropout=0.1, use_anchor=use_anchor)
        if family == "itransformer_noattn":
            kw["cross_attention"] = False
        model = cls(**kw)
        model, meta = IT.train_transformer(
            model, Z, tr_i, va_i, context=context, epochs=epochs, lr=5e-4,
            weight_decay=5e-3, batch=32, device=dev, verbose=verbose)
        meta["anchor_type"] = "learnable 2-term (a_h)" if use_anchor else "fixed persistence"

        def predict_levels(i):
            return IT.predict_transformer(model, Z, i, context=context)

    # ---- E. control: the champion architecture with a persistence anchor -----
    elif family == "xlstm_persist":
        tag = None
        df, meta = X.run(feat, wide, scaling, epochs=epochs, origins=H.ORIGINS,
                         verbose=verbose, tag="tmp", use_anchor=False,
                         use_calendar=False, purge_val=True, seed=seed,
                         context=context)
        meta["anchor_type"] = "fixed persistence"
        df = df[["origin", "h", "series", "y_true", "y_pred"]]
        extra["preds_frame"] = df
        predict_levels = None
    else:
        raise KeyError(family)

    fit_sec = round(time.time() - t0, 1)

    # ---- predictions at the 12 test origins ---------------------------------
    if "preds_frame" in extra:
        df = extra["preds_frame"].copy()
        df["model"] = f"{family}_s{seed}"
    else:
        rows = []
        for o in H.ORIGINS:
            i = periods.index(o)
            lv = predict_levels(i)
            if lv is None:
                if verbose:
                    print(f"  [warn] no window at origin {o}", flush=True)
                continue
            rows += rows_from_levels(f"{family}_s{seed}", o, lv, series, wide, scaling)
        df = pd.DataFrame(rows, columns=["model", "origin", "h", "series",
                                         "y_true", "y_pred"])
    if verbose:
        print(f"  predictions {len(df):,} rows / {df['origin'].nunique()} origins",
              flush=True)

    meta = dict(meta)
    meta.update({"family": family, "seed": int(seed),
                 "tag": f"{family}_s{seed}", "fit_sec": fit_sec,
                 "kernel": (kernel if "dlinear" in family else None),
                 "context": context, "epochs": epochs,
                 "target_form": "residual (anchor + learned delta)",
                 "device": dev, "plan": plan,
                 "n_params_reported": meta.get("n_params", meta.get("n_params_total"))})
    return df, meta


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, v in FAMILIES.items():
            print(f"  {k:22s} {v}")
        return 0
    if not args.family or args.seed is None:
        ap.error("--family and --seed are required unless --list is given")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.family}_s{args.seed}" + (f"_k{args.kernel}"
                                           if "dlinear" in args.family else "")
    out_csv = PRED_DIR / f"{tag}.csv"
    out_json = META_DIR / f"{tag}.json"
    if out_csv.exists() and not args.force:
        print(f"  [skip] {tag} already done", flush=True)
        return 0

    print("=" * 78, flush=True)
    print(f"  tag={tag}   {FAMILIES[args.family]}", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    df, meta = fit_family(args.family, args.seed, kernel=args.kernel,
                          epochs=args.epochs, context=args.context, verbose=True)
    meta["wall_sec"] = round(time.time() - t0, 1)
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")

    _v = meta.get("best_val_loss")
    _vtxt = f"{_v:.4g}" if isinstance(_v, (int, float)) else "n/a"
    print(f"  >>> {tag}  fit_sec={meta['fit_sec']}  wall={meta['wall_sec']}  "
          f"n_params={meta['n_params_reported']}  "
          f"val={_vtxt}  epoch={meta.get('best_epoch')}", flush=True)
    print(f"  saved {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
