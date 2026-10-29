"""Ablation: exogenous variables as EXTRA TOKENS in the iTransformer.

Why
---
`docs/DESIGN_FUSION_PROPOSAL.md` argues that on this panel the right way to add
heterogeneous data is NOT to widen the per-token feature vector (which the
xLSTM+FiLM runs did, and which cost +32 params/column against only 182 panel
windows) but to ADD TOKENS, because the iTransformer's variate projection is
shared across tokens, so extra tokens are almost free in parameters.

This script tests that claim directly:

    tokens = [23 retail series] + [K exogenous series]

The exogenous blocks enter as pseudo-variates with their own 12-month window,
go through the SAME shared Linear(L -> d_model) and the SAME encoder, and their
output heads are DISCARDED (loss is taken on the 23 core tokens only).  This is
the "cheap token fusion" variant of the proposal; it is deliberately NOT the
TimeXer cross-attention variant, which costs ~12k params per modality.

Protocol: identical to stage0/run_sota_baselines.py (context 12, epochs 100,
val_months 24, purged validation, origins 202312..202411, 3 seeds averaged).

Usage
-----
    stage0\\.venv\\Scripts\\python.exe stage0\\run_exog_token.py exog2
    stage0\\.venv\\Scripts\\python.exe stage0\\run_exog_token.py exog8
    stage0\\.venv\\Scripts\\python.exe stage0\\run_exog_token.py control   # 2 random tokens
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "eval"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))
import harness as H  # noqa: E402
import itransformer as X  # noqa: E402
import xlstm_direct as XD  # noqa: E402  (build_tensors lives here)

CONTEXT = 12
EPOCHS = 100
VAL_MONTHS = 24
SEEDS = (42, 7, 2024)
OUT = HERE / "results" / "exog_token"
HORIZONS = list(H.HORIZONS)
MAXH = max(HORIZONS)

BLOCKS = {
    # the group that showed the only weak positive signal in the FiLM runs
    "exog2": ["visitor_yoy", "mtr_yoy"],
    # everything the project ever built (incl. the two weather anomalies)
    "exog8": ["cpi_yoy", "rate_z", "eeri_yoy", "visitor_yoy", "mtr_yoy",
              "cargo_yoy", "rain_anom_z", "temp_anom_z"],
    # ------------------------------------------------------------------
    # NOTE on causality: exog_wide.csv is already built with the publication
    # lag (build_exog.py: "month t gets the value published for t-1"), and the
    # project verified it with verify_exog_crosscheck.py.  No extra shift is
    # applied here, so this ablation matches the exog definition used by the
    # FiLM runs it is meant to be compared with.
}


def load_all():
    feat = H.load_features()
    wide = H.load_panel()
    scaling = pd.read_csv(H.DATA / "scaling.csv", dtype={"series": str})
    return feat, wide, scaling


def exog_matrix(periods: list[str], cols: list[str] | None, seed: int = 20240) -> np.ndarray:
    """(P, K) z-scored exogenous levels aligned to `periods`, train-only scaling."""
    if cols is None:                                  # control: same-width noise
        rng = np.random.default_rng(seed)
        return rng.standard_normal((len(periods), 2))
    e = pd.read_csv(H.DATA / "exog_wide.csv", dtype={"period": str}).set_index("period")
    e = e.reindex(periods)[cols]
    tr = np.array([p <= H.TRAIN_END for p in periods])
    mu = e[tr].mean()
    sd = e[tr].std(ddof=0).replace(0.0, 1.0)
    return ((e - mu) / sd).to_numpy(dtype=float)


def origin_plan(Z2: np.ndarray, periods: list[str], context: int):
    """train / val period indices, purged.  Mirrors run_sota_baselines.origin_plan."""
    pidx = {p: i for i, p in enumerate(periods)}
    cut = pidx[H.TRAIN_END]
    val_end = cut - MAXH
    val_from = cut - VAL_MONTHS
    usable = {}
    for i in range(context - 1, len(periods)):
        w = Z2[i - context + 1:i + 1, :]
        ok = bool(np.isfinite(w).all())
        # every label must exist too
        for h in HORIZONS:
            if i + h >= len(periods) or not np.isfinite(Z2[i + h, :]).all():
                ok = False
        usable[i] = ok
    tr = [i for i in range(context - 1, val_from) if usable.get(i, False)]
    va = [i for i in range(val_from, val_end + 1) if usable.get(i, False)]
    if not va:
        raise ValueError("purge left no validation origins")
    max_val_target = periods[min(va[-1] + MAXH, len(periods) - 1)]
    assert max_val_target <= H.TRAIN_END, max_val_target
    return tr, va, {"n_train": len(tr), "n_val": len(va),
                    "train": [periods[tr[0]], periods[tr[-1]]],
                    "val": [periods[va[0]], periods[va[-1]]],
                    "max_val_target": max_val_target}


def windows(Z2: np.ndarray, origins, n_core: int, context: int):
    W = np.stack([Z2[i - context + 1:i + 1, :].T for i in origins])       # (n, S2, L)
    Y = np.stack([Z2[np.asarray(origins) + h, :n_core] for h in HORIZONS], axis=-1)  # (n, S, H)
    return torch.tensor(W, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def train(model, Z2, tr, va, n_core, device):
    model = model.to(device)
    Wtr, Ytr = windows(Z2, tr, n_core, CONTEXT)
    Wva, Yva = windows(Z2, va, n_core, CONTEXT)
    Wva, Yva = Wva.to(device), Yva.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = len(Wtr)
    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, 32):
            idx = perm[k:k + 32]
            xb, yb = Wtr[idx].to(device), Ytr[idx].to(device)
            opt.zero_grad()
            out = model(xb)[:, :n_core, :]           # discard exogenous token heads
            loss = ((out - yb) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(Wva)[:, :n_core, :] - Yva) ** 2).mean())
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_loss": best, "n_train": int(n)}


@torch.no_grad()
def predict(model, Z2, i, n_core, device):
    W = torch.tensor(Z2[i - CONTEXT + 1:i + 1, :].T[None], dtype=torch.float32, device=device)
    return model(W)[0, :n_core, :].cpu().numpy()


def run(key: str, wide, scaling, series, Z, periods, tr, va, extra: np.ndarray):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_core = len(series)
    Z2 = np.concatenate([Z, extra], axis=1) if extra.shape[1] else Z
    pidx = {p: i for i, p in enumerate(periods)}
    all_rows = []
    t0 = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = X.ITransformerNet(context=CONTEXT, d_model=32, n_heads_attn=4,
                                  layers=2, dropout=0.1)
        model, meta = train(model, Z2, tr, va, n_core, device)
        tag = f"{key}_s{seed}"
        n_tok = Z2.shape[1]
        for origin in H.ORIGINS:
            i = pidx[origin]
            lv = predict(model, Z2, i, n_core, device)
            for k, h in enumerate(HORIZONS):
                tgt = H.month_add(origin, h)
                if tgt not in wide.index:
                    continue
                yhat = H.inverse_transform(lv[:, k], np.asarray(series), scaling)
                for s, yp in zip(series, yhat):
                    all_rows.append((tag, origin, h, s, float(wide.loc[tgt, s]), float(yp)))
        n_par = int(sum(p.numel() for p in model.parameters()))
        print(f"  {tag}: tokens={n_tok} (core {n_core} + {n_tok - n_core}) "
              f"params={n_par} val={meta['best_val_loss']:.5f} "
              f"elapsed={time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame(all_rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])
    return df


def main(argv: list[str]) -> int:
    key = (argv[1] if len(argv) > 1 else "exog2")
    OUT.mkdir(parents=True, exist_ok=True)
    feat, wide, scaling = load_all()
    periods, series, F, Y, KF = XD.build_tensors(feat)
    Z = Y.astype(np.float64)
    n_core = len(series)

    extra = exog_matrix(periods, BLOCKS.get(key), seed=20240)
    # usable mask must consider exog finiteness -> drop any period with a nan
    if not np.isfinite(extra).all():
        keep = np.isfinite(extra).all(axis=1) & np.isfinite(Z).all(axis=1)
    else:
        keep = np.isfinite(Z).all(axis=1)
    Z2_full = np.concatenate([Z, extra], axis=1)
    Z2_full[~keep, :] = np.nan
    tr, va, plan = origin_plan(Z2_full, periods, CONTEXT)
    print(f"[{key}] core={n_core} extra={extra.shape[1]} params-add=~(K*L*d)"
          f"  train={plan['n_train']} val={plan['n_val']} "
          f"({plan['train'][0]}..{plan['train'][1]} / {plan['val'][0]}..{plan['val'][1]})", flush=True)

    df = run(key, wide, scaling, series, Z, periods, tr, va, extra)
    df.to_csv(OUT / f"preds_{key}.csv", index=False)

    naive_mae = H.seasonal_naive_mae_train(wide)
    m = H.metrics_from_frame(df, naive_mae)
    m.to_csv(OUT / f"metrics_{key}.csv", index=False)
    print(m[m["h"] == "ALL"].to_string(index=False))

    # ensemble of the 3 seeds (comparable to the ENS3 arms in the report)
    ens = (df.groupby(["origin", "h", "series", "y_true"], as_index=False)["y_pred"].mean())
    ens["model"] = f"{key}_ENS3"
    m2 = H.metrics_from_frame(ens, naive_mae)
    m2.to_csv(OUT / f"metrics_{key}_ENS3.csv", index=False)
    print("\nENS3:")
    print(m2[m2["h"] == "ALL"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
