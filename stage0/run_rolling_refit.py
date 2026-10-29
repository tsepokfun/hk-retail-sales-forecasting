"""TASK 1 — remove the protocol asymmetry: rolling refit with per-horizon masks.

THE PROBLEM
-----------
FINAL_REPORT's models were trained with `purge_val=True, val_months=24`, which
puts `val_from = 202312 - 24 = 202112`.  Training origins therefore stop at
2021-11: the weights that forecast 2024 never saw 2022, 2023 or early 2024.
The classical baselines reproduced in `stage0/repro_stats.py` are refit AT EACH
TEST ORIGIN, so they do see all of that history.  Comparing 0.85403 (ETS,
rolling) with 0.90198 (iTransformer, frozen in 2021) is therefore not a fair
test of the model.

WHAT THIS SCRIPT DOES
---------------------
Re-estimates the neural models at every test origin o, using every origin i
whose h=1 label exists at time o (i <= o-1), and applies a PER-HORIZON CAUSAL
MASK: at origin o only the horizons h with i + h <= o have happened, so only
those contribute to the loss.

    loss = sum_{(i,h): i+h<=o} (yhat - y)^2 / count

For i <= o-12 all four horizons are observed (identical to the old protocol);
for i = o-1 only h=1 is, and the rest are masked out.  Nothing is read that was
not published by origin o: no leakage.

Validation / early stopping uses only FULLY observed origins (i + 12 <= o), so
the stopping rule is causal too.  `--fixed-epochs N` disables early stopping.

Everything is evaluated on the locked protocol: origins 202312..202411,
h in {1,3,6,12}, 23 series = 1104 points, via harness.metrics_from_frame.

Usage
-----
    stage0\\.venv\\Scripts\\python.exe stage0\\run_rolling_refit.py itransformer
    stage0\\.venv\\Scripts\\python.exe stage0\\run_rolling_refit.py xlstm
    stage0\\.venv\\Scripts\\python.exe stage0\\run_rolling_refit.py both
    ... --seeds 42 --origins 202312,202401          # smoke test
"""
from __future__ import annotations

import argparse
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
import xlstm_direct as XD  # noqa: E402
import run_stage4_reconcile as R4  # noqa: E402

CONTEXT = 12
HEADS = list(H.HORIZONS)
MAXH = max(HEADS)
OUT = HERE / "results" / "rolling"


# ---------------------------------------------------------------------------
# masked window construction
# ---------------------------------------------------------------------------
def masked_windows_zi(Z: np.ndarray, origins: list[int], o_idx: int, n_core: int):
    """iTransformer layout: (n, S, L) windows + (n, S, H) targets + (n, 1, H) mask."""
    W, Yt, M = [], [], []
    for i in origins:
        W.append(Z[i - CONTEXT + 1:i + 1, :].T)                 # (S, L)
        tgt, msk = [], []
        for h in HEADS:
            ok = (i + h) <= o_idx and (i + h) < Z.shape[0]
            tgt.append(Z[i + h, :n_core] if ok else np.full(n_core, np.nan))
            msk.append(1.0 if ok else 0.0)
        Yt.append(np.stack(tgt, axis=-1))                       # (S, H)
        M.append(np.asarray(msk, dtype=np.float32))             # (H,)
    return (torch.tensor(np.stack(W), dtype=torch.float32),
            torch.tensor(np.stack(Yt), dtype=torch.float32),
            torch.tensor(np.stack(M)[:, None, :], dtype=torch.float32))


def masked_samples_xlstm(p_idx, s_idx, F, Y, KF, o_idx: int):
    """xLSTM layout: (n, T, F) windows + (n, H) targets + (n, H) mask."""
    Xs, Ss, Cs, Ts, Ms = [], [], [], [], []
    for i in p_idx:
        lo = i - CONTEXT + 1
        if lo < 0:
            continue
        for j in s_idx:
            win = F[lo:i + 1, j, :]
            if np.isnan(win).any():
                continue
            tg, mk = [], []
            for h in HEADS:
                ok = (i + h) <= o_idx and (i + h) < Y.shape[0]
                tg.append(float(Y[i + h, j]) if ok else np.nan)
                mk.append(1.0 if ok else 0.0)
            if sum(mk) == 0:
                continue
            Xs.append(win); Ss.append(j); Cs.append(KF[i]); Ts.append(tg); Ms.append(mk)
    return (np.asarray(Xs, np.float32), np.asarray(Ss, np.int64),
            np.asarray(Cs, np.float32), np.asarray(Ts, np.float32),
            np.asarray(Ms, np.float32))


def usable_origins(Z: np.ndarray, upto: int) -> list[int]:
    out = []
    for i in range(CONTEXT - 1, upto + 1):
        w = Z[i - CONTEXT + 1:i + 1, :]
        if np.isfinite(w).all():
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# rolling trainers
# ---------------------------------------------------------------------------
def train_itransformer(Z, origins, o_idx, n_core, device, fixed_epochs=None,
                       lr=5e-4, wd=5e-3, batch=32, val_block=12, no_partial=False,
                       max_epochs=100):
    model = X.ITransformerNet(context=CONTEXT, d_model=32, n_heads_attn=4,
                              layers=2, dropout=0.1).to(device)
    # causal validation: only origins whose h=12 label has happened
    va = [i for i in origins if i + MAXH <= o_idx][-val_block:]
    upper = (o_idx - MAXH) if no_partial else (o_idx - 1)
    tr = [i for i in origins if i <= upper and i not in va]
    Wtr, Ytr, Mtr = masked_windows_zi(Z, tr, o_idx, n_core)
    if len(va) >= 3:
        Wva, Yva, Mva = masked_windows_zi(Z, va, o_idx, n_core)
        Wva_d, Yva_d, Mva_d = Wva.to(device), Yva.to(device), Mva.to(device)
        va_mask = Mva_d.expand(-1, Yva_d.shape[1], -1) > 0
    else:
        Wva_d = Yva_d = Mva_d = va_mask = None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = fixed_epochs or max_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, best_ep = float("inf"), None, -1
    n = len(tr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            xb, yb, mb = Wtr[idx].to(device), Ytr[idx].to(device), Mtr[idx].to(device)
            opt.zero_grad()
            out = model(xb)
            m = mb.expand(-1, out.shape[1], -1) > 0
            if m.sum() == 0:
                continue
            loss = ((out - torch.nan_to_num(yb)) ** 2)[m].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if Wva_d is not None:
            model.eval()
            with torch.no_grad():
                o_va = model(Wva_d)
                v = float(((o_va - torch.nan_to_num(Yva_d)) ** 2)[va_mask].mean())
            if v < best:
                best, best_ep = v, ep
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        elif fixed_epochs is None:
            best_state = None
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"n_train": n, "n_val": len(va), "best_epoch": best_ep,
                   "best_val": (None if best == float("inf") else round(best, 6)),
                   "n_horizon_labels": int(Mtr.sum().item())}


def train_xlstm(feat_tensors, origins, o_idx, device, fixed_epochs=None,
                lr=5e-4, wd=5e-3, batch=128, seed=None, max_epochs=100):
    periods, series, F, Y, KF = feat_tensors
    model = XD.XLSTMForecaster(n_series=len(series), n_feat=F.shape[2] - 1,
                               dropout=0.3, use_anchor=True, use_calendar=False)
    model = model.to(device)
    va = [i for i in origins if i + MAXH <= o_idx][-12:]
    tr = [i for i in origins if i <= o_idx - 1 and i not in va]
    s_idx = np.arange(len(series))
    Xtr, Str, Ctr, Ttr, Mtr = masked_samples_xlstm(tr, s_idx, F, Y, KF, o_idx)
    Xva = Cva = Tva = Mva = None
    if len(va) >= 3:
        Xva, Sva, Cva, Tva, Mva = masked_samples_xlstm(va, s_idx, F, Y, KF, o_idx)

    def to_dev(*arrs):
        return [torch.tensor(a).to(device) if a is not None else None for a in arrs]

    Xd, Sd, Cd, Td, Md = to_dev(Xtr, Str, Ctr, Ttr, Mtr)
    if Xva is not None:
        Xv, Sv, Cv, Tv, Mv = to_dev(Xva, Sva, Cva, Tva, Mva)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = fixed_epochs or max_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, best_ep = float("inf"), None, -1
    n = Xd.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            out = model(Xd[idx], Sd[idx], Cd[idx])
            m = Md[idx] > 0
            if m.sum() == 0:
                continue
            loss = ((out - torch.nan_to_num(Td[idx])) ** 2)[m].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if Xva is not None:
            model.eval()
            with torch.no_grad():
                o_va = model(Xv, Sv, Cv)
                mv = Mv > 0
                v = float(((o_va - torch.nan_to_num(Tv)) ** 2)[mv].mean())
            if v < best:
                best, best_ep = v, ep
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"n_train": int(n), "n_val": len(va), "best_epoch": best_ep,
                   "best_val": (None if best == float("inf") else round(best, 6)),
                   "n_horizon_labels": int(Md.sum().item())}


# ---------------------------------------------------------------------------
def rows_itransformer(model, Z, o_idx, origin, series, wide, scaling, tag, device):
    n_core = len(series)
    W = torch.tensor(Z[o_idx - CONTEXT + 1:o_idx + 1, :].T[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        lv = model(W)[0, :n_core, :].cpu().numpy()
    return _to_rows(tag, origin, lv, series, wide, scaling)


def rows_xlstm(model, feat_tensors, o_idx, origin, series, wide, scaling, tag, device):
    """Predict at origin o.  NOTE: do NOT reuse masked_samples_xlstm here — at
    the origin itself no horizon has happened yet (i+h > o for every h>=1), so
    that builder would skip every sample and return an empty array."""
    periods, ser, F, Y, KF = feat_tensors
    lo = o_idx - CONTEXT + 1
    Xa = np.ascontiguousarray(F[lo:o_idx + 1, :, :].transpose(1, 0, 2))   # (S, T, F)
    Sa = np.arange(len(ser))
    Ca = np.repeat(KF[o_idx][None, :], len(ser), axis=0)
    with torch.no_grad():
        out = model(torch.tensor(Xa).to(device), torch.tensor(Sa).to(device),
                    torch.tensor(Ca).to(device)).cpu().numpy()      # (S, H)
    return _to_rows(tag, origin, out, series, wide, scaling)


def _to_rows(tag, origin, levels, series, wide, scaling):
    rows = []
    for k, h in enumerate(HEADS):
        tgt = H.month_add(origin, h)
        if tgt not in wide.index:
            continue
        yhat = H.inverse_transform(levels[:, k], np.asarray(series), scaling)
        for s, yp in zip(series, yhat):
            rows.append((tag, origin, h, s, float(wide.loc[tgt, s]), float(yp)))
    return rows


def add_mint(df: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    series = list(wide.columns)
    A = R4.build_A(series)
    W = np.diag(R4.naive_w(wide, series, H.TRAIN_END))
    piv = df.pivot_table(index=["origin", "h"], columns="series",
                         values="y_pred", aggfunc="first").reindex(columns=series)
    rec = R4.reconcile(piv.to_numpy(dtype=float), A, W)
    recdf = pd.DataFrame(rec, index=piv.index, columns=pd.Index(series, name="series"))
    lookup = recdf.stack().rename("yp").reset_index()
    lookup.columns = ["origin", "h", "series", "yp"]
    out = df.merge(lookup, on=["origin", "h", "series"], how="left")
    out["y_pred"] = out["yp"]
    out = out.drop(columns=["yp"])
    out["model"] = out["model"] + "+MinT_wls"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["itransformer", "xlstm", "both"])
    ap.add_argument("--seeds", default="42,7,2024")
    ap.add_argument("--origins", default=",".join(H.ORIGINS))
    ap.add_argument("--fixed-epochs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--val-block", type=int, default=12,
                    help="fully-observed origins used for early stopping")
    ap.add_argument("--no-partial", action="store_true",
                    help="drop origins whose h<12 labels are incomplete (isolates masking)")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    test_origins = args.origins.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feat = H.load_features()
    wide = H.load_panel()
    scaling = pd.read_csv(H.DATA / "scaling.csv", dtype={"series": str})
    periods, series, F, Y, KF = XD.build_tensors(feat)
    Z = XD_Y = Y.astype(np.float64)
    pidx = {p: i for i, p in enumerate(periods)}
    n_core = len(series)
    print(f"device={device}  series={n_core}  periods={len(periods)}  "
          f"test origins={len(test_origins)}  seeds={seeds}", flush=True)

    naive_mae = H.seasonal_naive_mae_train(wide)
    summary = []

    for name in (["itransformer", "xlstm"] if args.which == "both" else [args.which]):
        t0 = time.perf_counter()
        all_rows = []
        for origin in test_origins:
            o_idx = pidx[origin]
            origins = usable_origins(Z, o_idx - 1)
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                tag = f"{name}_rolling_s{seed}"
                if name == "itransformer":
                    model, meta = train_itransformer(
                        Z, origins, o_idx, n_core, device,
                        fixed_epochs=args.fixed_epochs,
                        val_block=args.val_block, no_partial=args.no_partial,
                        max_epochs=args.epochs)
                    all_rows += rows_itransformer(model, Z, o_idx, origin, series,
                                                  wide, scaling, tag, device)
                else:
                    model, meta = train_xlstm(
                        (periods, series, F, Y, KF), origins, o_idx, device,
                        fixed_epochs=args.fixed_epochs,
                        max_epochs=args.epochs)
                    all_rows += rows_xlstm(model, (periods, series, F, Y, KF), o_idx,
                                           origin, series, wide, scaling, tag, device)
            print(f"  [{name}] origin {origin}  train_n={meta['n_train']} "
                  f"labels={meta['n_horizon_labels']} best_ep={meta['best_epoch']} "
                  f"elapsed={time.perf_counter() - t0:.0f}s", flush=True)

        df = pd.DataFrame(all_rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])
        df.to_csv(OUT / f"preds_{name}_rolling{args.tag}.csv", index=False)
        ens = df.groupby(["origin", "h", "series", "y_true"], as_index=False)["y_pred"].mean()
        ens["model"] = f"{name}_rolling_ENS{len(seeds)}"
        both = pd.concat([df, ens], ignore_index=True)
        drec = add_mint(ens, wide)
        both = pd.concat([both, drec], ignore_index=True)
        both.to_csv(OUT / f"preds_{name}_rolling_mint{args.tag}.csv", index=False)
        m = H.metrics_from_frame(both, naive_mae)
        m.to_csv(OUT / f"metrics_{name}_rolling{args.tag}.csv", index=False)
        print(m[m["h"] == "ALL"].to_string(index=False), flush=True)
        summary.append({"model": name, "seconds": round(time.perf_counter() - t0, 1),
                        "n_points": len(df)})

    pd.DataFrame(summary).to_csv(OUT / f"rolling_meta{args.tag}.csv", index=False)
    print("done in", round(time.perf_counter() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
