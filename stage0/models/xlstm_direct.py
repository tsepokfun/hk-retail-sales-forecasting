"""
Stage 0.5 - xLSTM with Direct Multi-Horizon heads, LEARNABLE ANCHOR,
and KNOWN-FUTURE calendar features.

Two upgrades over v1 (see RESULTS.md section 3 for the diagnosis):

  (A) LEARNABLE ANCHOR instead of a fixed persistence base.
      v1 :  yhat_h = y_z(t)                            + head_h(hidden)
      v2 :  yhat_h = a_h*y_z(t) + (1-a_h)*y_z(t+h-12)  + head_h(hidden)

      The fixed anchor y_z(t) is the right predictor only at h=12, where t+12
      is the SAME calendar month as t. At h=3 and h=6 the target sits in a
      different season, so the head had to spend capacity undoing the seasonal
      gap - which is why v1 lost to Seasonal Naive at h=3 and h=6.

      y_z(t+h-12) is KNOWN at origin t for every h:
          h=1 -> t-11    h=3 -> t-9    h=6 -> t-6    h=12 -> t
      No leakage. Only 4 extra parameters.

  (B) KNOWN-FUTURE CALENDAR FEATURES per head. At origin t the calendar of the
      target month t+h is fully known, so each head receives that month's LNY
      window shares and month-of-year sin/cos.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "eval"))

import harness as H  # noqa: E402

CONTEXT = 12
EMBED = 32
SERIES_EMB = 32
BLOCKS = 2
HEADS = H.HORIZONS
SEASON = 12
SEED = 42
KF_DIM = len(H.KNOWNFUT_COLS)
DROPOUT = 0.3          # module-level so `dropout=None` resolves at CALL time
                       # (same default-arg-binding trap as SEED: a `dropout=0.3`
                       #  signature default would freeze the value at import)


def set_seed(s=None):
    # NOTE: default must be resolved at CALL time, not def time, otherwise
    # `xlstm_direct.SEED = x` has no effect (classic Python default-arg trap).
    if s is None:
        s = SEED
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


class XLSTMForecaster(nn.Module):
    def __init__(self, n_series, n_feat, embed=None, blocks=None,
                 series_emb=None, context=None, dropout=None, kf_dim=KF_DIM,
                 use_anchor=True, use_calendar=False, anchor_init=2.0):
        super().__init__()
        from xlstm import (
            xLSTMBlockStack, xLSTMBlockStackConfig,
            mLSTMBlockConfig, mLSTMLayerConfig,
            sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig,
        )
        # resolve the module constants at CALL time, not at def time, so that a
        # hyper-parameter sweep can pass explicit values without monkeypatching
        # globals (patching X.CONTEXT after import only moves *some* of the five
        # places CONTEXT is read -- see train_model's factory assertion below)
        embed = EMBED if embed is None else int(embed)
        blocks = BLOCKS if blocks is None else int(blocks)
        series_emb = SERIES_EMB if series_emb is None else int(series_emb)
        context = CONTEXT if context is None else int(context)
        dropout = DROPOUT if dropout is None else float(dropout)
        self.context = context
        self.dropout_p = dropout
        self.inp = nn.Sequential(nn.Linear(n_feat, embed), nn.LayerNorm(embed), nn.GELU())
        self.ser = nn.Embedding(n_series, series_emb)
        self.ser_proj = nn.Linear(series_emb, embed)
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4, conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent"),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")),
            context_length=context, num_blocks=blocks, embedding_dim=embed,
            slstm_at=[blocks - 1],
        )
        self.backbone = xLSTMBlockStack(cfg)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed)
        self.heads = nn.ModuleList([nn.Linear(embed + kf_dim, 1) for _ in HEADS])
        self.use_anchor = bool(use_anchor)
        self.use_calendar = bool(use_calendar)
        self.heads = nn.ModuleList([
            nn.Linear(embed + (kf_dim if use_calendar else 0), 1) for _ in HEADS])
        self.anchor_logit = nn.Parameter(torch.full((len(HEADS),), float(anchor_init)))

    def _embed_input(self, x, sid):
        """input projection + series offset.  Split out so that
        models/xlstm_exog.py can override ONLY this step to insert FiLM gating
        without forking forward()."""
        h = self.inp(x[:, :, :-1])
        return h + self.ser_proj(self.ser(sid)).unsqueeze(1)

    def _backbone(self, h):
        """the sequence mixer.  Split out so an ablation can swap xLSTM for a
        plain nn.LSTM (or GRU) without touching the input projection, the anchor,
        the loss or the training loop -- otherwise the comparison confounds two
        changes at once."""
        return self.backbone(h)

    def forward(self, x, sid, cal):
        anchor = self._anchor_terms(x)
        h = self._embed_input(x, sid)
        h = self._backbone(h)
        h = self.norm(self.drop(h[:, -1, :]))
        return anchor + self._head_delta(h, x, sid, cal)

    def _anchor_terms(self, x):
        """(B, n_heads) anchor  a_h*y_z(t) + (1-a_h)*y_z(t+h-12).

        Split out of forward() so a subclass that needs to change how the HEADS
        are computed (e.g. the horizon-gated FiLM variant) reuses the anchor
        arithmetic instead of copy-pasting it -- a duplicated anchor would be a
        silent way for two "ablations" to differ in more than the one thing they
        claim to differ in."""
        L = x.shape[1]
        base_t = x[:, -1, -1].unsqueeze(1)
        seasonal = []
        for h in HEADS:
            idx = L + h - SEASON - 1
            seasonal.append(base_t[:, 0] if idx < 0 else x[:, idx, -1])
        base_s = torch.stack(seasonal, dim=1)
        if self.use_anchor:
            a = torch.sigmoid(self.anchor_logit).unsqueeze(0)
            return a * base_t + (1.0 - a) * base_s
        return base_t                       # fixed persistence (v1 behaviour)

    def _head_delta(self, h, x, sid, cal):
        """(B, n_heads) per-head residual on top of the anchor.  Split out of
        forward() so a subclass can insert per-head terms (horizon gating)
        without forking the forward pass."""
        if self.use_calendar:
            return torch.cat(
                [self.heads[k](torch.cat([h, cal[:, k, :]], dim=-1)) for k in range(len(HEADS))],
                dim=-1)
        return torch.cat([self.heads[k](h) for k in range(len(HEADS))], dim=-1)

    def anchor_weights(self):
        return torch.sigmoid(self.anchor_logit).detach().cpu().numpy()


def build_tensors(feat):
    periods = sorted(feat["period"].unique())
    series = sorted(feat["series"].unique(), key=lambda s: (len(s), s))
    pidx = {p: i for i, p in enumerate(periods)}
    sidx = {s: i for i, s in enumerate(series)}
    nf = len(H.FEATURE_COLS)
    F = np.full((len(periods), len(series), nf + 1), np.nan, dtype=np.float32)
    Y = np.full((len(periods), len(series)), np.nan, dtype=np.float32)
    for row in feat.itertuples(index=False):
        i, j = pidx[row.period], sidx[row.series]
        F[i, j, :nf] = np.asarray([getattr(row, c) for c in H.FEATURE_COLS], dtype=np.float32)
        F[i, j, nf] = row.y_z
        Y[i, j] = row.y_z
    KF = H.build_known_future(periods, HEADS)
    return periods, series, F, Y, KF


def make_samples(p_idx, s_idx, F, Y, KF, context=CONTEXT, horizons=HEADS):
    X, S, C, T = [], [], [], []
    for i in p_idx:
        lo = i - context + 1
        if lo < 0:
            continue
        for j in s_idx:
            win = F[lo:i + 1, j, :]
            if np.isnan(win).any():
                continue
            tg = [Y[i + h, j] if i + h < Y.shape[0] else np.nan for h in horizons]
            if any(np.isnan(v) for v in tg):
                continue
            X.append(win); S.append(j); C.append(KF[i]); T.append(tg)
    return (np.asarray(X, np.float32), np.asarray(S, np.int64),
            np.asarray(C, np.float32), np.asarray(T, np.float32))


def train_model(F, Y, KF, periods, series, cutoff_period, epochs=100, lr=5e-4,
                batch=128, weight_decay=5e-3, val_months=24, verbose=True,
                loss_weighted=False, use_anchor=True, use_calendar=False, anchor_init=2.0,
                anchor_no_wd=False, purge_val=False, model_factory=None, seed=None,
                val_tail=None, context=None, embed=None, blocks=None, dropout=None):
    """purge_val=False reproduces the Stage-0 baseline exactly (leaky val, kept for
    the before/after comparison).  purge_val=True is the CORRECT protocol: a
    validation origin is only legal if every horizon's target still falls inside
    the training period, otherwise early stopping reads 2024 test-period truth.

    Hyper-parameters (Stage H): `context`/`embed`/`blocks`/`dropout` default to
    None -> the module constants, i.e. the shipped configuration.  They are
    resolved HERE, at call time, and applied to the three places that must agree:
    the sample window (`make_samples`), the model constructor, and `predict_at`.
    Patching `X.CONTEXT` from outside would only hit two of the three, because
    `make_samples(context=CONTEXT)` and `XLSTMForecaster(context=CONTEXT)` bound
    their defaults at import time.
    """
    ctx = CONTEXT if context is None else int(context)
    emb = EMBED if embed is None else int(embed)
    blk = BLOCKS if blocks is None else int(blocks)
    drp = DROPOUT if dropout is None else float(dropout)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pidx = {p: i for i, p in enumerate(periods)}
    cut = pidx[cutoff_period]
    first = ctx - 1
    val_from = cut - val_months
    val_end = (cut - max(HEADS)) if purge_val else cut
    if val_tail is not None:
        # take the LAST `val_tail` legal origins instead of a lookback window:
        # this keeps the validation block anchored to the end of the training
        # period, so shrinking it buys extra TRAINING origins rather than just
        # discarding validation ones.  k=6 -> origins 2022-07..2022-12.
        val_from = val_end - (int(val_tail) - 1)
    tr_i = list(range(first, val_from))
    va_i = list(range(val_from, val_end + 1))
    if not va_i:
        raise ValueError(f"purge_val left no validation origins (val_from={val_from}, "
                         f"val_end={val_end}); increase val_months")
    sidx = list(range(len(series)))
    Xtr, Str, Ctr, Ttr = make_samples(tr_i, sidx, F, Y, KF, context=ctx)
    Xva, Sva, Cva, Tva = make_samples(va_i, sidx, F, Y, KF, context=ctx)
    if verbose:
        print(f"  train samples {len(Xtr):,}   val samples {len(Xva):,}   device {dev}")
        print(f"    train origins {periods[tr_i[0]]}..{periods[tr_i[-1]]}   "
              f"val origins {periods[va_i[0]]}..{periods[va_i[-1]]}")
        print(f"    hp: context={ctx} embed={emb} blocks={blk} dropout={drp} "
              f"wd={weight_decay} lr={lr} epochs={epochs} batch={batch}")
    set_seed(seed)
    if model_factory is None:
        model = XLSTMForecaster(len(series), F.shape[2] - 1, embed=emb, blocks=blk,
                                context=ctx, dropout=drp,
                                use_anchor=use_anchor, use_calendar=use_calendar,
                                anchor_init=anchor_init).to(dev)
    else:
        # injected by models/xlstm_exog.py so the FiLM-gated variant reuses this
        # exact training loop instead of forking a second copy of it
        model = model_factory(len(series), F.shape[2] - 1).to(dev)
        # The factory cannot see these four values, so if it forgot to pass
        # `context=`, the model would silently use context_length=12 while the
        # SAMPLES were cut with 24 -> xLSTM stack shape mismatch or, worse, a
        # silently different experiment.  Fail loudly instead.
        if int(getattr(model, "context", -1)) != ctx:
            raise ValueError(
                f"model_factory built a model with context="
                f"{getattr(model, 'context', None)} but train_model was asked for "
                f"ctx={ctx}; thread `context=...` through the factory.")
        if abs(float(getattr(model, "dropout_p", drp)) - drp) > 1e-12:
            raise ValueError(
                f"model_factory built dropout={getattr(model, 'dropout_p', None)} "
                f"but train_model was asked for dropout={drp}")
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  parameters: {n_par:,}")
    w_series = None
    if loss_weighted:
        raw = np.exp(np.nanmean(Y[tr_i, :], axis=0))
        w = np.log1p(raw - raw.min() + 1.0)
        w_series = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
        if verbose:
            print(f"  scale-weighted loss ON  (w {w.min():.2f}..{w.max():.2f})")
    if anchor_no_wd:
        others = [p_ for n_, p_ in model.named_parameters() if n_ != 'anchor_logit']
        opt = torch.optim.AdamW(
            [{'params': others, 'weight_decay': weight_decay},
             {'params': [model.anchor_logit], 'weight_decay': 0.0}], lr=lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtr_t = torch.from_numpy(Xtr); Str_t = torch.from_numpy(Str)
    Ctr_t = torch.from_numpy(Ctr); Ttr_t = torch.from_numpy(Ttr)
    Xva_t = torch.from_numpy(Xva).to(dev); Sva_t = torch.from_numpy(Sva).to(dev)
    Cva_t = torch.from_numpy(Cva).to(dev); Tva_t = torch.from_numpy(Tva).to(dev)
    n = len(Xtr_t)
    best, best_ep, best_state = float("inf"), -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            xb = Xtr_t[idx].to(dev); sb = Str_t[idx].to(dev)
            cb = Ctr_t[idx].to(dev); yb = Ttr_t[idx].to(dev)
            opt.zero_grad()
            se = (model(xb, sb, cb) - yb) ** 2
            if w_series is not None:
                se = se * w_series[sb].unsqueeze(1)
            loss = se.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = ((model(Xva_t, Sva_t, Cva_t) - Tva_t) ** 2).mean().item()
        if vloss < best:
            best, best_ep = vloss, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep == 1 or ep % 20 == 0 or ep == epochs):
            print(f"    ep {ep:3d}  train {tot / n:.4f}  val {vloss:.4f}"
                  f"{'  *' if ep == best_ep else ''}")
    if best_state is not None:
        model.load_state_dict(best_state)
    aw = model.anchor_weights()
    if verbose:
        print(f"  best val {best:.4f} @ epoch {best_ep}")
        print("  learned anchor a_h  (a->1 persist y_z(t), a->0 seasonal y_z(t+h-12)):"
              if use_anchor else "  anchor: FIXED persistence (ablation)")
        for k, hh in enumerate(HEADS):
            print(f"     h={hh:2d}: a={aw[k]:.4f}  (persist {aw[k]*100:5.1f}% / seasonal {(1-aw[k])*100:5.1f}%)")
    meta = {"n_params": int(n_par), "best_val_loss": float(best), "best_epoch": int(best_ep),
            "n_train_samples": int(len(Xtr)), "n_val_samples": int(len(Xva)),
            "context": ctx, "embedding_dim": emb, "num_blocks": blk,
            "dropout": drp, "weight_decay": float(weight_decay), "lr": float(lr),
            "batch": int(batch), "epochs": int(epochs), "n_series_emb": int(SERIES_EMB),
            "device": dev, "anchor_weights": {str(h): float(aw[k]) for k, h in enumerate(HEADS)},
            "known_future": (H.KNOWNFUT_COLS if use_calendar else []),
            "use_anchor": bool(use_anchor), "use_calendar": bool(use_calendar),
            "loss_weighted": bool(loss_weighted),
            "purge_val": bool(purge_val),
            "val_tail": (int(val_tail) if val_tail is not None else None),
            "val_origins": [periods[va_i[0]], periods[va_i[-1]]],
            "train_origins": [periods[tr_i[0]], periods[tr_i[-1]]],
            "max_val_target_period": periods[min(va_i[-1] + max(HEADS), len(periods) - 1)]}
    return model, meta


@torch.no_grad()
def predict_at(model, F, KF, periods, series, origin, context=None):
    dev = next(model.parameters()).device
    model.eval()
    # read the window from the MODEL, not the module constant: a sweep run with
    # context=24 would otherwise predict from a 12-month window on a model
    # configured for 24 and silently emit garbage instead of raising.
    ctx = int(getattr(model, "context", CONTEXT) if context is None else context)
    pidx = {p: i for i, p in enumerate(periods)}
    i = pidx[origin]
    lo = i - ctx + 1
    if lo < 0:
        return None
    X = np.stack([F[lo:i + 1, j, :] for j in range(len(series))]).astype(np.float32)
    if np.isnan(X).any():
        return None
    S = np.arange(len(series), dtype=np.int64)
    C = np.repeat(KF[i][None, :, :], len(series), axis=0).astype(np.float32)
    out = model(torch.from_numpy(X).to(dev), torch.from_numpy(S).to(dev),
                torch.from_numpy(C).to(dev)).cpu().numpy()
    return {h: out[:, k] for k, h in enumerate(HEADS)}


def run(feat, wide, scaling, epochs=100, origins=None, verbose=True,
        tag="xLSTM", loss_weighted=False, use_anchor=True, use_calendar=False,
        anchor_init=2.0, anchor_no_wd=False, purge_val=False, seed=None,
        val_tail=None, context=None, embed=None, blocks=None, dropout=None,
        weight_decay=5e-3, lr=5e-4, batch=128):
    origins = origins or H.ORIGINS
    periods, series, F, Y, KF = build_tensors(feat)
    if verbose:
        print(f"  cube {F.shape}   calendar {KF.shape}   NaN cells {int(np.isnan(F).sum()):,}")
    t0 = time.time()
    model, meta = train_model(F, Y, KF, periods, series, H.TRAIN_END,
                              epochs=epochs, verbose=verbose, loss_weighted=loss_weighted,
                              use_anchor=use_anchor, use_calendar=use_calendar,
                              anchor_init=anchor_init, anchor_no_wd=anchor_no_wd,
                              purge_val=purge_val, seed=seed, val_tail=val_tail,
                              context=context, embed=embed, blocks=blocks,
                              dropout=dropout, weight_decay=weight_decay, lr=lr,
                              batch=batch)
    meta["train_sec"] = round(time.time() - t0, 1)
    meta["seed"] = seed if seed is not None else SEED
    rows = []
    for o in origins:
        pr = predict_at(model, F, KF, periods, series, o)
        if pr is None:
            continue
        for h in HEADS:
            tgt = H.month_add(o, h)
            if tgt not in wide.index:
                continue
            yhat = H.inverse_transform(pr[h], np.asarray(series), scaling)
            for s, yp in zip(series, yhat):
                rows.append((tag, o, h, s, float(wide.loc[tgt, s]), float(yp)))
    df = pd.DataFrame(rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])
    if verbose:
        print(f"  predictions {len(df):,} rows / {df['origin'].nunique()} origins")
    return df, meta