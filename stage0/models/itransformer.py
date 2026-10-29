"""
iTransformer (Liu et al., ICLR 2024) -- and the channel-independent twin needed
to answer ablation #9 of 01_ARCHITECTURE.md §7 ("通道獨立 vs 通道混合").

WHAT IS INVERTED, EXACTLY
-------------------------
A classical time-series Transformer treats (time, channel) as (sequence, feature):
every token is one MONTH carrying 23 numbers, so attention mixes channels inside
every token and mixes time across tokens.

iTransformer swaps the axes: every token is one SERIES (a "variate"), and the
token's embedding IS that series' whole lookback window.  Attention therefore
runs ACROSS RETAIL CATEGORIES at a fixed calendar month, and no linear layer
ever sums two categories together before attention.

    zi   = y_z window of series i,  length L=12                 (S=23 tokens)
    tk_i = Linear(L -> d_model)(zi)      # variate token embedding
    H    = TransformerEncoder(tk)        # attention ACROSS SERIES
    d_i  = Linear(d_model -> n_heads)(H_i)   # 4 direct-horizon residuals
    yhat(t+h) = a_h*y_z(t) + (1-a_h)*y_z(t+h-12) + d_i,h

SIMPLIFICATIONS -- stated here so the report can cite them (the brief requires
this, and each one is a reason the conclusion could be weaker than a full run):

  S1  Token content = the 12-step `y_z` window ONLY.  The 27 engineered columns
      the xLSTM champion sees (lags, rolling means/stds, month dummies, LNY
      windows, regime dummies) are NOT given to the Transformer arms.
      Direction of the bias: this handicaps the Transformer arms, so a LOSS is
      NOT attributable to channel mixing alone -> the comparison is a lower
      bound on what channel mixing could do, not an upper bound.  A WIN, on the
      other hand, would be strong evidence, because it would happen with less
      input.
  S2  Embedding is ONE `nn.Linear(L -> d_model)`; the paper's 2-layer MLP is not
      reproduced.  Affects capacity by a few hundred parameters.
  S3  d_model=32, heads=4, layers=2, ff=4*d_model, dropout=0.1, GELU,
      post-LN as in the paper's `nn.TransformerEncoderLayer` default.
  S4  Per-window instance normalisation is KEPT from the paper:
          z~ = (z - mean_L(z)) / std_L(z)
      but the forecast is a RESIDUAL, so the mean is not added back (the anchor
      supplies the level).  Only the scale is inverted: delta = delta~ * std_L.
      This is the one place where the paper's de-normalisation does not carry
      over verbatim, because the paper forecasts levels.
  S5  One shared output head `Linear(d_model -> 4)` applied to every variate
      token -- the paper's own design (a linear map from d_model to the
      forecast length, applied per variate).  No variate identity embedding,
      also as in the paper: tokens are distinguished by content only.
  S6  Training loop: the champion's optimiser/schedule (AdamW lr 5e-4,
      weight_decay 5e-3, cosine over 100 epochs, grad-clip 1.0, MSE in y_z
      space on the ANCHORED forecast so a learnable anchor gets gradients,
      early stopping on the purged validation block whose targets are all
      <= 202312).  Batch 32 instead of 128 because a channel-dependent model
      consumes ONE sample per origin (182 train origins) and 128 would give 1.4
      optimiser steps per epoch.  Documented rather than silently tuned.

THE TWIN (`TimeTokenCINet`) -- why it exists
--------------------------------------------
Comparing iTransformer with the xLSTM answers "channel mixing vs channel
independence" only if the backbone is held fixed, and the xLSTM's backbone is
not a Transformer.  `TimeTokenCINet` is the SAME model with the axes swapped
back: identical embedding width, identical `nn.TransformerEncoderLayer` stack,
identical readout, identical loop, identical anchor -- the only difference is
that attention runs over the 12 TIME steps of ONE series instead of over the 23
series of ONE month.  iTransformer vs TimeTokenCINet is therefore ablation #9
with capacity and optimiser held constant, which the xLSTM comparison cannot be.

`selftest()` proves the two arms actually behave as claimed instead of assuming
it: perturbing ONE series must move the OTHER series' forecasts under
iTransformer (mixing) and must leave them BIT-EXACTLY unchanged under the twin
(channel independence).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eval import harness as H  # noqa: E402

HEADS = H.HORIZONS
SEASON = 12


# ---------------------------------------------------------------------------
# anchor -- same arithmetic as the xLSTM champion, so the anchor is not a confound
# ---------------------------------------------------------------------------
class _Anchor(nn.Module):
    def __init__(self, context: int, n_heads: int = len(HEADS),
                 use_anchor: bool = False, anchor_init: float = 2.0):
        super().__init__()
        self.context = int(context)
        self.use_anchor = bool(use_anchor)
        if self.use_anchor:
            self.anchor_logit = nn.Parameter(torch.full((n_heads,), float(anchor_init)))
        else:
            self.register_buffer("anchor_logit", torch.zeros(n_heads))

    def anchor_terms(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: (B, S, L) -> (B, S, n_heads).

        use_anchor=False : z(t) broadcast over the four heads (pure persistence,
                           i.e. the brief's residual form).
        use_anchor=True  : a_h*z(t) + (1-a_h)*z(t+h-12), identical index
                           arithmetic to `xlstm_direct.XLSTMForecaster.
                           _anchor_terms` (idx = L + h - SEASON - 1, fallback to
                           z(t) when idx < 0) so the two models cannot silently
                           differ in their anchor.
        """
        L = Z.shape[-1]
        base_t = Z[:, :, -1:]                               # (B,S,1)  z(t)
        if not self.use_anchor:
            return base_t.expand(-1, -1, len(HEADS))
        seasonal = []
        for h in HEADS:
            idx = L + h - SEASON - 1
            seasonal.append(Z[:, :, -1] if idx < 0 else Z[:, :, idx])
        base_s = torch.stack(seasonal, dim=-1)               # (B,S,n_heads)
        a = torch.sigmoid(self.anchor_logit).view(1, 1, -1)
        return a * base_t + (1.0 - a) * base_s

    def anchor_weights(self) -> np.ndarray:
        return torch.sigmoid(self.anchor_logit).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 1) iTransformer -- tokens are SERIES
# ---------------------------------------------------------------------------
class ITransformerNet(_Anchor):
    def __init__(self, context: int = 12, d_model: int = 32, n_heads_attn: int = 4,
                 layers: int = 2, n_out: int = len(HEADS), dropout: float = 0.1,
                 ff_mult: int = 4, use_anchor: bool = False,
                 anchor_init: float = 2.0, instance_norm: bool = True,
                 cross_attention: bool = True):
        super().__init__(context=context, use_anchor=use_anchor, anchor_init=anchor_init)
        self.d_model = int(d_model)
        self.instance_norm = bool(instance_norm)
        # cross_attention=False is the sharpest possible form of ablation #9:
        # the token embedding, the encoder LAYERS (norms + feed-forward) and the
        # readout are all unchanged, the parameter count is unchanged -- only the
        # multi-head attention sublayer is bypassed, so no token can read any
        # other token.  Whatever is left is a per-series MLP on a 12-number
        # window, i.e. channel independence with everything else held fixed.
        # (`selftest_noattn()` asserts that a perturbation of one series cannot
        # move any other series' forecast, and that the parameter count is
        # identical to the full model.)
        self.cross_attention = bool(cross_attention)
        self.embed = nn.Linear(self.context, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads_attn,
            dim_feedforward=int(ff_mult) * self.d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             norm=nn.LayerNorm(self.d_model))
        self.proj = nn.Linear(self.d_model, n_out)

    def _encode(self, tok: torch.Tensor) -> torch.Tensor:
        if self.cross_attention:
            return self.encoder(tok)
        h = tok
        for layer in self.encoder.layers:
            # post-LN layer with the attention sublayer replaced by the identity
            # residual: same shapes, same feed-forward, same LayerNorm count.
            h = layer.norm2(h + layer._ff_block(layer.norm1(h)))
        return self.encoder.norm(h) if self.encoder.norm is not None else h

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z: (B, S, L) y_z windows -> (B, S, n_heads) forecasts in y_z units"""
        if self.instance_norm:
            mu = Z.mean(dim=-1, keepdim=True)
            sd = Z.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            Zn = (Z - mu) / sd
        else:
            sd = torch.ones_like(Z[:, :, -1:])
            Zn = Z
        tok = self.embed(Zn)                    # (B,S,d) -- one token per series
        h = self._encode(tok)                   # attention ACROSS SERIES
        delta = self.proj(h) * sd               # de-normalise the residual only
        return self.anchor_terms(Z) + delta


# ---------------------------------------------------------------------------
# 2) the twin -- tokens are TIME steps of one series (channel independent)
# ---------------------------------------------------------------------------
class TimeTokenCINet(_Anchor):
    """Same capacity, attention over the 12 time steps of each series alone."""

    def __init__(self, context: int = 12, d_model: int = 32, n_heads_attn: int = 4,
                 layers: int = 2, n_out: int = len(HEADS), dropout: float = 0.1,
                 ff_mult: int = 4, use_anchor: bool = False,
                 anchor_init: float = 2.0, instance_norm: bool = True):
        super().__init__(context=context, use_anchor=use_anchor, anchor_init=anchor_init)
        self.d_model = int(d_model)
        self.instance_norm = bool(instance_norm)
        self.embed = nn.Linear(1, self.d_model)          # scalar month -> d_model
        self.pos = nn.Parameter(torch.zeros(self.context, self.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads_attn,
            dim_feedforward=int(ff_mult) * self.d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             norm=nn.LayerNorm(self.d_model))
        self.proj = nn.Linear(self.d_model, n_out)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        B, S, L = Z.shape
        if self.instance_norm:
            mu = Z.mean(dim=-1, keepdim=True)
            sd = Z.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            Zn = (Z - mu) / sd
        else:
            sd = torch.ones_like(Z[:, :, -1:])
            Zn = Z
        # (B,S,L) -> (B*S, L): every series is an independent sequence
        flat = Zn.reshape(B * S, L)
        tok = self.embed(flat.unsqueeze(-1)) + self.pos.unsqueeze(0)   # (B*S,L,d)
        h = self.encoder(tok)                                         # time attention
        last = h[:, -1, :]                                            # (B*S,d)
        delta = self.proj(last).view(B, S, -1) * sd                   # (B,S,n_out)
        return self.anchor_terms(Z) + delta


# ---------------------------------------------------------------------------
# windows / training / prediction
# ---------------------------------------------------------------------------
def build_windows(Z: np.ndarray, origins, context: int):
    """-> (n, S, L) lookback windows and (n, S, n_heads) LEVEL targets y_z(t+h).

    Targets are LEVELS, not residuals, because the model emits `anchor + delta`
    and the loss is taken on that anchored forecast -- exactly the xLSTM loop's
    convention, so a learnable anchor receives gradients instead of being a
    constant offset the loss cannot see.
    """
    o = np.asarray(origins)
    # Z[i-c+1:i+1, :] is (L, S); the model wants (S, L) per sample
    W = np.stack([Z[i - context + 1:i + 1, :].T for i in origins])     # (n,S,L)
    Yt = np.stack([Z[o + h, :] for h in HEADS], axis=-1)               # (n,S,H)
    return torch.tensor(W, dtype=torch.float32), torch.tensor(Yt, dtype=torch.float32)


def train_transformer(model: nn.Module, Z: np.ndarray, origins_tr, origins_va,
                      context: int = 12, epochs: int = 100, lr: float = 5e-4,
                      weight_decay: float = 5e-3, batch: int = 32,
                      device: str | None = None, verbose: bool = False):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    Wtr, Ytr = build_windows(Z, origins_tr, context)
    Wva, Yva = build_windows(Z, origins_va, context)
    n = len(Wtr)
    if n == 0:
        raise ValueError("no training origins")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Wva_t, Yva_t = Wva.to(dev), Yva.to(dev)
    best, best_ep, best_state = float("inf"), -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            xb, yb = Wtr[idx].to(dev), Ytr[idx].to(dev)
            opt.zero_grad()
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = float(((model(Wva_t) - Yva_t) ** 2).mean())
        if vloss < best:
            best, best_ep = vloss, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if verbose and (ep == 1 or ep % 25 == 0 or ep == epochs):
            print(f"    ep {ep:3d}  train {tot / n:.4f}  val {vloss:.4f}"
                  f"{'  *' if ep == best_ep else ''}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    meta = {"best_val_loss": float(best), "best_epoch": int(best_ep),
            "n_train_origins": int(n), "n_val_origins": int(len(Wva)),
            "n_series_per_sample": int(Z.shape[1]),
            "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "epochs": int(epochs), "batch": int(batch), "lr": float(lr),
            "weight_decay": float(weight_decay)}
    if hasattr(model, "anchor_weights"):
        meta["anchor_weights"] = {str(h): float(w) for h, w in zip(HEADS, model.anchor_weights())}
    return model, meta


@torch.no_grad()
def predict_transformer(model, Z: np.ndarray, origin_idx: int, context: int = 12):
    """(n_series, n_heads) forecast levels in y_z units at one period index"""
    dev = next(model.parameters()).device
    W = torch.tensor(Z[origin_idx - context + 1:origin_idx + 1, :].T[None, :, :],
                     dtype=torch.float32, device=dev)          # (1, S, L)
    return model(W)[0].cpu().numpy()


# ---------------------------------------------------------------------------
# selftest -- CPU, no data, ~3 s
# ---------------------------------------------------------------------------
def selftest(verbose: bool = True) -> dict:
    torch.manual_seed(0)
    S, L, B = 23, 12, 4
    Z = torch.randn(B, S, L)

    it = ITransformerNet().eval()
    ci = TimeTokenCINet().eval()

    # P1: zero the readout -> output must equal the anchor exactly
    with torch.no_grad():
        for m in (it, ci):
            nn.init.zeros_(m.proj.weight)
            nn.init.zeros_(m.proj.bias)
        anchor = it.anchor_terms(Z)
        p1_it = torch.allclose(it(Z), anchor, atol=1e-6)
        p1_ci = torch.allclose(ci(Z), anchor, atol=1e-6) and torch.equal(ci(Z), anchor)

    # randomise everything, then perturb ONE series only
    torch.manual_seed(1)
    it = ITransformerNet().eval()
    torch.manual_seed(1)
    ci = TimeTokenCINet().eval()
    torch.manual_seed(2)
    for p in it.parameters():
        p.data.normal_(0.0, 0.3)
    torch.manual_seed(2)
    for p in ci.parameters():
        p.data.normal_(0.0, 0.3)

    # NOTE on the perturbation: adding a CONSTANT to a whole window is invisible
    # to these models, because per-window instance normalisation removes it
    # (mu moves with the shift, sd does not).  A shape-changing perturbation is
    # required, otherwise this test measures nothing but float noise.
    Z2 = Z.clone()
    Z2[:, 0, -3:] = Z2[:, 0, -3:] + 5.0        # perturb series 0's shape only
    with torch.no_grad():
        a_it, b_it = it(Z), it(Z2)
        a_ci, b_ci = ci(Z), ci(Z2)
    others = list(range(1, S))
    d_it = float((a_it[:, others] - b_it[:, others]).abs().max())
    d_ci = float((a_ci[:, others] - b_ci[:, others]).abs().max())
    d_self_it = float((a_it[:, 0] - b_it[:, 0]).abs().max())
    d_self_ci = float((a_ci[:, 0] - b_ci[:, 0]).abs().max())

    n_it = sum(p.numel() for p in ITransformerNet().parameters() if p.requires_grad)
    n_ci = sum(p.numel() for p in TimeTokenCINet().parameters() if p.requires_grad)

    res = {
        "P1_iTransformer_init_is_persistence": p1_it,
        "P1_TimeTokenCI_init_is_persistence": p1_ci,
        "S1_iTransformer_other_series_max_abs_change": d_it,
        "S1_iTransformer_mixes_channels": d_it > 0.0,
        "S1_iTransformer_own_series_max_abs_change": d_self_it,
        "S2_TimeTokenCI_other_series_max_abs_change": d_ci,
        "S2_TimeTokenCI_is_channel_independent": d_ci == 0.0,
        "S2_TimeTokenCI_own_series_max_abs_change": d_self_ci,
        "S2_TimeTokenCI_own_series_reacts": d_self_ci > 0.0,
        "P3_params_iTransformer": n_it,
        "P3_params_TimeTokenCI": n_ci,
        "P3_capacity_gap_pct": round(100.0 * abs(n_it - n_ci) / n_it, 2),
        "PASS": bool(p1_it and p1_ci and d_it > 0.0 and d_ci == 0.0 and d_self_ci > 0.0),
    }
    if verbose:
        print("itransformer selftest")
        for k, v in res.items():
            print(f"  {k:44s} {v}")
        print(f"\n  {'PASS' if res['PASS'] else 'FAIL'}")
    return res


def selftest_noattn(verbose: bool = True) -> dict:
    """The sharpest control for ablation #9.

    Same class, same token embedding, same feed-forward, same readout, SAME
    parameter count -- the self-attention sublayer is bypassed.  If perturbing
    series 0 still moves series 1, the flag is broken and the ablation means
    nothing.
    """
    torch.manual_seed(0)
    S, L, B = 23, 12, 4
    Z = torch.randn(B, S, L)
    Z2 = Z.clone()
    Z2[:, 0, -3:] = Z2[:, 0, -3:] + 5.0
    m = ITransformerNet(cross_attention=False).eval()
    for p in m.parameters():
        p.data.normal_(0.0, 0.3)
    with torch.no_grad():
        a, b = m(Z), m(Z2)
    others = list(range(1, S))
    d_other = float((a[:, others] - b[:, others]).abs().max())
    d_self = float((a[:, 0] - b[:, 0]).abs().max())
    n_off = sum(p.numel() for p in ITransformerNet(cross_attention=False).parameters()
                if p.requires_grad)
    n_on = sum(p.numel() for p in ITransformerNet(cross_attention=True).parameters()
               if p.requires_grad)
    res = {"noattn_other_series_max_abs_change": d_other,
           "noattn_is_channel_independent": d_other == 0.0,
           "noattn_own_series_max_abs_change": d_self,
           "noattn_own_series_reacts": d_self > 0.0,
           "params_noattn": n_off, "params_full": n_on,
           "params_identical": bool(n_off == n_on),
           "PASS": bool(d_other == 0.0 and d_self > 0.0 and n_off == n_on)}
    if verbose:
        print("itransformer no-attention ablation selftest")
        for k, v in res.items():
            print(f"  {k:40s} {v}")
        print(f"\n  {'PASS' if res['PASS'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    ok = selftest()["PASS"]
    print()
    ok = selftest_noattn()["PASS"] and ok
    raise SystemExit(0 if ok else 1)
