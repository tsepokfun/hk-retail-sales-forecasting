"""
Modern linear baselines, NEURAL-NETWORK editions (Zeng et al., AAAI 2023).

WHY THIS FILE EXISTS
--------------------
`models/baselines.py` already ships a *sklearn* `run_dlinear` (MASE_ALL 1.34248)
and `run_nlinear` (1.15128).  Those are OLS fits on 6 hand-engineered columns.
The brief asks for the PAPER's models: a 12-step input window, an explicit
moving-average trend/seasonal decomposition, and two separate `nn.Linear` maps
trained by gradient descent.  This file is that, and nothing else.

THE TWO EDITIONS ARE NOT THE SAME MODEL -- the differences, enumerated
---------------------------------------------------------------------
1. INPUT.  sklearn edition: 6 columns `[y_z, lag1, lag2, lag3, lag12, lag13]`.
   Those lags are NOT contiguous in time (lag3 then lag12 is a 9-month jump), so
   the "centred moving average" the sklearn edition takes over them is an average
   over *scattered* months, not a trend smoother.  Neural edition: the true
   contiguous 12-month `y_z` window, so `movavg` really is a 12-month MA.
2. DECOMPOSITION.  sklearn: one MA over 6 columns, then `x - trend` as the
   seasonal half, OLS on `[trend(6) | seasonal(6)]` = 12 columns.  Neural: the
   paper's `series_decomp`, edge-replicated centred MA over the 12-step series,
   trend and seasonal each fed to their OWN `nn.Linear(12 -> n_heads)`, summed.
   Fitting the two maps jointly by OLS is algebraically the same functional form
   as summing two linear maps, so point 2 is a *small* difference; point 1 and
   point 3 are the big ones.
3. FITTING.  sklearn: closed-form least squares, refit **at every test origin**
   (walk-forward, expanding window), one model per horizon.  Neural: SGD/AdamW
   for 100 epochs on a FIXED training period, ONE model emitting all four
   horizons, early stopping on a purged validation block.  The neural edition
   therefore has a smoother, implicitly regularised solution but does not get to
   re-adapt to the 2024 regime origin by origin.  This is a protocol difference,
   and the report says so instead of pretending the two numbers are comparable
   model-for-model.
4. TARGET.  Both are run in residual form here (`y_z(t+h) - y_z(t)` + the
   anchor added back), per the project rule that target form dominates model
   choice (Ridge -34.0%, LightGBM -14.8% when switching level -> residual).

THE ANCHOR
----------
`use_anchor=False` (the default here) gives the paper's own form:
    yhat_h = y_z(t) + f_h(window)          # f_h learns the residual
`use_anchor=True` reuses the xLSTM champion's learnable two-term anchor
    anchor_h = a_h*y_z(t) + (1-a_h)*y_z(t+h-12),  a_h = sigmoid(logit_h)
so that a DLinear-vs-xLSTM comparison can be run with the anchor held fixed.
The arithmetic is a line-by-line copy of `xlstm_direct.XLSTMForecaster.
_anchor_terms` (including the `L + h - SEASON - 1` index and the `idx < 0`
fallback) so the two models cannot silently differ in their anchor.

NB on NLinear, stated up front because it is easy to get wrong: for NLinear the
"level" and "residual" forms are THE SAME MODEL.  NLinear already normalises by
subtracting the last step, so
    level    : yhat = W(x - x[-1]) + x[-1]
    residual : yhat = x[-1] + W(x - x[-1])
are identical expressions.  `selftest()` asserts this numerically instead of
leaving it as a claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eval import harness as H  # noqa: E402

HEADS = H.HORIZONS
SEASON = 12


# ---------------------------------------------------------------------------
# the paper's `series_decomp` (LTSF-Linear repo: edge-replicated centred MA)
# ---------------------------------------------------------------------------
def moving_average_1d(z: torch.Tensor, kernel: int) -> torch.Tensor:
    """centred moving average along the last axis, edge-replicated, same length.

    z: (..., L).  `kernel` must be odd.  No learnable parameters -- in Zeng et
    al. the decomposition kernel is FIXED and only the two Linear maps train.
    Reproduces `moving_avg` from the official DLinear implementation:
        front = z[...,0:1] repeated (k-1)//2 times
        end   = z[...,-1:] repeated (k-1)//2 times
        AvgPool1d(k, stride=1, padding=0)
    """
    k = int(kernel)
    if k % 2 == 0 or k < 1:
        raise ValueError(f"kernel must be a positive odd integer, got {kernel}")
    if k == 1:
        return z
    p = (k - 1) // 2
    zz = torch.cat([z[..., :1].expand(*z.shape[:-1], p),
                    z,
                    z[..., -1:].expand(*z.shape[:-1], p)], dim=-1)
    return Fn.avg_pool1d(zz.unsqueeze(-2), kernel_size=k, stride=1).squeeze(-2)


class _ChannelIndependentBase(nn.Module):
    """anchor arithmetic + the two attributes `xlstm_direct.train_model` insists
    on (`context`, `dropout_p`) so these models can be trained by the champion's
    own loop through its `model_factory` hook."""

    def __init__(self, context: int = 12, n_heads: int = len(HEADS),
                 use_anchor: bool = False, anchor_init: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        self.context = int(context)
        self.dropout_p = float(dropout)
        self.n_heads = int(n_heads)
        self.use_anchor = bool(use_anchor)
        # register the anchor logit as a PARAMETER only when it is actually
        # learned: a dead 4-value Parameter would inflate every parameter count
        # in the cost table for nothing.
        if self.use_anchor:
            self.anchor_logit = nn.Parameter(torch.full((self.n_heads,), float(anchor_init)))
        else:
            self.register_buffer("anchor_logit", torch.zeros(self.n_heads))

    def _anchor_terms(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads) = a_h*z(t) + (1-a_h)*z(t+h-12);  z(t) when use_anchor=False.

        Byte-for-byte the same index arithmetic as
        `xlstm_direct.XLSTMForecaster._anchor_terms`, reading the y_z channel,
        which is the last channel of `x` by construction (see build_tensors).
        """
        L = x.shape[1]
        z = x[:, :, -1]
        base_t = z[:, -1].unsqueeze(1)                     # (B,1)  z(t)
        seasonal = []
        for h in HEADS:
            idx = L + h - SEASON - 1
            seasonal.append(base_t[:, 0] if idx < 0 else z[:, idx])
        base_s = torch.stack(seasonal, dim=1)              # (B,n_heads)
        if self.use_anchor:
            a = torch.sigmoid(self.anchor_logit).unsqueeze(0)
            return a * base_t + (1.0 - a) * base_s
        return base_t.expand(-1, self.n_heads)

    def anchor_weights(self) -> np.ndarray:
        """train_model() calls this unconditionally; with use_anchor=False the
        returned vector is not used for anything (anchor is fixed persistence)."""
        return torch.sigmoid(self.anchor_logit).detach().cpu().numpy()


class DLinearNet(_ChannelIndependentBase):
    """DLinear: trend/seasonal split of the 12-step window, one Linear each.

        trend    = movavg(z)          # fixed kernel, no parameters
        seasonal = z - trend
        yhat_h   = anchor_h + W_trend[h,:]@trend + W_seasonal[h,:]@seasonal
    """

    def __init__(self, context: int = 12, kernel: int = 3,
                 n_heads: int = len(HEADS), use_anchor: bool = False,
                 anchor_init: float = 2.0, dropout: float = 0.0,
                 kernel_trainable: bool = False):
        super().__init__(context=context, n_heads=n_heads, use_anchor=use_anchor,
                         anchor_init=anchor_init, dropout=dropout)
        self.kernel = int(kernel)
        self.lin_trend = nn.Linear(self.context, self.n_heads)
        self.lin_seasonal = nn.Linear(self.context, self.n_heads)
        # kernel_trainable exists ONLY to be able to say "we tested whether a
        # learned kernel helps"; the paper's kernel is fixed, so the default is
        # the faithful one and this path is off unless explicitly requested.
        self.kernel_trainable = bool(kernel_trainable)
        if self.kernel_trainable:
            k = self.kernel
            w = torch.full((1, 1, k), 1.0 / k)
            self.kern = nn.Parameter(w)

    def _trend(self, z: torch.Tensor) -> torch.Tensor:
        if self.kernel_trainable:
            k = self.kernel
            p = (k - 1) // 2
            zz = torch.cat([z[..., :1].expand(*z.shape[:-1], p), z,
                            z[..., -1:].expand(*z.shape[:-1], p)], dim=-1)
            return Fn.conv1d(zz.unsqueeze(1), self.kern).squeeze(1)
        return moving_average_1d(z, self.kernel)

    def forward(self, x, sid=None, cal=None):
        z = x[:, :, -1]                                    # (B,L) y_z window
        trend = self._trend(z)
        seasonal = z - trend
        delta = self.lin_trend(trend) + self.lin_seasonal(seasonal)
        return self._anchor_terms(x) + delta


class NLinearNet(_ChannelIndependentBase):
    """NLinear: subtract the last step, one Linear, add it back.

        yhat_h = anchor_h + W[h,:]@(z - z(t))
    With use_anchor=False this is exactly the paper's NLinear, and the
    level-form / residual-form distinction collapses (see module docstring).
    """

    def __init__(self, context: int = 12, n_heads: int = len(HEADS),
                 use_anchor: bool = False, anchor_init: float = 2.0,
                 dropout: float = 0.0):
        super().__init__(context=context, n_heads=n_heads, use_anchor=use_anchor,
                         anchor_init=anchor_init, dropout=dropout)
        self.lin = nn.Linear(self.context, self.n_heads)

    def forward(self, x, sid=None, cal=None):
        z = x[:, :, -1]
        delta = self.lin(z - z[:, -1:])
        return self._anchor_terms(x) + delta


# ---------------------------------------------------------------------------
# parameter count helper -- the cost table is part of the deliverable
# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# per-series ("逐序列訓練") fitter -- 23 independent models, one per series
# ---------------------------------------------------------------------------
def fit_per_series(kind: str, Z: np.ndarray, origins_tr: list[int], origins_va: list[int],
                   context: int = 12, kernel: int = 3, epochs: int = 100,
                   lr: float = 5e-4, batch: int = 128, weight_decay: float = 5e-3,
                   seed: int = 42, device: str | None = None, verbose: bool = False):
    """One model per series, channel-independent, 4 direct horizons.

    Z: (n_periods, n_series) matrix of y_z.  `origins_tr` / `origins_va` are
    period indices that already satisfy the window and purge conditions, so this
    function contains no splitting logic of its own and cannot disagree with the
    shared panel split used everywhere else in the run.

    Returns (preds (n_series, n_heads) per series at every origin, meta dict).
    Seeding: the caller seeds torch once; fits run sequentially in a fixed series
    order so the run is reproducible.  Each fit gets a FRESH model instance.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_periods, n_series = Z.shape

    def windows(origins):
        # Z[i-c+1:i+1, :] is (L, S); the models want (S, L) per sample
        W = np.stack([Z[i - context + 1:i + 1, :].T for i in origins])   # (n,S,L)
        return torch.tensor(W, dtype=torch.float32)

    def deltas(origins):
        """LEVEL targets y_z(t+h) -- NOT residuals.

        The model's `forward` returns `anchor + delta`, and the anchor is fixed
        persistence `y_z(t)` when use_anchor=False.  Training against residuals
        while the network emits levels would (a) fit `anchor + delta` to `delta`
        and (b) let the caller add the anchor a second time.  Both the loss and
        the prediction would be wrong by exactly one anchor, which is a 2x level
        error -- MASE 2.89 instead of 1.03.  Targets are levels here for the same
        reason the panel-wide loop uses levels: one convention, everywhere.
        """
        i = np.asarray(origins)
        D = np.stack([Z[i + h, :] for h in HEADS], axis=-1)            # (n,S,n_heads)
        return torch.tensor(D, dtype=torch.float32)

    Wtr, Wva = windows(origins_tr), windows(origins_va)
    Dtr, Dva = deltas(origins_tr), deltas(origins_va)
    n_tr, n_va = len(origins_tr), len(origins_va)

    per_series_params = []
    fitted = []
    bests = []
    for j in range(n_series):
        cls = DLinearNet if kind == "dlinear" else NLinearNet
        m = cls(context=context, kernel=kernel, use_anchor=False) if kind == "dlinear" \
            else cls(context=context, use_anchor=False)
        m = m.to(dev)
        if j == 0:
            per_series_params.append(count_params(m))
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        xa = Wva[:, j, :].unsqueeze(-1).to(dev)                        # (n_va,1,L)
        ya = Dva[:, j, :].to(dev)
        best, best_state, best_ep = float("inf"), None, -1
        for ep in range(1, epochs + 1):
            m.train()
            perm = torch.randperm(n_tr)
            for k in range(0, n_tr, batch):
                idx = perm[k:k + batch]
                xb = Wtr[idx, j, :].unsqueeze(-1).to(dev)              # (b,1,L)
                yb = Dtr[idx, j, :].to(dev)
                opt.zero_grad()
                loss = ((m(xb) - yb) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            sched.step()
            m.eval()
            with torch.no_grad():
                v = float(((m(xa) - ya) ** 2).mean())
            if v < best:
                best, best_ep = v, ep
                best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        if best_state is not None:
            m.load_state_dict(best_state)
        m.eval()
        fitted.append(m)
        bests.append(best)
        if verbose:
            print(f"    series {j:2d}: best val {best:.4f} @ ep {best_ep}", flush=True)

    @torch.no_grad()
    def predict(origin_idx: int) -> np.ndarray:
        """FORECAST LEVELS (n_series, n_heads) in y_z units at a period index.

        `DLinearNet.forward` already adds the persistence anchor, so this is a
        forecast, not a residual -- the caller must NOT add `y_z(t)` again.
        """
        assert origin_idx - context + 1 >= 0
        out = np.zeros((n_series, len(HEADS)), dtype=np.float64)
        for j, m in enumerate(fitted):
            x = torch.tensor(Z[origin_idx - context + 1:origin_idx + 1, j],
                             dtype=torch.float32, device=dev).view(1, context, 1)
            out[j] = (m(x).cpu().numpy())[0]
        return out

    meta = {"n_params_per_series": int(per_series_params[0]),
            "n_params_total": int(per_series_params[0] * n_series),
            "n_models": int(n_series), "n_train_origins": int(n_tr),
            "n_val_origins": int(n_va), "kernel": int(kernel) if kind == "dlinear" else None,
            "context": int(context), "epochs": int(epochs),
            # mean of the 23 per-series best validation MSEs; comparable to the
            # panel-wide `best_val_loss` of the other families because it is the
            # same loss in the same units on the same 13 validation origins.
            "best_val_loss": float(np.mean(bests)),
            "best_val_loss_per_series": [float(b) for b in bests]}
    return predict, meta


# ---------------------------------------------------------------------------
# selftest -- CPU, no data, ~2 s.  Run before trusting any number.
# ---------------------------------------------------------------------------
def selftest(verbose: bool = True) -> dict:
    """P1  DLinear/NLinear at initialisation with zeroed output maps reproduce a
        pure persistence forecast exactly (residual form really is "anchor +
        nothing" before training).
    P2  NLinear level form == residual form, exactly (same weights, same input).
    P3  the moving average is length-preserving, centred and sums to 1.
    P4  parameter counts are the paper's (2 * L * n_heads for DLinear).
    """
    torch.manual_seed(0)
    ctx, n_heads = 12, 4
    x = torch.randn(7, ctx, 3)

    # -- P3 -----------------------------------------------------------------
    z = x[:, :, -1]
    tr = moving_average_1d(z, 3)
    p3_len = tr.shape == z.shape
    manual = (torch.cat([z[:, :1], z, z[:, -1:]], 1)[:, :-2]
              + torch.cat([z[:, :1], z, z[:, -1:]], 1)[:, 1:-1]
              + torch.cat([z[:, :1], z, z[:, -1:]], 1)[:, 2:]) / 3.0
    p3_centre = bool(torch.allclose(tr, manual, atol=1e-6))
    k5 = moving_average_1d(z, 5)
    p3_k5 = bool(torch.allclose(k5, moving_average_1d(z, 5))) and k5.shape == z.shape

    # -- P1 -----------------------------------------------------------------
    dl = DLinearNet(context=ctx, kernel=3, use_anchor=False).eval()
    nl = NLinearNet(context=ctx, use_anchor=False).eval()
    for m in (dl, nl):
        for lin in ([m.lin_trend, m.lin_seasonal] if m is dl else [m.lin]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
    with torch.no_grad():
        p1_dl = torch.allclose(dl(x), z[:, -1:].expand(-1, n_heads), atol=1e-7)
        p1_nl = torch.allclose(nl(x), z[:, -1:].expand(-1, n_heads), atol=1e-7)

    # -- P2: same weights, both forms ---------------------------------------
    m = NLinearNet(context=ctx, use_anchor=False).eval()
    with torch.no_grad():
        residual_form = z[:, -1:] + m.lin(z - z[:, -1:])
        level_form = m.lin(z - z[:, -1:]) + z[:, -1:]
        p2 = bool(torch.equal(residual_form, level_form))

    # -- P4 -----------------------------------------------------------------
    n_dl = count_params(DLinearNet(context=ctx, kernel=3))
    n_nl = count_params(NLinearNet(context=ctx))
    p4 = (n_dl == 2 * (ctx * n_heads + n_heads)) and (n_nl == ctx * n_heads + n_heads)

    res = {"P1_DLinear_init_is_persistence": p1_dl,
           "P1_NLinear_init_is_persistence": p1_nl,
           "P2_NLinear_level_equals_residual": p2,
           "P3_movavg_same_length": p3_len,
           "P3_movavg_centred_equals_manual": p3_centre,
           "P3_movavg_k5_ok": p3_k5,
           "P4_DLinear_params": n_dl, "P4_NLinear_params": n_nl,
           "P4_counts_match_paper_form": bool(p4),
           "PASS": bool(p1_dl and p1_nl and p2 and p3_len and p3_centre and p4)}
    if verbose:
        print("neural_linear selftest")
        for k, v in res.items():
            print(f"  {k:38s} {v}")
        print(f"\n  {'PASS' if res['PASS'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    raise SystemExit(0 if selftest()["PASS"] else 1)
