"""
Stage 1 — Series-Conditioned FiLM gating of the exogenous block.

DESIGN (matches 01_ARCHITECTURE.md §4.1 / 決策 1)

    h_in  = Proj_endo(x_endo)  +  scale * ( gate_i  ⊙  Proj_exog(x_exog) )
    gate_i = tanh(W_g @ series_emb_i)          shape (n_series, embed)

`gate_i` is a per-series, per-embedding-dimension multiplier learned from the
series identity alone, so the *same* weight set can let series 12 (discretionary
apparel) lean on visitor arrivals while series 40 (supermarket) ignores it.  This
is the only way to give branch-specific exogenous variables to a
channel-independent model without un-sharing the weights.

WHY `Proj_exog` IS ZERO-INITIALISED -- AND WHY A LEARNABLE `scale` WOULD BE A TRAP
----------------------------------------------------------------------------------
`Proj_exog` is zero-initialised, so at step 0 the exogenous branch contributes
exactly nothing and the network is *identical* to the no-exogenous champion.
Training then decides how much of the exogenous block to let in.

The obvious companion -- a learnable scalar `exog_scale` initialised to 0.0 --
is a FATAL design error here, and this module does NOT do it:

    h = inp_endo(num) + exog_scale * (g * inp_exog(exo))

    d loss / d exog_scale  is proportional to  (g * inp_exog(exo))  = 0  at init
    d loss / d inp_exog    is proportional to  exog_scale           = 0  at init

Both gradients are EXACTLY zero, so the whole exogenous branch is permanently
dead -- it can never leave zero.  This is the same degenerate-equilibrium trap
that makes `a_12` unlearnable in the anchor (see 01_ARCHITECTURE.md Q13).

The fix is to make the scale a FIXED constant 1.0 and zero-init only
`inp_exog`.  Then

    h(step 0) = inp_endo(num) + 1.0 * (g * 0) = inp_endo(num)   -> baseline exactly
    d loss / d inp_exog  is proportional to  1.0 * g * exo      -> non-zero

so the branch starts switched off but can switch itself on.  There is a unit
test for this (`selftest`) that fails loudly if the gradient comes back as 0.

WHAT IS *NOT* CHANGED
---------------------
`runtime` reuses `xlstm_direct.train_model` verbatim through its `model_factory`
hook, so the loss, the optimiser, the schedule, the anchor parameterisation and
the early-stopping rule are byte-identical to the Stage-0 baseline.  The only
difference is the input projection.  That is deliberate: an ablation that also
changes the training loop proves nothing.

Usage
    from models import xlstm_exog as XE
    df, meta = XE.run_exog(feat, wide, scaling, exog, groups["all"],
                           tag="xLSTM_exog_all_s42", epochs=100, seed=42)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import harness as H          # noqa: E402
from models import xlstm_direct as X   # noqa: E402


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class XLSTMForecasterFiLM(X.XLSTMForecaster):
    """XLSTMForecaster whose input projection is split into endo + gated exog."""

    def __init__(self, n_series, n_total_feat, n_numeric, n_exog, **kw):
        super().__init__(n_series, n_total_feat, **kw)
        assert n_numeric + n_exog == n_total_feat, (
            f"n_numeric({n_numeric}) + n_exog({n_exog}) != n_total_feat({n_total_feat})")
        self.n_numeric = int(n_numeric)
        self.n_exog = int(n_exog)
        embed = self.inp[0].out_features
        series_emb = self.ser.embedding_dim
        self.inp_endo = nn.Linear(self.n_numeric, embed)
        self.inp_exog = nn.Linear(self.n_exog, embed)
        self.gate = nn.Linear(series_emb, embed)
        # Rebuild the tail of the baseline's `inp` chain (LayerNorm -> GELU) so
        # that the FiLM variant is EXACTLY "baseline input path, except the
        # projection is split in two and the exogenous half is gated".
        # Dropping the LayerNorm would silently turn the ablation into a
        # two-variable comparison (split + missing normalisation), and the
        # parameter delta would be 1,280 instead of the true 1,344.
        self.in_ln = nn.LayerNorm(embed)
        self.in_act = nn.GELU()
        # --- switch the exogenous branch off at step 0, but keep it trainable -
        # `inp_exog` zero -> contribution is exactly 0 -> step 0 == no-exog model.
        # The scale stays a FIXED 1.0 (see the module docstring): a learnable
        # zero-initialised scale would give d(loss)/d(inp_exog) == 0 forever.
        nn.init.zeros_(self.inp_exog.weight)
        nn.init.zeros_(self.inp_exog.bias)
        nn.init.uniform_(self.gate.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.gate.bias)
        self.register_buffer("exog_scale", torch.tensor(1.0))
        # the inherited `inp` is now unused; drop it so it cannot be trained by
        # accident, and normalise inp_endo so the two variants are comparable
        self.inp = nn.Identity()

    def _embed_input(self, x, sid):
        num = x[:, :, : self.n_numeric]
        exo = x[:, :, self.n_numeric: -1]
        g = torch.tanh(self.gate(self.ser(sid)))              # (B, embed)
        h = self.inp_endo(num) + self.exog_scale * (g.unsqueeze(1) * self.inp_exog(exo))
        h = self.in_act(self.in_ln(h))                        # baseline's LN -> GELU
        return h + self.ser_proj(self.ser(sid)).unsqueeze(1)

    def gate_values(self) -> np.ndarray:
        """(n_series, embed) tanh gate, detached -- for inspection"""
        with torch.no_grad():
            ids = torch.arange(self.ser.num_embeddings, device=self.ser.weight.device)
            return torch.tanh(self.gate(self.ser(ids))).cpu().numpy()


# ---------------------------------------------------------------------------
# tensors
# ---------------------------------------------------------------------------
def build_tensors_exog(feat: pd.DataFrame, exog: pd.DataFrame,
                       exog_cols: list[str], context: int | None = None,
                       standardize: bool = True, train_end: str = "202312",
                       verbose: bool = True):
    """(periods, series, F, Y, KF, scaler) with the exogenous channels inserted
    immediately before the anchor channel.

    Layout of the last axis:  [0 .. n_numeric-1] numeric  |  [n_numeric .. -2]
    exog  |  [-1] y_z(t)   -- the anchor channel must stay last, because
    XLSTMForecaster.forward reads x[:, :, -1] for the persistence/seasonal anchor.

    WHY standardize=True IS THE DEFAULT
    -----------------------------------
    The endogenous block is already z-scored per series, but the raw exogenous
    columns are on wildly different scales:

        cpi_yoy    ~ 0.008      rate_z      ~ +/-3
        eeri_yoy   ~ 0.02       visitor_yoy ~ +/-7.2   (COVID collapse)
        mtr_yoy    ~ 0.7        rain_anom_z ~ +/-5.7

    i.e. a 1000x spread, all fed to ONE `nn.Linear` shared with the endogenous
    block.  The first un-standardised ablation showed exactly the fingerprint of
    that mistake: `all` reached MASE 1.88 on seed 42 while `none` reached 1.02,
    and the bias flipped sign (+18% vs -5%).  Standardising per column with
    TRAIN-ONLY statistics removes that artefact, and the scaler is computed on
    `period <= train_end` so no test-period information enters the scale.

    Set `standardize=False` to reproduce the flawed first pass.

    NaN POLICY
    ----------
    `make_samples` drops any origin whose context window contains a NaN in ANY
    channel, so a single NaN month in the exogenous block silently deletes up to
    `context` training origins.  That is unacceptable to discover after the fact,
    so instead of blindly filling or blindly raising, this function *measures*
    whether the exogenous block costs any origin that the numeric block did not
    already cost, and refuses to build if it does.

    `visitor_yoy` is NaN for 2004-10..2005-01 because the visitor series starts in
    2004-01 and a shifted YoY needs t-13.  Those four months sit inside windows
    that the lag warm-up has already discarded, so the measured cost is zero --
    but the check proves it rather than assuming it.
    """
    context = X.CONTEXT if context is None else context
    periods, series, F, Y, KF = X.build_tensors(feat)
    n_numeric = F.shape[2] - 1

    ex = exog.copy()
    ex.index = ex.index.astype(str)
    missing = [c for c in exog_cols if c not in ex.columns]
    if missing:
        raise KeyError(f"exog_wide.csv is missing {missing}")
    # exog rows are per PERIOD (not per series) -> broadcast over the series axis
    E = ex.loc[periods, exog_cols].to_numpy(dtype=np.float64)      # (P, d)

    scaler = None
    if standardize:
        train_mask = np.array([p <= train_end for p in periods])
        mu = np.nanmean(E[train_mask], axis=0)
        sd = np.nanstd(E[train_mask], axis=0, ddof=0)
        if not np.isfinite(sd).all() or (sd == 0).any():
            raise RuntimeError(f"degenerate exog scaler: sd={sd}")
        E = (E - mu) / sd
        scaler = {"train_end": train_end,
                  "n_train_months": int(train_mask.sum()),
                  "mu": {c: float(m) for c, m in zip(exog_cols, mu)},
                  "sd": {c: float(s) for c, s in zip(exog_cols, sd)}}

    E = E.astype(np.float32)
    E = np.repeat(E[:, None, :], len(series), axis=1)               # (P, S, d)

    F2 = np.concatenate([F[:, :, :n_numeric], E, F[:, :, n_numeric:]], axis=2)
    assert F2.shape[2] == n_numeric + len(exog_cols) + 1, F2.shape

    # ---- audit: the exog block must not cost an otherwise-usable origin ----
    def usable(FF):
        return {i for i in range(len(periods))
                if i - context + 1 >= 0
                and not np.isnan(FF[i - context + 1: i + 1]).any()}

    ok_num, ok_all = usable(F), usable(F2)
    lost = sorted(ok_num - ok_all)
    if lost:
        detail = [periods[i] for i in lost]
        raise ValueError(
            f"the exogenous block would drop {len(lost)} otherwise-usable "
            f"origin(s): {detail}  -- NaNs present in "
            f"{[c for c in exog_cols if ex.loc[periods, c].isna().any()]}")
    if verbose:
        nan_cells = int(np.isnan(E).any(axis=2).sum())
        tag = "z-scored (train-only)" if standardize else "RAW (not standardised)"
        print(f"  exog audit: {nan_cells} NaN cell(s) in the block, "
              f"0 additional origins lost (usable origins {len(ok_all)}); "
              f"scaling = {tag}")
        if standardize:
            print("    " + "  ".join(f"{c}: mu={scaler['mu'][c]:+.4f} "
                                     f"sd={scaler['sd'][c]:.4f}"
                                     for c in exog_cols))
    return periods, series, F2, Y, KF, scaler


# ---------------------------------------------------------------------------
# runner -- mirrors xlstm_direct.run but with the FiLM model + exog tensor
# ---------------------------------------------------------------------------
def run_exog(feat, wide, scaling, exog, exog_cols, tag="xLSTM_exog",
             epochs=100, origins=None, seed=None, verbose=True,
             purge_val=True, standardize=True, val_tail=None):
    origins = origins or H.ORIGINS
    if seed is not None:
        X.set_seed(seed)
    periods, series, F, Y, KF, ex_scaler = build_tensors_exog(
        feat, exog, exog_cols, standardize=standardize, verbose=verbose)
    n_numeric = len(H.FEATURE_COLS)
    n_exog = len(exog_cols)
    if verbose:
        print(f"  cube {F.shape} (numeric {n_numeric} + exog {n_exog} + anchor 1)")

    def factory(n_series, n_feat_total):
        return XLSTMForecasterFiLM(
            n_series, n_feat_total, n_numeric, n_exog,
            use_anchor=True, use_calendar=False)

    t0 = time.time()
    model, meta = X.train_model(F, Y, KF, periods, series, H.TRAIN_END,
                               epochs=epochs, verbose=verbose,
                               purge_val=purge_val, model_factory=factory,
                               seed=seed, val_tail=val_tail)
    meta["train_sec"] = round(time.time() - t0, 1)
    meta["tag"] = tag
    meta["exog_cols"] = list(exog_cols)
    meta["n_exog"] = n_exog
    meta["film"] = True
    meta["standardize_exog"] = bool(standardize)
    meta["exog_scaler"] = ex_scaler
    meta["exog_weight_norm"] = float(model.inp_exog.weight.detach().norm().cpu())

    rows = []
    for o in origins:
        pr = X.predict_at(model, F, KF, periods, series, o)
        if pr is None:
            continue
        for h in X.HEADS:
            tgt = H.month_add(o, h)
            if tgt not in wide.index:
                continue
            yhat = H.inverse_transform(pr[h], np.asarray(series), scaling)
            for s, yp in zip(series, yhat):
                rows.append((tag, o, h, s, float(wide.loc[tgt, s]), float(yp)))
    df = pd.DataFrame(rows, columns=["model", "origin", "h", "series", "y_true", "y_pred"])
    if verbose:
        print(f"  predictions {len(df):,} rows / {df['origin'].nunique()} origins")
    gv = model.gate_values()
    meta["gate_absmean_by_series"] = {
        s: float(np.abs(gv[i]).mean()) for i, s in enumerate(series)}
    return df, meta


# ---------------------------------------------------------------------------
# feature groups for the orthogonal ablation
# ---------------------------------------------------------------------------
GROUPS = {
    "none":        [],
    "macro":       ["cpi_yoy", "rate_z", "eeri_yoy"],
    "people":      ["visitor_yoy", "mtr_yoy", "rain_anom_z", "temp_anom_z"],
    "logistics":   ["cargo_yoy"],
    "macro_fin":   ["cpi_yoy", "rate_z", "eeri_yoy"],
    "people_weather_logistics": ["visitor_yoy", "mtr_yoy",
                                 "rain_anom_z", "temp_anom_z", "cargo_yoy"],
    "all":         ["cpi_yoy", "rate_z", "eeri_yoy", "visitor_yoy", "mtr_yoy",
                    "cargo_yoy", "rain_anom_z", "temp_anom_z"],
}
GROUPS["people_weather_logistics"] = GROUPS["people"] + GROUPS["logistics"]

ALL_COLS = GROUPS["all"]


def load_exog(path: Path | str | None = None) -> pd.DataFrame:
    p = Path(path) if path else ROOT / "data" / "exog_wide.csv"
    e = pd.read_csv(p, dtype={"period": str})
    return e.set_index("period")


# ---------------------------------------------------------------------------
# selftest -- CPU only, no data, ~2 seconds.  Run it before trusting any number.
# ---------------------------------------------------------------------------
def selftest(verbose: bool = True) -> dict:
    """Three properties that MUST hold, else the ablation is meaningless:

    P1  step 0 == the no-exog model.  Changing the exogenous inputs changes
        nothing while inp_exog is still zero, i.e. the branch starts off.
    P2  the branch is NOT dead.  d(loss)/d(inp_exog.weight) is non-zero after
        one backward, so training can switch it on.
    P3  parameter delta is small and known.  The FiLM variant must not silently
        buy its way to a better score with a much bigger network.
    """
    torch.manual_seed(0)
    n_series, n_num, n_ex, ctx = 5, 27, 8, 12
    m = XLSTMForecasterFiLM(n_series, n_num + n_ex, n_num, n_ex,
                            use_anchor=True, use_calendar=False)
    m.eval()

    x = torch.randn(16, ctx, n_num + n_ex + 1)
    sid = torch.randint(0, n_series, (16,))
    cal = torch.zeros(16, len(X.HEADS), X.KF_DIM)

    with torch.no_grad():
        out_a = m(x, sid, cal)
        x2 = x.clone()
        x2[:, :, n_num:n_num + n_ex] = torch.randn(16, ctx, n_ex) * 100.0
        out_b = m(x2, sid, cal)
    p1 = bool(torch.allclose(out_a, out_b, atol=0, rtol=0))
    max_p1 = float((out_a - out_b).abs().max())

    m.train()
    m.zero_grad()
    loss = (m(x, sid, cal) ** 2).mean()
    loss.backward()
    g = float(m.inp_exog.weight.grad.abs().sum())
    gg = float(m.gate.weight.grad.abs().sum())
    p2 = g > 0.0

    n_film = sum(p.numel() for p in m.parameters() if p.requires_grad)
    base = X.XLSTMForecaster(n_series, n_num, use_anchor=True, use_calendar=False)
    n_base = sum(p.numel() for p in base.parameters() if p.requires_grad)

    res = {
        "P1_epoch0_identical_to_no_exog": p1,
        "P1_max_abs_output_diff": max_p1,
        "P2_inp_exog_grad_abs_sum": g,
        "P2_gate_grad_abs_sum": gg,
        "P2_branch_trainable": p2,
        "P3_film_params": n_film,
        "P3_base_params": n_base,
        "P3_delta": n_film - n_base,
        "PASS": bool(p1 and p2),
    }
    if verbose:
        print("FiLM selftest")
        for k, v in res.items():
            print(f"  {k:34s} {v}")
        print(f"\n  {'PASS' if res['PASS'] else 'FAIL'}")
    return res


# ---------------------------------------------------------------------------
# Stage H -- HORIZON-SPECIFIC FiLM gating
# ---------------------------------------------------------------------------
# THE HYPOTHESIS
# --------------
# `XLSTMForecasterFiLM` injects the exogenous block at the BOTTOM, in the shared
# input projection:
#
#     h_in = Proj_endo(x_endo) + gate_i ⊙ Proj_exog(x_exog)      # one hidden state
#
# All four direct-multi-horizon heads then read the SAME final hidden state, so
# an exogenous channel that only carries information about h=12 (Stage-1 evidence:
# `visitor_yoy` / `mtr_yoy` beat a width-matched noise block at h=12 with
# p≈0.033–0.035, while the 5-column block collapses to MASE 2.10 at h=12) has no
# way to be "used only where it helps".  It perturbs the shared representation and
# therefore also the h=1 / h=3 / h=6 forecasts, where it is at best noise.
#
# THE IMPLEMENTATION CHOSEN, AND WHY THIS ONE
# -------------------------------------------
# The exogenous block is NOT in the shared input projection here.  The backbone
# is the `none` baseline's endogenous-only path, and each head k gets its own
# gated exogenous term added to the SHARED hidden state just before its own
# linear map:
#
#     h       = Backbone(Proj_endo(x_endo) + series_emb)          # == `none`
#     pool    = [ x_exog(t) , mean_{t-11..t} x_exog ]              # (2*n_exog,)
#     gate_k  = tanh(W_g series_emb_i)[k]  ⊙ mask_k                # (embed,)
#     delta_k = head_k( h + gate_k ⊙ Proj_exog_k(pool) )
#     yhat_k  = anchor_k + delta_k
#
# Three consequences, each of which is a reason for the choice:
#  1. `mask_k == 0` makes head k's output MATHEMATICALLY independent of the
#     exogenous inputs -- exactly 0.0 difference, not "small".  That is the only
#     way the required selftest (h=1/h=3 insensitivity) can be passed at all:
#     any bottom injection would leak through the shared hidden state and the
#     best achievable difference would be "small but non-zero".  Measured: see
#     `selftest_horizon()`.
#  2. It keeps ONE shared backbone and ONE shared hidden state, so the only thing
#     that differs between the three arms (`none`, `_full`, `_h12`) is where the
#     exogenous information is allowed to act.  Freezing a per-head gate would be
#     meaningless if each head also had its own backbone.
#  3. `gate_k` is still a function of the series embedding, so the per-series
#     conditioning of the original design survives -- series 12 (discretionary
#     apparel) and series 40 (supermarket) can weight the same exogenous channel
#     differently at the same horizon.
#
# WHY THE POOL IS [last, window-mean] AND NOT JUST THE WINDOW MEAN
# ---------------------------------------------------------------
# The brief suggests "Proj_exog_h(x_exog_pooled)".  A 12-month mean alone throws
# away the most recent regime, and the recent regime is exactly what a direct
# h=1/h=3 forecast needs; using ONLY x_exog(t) throws away the annual level that
# the bottom-FiLM arm can reconstruct from the full sequence.  Feeding both costs
# 64 extra parameters per head and removes a handicap that would have made the
# horizon-gated arm lose for a reason unrelated to the hypothesis.  Both terms are
# observed at the origin t -- no leakage.
#
# THE DEGENERATE-EQUILIBRIUM TRAP IS AVOIDED THE SAME WAY AS IN FiLM
# -----------------------------------------------------------------
# `Proj_exog_k` is zero-initialised while the gate stays a FIXED-shape,
# small-non-zero number (a learnable zero-initialised scale would freeze both
# gradient paths; see this module's main docstring).  So at step 0 every OPEN
# head is exactly the `none` model, and d(loss)/d(Proj_exog_k) ∝ gate_k ≠ 0 lets
# the branch switch itself on.  CLOSED heads additionally carry mask_k = 0, which
# zeroes both the contribution AND its gradient.
class XLSTMForecasterFiLMHorizon(X.XLSTMForecaster):
    """Per-head (horizon-specific) gated injection of the exogenous block."""

    def __init__(self, n_series, n_total_feat, n_numeric, n_exog,
                 open_heads=(1, 3, 6, 12), **kw):
        super().__init__(n_series, n_total_feat, **kw)
        if kw.get("use_calendar", False):
            raise NotImplementedError(
                "the horizon-gated variant does not implement the known-future "
                "calendar concat; every arm of this experiment uses "
                "use_calendar=False")
        assert n_numeric + n_exog == n_total_feat, (
            f"n_numeric({n_numeric}) + n_exog({n_exog}) != n_total_feat({n_total_feat})")
        self.n_numeric = int(n_numeric)
        self.n_exog = int(n_exog)
        n_h = len(X.HEADS)
        embed = int(self.inp[0].out_features)
        series_emb = int(self.ser.embedding_dim)
        self.embed = embed
        # Replace the (numeric + exog)-wide projection with the ENDO-ONLY
        # projection.  Without this the backbone would still see the exog block
        # and the closed-head invariance below would be false.
        self.inp = nn.Sequential(nn.Linear(self.n_numeric, embed),
                                 nn.LayerNorm(embed), nn.GELU())
        # (n_series, n_heads, embed) gate, produced as one Linear and viewed
        self.exog_gate = nn.Linear(series_emb, n_h * embed)
        self.exog_proj = nn.ModuleList(
            [nn.Linear(2 * self.n_exog, embed) for _ in range(n_h)])
        for p in self.exog_proj:
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)
        nn.init.uniform_(self.exog_gate.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.exog_gate.bias)
        # STRUCTURAL freeze: a 0/1 mask, not an initialisation and not
        # requires_grad=False.  AdamW's decoupled weight decay would keep
        # shrinking a "frozen" gate even with zero gradient, so relying on the
        # value would be wrong; 0 * anything *is* exactly 0.
        mask = torch.zeros(n_h, embed)
        for k, hh in enumerate(X.HEADS):
            if int(hh) in set(int(o) for o in open_heads):
                mask[k] = 1.0
        self.register_buffer("head_mask", mask.view(1, n_h, embed))
        self.open_heads = tuple(int(o) for o in open_heads)
        self.closed_heads = tuple(int(hh) for hh in X.HEADS
                                  if int(hh) not in self.open_heads)

    # -- the backbone sees the endogenous block only -------------------------
    def _embed_input(self, x, sid):
        h = self.inp(x[:, :, : self.n_numeric])
        return h + self.ser_proj(self.ser(sid)).unsqueeze(1)

    # -- per-head gated exogenous term ---------------------------------------
    def _head_delta(self, h, x, sid, cal):
        exo = x[:, :, self.n_numeric: -1]                          # (B, L, n_exog)
        pooled = torch.cat([exo[:, -1, :], exo.mean(dim=1)], dim=-1)   # (B, 2*n_exog)
        n_h = len(X.HEADS)
        g = torch.tanh(self.exog_gate(self.ser(sid))).view(-1, n_h, self.embed)
        g = g * self.head_mask                                     # (B, n_h, embed)
        outs = []
        for k in range(n_h):
            z = g[:, k, :] * self.exog_proj[k](pooled)              # (B, embed)
            outs.append(self.heads[k](h + z))
        return torch.cat(outs, dim=-1)

    def gate_values(self) -> np.ndarray:
        """(n_heads, n_series, embed) tanh gate AFTER the mask, detached."""
        with torch.no_grad():
            ids = torch.arange(self.ser.num_embeddings,
                               device=self.ser.weight.device)
            g = torch.tanh(self.exog_gate(self.ser(ids)))
            g = g.view(-1, len(X.HEADS), self.embed) * self.head_mask
            return g.permute(1, 0, 2).cpu().numpy()


def run_exog_horizon(feat, wide, scaling, exog, exog_cols, open_heads=(1, 3, 6, 12),
                     tag="xLSTM_exog_hz", epochs=100, origins=None, seed=None,
                     verbose=True, purge_val=True, standardize=True, val_tail=None,
                     context=None, embed=None, blocks=None, dropout=None,
                     weight_decay=5e-3):
    """`run_exog`, but with the horizon-gated model.  Same data, same loop."""
    origins = origins or H.ORIGINS
    ctx = X.CONTEXT if context is None else int(context)
    drp = X.DROPOUT if dropout is None else float(dropout)
    if seed is not None:
        X.set_seed(seed)
    periods, series, F, Y, KF, ex_scaler = build_tensors_exog(
        feat, exog, exog_cols, context=ctx, standardize=standardize, verbose=verbose)
    n_numeric = len(H.FEATURE_COLS)
    n_exog = len(exog_cols)
    if verbose:
        print(f"  cube {F.shape} (numeric {n_numeric} + exog {n_exog} + anchor 1)")
        print(f"  horizon gates: open={list(open_heads)}  "
              f"closed={[h for h in X.HEADS if h not in open_heads]}")

    def factory(n_series, n_feat_total):
        # `context` and `dropout` MUST be threaded: train_model asserts that the
        # model agrees with the window it cut (see its docstring).
        return XLSTMForecasterFiLMHorizon(
            n_series, n_feat_total, n_numeric, n_exog, open_heads=tuple(open_heads),
            context=ctx, embed=embed, blocks=blocks, dropout=drp,
            use_anchor=True, use_calendar=False)

    t0 = time.time()
    model, meta = X.train_model(F, Y, KF, periods, series, H.TRAIN_END,
                               epochs=epochs, verbose=verbose,
                               purge_val=purge_val, model_factory=factory,
                               seed=seed, val_tail=val_tail, context=ctx,
                               embed=embed, blocks=blocks, dropout=drp,
                               weight_decay=weight_decay)
    meta["train_sec"] = round(time.time() - t0, 1)
    meta["tag"] = tag
    meta["exog_cols"] = list(exog_cols)
    meta["n_exog"] = n_exog
    meta["film"] = "horizon"
    meta["open_heads"] = list(open_heads)
    meta["closed_heads"] = [h for h in X.HEADS if h not in open_heads]
    meta["standardize_exog"] = bool(standardize)
    meta["exog_scaler"] = ex_scaler
    meta["exog_proj_norm_by_head"] = {
        str(h): float(model.exog_proj[k].weight.detach().norm().cpu())
        for k, h in enumerate(X.HEADS)}
    gv = model.gate_values()                       # (n_heads, n_series, embed)
    meta["gate_absmean_by_head"] = {
        str(h): float(np.abs(gv[k]).mean()) for k, h in enumerate(X.HEADS)}

    rows = []
    for o in origins:
        pr = X.predict_at(model, F, KF, periods, series, o)
        if pr is None:
            continue
        for h in X.HEADS:
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


def selftest_horizon(verbose: bool = True) -> dict:
    """The premise of task C.  If this fails, the horizon-gating experiment is
    meaningless no matter what MASE comes out.

    H1  CLOSED heads are BIT-EXACTLY insensitive to the exogenous inputs.
        Two different exogenous tensors (one scaled x100) must give
        max|Δ| == 0.0 on the h=1/h=3 head outputs.  Asserted with torch.equal,
        i.e. exact, not allclose.
        Run AFTER randomising exog_gate AND exog_proj, so the invariance cannot
        be an artefact of the zero initialisation -- it must hold for a "trained"
        gate too, because the mask is structural.
    H2  OPEN heads DO respond: max|Δ| > 0 for h=6 and h=12 in exactly the same
        forward pass.  Without this the test could pass on a model whose exog
        branch is entirely dead.
    H3  at init (zero proj) the full-open model equals the `none` model exactly,
        and d(loss)/d(exog_proj) != 0 (the branch can wake up).
    H4  the parameter cost is reported (no silent capacity purchase).
    """
    torch.manual_seed(0)
    n_series, n_num, n_ex, ctx = 5, 27, 2, 12
    B = 16
    x = torch.randn(B, ctx, n_num + n_ex + 1)
    sid = torch.randint(0, n_series, (B,))
    cal = torch.zeros(B, len(X.HEADS), X.KF_DIM)

    def perturb(xx):
        y = xx.clone()
        y[:, :, n_num:n_num + n_ex] = torch.randn(B, ctx, n_ex) * 100.0
        return y

    # ---------------- H3 first, on a FRESH model (before randomisation) ------
    m0 = XLSTMForecasterFiLMHorizon(n_series, n_num + n_ex, n_num, n_ex,
                                    open_heads=(1, 3, 6, 12)).eval()
    with torch.no_grad():
        base_out = m0(x, sid, cal)
        # "step 0 == no-exog model" tested operationally: with the zero-init
        # projections, replacing the exogenous inputs wholesale must not move ANY
        # head -- including the open ones.
        p3_diff = float((base_out - m0(perturb(x), sid, cal)).abs().max())

    m0.train(); m0.zero_grad()
    loss = (m0(x, sid, cal) ** 2).mean()
    loss.backward()
    grads_open = {str(h): float(m0.exog_proj[k].weight.grad.abs().sum())
                  for k, h in enumerate(X.HEADS)}

    # ---------------- H1/H2 on a RANDOMISED (= "trained") model -------------
    m1 = XLSTMForecasterFiLMHorizon(n_series, n_num + n_ex, n_num, n_ex,
                                    open_heads=(6, 12)).eval()
    with torch.no_grad():
        m1.exog_gate.weight.normal_(0.0, 0.5)
        m1.exog_gate.bias.normal_(0.0, 0.5)
        for p in m1.exog_proj:
            p.weight.normal_(0.0, 0.5)
            p.bias.normal_(0.0, 0.5)
    with torch.no_grad():
        a = m1(x, sid, cal)
        b = m1(perturb(x), sid, cal)
    d = (a - b).abs().amax(dim=0)                    # (n_heads,)
    per_head = {str(h): float(d[k]) for k, h in enumerate(X.HEADS)}
    closed = [h for h in X.HEADS if h in (1, 3)]
    openh = [h for h in X.HEADS if h in (6, 12)]
    h1_exact_zero = all(float(d[k]) == 0.0 for k, h in enumerate(X.HEADS) if h in closed)
    h2_responds = all(float(d[k]) > 0.0 for k, h in enumerate(X.HEADS) if h in openh)
    # bitwise, not allclose
    h1_tensor_equal = all(torch.equal(a[:, k], b[:, k]) for k, h in enumerate(X.HEADS)
                          if h in closed)
    # gradients into the CLOSED heads' projections must be exactly zero
    m1.train(); m1.zero_grad()
    (m1(x, sid, cal) ** 2).mean().backward()
    closed_grad = {str(h): float(m1.exog_proj[k].weight.grad.abs().sum())
                   for k, h in enumerate(X.HEADS)}
    closed_grad_zero = all(closed_grad[str(h)] == 0.0 for h in closed)

    n_hz = sum(p.numel() for p in m0.parameters() if p.requires_grad)
    n_base = sum(p.numel() for p in X.XLSTMForecaster(n_series, n_num).parameters()
                 if p.requires_grad)

    res = {
        "H1_max_abs_diff_closed_heads": max(per_head[str(h)] for h in closed),
        "H1_bitwise_equal_closed_heads": h1_tensor_equal,
        "H1_closed_heads_exact_zero": h1_exact_zero,
        "H2_max_abs_diff_open_heads": {str(h): per_head[str(h)] for h in openh},
        "H2_open_heads_respond": h2_responds,
        "H3_step0_exog_perturbation_diff_all_open": p3_diff,
        "H3_step0_is_zero": p3_diff == 0.0,
        "H3_grad_abs_sum_by_head_at_init": grads_open,
        "H3_branch_trainable": all(v > 0.0 for v in grads_open.values()),
        "H4_horizon_params": n_hz,
        "H4_base_params": n_base,
        "H4_delta": n_hz - n_base,
        "H5_closed_head_grad_abs_sum": {k: v for k, v in closed_grad.items()
                                        if int(k) in closed},
        "H5_closed_head_grad_is_zero": closed_grad_zero,
        "mask_sum": float(m1.head_mask.sum()),
        "PASS": bool(h1_exact_zero and h1_tensor_equal and h2_responds and p3_diff == 0.0
                     and all(v > 0.0 for v in grads_open.values()) and closed_grad_zero),
    }
    if verbose:
        print("HORIZON-GATE selftest")
        for k, v in res.items():
            print(f"  {k:46s} {v}")
        print(f"\n  {'PASS' if res['PASS'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    r = selftest()
    print()
    rh = selftest_horizon()
    raise SystemExit(0 if (r["PASS"] and rh["PASS"]) else 1)

