"""
Stage 3 — a plain nn.LSTM backbone, everything else held fixed.

WHY
---
Ablation item #4 of `01_ARCHITECTURE.md` §7 asks whether xLSTM itself earns its
place.  Comparing "xLSTM run A" against "LSTM run B" from two different scripts
is worthless, because any difference could come from the input projection, the
anchor, the loss, the schedule, the early-stopping rule, or the seed handling.

So this module changes EXACTLY ONE thing: the sequence mixer.  `LSTMForecaster`
subclasses the shipped `XLSTMForecaster`, keeps its input projection
(`inp` / `inp_endo` / `inp_exog` / gate / LayerNorm / GELU), keeps its series
embedding, keeps its learnable persistence-seasonal anchor, keeps its four heads,
and keeps its `forward` -- and only swaps `_backbone`.

    xLSTM :  xLSTMBlockStack(2 blocks, mLSTM x1 + sLSTM x1, embed=32)
    LSTM  :  nn.LSTM(input_size=32, hidden_size=32, num_layers=2,
                     batch_first=True, dropout=0.3)              -> last hidden

Deterministic-mixer note: `nn.LSTM` has no internal dropout on the final layer,
so with num_layers=2 the specified dropout applies between layer 1 and 2 only.
That matches the shipped xLSTM stack, which likewise applies its dropout once.

Usage
    .venv\\Scripts\\python.exe -m stage3.lstm_backbone      (selftest, CPU, ~2 s)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "stage0") not in sys.path:
    sys.path.insert(0, str(ROOT / "stage0"))

from models import xlstm_direct as X      # noqa: E402


class LSTMForecaster(X.XLSTMForecaster):
    """XLSTMForecaster with the xLSTM block stack replaced by a 2-layer LSTM."""

    def __init__(self, n_series, n_feat, embed=X.EMBED, blocks=X.BLOCKS,
                 series_emb=X.SERIES_EMB, context=X.CONTEXT, dropout=0.3,
                 kf_dim=X.KF_DIM, use_anchor=True, use_calendar=False,
                 anchor_init=2.0, lstm_dropout=0.3):
        super().__init__(n_series, n_feat, embed=embed, blocks=blocks,
                         series_emb=series_emb, context=context, dropout=dropout,
                         kf_dim=kf_dim, use_anchor=use_anchor,
                         use_calendar=use_calendar, anchor_init=anchor_init)
        self.lstm = nn.LSTM(input_size=embed, hidden_size=embed, num_layers=2,
                            batch_first=True, dropout=lstm_dropout)
        # the inherited xLSTM stack is now unused; drop it so it cannot be
        # trained and does not pollute the parameter count
        self.backbone = nn.Identity()

    def _backbone(self, h):
        out, _ = self.lstm(h)
        return out


class LSTMForecasterFiLM(X.XLSTMForecaster):
    """LSTM backbone AND the Series-Conditioned FiLM input path from
    models/xlstm_exog.py, so text can be fed to the LSTM the same way."""

    def __init__(self, n_series, n_total_feat, n_numeric, n_exog,
                 embed=X.EMBED, blocks=X.BLOCKS, series_emb=X.SERIES_EMB,
                 context=X.CONTEXT, dropout=0.3, kf_dim=X.KF_DIM,
                 lstm_dropout=0.3):
        super().__init__(n_series, n_total_feat, embed=embed, blocks=blocks,
                         series_emb=series_emb, context=context, dropout=dropout,
                         kf_dim=kf_dim, use_anchor=True, use_calendar=False,
                         anchor_init=2.0)
        self.n_numeric = int(n_numeric)
        self.n_exog = int(n_exog)
        self.inp_endo = nn.Linear(self.n_numeric, embed)
        self.inp_exog = nn.Linear(self.n_exog, embed)
        self.gate = nn.Linear(series_emb, embed)
        self.in_ln = nn.LayerNorm(embed)
        self.in_act = nn.GELU()
        nn.init.zeros_(self.inp_exog.weight)
        nn.init.zeros_(self.inp_exog.bias)
        nn.init.uniform_(self.gate.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.gate.bias)
        self.register_buffer("exog_scale", torch.tensor(1.0))
        self.inp = nn.Identity()
        self.lstm = nn.LSTM(input_size=embed, hidden_size=embed, num_layers=2,
                            batch_first=True, dropout=lstm_dropout)
        self.backbone = nn.Identity()

    def _embed_input(self, x, sid):
        num = x[:, :, : self.n_numeric]
        exo = x[:, :, self.n_numeric: -1]
        g = torch.tanh(self.gate(self.ser(sid)))
        h = self.inp_endo(num) + self.exog_scale * (g.unsqueeze(1) * self.inp_exog(exo))
        h = self.in_act(self.in_ln(h))
        return h + self.ser_proj(self.ser(sid)).unsqueeze(1)

    def _backbone(self, h):
        out, _ = self.lstm(h)
        return out


def selftest(verbose: bool = True) -> dict:
    """same three properties the FiLM selftest checks, for the LSTM backbone"""
    torch.manual_seed(0)
    n_series, n_num, ctx = 5, 27, X.CONTEXT
    base = X.XLSTMForecaster(n_series, n_num, use_anchor=True, use_calendar=False)
    lstm = LSTMForecaster(n_series, n_num)
    film = LSTMForecasterFiLM(n_series, n_num + 4, n_num, 4)

    x = torch.randn(8, ctx, n_num + 1)
    sid = torch.randint(0, n_series, (8,))
    cal = torch.zeros(8, len(X.HEADS), X.KF_DIM)
    for name, m in (("xLSTM", base), ("LSTM", lstm)):
        m.eval()
        with torch.no_grad():
            out = m(x, sid, cal)
        assert out.shape == (8, len(X.HEADS)), (name, out.shape)

    # FiLM+LSTM: exogenous branch must be OFF at step 0
    x2 = torch.randn(8, ctx, n_num + 4 + 1)
    film.eval()
    with torch.no_grad():
        a = film(x2, sid, cal)
        x3 = x2.clone()
        x3[:, :, n_num:n_num + 4] = torch.randn(8, ctx, 4) * 100
        b = film(x3, sid, cal)
    p1 = bool(torch.allclose(a, b, atol=0, rtol=0))

    film.train(); film.zero_grad()
    ((film(x2, sid, cal)) ** 2).mean().backward()
    p2 = float(film.inp_exog.weight.grad.abs().sum()) > 0

    res = {
        "P1_film_lstm_epoch0_identical": p1,
        "P2_film_lstm_branch_trainable": p2,
        "n_params_xLSTM_backbone": sum(p.numel() for p in base.parameters()),
        "n_params_LSTM_backbone": sum(p.numel() for p in lstm.parameters()),
        "n_params_LSTM_FiLM": sum(p.numel() for p in film.parameters()),
    }
    res["PASS"] = bool(p1 and p2)
    if verbose:
        print("LSTM backbone selftest")
        for k, v in res.items():
            print(f"  {k:34s} {v}")
        print(f"\n  {'PASS' if res['PASS'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    r = selftest()
    raise SystemExit(0 if r["PASS"] else 1)
