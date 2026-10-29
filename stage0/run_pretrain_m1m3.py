"""
Stage P - PRETRAIN an xLSTM backbone on the external monthly benchmarks
          (M1 + M3), then FINE-TUNE it on the Hong Kong retail panel.

Question
--------
Hong Kong only has 23 series / 4,186 training windows.  M1-monthly (617 series)
and M3-monthly (1,428 series) are external monthly series with no HK
`series_id`.  Does pretraining on them buy any MASE on 2024?

Protocol (locked, identical to stage0/run_purged_val.py)
-------------------------------------------------------
* features      : the SAME 27 definitions as stage0/data/make_features.py
                  (lag1,2,3,12,13 / roll{3,6,12}_{mean,std} / month_1..12 /
                   lny_pre14=lny_post7=0 / covid=0 / voucher=0)
* context 12, direct horizons h = 1,3,6,12, learnable anchor a_h
* fine-tune     : purge_val=True, val_months=24, epochs=100, seeds 42/7/2024
* evaluation    : 12 origins 202312..202411 x 4 horizons x 23 series = 1,104 pts
                  through the LOCKED eval/harness.py
* ensemble      : ENS3 = arithmetic mean of the three seeds' forecasts,
                  then MinT_wls reconciliation (run_stage4_reconcile.reconcile)

Three arms, all sharing ONE code path
-------------------------------------
  scratch    XLSTMForecaster built from random init            (the incumbent)
  pretrained XLSTMForecaster with the pretrained backbone loaded,
             series embedding DISCARDED and re-initialised   (the treatment)

Both arms go through the same `model_factory`, so the only difference between
them is the `load_state_dict` call.  Because `train_model` calls `set_seed(seed)`
BEFORE it calls the factory, the HK series embedding is initialised from the
same RNG state in both arms -- so this is a controlled comparison, not a
re-run of two loosely similar scripts.

Why the pretraining model has no series embedding
-------------------------------------------------
M1/M3 have no HK `series_id`.  Rather than feed a meaningless id through
`nn.Embedding`, the pretraining variant drops the `ser`/`ser_proj` path
entirely: `h = inp(x)` with no per-series offset.  Every other parameter
(inp / backbone / norm / heads / anchor_logit) has the SAME shape as in
`XLSTMForecaster`, so the transfer is a plain `load_state_dict(strict=False)`
and the skipped keys are reported, never silently ignored.

Usage
-----
    # smoke test: 50 M1 series only, 1 pretrain epoch, 1 seed, 2 fine-tune epochs
    .venv\\Scripts\\python.exe run_pretrain_m1m3.py --smoke

    # full run
    .venv\\Scripts\\python.exe run_pretrain_m1m3.py

    # rebuild every table from the per-run prediction files (no refitting)
    .venv\\Scripts\\python.exe run_pretrain_m1m3.py --combine-only

Outputs (stage0/results/pretrain/)
    pretrain_loss.csv            epoch-by-epoch train/val loss + wall clock
    pretrain_meta.json           window counts, drop log, timings, shapes
    pretrain_backbone.pt         the transferred state_dict
    preds_{scratch,pretrained}_s<seed>.csv
    finetune_meta.json           per-run metadata incl. load report
    metrics.csv                  per (model, h) through the locked harness
    summary.csv                  the headline table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent          # .../stage0
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "models"))

import harness as H                        # noqa: E402  (LOCKED, read-only)
import xlstm_direct as X                   # noqa: E402  (LOCKED, read-only)
import run_stage4_reconcile as R4          # noqa: E402  (reconciliation utils)

PROJ = ROOT.parent
BENCH = PROJ / "data_external" / "raw" / "benchmarks"
RES = ROOT / "results" / "pretrain"

# M1/M3 monthly .tsf -> the two families used for pretraining.
# Quarterly/yearly/other are deliberately NOT used (different sampling frequency
# would need a second feature definition; that is out of scope here).
TSF = {
    "M1": BENCH / "M1" / "m1_monthly_dataset_extracted" / "m1_monthly_dataset.tsf",
    "M3": BENCH / "M3" / "m3_monthly_dataset_extracted" / "m3_monthly_dataset.tsf",
}

SEEDS = (42, 7, 2024)
PRETRAIN_SEED = 0
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 100
BATCH = 128
LR = 5e-4
WEIGHT_DECAY = 5e-3

CONTEXT = X.CONTEXT                        # 12
HEADS = tuple(H.HORIZONS)                  # (1, 3, 6, 12)
NFEAT = len(H.FEATURE_COLS)                # 27
KF_DIM = len(H.KNOWNFUT_COLS)              # 4
LAGS = (1, 2, 3, 12, 13)
ROLL = (3, 6, 12)


# ===========================================================================
# A. .tsf parsing
# ===========================================================================
def parse_tsf(path: Path) -> tuple[dict, list[dict]]:
    """Parse a Monash `.tsf` file.

    Data lines are `NAME:START_TIMESTAMP:v1,v2,...`.  `@horizon` marks how many
    trailing observations are the competition holdout -- those must NEVER enter
    pretraining, so the caller slices them off explicitly rather than trusting
    that the values happen to be last.

    START_TIMESTAMP carries the month-of-year alignment we need for the
    `month_1..12` one-hot.  Both files document that some series have a
    substituted start date (M1: 15 series year->00, +3 series month->01;
    M3: last 29 series -> 1900-01-01).  Those keep month 01, which is what the
    archive itself assumes; the count is reported in the meta, not hidden.
    """
    meta: dict[str, str] = {}
    recs: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        in_data = False
        for line in f:
            line = line.rstrip("\n")
            if not in_data:
                if line.startswith("@data"):
                    in_data = True
                elif line.startswith("@"):
                    key, _, val = line.partition(" ")
                    meta[key[1:].strip()] = val.strip()
                continue
            if not line.strip():
                continue
            parts = line.split(":")
            if len(parts) == 3:
                name, start, vals = parts
            elif len(parts) == 2:                      # no timestamp member
                name, start, vals = parts[0], None, parts[1]
            else:                                       # timestamp is dismembered
                name, start, vals = parts[0], ":".join(parts[1:-1]), parts[-1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arr = np.fromstring(vals, sep=",") if vals else np.array([])
            recs.append({"name": name.strip(), "start": start, "v": arr})
    return meta, recs


def start_month(start: str | None) -> int:
    """month-of-year of the first observation; January when unusable."""
    if not start:
        return 1
    try:
        m = int(start.split("-")[1])
    except (IndexError, ValueError):
        return 1
    return m if 1 <= m <= 12 else 1


# ===========================================================================
# B. per-series features, EXACTLY the make_features.py definitions
# ===========================================================================
def series_features(y_log: np.ndarray, month0: int) -> np.ndarray:
    """(n, 27) feature matrix for one series, column order == H.FEATURE_COLS.

    Warm-up rows that cannot support lag13 / roll12 are left as NaN, exactly
    like make_features.py; `make_samples` then refuses any window touching them.
    """
    n = len(y_log)
    F = np.full((n, NFEAT), np.nan, dtype=np.float32)
    for i in range(n):
        col = 0
        for L in LAGS:
            j = i - L
            if j >= 0:
                F[i, col] = y_log[j]
            col += 1
        for w in ROLL:
            lo = i - w + 1
            if lo >= 0:
                seg = y_log[lo:i + 1]
                F[i, col] = seg.mean()
                F[i, col + 1] = seg.std(ddof=0)        # make_features uses ddof=0
            col += 2
        mm = ((month0 - 1 + i) % 12) + 1
        for k in range(1, 13):
            F[i, col] = 1.0 if k == mm else 0.0
            col += 1
        # lunar-new-year shares: M1/M3 carry no lunar calendar -> 0
        F[i, col] = 0.0
        F[i, col + 1] = 0.0
        col += 2
        # HK-specific regime dummies: meaningless outside HK -> 0
        F[i, col] = 0.0
        F[i, col + 1] = 0.0
    assert col + 2 == NFEAT, (col, NFEAT)
    return F


def build_external_panels(families: list[str], horizon_from_file: bool = True,
                          max_series: int | None = None) -> tuple[list[dict], dict]:
    """Parse the .tsf files and turn every usable series into a feature panel.

    The `.tsf` holdout (the trailing `@horizon` observations) is removed BEFORE
    anything else, so no holdout value can reach the features, mu/sigma, or the
    targets.
    """
    panels: list[dict] = []
    log: dict = {"families": {}, "drops": []}

    for fam in families:
        path = TSF[fam]
        meta, recs = parse_tsf(path)
        horizon = int(meta.get("horizon", "18")) if horizon_from_file else int(meta.get("horizon", "0"))
        if max_series is not None:
            recs = recs[:max_series]
        fam_log = {
            "tsf": str(path.relative_to(PROJ)),
            "tsf_bytes": path.stat().st_size,
            "file_meta": meta,
            "n_series_in_file": len(recs),
            "holdout_len": horizon,
            "n_placeholder_start": int(sum(
                1 for r in recs if (r["start"] or "").startswith("1900-01-01"))),
            "n_no_start": int(sum(1 for r in recs if r["start"] is None)),
        }

        n_nonpos = n_nan = n_const = n_short = 0
        lengths = []
        for rec in recs:
            v = np.asarray(rec["v"], dtype=np.float64)
            if horizon:
                assert len(v) > horizon, (rec["name"], len(v), horizon)
                v = v[: len(v) - horizon]              # <-- holdout cut, first thing
            n = len(v)
            lengths.append(n)
            if not np.isfinite(v).all():
                n_nan += 1
                log["drops"].append({"family": fam, "series": rec["name"],
                                     "reason": "non_finite_in_train_segment"})
                continue
            if (v <= 0).any():
                n_nonpos += 1
                log["drops"].append({"family": fam, "series": rec["name"],
                                     "reason": "non_positive_in_train_segment",
                                     "min_value": float(v.min()),
                                     "n_nonpositive": int((v <= 0).sum())})
                continue
            y_log = np.log(v)
            mu, sd = float(y_log.mean()), float(y_log.std(ddof=0))   # train segment only
            if not np.isfinite(sd) or sd <= 1e-12:
                n_const += 1
                log["drops"].append({"family": fam, "series": rec["name"],
                                     "reason": "constant_series_sd=0"})
                continue
            y_z = (y_log - mu) / sd
            m0 = start_month(rec["start"])
            F = series_features(y_log, m0)

            # first origin whose 12-month window is NaN-free (lag13 binds)
            first = None
            for i in range(n):
                lo = i - CONTEXT + 1
                if lo < 0:
                    continue
                if np.isnan(F[lo:i + 1]).any():
                    continue
                first = i
                break
            last = n - 1 - max(HEADS)                  # all 4 targets in range
            if first is None or last < first:
                n_short += 1
                log["drops"].append({"family": fam, "series": rec["name"],
                                     "reason": "too_short_for_context+horizon",
                                     "train_len": n,
                                     "first_legal_origin": first,
                                     "last_legal_origin": last})
                continue
            panels.append({"family": fam, "series": rec["name"], "month0": m0,
                           "F": F, "y_z": y_z.astype(np.float32),
                           "first": first, "last": last, "train_len": n,
                           "mu_log": mu, "sigma_log": sd})

        fam_log.update({
            "n_series_usable": int(sum(1 for p in panels if p["family"] == fam)),
            "train_len_min": int(min(lengths)) if lengths else 0,
            "train_len_max": int(max(lengths)) if lengths else 0,
            "train_len_mean": round(float(np.mean(lengths)), 2) if lengths else 0.0,
            "train_obs_total": int(np.sum(lengths)),
            "n_dropped": {"non_positive": n_nonpos, "non_finite": n_nan,
                          "constant": n_const, "too_short": n_short,
                          "total": n_nonpos + n_nan + n_const + n_short},
        })
        log["families"][fam] = fam_log
        print(f"  [{fam}] {len(recs)} series in file -> {fam_log['n_series_usable']} usable "
              f"(dropped {fam_log['n_dropped']['total']}: "
              f"non-positive {n_nonpos}, non-finite {n_nan}, "
              f"constant {n_const}, too-short {n_short})")
    return panels, log


# ===========================================================================
# C. windows
# ===========================================================================
def build_windows(panels: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Replicate `xlstm_direct.make_samples` exactly, one series at a time.

    make_samples() works on a (period x series x feat) cube, which cannot hold
    2,045 series with different start dates and lengths.  The per-series loop
    below applies the identical three rules: a 12-row window with no NaN, and
    all four targets present inside the series' own training segment.

    The last channel of each window is y_z itself, because
    `XLSTMForecaster._anchor_terms` reads the anchor out of `x[:, :, -1]`; the
    input projection drops it via `x[:, :, :-1]`.
    """
    Xs, Cs, Ts = [], [], []
    for p in panels:
        F, yz, m0 = p["F"], p["y_z"], p["month0"]
        for i in range(p["first"], p["last"] + 1):
            lo = i - CONTEXT + 1
            win = F[lo:i + 1]
            if np.isnan(win).any():
                continue
            tgt = [yz[i + h] for h in HEADS]
            if any(not np.isfinite(t) for t in tgt):
                continue
            full = np.empty((CONTEXT, NFEAT + 1), dtype=np.float32)
            full[:, :NFEAT] = win
            full[:, NFEAT] = yz[lo:i + 1]
            Xs.append(full)
            # known-future calendar of the TARGET month.  The champion config has
            # use_calendar=False so these are unused, but they are built honestly
            # (sin/cos of month-of-year, lny shares = 0) so that flipping
            # use_calendar on would not silently feed zeros.
            cal = np.zeros((len(HEADS), KF_DIM), dtype=np.float32)
            for k, h in enumerate(HEADS):
                mm = ((m0 - 1 + i + h) % 12) + 1
                ang = 2.0 * np.pi * (mm - 1) / 12.0
                cal[k] = (0.0, 0.0, np.sin(ang), np.cos(ang))
            Cs.append(cal)
            Ts.append(tgt)
    return (np.asarray(Xs, dtype=np.float32), np.asarray(Cs, dtype=np.float32),
            np.asarray(Ts, dtype=np.float32), panels)


# ===========================================================================
# D. the pretraining backbone: XLSTMForecaster WITHOUT the series embedding
# ===========================================================================
class XLSTMForecasterNoSID(nn.Module):
    """Copy of `xlstm_direct.XLSTMForecaster` with the series-embedding path
    removed.

    Identical in every other respect -- same input projection, same 2-block
    xLSTM stack (mLSTM -> sLSTM, backend="vanilla"), same LayerNorm/dropout,
    same four direct-horizon heads, same learnable anchor a_h.  The only
    difference is that `_embed_input` returns `inp(x)` instead of
    `inp(x) + ser_proj(ser(sid))`, so every remaining parameter has exactly the
    shape it has in `XLSTMForecaster` and `load_state_dict(strict=False)` is a
    plain transfer.

    `sLSTMLayerConfig.backend` MUST be "vanilla": the package default is "cuda",
    whose DLL fails to load on this machine.
    """

    def __init__(self, n_feat, embed=None, blocks=None, context=None,
                 dropout=None, kf_dim=KF_DIM, use_anchor=True, use_calendar=False,
                 anchor_init=2.0):
        super().__init__()
        from xlstm import (
            xLSTMBlockStack, xLSTMBlockStackConfig,
            mLSTMBlockConfig, mLSTMLayerConfig,
            sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig,
        )
        embed = X.EMBED if embed is None else int(embed)
        blocks = X.BLOCKS if blocks is None else int(blocks)
        context = CONTEXT if context is None else int(context)
        dropout = X.DROPOUT if dropout is None else float(dropout)
        self.context = context
        self.dropout_p = dropout
        self.n_feat = int(n_feat)
        self.inp = nn.Sequential(nn.Linear(n_feat, embed), nn.LayerNorm(embed), nn.GELU())
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
        self.use_anchor = bool(use_anchor)
        self.use_calendar = bool(use_calendar)
        self.heads = nn.ModuleList([
            nn.Linear(embed + (kf_dim if use_calendar else 0), 1) for _ in HEADS])
        self.anchor_logit = nn.Parameter(torch.full((len(HEADS),), float(anchor_init)))

    def _embed_input(self, x, sid=None):
        return self.inp(x[:, :, :-1])

    def _anchor_terms(self, x):
        L = x.shape[1]
        base_t = x[:, -1, -1].unsqueeze(1)
        seasonal = []
        for h in HEADS:
            idx = L + h - X.SEASON - 1
            seasonal.append(base_t[:, 0] if idx < 0 else x[:, idx, -1])
        base_s = torch.stack(seasonal, dim=1)
        if self.use_anchor:
            a = torch.sigmoid(self.anchor_logit).unsqueeze(0)
            return a * base_t + (1.0 - a) * base_s
        return base_t

    def _head_delta(self, h, x, sid, cal):
        if self.use_calendar:
            return torch.cat([self.heads[k](torch.cat([h, cal[:, k, :]], dim=-1))
                              for k in range(len(HEADS))], dim=-1)
        return torch.cat([self.heads[k](h) for k in range(len(HEADS))], dim=-1)

    def forward(self, x, sid=None, cal=None):
        anchor = self._anchor_terms(x)
        h = self._backbone(self._embed_input(x, sid))
        h = self.norm(self.drop(h[:, -1, :]))
        return anchor + self._head_delta(h, x, sid, cal)

    def _backbone(self, h):
        return self.backbone(h)

    def anchor_weights(self):
        return torch.sigmoid(self.anchor_logit).detach().cpu().numpy()


# ===========================================================================
# E. pretraining loop
# ===========================================================================
def pretrain(Xw, Cw, Tw, epochs=PRETRAIN_EPOCHS, seed=PRETRAIN_SEED,
             batch=BATCH, lr=LR, weight_decay=WEIGHT_DECAY, val_frac=0.05,
             verbose=True) -> tuple[XLSTMForecasterNoSID, dict, pd.DataFrame]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(Xw)
    rng = np.random.default_rng(0)                 # split fixed independent of model seed
    order = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, tr_idx = order[:n_val], order[n_val:]

    Xt = torch.from_numpy(Xw[tr_idx]).to(dev)
    Ct = torch.from_numpy(Cw[tr_idx]).to(dev)
    Tt = torch.from_numpy(Tw[tr_idx]).to(dev)
    Xv = torch.from_numpy(Xw[val_idx]).to(dev)
    Cv = torch.from_numpy(Cw[val_idx]).to(dev)
    Tv = torch.from_numpy(Tw[val_idx]).to(dev)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = XLSTMForecasterNoSID(n_feat=NFEAT, embed=X.EMBED, blocks=X.BLOCKS,
                                 context=CONTEXT, dropout=X.DROPOUT,
                                 use_anchor=True, use_calendar=False).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if verbose:
        print(f"  pretrain device {dev}  params {n_par:,}  "
              f"train windows {len(tr_idx):,}  val windows {len(val_idx):,}")
        print(f"  {epochs} epochs, batch {batch}, lr {lr}, wd {weight_decay}")

    rows = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_idx), device=dev)
        tot = 0.0
        for k in range(0, len(tr_idx), batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            se = (model(Xt[idx], None, Ct[idx]) - Tt[idx]) ** 2
            loss = se.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vt = 0.0
            for k in range(0, len(val_idx), 512):
                sl = slice(k, k + 512)
                vt += ((model(Xv[sl], None, Cv[sl]) - Tv[sl]) ** 2).mean().item() * len(Xv[sl])
            vloss = vt / len(val_idx)
        tr_loss = tot / len(tr_idx)
        rows.append({"epoch": ep, "train_loss": round(tr_loss, 6),
                     "val_loss": round(vloss, 6),
                     "lr": round(float(sched.get_last_lr()[0]), 8),
                     "elapsed_sec": round(time.time() - t0, 1)})
        if verbose:
            print(f"    ep {ep:3d}  train {tr_loss:.4f}  val {vloss:.4f}  "
                  f"({rows[-1]['elapsed_sec']:.0f}s)", flush=True)
    wall = time.time() - t0

    meta = {
        "n_params": int(n_par),
        "n_windows_total": int(n),
        "n_pretrain_windows": int(len(tr_idx)),
        "n_val_windows": int(len(val_idx)),
        "val_frac": float(val_frac),
        "epochs": int(epochs), "batch": int(batch), "lr": float(lr),
        "weight_decay": float(weight_decay), "seed": int(seed),
        "device": dev,
        "loss_first_epoch": rows[0]["train_loss"],
        "loss_last_epoch": rows[-1]["train_loss"],
        "val_first_epoch": rows[0]["val_loss"],
        "val_last_epoch": rows[-1]["val_loss"],
        "val_best": float(min(r["val_loss"] for r in rows)),
        "val_best_epoch": int(min(rows, key=lambda r: r["val_loss"])["epoch"]),
        "pretrain_sec": round(wall, 1),
        "anchor_weights": {str(h): float(v)
                           for h, v in zip(HEADS, model.anchor_weights())},
    }
    return model, meta, pd.DataFrame(rows)


# ===========================================================================
# F. fine-tuning: the LOCKED harness's own training loop, with a weight hook
# ===========================================================================
def make_finetune_factory(state_dict: dict | None, ctx: int, emb: int, blk: int,
                          drp: float, load_report: dict | None = None):
    """Build the HK `XLSTMForecaster` exactly as `train_model` would, optionally
    after loading the pretrained backbone.

    `train_model` calls `set_seed(seed)` before invoking the factory, so in BOTH
    arms the HK series embedding is drawn from an identical RNG state; loading
    `state_dict` consumes no randomness.  The two arms therefore differ in one
    thing only: whether the backbone carries the pretrained weights.

    The pretrained series-embedding keys do not exist (the pretraining model had
    no `ser`/`ser_proj`), and any that did would be dropped here anyway -- HK
    series ids are unrelated to M1/M3 series names, so there is nothing to
    transfer.
    """
    def factory(n_series: int, n_feat: int):
        model = X.XLSTMForecaster(n_series, n_feat, embed=emb, blocks=blk,
                                  context=ctx, dropout=drp,
                                  use_anchor=True, use_calendar=False,
                                  anchor_init=2.0)
        if state_dict is not None:
            own = model.state_dict()
            skip_prefix = ("ser.", "ser_proj.")
            loadable = {k: v for k, v in state_dict.items()
                        if k in own and own[k].shape == v.shape
                        and not k.startswith(skip_prefix)}
            dropped = sorted(k for k in state_dict if k not in loadable)
            missing = sorted(k for k in own if k not in loadable)
            model.load_state_dict(loadable, strict=False)
            if load_report is not None:
                load_report.update({
                    "n_keys_in_checkpoint": len(state_dict),
                    "n_keys_loaded": len(loadable),
                    "keys_loaded": sorted(loadable),
                    "keys_dropped_from_checkpoint": dropped,
                    "keys_left_at_random_init": missing,
                    "n_params_transferred": int(sum(
                        v.numel() for v in loadable.values())),
                })
        return model
    return factory


def fit_one(feat, wide, scaling, arm: str, seed: int, epochs: int,
            state_dict: dict | None, verbose: bool = True):
    """One fine-tune run through `X.train_model` + `X.predict_at` (the locked
    modules' own functions; nothing here re-implements the protocol)."""
    periods, series, F, Y, KF = X.build_tensors(feat)
    load_report: dict = {}
    factory = make_finetune_factory(
        state_dict if arm == "pretrained" else None,
        ctx=CONTEXT, emb=X.EMBED, blk=X.BLOCKS, drp=X.DROPOUT,
        load_report=load_report)

    t0 = time.time()
    model, meta = X.train_model(
        F, Y, KF, periods, series, H.TRAIN_END,
        epochs=epochs, purge_val=True, val_months=24, seed=seed,
        verbose=verbose, model_factory=factory)
    train_sec = time.time() - t0

    rows = []
    for o in H.ORIGINS:
        pr = X.predict_at(model, F, KF, periods, series, o)
        if pr is None:
            continue
        for h in HEADS:
            tgt = H.month_add(o, h)
            if tgt not in wide.index:
                continue
            yhat = H.inverse_transform(pr[h], np.asarray(series), scaling)
            for s, yp in zip(series, yhat):
                rows.append((o, h, s, float(wide.loc[tgt, s]), float(yp)))
    df = pd.DataFrame(rows, columns=["origin", "h", "series", "y_true", "y_pred"])

    meta = dict(meta)
    meta.update({"tag": f"{arm}_s{seed}", "arm": arm, "seed": seed,
                 "train_sec": round(train_sec, 1), "n_points": int(len(df)),
                 "load_report": load_report})
    if verbose:
        print(f"  {arm} seed={seed}: {len(df):,} points, {train_sec:.1f}s, "
              f"best_val {meta['best_val_loss']:.4f} @ ep {meta['best_epoch']}")
    return df, meta


# ===========================================================================
# G. ensemble + reconciliation + metrics
# ===========================================================================
def ensemble_and_mint(frames: list[pd.DataFrame], wide: pd.DataFrame,
                      label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ENS3 = arithmetic mean of the seeds' forecasts, then MinT_wls."""
    series = list(wide.columns)
    keys = ["origin", "h", "series"]
    idx = [f.set_index(keys) for f in frames]
    P = pd.concat([d["y_pred"] for d in idx], axis=1)
    y_true = idx[0]["y_true"].reindex(P.index)
    for d in idx[1:]:
        d_yt = d["y_true"].reindex(P.index)
        assert np.allclose(d_yt.to_numpy(float), y_true.to_numpy(float)), "y_true drift"

    ens = P.mean(axis=1).rename("y_pred").reset_index()
    ens["y_true"] = y_true.reindex(pd.MultiIndex.from_frame(ens[keys])).to_numpy()
    ens["model"] = label

    A = R4.build_A(series)
    W = np.diag(R4.naive_w(wide, series, H.TRAIN_END))
    piv = ens.pivot_table(index=["origin", "h"], columns="series",
                          values="y_pred", aggfunc="first").reindex(columns=series)
    rec = R4.reconcile(piv.to_numpy(dtype=float), A, W)
    recdf = pd.DataFrame(rec, index=piv.index, columns=pd.Index(series, name="series"))
    lookup = recdf.stack().rename("y_pred_rec").reset_index()
    lookup.columns = keys + ["y_pred_rec"]
    minted = ens.merge(lookup, on=keys, how="left")
    assert len(minted) == len(ens) and minted["y_pred_rec"].notna().all()
    minted["y_pred"] = minted["y_pred_rec"]
    minted = minted.drop(columns=["y_pred_rec"])
    minted["model"] = label + "+MinT_wls"
    return ens, minted


def seed_spread(all_preds: pd.DataFrame, naive_mae: pd.Series,
                seeds) -> pd.DataFrame:
    """Per-arm, per-seed MASE_ALL.  This is the empirical noise floor that any
    claimed improvement has to clear -- the threshold is not assumed."""
    rows = []
    for model, g in all_preds.groupby("model", sort=False):
        if "_s" not in model or "ENS" in model:
            continue
        m = H.metrics_from_frame(g, naive_mae)
        row = m[m["h"].astype(str) == "ALL"].iloc[0]
        rows.append({"arm": model.rsplit("_s", 1)[0],
                     "seed": int(model.rsplit("_s", 1)[1]),
                     "MASE_ALL": float(row["MASE"]),
                     "WAPE_ALL": float(row["WAPE"])})
    df = pd.DataFrame(rows)
    extra = []
    for arm, g in df.groupby("arm", sort=False):
        v = g["MASE_ALL"].to_numpy(float)
        extra.append({"arm": arm, "seed": "MEAN",
                      "MASE_ALL": float(v.mean()),
                      "WAPE_ALL": float(g["WAPE_ALL"].mean())})
        extra.append({"arm": arm, "seed": "STD(ddof=1)",
                      "MASE_ALL": float(v.std(ddof=1)),
                      "WAPE_ALL": float(g["WAPE_ALL"].to_numpy(float).std(ddof=1))})
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def stats_report(all_preds: pd.DataFrame, naive_mae: pd.Series) -> pd.DataFrame:
    """Pairwise tests between the two arms' ENSEMBLES.

    Three aggregation levels, because on a PANEL they disagree and only
    reporting the pooled one would overstate the result (see
    harness.dm_time_aggregated's own docstring):
      pooled   DM on every (origin, series) point   -- treats 23 series as
                                                       independent, too optimistic
      series   Wilcoxon on the 23 per-series MAEs   -- "better on most categories?"
      time     Wilcoxon on the 12 per-origin MAEs   -- "better on the average month?"
    """
    rows = []
    pairs = [("xLSTM_scratch_ENS3", "xLSTM_pretrained_ENS3"),
             ("xLSTM_scratch_ENS3+MinT_wls", "xLSTM_pretrained_ENS3+MinT_wls")]
    key = ["origin", "series"]
    for a_name, b_name in pairs:
        for h in list(HEADS) + ["ALL"]:
            sub = all_preds if h == "ALL" else all_preds[all_preds["h"] == h]
            a = sub[sub["model"] == a_name].set_index(key)
            b = sub[sub["model"] == b_name].set_index(key)
            if a.empty or b.empty:
                continue
            eb = b["y_pred"].reindex(a.index)
            ea = a["y_true"] - a["y_pred"]
            ebb = b["y_true"].reindex(a.index) - eb
            lag = 1 if h == "ALL" else int(h)
            for power in (1, 2):
                r = H.dm_test(ea.to_numpy(), ebb.to_numpy(), h=lag, power=power)
                rows.append({"pair": f"{a_name} vs {b_name}", "h": h,
                             "level": "pooled", "n": r["n"],
                             "stat": r["dm"], "p_value": r["p_value"],
                             "better": r["better"], "loss_power": power})
        both = pd.concat([all_preds[all_preds["model"] == a_name],
                          all_preds[all_preds["model"] == b_name]], ignore_index=True)
        for fn, lvl in ((H.dm_series_level, "series"), (H.dm_time_aggregated, "time")):
            t = fn(both, baseline=a_name)
            t = t[t["model"] == b_name]
            for r in t.itertuples(index=False):
                col = "series" if lvl == "series" else "origins"
                rows.append({"pair": f"{a_name} vs {b_name}", "h": int(r.h),
                             "level": lvl, "n": int(getattr(r, f"n_{col}")),
                             "stat": np.nan, "p_value": float(r.p_wilcoxon),
                             "better": ("scratch" if r.better == a_name else "pretrained"),
                             "loss_power": 1,
                             f"{col}_won": int(getattr(r, f"{col}_won")),
                             f"{col}_lost": int(getattr(r, f"{col}_lost")),
                             "median_diff": float(r.median_diff)})
    return pd.DataFrame(rows)


def metrics_for(all_preds: pd.DataFrame, naive_mae: pd.Series) -> pd.DataFrame:
    return H.metrics_from_frame(all_preds, naive_mae)


# ===========================================================================
# H. driver
# ===========================================================================
def load_hk():
    feat = pd.read_csv(ROOT / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(ROOT / "data" / "scaling.csv", dtype={"series": str})
    return feat, wide, scaling


def summarise(all_preds: pd.DataFrame, naive_mae: pd.Series,
              baseline_mase: float = 0.94810) -> pd.DataFrame:
    """One row per model: seed count, MASE_ALL, WAPE_ALL, delta vs from-scratch."""
    rows = []
    for model, g in all_preds.groupby("model", sort=False):
        m = H.metrics_from_frame(g, naive_mae)
        row = m[m["h"].astype(str) == "ALL"].iloc[0]
        per_h = {f"MASE_h{int(r.h)}": float(r.MASE)
                 for r in m.itertuples(index=False) if str(r.h) != "ALL"}
        rows.append({"model": model, "n_points": int(row["n_points"]),
                     "MASE_ALL": float(row["MASE"]), "WAPE_ALL": float(row["WAPE"]),
                     **per_h})
    df = pd.DataFrame(rows)

    # "relative to from-scratch" is measured against THIS script's own control
    # arm, not against the published 0.94810, so the comparison is controlled.
    base = df.loc[df["model"] == "xLSTM_scratch_ENS3", "MASE_ALL"]
    base_v = float(base.iloc[0]) if len(base) else np.nan
    df["vs_scratch_ENS3"] = df["MASE_ALL"] - base_v
    df["vs_scratch_pct"] = 100.0 * (df["MASE_ALL"] - base_v) / base_v
    base_pub = baseline_mase
    df["vs_published_0.94810"] = df["MASE_ALL"] - base_pub
    df["n_seeds"] = 3
    return df


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="50 M1 series, 1 pretrain epoch, 1 seed, 2 fine-tune epochs")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    ap.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--families", nargs="+", default=["M1", "M3"])
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--outdir", default=None,
                    help="override the output directory (default results/pretrain)")
    args = ap.parse_args()

    if args.smoke:
        args.families, args.max_series = ["M1"], 50
        args.epochs, args.finetune_epochs, args.seeds = 1, 2, [42]
        # a smoke run must NOT leave artifacts where the full run looks for them:
        # `preds_scratch_s42.csv` from a 2-epoch run would be picked up as
        # "already done" and silently become the control arm.
        args.outdir = args.outdir or "results/pretrain_smoke"

    global RES
    if args.outdir:
        RES = (ROOT / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    RES.mkdir(parents=True, exist_ok=True)
    feat, wide, scaling = load_hk()
    naive_mae = H.seasonal_naive_mae_train(wide)
    print("=" * 78)
    print("STAGE P - pretrain xLSTM on M1+M3, fine-tune on the HK retail panel")
    print("=" * 78)
    print(f"  HK panel {wide.shape[0]} periods x {wide.shape[1]} series   "
          f"features {NFEAT}   context {CONTEXT}   horizons {HEADS}")

    if not args.combine_only:
        # ---- A. parse + features -----------------------------------------
        print("\n--- A. parse .tsf, build the 27 HK-identical features ---")
        t_parse = time.time()
        panels, plog = build_external_panels(args.families,
                                             max_series=args.max_series)
        Xw, Cw, Tw, _ = build_windows(panels)
        parse_sec = time.time() - t_parse
        print(f"  panels {len(panels)}   windows {len(Xw):,}   "
              f"X {Xw.shape}  C {Cw.shape}  T {Tw.shape}   ({parse_sec:.1f}s)")
        print(f"  HK windows for comparison: 4,186 train (+299 val)")
        assert Xw.shape[1:] == (CONTEXT, NFEAT + 1), Xw.shape
        assert Tw.shape[1] == len(HEADS) and Cw.shape[1:] == (len(HEADS), KF_DIM)

        # ---- B1. pretrain -------------------------------------------------
        print("\n--- B1. pretrain the no-series-embedding backbone ---")
        model, pmeta, curve = pretrain(Xw, Cw, Tw, epochs=args.epochs)
        curve.to_csv(RES / "pretrain_loss.csv", index=False)
        torch.save(model.state_dict(), RES / "pretrain_backbone.pt")
        pmeta.update({"parse_sec": round(parse_sec, 1),
                      "families": args.families,
                      "max_series": args.max_series,
                      "feature_cols": list(H.FEATURE_COLS),
                      "drop_log": plog,
                      "hk_train_windows": 4186})
        (RES / "pretrain_meta.json").write_text(
            json.dumps(pmeta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  loss  ep1 {pmeta['loss_first_epoch']:.4f} -> "
              f"ep{args.epochs} {pmeta['loss_last_epoch']:.4f}   "
              f"val {pmeta['val_first_epoch']:.4f} -> {pmeta['val_last_epoch']:.4f} "
              f"(best {pmeta['val_best']:.4f} @ ep {pmeta['val_best_epoch']})")
        print(f"  pretrain wall {pmeta['pretrain_sec']:.1f}s")
        state = torch.load(RES / "pretrain_backbone.pt", map_location="cpu")

        # ---- B2. fine-tune ------------------------------------------------
        print("\n--- B2. fine-tune on HK: control vs pretrained ---")
        metas = []
        for arm in ("scratch", "pretrained"):
            for sd in args.seeds:
                tag = f"{arm}_s{sd}"
                out = RES / f"preds_{tag}.csv"
                if out.exists() and not args.smoke:
                    print(f"  [skip] {tag} already done")
                    df = pd.read_csv(out, dtype={"series": str, "origin": str})
                    metas.append(json.loads(
                        (RES / f"meta_{tag}.json").read_text(encoding="utf-8")))
                    continue
                df, m = fit_one(feat, wide, scaling, arm, sd,
                                args.finetune_epochs,
                                state if arm == "pretrained" else None,
                                verbose=True)
                df.to_csv(out, index=False)
                (RES / f"meta_{tag}.json").write_text(
                    json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
                metas.append(m)
        (RES / "finetune_meta.json").write_text(
            json.dumps(metas, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- C. combine -------------------------------------------------------
    print("\n--- C. ENS3 + MinT_wls + metrics through the locked harness ---")
    all_preds, frames_meta = [], []
    for arm in ("scratch", "pretrained"):
        frames = []
        for sd in args.seeds:
            f = RES / f"preds_{arm}_s{sd}.csv"
            if not f.exists():
                raise SystemExit(f"missing {f} -- run without --combine-only")
            d = pd.read_csv(f, dtype={"series": str, "origin": str})
            d["model"] = f"xLSTM_{arm}_s{sd}"
            frames.append(d)
        all_preds.extend(frames)
        ens, minted = ensemble_and_mint(frames, wide, f"xLSTM_{arm}_ENS{len(frames)}")
        all_preds.extend([ens, minted])

    all_preds = pd.concat(all_preds, ignore_index=True)
    all_preds.to_csv(RES / "pretrain_predictions.csv", index=False)
    metrics = metrics_for(all_preds, naive_mae)
    metrics.to_csv(RES / "metrics.csv", index=False)
    summary = summarise(all_preds, naive_mae)
    summary.to_csv(RES / "summary.csv", index=False)
    spread = seed_spread(all_preds, naive_mae, args.seeds)
    spread.to_csv(RES / "seed_spread.csv", index=False)
    stats = stats_report(all_preds, naive_mae)
    stats.to_csv(RES / "stats.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 78)
    print("  SUMMARY  (1,104 test points: 12 origins x 4 horizons x 23 series)")
    print("=" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    bad = summary[~np.isclose(summary["n_points"], 1104)]
    if len(bad):
        print(f"\n  !! WARNING: rows with n_points != 1104:\n{bad.to_string(index=False)}")
    else:
        print("\n  every row scored on exactly 1,104 points")

    print("\n  per-horizon MASE")
    piv = metrics.pivot_table(index="model", columns="h", values="MASE", sort=False)
    cols = [c for c in piv.columns if str(c) != "ALL"] + ["ALL"]
    print(piv[cols].to_string(float_format=lambda v: f"{v:.5f}"))

    print("\n  per-seed MASE_ALL (the noise floor any gain must clear)")
    print(spread.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n  significance (scratch ENS3 -> pretrained ENS3)")
    pv = stats[stats["pair"].str.contains("scratch_ENS3 vs")]
    print(pv[["pair", "h", "level", "n", "stat", "p_value", "better",
              "loss_power"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
