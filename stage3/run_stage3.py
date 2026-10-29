"""
Stage 3 — news text fusion ablation, plus the LSTM backbone control, plus the
purge_val window sensitivity check (k = 6).

CELLS (each x 3 seeds 42 / 7 / 2024, all purge_val=True)

    none                0 columns                         (reuse Stage-1 result)
    text_scalar         3 dictionary/scalar columns       vs random3 (width 3)
    text_emb            4 PCA text columns                vs random4 (width 4)
    visitor_mtr_text    2 people columns + 4 PCA (6)      vs random6 (width 6)
    LSTM_none           plain nn.LSTM, 0 exogenous
    LSTM_text_emb       plain nn.LSTM + 4 PCA text
    none_k6             0 columns, validation = last 6 legal origins
    visitor_mtr_k6      2 people columns, same k=6 validation

WHY THE WIDTH-MATCHED NOISE CONTROL AGAIN
    Stage 1 established that every extra input column costs accuracy on its own
    (`stage1/RESULTS.md` §7).  A text cell is therefore only allowed to claim
    anything if it beats a NOISE block of the same width.  `random3` is reused
    from Stage 1 because it already exists at the right width; `random4` and
    `random6` are added here.

WHY THE k=6 CELLS
    The shipped protocol uses `val_from = cutoff - 24`, purged, which discards
    2023 entirely and trains only up to 2021-11.  Anchoring validation to the
    LAST k legal origins instead lets the training set extend to 2022-06.  This
    measures whether that extra 7 months of (post-COVID-reopening) history is
    worth more than the wider validation block.

Usage
    .venv\\Scripts\\python.exe ../stage3/run_stage3.py --only text_emb:42
    .venv\\Scripts\\python.exe ../stage3/run_stage3.py --combine-only
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
STAGE0 = ROOT.parent / "stage0"
for p in (str(ROOT), str(STAGE0)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval import harness as H              # noqa: E402
from models import xlstm_direct as X       # noqa: E402
from models import xlstm_exog as XE        # noqa: E402
import lstm_backbone as LB                 # noqa: E402

RES = ROOT / "results"
DATA = ROOT / "data"
PRED = RES / "preds"
META = RES / "meta"

SC_COLS = ["news_log_count", "news_fin_share", "news_sent_net"]
EMB_COLS = [f"news_pc{i+1}" for i in range(4)]
PEOPLE = ["visitor_yoy", "mtr_yoy"]

CELLS = {
    "text_scalar": (SC_COLS, "xLSTM", None),
    "text_emb": (EMB_COLS, "xLSTM", None),
    "visitor_mtr_text": (PEOPLE + EMB_COLS, "xLSTM", None),
    "random4": (EMB_COLS, "xLSTM", "noise"),
    "random6": (PEOPLE + EMB_COLS, "xLSTM", "noise"),
    "LSTM_none": ([], "LSTM", None),
    "LSTM_text_emb": (EMB_COLS, "LSTM", None),
    "none_k6": ([], "xLSTM", None),
    "visitor_mtr_k6": (PEOPLE, "xLSTM", None),
}
CELL_ORDER = ["none", "text_scalar", "random3", "text_emb", "random4",
              "visitor_mtr", "visitor_mtr_text", "random6",
              "LSTM_none", "LSTM_text_emb", "none_k6", "visitor_mtr_k6"]
LABEL = {
    "none": "baseline, no exogenous (Stage-1)",
    "text_scalar": "news dictionary/scalar (3)",
    "random3": "3 noise columns (Stage-1 control)",
    "text_emb": "news PCA embedding (4)",
    "random4": "4 noise columns (width-4 control)",
    "visitor_mtr": "visitor + MTR (2), Stage-1",
    "visitor_mtr_text": "visitor + MTR + news PCA (6)",
    "random6": "6 noise columns (width-6 control)",
    "LSTM_none": "plain nn.LSTM backbone, no exogenous",
    "LSTM_text_emb": "plain nn.LSTM backbone + news PCA (4)",
    "none_k6": "baseline, validation = last 6 legal origins",
    "visitor_mtr_k6": "visitor + MTR, validation = last 6 legal origins",
}
K6_CELLS = {"none_k6", "visitor_mtr_k6"}


def load_all():
    feat = pd.read_csv(STAGE0 / "data" / "features_long.csv",
                       dtype={"series": str, "period": str})
    wide = pd.read_csv(STAGE0 / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    scaling = pd.read_csv(STAGE0 / "data" / "scaling.csv", dtype={"series": str})
    exog = XE.load_exog(STAGE0 / "data" / "exog_wide.csv")
    news = pd.read_csv(DATA / "news_text_wide.csv", index_col=0, dtype={0: str})
    news.index = news.index.astype(str)
    combined = pd.concat([exog, news], axis=1)
    return feat, wide, scaling, combined


def noise_block(index, cols, seed=12345):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((len(index), len(cols))).astype(np.float32),
                        index=index, columns=list(cols))


def fit_one(feat, wide, scaling, exog, cell, seed, epochs):
    cols, backbone, mode = CELLS[cell]
    k6 = cell in K6_CELLS
    tag = f"{cell}_s{seed}"
    print("\n" + "=" * 78, flush=True)
    print(f"  tag={tag}  cell={cell} ({LABEL[cell]})  backbone={backbone}  "
          f"seed={seed}  cols={len(cols)}  k6={k6}", flush=True)
    print("=" * 78, flush=True)

    t0 = time.time()
    if not cols and backbone == "xLSTM":
        df, meta = X.run(feat, wide, scaling, epochs=epochs, tag=tag,
                         purge_val=True, verbose=True, seed=seed,
                         val_tail=(6 if k6 else None))
        meta = dict(meta)
    else:
        block = exog
        if mode == "noise":
            block = noise_block(exog.index, cols)
        n_numeric = len(H.FEATURE_COLS)
        n_exog = len(cols)

        def factory(n_series, n_feat_total, _bb=backbone, _nn=n_numeric, _ne=n_exog):
            if _bb == "LSTM":
                if _ne == 0:
                    return LB.LSTMForecaster(n_series, n_feat_total)
                return LB.LSTMForecasterFiLM(n_series, n_feat_total, _nn, _ne)
            if _ne == 0:
                return X.XLSTMForecaster(n_series, n_feat_total,
                                         use_anchor=True, use_calendar=False)
            return XE.XLSTMForecasterFiLM(n_series, n_feat_total, _nn, _ne,
                                          use_anchor=True, use_calendar=False)

        periods, series, F, Y, KF, scaler = XE.build_tensors_exog(
            feat, block, cols, verbose=True)
        model, meta = X.train_model(F, Y, KF, periods, series, H.TRAIN_END,
                                   epochs=epochs, verbose=True, purge_val=True,
                                   model_factory=factory, seed=seed,
                                   val_tail=(6 if k6 else None))
        meta = dict(meta)
        meta["exog_scaler"] = scaler
        if hasattr(model, "inp_exog"):
            meta["exog_weight_norm"] = float(model.inp_exog.weight.detach().norm().cpu())

        rows = []
        for o in H.ORIGINS:
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
        df = pd.DataFrame(rows, columns=["model", "origin", "h", "series",
                                        "y_true", "y_pred"])

    meta.update({"tag": tag, "cell": cell, "seed": seed, "backbone": backbone,
                 "label": LABEL[cell], "n_exog": len(cols),
                 "exog_cols": list(cols), "val_tail": (6 if k6 else None),
                 "train_sec": round(time.time() - t0, 1)})
    df["cell"] = cell
    df["seed"] = seed
    df["backbone"] = backbone
    df.to_csv(PRED / f"{tag}.csv", index=False)
    (META / f"{tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  saved {tag}  ({len(df):,} rows, {meta['train_sec']}s)", flush=True)
    return df, meta


def combine(naive_mae, extra_pred_dirs):
    frames = []
    for d in [*extra_pred_dirs, PRED]:
        for f in sorted(Path(d).glob("*.csv")):
            x = pd.read_csv(f, dtype={"series": str, "origin": str})
            if "cell" not in x.columns:
                continue
            frames.append(x)
    if not frames:
        print("nothing to combine")
        return 1
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(RES / "stage3_predictions_all.csv", index=False)

    rows = []
    for tag, g in pred.groupby("model", sort=False):
        one = H.metrics_from_frame(g, naive_mae)
        for _, r in one.iterrows():
            rows.append({"tag": tag, "cell": r["model"] if False else g["cell"].iloc[0],
                         "backbone": g["backbone"].iloc[0],
                         "seed": int(g["seed"].iloc[0]), "h": r["h"],
                         "MASE": r["MASE"], "WAPE": r["WAPE"],
                         "bias_pct": r["bias_pct"], "n": r["n_points"]})
    m = pd.DataFrame(rows)
    m.to_csv(RES / "stage3_metrics_by_h.csv", index=False)

    print("\n" + "=" * 96)
    print("  MASE by cell x horizon (mean over seeds; lower is better)")
    print("=" * 96)
    piv = m.pivot_table(index="cell", columns="h", values="MASE", aggfunc="mean")
    cols = [c for c in piv.columns if str(c) in {"1", "3", "6", "12", "ALL"}]
    cols = sorted(cols, key=lambda c: 99 if str(c) == "ALL" else int(c))
    piv = piv.reindex([c for c in CELL_ORDER if c in piv.index])
    print(piv[cols].round(5).to_string())

    print("\n  --- ALL horizon, per seed ---")
    aw = m[m["h"].astype(str) == "ALL"].pivot_table(
        index="cell", columns="seed", values="MASE", aggfunc="mean")
    aw = aw.reindex([c for c in CELL_ORDER if c in aw.index])
    print(aw.round(5).to_string())

    # ---- width-matched pairs ------------------------------------------------
    print("\n" + "=" * 96)
    print("  WIDTH-MATCHED PAIRS  (real minus control; negative = the real columns help)")
    print("=" * 96)
    pairs = [("text_scalar", "random3"), ("text_emb", "random4"),
             ("visitor_mtr_text", "random6"), ("visitor_mtr", "none"),
             ("text_emb", "text_scalar"), ("LSTM_text_emb", "LSTM_none"),
             ("LSTM_none", "none"), ("visitor_mtr_k6", "none_k6"),
             ("none_k6", "none")]
    rows = []
    for a, b in pairs:
        if a not in aw.index or b not in aw.index:
            continue
        va = aw.loc[a].mean()
        vb = aw.loc[b].mean()
        rows.append({"real": a, "control": b, "MASE_real": round(va, 5),
                     "MASE_control": round(vb, 5),
                     "delta": round(va - vb, 5),
                     "pct": round(100 * (va / vb - 1), 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- DM: each cell vs none, and vs its width-matched control -----------
    print("\n" + "=" * 96)
    print("  DIEBOLD-MARIANO (power=2, pooled; dm > 0 => A worse => B better)")
    print("=" * 96)
    dm_rows = []
    refs = {"text_scalar": "random3", "text_emb": "random4",
            "visitor_mtr_text": "random6", "LSTM_none": "none",
            "LSTM_text_emb": "LSTM_none", "none_k6": "none",
            "visitor_mtr_k6": "none_k6"}
    for cell in CELL_ORDER:
        if cell == "none":
            continue
        for ref in ["none"] + ([refs[cell]] if cell in refs else []):
            if ref == cell or ref not in set(pred["cell"].unique()):
                continue
            for h in H.HORIZONS:
                a = pred[(pred["cell"] == ref) & (pred["h"] == h)]
                b = pred[(pred["cell"] == cell) & (pred["h"] == h)]
                if a.empty or b.empty:
                    continue
                ea = (a["y_true"] - a["y_pred"]).groupby([a["origin"], a["series"]]).mean()
                eb = (b["y_true"] - b["y_pred"]).groupby([b["origin"], b["series"]]).mean()
                idx = ea.index.intersection(eb.index)
                if len(idx) < 10:
                    continue
                r = H.dm_test(ea.loc[idx].to_numpy(), eb.loc[idx].to_numpy(), h=h, power=2)
                r.update({"A": ref, "B": cell, "h": h})
                dm_rows.append(r)
    dm = pd.DataFrame(dm_rows)
    if not dm.empty:
        dm = dm[["A", "B", "h", "n", "dm", "p_value", "better"]]
        dm.to_csv(RES / "stage3_dm.csv", index=False)
        sig = dm[dm["p_value"] < 0.05]
        print(f"  {len(dm)} tests, {len(sig)} significant at 5%")
        if len(sig):
            print(sig.to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--only", default=None)
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()

    PRED.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    feat, wide, scaling, exog = load_all()
    naive_mae = H.seasonal_naive_mae_train(wide)

    if args.only:
        cell, seed_s = args.only.split(":")
        tag = f"{cell}_s{seed_s}"
        if (PRED / f"{tag}.csv").exists():
            print(f"  [skip] {tag}")
            return 0
        fit_one(feat, wide, scaling, exog, cell, int(seed_s), args.epochs)
        return 0

    if not args.combine_only:
        t0 = time.time()
        for cell in args.cells:
            for seed in args.seeds:
                tag = f"{cell}_s{seed}"
                if (PRED / f"{tag}.csv").exists():
                    print(f"  [skip] {tag} already done", flush=True)
                    continue
                fit_one(feat, wide, scaling, exog, cell, seed, args.epochs)
        print(f"\n  fitting wall time {time.time()-t0:.0f}s", flush=True)

    # Stage-1 predictions for the "none", "random3" and "visitor_mtr" reference
    # cells, reshaped into this file's schema so the comparison table is complete.
    s1 = STAGE0 / "results" / "exog_preds_z"
    bridged = RES / "preds_from_stage1"
    bridged.mkdir(parents=True, exist_ok=True)
    alias = {"xLSTM_none_s": "none", "xLSTM_random3_s": "random3",
             "xLSTM_visitor_mtr_s": "visitor_mtr", "xLSTM_random_s": "random",
             "xLSTM_visitor_s": "visitor", "xLSTM_rain_temp_s": "rain_temp",
             "xLSTM_random2_s": "random2"}
    for f in sorted(s1.glob("*.csv")):
        stem = f.stem
        for pre, cell in alias.items():
            if stem.startswith(pre):
                seed = stem[len(pre):]
                out = bridged / f"{cell}_s{seed}.csv"
                if out.exists():
                    break
                d = pd.read_csv(f, dtype={"series": str, "origin": str})
                d["cell"] = cell
                d["seed"] = int(seed)
                d["backbone"] = "xLSTM"
                d.to_csv(out, index=False)
                break
    return combine(naive_mae, [bridged])


if __name__ == "__main__":
    raise SystemExit(main())
