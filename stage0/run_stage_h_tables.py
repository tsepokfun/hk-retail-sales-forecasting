"""
Stage H -- build the three report tables from whatever fits have landed.

Everything here is post-hoc analysis of results/stage_h_preds/*.csv and
results/stage_h_meta/*.json plus the pre-existing Stage-1 tree
results/exog_preds_z/.  No training happens in this file.

OUTPUT (results/)
    stage_h_tableA_ensemble.csv    3-seed vs 5-seed ensemble, per horizon + ALL
    stage_h_tableA_dm.csv          DM: 5-seed ens vs 3-seed ens, and each seed
    stage_h_tableB_sweep.csv       hyper-parameter grid, one row per config
    stage_h_tableC_horizon.csv     horizon-gate arms x seeds
    stage_h_tableC_dm.csv          DM of each arm's 3-seed ensemble vs `none`'s
    stage_h_costs.csv              per-fit wall/train seconds

Usage
    .venv\\Scripts\\python.exe run_stage_h_tables.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H            # noqa: E402

RES = ROOT / "results"
PRED_DIR = RES / "stage_h_preds"
META_DIR = RES / "stage_h_meta"
STAGE1_DIR = RES / "exog_preds_z"
KEYS = ["origin", "h", "series"]
BASE_MASE = 0.98355          # the number the brief fixes as the reference


# ---------------------------------------------------------------------------
def naive_mae() -> pd.Series:
    return H.seasonal_naive_mae_train(H.load_panel())


def load_pred(tag: str, dirs=(STAGE1_DIR, PRED_DIR), name: str | None = None) -> pd.DataFrame:
    for d in dirs:
        f = d / f"{tag}.csv"
        if f.exists():
            df = pd.read_csv(f, dtype={"origin": str, "series": str})
            df["model"] = name or tag
            return df[["model", "origin", "h", "series", "y_true", "y_pred"]]
    raise FileNotFoundError(tag)


def ens(preds: list[pd.DataFrame], name: str) -> pd.DataFrame:
    """level-space mean of several models' forecasts on the common grid."""
    key = KEYS
    yt = preds[0].set_index(key)["y_true"]
    acc = None
    for p in preds:
        s = p.set_index(key)["y_pred"].reindex(yt.index)
        acc = s if acc is None else acc + s
    out = (acc / len(preds)).rename("y_pred").to_frame()
    out["y_true"] = yt
    out = out.reset_index()
    out["model"] = name
    return out[["model"] + key + ["y_true", "y_pred"]]


def metrics(df: pd.DataFrame, naive, cols=("MASE", "WAPE", "bias_pct")) -> pd.DataFrame:
    m = H.metrics_from_frame(df, naive)
    m["h"] = m["h"].astype(str)
    return m[["model", "h", "n_points", *cols]]


def dm_pair(a: pd.DataFrame, b: pd.DataFrame, power: int = 2,
            pooled_lag: int = 11) -> pd.DataFrame:
    """DM per horizon (NW lag h-1, harness convention) plus one POOLED row.

    The pooled row is explicitly labelled optimistic: the 4 horizons overlap in
    target months, so it uses lag = max(HORIZONS)-1 = 11 and is reported only as
    a companion to MASE_ALL, never as the primary test.
    """
    rows = []
    ka = a.set_index(KEYS)
    kb = b.set_index(KEYS)
    idx = ka.index.intersection(kb.index)
    for h in sorted(a["h"].unique()):
        ia = [i for i in idx if i[1] == h]
        ea = (ka.loc[ia, "y_true"] - ka.loc[ia, "y_pred"]).to_numpy()
        eb = (kb.loc[ia, "y_true"] - kb.loc[ia, "y_pred"]).to_numpy()
        for pw in (2, 1):
            r = H.dm_test(ea, eb, h=int(h), power=pw)
            rows.append({"scope": f"h={h}", "power": pw, "n": r["n"], "dm": r["dm"],
                         "p": r["p_value"], "better": r["better"]})
    ea = (ka.loc[idx, "y_true"] - ka.loc[idx, "y_pred"]).to_numpy()
    eb = (kb.loc[idx, "y_true"] - kb.loc[idx, "y_pred"]).to_numpy()
    for pw in (2, 1):
        r = H.dm_test(ea, eb, h=pooled_lag, power=pw)
        rows.append({"scope": "POOLED(optimistic)", "power": pw, "n": r["n"],
                     "dm": r["dm"], "p": r["p_value"], "better": r["better"]})
    return pd.DataFrame(rows)


def load_metas(prefix: str | None = None) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(META_DIR / "*.json"))):
        m = json.loads(Path(f).read_text(encoding="utf-8"))
        m["_file"] = Path(f).name
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def table_A(naive):
    tags3 = ["xLSTM_none_s42", "xLSTM_none_s7", "xLSTM_none_s2024"]
    p3 = [load_pred(t) for t in tags3]
    # stage-H re-fit of seed 42 exists too: use it only as a determinism proof
    out, rows = [], []
    for p, t in zip(p3, tags3):
        rows.append(metrics(p, naive).assign(kind="per_seed"))
    e3 = ens(p3, "ENS3_level")
    rows.append(metrics(e3, naive).assign(kind="ENS3_level"))
    e3l = ens([p.assign(y_pred=np.log(p.y_pred)) for p in p3], "ENS3_log")
    e3l["y_pred"] = np.exp(e3l["y_pred"])
    rows.append(metrics(e3l, naive).assign(kind="ENS3_log"))

    tags5 = tags3 + ["xLSTM_base_s1", "xLSTM_base_s1234"]
    try:
        p5 = [load_pred(t) for t in tags5]
        for p, t in zip(p5[3:], tags5[3:]):
            rows.append(metrics(p, naive).assign(kind="per_seed"))
        e5 = ens(p5, "ENS5_level")
        rows.append(metrics(e5, naive).assign(kind="ENS5_level"))
        e5l = ens([p.assign(y_pred=np.log(p.y_pred)) for p in p5], "ENS5_log")
        e5l["y_pred"] = np.exp(e5l["y_pred"])
        rows.append(metrics(e5l, naive).assign(kind="ENS5_log"))
    except FileNotFoundError as e:
        print(f"  [tableA] missing {e}; 5-seed part skipped")
        e5 = None
    A = pd.concat(rows, ignore_index=True)

    dmA = []
    if e5 is not None:
        d = dm_pair(e5, e3)
        dmA.append(d.assign(comparison="ENS5 vs ENS3"))
        for t, p in zip(tags3 + tags5[3:], p5):
            d = dm_pair(p, e5)
            dmA.append(d.assign(comparison=f"seed {t} vs ENS5"))
    dmA = pd.concat(dmA, ignore_index=True) if dmA else pd.DataFrame()
    return A, dmA, e3, e5


def table_B(naive, metas: pd.DataFrame):
    rows = []
    for _, m in metas.iterrows():
        cfg = m.get("config")
        if not str(cfg).startswith(("base", "ctx", "emb", "blk", "dp", "wd", "lr", "b")):
            continue
        if str(cfg).startswith("hz"):
            continue
        tag = m["tag"]
        try:
            p = load_pred(tag)
        except FileNotFoundError:
            continue
        mm = metrics(p, naive)
        allrow = mm[mm.h == "ALL"].iloc[0]
        rec = {"config": cfg, "seed": int(m["seed"]), "tag": tag,
               "n_train": int(m["n_train_samples"]), "n_val": int(m["n_val_samples"]),
               "n_params": int(m["n_params"]),
               "val_loss": float(m["best_val_loss"]),
               "best_epoch": int(m["best_epoch"]),
               "MASE_ALL": float(allrow["MASE"]),
               "WAPE_ALL": float(allrow["WAPE"]),
               "delta_vs_0.98355": float(allrow["MASE"]) - BASE_MASE,
               "pct_vs_0.98355": 100 * (float(allrow["MASE"]) - BASE_MASE) / BASE_MASE,
               "val_origins": "|".join(m["val_origins"]),
               "max_val_target_period": m["max_val_target_period"],
               "train_sec": m.get("train_sec"), "wall_sec": m.get("wall_sec"),
               "hp": dict(m.get("hp") or {})}
        for h in ("1", "3", "6", "12"):
            rec[f"MASE_h{h}"] = float(mm[mm.h == h].iloc[0]["MASE"])
        rows.append(rec)
    B = pd.DataFrame(rows)
    if B.empty:
        return B
    # aggregate over seeds per config
    agg = B.groupby("config").agg(n_seeds=("seed", "nunique"),
                                  MASE_mean=("MASE_ALL", "mean"),
                                  MASE_std=("MASE_ALL", lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan),
                                  val_loss=("val_loss", "mean"),
                                  n_train=("n_train", "first"),
                                  n_params=("n_params", "first")).reset_index()
    agg["delta_vs_0.98355"] = agg["MASE_mean"] - BASE_MASE
    agg["pct_vs_0.98355"] = 100 * agg["delta_vs_0.98355"] / BASE_MASE
    return B.merge(agg, on="config", suffixes=("", "_agg"))


def table_C(naive, metas: pd.DataFrame):
    """Horizon-gate arms.

    The matched baseline is the Stage-1 `none` trio (seeds 42 / 7 / 2024), NOT a
    fresh run: it is the same model, same protocol, same three seeds, so the
    per-seed pairing below is exact.  Rerunning `none` would add GPU minutes and
    produce nothing but a determinism check we already did (base_s42 ==
    xLSTM_none_s42 to 5 decimals).
    """
    BASE_TAGS = ["xLSTM_none_s42", "xLSTM_none_s7", "xLSTM_none_s2024"]
    cells = [c for c in metas["config"].dropna().unique() if str(c).startswith("hz-")]
    rows, dmrows, per_cell = [], [], {}
    base_seed_mase = {}
    try:
        bpreds = [load_pred(t, name=f"none_s{t.split('_s')[-1]}") for t in BASE_TAGS]
        per_cell["hz-none"] = ens(bpreds, "ENS3_none")
        for t, p in zip(BASE_TAGS, bpreds):
            m = metrics(p, naive)
            a = m[m.h == "ALL"].iloc[0]
            seed = int(t.split("_s")[-1])
            base_seed_mase[seed] = float(a["MASE"])
            rec = {"cell": "none(Stage-1)", "seed": seed, "n_params": 20496,
                   "n_train": 4186, "val_loss": np.nan, "MASE_ALL": float(a["MASE"]),
                   "WAPE_ALL": float(a["WAPE"]), "delta_vs_0.98355": float(a["MASE"]) - BASE_MASE,
                   "open_heads": "[]", "closed_heads": "[]", "exog_cols": "",
                   "max_val_target_period": "—", "paired_delta_vs_none": 0.0}
            for h in ("1", "3", "6", "12"):
                rec[f"MASE_h{h}"] = float(m[m.h == h].iloc[0]["MASE"])
            rows.append(rec)
    except FileNotFoundError as e:
        print(f"  [tableC] no Stage-1 none baseline: {e}")
    for c in sorted(cells):
        sub = metas[metas["config"] == c].sort_values("seed")
        preds = []
        for _, m in sub.iterrows():
            try:
                p = load_pred(m["tag"])
            except FileNotFoundError:
                continue
            preds.append(p)
            mm = metrics(p, naive)
            allrow = mm[mm.h == "ALL"].iloc[0]
            rec = {"cell": c.replace("hz-", ""), "seed": int(m["seed"]),
                   "n_params": int(m["n_params"]), "n_train": int(m["n_train_samples"]),
                   "val_loss": float(m["best_val_loss"]),
                   "MASE_ALL": float(allrow["MASE"]),
                   "WAPE_ALL": float(allrow["WAPE"]),
                   "delta_vs_0.98355": float(allrow["MASE"]) - BASE_MASE,
                   "paired_delta_vs_none": (float(allrow["MASE"])
                                            - base_seed_mase.get(int(m["seed"]), np.nan)),
                   "open_heads": str(m.get("hz_open_heads")),
                   "closed_heads": str(m.get("hz_closed_heads")),
                   "exog_cols": ",".join(m.get("exog_cols") or []),
                   "max_val_target_period": m["max_val_target_period"],
                   "train_sec": m.get("train_sec"), "wall_sec": m.get("wall_sec")}
            for h in ("1", "3", "6", "12"):
                rec[f"MASE_h{h}"] = float(mm[mm.h == h].iloc[0]["MASE"])
            rows.append(rec)
        if preds:
            per_cell[c] = ens(preds, f"ENS{len(preds)}_{c}")
    C = pd.DataFrame(rows)
    # DM: each arm's ensemble vs the `none` ensemble
    if "hz-none" in per_cell:
        b = per_cell["hz-none"]
        for c, e in per_cell.items():
            if c == "hz-none":
                continue
            dmrows.append(dm_pair(e, b).assign(comparison=f"{c} vs hz-none"))
        dmC = pd.concat(dmrows, ignore_index=True) if dmrows else pd.DataFrame()
    else:
        dmC = pd.DataFrame()
    return C, dmC, per_cell


def costs() -> pd.DataFrame:
    rows = []
    log = RES / "stage_h_campaign.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            m = re.search(r"attempt=(\d+) rc=(-?\d+) wall=([\d.]+)s", parts[2])
            if m:
                rows.append({"spec": parts[1], "attempt": int(m.group(1)),
                             "rc": int(m.group(2)), "wall_sec": float(m.group(3)),
                             "verdict": parts[3]})
    return pd.DataFrame(rows)


def md_table(header: list[str], rows: list[list], align: str | None = None) -> str:
    a = align or ("l" + "r" * (len(header) - 1))
    sep = {"l": ":---", "r": "---:"}
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(sep[c] for c in a) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def f5(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if v != v else f"{v:.5f}"


def f3(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if v != v else f"{v:.3f}"


def pct(v, ref: float) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{100 * (v - ref) / ref:+.2f}%"


def emit_markdown(A, dmA, B, C, dmC, K, per_cell, naive) -> str:
    L = []
    add = L.append
    add("<!-- generated by run_stage_h_tables.py -- do not hand-edit numbers -->")
    add("")
    add("## 表 A 原始輸出")
    add("")
    hdr = ["配置 / 種子", "h=1", "h=3", "h=6", "h=12", "MASE_ALL", "WAPE_ALL", "bias%"]
    rows = []
    wape = A.pivot_table(index=["kind", "model"], columns="h", values="WAPE", sort=False)
    bias = A.pivot_table(index=["kind", "model"], columns="h", values="bias_pct", sort=False)
    for (kind, model), g in A.groupby(["kind", "model"], sort=False):
        g = g.set_index("h")
        rows.append([f"{kind} · `{model}`"] +
                    [f5(g.loc[h, "MASE"]) for h in ("1", "3", "6", "12", "ALL")] +
                    [f5(wape.loc[(kind, model), "ALL"]), f3(bias.loc[(kind, model), "ALL"])])
    add(md_table(hdr, rows))
    add("")
    if len(dmA):
        add("**DM（power=2；`better` = 損失較低的一方）**")
        add("")
        rows = []
        for r in dmA.itertuples(index=False):
            rows.append([r.comparison, r.scope, r.power, r.n,
                         f"{r.dm:+.4f}", f"{r.p:.6f}", r.better])
        add(md_table(["比較 (A vs B)", "範圍", "power", "n", "DM", "p", "較優"], rows))
        add("")
    add("## 表 B 原始輸出")
    add("")
    if len(B):
        rows = []
        for r in B.itertuples(index=False):
            rows.append([r.config, r.seed, r.n_train, r.n_params, f5(r.val_loss),
                         f5(r.MASE_h1), f5(r.MASE_h3), f5(r.MASE_h6), f5(r.MASE_h12),
                         f5(r.MASE_ALL), pct(r.MASE_ALL, 0.98355),
                         pct(r.MASE_ALL, 1.01743)])
        add(md_table(["config", "seed", "n_train", "參數量", "val_loss", "h=1", "h=3",
                      "h=6", "h=12", "MASE_ALL", "vs 0.98355", "vs s42 1.01743"], rows))
        add("")
        add("per-config 彙總（以驗證損失排序 —— 挑選依據**不是** 2024 測試集）")
        add("")
        rows = []
        for cfg, g in B.groupby("config", sort=False):
            v = g["MASE_ALL"].to_numpy(float)
            rows.append([cfg, int(g["seed"].nunique()), int(g["n_train"].iloc[0]),
                         int(g["n_params"].iloc[0]), f5(g["val_loss"].mean()),
                         f5(g["MASE_h1"].mean()), f5(g["MASE_h3"].mean()),
                         f5(g["MASE_h6"].mean()), f5(g["MASE_h12"].mean()),
                         f5(v.mean()),
                         f5(np.std(v, ddof=1)) if len(v) > 1 else "—",
                         pct(v.mean(), 0.98355)])
        add(md_table(["config", "種子數", "n_train", "參數量", "val_loss(mean)", "h=1",
                      "h=3", "h=6", "h=12", "MASE_ALL(mean)", "std", "vs 0.98355"], rows))
        add("")
    add("## 表 C 原始輸出")
    add("")
    if len(C):
        rows = []
        for r in C.itertuples(index=False):
            d = getattr(r, "paired_delta_vs_none", np.nan)
            rows.append([r.cell, r.seed, r.n_params, f5(r.val_loss), f5(r.MASE_h1),
                         f5(r.MASE_h3), f5(r.MASE_h6), f5(r.MASE_h12), f5(r.MASE_ALL),
                         pct(r.MASE_ALL, 0.98355),
                         "—" if d != d else f"{d:+.5f}"])
        add(md_table(["cell", "seed", "參數量", "val_loss", "h=1", "h=3", "h=6", "h=12",
                      "MASE_ALL", "vs 0.98355", "同種子配對差 vs none"], rows))
        add("")
        add("per-cell 平均 ± 標準差（逐種子 MASE_ALL）")
        add("")
        rows = []
        for cell, g in C.groupby("cell", sort=False):
            v = g["MASE_ALL"].to_numpy(float)
            rows.append([cell, len(v), f5(v.mean()),
                         f5(np.std(v, ddof=1)) if len(v) > 1 else "—",
                         f5(g["MASE_h1"].mean()), f5(g["MASE_h3"].mean()),
                         f5(g["MASE_h6"].mean()), f5(g["MASE_h12"].mean()),
                         pct(v.mean(), 0.98355)])
        add(md_table(["cell", "種子數", "MASE_ALL 平均", "std", "h=1", "h=3", "h=6",
                      "h=12", "vs 0.98355"], rows))
        add("")
    if len(dmC):
        add("**DM：各臂 3 種子集成 vs `none` 3 種子集成**")
        add("")
        rows = []
        for r in dmC.itertuples(index=False):
            rows.append([r.comparison, r.scope, r.power, r.n, f"{r.dm:+.4f}",
                         f"{r.p:.6f}", r.better])
        add(md_table(["比較", "範圍", "power", "n", "DM", "p", "較優"], rows))
        add("")
    if per_cell:
        add("集成後逐視界 MASE（level-space 平均）")
        add("")
        rows = []
        naive_ref = naive
        for name, p in per_cell.items():
            m = H.metrics_from_frame(p, naive_ref)
            m["h"] = m["h"].astype(str)
            m = m.set_index("h")
            rows.append([name] + [f5(m.loc[h, "MASE"]) for h in ("1", "3", "6", "12", "ALL")])
        add(md_table(["集成", "h=1", "h=3", "h=6", "h=12", "MASE_ALL"], rows))
        add("")
    add("## 成本")
    add("")
    if len(K):
        ok = K[K.verdict == "OK"]
        add(md_table(["項目", "值"], [
            ["成功 fit 數", int(len(ok))],
            ["失敗/重試次數", int(len(K) - len(ok))],
            ["總 wall time", f"{ok.wall_sec.sum() / 60:.1f} min"],
            ["單次平均 wall", f"{ok.wall_sec.mean():.1f} s"],
        ]))
    return "\n".join(L) + "\n"


def val_vs_test(naive, B: pd.DataFrame) -> pd.DataFrame:
    """Does validation loss rank the configurations the same way the test set does?

    This is the question that decides whether the whole sweep is informative.  It
    is answered twice, because they are different questions:
      * across CONFIGURATIONS at a fixed seed 42  (n=7)
      * across SEEDS at a fixed configuration     (n=5, val loss taken from the
        Stage-1 metas for seeds 7 / 2024 which were never re-run here)
    A non-positive Spearman rho means the selector carries no information, so any
    "best config" claim is noise.
    """
    from scipy import stats as st
    rows = []
    s42 = B[B.seed == 42][["config", "val_loss", "MASE_ALL"]].dropna()
    if len(s42) >= 4:
        rho, p = st.spearmanr(s42.val_loss, s42.MASE_ALL)
        rows.append({"question": "across configs (seed 42)", "n": len(s42),
                     "spearman_rho": round(float(rho), 4), "p": round(float(p), 4),
                     "members": ",".join(s42.config)})
    seeds: dict[int, tuple[float, float]] = {}
    for r in B[B.config == "base"].itertuples(index=False):
        seeds[int(r.seed)] = (float(r.val_loss), float(r.MASE_ALL))
    # seeds 7 / 2024 were never re-run under Stage H: read their val loss from the
    # Stage-1 metas and their test MASE from the Stage-1 predictions, so the
    # seed-level correlation has 5 points instead of 3.
    for seed, tag in ((7, "xLSTM_none_s7"), (2024, "xLSTM_none_s2024")):
        f = RES / "exog_meta_z" / f"{tag}.json"
        if seed in seeds or not f.exists():
            continue
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
            mm = metrics(load_pred(tag), naive)
            seeds[seed] = (float(m["best_val_loss"]),
                           float(mm[mm.h == "ALL"].iloc[0]["MASE"]))
        except FileNotFoundError:
            pass
    pts = [(v, t) for v, t in seeds.values() if t is not None]
    if len(pts) >= 4:
        rho, p = st.spearmanr([a for a, _ in pts], [b for _, b in pts])
        rows.append({"question": "across seeds (base config)", "n": len(pts),
                     "spearman_rho": round(float(rho), 4), "p": round(float(p), 4),
                     "members": ",".join(str(k) for k in seeds)})
    return pd.DataFrame(rows)


def table_B_dm(naive, metas: pd.DataFrame):
    """ENS3(config) vs ENS3(base) on the SAME three seeds 42/7/2024.

    A single-seed config number cannot carry a claim (per-seed std is 0.039), so
    every configuration that got its 3 seeds is tested here as a 3-seed ensemble
    against the 3-seed `none` ensemble, which is the like-for-like comparison.
    """
    SEEDS = [42, 7, 2024]
    ens_by_cfg, rows = {}, []
    base_preds = []
    for s in SEEDS:
        tag = f"xLSTM_none_s{s}" if s != 42 else "xLSTM_none_s42"
        try:
            base_preds.append(load_pred(f"xLSTM_none_s{s}"))
        except FileNotFoundError:
            return pd.DataFrame()
    ens_by_cfg["base"] = ens(base_preds, "ENS3_base")
    for cfg in sorted(metas["config"].dropna().unique()):
        if not str(cfg).startswith(("ctx", "emb", "blk", "dp", "wd", "lr", "b")) \
                or str(cfg).startswith("hz"):
            continue
        preds = []
        ok = True
        for s in SEEDS:
            try:
                preds.append(load_pred(f"xLSTM_{cfg}_s{s}"))
            except FileNotFoundError:
                ok = False
                break
        if ok:
            ens_by_cfg[cfg] = ens(preds, f"ENS3_{cfg}")
    b = ens_by_cfg["base"]
    for cfg, e in ens_by_cfg.items():
        if cfg == "base":
            continue
        mb = metrics(e, naive).set_index("h")["MASE"]
        mB = metrics(b, naive).set_index("h")["MASE"]
        rows.append(dm_pair(e, b).assign(comparison=f"ENS3_{cfg} vs ENS3_base",
                                         MASE_ALL=float(mb.loc["ALL"]),
                                         MASE_ALL_base=float(mB.loc["ALL"])))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    naive = naive_mae()
    metas = load_metas()

    A, dmA, e3, e5 = table_A(naive)
    A.to_csv(RES / "stage_h_tableA_ensemble.csv", index=False)
    dmA.to_csv(RES / "stage_h_tableA_dm.csv", index=False)

    B = table_B(naive, metas)
    B.to_csv(RES / "stage_h_tableB_sweep.csv", index=False)

    C, dmC, per_cell = table_C(naive, metas)
    C.to_csv(RES / "stage_h_tableC_horizon.csv", index=False)
    dmC.to_csv(RES / "stage_h_tableC_dm.csv", index=False)

    K = costs()
    K.to_csv(RES / "stage_h_costs.csv", index=False)

    V = val_vs_test(naive, B)
    V.to_csv(RES / "stage_h_val_vs_test.csv", index=False)

    BD = table_B_dm(naive, metas)
    BD.to_csv(RES / "stage_h_tableB_dm.csv", index=False)

    md = emit_markdown(A, dmA, B, C, dmC, K, per_cell, naive)
    (RES / "stage_h_tables_generated.md").write_text(md, encoding="utf-8")

    pd.set_option("display.width", 250)
    print("\n=== TABLE A ===")
    print(A.pivot_table(index=["kind", "model"], columns="h", values="MASE",
                        sort=False).to_string(float_format=lambda v: f"{v:.5f}"))
    if len(dmA):
        print(dmA.to_string(index=False))
    print("\n=== TABLE B ===")
    if len(B):
        print(B[["config", "seed", "n_seeds", "n_train", "n_params", "val_loss",
                 "MASE_h1", "MASE_h3", "MASE_h6", "MASE_h12", "MASE_ALL",
                 "pct_vs_0.98355"]].to_string(index=False,
                                              float_format=lambda v: f"{v:.5f}"))
    print("\n=== TABLE C ===")
    if len(C):
        print(C[["cell", "seed", "n_params", "val_loss", "MASE_h1", "MASE_h3",
                 "MASE_h6", "MASE_h12", "MASE_ALL"]].to_string(
                     index=False, float_format=lambda v: f"{v:.5f}"))
        if len(dmC):
            print(dmC.to_string(index=False))
    print("\n=== TABLE B DM (3-seed ensemble vs 3-seed base) ===")
    print(BD.to_string(index=False) if len(BD) else "(needs 3 seeds per config)")
    print("\n=== VAL vs TEST ===")
    print(V.to_string(index=False) if len(V) else "(not enough rows)")
    print("\n=== COSTS ===")
    print(K.to_string(index=False) if len(K) else "(no campaign log)")
    if len(K):
        print(f"  fits={len(K[K.verdict == 'OK'])}  total wall={K.wall_sec.sum()/60:.1f} min")
    print("\nwrote results/stage_h_tables_generated.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
