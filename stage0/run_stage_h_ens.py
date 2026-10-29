"""
Stage H, task A -- INDEPENDENT re-derivation of the seed-ensemble numbers.

Why a separate script
---------------------
The three-seed ensemble (0.94810) and the 3-seed mean-of-per-seed-MASE (0.98355)
are DIFFERENT quantities and the project's write-up has never stored the code
that produced 0.94810.  So this script recomputes it from the raw per-seed
prediction CSVs and prints BOTH numbers side by side, which is the only way to
prove the two are not being confused:

    mean of per-seed MASE_ALL   0.98355   <- per-SEED score, averaged
    MASE of the mean forecast   0.94810   <- per-POINT score of the ensemble

The ensemble can never equal the mean of the per-seed scores, because averaging
forecasts removes independent variance (Jensen: f(x)=x^2 is convex, so
mean(MSE of seeds) >= MSE of mean).

Two averaging spaces are supported, and the difference is reported rather than
hidden:
    'level' : yhat_ens = mean_s(yhat_s)                (arithmetic, HK$m)
    'log'   : yhat_ens = exp(mean_s(log yhat_s)) = geometric mean
              (this is the arithmetic mean in the z-space the model actually
               trains in, since y_z = (log y - mu)/sd and sd is per-series only)

Usage
    .venv\\Scripts\\python.exe run_stage_h_ens.py --tags xLSTM_none_s42 xLSTM_none_s7 xLSTM_none_s2024
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H            # noqa: E402

RES = ROOT / "results"
PRED_DIR = RES / "exog_preds_z"

KEYS = ["origin", "h", "series"]


# ---------------------------------------------------------------------------
def resolve(tag: str, dirs: list[Path]) -> Path:
    """find <tag>.csv in the first directory that has it (tags differ between the
    Stage-1 tree `xLSTM_none_s*` and the Stage-H tree `xLSTM_base_s*`; the
    protocol is identical, only the name changed)."""
    for d in dirs:
        f = d / f"{tag}.csv"
        if f.exists():
            return f
    raise FileNotFoundError(f"{tag}.csv not found in {[str(d) for d in dirs]}")


def load_tags(tags: list[str], dirs: list[Path]) -> pd.DataFrame:
    """wide frame indexed by (origin,h,series) with one column per tag."""
    frames = []
    for t in tags:
        f = resolve(t, dirs)
        d = pd.read_csv(f, dtype={"origin": str, "series": str})
        d = d[KEYS + ["y_true", "y_pred"]].copy()
        d["y_pred"] = d["y_pred"].astype(float)
        # a duplicate (origin,h,series) would silently break the pivot
        assert not d.duplicated(KEYS).any(), f"{t}: duplicate keys"
        frames.append(d.set_index(KEYS))
    out = pd.concat({t: f["y_pred"] for t, f in zip(tags, frames)}, axis=1)
    # y_true must agree across seeds -- it is the same panel target
    ref = frames[0]["y_true"]
    for t, f in zip(tags[1:], frames[1:]):
        d = (f["y_true"] - ref.reindex(f.index)).abs().max()
        if not (d == 0 or np.isnan(d)):
            raise AssertionError(f"{t}: y_true differs from {tags[0]} by {d}")
    out["y_true"] = ref.reindex(out.index)
    return out


def to_frame(wide: pd.DataFrame, ypred: np.ndarray, model: str) -> pd.DataFrame:
    out = wide.reset_index()[KEYS].copy()
    out["model"] = model
    out["y_true"] = wide["y_true"].to_numpy()
    out["y_pred"] = ypred
    return out


def combine(wide: pd.DataFrame, tags: list[str], how: str = "level") -> np.ndarray:
    cols = wide[tags].to_numpy(dtype=float)
    if how == "level":
        return cols.mean(axis=1)
    if how == "log":
        return np.exp(np.log(cols).mean(axis=1))
    raise ValueError(how)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--model-prefix", default="xLSTM_none")
    ap.add_argument("--dirs", nargs="+", default=["exog_preds_z", "stage_h_preds"],
                    help="prediction trees to search for each tag")
    ap.add_argument("--out", default=None, help="write per-horizon table here")
    ap.add_argument("--dump", default=None, help="write ensembled predictions here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    dirs = [RES / d for d in args.dirs]
    wide = load_tags(args.tags, dirs)
    wide = wide.sort_index()
    naive = H.seasonal_naive_mae_train(H.load_panel())
    if not args.quiet:
        print(f"  tags {args.tags}")
        print(f"  aligned grid: {len(wide):,} rows  "
              f"origins {sorted(wide.index.get_level_values('origin').unique())}")
        print(f"  horizons  {sorted(wide.index.get_level_values('h').unique())}   "
              f"series {wide.index.get_level_values('series').nunique()}")

    rows, frames = [], []
    # ---- per-SEED scores (what 0.98355 is) -------------------------------
    for t in args.tags:
        fr = to_frame(wide, wide[t].to_numpy(), t)
        frames.append(fr)
        m = H.metrics_from_frame(fr, naive)
        rows.append(m.assign(kind="per_seed", seed=t))
    # ---- ENSEMBLE scores (what 0.94810 is) -------------------------------
    for how in ("level", "log"):
        nm = f"{args.model_prefix}_ENS{len(args.tags)}_{how}"
        fr = to_frame(wide, combine(wide, args.tags, how), nm)
        frames.append(fr)
        m = H.metrics_from_frame(fr, naive)
        rows.append(m.assign(kind="ensemble", seed=nm))
    allm = pd.concat(rows, ignore_index=True)

    pred = pd.concat(frames, ignore_index=True)
    if args.dump:
        pred.to_csv(args.dump, index=False)
    if args.out:
        allm.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    if not args.quiet:
        print("\n  per-horizon MASE   (rows = run, cols = h)")
        piv = allm.pivot_table(index=["kind", "seed"], columns="h",
                               values="MASE", sort=False)
        cols = [c for c in piv.columns if c != "ALL"] + ["ALL"]
        print(piv[cols].to_string(float_format=lambda v: f"{v:.5f}"))

        print("\n  per-horizon WAPE")
        pivw = allm.pivot_table(index=["kind", "seed"], columns="h",
                                values="WAPE", sort=False)
        print(pivw[cols].to_string(float_format=lambda v: f"{v:.5f}"))

        per = allm[(allm.kind == "per_seed") & (allm.h == "ALL")]["MASE"]
        ens = allm[(allm.kind == "ensemble") & (allm.h == "ALL")]
        print(f"\n  mean of per-seed MASE_ALL ............ {per.mean():.5f} "
              f"(std {per.std(ddof=1):.5f})")
        for _, r in ens.iterrows():
            print(f"  MASE_ALL of ensemble [{r['seed']}] ... {r['MASE']:.5f}  "
                  f"({100*(r['MASE']-per.mean())/per.mean():+.3f}% vs per-seed mean)")

        # ---- DM: every seed against the ensemble, per horizon -------------
        print("\n  DIEBOLD-MARIANO  seed (A) vs ensemble (B), power=2")
        print("  dm > 0 => seed has higher loss => ensemble better")
        dmrows = []
        ensname = f"{args.model_prefix}_ENS{len(args.tags)}_level"
        for h in sorted(pred["h"].unique()):
            e = pred[(pred.model == ensname) & (pred.h == h)].set_index(KEYS)
            for t in args.tags:
                a = pred[(pred.model == t) & (pred.h == h)].set_index(KEYS)
                idx = a.index.intersection(e.index)
                ea = (a.loc[idx, "y_true"] - a.loc[idx, "y_pred"]).to_numpy()
                eb = (e.loc[idx, "y_true"] - e.loc[idx, "y_pred"]).to_numpy()
                r = H.dm_test(ea, eb, h=h, power=2)
                dmrows.append({"h": h, "A": t, "B": ensname, **r})
            # ensemble vs the MEAN of per-seed?  can't be a pair -- skip
        dm = pd.DataFrame(dmrows)
        print(dm[["h", "A", "dm", "p_value", "better", "n"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
