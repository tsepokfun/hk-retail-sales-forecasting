"""Goal-round audit: recompute the headline MASE_ALL / WAPE_ALL figures that
FINAL_REPORT.md quotes, straight from the stored per-point prediction csv files,
using the LOCKED harness (stage0/eval/harness.py).  Read-only: writes nothing.

Purpose: establish a verified anchor table before any new reproduction runs, so
that new numbers can be compared against the same denominator and the same
MASE implementation.

Run:  stage0\\.venv\\Scripts\\python.exe stage0\\audit_existing_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "eval"))
import harness as H  # noqa: E402

RES = ROOT / "results"


def _metrics(df: pd.DataFrame, naive_mae: pd.Series) -> pd.DataFrame:
    """metrics_from_frame expects model/origin/h/series/y_true/y_pred."""
    keep = ["model", "origin", "h", "series", "y_true", "y_pred"]
    return H.metrics_from_frame(df[keep].copy(), naive_mae)


def _ens3(paths: list[Path], label: str) -> pd.DataFrame:
    """Average y_pred across seed files, aligned on (origin, h, series)."""
    frames = []
    for p in paths:
        d = pd.read_csv(p, dtype={"series": str, "origin": str})
        frames.append(d)
    base = frames[0][["origin", "h", "series", "y_true"]].copy()
    stacked = pd.concat(
        [f[["origin", "h", "series", "y_pred"]] for f in frames], ignore_index=True
    )
    mean = stacked.groupby(["origin", "h", "series"], as_index=False)["y_pred"].mean()
    out = base.merge(mean, on=["origin", "h", "series"], how="left")
    out["model"] = label
    return out


def main() -> int:
    wide = H.load_panel()
    naive_mae = H.seasonal_naive_mae_train(wide)
    print(f"panel: {wide.shape[0]} months x {wide.shape[1]} series")
    print(f"naive_mae series: n={len(naive_mae)}  head={naive_mae.head(3).round(3).to_dict()}\n")

    # ---- 1. the 7-model table in predictions.csv -------------------------
    p = RES / "predictions.csv"
    d = pd.read_csv(p, dtype={"series": str, "origin": str})
    print(f"[1] {p.name}  rows={len(d)}  models={sorted(d['model'].unique())}")
    print(_metrics(d, naive_mae).to_string(index=False), "\n")

    # ---- 2. the ensemble ladder (audit_ladder_predictions.csv) -----------
    p = RES / "audit_ladder_predictions.csv"
    if p.exists():
        d = pd.read_csv(p, dtype={"series": str, "origin": str})
        print(f"[2] {p.name}  rows={len(d)}  models={sorted(d['model'].unique())}")
        print(_metrics(d, naive_mae).to_string(index=False), "\n")

    # ---- 3. stage_h ens5 -------------------------------------------------
    p = RES / "stage_h_ens5_preds.csv"
    if p.exists():
        d = pd.read_csv(p, dtype={"series": str, "origin": str})
        print(f"[3] {p.name}  rows={len(d)}  models={sorted(d['model'].unique())}")
        print(_metrics(d, naive_mae).to_string(index=False), "\n")

    # ---- 4. the SOTA transformer arms: single seed + ENS3 ----------------
    sota = RES / "sota_preds"
    for stem in ["itransformer_anchor", "itransformer", "itransformer_noattn", "timetf_anchor"]:
        files = {s: sota / f"{stem}_s{s}.csv" for s in ("42", "7", "2024")}
        if not all(f.exists() for f in files.values()):
            print(f"[4] {stem}: missing seed files -> skipped")
            continue
        single = pd.read_csv(files["42"], dtype={"series": str, "origin": str})
        m1 = _metrics(single, naive_mae)
        ens = _ens3(list(files.values()), f"{stem}_ENS3")
        m3 = _metrics(ens, naive_mae)
        print(f"[4] {stem}   s42 only:")
        print(m1.to_string(index=False))
        print(f"[4] {stem}   ENS3 (mean of s42/s7/s2024):")
        print(m3.to_string(index=False), "\n")

    # ---- 5. the reference seasonal naive file ----------------------------
    p = RES / "_sota_ref_SeasonalNaive.csv"
    if p.exists():
        d = pd.read_csv(p, dtype={"series": str, "origin": str})
        print(f"[5] {p.name}  rows={len(d)}")
        print(_metrics(d, naive_mae).to_string(index=False), "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
