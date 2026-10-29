"""
Stage H -- the conservative significance tests for the headline claims.

`dm_pair` in run_stage_h_tables.py pools 276 points and therefore treats the 23
retail categories in a given month as independent observations.  They are not:
the whole sector moves with the same macro shock, so pooled DM understates the
variance.  This project already knows that (harness.py's docstring says so), so
the two aggregation levels the harness provides are applied here to the Stage-H
claims that actually matter:

    dm_series_level   one number per SERIES   (N=23, Wilcoxon)
                      -> "is the model better on most retail categories?"
    dm_time_aggregated one number per ORIGIN  (N=12, Wilcoxon)
                      -> "is the model better in the average month?"

Everything is compared on 3-seed level-space ensembles, matching the reported
numbers.  No training happens here.

Usage
    .venv\\Scripts\\python.exe run_stage_h_robust.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H                                     # noqa: E402
from run_stage_h_tables import metrics, naive_mae, ens, load_pred  # noqa: E402

RES = ROOT / "results"
SEEDS = [42, 7, 2024]

# label -> tag pattern ({} -> seed)
CLAIMS = {
    "ENS3_base":   "xLSTM_none_s{}",
    "ENS3_ctx24":  "xLSTM_ctx24_s{}",
    "ENS3_emb64":  "xLSTM_emb64_s{}",
    "ENS3_blk3":   "xLSTM_blk3_s{}",
    "ENS3_hzvisitor_mtr_full":   "xLSTM_hz-visitor_mtr_full_s{}",
    "ENS3_hzvisitor_mtr_h12":    "xLSTM_hz-visitor_mtr_h12_s{}",
    "ENS3_hzvisitor_mtr_h1h3":   "xLSTM_hz-visitor_mtr_h1h3_s{}",
    "ENS3_hzvisitor_mtr_h6only": "xLSTM_hz-visitor_mtr_h6only_s{}",
}


def main() -> int:
    naive = naive_mae()
    frames = {}
    for label, pat in CLAIMS.items():
        preds = []
        try:
            for s in SEEDS:
                preds.append(load_pred(pat.format(s)))
        except FileNotFoundError:
            print(f"  [skip] {label}: needs 3 seeds")
            continue
        frames[label] = ens(preds, label)

    if "ENS3_base" not in frames:
        print("no baseline")
        return 1
    allp = pd.concat(frames.values(), ignore_index=True)
    allp.to_csv(RES / "stage_h_robust_input.csv", index=False)

    ser = H.dm_series_level(allp, baseline="ENS3_base")
    tim = H.dm_time_aggregated(allp, baseline="ENS3_base")
    ser.to_csv(RES / "stage_h_robust_dm_series.csv", index=False)
    tim.to_csv(RES / "stage_h_robust_dm_time.csv", index=False)

    pd.set_option("display.width", 220)
    print("series-level Wilcoxon (N=23 series, mean |err| per series, per horizon)")
    print(ser[["h", "model", "n_series", "series_won", "series_lost",
               "median_diff", "p_wilcoxon", "better"]].to_string(index=False))

    print("\ntime-aggregated Wilcoxon (N=12 origins, mean |err| per origin, per horizon)")
    print(tim[["h", "model", "n_origins", "origins_won", "origins_lost",
               "median_diff", "p_wilcoxon", "better"]].to_string(index=False))

    # one-line verdicts for the report
    print("\nper-model summary across the 4 horizons:")
    for m in sorted(allp["model"].unique()):
        if m == "ENS3_base":
            continue
        s = ser[ser.model == m]
        t = tim[tim.model == m]
        print(f"  {m:28s} series-level sig(h): "
              f"{','.join(str(r.h) for r in s.itertuples() if r.sig) or '—':>10s}   "
              f"time-level sig(h): "
              f"{','.join(str(r.h) for r in t.itertuples() if r.sig) or '—':>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
