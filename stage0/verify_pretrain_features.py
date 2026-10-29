"""
Stage P / verification - prove that the 27 features built for the M1/M3
pretraining set are the SAME functions as stage0/data/make_features.py.

Why this exists
---------------
`run_pretrain_m1m3.py` re-implements the feature definitions for the external
benchmarks, because make_features.py only reads `data/panel_wide.csv`.  A
re-implementation that is *almost* right would quietly invalidate the whole
pretraining experiment: the backbone would be pretrained on a different input
distribution than it is fine-tuned on, and the transfer result would mean
nothing.  So the re-implementation is diffed against the SHIPPED feature table,
column by column, on the Hong Kong panel itself.

Only the 23 non-regime columns can be compared this way: the HK panel has real
`lny_pre14`/`lny_post7`/`covid_border_closed`/`consumption_voucher` values,
whereas M1/M3 are forced to 0 for all four (documented in the task brief).  The
four regime columns are therefore asserted to be the constants 0 for external
data by construction, and reported as such.

Usage
    .venv\\Scripts\\python.exe verify_pretrain_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import harness as H                                   # noqa: E402
import run_pretrain_m1m3 as P                          # noqa: E402

REGIME = ["lny_pre14", "lny_post7", "covid_border_closed", "consumption_voucher"]
COMPARED = [c for c in H.FEATURE_COLS if c not in REGIME]      # 23 columns


def main() -> int:
    print("=" * 78)
    print("VERIFY - external feature builder vs stage0/data/make_features.py")
    print("=" * 78)

    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    feat = pd.read_csv(ROOT / "data" / "features_long.csv",
                       dtype={"period": str, "series": str})
    feat["period"] = feat["period"].astype(str).str.zfill(6)

    periods = list(wide.index)
    month0 = int(periods[0][4:])
    print(f"  panel {wide.shape[0]} periods x {wide.shape[1]} series   "
          f"first period {periods[0]} -> month0={month0}")
    print(f"  comparing {len(COMPARED)} of {len(H.FEATURE_COLS)} columns "
          f"(the 4 regime dummies are forced to 0 for M1/M3 by construction)")

    train_periods = [p for p in periods if p <= H.TRAIN_END]
    worst = {c: 0.0 for c in COMPARED}
    worst_z = 0.0
    n_rows = 0
    for s in wide.columns:
        v = wide[s].to_numpy(dtype=float)
        assert (v > 0).all(), s
        y_log = np.log(v)
        mu = float(y_log[[i for i, p in enumerate(periods) if p <= H.TRAIN_END]].mean())
        sd = float(y_log[[i for i, p in enumerate(periods) if p <= H.TRAIN_END]].std(ddof=0))
        y_z = (y_log - mu) / sd

        mine = P.series_features(y_log, month0)
        ref = feat[feat["series"] == s].set_index("period").loc[periods]
        n_rows += len(ref)

        for j, c in enumerate(H.FEATURE_COLS):
            if c in REGIME:
                continue
            a = mine[:, j].astype(np.float64)
            b = ref[c].to_numpy(dtype=np.float64)
            both = np.isfinite(a) & np.isfinite(b)
            if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                raise AssertionError(f"{s}/{c}: NaN pattern differs")
            if both.any():
                worst[c] = max(worst[c], float(np.nanmax(np.abs(a[both] - b[both]))))
        worst_z = max(worst_z, float(np.max(np.abs(y_z - ref["y_z"].to_numpy(float)))))

    print(f"  rows checked: {n_rows:,}  ({len(wide.columns)} series x {len(periods)} periods)")
    print(f"\n  {'column':24s} {'max |mine - make_features|':>26s}")
    bad = []
    for c in COMPARED:
        flag = "" if worst[c] < 1e-5 else "   <-- MISMATCH"
        if flag:
            bad.append(c)
        print(f"  {c:24s} {worst[c]:26.3e}{flag}")
    print(f"\n  y_z (train-only mu/sigma)   max |diff| = {worst_z:.3e}")

    ok = not bad and worst_z < 1e-5
    print("\n  RESULT:", "PASS - the external features use the shipped definitions"
          if ok else f"FAIL - mismatched columns {bad}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
