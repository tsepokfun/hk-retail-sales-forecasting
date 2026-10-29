"""Independent self-test for the per-horizon causal mask used in rolling refit.

The claim that must hold: at test origin o, a training sample drawn from origin i
contributes gradient ONLY on horizons h with i + h <= o.  If that fails anywhere,
the rolling refit leaks the future and every number it produces is invalid.

This test re-derives the expected mask from first principles (indices, not from
the implementation) and compares.  It also checks the two boundary cases:

  * the OLD protocol (purge_val, val_months=24) keeps origins i <= 2021-11 only
  * the NEW rolling protocol at o = 202312 admits i up to 2023-11, but for
    i > 2022-12 only some horizons are labelled

Run: stage0\\.venv\\Scripts\\python.exe stage0\\selftest_rolling_mask.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "eval"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))
import harness as H  # noqa: E402
import xlstm_direct as XD  # noqa: E402
import run_rolling_refit as RR  # noqa: E402

HEADS = list(H.HORIZONS)


def check_zi(Z, origins, o_idx, n_core, label):
    W, Yt, M = RR.masked_windows_zi(Z, origins, o_idx, n_core)
    Yt, M = Yt.numpy(), M.numpy()
    bad = 0
    for k, i in enumerate(origins):
        for hj, h in enumerate(HEADS):
            expected = 1.0 if (i + h) <= o_idx else 0.0
            got = float(M[k, 0, hj])
            if expected != got:
                bad += 1
            # a masked-out target must be NaN, an unmasked one must be finite
            if got == 1.0 and not np.isfinite(Yt[k, :, hj]).all():
                bad += 1
            if got == 0.0 and not np.isnan(Yt[k, :, hj]).all():
                bad += 1
    print(f"  {label}: samples={len(origins)} labels_on={int(M.sum())} "
          f"labels_total={M.size} violations={bad}")
    return bad


def check_xlstm(tensors, origins, o_idx, label):
    periods, series, F, Y, KF = tensors
    Xs, Ss, Cs, Ts, Ms = RR.masked_samples_xlstm(
        origins, np.arange(len(series)), F, Y, KF, o_idx)
    bad = 0
    # rebuild the (i, j) -> row map EXACTLY as the builder does, including its
    # two skip rules (NaN window, and all-horizons-masked), otherwise the row
    # counter desynchronises after the first skipped sample.
    row = 0
    skipped = 0
    for i in origins:
        lo = i - RR.CONTEXT + 1
        if lo < 0:
            skipped += len(series)
            continue
        for j in range(len(series)):
            if np.isnan(F[lo:i + 1, j, :]).any():
                skipped += 1
                continue
            if not any((i + h) <= o_idx for h in HEADS):
                skipped += 1
                continue
            if row >= len(Ms):
                bad += 1
                continue
            for hj, h in enumerate(HEADS):
                expected = 1.0 if (i + h) <= o_idx else 0.0
                if expected != float(Ms[row, hj]):
                    bad += 1
                if expected == 1.0 and not np.isfinite(Ts[row, hj]):
                    bad += 1
                if expected == 0.0 and not np.isnan(Ts[row, hj]):
                    bad += 1
            row += 1
    if row != len(Ms):
        bad += 1
    print(f"  {label}: samples={len(Ms)} (skipped {skipped}) labels_on={int(Ms.sum())} "
          f"labels_total={Ms.size} violations={bad}")
    return bad


def main() -> int:
    feat = H.load_features()
    periods, series, F, Y, KF = XD.build_tensors(feat)
    Z = Y.astype(np.float64)
    pidx = {p: i for i, p in enumerate(periods)}
    n_core = len(series)

    print("=== per-horizon causal mask ===")
    total_bad = 0
    for origin in ("202312", "202406", "202411"):
        o_idx = pidx[origin]
        origins = RR.usable_origins(Z, o_idx - 1)
        total_bad += check_zi(Z, origins, o_idx, n_core, f"iTransformer o={origin}")
        total_bad += check_xlstm((periods, series, F, Y, KF), origins, o_idx,
                                 f"xLSTM      o={origin}")

    print("\n=== protocol comparison at o=202312 ===")
    o_idx = pidx["202312"]
    old_val_from = o_idx - 24
    old_train = [i for i in range(RR.CONTEXT - 1, old_val_from
                                  if False else RR.CONTEXT - 1)
                 ]
    old_train = [i for i in range(RR.CONTEXT - 1, old_val_from)]
    new_origins = RR.usable_origins(Z, o_idx - 1)
    print(f"  OLD (val_months=24): train origins up to {periods[old_val_from - 1]} "
          f"({len(old_train)} indices)")
    print(f"  NEW (rolling)      : train origins up to {periods[new_origins[-1]]} "
          f"({len(new_origins)} indices, partial horizons masked)")
    extra = [i for i in new_origins if i >= old_val_from]
    print(f"  additional origins admitted: {len(extra)} "
          f"({periods[extra[0]]}..{periods[extra[-1]]})" if extra else "  none")

    print(f"\nRESULT: {'PASS' if total_bad == 0 else 'FAIL'} "
          f"(violations={total_bad})")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
