"""AUDIT — the harness's OWN significance tests (not re-implemented here) on the
key 2024-test pairs, so the audit does not rest on a home-made test."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H          # noqa: E402
import audit_selection as AS           # noqa: E402

RES = ROOT / "results"
KEEP = ["SeasonalNaive", "xLSTM_ensemble", "kNN_residual",
        "LightGBMTuned_residual", "LSTM_none", "Drift", "Ridge_residual"]


def main() -> int:
    frames = AS.load_test_grid()
    big = pd.concat([frames[m] for m in KEEP], ignore_index=True)
    big["h"] = big["h"].astype(int)

    s = H.dm_series_level(big, baseline="SeasonalNaive")
    t = H.dm_time_aggregated(big, baseline="SeasonalNaive")
    s.to_csv(RES / "audit_dm_series_official.csv", index=False)
    t.to_csv(RES / "audit_dm_time_official.csv", index=False)
    print("=" * 96)
    print("harness.dm_series_level — one obs per SERIES (N=23), vs SeasonalNaive")
    print("=" * 96)
    print(s.to_string(index=False))
    print("\n" + "=" * 96)
    print("harness.dm_time_aggregated — one obs per ORIGIN (T=12), vs SeasonalNaive")
    print("=" * 96)
    print(t.to_string(index=False))

    p = H.dm_test(
        (frames["xLSTM_ensemble"]["y_true"] - frames["xLSTM_ensemble"]["y_pred"]).to_numpy(),
        (frames["SeasonalNaive"]["y_true"] - frames["SeasonalNaive"]["y_pred"]).to_numpy(),
        h=1, power=1)
    print("\n  pooled all-horizon DM (power=1), xLSTM_ensemble vs SeasonalNaive:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
