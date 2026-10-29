"""AUDIT — how large is the seed noise in validation block V1, compared with the
gaps the deployable rule has to resolve?

The xLSTM is the only stochastic candidate; its per-seed V1 MASE gives the
band that any "model A beats model B" claim inside V1 has to clear.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H          # noqa: E402
import audit_selection as AS           # noqa: E402

RES = ROOT / "results"


def main() -> int:
    wide = H.load_panel()
    series = sorted([str(c) for c in wide.columns], key=lambda s: (len(s), s))
    denom = AS.denom_for(wide, AS.FROZEN_CUTOFF, series)

    rows = []
    for s in (42, 7, 2024):
        d = AS.read(RES / f"stage2_oof_s{s}.csv")
        d["model"] = f"xLSTM_s{s}"
        m = H.metrics_from_frame(d, denom)
        m["seed"] = s
        rows.append(m)
    m = pd.concat(rows, ignore_index=True)
    piv = m.pivot_table(index="seed", columns="h", values="MASE", aggfunc="first")
    print("per-seed V1 MASE (xLSTM, frozen <=%s, origins %s..%s):"
          % (AS.FROZEN_CUTOFF, AS.V1_LO, AS.V1_HI))
    print(piv.round(5).to_string())
    spr = (piv.max() - piv.min()).round(5)
    print("\nseed spread (max - min):")
    print(spr.to_string())
    print("\nsd over seeds:")
    print(piv.std(ddof=1).round(5).to_string())
    spr.to_frame("seed_spread_v1").to_csv(RES / "audit_v1_seed_spread.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
