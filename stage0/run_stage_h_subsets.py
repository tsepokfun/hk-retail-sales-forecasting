"""
Stage H, task A follow-up -- is the 3->5 seed ensemble gain REAL, or was the
original 3-seed set just a bad draw?

The per-seed numbers make the question unavoidable:

    seed  42   1.01743     <- the worst seed, and the one the project reports first
    seed   7   0.95989
    seed 2024  0.97333
    seed   1   0.91714     <- better than the 3-seed ENSEMBLE built from the other three
    seed 1234  0.93382

so "5-seed ensemble 0.91230 vs 3-seed ensemble 0.94810" mixes two effects:
more diversity, and dropping the unlucky draw.  This script separates them by
scoring EVERY 3-subset of the 5 seeds (C(5,3)=10) and every 4-subset (C(5,4)=5).

If the original {42,7,2024} ensemble sits at the BOTTOM of the 10, then quoting
0.94810 as "the 3-seed ensemble" understates the method and the honest baseline
for any future comparison is the median subset, not that one.

Usage
    .venv\\Scripts\\python.exe run_stage_h_subsets.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H            # noqa: E402
from run_stage_h_ens import combine, load_tags, to_frame   # noqa: E402

RES = ROOT / "results"
DIRS = [RES / "exog_preds_z", RES / "stage_h_preds"]
ORIG3 = ("xLSTM_none_s42", "xLSTM_none_s7", "xLSTM_none_s2024")
ALL5 = ORIG3 + ("xLSTM_base_s1", "xLSTM_base_s1234")


def main() -> int:
    naive = H.seasonal_naive_mae_train(H.load_panel())
    wide = load_tags(list(ALL5), DIRS).sort_index()
    rows = []
    for k in (1, 2, 3, 4, 5):
        for combo in itertools.combinations(ALL5, k):
            nm = f"ENS{k}:" + ",".join(c.replace("xLSTM_none_", "").replace("xLSTM_base_", "")
                                       for c in combo)
            fr = to_frame(wide, combine(wide, list(combo), "level"), nm)
            m = H.metrics_from_frame(fr, naive)
            allrow = m[m.h.astype(str) == "ALL"].iloc[0]
            rec = {"k": k, "subset": ",".join(c.replace("xLSTM_none_", "")
                                              .replace("xLSTM_base_", "") for c in combo),
                   "tag_tuple": "|".join(combo),
                   "MASE_ALL": float(allrow["MASE"]), "WAPE_ALL": float(allrow["WAPE"]),
                   "is_original": set(combo) == set(ORIG3)}
            for h in ("1", "3", "6", "12"):
                rec[f"MASE_h{h}"] = float(m[m.h.astype(str) == h].iloc[0]["MASE"])
            rows.append(rec)
    S = pd.DataFrame(rows)
    S.to_csv(RES / "stage_h_tableA_subsets.csv", index=False)

    pd.set_option("display.width", 200)
    print("every ensemble size x subset (level-space mean forecast)\n")
    print(S[["k", "subset", "MASE_h1", "MASE_h3", "MASE_h6", "MASE_h12",
             "MASE_ALL", "is_original"]].to_string(
        index=False, float_format=lambda v: f"{v:.5f}"))

    print("\nrank of the reported 3-seed ensemble among the 10 3-subsets:")
    s3 = S[S.k == 3].sort_values("MASE_ALL").reset_index(drop=True)
    for i, r in s3.iterrows():
        print(f"  {i+1:2d}. {r['subset']:28s} {r['MASE_ALL']:.5f}"
              f"{'   <- the reported one' if r['is_original'] else ''}")
    print(f"\n  3-subset MASE_ALL: min {s3.MASE_ALL.min():.5f}  "
          f"median {s3.MASE_ALL.median():.5f}  max {s3.MASE_ALL.max():.5f}")
    for k in (4, 5):
        sk = S[S.k == k]
        print(f"  {k}-subset MASE_ALL: min {sk.MASE_ALL.min():.5f}  "
              f"median {sk.MASE_ALL.median():.5f}  max {sk.MASE_ALL.max():.5f}"
              + ("  (only one 5-subset)" if k == 5 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
