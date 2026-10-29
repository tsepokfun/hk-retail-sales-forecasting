"""Diagnose WHERE rolling refit helps or hurts, per test origin.

Rolling iTransformer came out WORSE than the frozen one overall (0.94635 vs
0.90198).  Before blaming the protocol, find out whether the loss is uniform or
concentrated at the late origins — the run log shows `best_ep` collapsing from
~78 to ~6 after origin 202408, which points at the causal validation block
becoming a bad early-stopping signal, not at the protocol being wrong.

Compares, per origin: frozen iTransformer ENS3 vs rolling iTransformer ENS3,
using the same per-series MASE denominator.

Run: stage0\\.venv\\Scripts\\python.exe stage0\diag_rolling_by_origin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "eval"))
import harness as H  # noqa: E402

RES = HERE / "results"


def ens3(stem: str) -> pd.DataFrame:
    frames = [pd.read_csv(RES / "sota_preds" / f"{stem}_s{s}.csv",
                          dtype={"series": str, "origin": str})
              for s in ("42", "7", "2024")]
    base = frames[0][["origin", "h", "series", "y_true"]].copy()
    st = pd.concat([f[["origin", "h", "series", "y_pred"]] for f in frames], ignore_index=True)
    m = st.groupby(["origin", "h", "series"], as_index=False)["y_pred"].mean()
    out = base.merge(m, on=["origin", "h", "series"], how="left")
    out["model"] = stem
    return out


def main() -> int:
    wide = H.load_panel()
    naive = H.seasonal_naive_mae_train(wide)

    frozen = ens3("itransformer_anchor")
    rolling = pd.read_csv(RES / "rolling" / "preds_itransformer_rolling_mint.csv",
                          dtype={"series": str, "origin": str})
    rolling = rolling[rolling["model"].str.endswith("ENS3")].copy()
    rolling["model"] = "itransformer_rolling"

    rows = []
    for origin, g in pd.concat([frozen, rolling], ignore_index=True).groupby("origin"):
        for model, gg in g.groupby("model"):
            m = H.metrics_from_frame(gg, naive)
            a = m[m["h"].astype(str) == "ALL"].iloc[0]
            rows.append({"origin": origin, "model": model,
                         "MASE": float(a["MASE"]), "WAPE": float(a["WAPE"]),
                         "bias_pct": float(a["bias_pct"])})
    d = pd.DataFrame(rows)
    piv = d.pivot(index="origin", columns="model", values="MASE")
    piv["delta"] = piv["itransformer_rolling"] - piv["itransformer_anchor"]
    print("=== per-origin MASE (ALL horizons) ===")
    print(piv.round(4).to_string())
    print()
    print("mean delta:", round(float(piv["delta"].mean()), 4),
          " | worse at", int((piv["delta"] > 0).sum()), "of", len(piv), "origins")
    print()
    # horizon-level view
    hrows = []
    for h in H.HORIZONS:
        for model, gg in pd.concat([frozen, rolling], ignore_index=True).groupby("model"):
            m = H.metrics_from_frame(gg[gg["h"] == h], naive)
            a = m[m["h"].astype(str) == "ALL"].iloc[0]
            hrows.append({"h": h, "model": model, "MASE": float(a["MASE"])})
    hp = pd.DataFrame(hrows).pivot(index="h", columns="model", values="MASE")
    hp["delta"] = hp["itransformer_rolling"] - hp["itransformer_anchor"]
    print("=== per-horizon MASE ===")
    print(hp.round(4).to_string())
    piv.to_csv(RES / "rolling" / "diag_itransformer_by_origin.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
