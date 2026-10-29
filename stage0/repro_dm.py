"""Diebold-Mariano tests for the reproduced statistical baselines.

Compares ETS (both protocols) and Theta against:
  * SeasonalNaive   (the project's reference line)
  * iTransformer+anchor ENS3 (the project's best model, 0.90198)

Uses ONLY the harness's own functions (dm_series_level / dm_time_aggregated),
so the tests are the same ones quoted in FINAL_REPORT.md.

Run: stage0\\.venv\\Scripts\\python.exe stage0\\repro_dm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "eval"))
import harness as H  # noqa: E402

RES = HERE / "results"
REPRO = RES / "repro"


def load(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype={"series": str, "origin": str})


def ens3(stem: str) -> pd.DataFrame:
    frames = [load(RES / "sota_preds" / f"{stem}_s{s}.csv") for s in ("42", "7", "2024")]
    base = frames[0][["origin", "h", "series", "y_true"]].copy()
    st = pd.concat([f[["origin", "h", "series", "y_pred"]] for f in frames], ignore_index=True)
    m = st.groupby(["origin", "h", "series"], as_index=False)["y_pred"].mean()
    out = base.merge(m, on=["origin", "h", "series"], how="left")
    out["model"] = stem + "_ENS3"
    return out


def main() -> int:
    parts = []
    parts.append(load(RES / "predictions.csv").query("model == 'SeasonalNaive'"))
    parts.append(ens3("itransformer_anchor"))
    for f, name in [("repro_preds_ets.csv", "ETS_HW_add"),
                    ("repro_preds_ets_frozen.csv", "ETS_HW_add_FROZEN"),
                    ("repro_preds_theta.csv", "Theta_seas"),
                    ("repro_preds_sarimax.csv", "SARIMAX_111_0112")]:
        p = REPRO / f
        if p.exists():
            d = load(p)
            d["model"] = name
            parts.append(d)
    preds = pd.concat(parts, ignore_index=True)
    print(preds.groupby("model").size().to_string(), "\n")

    for base in ("SeasonalNaive", "itransformer_anchor_ENS3"):
        for fn, tag in [(H.dm_series_level, "series-level"), (H.dm_time_aggregated, "time-aggregated")]:
            try:
                t = fn(preds, baseline=base)
            except Exception as exc:                       # noqa: BLE001
                print(f"[{tag} vs {base}] FAILED: {type(exc).__name__}: {exc}")
                continue
            keep = list(t.columns)
            print(f"--- {tag} vs {base} ---")
            print(t[keep].to_string(index=False))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
