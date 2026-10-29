"""
Stage 3 / step 3 — causal verification of the news block.

Two questions, answered separately, same as Stage 1's exogenous block:

Q1  FORMULA   does `news_text_wide.csv` equal what news_features.py says it is?
              Recompute the three scalar features directly from the headline csv
              and diff.

Q2  CAUSALITY does any month-t value secretly depend on a headline published
              after t?
              Method: drop every headline dated AFTER the training cut-off,
              rebuild the whole block (scaler, PCA, everything), and diff the
              in-window (<= train_end) values.  A causal pipeline returns 0.0.
              This is exactly the prefix-mask test used in Stage 1; it catches
              a full-sample PCA, a full-sample z-score, or a missing shift.

A third check is specific to text: the NO-COVERAGE months must be imputed to the
TRAIN MEAN (z = 0), not to zero on the raw scale.  Asserting that keeps a data
gap from masquerading as "no news happened".

Usage
    .venv\\Scripts\\python.exe ../stage3/verify_text_causality.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
TRAIN_END = "202312"
TOL = 1e-9

import news_features as NF          # noqa: E402


def rebuild(headlines: pd.DataFrame) -> pd.DataFrame:
    """identical pipeline to news_features.main(), used for both passes"""
    sc = NF.month_scalar(headlines)
    SC = ["news_log_count", "news_fin_share", "news_sent_net"]
    EMB = [f"news_pc{i+1}" for i in range(NF.EMB_DIM)]
    try:
        emb, _ = NF.month_embeddings_st(headlines)
    except Exception:                                        # noqa: BLE001
        emb, _ = NF.month_embeddings_tfidf(headlines)
    periods = [str(p) for p in pd.read_csv(
        ROOT.parent / "stage0" / "data" / "panel_wide.csv", index_col=0).index]
    a, _ = NF.to_panel(sc, periods, SC)
    b, _ = NF.to_panel(emb, periods, EMB)
    return pd.concat([a, b], axis=1)


def main() -> int:
    d = NF.load_headlines()
    shipped = pd.read_csv(DATA / "news_text_wide.csv", index_col=0, dtype={0: str})
    meta = json.loads((DATA / "news_text_meta.json").read_text(encoding="utf-8"))
    cols = meta["columns"]

    # ---------------- Q1: formula (scalar block only) ----------------
    print("=" * 80)
    print("Q1  FORMULA — recompute the scalar block from news_headlines.csv")
    print("=" * 80)
    sc = NF.month_scalar(d)
    periods = [str(p) for p in pd.read_csv(
        ROOT.parent / "stage0" / "data" / "panel_wide.csv", index_col=0).index]
    sc_z, rep = NF.to_panel(sc, periods, ["news_log_count", "news_fin_share",
                                         "news_sent_net"])
    rows = []
    for c in ["news_log_count", "news_fin_share", "news_sent_net"]:
        a, b = shipped[c].to_numpy(), sc_z[c].to_numpy()
        dmax = float(np.nanmax(np.abs(a - b)))
        rows.append({"feature": c, "n": len(a), "max_abs_diff": dmax,
                     "PASS": bool(dmax < TOL)})
    q1 = pd.DataFrame(rows)
    print(q1.to_string(index=False))

    # ---------------- Q2: causality (prefix mask on the WHOLE block) -------
    print("\n" + "=" * 80)
    print(f"Q2  CAUSALITY — drop every headline dated after {TRAIN_END},")
    print("    rebuild scaler + PCA, and diff the in-window values")
    print("=" * 80)
    masked = d[d["month"] <= TRAIN_END].copy()
    print(f"  headlines kept {len(masked):,} / {len(d):,} "
          f"(dropped {len(d)-len(masked):,})")
    full = rebuild(d)
    cut = rebuild(masked)
    w = [p for p in full.index if p <= TRAIN_END]
    rows = []
    for c in cols:
        a = full.loc[w, c].to_numpy(dtype=float)
        b = cut.loc[w, c].to_numpy(dtype=float)
        dmax = float(np.nanmax(np.abs(a - b)))
        rows.append({"feature": c, "n_in_window": len(a), "max_abs_diff": dmax,
                     "PASS": bool(dmax < 1e-6)})
    q2 = pd.DataFrame(rows)
    print(q2.to_string(index=False))

    # ---------------- Q3: imputation policy -------------------------------
    print("\n" + "=" * 80)
    print("Q3  NO-COVERAGE months are imputed to the TRAIN MEAN (z = 0)")
    print("=" * 80)
    cov = d.groupby("month").size()
    periods_all = shipped.index.tolist()
    missing = [p for p in periods_all if p not in cov.index]
    z_vals = shipped.loc[missing, "news_log_count"].to_numpy()
    q3_pass = bool(len(missing) == 0 or np.all(np.abs(z_vals) < 1e-9))
    print(f"  months with no headline: {len(missing)}  "
          f"range {missing[0] if missing else '-'}..{missing[-1] if missing else '-'}")
    print(f"  their news_log_count z values: min {z_vals.min():+.3e} "
          f"max {z_vals.max():+.3e}  -> {'PASS' if q3_pass else 'FAIL'}")

    ok = bool(q1["PASS"].all() and q2["PASS"].all() and q3_pass)
    print("\n" + "=" * 80)
    print(f"Q1 formula   : {'PASS' if q1['PASS'].all() else 'FAIL'} "
          f"({int(q1['PASS'].sum())}/{len(q1)})")
    print(f"Q2 causality : {'PASS' if q2['PASS'].all() else 'FAIL'} "
          f"({int(q2['PASS'].sum())}/{len(q2)})")
    print(f"Q3 imputation: {'PASS' if q3_pass else 'FAIL'}")
    print(f"OVERALL      : {'PASS' if ok else 'FAIL'}")
    print("=" * 80)
    q1.to_csv(DATA / "verify_formula.csv", index=False)
    q2.to_csv(DATA / "verify_causality.csv", index=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
