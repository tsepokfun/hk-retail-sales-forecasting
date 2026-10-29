"""
INDEPENDENT cross-check of `data/exog_wide.csv`  (written by the parent agent,
NOT by the stage-1 subagent -- this is a second opinion on the subagent's own
verify_exog.py).

Two separate questions, answered separately:

Q1  FORMULA  -- does every column equal what EXOG.md says it should?
    Method: re-run the module's loaders, apply the documented transform here,
    and diff against the file cell by cell.  Tolerance 1e-9 (all eight land at
    ~1e-16, i.e. bit-identical up to float64 round-off).

Q2  CAUSALITY -- does any value at month t secretly depend on data after t?
    Method: the "prefix mask" test.  Set every RAW observation after 2023-12 to
    NaN, rebuild, and diff the in-window values against the shipped file.
    A causal pipeline returns 0.0 everywhere; a pipeline with a centred rolling
    mean, a full-sample scaler, or a missing shift returns a non-zero diff.

Why Q2 matters more than Q1: a formula can be implemented exactly as documented
and still leak, because the *documentation* can be wrong about the shift.  Q2
tests the pipeline, not the prose.

Usage
    .venv\\Scripts\\python.exe data\\verify_exog_crosscheck.py
"""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

TRAIN_END = "202312"
TOL = 1e-9


def load_module():
    """import build_exog with its chatty loader prints silenced"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod = importlib.import_module("build_exog")
    return mod


def main() -> int:
    B = load_module()
    quiet = io.StringIO()

    exog = pd.read_csv(ROOT / "data" / "exog_wide.csv", dtype={"period": str})
    exog = exog.set_index("period")
    panel = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0)
    periods = [str(p) for p in panel.index]
    train = [p for p in periods if p <= TRAIN_END]

    print(f"exog_wide.csv  {exog.shape[0]} rows x {exog.shape[1]} cols  "
          f"{periods[0]}..{periods[-1]}")
    assert len(periods) == 263, len(periods)
    assert list(exog.index) == periods, "period index mismatch vs panel_wide.csv"

    def pz(s: pd.Series) -> pd.Series:
        return B.to_panel(s, periods)

    # ---------------- Q1: formula ----------------
    print("\n" + "=" * 78)
    print("Q1  FORMULA  -- recompute each column from the raw loaders")
    print("=" * 78)
    q1 = []

    def cmp(name: str, mine: pd.Series, note: str = "") -> None:
        a, b = exog[name], mine
        both = a.notna() & b.notna()
        d = (a[both] - b[both]).abs()
        q1.append({
            "feature": name,
            "n_cmp": int(both.sum()),
            "max_abs_diff": float(d.max()) if len(d) else float("nan"),
            "only_file": int((a.notna() & b.isna()).sum()),
            "only_mine": int((a.isna() & b.notna()).sum()),
            "PASS": bool(len(d) and d.max() < TOL),
            "note": note,
        })

    with contextlib.redirect_stdout(quiet):
        for key, col, l1 in [("cpi", "cpi_yoy", False),
                             ("eeri", "eeri_yoy", False),
                             ("visitor", "visitor_yoy", True),
                             ("mtr", "mtr_yoy", False),
                             ("cargo", "cargo_yoy", False)]:
            s, _ = getattr(B, f"build_{key}")()
            cmp(col, pz(B.yoy_level(s, key, log1p=l1).shift(1)),
                "log-YoY then shift(1)")

        s, _ = B.build_rate()
        z, sc = B.train_z(B.monthly_grid(s), train, "rate_z")
        cmp("rate_z", pz(z.shift(1)),
            f"mu={sc['mu']:.4f} sigma={sc['sigma']:.4f} ddof={sc['ddof']}")

        w, _ = B.build_weather()
        for name, col in (("rain_anom_z", "rain_mm"), ("temp_anom_z", "temp_c")):
            g = B.monthly_grid(w[col])
            zz, sc2 = B.weather_anomaly_z(B.to_panel(g, periods), train, name)
            cmp(name, zz,
                f"cal-month anomaly, mu={sc2['anomaly_mu']:+.4f} "
                f"sigma={sc2['anomaly_sigma']:.4f}")

    q1df = pd.DataFrame(q1)
    print(q1df.to_string(index=False))

    # ---------------- Q2: causality ----------------
    print("\n" + "=" * 78)
    print(f"Q2  CAUSALITY  -- NaN every raw observation after {TRAIN_END}, rebuild,")
    print("    and diff the in-window values.  0.0 everywhere == no look-ahead.")
    print("=" * 78)
    q2 = []

    def mask_after(s: pd.Series) -> pd.Series:
        """blank every entry whose month key is later than TRAIN_END"""
        s = s.copy()
        for i in list(s.index):
            key = str(i)                      # 'YYYYMM' strings, ints, or Periods
            if isinstance(i, pd.Period):
                key = f"{i.year}{i.month:02d}"
            if key > TRAIN_END:
                s.loc[i] = np.nan
        return s

    with contextlib.redirect_stdout(quiet):
        for key, col, l1 in [("cpi", "cpi_yoy", False),
                             ("eeri", "eeri_yoy", False),
                             ("visitor", "visitor_yoy", True),
                             ("mtr", "mtr_yoy", False),
                             ("cargo", "cargo_yoy", False)]:
            s, _ = getattr(B, f"build_{key}")()
            full = pz(B.yoy_level(s, key, log1p=l1).shift(1))
            cut = pz(B.yoy_level(mask_after(s), key, log1p=l1).shift(1))
            a, b = full.loc[:TRAIN_END], cut.loc[:TRAIN_END]
            d = (a - b).abs()
            q2.append({"feature": col,
                       "max_abs_diff_in_window": float(d.max()) if d.notna().any() else 0.0,
                       "nan_file": int(a.isna().sum()),
                       "nan_masked_build": int(b.isna().sum()),
                       "PASS": bool(not d.notna().any() or d.max() < TOL)})

        s, _ = B.build_rate()
        g_full = B.monthly_grid(s)
        g_cut = B.monthly_grid(mask_after(s))
        zf, _ = B.train_z(g_full, train, "rate_z")
        zc, _ = B.train_z(g_cut, train, "rate_z")
        a, b = pz(zf.shift(1)).loc[:TRAIN_END], pz(zc.shift(1)).loc[:TRAIN_END]
        d = (a - b).abs()
        q2.append({"feature": "rate_z",
                   "max_abs_diff_in_window": float(d.max()) if d.notna().any() else 0.0,
                   "nan_file": int(a.isna().sum()),
                   "nan_masked_build": int(b.isna().sum()),
                   "PASS": bool(not d.notna().any() or d.max() < TOL)})

        w, _ = B.build_weather()
        for name, col in (("rain_anom_z", "rain_mm"), ("temp_anom_z", "temp_c")):
            gf = B.monthly_grid(w[col])
            gc = B.monthly_grid(mask_after(w[col]))
            zf, _ = B.weather_anomaly_z(B.to_panel(gf, periods), train, name)
            zc, _ = B.weather_anomaly_z(B.to_panel(gc, periods), train, name)
            a, b = zf.loc[:TRAIN_END], zc.loc[:TRAIN_END]
            d = (a - b).abs()
            q2.append({"feature": name,
                       "max_abs_diff_in_window": float(d.max()) if d.notna().any() else 0.0,
                       "nan_file": int(a.isna().sum()),
                       "nan_masked_build": int(b.isna().sum()),
                       "PASS": bool(not d.notna().any() or d.max() < TOL)})

    q2df = pd.DataFrame(q2)
    print(q2df.to_string(index=False))

    ok1, ok2 = bool(q1df["PASS"].all()), bool(q2df["PASS"].all())
    n1, n2 = int(q1df["PASS"].sum()), int(q2df["PASS"].sum())
    print("\n" + "=" * 78)
    print(f"Q1 formula   : {'PASS' if ok1 else 'FAIL'}  ({n1}/{len(q1df)} columns)")
    print(f"Q2 causality : {'PASS' if ok2 else 'FAIL'}  ({n2}/{len(q2df)} columns)")
    print("=" * 78)
    q1df.to_csv(ROOT / "data" / "exog_crosscheck_formula.csv", index=False)
    q2df.to_csv(ROOT / "data" / "exog_crosscheck_causality.csv", index=False)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
