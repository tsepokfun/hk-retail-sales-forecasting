"""
Build the two business-facing tables for FINAL_REPORT v5.0 — from SAVED predictions only.

No training happens here. Every number is read from an existing prediction csv and
aggregated with the project's locked definitions.

WHAT "香港零售總銷售額" MEANS IN THIS PROJECT
    The panel has 23 outlet-type series, of which 15 are children of 4 parents:
        2 = 3+4+5+6+7   12 = 13+14   18 = 19+23+51   35 = 36+37+38+39+40
    Summing all 23 would DOUBLE COUNT.  The true retail total is the sum of the 8
    TOP-LEVEL series only:
        2, 12, 18, 35 (parents) + 8, 9, 30, 32 (standalone)
    Verified: that sum is HK$36,517m for 2024-01 and averages HK$31,404m/month over
    2024, which is the right order of magnitude for Hong Kong retail.

OUTPUT
    results/business_table_A.csv   monthly total, actual vs 4 models, h=1
    results/business_table_B.csv   4 representative series, MAE/WAPE/accuracy
    results/BUSINESS_TABLES.md     both tables rendered as markdown
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eval import harness as H            # noqa: E402

RES = ROOT / "results"

PARENTS = {"2", "12", "18", "35"}
CHILDREN = {"3", "4", "5", "6", "7", "13", "14", "19", "23", "51",
            "36", "37", "38", "39", "40"}

# model label -> (loader, list of files, ensemble?)
MODELS = {
    # NOTE: audit_v*_preds_*.csv are the 2022 VALIDATION block, not the test set.
    # The canonical test-period baseline lives in results/predictions.csv.
    "SeasonalNaive": ("single", [RES / "predictions.csv"]),
    "LightGBM(tuned)": ("single", [RES / "extra_preds_LightGBMTuned_residual.csv"]),
    "xLSTM": ("ens", [RES / "exog_preds_z" / f"xLSTM_none_s{s}.csv"
                      for s in (42, 7, 2024)]),
    "iTransformer": ("ens", [RES / "sota_preds" / f"itransformer_anchor_s{s}.csv"
                             for s in (42, 7, 2024)]),
}

SERIES_FOCUS = {
    "32": ("珠寶鐘錶及名貴禮物", "旅客敏感 / tourist-sensitive"),
    "12": ("服裝鞋類及有關製品", "季節潮流 / seasonal fashion"),
    "19": ("汽車及汽車零件", "大宗耐用 / big-ticket durable"),
    "8": ("超級市場", "民生必需 / essential staple"),
}


def load(label: str) -> pd.DataFrame:
    kind, files = MODELS[label]
    parts = []
    for f in files:
        if not f.exists():
            raise SystemExit(f"missing prediction file: {f}")
        d = pd.read_csv(f, dtype={"series": str, "origin": str})
        d["series"] = d["series"].astype(str)          # guard against int64 read
        if label == "SeasonalNaive" and "model" in d.columns:
            d = d[d["model"] == "SeasonalNaive"].copy()
        if d["origin"].min() < "202312":
            raise SystemExit(f"{f} starts at origin {d['origin'].min()} -- not the test block")
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    if kind == "ens":
        d = d.groupby(["origin", "h", "series"], as_index=False).agg(
            y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    else:
        d = d.groupby(["origin", "h", "series"], as_index=False).agg(
            y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    return d


def main() -> int:
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    top = [c for c in wide.columns if c not in CHILDREN]
    assert len(top) == 8, top

    preds = {m: load(m) for m in MODELS}
    lgbm_plain = RES / "audit_v2_preds_LightGBM_residual.csv"

    # ---------------- Table A: h=1 monthly total -------------------------
    rows = []
    base = preds["SeasonalNaive"]
    base = base[base["h"] == 1]
    for (origin,), g in base.groupby(["origin"]):
        tgt = H.month_add(origin, 1)
        if not ("202401" <= tgt <= "202412"):
            continue
        rec = {"target_month": tgt, "origin": origin}
        for m in MODELS:
            d = preds[m]
            sub = d[(d["h"] == 1) & (d["origin"] == origin)]
            top_sum = sub[sub["series"].isin(top)]
            rec[f"{m}_total"] = float(top_sum["y_pred"].sum())
            rec[f"{m}_n"] = int(len(top_sum))
        rec["actual_total"] = float(wide.loc[tgt, top].sum())
        rows.append(rec)
    A = pd.DataFrame(rows).sort_values("target_month").reset_index(drop=True)
    for m in MODELS:
        A[f"{m}_err_m"] = A[f"{m}_total"] - A["actual_total"]
        A[f"{m}_err_pct"] = 100 * A[f"{m}_err_m"] / A["actual_total"]
    A.to_csv(RES / "business_table_A.csv", index=False)

    print("=" * 100)
    print("TABLE A — 2024 monthly Hong Kong retail TOTAL, h=1, HK$ million")
    print(f"  total = sum of the {len(top)} top-level series {top}")
    print("=" * 100)
    show = ["target_month", "actual_total"] + [f"{m}_total" for m in MODELS]
    print(A[show].round(1).to_string(index=False))
    print("\n  mean absolute percentage error by model (monthly total):")
    for m in MODELS:
        mape = float((A[f"{m}_err_m"].abs() / A["actual_total"]).mean() * 100)
        bias = float(A[f"{m}_err_m"].sum())
        print(f"    {m:16s} MAPE {mape:5.2f}%   total signed error {bias:+9.1f} HK$m")
    print("\n  iTransformer vs SeasonalNaive, monthly total error (HK$m, "
          "positive = over-prediction):")
    diff = A["iTransformer_err_m"] - A["SeasonalNaive_err_m"]
    print(f"    iTransformer over-predicts LESS by {float(diff.mean()):.1f} HK$m/month on average"
          f"  (total {float(diff.sum()):+.1f} HK$m over 12 months)")

    # ---------------- Table B: four representative series ---------------
    rows = []
    for s, (cn, note) in SERIES_FOCUS.items():
        rec = {"series": s, "name_cn": cn, "note": note}
        for m in MODELS:
            d = preds[m]
            d = d[d["series"] == s]
            if d.empty:
                continue
            e = (d["y_true"] - d["y_pred"]).abs()
            rec[f"{m}_mean_actual"] = float(d["y_true"].mean())
            rec[f"{m}_MAE"] = float(e.mean())
            rec[f"{m}_WAPE"] = float(e.sum() / d["y_true"].abs().sum())
            rec[f"{m}_acc"] = 1.0 - rec[f"{m}_WAPE"]
        rows.append(rec)
    B = pd.DataFrame(rows)
    B.to_csv(RES / "business_table_B.csv", index=False)

    print("\n" + "=" * 100)
    print("TABLE B — four representative series, all horizons pooled, HK$ million")
    print("=" * 100)
    for _, r in B.iterrows():
        print(f"\n  [{r['series']}] {r['name_cn']}  ({r['note']})")
        print(f"      mean actual per point {r['xLSTM_mean_actual']:,.1f} HK$m")
        for m in MODELS:
            if f"{m}_MAE" in r and pd.notna(r.get(f"{m}_MAE")):
                print(f"      {m:16s} MAE {r[f'{m}_MAE']:8.1f} HK$m   "
                      f"WAPE {100*r[f'{m}_WAPE']:5.2f}%   accuracy {100*r[f'{m}_acc']:5.2f}%")

    # ---------------- headline comparisons -----------------------------
    print("\n" + "=" * 100)
    print("HEADLINE COMPARISONS (MASE_ALL, 2024 test, from the locked harness)")
    print("=" * 100)
    ref = {"SeasonalNaive": 1.03029, "LightGBM_residual(untuned)": 1.08920,
           "LightGBMTuned_residual": 1.04108,
           "xLSTM_ensemble": 0.94810, "xLSTM_ensemble+MinT_wls": 0.94455,
           "iTransformer+anchor_ensemble": 0.90198}
    for k, v in ref.items():
        print(f"  {k:30s} {v:.5f}")
    for a, b in [("xLSTM_ensemble+MinT_wls", "SeasonalNaive"),
                 ("iTransformer+anchor_ensemble", "SeasonalNaive"),
                 ("xLSTM_ensemble+MinT_wls", "LightGBMTuned_residual"),
                 ("iTransformer+anchor_ensemble", "LightGBMTuned_residual"),
                 ("xLSTM_ensemble+MinT_wls", "LightGBM_residual(untuned)"),
                 ("iTransformer+anchor_ensemble", "LightGBM_residual(untuned)")]:
        print(f"  {a} vs {b}: {100*(ref[a]/ref[b]-1):+.2f}%")

    # ---------------- render markdown ----------------------------------
    lines = ["# 業務對照表（由 predictions csv 實算，未經訓練）", ""]
    lines += ["## 表 A｜2024 年香港零售總銷售額：實際 vs 各模型預測（h=1，HK$ million）",
              "",
              f"總額 = {len(top)} 條頂層序列之和 `{top}`（避開 15 條子序列的重複計數）。",
              "", "| 目標月 | 實際總額 | Seasonal Naive | LightGBM(tuned) | xLSTM | iTransformer |",
              "|---|---:|---:|---:|---:|---:|"]
    for _, r in A.iterrows():
        lines.append(f"| {r['target_month']} | {r['actual_total']:,.0f} | "
                     f"{r['SeasonalNaive_total']:,.0f} | {r['LightGBM(tuned)_total']:,.0f} | "
                     f"{r['xLSTM_total']:,.0f} | {r['iTransformer_total']:,.0f} |")
    lines += ["", "| 目標月 | Naive 誤差 | LGBM 誤差 | xLSTM 誤差 | iTf 誤差 | Naive 誤差% | LGBM 誤差% | xLSTM 誤差% | iTf 誤差% |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in A.iterrows():
        lines.append(
            f"| {r['target_month']} | {r['SeasonalNaive_err_m']:+,.0f} | "
            f"{r['LightGBM(tuned)_err_m']:+,.0f} | {r['xLSTM_err_m']:+,.0f} | "
            f"{r['iTransformer_err_m']:+,.0f} | {r['SeasonalNaive_err_pct']:+.2f}% | "
            f"{r['LightGBM(tuned)_err_pct']:+.2f}% | {r['xLSTM_err_pct']:+.2f}% | "
            f"{r['iTransformer_err_pct']:+.2f}% |")
    lines += ["", "## 表 B｜四個代表性行業（全視界 pooled，HK$ million）", ""]
    lines += ["| 序列 | 行業 | 特性 | 平均實際 | Naive MAE | xLSTM MAE | iTf MAE | xLSTM 準確率 | iTf 準確率 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in B.iterrows():
        lines.append(
            f"| {r['series']} | {r['name_cn']} | {r['note']} | "
            f"{r['xLSTM_mean_actual']:,.1f} | {r['SeasonalNaive_MAE']:,.1f} | "
            f"{r['xLSTM_MAE']:,.1f} | {r['iTransformer_MAE']:,.1f} | "
            f"{100*r['xLSTM_acc']:.2f}% | {100*r['iTransformer_acc']:.2f}% |")
    (RES / "BUSINESS_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {RES/'BUSINESS_TABLES.md'}  ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
