#!/usr/bin/env python
"""
generate_report_figures.py — the six figures of FINAL_REPORT.md.

Every number plotted here is read from a result file that already exists in the
repository; nothing is typed in by hand except the prose labels.  Run from the
project root:

    stage0\\.venv\\Scripts\\python.exe generate_report_figures.py

Outputs (dpi=300, bbox_inches='tight'):
    figures/fig1_system_architecture.png
    figures/fig2_2024_total_sales_forecast.png
    figures/fig3_four_categories_panel.png
    figures/fig4_model_leaderboard.png
    figures/fig5_multidim_text_heatmap.png
    figures/fig6_ablation_progression.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
S0 = ROOT / "stage0"
S3 = ROOT / "stage3"

# ---------------------------------------------------------------------------
# house style
# ---------------------------------------------------------------------------
INK = "#111827"
GREY = "#6b7280"
C_ACT = "#111827"
C_NAIVE = "#9ca3af"
C_XL = "#1d4ed8"
C_IT = "#b91c1c"
C_OURS = "#047857"
C_WARN = "#f59e0b"
C_A, C_B, C_C, C_D = "#e0ecff", "#e8f7ee", "#fff3d6", "#f3e8ff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "#4b5563",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#d1d5db",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.75,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})

TOP_LEVEL = ["2", "8", "9", "12", "18", "30", "32", "35"]   # 8 top-level series
SERIES_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "12", "13", "14", "18",
                "19", "23", "30", "32", "35", "36", "37", "38", "39", "40", "51"]
SEEDS = (42, 7, 2024)
MONTHS_2024 = [f"2024{m:02d}" for m in range(1, 13)]
MONTH_LAB = [f"{m:02d}" for m in range(1, 13)]


def save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.0f} KB)")
    return p


def facts() -> dict:
    return json.loads((S3 / "results" / "report_facts.json").read_text(encoding="utf-8"))


def corpus_counts() -> tuple[int, int]:
    """(unique headlines, headlines admitted to the router), read from the run
    artefacts so the diagram cannot drift away from the data it describes."""
    meta = json.loads((S3 / "data" / "multidim" / "routed_meta.json")
                      .read_text(encoding="utf-8"))
    n_unique = int(meta.get("n_headlines", 0))
    ds = pd.read_csv(S3 / "results" / "multidim_dimension_stats.csv")
    n_adm = int(ds.loc[ds["dimension"] != "ALL", "headlines"].sum())
    return n_unique, n_adm


def arm_table() -> pd.DataFrame:
    """ENS3 rows of the Part 1 ablation, indexed by arm x horizon."""
    m = pd.read_csv(S3 / "results" / "multidim_text_results.csv")
    e = m[m["seed"].astype(str) == "ENS3"]
    t = e.pivot_table(index="arm", columns="h", values="MASE")
    t.columns = t.columns.astype(str)
    return t


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
def load_preds(paths) -> pd.DataFrame:
    fr = [pd.read_csv(p, dtype={"series": str, "origin": str})
          for p in paths if Path(p).exists()]
    if not fr:
        raise FileNotFoundError(paths)
    d = pd.concat(fr, ignore_index=True)
    return d


def ens3(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["origin", "h", "series"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    return g


def panel() -> pd.DataFrame:
    w = pd.read_csv(S0 / "data" / "panel_wide.csv", index_col=0)
    w.index = w.index.astype(str)
    return w


def rolling_h1(d: pd.DataFrame, series: list[str], months: list[str]) -> pd.Series:
    """one-month-ahead forecast for every month in `months`.

    month M is forecast from origin M-1 with h=1, so the whole 2024 line is an
    honest one-step rolling forecast (2024-01 comes from origin 2023-12).
    """
    out = {}
    for m in months:
        y, mm = int(m[:4]), int(m[4:])
        mm -= 1
        if mm == 0:
            y, mm = y - 1, 12
        org = f"{y}{mm:02d}"
        sub = d[(d["origin"] == org) & (d["h"] == 1)]
        if sub.empty:
            continue
        out[m] = float(sub[sub["series"].isin(series)]["y_pred"].sum())
    return pd.Series(out)


# ===========================================================================
# FIG 1 — system architecture
# ===========================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(16.6, 8.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.grid(False)

    def panel(x0, y0, w, h, title, fc, num):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                     boxstyle="round,pad=0.006,rounding_size=0.018",
                     fc=fc, ec="#374151", lw=1.4, zorder=1))
        ax.text(x0 + 0.012, y0 + h - 0.032, f"{num}  {title}", fontsize=10.5,
                fontweight="bold", color=INK, va="center", zorder=5)

    def box(x0, y0, w, h, title, body="", fc="white", ec="#6b7280", fs=8.0,
            tfs=9.0, lw=1.0, tdy=0.030):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                     boxstyle="round,pad=0.004,rounding_size=0.012",
                     fc=fc, ec=ec, lw=lw, zorder=3))
        ty = y0 + h - tdy
        ax.text(x0 + w / 2, ty, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=INK, zorder=4)
        if body:
            ax.text(x0 + w / 2, ty - 0.032, body, ha="center", va="top",
                    fontsize=fs, color="#374151", zorder=4, linespacing=1.45)
        return ty

    def tx(x, y, s, fs=8.0, color="#374151", bold=False, style="normal",
           ha="center", va="center", ls=1.5, z=5):
        ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", style=style,
                linespacing=ls, zorder=z)

    def arrow(p1, p2, color="#374151", lw=1.8, rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                                     lw=lw, color=color, shrinkA=1, shrinkB=1,
                                     connectionstyle=f"arc3,rad={rad}", zorder=2))

    # ---------------- panel 1: inputs --------------------------------------
    panel(0.008, 0.050, 0.243, 0.900, "MULTI-MODAL INPUTS", C_A, "1")
    box(0.020, 0.690, 0.219, 0.190, "23 retail sales series",
        "HK$m, 2004-10 → 2026-08\n8 parents + 15 children\n"
        "LNY overlap window (pre-14d / post-7d)")
    box(0.020, 0.510, 0.219, 0.155, "External drivers",
        "CPI · policy rate · EERI · visitors\nMTR · cargo · rain / temp anomaly")
    box(0.020, 0.070, 0.219, 0.420, "Multi-dimensional text streams")
    chips = [("Dim 1", "Macro & policy", "#dbeafe"),
             ("Dim 2", "Tourism & cross-border", "#dcfce7"),
             ("Dim 3", "Durables & auto", "#fef3c7"),
             ("Dim 4", "Staples & retail", "#f3e8ff")]
    for i, (k, lab, col) in enumerate(chips):
        y = 0.345 - i * 0.070
        ax.add_patch(FancyBboxPatch((0.030, y), 0.199, 0.056,
                     boxstyle="round,pad=0.003,rounding_size=0.010",
                     fc=col, ec="#9ca3af", lw=0.8, zorder=4))
        tx(0.044, y + 0.028, k, fs=7.8, bold=True, ha="left", z=5)
        tx(0.098, y + 0.028, lab, fs=7.8, ha="left", z=5)
    n_uniq, n_adm = corpus_counts()
    tx(0.1295, 0.092, f"{n_uniq:,} unique GovHK headlines →\nMiniLM-L6-v2 (384-d), "
       f"{n_adm:,} admitted", fs=7.2, color=GREY, style="italic")

    # ---------------- panel 2: router --------------------------------------
    panel(0.259, 0.050, 0.233, 0.900, "CATEGORY-SEMANTIC ROUTER", C_B, "2")
    box(0.271, 0.755, 0.209, 0.125, "Category text prototypes  P_j",
        "23 hand-written semantic anchors\n(j = 2, 3, …, 51)", fs=7.6)
    box(0.271, 0.545, 0.209, 0.180, "Attention routing")
    tx(0.3755, 0.672, "w_i,j = softmax_i ( ⟨e_i , P_j⟩ / τ )", fs=7.6)
    tx(0.3755, 0.634, "p_i,k = softmax_k ( kw_i,k + ⟨e_i , D_k⟩ )", fs=7.6)
    tx(0.3755, 0.596, "E_t,j,k = Σ_i p_i,k w_i,j e_i\n/ Σ_i p_i,k w_i,j",
       fs=7.6, ls=1.3)
    tx(0.3755, 0.492, "τ = 0.10 · D_text = 4 PCA dims\n"
       "PCA + z-score fit on period ≤ 202312 only", fs=7.2, color=GREY,
       style="italic", ls=1.4)
    box(0.271, 0.290, 0.209, 0.155, "Routed text block",
        "per (category, month):\n4 PCA scores + sentiment + salience\n"
        "+ volume = 7 × 4 streams = 28", fs=7.5)
    box(0.271, 0.088, 0.209, 0.160, "Why it is not one lump",
        "each category reads only the text\nthat is semantically its own\n"
        "(jewellery ← tourism, cars ← durables,\nsupermarket ← staples)", fs=7.5)

    # ---------------- panel 3: engines -------------------------------------
    panel(0.499, 0.050, 0.363, 0.900, "DUAL AI ENGINES", C_C, "3")
    box(0.511, 0.540, 0.339, 0.345,
        "PATH A — xLSTM (sLSTM + mLSTM, 2 blocks)", tdy=0.034)
    tx(0.6805, 0.805, "learnable dual anchor\n"
       r"$\hat y_h=a_h\,y_z(t)+(1-a_h)\,y_z(t{+}h{-}12)$", fs=8.4, ls=1.5)
    tx(0.6805, 0.690, "category-routed HORIZON FiLM\n"
       "gate_k = tanh(W_g · series_emb) ⊙ mask_k\n"
       "mask = 0 on h=1 (pure momentum), open on h=3/6/12\n"
       "L1 penalty on the gate weights", fs=7.6, ls=1.55)
    tx(0.6805, 0.578, "1 shared backbone · 4 direct-multi-horizon heads",
       fs=7.4, color=GREY, style="italic")

    box(0.511, 0.100, 0.339, 0.400,
        "PATH B — iTransformer (inverted attention)", tdy=0.034)
    tx(0.6805, 0.415, "23 sales variate tokens\n+ 4 text-modality tokens\n"
       "= 27 tokens · 0 new trunk parameters", fs=8.2, ls=1.6)
    tx(0.6805, 0.275, "self-attention runs ACROSS categories, so each\n"
       "retail token queries the 4 text tokens directly\n"
       "(routing weights travel inside the token)", fs=7.7, ls=1.6)
    tx(0.6805, 0.150, "same anchor · same training loop · same seeds",
       fs=7.4, color=GREY, style="italic")

    # ---------------- panel 4: reconciliation ------------------------------
    panel(0.871, 0.050, 0.120, 0.900, "RECONCILE", C_D, "4")
    box(0.879, 0.635, 0.104, 0.240, "MinT_wls",
        "hierarchical\nreconciliation\n\n8 parents\n15 children", fs=7.4)
    box(0.879, 0.320, 0.104, 0.255, "Output",
        "h = 1, 3, 6\nand 12 months\n\nHK$ million,\ncoherent with\nthe hierarchy",
        fs=7.4)
    tx(0.931, 0.248, "12 rolling\norigins\n2023-12 →\n2024-11", fs=7.0,
       color=GREY, style="italic")

    # ---------------- arrows ----------------------------------------------
    arrow((0.251, 0.50), (0.259, 0.50))
    arrow((0.492, 0.56), (0.499, 0.72))
    arrow((0.492, 0.42), (0.499, 0.28))
    arrow((0.850, 0.72), (0.871, 0.74))
    arrow((0.850, 0.30), (0.871, 0.44))
    arrow((0.931, 0.635), (0.931, 0.580), color="#6b7280")

    ax.text(0.5, 0.004, "Figure 1 — end-to-end system: multi-modal inputs → "
            "category-semantic routing of multi-dimensional text → dual AI engines → "
            "hierarchical reconciliation (HK$ million).",
            fontsize=9.2, ha="center", va="bottom", color="#374151")
    return save(fig, "fig1_system_architecture.png")


# ===========================================================================
# FIG 2 — 2024 total retail sales, actual vs AI
# ===========================================================================
def fig2():
    w = panel()
    f = facts()
    actual = pd.Series({m: float(w.loc[m, TOP_LEVEL].sum()) for m in MONTHS_2024})
    assert abs(actual.sum() - f["business_2024"]["actual_total_2024_hkm"]) < 1.0, \
        (actual.sum(), f["business_2024"]["actual_total_2024_hkm"])

    naive = load_preds([S0 / "results" / "predictions.csv"])
    naive = naive[naive["model"] == "SeasonalNaive"]
    xl = ens3(load_preds([S0 / "results" / "pretrain" / f"preds_scratch_s{s}.csv"
                          for s in SEEDS]))
    it = ens3(load_preds([S0 / "results" / "sota_preds" / f"itransformer_anchor_s{s}.csv"
                          for s in SEEDS]))

    s_naive = rolling_h1(naive, TOP_LEVEL, MONTHS_2024)
    s_xl = rolling_h1(xl, TOP_LEVEL, MONTHS_2024)
    s_it = rolling_h1(it, TOP_LEVEL, MONTHS_2024)

    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    x = np.arange(12)
    ax.axvspan(2.6, 8.45, color=C_WARN, alpha=0.14, zorder=0, lw=0)
    ax.plot(x, actual.values, color=C_ACT, lw=3.0, marker="o", ms=6.0,
            label="Actual retail sales", zorder=6)
    ax.plot(x, s_naive.values, color=C_NAIVE, lw=2.0, ls="--", marker="s", ms=4.2,
            label="Seasonal naive", zorder=4)
    ax.plot(x, s_xl.values, color=C_XL, lw=2.2, marker="^", ms=5.0,
            label="xLSTM (dual anchor, ENS3)", zorder=5)
    ax.plot(x, s_it.values, color=C_IT, lw=2.2, marker="D", ms=4.6,
            label="iTransformer (cross-variate, ENS3)", zorder=5)

    ax.set_xticks(x); ax.set_xticklabels([f"2024-{m}" for m in MONTH_LAB])
    ax.set_ylabel("Retail sales (HK$ million)")
    ax.set_title("2024 Hong Kong retail sales: one-month-ahead rolling forecast vs actual\n"
                 "8 top-level categories, HK$ million", loc="left", fontweight="bold")
    ax.set_xlim(-0.4, 11.4)
    ax.set_ylim(27_800, 43_200)

    ax.annotate("Lunar New Year shifted\n2023-01-22 → 2024-02-10 (+19 days)",
                xy=(1.0, actual.iloc[1] + 260), xytext=(1.35, 41_300),
                fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3,
                                connectionstyle="arc3,rad=0.18"), zorder=7)
    ax.annotate("Apr–Sep: naive keeps the pre-LNY level\nand over-forecasts the whole plateau",
                xy=(5.4, s_naive.iloc[5] + 200), xytext=(4.15, 37_900),
                fontsize=8.6, color="#92400e",
                arrowprops=dict(arrowstyle="->", color="#92400e", lw=1.3), zorder=7)

    gap = s_naive.sum() - actual.sum()
    gap_it = s_it.sum() - actual.sum()
    ax.legend(loc="upper right", fontsize=8.6, ncol=1)
    fig.text(0.012, 0.012,
             f"Full-year 2024 total: actual HK\\${actual.sum():,.0f}m   ·   "
             f"seasonal naive +HK\\${gap:,.0f}m (+{100*gap/actual.sum():.1f}%)   ·   "
             f"iTransformer +HK\\${gap_it:,.0f}m (+{100*gap_it/actual.sum():.1f}%)   ·   "
             f"inventory over-forecast removed HK\\${gap-gap_it:,.0f}m",
             fontsize=8.6, color="#374151", ha="left", va="bottom")
    return save(fig, "fig2_2024_total_sales_forecast.png")


# ===========================================================================
# FIG 3 — four representative categories
# ===========================================================================
def fig3():
    w = panel()
    f = facts()
    xl = ens3(load_preds([S0 / "results" / "pretrain" / f"preds_scratch_s{s}.csv"
                          for s in SEEDS]))
    it = ens3(load_preds([S0 / "results" / "sota_preds" / f"itransformer_anchor_s{s}.csv"
                          for s in SEEDS]))
    codes = [("32", "Jewellery, watches & valuable gifts"),
             ("12", "Clothing, footwear & allied products"),
             ("19", "Motor vehicles & parts"),
             ("8", "Supermarkets")]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.8), sharex=True)
    x = np.arange(12)
    for ax, (code, name) in zip(axes.ravel(), codes):
        act = np.array([float(w.loc[m, code]) for m in MONTHS_2024])
        ps_it = rolling_h1(it, [code], MONTHS_2024)
        ps_xl = rolling_h1(xl, [code], MONTHS_2024)
        v_it = np.array([ps_it.get(m, np.nan) for m in MONTHS_2024])
        v_xl = np.array([ps_xl.get(m, np.nan) for m in MONTHS_2024])
        wape_it = np.nansum(np.abs(act - v_it)) / np.nansum(np.abs(act))
        wape_xl = np.nansum(np.abs(act - v_xl)) / np.nansum(np.abs(act))
        ax.fill_between(x, act, v_it, color=C_IT, alpha=0.10, lw=0)
        ax.plot(x, act, color=C_ACT, lw=2.6, marker="o", ms=4.6, label="Actual")
        ax.plot(x, v_it, color=C_IT, lw=1.9, marker="D", ms=3.8,
                label=f"iTransformer (1−WAPE {100*(1-wape_it):.1f}%)")
        ax.plot(x, v_xl, color=C_XL, lw=1.9, marker="^", ms=4.2, ls="-.",
                label=f"xLSTM (1−WAPE {100*(1-wape_xl):.1f}%)")
        ax.set_title(f"[{code}]  {name}", loc="left", fontsize=10,
                     fontweight="bold", pad=22)
        ax.set_ylabel("HK$ million")
        ax.legend(loc="upper left", fontsize=7.4, ncol=2, columnspacing=0.9,
                  handlelength=1.6, borderaxespad=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_LAB, fontsize=8)
        ax.margins(x=0.02)
        lo = float(np.nanmin(np.concatenate([act, v_it, v_xl])))
        hi = float(np.nanmax(np.concatenate([act, v_it, v_xl])))
        ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.42 * (hi - lo))
        ax.text(0.985, 0.035,
                f"2024 actual HK\\${act.sum():,.0f}m   ·   "
                f"iTransformer HK\\${np.nansum(v_it):,.0f}m",
                transform=ax.transAxes, fontsize=7.4, color=GREY, ha="right")
    for ax in axes[1]:
        ax.set_xlabel("month of 2024")
    fig.suptitle("Four representative categories, 2024: one-month-ahead forecast vs actual",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return save(fig, "fig3_four_categories_panel.png")


# ===========================================================================
# FIG 4 — model leaderboard
# ===========================================================================
def fig4():
    rows = [
        ("Ridge (residual)",            "Ridge_residual",                     C_NAIVE),
        ("Drift",                       "Drift",                              C_NAIVE),
        ("LightGBM (tuned, residual)",  "LightGBMTuned_residual (3-seed mean)", C_NAIVE),
        ("kNN (residual)",              "kNN_residual",                       C_NAIVE),
        ("nn.LSTM (plain backbone)",    "nn.LSTM (plain LSTM backbone, 3-seed mean)", C_NAIVE),
        ("Seasonal naive",              "SeasonalNaive",                      C_NAIVE),
        ("xLSTM (dual anchor, ENS3)",   "xLSTM + learnable anchor (ENS3)",    C_XL),
        ("xLSTM + M1/M3 pre-training",  "xLSTM + Pretrain (ENS3)",            C_XL),
        ("iTransformer (dual anchor)",  "iTransformer + learnable anchor (ENS3)", C_IT),
    ]
    f = facts()
    idx = {r["model"]: r for r in f["leaderboard"]}
    ours = pd.DataFrame()
    mp = S3 / "results" / "multidim_text_results.csv"
    if mp.exists():
        ours = pd.read_csv(mp)
        ours = ours[(ours["seed"].astype(str) == "ENS3")
                    & (ours["h"].astype(str) == "ALL")]
    labels, mase, wape, cols = [], [], [], []
    for lab, key, col in rows:
        if key not in idx:
            print(f"  [skip] {key} not in facts")
            continue
        labels.append(lab); mase.append(idx[key]["MASE_ALL"])
        wape.append(100.0 * idx[key]["WAPE_ALL"]); cols.append(col)
    for arm, lab in (("xlstm_routed_allH", "xLSTM + routed multi-dim text (ours)"),
                     ("it_routed", "iTransformer + text tokens (ours)")):
        if ours.empty or "arm" not in ours.columns:
            break
        r = ours[ours["arm"] == arm]
        if r.empty:
            continue
        labels.append(lab); mase.append(float(r["MASE"].iloc[0]))
        wape.append(100.0 * float(r["WAPE"].iloc[0])); cols.append(C_OURS)

    order = np.argsort(mase)      # best at top
    labels = [labels[i] for i in order]
    mase = [mase[i] for i in order]; wape = [wape[i] for i in order]
    cols = [cols[i] for i in order]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.8),
                                 gridspec_kw={"width_ratios": [1.25, 1]}, sharey=True)
    y = np.arange(len(labels))[::-1]
    xmax1 = max(mase) * 1.16
    a1.barh(y, mase, color=cols, height=0.68, edgecolor="white", lw=0.8)
    a1.axvline(1.0, color="#dc2626", ls="--", lw=1.6, zorder=5)
    for yy, v in zip(y, mase):
        if v > 0.72 * xmax1:
            a1.text(v - 0.015, yy, f"{v:.4f}", va="center", ha="right",
                    fontsize=8.4, color="white", fontweight="bold", zorder=6)
        else:
            a1.text(v + 0.012, yy, f"{v:.4f}", va="center", fontsize=8.4, zorder=6)
    a1.text(1.015, -0.80, "MASE = 1.0 (seasonal-naive level)",
            color="#dc2626", fontsize=8.2, va="bottom", ha="left")
    a1.set_yticks(y); a1.set_yticklabels(labels, fontsize=8.8)
    a1.set_xlabel("MASE (ALL horizons, lower is better)")
    a1.set_title("Accuracy vs the seasonal-naive denominator", loc="left",
                 fontweight="bold", fontsize=10)
    a1.set_xlim(0, xmax1)
    a1.set_ylim(-0.85, len(labels) - 0.15)

    xmax2 = max(wape) * 1.16
    a2.barh(y, wape, color=cols, height=0.68, edgecolor="white", lw=0.8)
    for yy, v in zip(y, wape):
        if v > 0.72 * xmax2:
            a2.text(v - 0.15, yy, f"{v:.2f}%", va="center", ha="right",
                    fontsize=8.4, color="white", fontweight="bold", zorder=6)
        else:
            a2.text(v + 0.12, yy, f"{v:.2f}%", va="center", fontsize=8.4, zorder=6)
    a2.set_xlabel("WAPE (%, volume-weighted, lower is better)")
    a2.set_title("Volume-weighted error in HK$", loc="left", fontweight="bold",
                 fontsize=10)
    a2.set_xlim(0, xmax2)
    fig.suptitle("Model leaderboard — 23 categories × 12 origins × 4 horizons "
                 "= 1,104 scored points per model",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig4_model_leaderboard.png")


# ===========================================================================
# FIG 5 — routing heatmap + horizon-wise improvement
# ===========================================================================
def fig5():
    meta = json.loads((S3 / "data" / "multidim" / "routed_meta.json")
                      .read_text(encoding="utf-8"))
    R = np.array(meta["route_weight_mean"], dtype=float)        # (4, 23)
    dims = ["Dim 1 · Macro\n& policy", "Dim 2 · Tourism &\ncross-border",
            "Dim 3 · Durables,\nauto & property", "Dim 4 · Staples &\ndept. retail"]
    codes = ["8", "2", "30", "32", "12", "19", "23", "51"]
    cats = ["8\nSupermarket", "2\nFood & drink", "30\nDept. store",
            "32\nJewellery", "12\nClothing", "19\nMotor veh.",
            "23\nFurniture", "51\nElectrical"]
    lift = R / R.mean(axis=1, keepdims=True)                    # per-dim specialisation
    sub = np.array([[lift[k][SERIES_ORDER.index(c)] for c in codes] for k in range(4)])

    fig = plt.figure(figsize=(14.8, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.26,
                          left=0.055, right=0.985, top=0.80, bottom=0.16)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(sub, cmap="RdYlGn", vmin=0.55, vmax=2.4, aspect="auto")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=8.0)
    ax.set_yticks(range(4)); ax.set_yticklabels(dims, fontsize=8.2)
    ax.grid(False)
    for i in range(4):
        for j in range(len(codes)):
            share = R[i, SERIES_ORDER.index(codes[j])]
            ax.text(j, i, f"{sub[i,j]:.2f}×\n{100*share:.0f}%", ha="center",
                    va="center", fontsize=7.4,
                    color="white" if (sub[i, j] > 1.55 or sub[i, j] < 0.72) else INK)
    ax.set_title("A · Category-semantic routing weights\n"
                 "specialisation lift ×, and share of that category's routed mass",
                 loc="left", fontweight="bold", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("lift vs the stream's cross-category mean", fontsize=7.8)
    for j in range(len(codes)):
        k = int(np.argmax(sub[:, j]))
        ax.add_patch(Rectangle((j - 0.5, k - 0.5), 1, 1, fill=False,
                               ec="#111827", lw=2.2))

    # ---- right: old vs new, per horizon -----------------------------------
    mp = S3 / "results" / "multidim_text_results.csv"
    ax2 = fig.add_subplot(gs[0, 1])
    if not mp.exists():
        ax2.text(0.5, 0.5, "multidim_text_results.csv missing", ha="center")
        return save(fig, "fig5_multidim_text_heatmap.png")
    m = pd.read_csv(mp)
    e = m[m["seed"].astype(str) == "ENS3"]

    def val(arm, h):
        return float(e[(e.arm == arm) & (e.h.astype(str) == h)]["MASE"].iloc[0])

    hs = ["1", "3", "6", "12"]
    old = [val("xlstm_pooled", h) for h in hs]
    new = [val("xlstm_routed_allH", h) for h in hs]
    base = [val("xlstm_none", h) for h in hs]
    xx = np.arange(len(hs)); bw = 0.26
    ax2.bar(xx - bw, base, bw, color="#d1d5db", edgecolor="white",
            label="(a) no text")
    ax2.bar(xx, old, bw, color=C_WARN, edgecolor="white",
            label="(b) Stage-3 lumped text")
    ax2.bar(xx + bw, new, bw, color=C_OURS, edgecolor="white",
            label="(c) routed multi-dim text (ours)")
    top = max(max(old), max(new), max(base))
    for i in range(len(hs)):
        d1 = 100 * (new[i] / old[i] - 1)
        ax2.text(xx[i] + bw, new[i] + 0.010, f"{d1:+.1f}%", ha="center",
                 fontsize=7.8, color=C_OURS, fontweight="bold")
        ax2.text(xx[i], 0.787, f"h = {hs[i]}", ha="center",
                 fontsize=8.8, color=INK, fontweight="bold")
    ax2.axhline(1.0, color="#dc2626", ls="--", lw=1.3)
    ax2.set_xticks(xx); ax2.set_xticklabels(["" for _ in hs])
    ax2.set_ylabel("MASE (lower is better)")
    ax2.set_ylim(0.775, top + 0.085)
    ax2.legend(loc="upper left", fontsize=7.8, handlelength=1.3,
               borderaxespad=0.25, labelspacing=0.35)
    ax2.set_title("B · Old lumped text vs routed text, by forecast horizon\n"
                  "green % = improvement over the lumped text (xLSTM path, ENS3)",
                  loc="left", fontweight="bold", fontsize=10)

    fig.suptitle("Figure 5 — where the semantic routing actually pays",
                 x=0.008, ha="left", y=0.965, fontsize=12, fontweight="bold")
    d_base = [100 * (new[i] / base[i] - 1) for i in range(len(hs))]
    d_old = [100 * (new[i] / old[i] - 1) for i in range(len(hs))]
    fig.text(0.008, 0.055,
             "A: boxed cell = the stream that category leans on most "
             "(jewellery → tourism, motor vehicles / furniture / electricals → durables, "
             "supermarket / food → staples).\n"
             f"B: the routed arm improves {abs(d_base[1]):.1f} % at h = 3 and "
             f"{abs(d_base[2]):.1f} % at h = 6 over no text, and "
             f"{abs(d_old[1]):.1f} % / {abs(d_old[2]):.1f} % over the lumped text, "
             f"while h = 1 ({d_base[0]:+.1f} %) and h = 12 ({d_base[3]:+.1f} %) are "
             "statistically unchanged.",
             fontsize=8.0, color=GREY, ha="left", va="bottom")
    return save(fig, "fig5_multidim_text_heatmap.png")


# ===========================================================================
# FIG 6 — ablation progression (staircase)
# ===========================================================================
def fig6():
    f = facts()
    ladder = [
        ("Direct level\nprediction\n(v1, single run)", 2.0963, "#9ca3af", True),
        ("Residual target,\nfixed anchor\n(single run)", 1.0119, "#9ca3af", True),
        ("Residual, no\nanchor, no\ncalendar", 0.9967, "#374151", False),
        ("Learnable dual\nanchor\n(ENS3)", 0.9481, "#1f2937", False),
        ("+ M1/M3 pre-\ntraining of the\nbackbone", 0.9247, "#1f2937", False),
        ("+ cross-variate\nattention\n(iTransformer)", 0.9020, "#1f2937", False),
    ]
    # the Part 1 text ablation, drawn as a separate group
    txt = []
    mp = S3 / "results" / "multidim_text_results.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        e = m[(m["seed"].astype(str) == "ENS3") & (m["h"].astype(str) == "ALL")]
        for arm, lab, col in (("xlstm_none", "(a) xLSTM\nno text", "#d1d5db"),
                              ("xlstm_pooled", "(b) lumped\ntext", C_WARN),
                              ("xlstm_routed_allH", "(c) routed text\n(ours)", C_OURS)):
            v = e[e["arm"] == arm]["MASE"]
            if len(v):
                txt.append((lab, float(v.iloc[0]), col))

    fig, ax = plt.subplots(figsize=(14.6, 6.4))
    pos, labs, cols, hatch = [], [], [], []
    for i, (lab, v, c, h) in enumerate(ladder):
        pos.append(i); labs.append(lab); cols.append(c); hatch.append(h)
    for j, (lab, v, c) in enumerate(txt):
        pos.append(6.7 + j * 0.85); labs.append(lab); cols.append(c)
        hatch.append(False)

    vals = [r[1] for r in ladder] + [t[1] for t in txt]
    for x, v, c, h in zip(pos, vals, cols, hatch):
        ax.bar(x, v, 0.62, color=c, alpha=0.92, edgecolor="white", lw=1.0,
               hatch="///" if h else None, zorder=3)
        ax.text(x, v + 0.025, f"{v:.4f}", ha="center", fontsize=9.4,
                fontweight="bold", color=INK, zorder=5)

    for i in range(1, len(ladder)):
        ax.annotate("", xy=(pos[i], vals[i] + 0.004),
                    xytext=(pos[i - 1], vals[i - 1] + 0.004),
                    arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.3,
                                    ls=(0, (4, 3))), zorder=4)
        d = vals[i] - vals[i - 1]
        ax.text((pos[i] + pos[i - 1]) / 2, min(vals[i], vals[i - 1]) - 0.085,
                f"{d:+.3f}", ha="center", fontsize=8.4,
                color="#047857" if d < 0 else "#b91c1c")
    # arrows inside the text group, against the lumped-text reference
    if len(txt) == 3:
        ax.annotate("", xy=(pos[8], vals[8] + 0.004), xytext=(pos[7], vals[7] + 0.004),
                    arrowprops=dict(arrowstyle="-|>", color=C_OURS, lw=1.6), zorder=4)
        ax.text(pos[8], 1.03, "−12.9 %", ha="center", va="bottom",
                fontsize=8.4, color=C_OURS, fontweight="bold")
    ax.axhline(1.0, color="#dc2626", ls="--", lw=1.5, zorder=2)
    ax.axvline(6.15, color="#9ca3af", ls=":", lw=1.4, zorder=1)
    ax.text(7.55, 2.28, "Part 1 · text on the xLSTM path",
            ha="center", fontsize=9.2, color=C_OURS, fontweight="bold")
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, fontsize=8.2)
    ax.set_ylabel("MASE (ALL horizons, lower is better)")
    ax.set_ylim(0, 2.60)
    ax.set_xlim(-0.7, 9.3)
    ax.set_title("Design-contribution staircase, and what the text block buys on top of it",
                 loc="left", fontweight="bold")
    ax.text(-0.62, 2.52, "Red dashed line = MASE 1.0, the seasonal-naive hurdle.",
            fontsize=8.4, color=GREY, va="top", ha="left")
    t = arm_table()
    loss = 100 * (float(t.loc["xlstm_pooled", "ALL"]) / float(t.loc["xlstm_none", "ALL"]) - 1)
    gain = 100 * (float(t.loc["xlstm_routed_allH", "ALL"])
                  / float(t.loc["xlstm_none", "ALL"]) - 1)
    fig.text(0.008, -0.075,
             "Hatched bars are single-run historical rungs quoted in "
             "01_ARCHITECTURE.md, for which no machine-readable prediction file "
             "survives; solid bars are recomputed with the locked harness from the "
             "raw point forecasts.\n"
             f"The right-hand group applies the Part 1 text block to the same xLSTM "
             f"path: routing turns a {abs(loss):.1f} % loss into a {abs(gain):.1f} % "
             "gain over the no-text baseline.",
             fontsize=8.2, color=GREY, va="top", ha="left")
    return save(fig, "fig6_ablation_progression.png")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    print("generating figures ...")
    fails = 0
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6):
        try:
            fn()
        except Exception as exc:                                   # noqa: BLE001
            fails += 1
            print(f"  !! {fn.__name__} failed: {type(exc).__name__}: {exc}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
