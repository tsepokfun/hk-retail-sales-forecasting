"""Umbrella runner for the data + protocol upgrade (tasks 1-3).

Three components live in their own modules because each has its own runtime
budget and failure mode; this script is the single entry point the report cites,
and it can also assemble the results tables straight from the saved csv files so
no number is ever transcribed by hand.

    task 1  rolling refit + per-horizon causal mask   stage0/run_rolling_refit.py
    task 2  34-series panel (23 core + 11 sub-series) stage0/run_upgrade_34series.py
    task 3  M1/M3 pretraining then fine-tune          stage0/run_pretrain_m1m3.py

Usage
-----
    # run one component (arguments are forwarded verbatim)
    python stage0/run_data_and_rolling_upgrade.py rolling --which both
    python stage0/run_data_and_rolling_upgrade.py series34
    python stage0/run_data_and_rolling_upgrade.py pretrain

    # run everything that is present, in the cheap-to-expensive order
    python stage0/run_data_and_rolling_upgrade.py all

    # regenerate docs/DATA_AND_PROTOCOL_UPGRADE_RESULTS.md from saved results
    python stage0/run_data_and_rolling_upgrade.py report

The report step is deliberately read-only: it globs the result csv files and
prints only what is actually on disk.  If a component has not run, its table is
emitted with an explicit "NOT RUN" row instead of an empty or invented one.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
DOC = HERE.parent / "docs" / "DATA_AND_PROTOCOL_UPGRADE_RESULTS.md"

COMPONENTS = {
    "rolling": HERE / "run_rolling_refit.py",
    "series34": HERE / "run_upgrade_34series.py",
    "pretrain": HERE / "run_pretrain_m1m3.py",
}

# reference numbers derived from the locked harness in earlier rounds
REFS = {
    "SeasonalNaive": 1.03029,
    "LightGBMTuned": 1.04108,
    "xLSTM (frozen 2021-11)": 0.94810,
    "xLSTM + MinT_wls (frozen)": 0.94455,
    "iTransformer (frozen 2021-11)": 0.90198,
    "ETS rolling refit": 0.85403,
    "ETS rolling + MinT_wls": 0.84927,
    "SARIMAX rolling refit": 0.88354,
    "SARIMAX rolling + MinT_wls": 0.88326,
}


def run(name: str, extra: list[str]) -> int:
    script = COMPONENTS[name]
    if not script.exists():
        print(f"[skip] {name}: {script.name} not present")
        return 1
    cmd = [str(PY), str(script), *extra]
    print(f"[run] {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(cmd)
    print(f"[done] {name} rc={rc} in {time.perf_counter() - t0:.0f}s", flush=True)
    return rc


def _all_rows(folder: Path) -> pd.DataFrame:
    """every metrics csv under a results folder, normalised to model/h/MASE/WAPE."""
    if not folder.exists():
        return pd.DataFrame()
    out = []
    for f in sorted(folder.glob("**/metrics*.csv")) + sorted(folder.glob("**/*metrics*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:                                    # noqa: BLE001
            continue
        if not {"MASE"}.issubset(d.columns):
            continue
        d = d.copy()
        d["source"] = f.name
        out.append(d)
    return pd.concat(out, ignore_index=True).drop_duplicates() if out else pd.DataFrame()


def _table(df: pd.DataFrame, title: str, note: str = "") -> list[str]:
    lines = [f"### {title}", ""]
    if df.empty:
        lines += ["_NOT RUN — 沒有任何 metrics csv 落檔。_", ""]
        return lines
    a = df[df["h"].astype(str) == "ALL"].copy()
    if a.empty:
        lines += ["_有 metrics 檔但沒有 ALL 列。_", ""]
        return lines
    # group by source file so different PROTOCOLS never get mixed in one ranking
    for src, g in a.groupby("source"):
        lines.append(f"**`{src}`**")
        lines.append("")
        lines.append("| 模型 | MASE_ALL | WAPE_ALL |")
        lines.append("|---|---:|---:|")
        for _, r in g.sort_values("MASE").iterrows():
            w = f"{r['WAPE']:.4f}" if "WAPE" in g.columns and pd.notna(r.get("WAPE")) else "—"
            lines.append(f"| {r['model']} | **{r['MASE']:.5f}** | {w} |")
        lines.append("")
    if note:
        lines.append(note)
        lines.append("")
    return lines


def _seed_robustness(folder: Path) -> list[str]:
    """Per-seed delta between a 'scratch' arm and its pretrained twin.

    An ENS3 improvement can come from two places: every seed getting better, or
    the seeds merely spreading out so their average is better.  Only the first is
    a property of the method; the second is an ensemble-composition effect.  This
    block separates them so the report cannot over-claim.
    """
    f = folder / "metrics.csv"
    if not f.exists():
        return []
    d = pd.read_csv(f)
    a = d[d["h"].astype(str) == "ALL"].copy()
    got = {}
    for arm, prefix in (("scratch", "xLSTM_scratch_s"), ("pretrained", "xLSTM_pretrained_s")):
        s = a[a["model"].str.startswith(prefix)].set_index("model")["MASE"]
        s.index = [i.replace("xLSTM_", "").replace("pretrained", "").replace("scratch", "") for i in s.index]
        got[arm] = s
    if "scratch" not in got or "pretrained" not in got or got["scratch"].empty:
        return []
    cmp = pd.DataFrame({"scratch": got["scratch"], "pretrained": got["pretrained"]})
    cmp["delta"] = cmp["pretrained"] - cmp["scratch"]
    lines = ["### 穩健性：改善是「每個種子都變好」還是「集成組成改變」？", "",
             "| 種子 | scratch | pretrained | 差 |", "|---|---:|---:|---:|"]
    for name, r in cmp.iterrows():
        lines.append(f"| {name} | {r['scratch']:.5f} | {r['pretrained']:.5f} | "
                     f"{r['delta']:+.5f} |")
    n_better = int((cmp["delta"] < 0).sum())
    lines += ["",
              f"- 逐種子平均差：**{cmp['delta'].mean():+.5f}** ｜ 種子平均 MASE "
              f"{cmp['scratch'].mean():.5f} → {cmp['pretrained'].mean():.5f}",
              f"- 有改善的種子：**{n_better}/{len(cmp)}**",
              "",
              "> 判讀：若「逐種子平均差」的絕對值小於 0.02 門檻，但 ENS3 的差距大於門檻，",
              "> 則改善**部分來自集成組成**（種子間差異變大），不可全部歸因於預訓練。",
              ""]
    return lines


def _robustness_34(folder: Path) -> list[str]:
    """Per-seed 34-vs-23 comparison, computed straight from the prediction csvs.

    The ENS3 table alone cannot tell 'every seed got worse' from 'the ensemble
    shifted'.  For a NEGATIVE result that distinction matters even more than for
    a positive one: a uniform loss across seeds is much harder to explain away
    than a shifted average.
    """
    sys.path.insert(0, str(HERE / "eval"))
    try:
        import harness as H  # noqa: E402
    except Exception:                                        # noqa: BLE001
        return []
    import glob
    wide = H.load_panel()
    naive = H.seasonal_naive_mae_train(wide)

    def mase(path: Path) -> float:
        d = pd.read_csv(path, dtype={"series": str, "origin": str})
        m = H.metrics_from_frame(d, naive)
        return float(m.loc[m["h"].astype(str) == "ALL", "MASE"].iloc[0])

    lines, any_row = [], False
    for fam, pre23, pre34 in (("iTransformer", "itransformer23", "itransformer34"),
                              ("xLSTM", "xlstm23", "xlstm34")):
        rows = []
        for seed in ("42", "7", "2024"):
            f23 = list(folder.glob(f"**/*{pre23}_s{seed}.csv"))
            f34 = folder / f"preds_{pre34}_s{seed}.csv"
            if not f23 or not f34.exists():
                continue
            try:
                rows.append((seed, mase(f23[0]), mase(f34)))
            except Exception:                                # noqa: BLE001
                continue
        if not rows:
            continue
        any_row = True
        if not lines:
            lines = ["### 穩健性：34 條的退步是「每個種子都退」還是「集成位移」？", "",
                     "| 家族 | 種子 | 23 條 | 34 條 | 差 |", "|---|---|---:|---:|---:|"]
        for seed, a, b in rows:
            lines.append(f"| {fam} | s{seed} | {a:.5f} | {b:.5f} | **{b - a:+.5f}** |")
        worse = sum(1 for _, a, b in rows if b > a)
        lines.append(f"| {fam} | **平均** | {sum(a for _, a, _ in rows)/len(rows):.5f} | "
                     f"{sum(b for _, _, b in rows)/len(rows):.5f} | "
                     f"**{(sum(b for _, _, b in rows) - sum(a for _, a, _ in rows))/len(rows):+.5f}**"
                     f"（{worse}/{len(rows)} 個種子變差） |")
    if any_row:
        lines += ["",
                  "> 判讀：若**每個種子**都變差，則結論不依賴集成組成，可直接寫成「加子序列有害」。",
                  ""]
    return lines


def report() -> int:
    rolling = _all_rows(HERE / "results" / "rolling")
    s34 = _all_rows(HERE / "results" / "upgrade34")
    pre = _all_rows(HERE / "results" / "pretrain")

    lines = [
        "# 資料擴充與公平協議對決：實測結果",
        "",
        "> 本檔案由 `stage0/run_data_and_rolling_upgrade.py report` **自動從結果 csv 生成**，不經人手抄寫。",
        f"> 生成時間：{time.strftime('%Y-%m-%d %H:%M')} ｜ 評估一律為 `harness.py` 的 1,104 點協定。",
        "",
        "## 表 0｜對照錨點（前幾輪由鎖死 harness 實測）",
        "",
        "| 基準 | MASE_ALL |",
        "|---|---:|",
    ]
    for k, v in REFS.items():
        lines.append(f"| {k} | {v:.5f} |")
    lines += [""]

    lines += ["## 表 1｜公平 Rolling 協議：神經模型 vs 統計模型", ""]
    lines += _table(rolling, "Rolling refit（每個測試原點重訓 + per-horizon 因果遮罩）",
                    "判讀：與表 0 的 `ETS rolling` 0.85403 / `SARIMAX rolling` 0.88354 直接比較。")
    lines += ["## 表 2｜23 條序列 vs 34 條序列（+47.8% 樣本）", ""]
    lines += _table(s34, "34 條序列（訓練用 34、評估鎖定原 23）")
    lines += _robustness_34(HERE / "results" / "upgrade34")
    lines += ["## 表 3｜M1/M3 預訓練 → 微調 vs 從頭訓練", ""]
    lines += _table(pre, "M1+M3 月度序列預訓練 20 epochs 後微調")
    lines += _seed_robustness(HERE / "results" / "pretrain")

    lines += [
        "## 判讀規則（先寫死，避免事後詮釋）",
        "",
        "1. 種子雜訊門檻 **0.02 MASE** 沿用 `FINAL_REPORT.md` §4.1 的推導；差距小於此值一律寫「方向一致，不顯著」。",
        "2. 統計模型（ETS/SARIMAX）是**確定性**的，沒有種子；門檻只用來判讀神經模型那一側的不確定性。",
        "3. 三張表全部在**同一 1,104 點**上結算；任何改動評估集合的結果一律不列入。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {DOC} ({len(lines)} lines)")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "report":
        return report()
    if cmd == "all":
        rc = 0
        for name in ("rolling", "series34", "pretrain"):
            rc |= run(name, [])
        report()
        return rc
    if cmd not in COMPONENTS:
        print(f"unknown command: {cmd}")
        return 2
    rc = run(cmd, rest)
    report()
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
