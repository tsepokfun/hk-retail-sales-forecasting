"""Audit the downloaded C&SD tables: how many monthly series does each really
carry, do they overlap, and does the `panel_expansion.json` figure (47,858
cells) survive de-duplication?

Writes  data_external/tables_audit.json  and  docs/DATA_TABLE_AUDIT.md
Read-only with respect to stage0/data.

Run: stage0\\.venv\Scripts\python.exe stage0\audit_censtatd_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXT = ROOT.parent / "data_external"
RAW = EXT / "raw" / "censtatd"
SKIP = {"_enumeration_probe.json", "_validation.json", "table_index_en.json",
        "WebTableList.json"}
META = {"freq", "period", "figure", "sd_value", "svDesc", "sv"}
WINDOW = ("200410", "202608")


def series_key(row: dict) -> tuple:
    """the identifying code(s) of a series: every field that is not metadata
    and not a *Desc label (the label duplicates the code)."""
    return tuple(sorted((k, str(v)) for k, v in row.items()
                        if k not in META and not k.endswith("Desc")))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def audit(f: Path) -> dict:
    d = json.loads(f.read_text(encoding="utf-8"))
    rows = d.get("dataSet", [])
    hdr = d.get("header", {})
    monthly = [r for r in rows if r.get("freq") == "M" and num(r.get("figure")) is not None]
    svs = sorted({r.get("sv") for r in monthly if r.get("sv")})
    by_sv = {}
    for sv in svs:
        sub = [r for r in monthly if r.get("sv") == sv]
        keys = {series_key(r) for r in sub}
        per = {str(r["period"]) for r in sub}
        inwin = {p for p in per if WINDOW[0] <= p <= WINDOW[1]}
        by_sv[sv] = {"n_series": len(keys), "n_periods": len(per),
                     "n_periods_in_window": len(inwin),
                     "cells_in_window": len(keys) * len(inwin),
                     "series": sorted("|".join(v for _, v in k) for k in keys),
                     "p_min": min(per) if per else None, "p_max": max(per) if per else None}
    return {"table": f.stem, "title": str(hdr.get("title", ""))[:110],
            "rows_total": len(rows), "rows_monthly": len(monthly),
            "sv": by_sv}


def main() -> int:
    files = sorted(p for p in RAW.glob("*.json") if p.name not in SKIP)
    out = [audit(f) for f in files]

    # --- de-duplication: a table whose series list is a SUBSET of another
    # table's (same sv) contributes nothing new to the panel.
    for sv_probe in ("VAL_RS", "VOL_IDX_RS", "VAL_IDX_RS"):
        cands = [(t["table"], s[sv_probe]) for t in out for s in [t["sv"]] if sv_probe in s]
        for i, (na, a) in enumerate(cands):
            for j, (nb, b) in enumerate(cands):
                if i >= j:
                    continue
                sa, sb = set(a["series"]), set(b["series"])
                if sa and sb and sa <= sb:
                    a.setdefault("subset_of", []).append(nb)
                elif sa and sb and sb <= sa:
                    b.setdefault("subset_of", []).append(na)

    total_cells = sum(s["cells_in_window"] for t in out for s in t["sv"].values())
    json.dump({"window": WINDOW, "tables": out, "total_cells_raw": total_cells},
              (EXT / "tables_audit.json").open("w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    lines = ["# C&SD 已下載表：真實序列數、期間、重疊審計", "",
             f"> 產生：`stage0/audit_censtatd_tables.py` ｜ 視窗 {WINDOW[0]}–{WINDOW[1]} ｜ 只計 `freq=='M'` 且 `figure` 可解析為數值的列", "",
             "| 表號 | 標題 | sv | 序列數 | 期間數(窗內) | **窗內格數** | 期間範圍 | 被誰包含 |",
             "|---|---|---|---:|---:|---:|---|---|"]
    for t in sorted(out, key=lambda x: -max([s["cells_in_window"] for s in x["sv"].values()] or [0])):
        for sv, s in t["sv"].items():
            sub = ",".join(s.get("subset_of", [])) or "—"
            lines.append(f"| `{t['table']}` | {t['title'][:42]} | {sv} | {s['n_series']} | "
                         f"{s['n_periods_in_window']} | **{s['cells_in_window']}** | "
                         f"{s['p_min']}–{s['p_max']} | {sub} |")
    lines += ["", f"**原始合計（未去重）：{total_cells} 格**", "",
              "> 判讀規則：`被誰包含` 非空者，其序列集合是另一張表的子集，",
              "> 納入面板時不應重複計算。實際可用擴張必須扣掉這些。", ""]
    (ROOT.parent / "docs" / "DATA_TABLE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"tables audited: {len(out)}   raw cells in window: {total_cells}")
    for t in sorted(out, key=lambda x: -max([s["cells_in_window"] for s in x["sv"].values()] or [0]))[:12]:
        for sv, s in t["sv"].items():
            print(f"  {t['table']:14s} {sv:11s} series={s['n_series']:4d} "
                  f"periods={s['n_periods_in_window']:3d} cells={s['cells_in_window']:6d} "
                  f"subset_of={s.get('subset_of', [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
