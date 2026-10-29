"""Does the product-category split add up to the parent series?

Checks the two C&SD tables that could genuinely grow the retail panel:
  620-67011  supermarkets (OUTLET_TYPE=8) by broad product category
  620-67012  department stores (OUTLET_TYPE=30) by broad product category

For each: sum the children and compare with (a) the table's own Total row and
(b) the parent series in stage0/data/panel_wide.csv.  Only if the identity
holds exactly can these series be appended to the project's HIER hierarchy
(MinT_wls build_A) without breaking reconciliation.

Run: stage0\\.venv\\Scripts\\python.exe stage0\verify_subsplits.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "data_external" / "raw" / "censtatd"
CASES = [("620-67011", "8", "supermarkets"),
         ("620-67012", "30", "department stores"),
         # aggregate columns that duplicate the sum of the D* children
         ("620-67012", "30", "department stores")]
AGGREGATE_CODES = {"D1_D6"}


def table(tid: str) -> pd.DataFrame:
    d = json.loads((RAW / f"{tid}.json").read_text(encoding="utf-8"))
    rows = [x for x in d["dataSet"] if x.get("freq") == "M" and x.get("sv") == "VAL_RS"]
    df = pd.DataFrame(rows)
    df["v"] = pd.to_numeric(df["figure"], errors="coerce")
    df = df.dropna(subset=["v"])
    return df.pivot_table(index="period", columns="BROAD_PROD_CAT", values="v", aggfunc="first")


def stats(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    d = (a - b).abs()
    return round(float(d.max()), 4), round(float(d.median()), 4), int(d.notna().sum())


def main() -> int:
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0)
    wide.index = wide.index.astype(str)
    seen = set()
    for tid, parent, label in CASES:
        if (tid, parent) in seen:
            continue
        seen.add((tid, parent))
        p = table(tid)
        total = p.pop("") if "" in p.columns else None
        kids = [c for c in p.columns if c not in AGGREGATE_CODES]
        s = p[kids].sum(axis=1, min_count=1)
        par = wide[parent].reindex(p.index)
        print(f"{tid}  {label}  (parent series {parent})")
        print(f"  categories      : {sorted(p.columns)}")
        print(f"  children summed : {sorted(kids)}")
        print(f"  sum(kids) vs table Total : max={stats(s, total)[0]}  median={stats(s, total)[1]}  n={stats(s, total)[2]}")
        print(f"  sum(kids) vs panel parent: max={stats(s, par)[0]}  median={stats(s, par)[1]}  n={stats(s, par)[2]}")
        print(f"  Total row  vs panel parent: max={stats(total, par)[0]}  median={stats(total, par)[1]}  n={stats(total, par)[2]}")
        print(f"  mean ratio sum/Total     : {round(float((s / total).mean()), 6)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
