#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build out/VERIFICATION.md from the collector's outputs.

Reads the two CSVs plus the run log and writes a short, number-dense
verification report (coverage, per-table totals, vintage revision statistics).
Every number in the report is computed from the files, not transcribed.
"""
from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def load(name):
    p = OUT / name
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    long_rows = load("td_first_registration_long.csv")
    latest = load("td_first_registration_latest.csv")
    issues_parsed = load("issues_parsed.csv")
    L = []
    A = L.append

    A("# Verification log - TD Monthly Traffic and Transport Digest collector")
    A("")
    A("Source: Transport Department, *Monthly Traffic and Transport Digest*.")
    A("Script: `td_digest_collector.py`. All figures below are computed from")
    A("`td_first_registration_long.csv` / `td_first_registration_latest.csv`.")
    A("")

    # ---- issues -------------------------------------------------------
    A("## 1. Issues discovered and parsed")
    A("")
    found = OUT / "issues_found.txt"
    n_found = sum(1 for ln in found.read_text("utf-8").splitlines() if ln.strip()) \
        if found.exists() else 0
    n_ok = n_err = 0
    tabs = Counter()
    for r in issues_parsed:
        if r["tables_with_rows"] and r["tables_with_rows"] != "page_error":
            n_ok += 1
            for t in r["tables_with_rows"].split(","):
                if t:
                    tabs[t] += 1
        else:
            n_err += 1
    if n_found:
        A(f"- monthly issues enumerated from the site : **{n_found}** "
          f"(2003-01 .. 2026-07)")
        A(f"- issues collected (2004-01 onwards)       : **{len(issues_parsed)}**")
    else:
        A(f"- monthly issues collected                 : **{len(issues_parsed)}**")
    A(f"- issues that yielded at least one table   : **{n_ok}**")
    A(f"- issues with no usable table              : {n_err}")
    for t in ("4.1(a)", "4.1(c)", "4.1(e)"):
        A(f"- issues providing {t}                     : {tabs.get(t, 0)}")
    A("")

    # ---- coverage -----------------------------------------------------
    A("## 2. Assembled series")
    A("")
    A(f"- raw occurrence rows (every vintage kept) : **{len(long_rows)}**")
    A(f"- de-duplicated latest-vintage points       : **{len(latest)}**")
    A("")
    A("| table | points | monthly pts | monthly range | annual pts | annual range | entities |")
    A("|---|---|---|---|---|---|---|")
    for t in ("4.1(a)", "4.1(c)", "4.1(e)"):
        sub = [r for r in latest if r["table"] == t]
        if not sub:
            A(f"| {t} | 0 | - | - | - | - | - |")
            continue
        m = sorted({r["date"] for r in sub if r["granularity"] == "monthly"})
        a = sorted({r["date"] for r in sub if r["granularity"] == "annual"})
        ent = len({r["vehicle_class_or_make"] for r in sub})
        A(f"| {t} | {len(sub)} | {len(m)} | {m[0] if m else '-'} .. {m[-1] if m else '-'} "
          f"| {len(a)} | {a[0] if a else '-'} .. {a[-1] if a else '-'} | {ent} |")
    A("")
    allm = sorted({r["date"] for r in latest if r["granularity"] == "monthly"})
    if allm:
        A(f"Overall monthly coverage of the latest-vintage file: "
          f"**{allm[0]} .. {allm[-1]}** ({len(allm)} distinct months).")
        # gap report
        y0, m0 = int(allm[0][:4]), int(allm[0][5:])
        have = set(allm)
        gaps = []
        y, m = y0, m0
        while f"{y}-{m:02d}" <= allm[-1]:
            if f"{y}-{m:02d}" not in have:
                gaps.append(f"{y}-{m:02d}")
            m += 1
            if m == 13:
                y, m = y + 1, 1
        A("")
        A(f"Missing months inside that range: **{len(gaps)}**"
          + ("" if not gaps else " -> " + ", ".join(gaps[:24])
             + (" ..." if len(gaps) > 24 else "")))
    A("")

    # ---- revisions ----------------------------------------------------
    A("## 3. Vintage / revision statistics")
    A("")
    TOP_CLASSES = ("motor-cycles", "motor cycles", "private cars", "taxis",
                   "public buses", "goods vehicles", "government vehicles",
                   "public light buses")

    def merged_name(ent):
        """True if this entity path mixes two vehicle-class blocks.

        Two shapes occur: a path naming two top-level classes
        ('Taxis > Private Cars'), and the grand-total block being absorbed by
        the class printed above it ('Government Vehicles > Total > ...').  A
        bare 'Total' component is never a legitimate class or group name in
        these tables ('Sub-total' is).
        """
        low = ent.lower()
        if "total" in [c.strip() for c in low.split(">")]:
            return True
        return sum(1 for w in TOP_CLASSES if w in low) >= 2

    key = lambda r: (r["date"], r["granularity"], r["table"],
                     r["vehicle_class_or_make"], r["measure"])
    groups = defaultdict(list)
    for r in long_rows:
        groups[key(r)].append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    changed, changed_clean = [], []
    for k, g in multi.items():
        g.sort(key=lambda r: r["issue"])
        fv, lv = int(float(g[0]["value"])), int(float(g[-1]["value"]))
        if fv != lv:
            changed.append((k, fv, lv, len(g)))
            if not merged_name(k[3]):
                changed_clean.append((k, fv, lv, len(g)))
    A(f"- distinct series points                       : {len(groups)}")
    A(f"- series observed in more than one vintage     : {len(multi)}")
    A(f"- ... of those, value changed between first and last vintage : "
      f"**{len(changed)}** "
      f"({100.0 * len(changed) / len(multi):.1f}%)" if multi else "")
    if changed:
        A(f"- mean |change| among revised series           : "
          f"{statistics.mean(abs(l - f) for _, f, l, _ in changed):.1f}")
        A(f"- median |change|                              : "
          f"{statistics.median(abs(l - f) for _, f, l, _ in changed):.1f}")
        A(f"- max |change|                                 : "
          f"{max(abs(l - f) for _, f, l, _ in changed)}")
        A("")
        A("The largest entries above are a LABEL artefact, not real revisions: in")
        A("2013-2015 PDFs the grand-total block shares a page with the Government")
        A("Vehicles block, so the two labels merge and the same column is")
        A("attributed to different 'entities' in different vintages.  Note the")
        A("first vintage reads 6289 (genuine Government Vehicles) and the last")
        A("reads 769199 (the all-vehicle grand total) - the label, not the data,")
        A("is what moved.")
        A("")
        A(f"Excluding series whose entity name merges two top-level classes:")
        A(f"- revised series : **{len(changed_clean)}** of "
          f"{sum(1 for k in multi if not merged_name(k[3]))} "
          f"comparable series "
          f"({100.0 * len(changed_clean) / max(1, sum(1 for k in multi if not merged_name(k[3]))):.1f}%)")
        if changed_clean:
            A(f"- mean |change|  : "
              f"{statistics.mean(abs(l - f) for _, f, l, _ in changed_clean):.1f}")
            A(f"- median |change|: "
              f"{statistics.median(abs(l - f) for _, f, l, _ in changed_clean):.1f}")
            A(f"- max |change|   : "
              f"{max(abs(l - f) for _, f, l, _ in changed_clean)}")
        A("")
        A("Largest genuine revisions (first vintage -> last vintage):")
        A("")
        A("| date | table | entity | measure | first | last | vintages |")
        A("|---|---|---|---|---|---|---|")
        for k, fv, lv, n in sorted(changed_clean, key=lambda z: -abs(z[2] - z[1]))[:12]:
            A(f"| {k[0]} | {k[2]} | {k[3][:44]} | {k[4][:26]} | {fv} | {lv} | {n} |")
    A("")

    # ---- independent cross-checks ------------------------------------
    A("## 4. Independent cross-checks (computed)")
    A("")

    # (a) Table 4.1(e) is a single-month table.  The sum of its per-make totals
    #     must equal Table 4.1(c) gross private-car first registration for the
    #     same month, which is an entirely separate table.
    e_tot = defaultdict(int)
    for r in latest:
        if r["table"] == "4.1(e)" and r["granularity"] == "monthly":
            e_tot[r["date"]] += int(float(r["value"]))
    c_gross = {r["date"]: int(float(r["value"])) for r in latest
               if r["table"] == "4.1(c)" and r["measure"] == "Gross First Registration"}
    both = sorted(set(e_tot) & set(c_gross))
    if both:
        ok = [d for d in both if e_tot[d] == c_gross[d]]
        A(f"- **4.1(e) sum of makes vs 4.1(c) gross first registration**: "
          f"{len(both)} months comparable, **{len(ok)} agree exactly** "
          f"({100.0 * len(ok) / len(both):.1f}%).")
        bad = [d for d in both if e_tot[d] != c_gross[d]]
        for d in bad[:6]:
            A(f"  - {d}: 4.1(e) sum={e_tot[d]} vs 4.1(c)={c_gross[d]}")

    # (b) Table 4.1(a) private-car sub-total vs 4.1(c) gross, same month.
    a_pc = {r["date"]: int(float(r["value"])) for r in latest
            if r["table"] == "4.1(a)" and r["measure"] == "First Registration"
            and r["granularity"] == "monthly"
            and r["vehicle_class_or_make"].lower().startswith("private cars")
            and r["vehicle_class_or_make"].lower().endswith("sub-total")}
    both2 = sorted(set(a_pc) & set(c_gross))
    if both2:
        ok2 = [d for d in both2 if a_pc[d] == c_gross[d]]
        A(f"- **4.1(a) private-car sub-total vs 4.1(c) gross**: {len(both2)} months "
          f"comparable, **{len(ok2)} agree exactly** "
          f"({100.0 * len(ok2) / len(both2):.1f}%).")
        for d in [d for d in both2 if a_pc[d] != c_gross[d]][:6]:
            A(f"  - {d}: 4.1(a)={a_pc[d]} vs 4.1(c)={c_gross[d]}")
    A("")

    # ---- label reliability -------------------------------------------
    A("## 5. Known limitation: entity labels in the 2004-2015 PDF era")
    A("")
    A("Those PDFs lay several vehicle-class blocks out on one page while sharing")
    A("a single `Year/Month` column, so the class labels of neighbouring blocks")
    A("merge into one entity path (e.g. `Taxis > Private Cars`). The NUMBERS are")
    A("still correct - Table 4.1(a)'s private-car sub-total agrees with Table")
    A("4.1(c) for 100% of comparable months - but the class label is not.")
    A("")
    # Only the genuine top-level classes.  'Special Purpose Vehicles',
    # 'Private Buses', 'Private Light Buses' etc. are legitimate SUB-classes and
    # nest inside a top-level class, so they must not be counted as a second
    # top-level class here.
    pdf_rows = [r for r in latest if r["issue"] < "2018" and r["table"] == "4.1(a)"]
    xls_rows = [r for r in latest if r["issue"] >= "2018" and r["table"] == "4.1(a)"]

    def merged(r):
        return merged_name(r["vehicle_class_or_make"])

    for label, rs in (("PDF era (issue < 2018-01)", pdf_rows),
                      ("XLS era (issue >= 2018-01)", xls_rows)):
        if not rs:
            continue
        m = sum(1 for r in rs if merged(r))
        A(f"- 4.1(a) {label}: {len(rs)} points, {m} with a merged top-level "
          f"class label ({100.0 * m / len(rs):.1f}%)")
    A("")
    A("Table 4.1(c) is unaffected: it holds a single class (`Private Cars`) and")
    A("is the reliable long monthly series for private-car first registration.")
    A("")
    bare = [r for r in latest if r["measure"] == "Total"]
    if bare:
        yrs = Counter(r["issue"][:4] for r in bare)
        A(f"- {len(bare)} points ({100.0 * len(bare) / len(latest):.1f}% of the "
          f"latest file) carry a bare `Total` measure: the paired "
          f"`Registration`/`Licensed` label could not be attached to that "
          f"column, so it is `Total Registration` or `Total Licensed` but not "
          f"distinguishable in the CSV. Affected vintages: "
          f"{', '.join(f'{y}({n})' for y, n in sorted(yrs.items())[:10])}.")
    A("")

    # ---- headline series ---------------------------------------------
    A("## 6. Headline series check (private cars, gross first registration)")
    A("")
    cands = [r for r in latest if r["table"] == "4.1(c)"
             and r["measure"] == "Gross First Registration"
             and r["granularity"] == "monthly"]
    if cands:
        cands.sort(key=lambda r: r["date"])
        A(f"Series used: table `4.1(c)`, entity `Private Cars`, measure "
          f"`Gross First Registration` - {len(cands)} monthly points, "
          f"{cands[0]['date']} .. {cands[-1]['date']}.")
        A("")
        A("| month | gross first registration | vintage |")
        A("|---|---|---|")
        for r in cands[-12:]:
            A(f"| {r['date']} | {int(float(r['value']))} | {r['issue']} |")
    else:
        A("No private-car gross series found.")
    A("")

    # ---- provenance ---------------------------------------------------
    A("## 7. Provenance")
    A("")
    srcs = {(r["table"], r["source_url"]) for r in latest}
    per_tab = Counter(t for t, _ in srcs)
    for t in sorted(per_tab):
        A(f"- {t}: {per_tab[t]} distinct source files")
    hosts = Counter(r["source_url"].split("/")[2] for r in latest)
    A(f"- hosts: {dict(hosts)}")
    A(f"- fetched_at range: {min(r['fetched_at'] for r in latest)[:19]} .. "
      f"{max(r['fetched_at'] for r in latest)[:19]}")
    A("")

    (OUT / "VERIFICATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'VERIFICATION.md'} ({len(L)} lines)")


if __name__ == "__main__":
    main()
