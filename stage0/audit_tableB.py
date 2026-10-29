"""
AUDIT — TABLE B: the four-column per-horizon comparison.

  col 1  pure xLSTM_ensemble              (and the final model, MinT_wls)
  col 2  pure SeasonalNaive
  col 3  DEPLOYABLE per-horizon selection — decisions taken ONLY from
         validation block V1 (origins 2019-01..2021-11, every candidate trained
         on target_period <= 2017-12).  Rule:
             pick argmin V1 MASE at each horizon, but switch away from the
             incumbent xLSTM only when the series-level paired Wilcoxon
             (N=23) rejects equal per-series MAE at 5%.
         No 2024 label is read by the rule.
  col 4  ORACLE upper bound — per horizon, the best model ON THE 2024 TEST SET.
         Uses 2024 labels, NOT deployable.

Also reports MASE and WAPE side by side, because MASE weights the 23 series
equally while WAPE weights them by HK$; the two disagree at h=3 and h=12.

Writes: audit_tableB.csv, audit_test_dm.csv, audit_selector_rule.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import harness as H                       # noqa: E402
import audit_selection as AS                        # noqa: E402
from run_stage4_reconcile import (                  # noqa: E402
    build_A, naive_w, reconcile,
)

RES = ROOT / "results"
CANDIDATES = ["xLSTM_ensemble", "SeasonalNaive", "kNN_residual",
              "LightGBMTuned_residual"]
INCUMBENT = "xLSTM_ensemble"


def mint(frame, series, A, W, name):
    """MinT_wls on every (origin, h) vector of `frame`"""
    d = frame.copy()
    out = []
    for (o, h), g in d.groupby(["origin", "h"], sort=False):
        vec = g.set_index("series").loc[series, "y_pred"].to_numpy(float)
        rec = reconcile(vec.reshape(1, -1), A, W).ravel()
        g = g.copy()
        g["y_pred"] = [rec[list(series).index(s)] for s in g["series"]]
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res["model"] = name
    return res


def wcol(frames, name, naive_mae, series, A, W):
    """metrics row set for one column of the table"""
    f = frames[name]
    m = H.metrics_from_frame(f, naive_mae).copy()
    m["col"] = name
    return m


def main() -> int:
    wide = pd.read_csv(ROOT / "data" / "panel_wide.csv", index_col=0, dtype={0: str})
    wide.index = wide.index.astype(str)
    naive_mae = H.seasonal_naive_mae_train(wide)
    frames = AS.load_test_grid()
    ref = frames["SeasonalNaive"]
    series = sorted(ref["series"].unique(), key=lambda s: (len(s), s))
    A = build_A(series)
    W = np.diag(naive_w(wide, series, H.TRAIN_END))

    # coherent (MinT_wls) version of every candidate, built on this grid
    minted = {name: mint(frames[name], series, A, W, f"{name}__MinT_wls")
              for name in CANDIDATES}
    for k, v in minted.items():
        frames[f"{k}__MinT_wls"] = v

    # ------------------------------------------------------------------ rule
    v1 = pd.read_csv(RES / "audit_val_v1.csv")
    v1["h"] = v1["h"].astype(str)
    tests = pd.read_csv(RES / "audit_validation_tests.csv")
    tests["h"] = tests["h"].astype(str)
    v1m = v1[v1["model"].isin(CANDIDATES)].pivot_table(
        index="model", columns="h", values="MASE", aggfunc="first")

    picks, rule_log = {}, []
    for h in ("1", "3", "6", "12"):
        order = v1m[h].sort_values()
        winner = str(order.index[0])
        decision, p = "keep incumbent", np.nan
        if winner != INCUMBENT:
            row = tests[(tests["block"] == "V1") & (tests["h"] == h)
                        & (tests["A"] == winner) & (tests["B"] == INCUMBENT)]
            p = float(row["p_wilcoxon"].iloc[0]) if len(row) else np.nan
            decision = "switch" if (p == p and p < 0.05) else "keep incumbent (not significant)"
        picks[h] = winner if decision == "switch" else INCUMBENT
        rule_log.append({"h": h, "v1_argmin": winner,
                         "v1_MASE": round(float(order.iloc[0]), 5),
                         "v1_MASE_incumbent": round(float(v1m.loc[INCUMBENT, h]), 5),
                         "v1_MASE_runner_up": round(float(order.iloc[1]), 5),
                         "wilcoxon_p_vs_incumbent": (round(p, 6) if p == p else None),
                         "decision": decision, "deployed_model": picks[h]})
    rule = pd.DataFrame(rule_log)
    print("=" * 94)
    print("DEPLOYABLE RULE — evidence = validation block V1 only")
    print("=" * 94)
    print(rule.to_string(index=False))
    json.dump(rule_log, open(RES / "audit_selector_rule.json", "w"), indent=1)

    # -------------------------------------------------------------- hybrids
    cols = {}
    cols["xLSTM_ensemble"] = frames["xLSTM_ensemble"]
    cols["SeasonalNaive"] = frames["SeasonalNaive"]
    cols["Deployable_V1_per_h"] = pd.concat(
        [frames[m][frames[m]["h"] == int(h)] for h, m in picks.items()],
        ignore_index=True).assign(model="Deployable_V1_per_h")
    # coherent variant: each chosen model is MinT_wls-reconciled before mixing,
    # so the mixed forecast still satisfies the 2024 aggregation identities
    cols["Deployable_V1_per_h_MinT"] = pd.concat(
        [minted[m][minted[m]["h"] == int(h)] for h, m in picks.items()],
        ignore_index=True).assign(model="Deployable_V1_per_h_MinT")
    cols["xLSTM_ensemble_MinT_wls"] = minted[INCUMBENT]

    # oracle (MASE) and oracle (WAPE), both test-set chosen
    met = pd.read_csv(RES / "audit_model_by_h.csv")
    met["h"] = met["h"].astype(str)
    met = met[met["model"].isin(AS.PARTICIPANTS)]
    orc_pick, orc_wape_pick = {}, {}
    for h in ("1", "3", "6", "12"):
        c = met[met["h"] == h]
        orc_pick[h] = c.sort_values("MASE").iloc[0]["model"]
        orc_wape_pick[h] = c.sort_values("WAPE").iloc[0]["model"]
    src = dict(frames)
    src["xLSTM_ensemble_MinT_wls"] = minted[INCUMBENT]
    cols["Oracle_per_h_MASE"] = pd.concat(
        [src[m][src[m]["h"] == int(h)] for h, m in orc_pick.items()],
        ignore_index=True).assign(model="Oracle_per_h_MASE")
    cols["Oracle_per_h_WAPE"] = pd.concat(
        [src[m][src[m]["h"] == int(h)] for h, m in orc_wape_pick.items()],
        ignore_index=True).assign(model="Oracle_per_h_WAPE")
    print("\n  ORACLE picks by MASE:", orc_pick)
    print("  ORACLE picks by WAPE:", orc_wape_pick)

    rows = []
    for name, f in cols.items():
        m = H.metrics_from_frame(f, naive_mae)
        m["variant"] = name
        rows.append(m)
    tab = pd.concat(rows, ignore_index=True)
    tab["h"] = tab["h"].astype(str)
    tab["deployable"] = tab["variant"].str.startswith("Deployable")
    tab.to_csv(RES / "audit_tableB.csv", index=False)

    show = tab[["variant", "h", "MASE", "WAPE", "bias_pct", "n_points"]]
    print("\n" + "=" * 94)
    print("TABLE B — per-horizon selection (2024 test set)")
    print("=" * 94)
    print(show.round(5).to_string(index=False))
    print("\n  MASE_ALL summary:")
    allr = tab[tab["h"] == "ALL"][["variant", "MASE", "WAPE", "bias_pct"]]
    print(allr.round(5).to_string(index=False))

    # --------------------------------------------------------- test-set DM
    print("\n" + "=" * 94)
    print("PAIRED SERIES-LEVEL TEST on the 2024 test set (N=23 series)")
    print("  A vs B, Wilcoxon on per-series MAE; negative median_rel => A better")
    print("=" * 94)
    dm = []
    pairs = [("xLSTM_ensemble", "SeasonalNaive"), ("xLSTM_ensemble", "kNN_residual"),
             ("xLSTM_ensemble", "LightGBMTuned_residual")]
    for a, b in pairs:
        for h in H.HORIZONS:
            r = AS.pair_test(frames[a][frames[a]["h"] == h],
                             frames[b][frames[b]["h"] == h], a, b)
            ma = H.metrics_from_frame(frames[a][frames[a]["h"] == h], naive_mae)
            mb = H.metrics_from_frame(frames[b][frames[b]["h"] == h], naive_mae)
            r["h"] = h
            r["MASE_A"] = float(ma["MASE"].iloc[0])
            r["MASE_B"] = float(mb["MASE"].iloc[0])
            r["WAPE_A"] = float(ma["WAPE"].iloc[0])
            r["WAPE_B"] = float(mb["WAPE"].iloc[0])
            dm.append(r)
    dm = pd.DataFrame(dm)
    dm.to_csv(RES / "audit_test_dm.csv", index=False)
    print(dm[["h", "A", "B", "series_A_better", "series_B_better",
              "median_rel_diff_pct", "p_wilcoxon", "MASE_A", "MASE_B",
              "WAPE_A", "WAPE_B"]].round(5).to_string(index=False))

    json.dump({"rule": rule_log, "oracle_MASE": orc_pick,
               "oracle_WAPE": orc_wape_pick},
              open(RES / "audit_tableB_picks.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
