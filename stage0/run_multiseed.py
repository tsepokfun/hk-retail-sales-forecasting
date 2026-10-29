import sys, pandas as pd, numpy as np
from pathlib import Path
HERE = Path.cwd()
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE/"eval")); sys.path.insert(0,str(HERE/"models"))
import harness as H, xlstm_direct as X

wide = H.load_panel(); feat = H.load_features()
scaling = pd.read_csv(HERE/"data"/"scaling.csv", dtype={"series":str})
naive = H.seasonal_naive_mae_train(wide)

CONFIGS = [("v1_baseline", dict(use_anchor=False, use_calendar=False)),
           ("anchor_only", dict(use_anchor=True,  use_calendar=False))]
SEEDS = [42, 7, 2024]

res = []
for cname, kw in CONFIGS:
    for sd in SEEDS:
        X.SEED = sd
        print(f"\n{'='*70}\n{cname}  seed={sd}\n{'='*70}", flush=True)
        df, m = X.run(feat, wide, scaling, epochs=100, verbose=False, tag="xLSTM", **kw)
        t = H.metrics_from_frame(df, naive)
        row_all = t[t["h"].astype(str) == "ALL"].iloc[0]
        per = {}
        for r in t.itertuples(index=False):
            if str(r.h) != "ALL":
                per[int(r.h)] = float(r.MASE)
        rec = {"config": cname, "seed": sd, "params": int(m["n_params"]),
               "best_val": round(float(m["best_val_loss"]), 5),
               "MASE_ALL": float(row_all["MASE"]),
               "WAPE_ALL": round(float(row_all["WAPE"]) * 100, 3)}
        for h, v in per.items():
            rec[f"MASE_h{h}"] = round(v, 4)
        res.append(rec)
        print(f"  MASE_ALL {rec['MASE_ALL']:.5f}  WAPE_ALL {rec['WAPE_ALL']:.2f}%  "
              f"val {rec['best_val']:.5f}", flush=True)

r = pd.DataFrame(res)
r.to_csv(HERE/"results"/"stage0_multiseed.csv", index=False, encoding="utf-8")
print("\n" + "="*70)
print("MULTI-SEED RAW")
print("="*70)
print(r.to_string(index=False))

print("\n" + "="*70)
print("SUMMARY  (manual mean/std, avoids pandas agg) ")
print("="*70)
print(f"{'config':14s} {'MASE_ALL mean':>14s} {'std':>8s} {'WAPE mean':>10s} {'std':>7s} {'val mean':>9s}")
for c in ["v1_baseline", "anchor_only"]:
    s = r[r["config"] == c]
    if len(s) == 0:
        continue
    print(f"{c:14s} {s['MASE_ALL'].mean():14.5f} {s['MASE_ALL'].std(ddof=1):8.5f} "
          f"{s['WAPE_ALL'].mean():10.3f} {s['WAPE_ALL'].std(ddof=1):7.3f} "
          f"{s['best_val'].mean():9.5f}")
print("\nseed effect check: are the three seeds identical or different?")
for c in ["v1_baseline", "anchor_only"]:
    s = r[r["config"] == c]
    if len(s) == 0:
        continue
    n_unique = s["MASE_ALL"].nunique()
    print(f"  {c:14s}: {n_unique} distinct MASE values across {len(s)} seeds"
          f"  -> {'SEED HAS NO EFFECT (bug)' if n_unique == 1 else 'seed matters'}")