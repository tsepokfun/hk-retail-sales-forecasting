import sys, json, pandas as pd, numpy as np
from pathlib import Path
HERE = Path.cwd()
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE/"eval")); sys.path.insert(0,str(HERE/"models"))
import harness as H, xlstm_direct as X

wide = H.load_panel(); feat = H.load_features()
scaling = pd.read_csv(HERE/"data"/"scaling.csv", dtype={"series":str})
naive = H.seasonal_naive_mae_train(wide)

configs = [
  ("v1  no-anchor, no-calendar", dict(use_anchor=False, use_calendar=False)),
  ("    anchor only          ", dict(use_anchor=True,  use_calendar=False)),
  ("    calendar only        ", dict(use_anchor=False, use_calendar=True)),
  ("v2  anchor + calendar    ", dict(use_anchor=True,  use_calendar=True)),
]
out=[]
for name, kw in configs:
    print("\n" + "="*74); print(name); print("="*74)
    df, m = X.run(feat, wide, scaling, epochs=100, verbose=True, tag="xLSTM", **kw)
    t = H.metrics_from_frame(df, naive)
    allrow = t[t.h=="ALL"].iloc[0]
    per = {int(r.h): r.MASE for r in t.itertuples() if r.h!="ALL"}
    out.append({"config":name.strip(),"params":m["n_params"],"best_val":round(m["best_val_loss"],4),
                "best_ep":m["best_epoch"],"MASE_ALL":allrow["MASE"],"WAPE_ALL":round(allrow["WAPE"]*100,2),
                **{f"MASE_h{h}":round(v,4) for h,v in per.items()},
                "anchor":m.get("anchor_weights")})
    df.to_csv(HERE/"results"/f"ablation_{name.strip().replace(' ','_').replace(',','')}.csv", index=False)
r = pd.DataFrame(out)
r.to_csv(HERE/"results"/"stage0_ablation.csv", index=False, encoding="utf-8")
print("\n" + "="*74); print("ABLATION SUMMARY"); print("="*74)
print(r.drop(columns=["anchor"]).to_string(index=False))
print("\nanchor weights per config:")
for row in r.itertuples():
    print(f"  {row.config:28s} {row.anchor}")