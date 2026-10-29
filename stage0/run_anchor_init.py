import sys, pandas as pd, numpy as np
from pathlib import Path
HERE = Path.cwd()
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE/"eval")); sys.path.insert(0,str(HERE/"models"))
import harness as H, xlstm_direct as X

wide = H.load_panel(); feat = H.load_features()
scaling = pd.read_csv(HERE/"data"/"scaling.csv", dtype={"series":str})
naive = H.seasonal_naive_mae_train(wide)

CONFIGS = [
  ("init2.0  wd=on   (current)", dict(anchor_init=2.0, anchor_no_wd=False)),
  ("init0.5  wd=on            ", dict(anchor_init=0.0, anchor_no_wd=False)),
  ("init0.5  wd=OFF on anchor ", dict(anchor_init=0.0, anchor_no_wd=True)),
  ("init2.0  wd=OFF on anchor ", dict(anchor_init=2.0, anchor_no_wd=True)),
]
res=[]
for name, kw in CONFIGS:
    print(f"\n{'='*72}\n{name}\n{'='*72}", flush=True)
    df, m = X.run(feat, wide, scaling, epochs=100, verbose=True, tag="xLSTM",
                  use_anchor=True, use_calendar=False, **kw)
    t = H.metrics_from_frame(df, naive)
    a = t[t["h"].astype(str)=="ALL"].iloc[0]
    per = {int(r.h): round(float(r.MASE),4) for r in t.itertuples(index=False) if str(r.h)!="ALL"}
    aw = m["anchor_weights"]
    print(f"  anchor weights: {aw}")
    print(f"  -> MASE_ALL {a['MASE']:.5f}  WAPE_ALL {a['WAPE']*100:.2f}%", flush=True)
    res.append({"config":name.strip(),"MASE_ALL":float(a["MASE"]),
                "WAPE_ALL":round(float(a["WAPE"])*100,3),
                **{f"MASE_h{h}":v for h,v in per.items()},
                **{f"a_h{h}":round(aw[str(h)],4) for h in (1,3,6,12)}})
r = pd.DataFrame(res)
r.to_csv(HERE/"results"/"stage0_anchor_init.csv", index=False, encoding="utf-8")
print("\n"+"="*72); print("ANCHOR INIT / WEIGHT-DECAY EXPERIMENT"); print("="*72)
print(r.to_string(index=False))