"""Public top-notebook hand features for Ventilator (per-breath lags, leads, diffs, cumsum, area, ...)."""
import sys, time, numpy as np, pandas as pd
V = "/tmp/claude-0/data/ventilator/hold"; out = sys.argv[1]
import os; os.makedirs(out, exist_ok=True)
def feats(d):
    d = d.sort_values(["breath_id", "time_step"]).copy()
    g = d.groupby("breath_id")
    d["area"] = (d.time_step * d.u_in); d["area"] = d.groupby("breath_id")["area"].cumsum()
    d["u_in_cumsum"] = g.u_in.cumsum()
    d["time_step_diff"] = g.time_step.diff().fillna(0)
    for k in (1, 2, 3, 4):
        d[f"u_in_lag{k}"] = g.u_in.shift(k).fillna(0); d[f"u_out_lag{k}"] = g.u_out.shift(k).fillna(0)
        d[f"u_in_diff{k}"] = d.u_in - d[f"u_in_lag{k}"]
    for k in (1, 2):
        d[f"u_in_back{k}"] = g.u_in.shift(-k).fillna(0); d[f"u_out_back{k}"] = g.u_out.shift(-k).fillna(0)
    d["u_in_first"] = g.u_in.transform("first"); d["u_in_last"] = g.u_in.transform("last")
    d["u_in_max"] = g.u_in.transform("max"); d["u_in_mean"] = g.u_in.transform("mean")
    d["u_in_diffmax"] = d.u_in_max - d.u_in; d["u_in_diffmean"] = d.u_in_mean - d.u_in
    d["breath_step"] = g.cumcount()
    d["ewm"] = g.u_in.transform(lambda s: s.ewm(halflife=9).mean())
    d["roll_mean10"] = g.u_in.transform(lambda s: s.rolling(10, min_periods=1).mean())
    d["R_C"] = d.R.astype(str) + "_" + d.C.astype(str); d["R_C"] = d.R_C.astype("category")
    d["RC"] = d.R * d.C
    return d.sort_index()
t = time.time()
tr = pd.read_parquet(f"{V}/train.parquet"); te = pd.read_parquet(f"{V}/test.parquet")
feats(tr).drop(columns=["breath_id"]).to_parquet(f"{out}/train_features.parquet", index=False)
feats(te).drop(columns=["breath_id"]).to_parquet(f"{out}/test_features.parquet", index=False)
print("hand features", round(time.time() - t), "s")
