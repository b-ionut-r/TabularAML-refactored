"""Public top-kernel GA features: every JSON column flattened into its fields (totals as numbers), frequency
counts of the categorical fields over train + test, and visit time parts."""
import sys, os, json, time, numpy as np, pandas as pd
H = "/tmp/claude-0/data/ga/hold"; out = sys.argv[1]; os.makedirs(out, exist_ok=True); t = time.time()
tr, te = pd.read_parquet(f"{H}/train.parquet"), pd.read_parquet(f"{H}/test.parquet")
n = len(tr); y = tr.pop("target")
d = pd.concat([tr, te], ignore_index=True)
for c in ("device", "geoNetwork", "totals", "trafficSource"):
    f = pd.json_normalize(d[c].map(json.loads)); f.columns = [f"{c}.{k}" for k in f.columns]
    for k in f.columns:
        f[k] = f[k].astype(str) if f[k].dtype == object else f[k]
    d = pd.concat([d.drop(columns=c), f], axis=1)
num_tot = [k for k in d.columns if k.startswith("totals.")]
for k in num_tot:
    d[k] = pd.to_numeric(d[k], errors="coerce").astype("float32")
const = [k for k in d.columns if d[k].nunique(dropna=False) <= 1]; d = d.drop(columns=const)
for k in [k for k in d.columns if d[k].dtype == object and k not in ("sessionId", "fullVisitorId", "visitId")]:
    d[f"fq_{k}"] = d[k].map(d[k].value_counts()).astype("float32")
ts = pd.to_datetime(d.visitStartTime, unit="s")
d["hour"], d["dow"] = ts.dt.hour, ts.dt.dayofweek
d = d.drop(columns=["fullVisitorId", "visitId"])
a = d.iloc[:n].reset_index(drop=True); a["target"] = y.to_numpy()
a.to_parquet(f"{out}/train_features.parquet", index=False); d.iloc[n:].reset_index(drop=True).to_parquet(f"{out}/test_features.parquet", index=False)
print("hand columns", d.shape[1], round(time.time() - t), "s")
