"""Avazu sample that keeps whole user histories: user = device_id, or device_ip + device_model
when device_id is the shared null id (a99f214a). Keeps users whose hash falls in frac."""
import sys, subprocess, numpy as np, pandas as pd
z, out, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
parts = []
for f in ["train.csv", "valid.csv", "test.csv"]:
    p = subprocess.Popen(["unzip", "-p", z, f], stdout=subprocess.PIPE)
    for ch in pd.read_csv(p.stdout, chunksize=2_000_000, dtype=str):
        u = ch.device_id.where(ch.device_id != "a99f214a", ch.device_ip + "_" + ch.device_model)
        h = pd.util.hash_pandas_object(u, index=False).to_numpy()
        parts.append(ch[(h % 10_000) < frac * 10_000])
    p.wait()
d = pd.concat(parts, ignore_index=True)
for c in ["click", "hour", "C1", "banner_pos", "device_type", "device_conn_type"] + [f"C{i}" for i in range(14, 22)]:
    d[c] = pd.to_numeric(d[c])
d = d.drop(columns=["id"]).sort_values("hour", kind="stable").reset_index(drop=True)
d.to_parquet(out)
print(d.shape, d.click.mean(), d.hour.min(), d.hour.max(), d.device_ip.value_counts().head(3).to_dict())
