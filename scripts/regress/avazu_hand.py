"""Winners-style click-log features on the user-sampled Avazu file (label-free, train+test rows)."""
import sys, numpy as np, pandas as pd
d = pd.read_parquet(sys.argv[1])
h = pd.to_datetime(d.hour.astype(str), format="%y%m%d%H")
d["hod"] = h.dt.hour; d["dow"] = h.dt.dayofweek
t = (h - h.min()).dt.total_seconds() / 3600
day = h.dt.day
d["user"] = d.device_id.where(d.device_id != "a99f214a", d.device_ip + "_" + d.device_model)
for k in ["user", "device_ip", "device_id"]:
    g = d.groupby(k)
    d[f"cnt_{k}"] = g[k].transform("size")
    d[f"cnt_{k}_hour"] = d.groupby([k, "hour"])[k].transform("size")
    d[f"cnt_{k}_day"] = d.groupby([k, day])[k].transform("size")
    tt = t.groupby(d[k])
    d[f"prev_{k}"] = t - tt.shift(1); d[f"next_{k}"] = tt.shift(-1) - t
    d[f"nth_{k}"] = g.cumcount()
d["cnt_user_app"] = d.groupby(["user", "app_id"]).user.transform("size")
d["cnt_user_site"] = d.groupby(["user", "site_id"]).user.transform("size")
d = d.drop(columns=["user"])
d.to_parquet(sys.argv[2]); print(d.shape)
