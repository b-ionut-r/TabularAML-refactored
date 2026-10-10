"""Public top-kernel hand features for Microsoft Malware: frequency encodings over train+test rows of
every categorical / identifier column, version parts, and version recency (how far a machine's
AvSig / engine / app version trails the newest one seen for its OS / engine)."""
import sys, os, time, numpy as np, pandas as pd
H = "/tmp/claude-0/data/malware/hold"; out = sys.argv[1]; os.makedirs(out, exist_ok=True)
t = time.time()
tr = pd.read_parquet(f"{H}/train.parquet"); te = pd.read_parquet(f"{H}/test.parquet")
n = len(tr); y = tr.pop("HasDetections")
d = pd.concat([tr, te], ignore_index=True)
ids = [c for c in d.columns if c != "MachineIdentifier" and (isinstance(d[c].dtype, pd.CategoricalDtype)
       or not pd.api.types.is_numeric_dtype(d[c]) or ("Identifier" in c) or ("Version" in c))]
new = {}
for c in ids:
    new[f"fe_{c}"] = d[c].map(d[c].value_counts(dropna=False)).astype("float32").to_numpy()
def vkey(s, parts):
    v = s.astype(str).str.split(".", expand=True)
    return [pd.to_numeric(v[i], errors="coerce").astype("float32") if i in v else np.nan for i in parts]
for c in ("EngineVersion", "AppVersion", "AvSigVersion", "Census_OSVersion"):
    for i, x in zip((1, 2, 3), vkey(d[c], (1, 2, 3))):
        new[f"{c}_p{i}"] = np.asarray(x, dtype="float32")
av = new["AvSigVersion_p1"] * 1e5 + new["AvSigVersion_p2"]
eng = new["EngineVersion_p2"] * 1e5 + new["EngineVersion_p3"]
app = new["AppVersion_p1"] * 1e6 + new["AppVersion_p2"] * 1e3 + new["AppVersion_p3"]
for nm, k, by in (("av_lag_os", av, "Census_OSVersion"), ("av_lag_engine", av, "EngineVersion"),
                  ("engine_lag_os", eng, "Census_OSVersion"), ("app_lag_os", app, "Census_OSVersion")):
    g = pd.Series(k).groupby(d[by].astype(str).to_numpy()).transform("max").to_numpy()
    new[nm] = (g - k).astype("float32")
new["av_rank"] = pd.Series(av).rank(method="dense").to_numpy(dtype="float32")
F = pd.DataFrame(new)
A = pd.concat([d, F], axis=1)
a = A.iloc[:n].reset_index(drop=True); a["HasDetections"] = y.to_numpy()
a.to_parquet(f"{out}/train_features.parquet", index=False)
A.iloc[n:].reset_index(drop=True).to_parquet(f"{out}/test_features.parquet", index=False)
print("hand features", F.shape[1], round(time.time() - t), "s")
