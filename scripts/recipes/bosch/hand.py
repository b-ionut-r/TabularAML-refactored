"""Winners' Bosch features on top of every raw column: each part's start / end time over all stations and per
line, its duration, how many numeric fields it has, and the "magic" order features (Id gap to the previous and
next part by start time, start-time gap to the neighbouring Ids), computed over train + test rows together.

    python3 -I hand.py <hold dir> <out dir>
"""
import os, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

H, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tr, te = pq.read_table(f"{H}/train.parquet"), pq.read_table(f"{H}/test.parquet")
n = tr.num_rows
dates = [c for c in te.column_names if "_D" in c]
nums = [c for c in te.column_names if "_F" in c and pa.types.is_floating(te.schema.field(c).type)]


def mat(cols):
    return np.vstack([np.column_stack([t[c].to_numpy(zero_copy_only=False) for c in cols]) for t in (tr, te)])


D = mat(dates)
F = {}
with np.errstate(all="ignore"):
    F["start"], F["end"] = np.nanmin(D, 1), np.nanmax(D, 1)
    for ln in ("L0", "L1", "L2", "L3"):
        k = [j for j, c in enumerate(dates) if c.startswith(ln + "_")]
        F[f"{ln}_start"], F[f"{ln}_end"] = np.nanmin(D[:, k], 1), np.nanmax(D[:, k], 1)
del D
F["duration"] = F["end"] - F["start"]
X = mat(nums)
F["n_numeric"] = np.isfinite(X).sum(1).astype(np.float32)
F["numeric_mean"] = np.nanmean(X, 1)
del X
ids = np.concatenate([t["Id"].to_numpy() for t in (tr, te)]).astype(np.int64)
f = pd.DataFrame(F)
f["Id"] = ids
s = f.sort_values(["start", "Id"])
f.loc[s.index, "id_gap_prev"] = s["Id"].diff().to_numpy()
f.loc[s.index, "id_gap_next"] = -s["Id"].diff(-1).to_numpy()
s = f.sort_values("Id")
f.loc[s.index, "start_gap_prev_id"] = s["start"].diff().to_numpy()
f.loc[s.index, "start_gap_next_id"] = -s["start"].diff(-1).to_numpy()
f = f.drop(columns="Id").astype(np.float32)
for t, part, name in ((tr, f.iloc[:n], "train"), (te, f.iloc[n:], "test")):
    for c in f.columns:
        t = t.append_column(f"hand_{c}", pa.array(part[c].to_numpy()))
    pq.write_table(t, f"{out}/{name}_features.parquet")
print("hand columns added", f.shape[1])
