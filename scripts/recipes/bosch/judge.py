"""Bosch holdout judge (AUC; MCC at the best threshold on the held-out rows, the same rule for every arm)
derived from the Malware judge: bagged LightGBM (kaggle_late params, lr 0.1, <=2000 rounds), early stopping on the
latest 10% of training rows (sorted by AvSigVersion), refit on all rows with 3 seeds; AUC on the later block."""
import os, sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
H = "/tmp/claude-0/data/bosch/hold"; S = "/tmp/claude-0/-home-user-TabularAML-refactored/202b3b1c-2dba-5963-88b9-b4a0727be35c/scratchpad"
tag = sys.argv[1]
import pyarrow.parquet as pq
src = (f"{H}/train.parquet", f"{H}/test.parquet") if tag == "raw" else (f"{S}/bout/{tag}/train_features.parquet", f"{S}/bout/{tag}/test_features.parquet")
# Column by column from parquet into one float32 matrix per frame (categoricals as codes over
# train + test levels): no full pandas copy of a 9M-row frame is ever held.
# Column by column straight from the parquet files: a 400k x 4,264 table read whole needs 8 GB on its own.
ptr, pte = pq.ParquetFile(src[0]), pq.ParquetFile(src[1])
y = ptr.read(columns=["Response"]).column(0).to_numpy().astype(int)
yh = pd.read_parquet(f"{H}/hold_y.parquet").Response.to_numpy()
cols_ = [c for c in ptr.schema_arrow.names if c not in ("Response", "Id")]
cats = []
M = np.empty((ptr.metadata.num_rows, len(cols_)), dtype=np.float32); G = np.empty((pte.metadata.num_rows, len(cols_)), dtype=np.float32)
for j0 in range(0, len(cols_), 100):
    A_, B_ = ptr.read(columns=cols_[j0:j0 + 100]), pte.read(columns=cols_[j0:j0 + 100])
    for j, c in enumerate(cols_[j0:j0 + 100], j0):
        a_, b_ = A_.column(c).to_pandas(), B_.column(c).to_pandas()
        if pd.api.types.is_numeric_dtype(a_) and not isinstance(a_.dtype, pd.CategoricalDtype):
            M[:, j] = a_.to_numpy(dtype=np.float32, na_value=np.nan); G[:, j] = b_.to_numpy(dtype=np.float32, na_value=np.nan)
        else:
            cats.append(j)
            a_, b_ = a_.astype("category"), b_.astype("category")
            u = a_.cat.categories.union(b_.cat.categories)
            for X_, s_ in ((M, a_), (G, b_)):
                k = s_.cat.set_categories(u).cat.codes.to_numpy()
                X_[:, j] = np.where(k < 0, np.nan, k)
    del A_, B_
P = dict(objective="binary", metric="auc", learning_rate=0.1, num_leaves=63, min_child_samples=50, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20, num_threads=4, verbose=-1, force_col_wise=True, bin_construct_sample_cnt=50000)  # binning samples are held as doubles: 200k x 4,264 is 7 GB
np.save(f"{S}/bout_G_{tag}.npy", G); del G   # the test matrix waits on disk while the training set is binned
t0 = time.time(); N = len(M); n = int(0.9 * N)
D = lgb.Dataset(M, y, categorical_feature=cats, params=P, free_raw_data=True).construct()
del M
b = lgb.train(dict(P, seed=0), D.subset(np.arange(n)), 2000, valid_sets=[D.subset(np.arange(n, N))],
              callbacks=[lgb.early_stopping(100, verbose=False)])
k = int(b.best_iteration * 1.1) + 1
G = np.load(f"{S}/bout_G_{tag}.npy"); os.remove(f"{S}/bout_G_{tag}.npy")
p = np.mean([lgb.train(dict(P, seed=s), D, k).predict(G) for s in (0, 1, 2)], axis=0)
from sklearn.metrics import matthews_corrcoef
o = np.argsort(-p); mcc = max(matthews_corrcoef(yh, p >= p[o[j]]) for j in range(50, 3000, 10))
r = dict(tag=tag, auc=round(float(roc_auc_score(yh, p)), 5), mcc=round(float(mcc), 4), val_auc=round(float(b.best_score["valid_0"]["auc"]) if "auc" in b.best_score["valid_0"] else -1, 5),
         n_feat=G.shape[1], rounds=k, judge_s=round(time.time() - t0))
print("SCORE", json.dumps(r), flush=True); open(f"{S}/bscores.jsonl", "a").write(json.dumps(r) + "\n")
