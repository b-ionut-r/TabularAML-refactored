"""Malware holdout judge: bagged LightGBM (kaggle_late params, lr 0.1, <=2000 rounds), early stopping on the
latest 10% of training rows (sorted by AvSigVersion), refit on all rows with 3 seeds; AUC on the later block."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
H = "/tmp/claude-0/data/malware/hold"; S = "/tmp/claude-0/-home-user-TabularAML-refactored/202b3b1c-2dba-5963-88b9-b4a0727be35c/scratchpad"
tag = sys.argv[1]
import pyarrow.parquet as pq
src = (f"{H}/train.parquet", f"{H}/test.parquet") if tag == "raw" else (f"{S}/mout/{tag}/train_features.parquet", f"{S}/mout/{tag}/test_features.parquet")
# Column by column from parquet into one float32 matrix per frame (categoricals as codes over
# train + test levels): no full pandas copy of a 9M-row frame is ever held.
tr, te = pq.read_table(src[0]), pq.read_table(src[1])
y = tr.column("HasDetections").to_numpy().astype(int)
yh = pd.read_parquet(f"{H}/hold_y.parquet").HasDetections.to_numpy()
cols_ = [c for c in tr.column_names if c not in ("HasDetections", "MachineIdentifier")]
cats = []
M = np.empty((tr.num_rows, len(cols_)), dtype=np.float32); G = np.empty((te.num_rows, len(cols_)), dtype=np.float32)
for j, c in enumerate(cols_):
    a_, b_ = tr.column(c).to_pandas(), te.column(c).to_pandas()
    if pd.api.types.is_numeric_dtype(a_) and not isinstance(a_.dtype, pd.CategoricalDtype):
        M[:, j] = a_.to_numpy(dtype=np.float32, na_value=np.nan); G[:, j] = b_.to_numpy(dtype=np.float32, na_value=np.nan)
    else:
        cats.append(j)
        a_, b_ = a_.astype("category"), b_.astype("category")
        u = a_.cat.categories.union(b_.cat.categories)
        for X_, s_ in ((M, a_), (G, b_)):
            k = s_.cat.set_categories(u).cat.codes.to_numpy()
            X_[:, j] = np.where(k < 0, np.nan, k)
    tr, te = tr.drop_columns([c]), te.drop_columns([c])
del tr, te
P = dict(objective="binary", metric="auc", learning_rate=0.1, num_leaves=63, min_child_samples=50, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20, num_threads=4, verbose=-1)
t0 = time.time(); N = len(M); n = int(0.9 * N)
D = lgb.Dataset(M, y, categorical_feature=cats, params=P, free_raw_data=True).construct()
del M
b = lgb.train(dict(P, seed=0), D.subset(np.arange(n)), 2000, valid_sets=[D.subset(np.arange(n, N))],
              callbacks=[lgb.early_stopping(100, verbose=False)])
k = int(b.best_iteration * 1.1) + 1
p = np.mean([lgb.train(dict(P, seed=s), D, k).predict(G) for s in (0, 1, 2)], axis=0)
r = dict(tag=tag, auc=round(float(roc_auc_score(yh, p)), 5), val_auc=round(float(b.best_score["valid_0"]["auc"]) if "auc" in b.best_score["valid_0"] else -1, 5),
         n_feat=G.shape[1], rounds=k, judge_s=round(time.time() - t0))
print("SCORE", json.dumps(r), flush=True); open(f"{S}/mscores.jsonl", "a").write(json.dumps(r) + "\n")
