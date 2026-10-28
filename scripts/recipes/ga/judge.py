"""GA session holdout judge: bagged LightGBM regression (kaggle_late params, lr 0.1, <=2000 rounds), early stopping
on the latest 10% of training sessions, refit with 3 seeds; RMSE of log1p revenue on the later 20% of sessions.
String columns enter as categorical codes over train + test levels (LightGBM groups rare levels)."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
H = "/tmp/claude-0/data/ga/hold"; S = "/tmp/claude-0/-home-user-TabularAML-refactored/202b3b1c-2dba-5963-88b9-b4a0727be35c/scratchpad"
tag = sys.argv[1]
src = (f"{H}/train.parquet", f"{H}/test.parquet") if tag == "raw" else (f"{S}/gout/{tag}/train_features.parquet", f"{S}/gout/{tag}/test_features.parquet")
tr, te = pd.read_parquet(src[0]), pd.read_parquet(src[1])
yh = pd.read_parquet(f"{H}/hold_y.parquet").target.to_numpy()
y = tr.pop("target").to_numpy(dtype=float)
drop = [c for c in ("sessionId",) if c in tr]
tr, te = tr.drop(columns=drop), te.drop(columns=[c for c in drop if c in te])
cats = []
for j, c in enumerate(tr.columns):
    if not pd.api.types.is_numeric_dtype(tr[c]) or isinstance(tr[c].dtype, pd.CategoricalDtype):
        a, b = tr[c].astype(object).where(tr[c].notna(), "nan").astype(str), te[c].astype(object).where(te[c].notna(), "nan").astype(str)
        u = pd.Index(a.unique()).union(pd.Index(b.unique()))
        tr[c] = pd.Categorical(a, categories=u).codes; te[c] = pd.Categorical(b, categories=u).codes
        cats.append(j)
M, G = tr.to_numpy(np.float32), te.to_numpy(np.float32); del tr, te
P = dict(objective="regression", metric="rmse", learning_rate=0.1, num_leaves=63, min_child_samples=50, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20, num_threads=4, verbose=-1)
t0 = time.time(); N = len(M); n = int(0.9 * N)
D = lgb.Dataset(M, y, categorical_feature=cats, params=P, free_raw_data=True).construct()
b = lgb.train(dict(P, seed=0), D.subset(np.arange(n)), 2000, valid_sets=[D.subset(np.arange(n, N))], callbacks=[lgb.early_stopping(100, verbose=False)])
k = int(b.best_iteration * 1.1) + 1
p = np.mean([lgb.train(dict(P, seed=s), D, k).predict(G) for s in (0, 1, 2)], axis=0)
p = np.clip(p, 0, None)
r = dict(tag=tag, rmse=round(float(np.sqrt(np.mean((p - yh) ** 2))), 5), n_feat=G.shape[1], rounds=k, judge_s=round(time.time() - t0))
print("SCORE", json.dumps(r), flush=True); open(f"{S}/gscores.jsonl", "a").write(json.dumps(r) + "\n")
