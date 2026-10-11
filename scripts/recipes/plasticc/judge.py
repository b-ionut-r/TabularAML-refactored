"""PLAsTiCC holdout judge: bagged LightGBM multiclass (kaggle_late params, lr 0.1, <=2000 rounds), early stopping on
10% of training objects, refit with 3 seeds; multiclass logloss on held-out test objects (classes seen in training),
plain and with the contest's class weights (2 for classes 15 and 64) averaged per class."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
H = sys.argv[2] if len(sys.argv) > 2 else "hold"; S = sys.argv[3] if len(sys.argv) > 3 else "."
tag = sys.argv[1]
src = (f"{H}/train.parquet", f"{H}/test.parquet") if tag == "raw" else (f"{S}/pout/{tag}/train_features.parquet", f"{S}/pout/{tag}/test_features.parquet")
tr, te = pd.read_parquet(src[0]), pd.read_parquet(src[1])
h = pd.read_parquet(f"{H}/hold_y.parquet")
te = te.set_index("object_id").loc[h.object_id].reset_index() if "object_id" in te else te
classes = np.sort(pd.read_parquet(f"{H}/train.parquet").target.unique()); cmap = {c: i for i, c in enumerate(classes)}
y = tr.pop("target").map(cmap).to_numpy(); yh = h.target.map(cmap).to_numpy()
drop = [c for c in ("object_id",) if c in tr]
tr, te = tr.drop(columns=drop), te.drop(columns=[c for c in drop if c in te])
cats = [j for j, c in enumerate(tr.columns) if not pd.api.types.is_numeric_dtype(tr[c])]
for j in cats:
    c = tr.columns[j]; u = pd.Index(tr[c].astype(str).unique()).union(pd.Index(te[c].astype(str).unique()))
    tr[c] = pd.Categorical(tr[c].astype(str), categories=u).codes; te[c] = pd.Categorical(te[c].astype(str), categories=u).codes
M, G = tr.to_numpy(np.float32), te.to_numpy(np.float32)
P = dict(objective="multiclass", num_class=len(classes), metric="multi_logloss", learning_rate=0.1, num_leaves=63, min_child_samples=20,
         feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
t0 = time.time(); va = np.random.default_rng(0).random(len(M)) < 0.1
b = lgb.train(dict(P, seed=0), lgb.Dataset(M[~va], y[~va], categorical_feature=cats), 2000,
              valid_sets=[lgb.Dataset(M[va], y[va], categorical_feature=cats)], callbacks=[lgb.early_stopping(100, verbose=False)])
k = int(b.best_iteration * 1.1) + 1
p = np.mean([lgb.train(dict(P, seed=s), lgb.Dataset(M, y, categorical_feature=cats), k).predict(G) for s in (0, 1, 2)], axis=0)
p = np.clip(p, 1e-15, 1); p /= p.sum(1, keepdims=True)
ll = -np.log(p[np.arange(len(yh)), yh])
w = np.array([2.0 if c in (15, 64) else 1.0 for c in classes])
per = pd.Series(ll).groupby(yh).mean()
wll = float((per * w[per.index]).sum() / w[per.index].sum())
r = dict(tag=tag, logloss=round(float(ll.mean()), 5), wlogloss=round(wll, 5), n_feat=G.shape[1], rounds=k, judge_s=round(time.time() - t0))
print("SCORE", json.dumps(r), flush=True); open(f"{S}/pscores.jsonl", "a").write(json.dumps(r) + "\n")
