"""Ventilator holdout judge: bagged LightGBM (kaggle_late params, L1 objective, lr 0.1), trained and
scored on inspiratory rows (u_out == 0) as the contest scored; early stopping on 10% of breaths."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
V = "/tmp/claude-0/data/ventilator/hold"; S = "/tmp/claude-0/-home-user-TabularAML-refactored/202b3b1c-2dba-5963-88b9-b4a0727be35c/scratchpad"
tag = sys.argv[1]; shuffled = len(sys.argv) > 2 and sys.argv[2] == "shuffled"
if tag == "raw":
    tr, te = pd.read_parquet(f"{V}/train.parquet"), pd.read_parquet(f"{V}/test.parquet")
else:
    tr, te = pd.read_parquet(f"{S}/vout/{tag}/train_features.parquet"), pd.read_parquet(f"{S}/vout/{tag}/test_features.parquet")
raw_tr = pd.read_parquet(f"{V}/train.parquet", columns=["breath_id", "u_out", "pressure"])
h = pd.read_parquet(f"{V}/hold_y.parquet")
y = tr.pop("pressure").to_numpy() if "pressure" in tr else raw_tr.pressure.to_numpy()
if shuffled:
    y = np.random.default_rng(0).permutation(y)
for X in (tr, te):
    X.drop(columns=[c for c in ("id",) if c in X], inplace=True)
te = te[tr.columns]
for c in tr.columns:
    if not (pd.api.types.is_numeric_dtype(tr[c]) or isinstance(tr[c].dtype, pd.CategoricalDtype)):
        u = pd.Categorical(pd.concat([tr[c], te[c]]).astype(str)).categories
        tr[c] = pd.Categorical(tr[c].astype(str), categories=u); te[c] = pd.Categorical(te[c].astype(str), categories=u)
mi, mh = raw_tr.u_out.to_numpy() == 0, h.u_out.to_numpy() == 0
Xtr, ytr, gtr, Xte = tr[mi], y[mi], raw_tr.breath_id.to_numpy()[mi], te[mh]
P = dict(objective="l1", learning_rate=0.1, num_leaves=63, min_child_samples=50, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20, num_threads=4, verbose=-1)
t0 = time.time()
ub = np.unique(gtr); va = np.isin(gtr, np.random.default_rng(1).choice(ub, len(ub) // 10, replace=False))
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr[~va], ytr[~va]), 2000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
n = int(b.best_iteration * 1.1) + 1
D = lgb.Dataset(Xtr, ytr)
p = np.mean([lgb.train(dict(P, seed=s), D, n).predict(Xte) for s in (0, 1, 2)], axis=0)
mae = float(np.abs(h.pressure.to_numpy()[mh] - p).mean())
r = dict(tag=tag + ("_shuffled" if shuffled else ""), mae=round(mae, 4), n_feat=tr.shape[1], rounds=n, judge_s=round(time.time() - t0))
print("SCORE", json.dumps(r), flush=True); open(f"{S}/vscores.jsonl", "a").write(json.dumps(r) + "\n")
