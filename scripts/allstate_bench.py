"""Allstate Claims Severity (Kaggle 2016): raw vs FeatureForge on held-out rows.

Data: OpenML 42571 (the 188k-row training set; 116 categoricals, 14 numerics). Random 80/20
holdout whose features are the unlabeled rows; the judge fits log(loss + 200) and is scored by
MAE on loss, as the contest. ``hand`` adds the categorical crosses of the top public kernels
(all pairs of 35 strong categoricals, each pair as one category, lexically coded).
``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse
from itertools import combinations
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=1800); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/as/as.pq'); ap.add_argument('--log', default='allstate.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
df = pd.read_parquet(a.data)
y = df.pop('loss').to_numpy(dtype=float); df = df.drop(columns=['id'])
cats = [c for c in df.columns if c.startswith('cat')]
for c in cats:
    df[c] = df[c].astype(str)
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=a.seed)
Xtr, Xte = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
ytr, yte = y[itr], y[ite]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time(); info = {}
if a.arm == 'hand':
    top = ('cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,'
           'cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25').split(',')
    def lex(s):  # the kernels' lexical code of a level string ('A' = 1, 'AB' = 28, ...)
        r = 0
        for ch in s:
            r = r * 26 + (ord(ch) - ord('A') + 1)
        return r
    for X in (Xtr, Xte):
        new = {f'{p}_{q}': (X[p] + X[q]).map(lex).astype(float) for p, q in combinations(top, 2)}
        for k, v in new.items():
            X[k] = v
if a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='regression', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, np.log(ytr + 200), X_unlabeled=Xte)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, feats=f.new_columns_[:60])
    Xtr, Xte = f.transform_train(Xtr), f.transform(Xte)
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = pd.Categorical(X[c], categories=sorted(pd.unique(pd.concat([Xtr[c], Xte[c]]))))
P = dict(objective='regression_l1', learning_rate=0.03, num_leaves=63, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=a.seed, max_cat_to_onehot=8)
zt = np.log(ytr + 200)
perm = np.random.default_rng(a.seed).permutation(len(Xtr)); tr, va = np.sort(perm[:int(.85 * len(perm))]), np.sort(perm[int(.85 * len(perm)):])
b = lgb.train(P, lgb.Dataset(Xtr.iloc[tr], zt[tr]), 20000, valid_sets=[lgb.Dataset(Xtr.iloc[va], zt[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.exp(lgb.train(P, lgb.Dataset(Xtr, zt), int(b.best_iteration * 1.1) + 1).predict(Xte)) - 200
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, mae=float(np.mean(np.abs(p - yte))),
           fe_s=round(fe_t), total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
