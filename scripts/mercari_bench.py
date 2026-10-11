"""Mercari Price Suggestion (Kaggle 2018, $100k): raw vs FeatureForge on held-out rows.

Data: Hugging Face ``multabench/core-text-reg-mercari-marketplace`` (100k listings
of the contest's training set; target log1p(price)). Random 80/20 holdout per seed,
as the contest's test set was; the holdout's features are the unlabeled rows.
Metric: RMSLE of price (RMSE of log_price), lower is better. String columns go to
the judge as categoricals.
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=1800); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/mercari/data.parquet'); ap.add_argument('--log', default='mercari.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
df = pd.read_parquet(a.data)
y = df.pop('log_price').astype(float)
for c in df.columns:
    if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
        df[c] = df[c].astype(object)
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=a.seed)
Xtr, Xte = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
ytr, yte = y.iloc[itr].reset_index(drop=True), y.iloc[ite].reset_index(drop=True)
if a.shuffle:  # leakage control: permuted training labels
    ytr = pd.Series(np.random.default_rng(0).permutation(ytr.to_numpy()))
t0 = time.time(); info = {}
if a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='regression', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xte)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, feats=f.new_columns_[:60])
    Xtr, Xte = f.transform_train(Xtr), f.transform(Xte)
fe_t = time.time() - t0
obj = [c for c in Xtr.columns if Xtr[c].dtype == object or isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
for c in obj:
    cats = pd.Index(pd.unique(Xtr[c].astype(str)))
    Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20, num_threads=4, verbose=-1, seed=a.seed)
perm = np.random.default_rng(a.seed).permutation(len(Xtr)); tr, va = np.sort(perm[:int(.85 * len(perm))]), np.sort(perm[int(.85 * len(perm)):])
b = lgb.train(P, lgb.Dataset(Xtr.iloc[tr], ytr.iloc[tr]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte)
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, rmsle=float(np.sqrt(np.mean((p - yte.to_numpy()) ** 2))), best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
