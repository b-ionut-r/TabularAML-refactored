"""IEEE-CIS fraud: time-ordered holdout (last 20% of rows by TransactionDT), raw vs FeatureForge."""
import sys, time, json, warnings, argparse
warnings.filterwarnings('ignore')
import os; sys.path.insert(0, os.environ.get('REPO', '/home/user/TabularAML-refactored'))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from tabularaml.contest import ContestSolver
ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='raw'); ap.add_argument('--budget', type=float, default=600)
ap.add_argument('--frac', type=float, default=1.0); ap.add_argument('--kw', default='{}')
ap.add_argument('--transductive', action='store_true'); ap.add_argument('--end', type=float, default=1.0); ap.add_argument('--tag', default=''); ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
df = pd.read_parquet(os.environ.get('IEEE_DATA', '/home/user/data/ieee/ieee.parquet'))
if a.frac < 1: df = df.iloc[int(len(df) * (a.end - a.frac)):int(len(df) * a.end)].reset_index(drop=True)   # window ending at --end
y = df.pop('isFraud'); df = df.drop(columns=['TransactionID'])
n = int(len(df) * 0.8)
Xtr, Xte, ytr, yte = df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True), y.iloc[:n].reset_index(drop=True), y.iloc[n:].reset_index(drop=True)
if a.shuffle: ytr = pd.Series(np.random.default_rng(0).permutation(ytr.values))   # training labels only; holdout labels untouched
t0 = time.time(); info = {}
if a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='binary', time_budget=a.budget, random_state=0, n_jobs=4, verbose=True, **json.loads(a.kw)).fit(
        Xtr, ytr, X_unlabeled=Xte if a.transductive else None)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, feats=f.new_columns_[:60])
    Xtr, Xte = f.transform_train(Xtr), f.transform(Xte)
    pass  # (feature files not saved by the regression bench)
fe_t = time.time() - t0
import lightgbm as lgb
for c in Xtr.columns:
    if str(Xtr[c].dtype) in ('object', 'string', 'str'):
        u = pd.Categorical(pd.concat([Xtr[c], Xte[c]]).astype(str)).categories
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=u); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=u)
P = dict(objective='binary', learning_rate=0.05, num_leaves=127, min_child_samples=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_cat_to_onehot=8, cat_smooth=50, num_threads=4, verbose=-1, seed=0)
m = int(len(Xtr) * 0.85)   # early stopping on the most recent 15% of training rows
b = lgb.train(P, lgb.Dataset(Xtr.iloc[:m], ytr.iloc[:m]), 3000, valid_sets=[lgb.Dataset(Xtr.iloc[m:], ytr.iloc[m:])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
full = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1)
p = full.predict(Xte)
res = dict(arm=a.arm + a.tag, frac=a.frac, end=a.end, auc=roc_auc_score(yte, p), logloss=log_loss(yte, p), fe_s=round(fe_t), total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res, default=str))
open('/tmp/claude-0/runs/ieee.jsonl', 'a').write(json.dumps(res, default=str) + '\n')
