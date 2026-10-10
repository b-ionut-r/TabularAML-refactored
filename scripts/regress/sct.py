"""Santander Customer Transaction: stratified 80/20 holdout, raw vs FeatureForge (unlabeled = holdout + real test rows)."""
import sys, time, json, warnings, argparse
warnings.filterwarnings('ignore'); import os; sys.path.insert(0, os.environ.get('REPO', '/home/user/TabularAML-refactored'))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=1200); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--full-test', action='store_true', help='use every competition test row, synthetic ones included')
ap.add_argument('--no-test', action='store_true', help='do not use the competition test rows as unlabeled data')
a = ap.parse_args()
D = os.environ.get('SCT_DATA', '/tmp/claude-0/kg/sct/prep/')
df = pd.read_parquet(D + 'train.parquet'); te = pd.read_parquet(D + 'test.parquet')
y = df.pop('target'); df = df.drop(columns=['ID_code']); te = te.drop(columns=['ID_code'])
# Real test rows: those with at least one value unique within the test set (the rest are synthetic).
uniq = np.zeros(len(te), bool)
for c in te.columns:
    vc = te[c].value_counts(); uniq |= te[c].map(vc).to_numpy() == 1
te_real = te[uniq].reset_index(drop=True)
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=a.seed, stratify=y)
Xtr, Xte = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
ytr, yte = y.iloc[itr].reset_index(drop=True), y.iloc[ite].reset_index(drop=True)
t0 = time.time(); info = {}
if a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    U = Xte if a.no_test else pd.concat([Xte, te if a.full_test else te_real], ignore_index=True)
    f = FeatureForge(task='binary', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True, **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=U)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, feats=f.new_columns_[:60])
    Xtr, Xte = f.transform_train(Xtr), f.transform(Xte)
fe_t = time.time() - t0
P = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_child_samples=80, feature_fraction=0.3,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=a.seed)
perm = np.random.default_rng(a.seed).permutation(len(Xtr)); tr, va = np.sort(perm[:int(.85*len(perm))]), np.sort(perm[int(.85*len(perm)):])
b = lgb.train(P, lgb.Dataset(Xtr.iloc[tr], ytr.iloc[tr]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr.iloc[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte)
res = dict(arm=a.arm + a.tag, seed=a.seed, real_test=int(uniq.sum()), auc=roc_auc_score(yte, p), logloss=log_loss(yte, p), fe_s=round(fe_t), total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res, default=str)); open('/tmp/claude-0/runs/sct.jsonl', 'a').write(json.dumps(res, default=str) + '\n')
