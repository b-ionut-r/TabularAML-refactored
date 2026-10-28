"""Loan Default Prediction (Imperial College, Kaggle 2014): raw vs FeatureForge on held-out rows.

Data: OpenML 6331 (the 105k-row training set; 769 anonymous columns). Target: whether the loan
defaulted (loss > 0), the classification stage every top solution used; AUC on a stratified 80/20
holdout whose features are the unlabeled rows. ``hand`` adds the golden features the winners found
by brute force over column pairs (f528 - f527, f528 - f274, f527 - f274). ``--shuffle`` permutes the
training labels (leakage control: AUC must land near 0.5).
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=1800); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/ld/ld.pq'); ap.add_argument('--log', default='loandefault.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
df = pd.read_parquet(a.data)
y = (df.pop('loss') > 0).astype(int); df = df.drop(columns=['id'])
for c in df.columns:
    if isinstance(df[c].dtype, pd.CategoricalDtype):
        df[c] = pd.to_numeric(df[c].astype(str), errors='coerce')  # OpenML stores numbers as categories
df = df.astype(np.float64)
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=a.seed, stratify=y)
Xtr, Xte = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
ytr, yte = y.iloc[itr].reset_index(drop=True), y.iloc[ite].reset_index(drop=True)
if a.shuffle:
    ytr = pd.Series(np.random.default_rng(0).permutation(ytr.to_numpy()))
t0 = time.time(); info = {}
if a.arm == 'hand':
    for X in (Xtr, Xte):
        X['g1'] = X['f528'] - X['f527']; X['g2'] = X['f528'] - X['f274']; X['g3'] = X['f527'] - X['f274']
if a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='binary', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xte)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, feats=f.new_columns_[:60])
    Xtr, Xte = f.transform_train(Xtr), f.transform(Xte)
fe_t = time.time() - t0
P = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_child_samples=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=a.seed)
perm = np.random.default_rng(a.seed).permutation(len(Xtr)); tr, va = np.sort(perm[:int(.85 * len(perm))]), np.sort(perm[int(.85 * len(perm)):])
b = lgb.train(P, lgb.Dataset(Xtr.iloc[tr], ytr.iloc[tr]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr.iloc[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte)
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, auc=roc_auc_score(yte, p),
           fe_s=round(fe_t), total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
