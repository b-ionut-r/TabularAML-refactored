"""Porto Seguro Safe Driver Prediction (Kaggle 2017, $25k): raw vs FeatureForge on held-out rows.

Data: OpenML 42742 (the full 595k-row training set). Stratified 80/20 holdout per
seed; the holdout's features are the unlabeled rows. Metric: normalized Gini
(2 AUC - 1), as the contest. ``hand`` adds the usual hand-made features of top
public kernels (missing-value count, ps_car_13 x ps_reg_03, calc columns dropped).
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=1800); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/porto/porto.pq'); ap.add_argument('--log', default='porto.jsonl')
a = ap.parse_args()
df = pd.read_parquet(a.data)
y = df.pop('target').astype(int)
for c in df.columns:
    if isinstance(df[c].dtype, pd.CategoricalDtype):
        df[c] = pd.to_numeric(df[c].astype(str), errors='coerce')  # OpenML stores codes as categories
    df[c] = df[c].replace(-1, np.nan)  # the contest's missing-value marker
cats = [c for c in df.columns if c.endswith('_cat')]
for c in cats:
    df[c] = df[c].astype('Int64').astype(str).astype('category')
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=a.seed, stratify=y)
Xtr, Xte = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
ytr, yte = y.iloc[itr].reset_index(drop=True), y.iloc[ite].reset_index(drop=True)
t0 = time.time(); info = {}
if a.arm == 'hand':
    def hand(X):
        X = X.drop(columns=[c for c in X.columns if c.startswith('ps_calc')]).copy()
        X['n_missing'] = X.isna().sum(1); X['car13_reg03'] = X['ps_car_13'] * X['ps_reg_03']
        return X
    Xtr, Xte = hand(Xtr), hand(Xte)
if a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='binary', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xte)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, feats=f.new_columns_[:60])
    Xtr, Xte = f.transform_train(Xtr), f.transform(Xte)
fe_t = time.time() - t0
P = dict(objective='binary', learning_rate=0.02, num_leaves=24, min_child_samples=200, feature_fraction=0.4,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=4, verbose=-1, seed=a.seed)
perm = np.random.default_rng(a.seed).permutation(len(Xtr)); tr, va = np.sort(perm[:int(.85 * len(perm))]), np.sort(perm[int(.85 * len(perm)):])
b = lgb.train(P, lgb.Dataset(Xtr.iloc[tr], ytr.iloc[tr]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr.iloc[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte)
res = dict(arm=a.arm + a.tag, seed=a.seed, gini=2 * roc_auc_score(yte, p) - 1, fe_s=round(fe_t), total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
