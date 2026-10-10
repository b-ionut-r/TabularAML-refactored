"""Santander Customer Satisfaction (Kaggle 2016, $60k): raw vs FeatureForge vs the public hand features.

Data: OpenML 46634 (the contest's train.csv: 76,020 customers, 369 anonymous var* columns: imp_ amounts, ind_ flags,
num_ counts, saldo_ balances, var15 (age), var38 (mortgage value, 20% at one imputed value), var3 (region, -999999
missing); TARGET 1 = unsatisfied, 4%). The contest's test file was a random draw of the same customers, so the
holdout is a stratified random 20% (``--seed``) whose features are the unlabeled rows. Metric: AUC, as the contest.
Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds on all of them):
  ``raw``    the columns as given (ID dropped);
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults;
  ``hand``   the top public kernels' features: constant and duplicate columns dropped; var3 -999999 as missing;
             zeros per row; var38 at its imputed value flagged and log var38 elsewhere; sums and nonzero counts of
             the saldo_, imp_, num_ and ind_ families; saldo_var30 over var38; var15 below 23.
``--shuffle`` permutes the training labels (leakage control).

    python scripts/scs_bench.py --arm ff --seed 0
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/scs/scs.pq'); ap.add_argument('--log', default='scs.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
df = pd.read_parquet(a.data)
y = df.pop('TARGET').astype(int).to_numpy(); df = df.drop(columns=['ID'])
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=a.seed, stratify=y)
Xtr, Xte = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
ytr, yte = y[itr], y[ite]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time()
if a.arm == 'ff':
    tmp = Path(tempfile.mkdtemp(dir='/tmp/claude-0'))
    Xtr.assign(target=ytr).to_parquet(tmp / 'train.parquet'); Xte.to_parquet(tmp / 'test.parquet')
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
                    '--test', str(tmp / 'test.parquet'), '--target', 'target', '--task', 'binary', '--budget', str(a.budget),
                    '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else []), check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['target'])
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    shutil.rmtree(tmp, ignore_errors=True)
elif a.arm == 'hand':
    A = pd.concat([Xtr, Xte], ignore_index=True)
    A = A.loc[:, A.nunique() > 1]; A = A.T.drop_duplicates().T.astype(float)
    A['var3'] = A['var3'].replace(-999999, np.nan)
    new = {'n_zero': (A == 0).sum(1)}
    mode = A['var38'].round(6) == 117310.979016
    new['var38_mode'] = mode.astype(float); new['log_var38'] = np.where(mode, np.nan, np.log(A['var38']))
    for fam in ('saldo_', 'imp_', 'num_', 'ind_'):
        cols = [c for c in A.columns if c.startswith(fam)]
        new[fam + 'sum'] = A[cols].sum(1); new[fam + 'nnz'] = (A[cols] != 0).sum(1)
    new['saldo30_over_var38'] = A['saldo_var30'] / A['var38']; new['young'] = (A['var15'] < 23).astype(float)
    A = pd.concat([A, pd.DataFrame(new)], axis=1)
    Xtr, Xte = A.iloc[:len(Xtr)].reset_index(drop=True), A.iloc[len(Xtr):].reset_index(drop=True)
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='binary', learning_rate=0.02, num_leaves=31, min_child_samples=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=20, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, auc=roc_auc_score(yte, p),
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
