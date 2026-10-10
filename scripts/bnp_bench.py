"""BNP Paribas Cardif Claims Management (Kaggle 2016, $30k): raw vs FeatureForge vs the public hand features.

Data: the contest's train.csv (114,321 claims, 131 anonymous columns: 112 numeric, 19 categorical, among them
v22 with 18,210 levels; target 1 = claim suitable for accelerated approval), from the public Kaggle copy
``hjimbean/kaggle-classification-autofe-benchmark``. The contest's test file was a random draw of the same
claims, so the holdout is a stratified random 20% (``--seed``) whose features are the unlabeled rows.
Metric: log loss, as the contest (AUC reported too).
Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds on all of them):
  ``raw``    the columns as given (strings as categories);
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults;
  ``hand``   the top public kernels' features: missing values per row, counts of v22, v56, v125, v113, v79 and of
             v22 crossed with v56, v125, v79 and v113 over training and test rows together, and their
             out-of-fold target means (5 folds; test rows get the full-training mean).
``--shuffle`` permutes the training labels (leakage control).

    python scripts/bnp_bench.py --arm ff --seed 0
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/bnp/train.csv'); ap.add_argument('--log', default='bnp.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
df = pd.read_csv(a.data)
y = df.pop('target').astype(int).to_numpy(); df = df.drop(columns=['ID'])
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
    new = {'n_missing': A.isna().sum(axis=1).to_numpy(dtype=np.float32)}
    keys = [['v22'], ['v56'], ['v125'], ['v113'], ['v79'], ['v22', 'v56'], ['v22', 'v125'], ['v22', 'v79'], ['v22', 'v113']]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(Xtr, ytr))
    prior, m, ntr = ytr.mean(), 20.0, len(Xtr)
    te_tr, te_te = {}, {}
    for kk in keys:
        k = pd.factorize(A[kk].astype(str).fillna('nan').agg('|'.join, axis=1))[0]
        name = '+'.join(kk)
        new['n_' + name] = np.bincount(k)[k].astype(np.float32)
        ktr, kte, L = k[:ntr], k[ntr:], k.max() + 1
        te = np.zeros(ntr)
        for f, v in folds:
            te[v] = ((np.bincount(ktr[f], ytr[f], L) + m * prior) / (np.bincount(ktr[f], minlength=L) + m))[ktr[v]]
        te_tr['te_' + name] = te
        te_te['te_' + name] = ((np.bincount(ktr, ytr, L) + m * prior) / (np.bincount(ktr, minlength=L) + m))[kte]
    N = pd.DataFrame(new)
    Xtr = pd.concat([Xtr, N.iloc[:ntr].reset_index(drop=True), pd.DataFrame(te_tr)], axis=1)
    Xte = pd.concat([Xte, N.iloc[ntr:].reset_index(drop=True), pd.DataFrame(te_te)], axis=1)
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='binary', learning_rate=0.02, num_leaves=31, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=20, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, logloss=log_loss(yte, p), auc=roc_auc_score(yte, p),
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
