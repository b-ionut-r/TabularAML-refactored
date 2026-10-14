"""Liberty Mutual Group Property Inspection Prediction (Kaggle 2015, $25k): raw vs FeatureForge vs the public hand features.

Data: the contest's train.csv (50,999 properties, 32 anonymous columns T1_V1..T2_V15: 16 single-letter codes, 16
small integers; target Hazard, a count from 1 to 69) and test.csv (51,000), from the public Kaggle copy
``hjimbean/kaggle-classification-autofe-benchmark``. The test file was a random draw of the same properties, so the
holdout is a random 20% (``--seed``); the held-out rows and the contest's test rows are the unlabeled rows.
Metric: normalized Gini of the predicted hazard, as the contest.
Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds on all of them):
  ``raw``    the columns as given (letters as categories);
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults;
  ``hand``   the top public kernels' features: letters as their alphabet position (the winners found the codes
             ordered), T2_V10, T2_V7, T1_V13 and T1_V10 dropped, every column's count over training and test rows,
             and the out-of-fold mean hazard of each letter column (5 folds, smoothing 20).
``--shuffle`` permutes the training labels (leakage control).

    python scripts/liberty_bench.py --arm ff --seed 0
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/liberty/'); ap.add_argument('--log', default='liberty.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = Path(a.data)
tr, te_real = pd.read_csv(D / 'train.csv').drop(columns=['Id']), pd.read_csv(D / 'test.csv').drop(columns=['Id'])
y = tr.pop('Hazard').to_numpy(float)
itr, iho = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=a.seed)
Xtr, Xho = tr.iloc[itr].reset_index(drop=True), tr.iloc[iho].reset_index(drop=True)
ytr, yho = y[itr], y[iho]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
Xun = pd.concat([Xho, te_real], ignore_index=True)
letters = [c for c in tr.columns if tr[c].dtype == object]
def gini(y, p):
    o = np.argsort(-p, kind='stable'); c = np.cumsum(y[o]) / y.sum(); n = len(y)
    g = (c - np.arange(1, n + 1) / n).sum()
    o = np.argsort(-y, kind='stable'); c = np.cumsum(y[o]) / y.sum(); gmax = (c - np.arange(1, n + 1) / n).sum()
    return g / gmax
t0 = time.time()
if a.arm == 'ff':
    tmp = Path(tempfile.mkdtemp(dir='/tmp/claude-0'))
    Xtr.assign(Hazard=ytr).to_parquet(tmp / 'train.parquet'); Xun.to_parquet(tmp / 'test.parquet')
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
                    '--test', str(tmp / 'test.parquet'), '--target', 'Hazard', '--task', 'regression', '--budget', str(a.budget),
                    '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else []), check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['Hazard'])
    Xho = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns].iloc[:len(Xho)].reset_index(drop=True)
    shutil.rmtree(tmp, ignore_errors=True)
elif a.arm == 'hand':
    A = pd.concat([Xtr, Xun], ignore_index=True).drop(columns=['T2_V10', 'T2_V7', 'T1_V13', 'T1_V10'])
    new = {}
    for c in A.columns:
        k = pd.factorize(A[c])[0]; new['n_' + c] = np.bincount(k)[k].astype(np.float32)
    folds = list(KFold(5, shuffle=True, random_state=a.seed).split(Xtr))
    prior, m, ntr = ytr.mean(), 20.0, len(Xtr)
    te_tr, te_un = {}, {}
    for c in [c for c in letters if c in A.columns]:
        k = pd.factorize(A[c])[0]; ktr, kun, L = k[:ntr], k[ntr:], k.max() + 1
        te = np.zeros(ntr)
        for f, v in folds:
            te[v] = ((np.bincount(ktr[f], ytr[f], L) + m * prior) / (np.bincount(ktr[f], minlength=L) + m))[ktr[v]]
        te_tr['te_' + c] = te; te_un['te_' + c] = ((np.bincount(ktr, ytr, L) + m * prior) / (np.bincount(ktr, minlength=L) + m))[kun]
        A[c] = A[c].map(lambda s: ord(s) - 64)
    N = pd.concat([A, pd.DataFrame(new)], axis=1)
    Xtr = pd.concat([N.iloc[:ntr].reset_index(drop=True), pd.DataFrame(te_tr)], axis=1)
    Xho = pd.concat([N.iloc[ntr:].reset_index(drop=True), pd.DataFrame(te_un)], axis=1).iloc[:len(Xho)].reset_index(drop=True)
fe_t = time.time() - t0
for X in (Xtr, Xho):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xho[c] = pd.Categorical(Xho[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='poisson', learning_rate=0.02, num_leaves=31, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=20, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xho))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, gini=gini(yho, p),
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
