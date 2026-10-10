"""Amazon.com Employee Access Challenge (Kaggle 2013, $5k): raw vs FeatureForge vs the winners' crosses.

Data: OpenML 4135 (the contest's 32,769 training rows: the requested RESOURCE, the employee's manager and
seven role columns, all categorical ids; target 1 = access granted). The contest's test file was a random
draw of the same requests, so the holdout is a stratified random 20% (``--seed``) whose features are the
unlabeled rows. Metric: AUC.
Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds on all of them):
  ``raw``    the nine id columns as categories;
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults;
  ``hand``   the winners' features (Paul Duan / Benjamin Solecki and the other top teams): every pair and triple
             of the id columns (ROLE_CODE duplicates ROLE_TITLE and is left out) as a combined key, each with
             its count over training and test rows together and its out-of-fold target mean (5 folds; test
             rows get the full-training mean), on top of the raw columns.
``--shuffle`` permutes the training labels (leakage control: every arm must land near 0.5).

    python scripts/amazon_bench.py --arm ff --seed 0
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from itertools import combinations
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/amazon/amazon.pq'); ap.add_argument('--log', default='amazon.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
df = pd.read_parquet(a.data)
y = df.pop('target').astype(int).to_numpy()
df = df.apply(lambda s: pd.to_numeric(s.astype(str)).astype(np.int64))
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
    cols = [c for c in df.columns if c != 'ROLE_CODE']
    A = pd.concat([Xtr, Xte], ignore_index=True)
    new_tr, new_te = {}, {}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(Xtr, ytr))
    prior, m = ytr.mean(), 20.0
    for r in (2, 3):
        for cc in combinations(cols, r):
            k = pd.factorize(A[list(cc)].astype(str).agg('|'.join, axis=1))[0]
            name = '+'.join(cc)
            cnt = np.bincount(k)[k].astype(np.float32)
            new_tr['n_' + name], new_te['n_' + name] = cnt[:len(Xtr)], cnt[len(Xtr):]
            ktr, kte = k[:len(Xtr)], k[len(Xtr):]
            te = np.zeros(len(Xtr))
            for f, v in folds:
                s = np.bincount(ktr[f], ytr[f], minlength=k.max() + 1); n = np.bincount(ktr[f], minlength=k.max() + 1)
                te[v] = ((s + m * prior) / (n + m))[ktr[v]]
            s = np.bincount(ktr, ytr, minlength=k.max() + 1); n = np.bincount(ktr, minlength=k.max() + 1)
            new_tr['te_' + name], new_te['te_' + name] = te, ((s + m * prior) / (n + m))[kte]
    Xtr = pd.concat([Xtr, pd.DataFrame(new_tr)], axis=1); Xte = pd.concat([Xte, pd.DataFrame(new_te)], axis=1)
fe_t = time.time() - t0
for c in df.columns:  # the ids are categories for every arm
    if c in Xtr and not isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        cats = pd.Index(np.unique(Xtr[c].to_numpy()))
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats); Xte[c] = pd.Categorical(Xte[c], categories=cats)
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype) and c not in df.columns:
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='binary', learning_rate=0.02, num_leaves=31, min_child_samples=20, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=20, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, auc=roc_auc_score(yte, p), best_it=b.best_iteration,
           n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
