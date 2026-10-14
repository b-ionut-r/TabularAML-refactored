"""KDD Cup 2012 Track 2 (search-ad click-through, $10k): raw vs FeatureForge on held-out rows.

Data: OpenML 1216 (1,496,391 ad impressions: ad, advertiser, query, keyword, title, description, user and URL ids,
depth and position of the ad, impression count; target click). The holdout is a stratified random 20% (``--seed``)
whose features are the unlabeled rows. Metric: AUC, as the contest (log loss reported too).
Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds on all of them):
  ``raw``    the columns as given;
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults.
``--shuffle`` permutes the training labels (leakage control).

    python scripts/kdd12_bench.py --arm ff --seed 0
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/kdd12/kdd12.pq'); ap.add_argument('--log', default='kdd12.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--keep', default='', help='directory to keep the blind output')
a = ap.parse_args()
df = pd.read_parquet(a.data)
y = df.pop('click').astype(int).to_numpy()
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=a.seed, stratify=y)
Xtr, Xte = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
ytr, yte = y[itr], y[ite]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time()
if a.arm == 'ff':
    tmp = Path(tempfile.mkdtemp(dir='/tmp/claude-0'))
    Xtr.assign(click=ytr).to_parquet(tmp / 'train.parquet'); Xte.to_parquet(tmp / 'test.parquet')
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
                    '--test', str(tmp / 'test.parquet'), '--target', 'click', '--task', 'binary', '--budget', str(a.budget),
                    '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else []), check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['click'])
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    if a.keep:
        shutil.copytree(tmp / 'out', a.keep, dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, auc=roc_auc_score(yte, p), logloss=log_loss(yte, p),
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
