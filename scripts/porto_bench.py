"""Porto Seguro Safe Driver Prediction (Kaggle 2017, $25k): raw vs FeatureForge vs the public hand features.

Data: OpenML 42742 (the contest's 595,212 training rows: 57 anonymous ps_ind / ps_reg / ps_car / ps_calc columns,
-1 marks a missing value). The contest's test file was a random draw of the same drivers' policies, so the holdout
is a stratified random 20% (``--seed``) whose features are the unlabeled rows. Metric: normalized Gini (2 AUC - 1).
Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds on all of them):
  ``raw``    the columns as given (``_cat`` columns as categories);
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults;
  ``hand``   the top public kernels' features: ps_calc dropped; missing values and binary flags per row; ps_reg_03
             decoded to its integer (ps_reg_03^2 x 1600) and that integer's two parts (the "M" and "F" of the
             decoding kernel); ps_car_13 decoded (ps_car_13^2 x 48400); ps_car_13 x ps_reg_03; every ``_cat``
             column's count over training and test rows and its out-of-fold target mean (5 folds, smoothing 20).
``--shuffle`` permutes the training labels (leakage control).

    python scripts/porto_bench.py --arm ff --seed 0
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/porto/porto.pq'); ap.add_argument('--log', default='porto.jsonl')
ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--kaggle-types', action='store_true', help="type columns as Kaggle's CSV does: integers, -1 for missing")
ap.add_argument('--probe', default='', help='raw arm variants: rank11 / cat11 (ps_car_11_cat as frequency rank / category)')
a = ap.parse_args()
df = pd.read_parquet(a.data)
y = df.pop('target').astype(int).to_numpy()
for c in df.columns:
    if isinstance(df[c].dtype, pd.CategoricalDtype):
        df[c] = pd.to_numeric(df[c].astype(str), errors='coerce')  # OpenML stores codes as categories
    if not a.kaggle_types:
        df[c] = df[c].replace(-1, np.nan)  # the contest's missing-value marker
    else:
        df[c] = df[c].fillna(-1)  # OpenML has a few -1s as missing; Kaggle's CSV has -1 throughout
        if (df[c] % 1 == 0).all():
            df[c] = df[c].astype(np.int64)
cats = [c for c in df.columns if c.endswith('_cat')]
if not a.kaggle_types:
    for c in cats:
        df[c] = df[c].astype('Int64').astype(str).astype('category')
if a.probe == 'rank11':
    vc = df['ps_car_11_cat'].value_counts(); df['ps_car_11_cat'] = df['ps_car_11_cat'].map(pd.Series(np.arange(1, len(vc) + 1), index=vc.index))
elif a.probe == 'cat11':
    df['ps_car_11_cat'] = df['ps_car_11_cat'].astype(str).astype('category')
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
    A = A.drop(columns=[c for c in A.columns if c.startswith('ps_calc')])
    new = {'n_missing': A.isna().sum(1).to_numpy(np.float32),
           'n_bin': A[[c for c in A.columns if c.endswith('_bin')]].sum(1).to_numpy(np.float32)}
    r3 = np.round(A['ps_reg_03'] ** 2 * 1600)
    new['reg03_int'] = r3.to_numpy()
    # the decoding kernel: reg03_int = 27 * M + F with F in 1..27 (province / municipality guess)
    new['reg03_M'] = np.floor((r3 - 1) / 27).to_numpy(); new['reg03_F'] = (r3 - 27 * np.floor((r3 - 1) / 27)).to_numpy()
    new['car13_int'] = np.round(A['ps_car_13'] ** 2 * 48400).to_numpy()
    new['car13_x_reg03'] = (A['ps_car_13'] * A['ps_reg_03']).to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(Xtr, ytr))
    prior, m, ntr = ytr.mean(), 20.0, len(Xtr)
    te_tr, te_te = {}, {}
    for c in cats:
        k = pd.factorize(A[c].astype(object).fillna('nan').astype(str))[0]
        new['n_' + c] = np.bincount(k)[k].astype(np.float32)
        ktr, kte, L = k[:ntr], k[ntr:], k.max() + 1
        te = np.zeros(ntr)
        for f, v in folds:
            te[v] = ((np.bincount(ktr[f], ytr[f], L) + m * prior) / (np.bincount(ktr[f], minlength=L) + m))[ktr[v]]
        te_tr['te_' + c] = te
        te_te['te_' + c] = ((np.bincount(ktr, ytr, L) + m * prior) / (np.bincount(ktr, minlength=L) + m))[kte]
    N = pd.concat([A.reset_index(drop=True), pd.DataFrame(new)], axis=1)
    Xtr = pd.concat([N.iloc[:ntr].reset_index(drop=True), pd.DataFrame(te_tr)], axis=1)
    Xte = pd.concat([N.iloc[ntr:].reset_index(drop=True), pd.DataFrame(te_te)], axis=1)
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='binary', learning_rate=0.02, num_leaves=24, min_child_samples=200, feature_fraction=0.4,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, cat_smooth=20, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_kt' if a.kaggle_types else '') + (('_' + a.probe) if a.probe else '') + ('_shuffled' if a.shuffle else ''), seed=a.seed, gini=2 * roc_auc_score(yte, p) - 1,
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
