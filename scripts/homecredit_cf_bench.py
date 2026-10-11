"""Home Credit Default Risk through ``scripts/contest_features.py`` (blind pipeline), for checking families that
switch on from keyed child tables (``--history auto``).

Data: the contest files (application_train.csv and five child tables keyed by SK_ID_CURR, each with its time
column: bureau DAYS_CREDIT, previous_application DAYS_DECISION, installments_payments DAYS_INSTALMENT,
POS_CASH_balance and credit_card_balance MONTHS_BALANCE). ``--loans`` random loans (``--seed``), 20% held out
(stratified); child rows of the sampled loans only. Same bagged LightGBM as the other benches; AUC.
``--extra`` passes arguments to contest_features.py; ``--shuffle`` permutes the training labels.
"""
import sys, time, json, warnings, argparse, subprocess, tempfile
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--loans', type=int, default=100_000)
ap.add_argument('--tag', default=''); ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='data/hc/'); ap.add_argument('--log', default='hc_cf.jsonl'); ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = Path(a.data)
app = pd.read_csv(D / 'application_train.csv')
app = app.sample(a.loans, random_state=a.seed).reset_index(drop=True)
itr, ite = train_test_split(np.arange(len(app)), test_size=0.2, random_state=a.seed, stratify=app.TARGET)
tr, te = app.iloc[np.sort(itr)].reset_index(drop=True), app.iloc[np.sort(ite)].drop(columns=['TARGET']).reset_index(drop=True)
yte = app.TARGET.iloc[np.sort(ite)].to_numpy()
if a.shuffle:
    tr['TARGET'] = np.random.default_rng(0).permutation(tr.TARGET.to_numpy())
ytr = tr.TARGET.to_numpy()
tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
keep = set(app.SK_ID_CURR)
tables = {'bureau': ('bureau.csv', 'DAYS_CREDIT'), 'prev': ('previous_application.csv', 'DAYS_DECISION'),
          'inst': ('installments_payments.csv', 'DAYS_INSTALMENT'), 'pos': ('POS_CASH_balance.csv', 'MONTHS_BALANCE'),
          'cc': ('credit_card_balance.csv', 'MONTHS_BALANCE')}
args = []
for name, (f, tcol) in tables.items():
    parts = [ch[ch.SK_ID_CURR.isin(keep)] for ch in pd.read_csv(D / f, chunksize=2_000_000)]
    pd.concat(parts, ignore_index=True).to_parquet(tmp / f'{name}.parquet')
    args += ['--table', f'{name}={tmp / (name + ".parquet")}:SK_ID_CURR:{tcol}']
tr.to_parquet(tmp / 'train.parquet'); te.to_parquet(tmp / 'test.parquet')
t0 = time.time()
cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
       '--test', str(tmp / 'test.parquet'), '--target', 'TARGET', '--id', 'SK_ID_CURR', '--task', 'binary',
       '--budget', str(a.budget), '--out-dir', str(tmp / 'out')] + args + (a.extra.split() if a.extra else [])
subprocess.run(cmd, check=True)
fe_t = time.time() - t0
Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['SK_ID_CURR', 'TARGET'])
Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet').drop(columns=['SK_ID_CURR'])[Xtr.columns]
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_child_samples=100, feature_fraction=0.3,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, max_bin=127)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm='ff' + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, auc=float(roc_auc_score(yte, p)), best_it=b.best_iteration,
           n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=len(Xte))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')

import shutil
if 'tmp' in globals() and not getattr(a, 'keep', ''):
    shutil.rmtree(tmp, ignore_errors=True)   # bench outputs fill the disk otherwise
