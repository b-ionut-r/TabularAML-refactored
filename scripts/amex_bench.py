"""American Express Default Prediction (Kaggle 2022, $100k): monthly statements per customer, one default label each.

Data: public Kaggle copies of the contest files: the statements (``raddar/amex-data-integer-dtypes-parquet-format``,
train.parquet: 5.5M statements of 458,913 customers, 188 columns) and the labels (``train_labels.csv``, taken from
``jeonbyungsu/amex-data``, where they are joined onto the same statements).

Holdout built like the test file: ``--customers`` random customers (``--seed``), 20% of them held out whole (the test
file lists new customers; their statements are given, their labels are not). The test customers' statements also
come from a later period, which this holdout cannot copy. Labels come from train_labels only and reach only the
training customers. Metric: the contest's (mean of the normalised Gini and the default rate captured in the top 4%),
and AUC.

Arms (same bagged LightGBM: early stopping on 15% of the training customers, then 3 seeds on all of them):
  ``raw``   the customer's last statement;
  ``ff``    blind: ``scripts/contest_features.py`` at its defaults, main table = customer ids (+ labels), child table =
            every statement (``statements=...:customer_ID:S_2``), as the contest's files are laid out;
  ``hand``  the winners' aggregates per customer: last, mean, std, min, max, last - mean, last - previous statement,
            statement count (reference, not shipped);
  ``ff_hand`` the same contest_features run with the winners' aggregates added to the main table.
``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, gc
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--customers', type=int, default=100_000); ap.add_argument('--tag', default='')
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='', help='extra contest_features arguments')
ap.add_argument('--data', default='data/amex/'); ap.add_argument('--log', default='amex.jsonl'); ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--keep', default='')
a = ap.parse_args()
D = Path(a.data)
lab = pd.read_csv(D / 'train_labels.csv')
rng = np.random.default_rng(a.seed)
cust = rng.choice(lab.customer_ID.to_numpy(), a.customers, replace=False)
ho = set(rng.choice(cust, a.customers // 5, replace=False))
S = pd.read_parquet(D / 'train.parquet', filters=[('customer_ID', 'in', list(cust))])
S['S_2'] = pd.to_datetime(S['S_2'])
S = S.sort_values(['customer_ID', 'S_2']).reset_index(drop=True)
lab = lab[lab.customer_ID.isin(cust)].set_index('customer_ID')
ids_tr = np.array(sorted(set(cust) - ho)); ids_te = np.array(sorted(ho))
ytr, yte = lab.target.reindex(ids_tr).to_numpy(), lab.target.reindex(ids_te).to_numpy()
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
print('statements', len(S), 'train customers', len(ids_tr), 'held out', len(ids_te), flush=True)
feats = [c for c in S.columns if c not in ('customer_ID', 'S_2')]


def hand(S):
    g = S.groupby('customer_ID', sort=True)[feats]
    last = g.last(); mean = g.mean()
    end = S.groupby('customer_ID', sort=True).tail(1).index
    prev = S[feats].groupby(S.customer_ID).shift(1).loc[end].set_index(S.customer_ID.loc[end].to_numpy())
    out = [last.add_suffix('_last'), mean.add_suffix('_mean'), g.std().add_suffix('_std'), g.min().add_suffix('_min'),
           g.max().add_suffix('_max'), (last - mean).add_suffix('_last_mean'),
           (last - prev.reindex(last.index)).add_suffix('_last_prev'), g.size().rename('n_statements').to_frame()]
    return pd.concat(out, axis=1).astype(np.float32)


t0 = time.time(); info = {}
if a.arm == 'raw':
    L = S.groupby('customer_ID', sort=True)[feats].last()
    Xtr, Xte = L.reindex(ids_tr).reset_index(drop=True), L.reindex(ids_te).reset_index(drop=True)
elif a.arm == 'hand':
    H = hand(S)
    Xtr, Xte = H.reindex(ids_tr).reset_index(drop=True), H.reindex(ids_te).reset_index(drop=True)
else:
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    tmp.mkdir(parents=True, exist_ok=True)
    mtr = pd.DataFrame({'customer_ID': ids_tr, 'target': ytr}); mte = pd.DataFrame({'customer_ID': ids_te})
    if a.arm == 'ff_hand':
        H = hand(S)
        mtr = pd.concat([mtr, H.reindex(ids_tr).reset_index(drop=True)], axis=1)
        mte = pd.concat([mte, H.reindex(ids_te).reset_index(drop=True)], axis=1)
    mtr.to_parquet(tmp / 'train.parquet'); mte.to_parquet(tmp / 'test.parquet')
    S.to_parquet(tmp / 'statements.parquet'); del S; gc.collect()
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
           '--test', str(tmp / 'test.parquet'), '--target', 'target', '--id', 'customer_ID', '--task', 'binary',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out'),
           '--table', f'statements={tmp / "statements.parquet"}:customer_ID:S_2'] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet')
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert (Xtr.customer_ID.to_numpy() == ids_tr).all() and (Xte.customer_ID.to_numpy() == ids_te).all()
    Xtr = Xtr.drop(columns=['customer_ID', 'target']); Xte = Xte.drop(columns=['customer_ID'])[Xtr.columns]
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype('category')
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.3,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1, max_bin=127)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3


def amex_metric(y, p):
    o = np.argsort(-p); y = y[o]; w = np.where(y == 0, 20, 1)
    top = np.cumsum(w) <= 0.04 * w.sum(); d = y[top].sum() / y.sum()
    def gini(y, w):
        cw = np.cumsum(w); rnd = cw / cw[-1]; tp = np.cumsum(y * w) / (y * w).sum()
        return np.sum((tp - rnd) * w)
    return 0.5 * (gini(y, w) / gini(y[np.argsort(-y, kind='stable')], w[np.argsort(-y, kind='stable')]) + d)


res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, amex=float(amex_metric(yte, p)),
           auc=float(roc_auc_score(yte, p)), best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t),
           total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')

import shutil
if 'tmp' in globals() and not getattr(a, 'keep', ''):
    shutil.rmtree(tmp, ignore_errors=True)   # bench outputs fill the disk otherwise
