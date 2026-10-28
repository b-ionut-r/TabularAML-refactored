"""WSDM - KKBox's Churn Prediction Challenge (Kaggle 2017, $5k): will a music-streaming subscriber whose membership
expires in a month renew within 30 days. Log loss, as the contest.

Data: a public copy of the contest files (``qmdo97/kkboxdataset``: train.csv 992,931 users expiring in February 2017,
train_v2.csv 970,960 expiring in March 2017, transactions.csv 21,547,746 rows, members_v3.csv 6,769,473; row counts
as the contest's; Kaggle's own download needs the rules accepted). The contest trained on one month's expiries and
scored the next month's, so the held-out rows are the March users (train_v2) and training rows the February users
(train). Each row sees the transactions dated before the first day of its expiry month (``as_of_date``, 20170201 /
20170301; the contest's files ran to the end of the month before the test month). The 30 GB listening logs are left
out (they do not fit this machine). ``--sample`` users of each month (by a hash of the user id, ``--seed``) keep it in
memory. Held-out labels reach only the scorer.

Arms (same bagged LightGBM: early stopping on 15% of training rows, then 3 seeds):
  ``raw``  the user's member record (city, age, gender, registration channel and date);
  ``ff``   blind: ``scripts/contest_features.py`` at its defaults, transactions given as a related table
           (``tx=...:msno:transaction_date``, ``--time as_of_date``) and members as a one-row-per-user table;
  ``hand`` raw + the top public kernels' transaction features: the last transaction's plan days, list price, amount
           paid, discount, auto-renew and cancel flags, payment method and membership expiry (days after the as-of
           date), days since it, number of transactions, cancels, auto-renew share, total paid, distinct plans;
  ``ff_hand`` both. ``--shuffle`` permutes the training labels (leakage control). Reports runtime and peak memory.
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil, resource, zlib
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from sklearn.metrics import log_loss
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--sample', type=float, default=0.3); ap.add_argument('--tag', default='')
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='/home/user/data/kkbox/'); ap.add_argument('--log', default='kkbox.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--judge-seed', type=int, default=0)
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
a = ap.parse_args()
D = Path(a.data)
t0 = time.time()


def pick(df):
    h = np.array([zlib.crc32(f'{a.seed}:{u}'.encode()) for u in df.msno], dtype=np.uint64)
    return df[(h % 10_000) < a.sample * 10_000].reset_index(drop=True)


tr = pick(pd.read_parquet(D / 'train.parquet')).assign(as_of_date=20170201)
te = pick(pd.read_parquet(D / 'train_v2.parquet')).assign(as_of_date=20170301)
ytr, yte = tr.pop('is_churn').to_numpy(), te.pop('is_churn').to_numpy()
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
users = set(tr.msno) | set(te.msno)
tx = pq.read_table(D / 'transactions.parquet')
tx = tx.filter(pc.is_in(tx['msno'], value_set=pa.array(sorted(users))))
tx = tx.to_pandas()
tx = tx[tx.transaction_date < 20170301].reset_index(drop=True)
mem = pq.read_table(D / 'members_v3.parquet').to_pandas()
mem = mem[mem.msno.isin(users)].reset_index(drop=True)
print(f'train {len(tr)} users (churn {ytr.mean():.3f}), held out {len(te)} (churn {yte.mean():.3f}); '
      f'{len(tx)} transactions, {len(mem)} members', flush=True)


def raw(X):
    return X[['msno', 'as_of_date']].merge(mem, on='msno', how='left').drop(columns=['msno', 'as_of_date'])


def days(v):
    return pd.to_datetime(pd.Series(v).astype('Int64').astype(str), format='%Y%m%d', errors='coerce')


def hand(X):
    out = []
    for d, g in X.groupby('as_of_date'):
        t = tx[(tx.transaction_date < d) & tx.msno.isin(set(g.msno))].sort_values(['msno', 'transaction_date', 'membership_expire_date'])
        last = t.groupby('msno').tail(1).set_index('msno')
        agg = t.groupby('msno').agg(n_tx=('is_cancel', 'size'), n_cancel=('is_cancel', 'sum'), auto_share=('is_auto_renew', 'mean'),
                                    paid_sum=('actual_amount_paid', 'sum'), n_plans=('payment_plan_days', 'nunique'))
        asof = pd.Timestamp(str(d))
        f = pd.DataFrame(index=g.msno)
        for c in ['payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'is_cancel', 'payment_method_id']:
            f[f'last_{c}'] = last[c].reindex(g.msno).to_numpy()
        f['last_discount'] = f.last_plan_list_price - f.last_actual_amount_paid
        f['last_expire_days'] = (days(last.membership_expire_date.reindex(g.msno).to_numpy()) - asof).dt.days.to_numpy()
        f['last_tx_age'] = (asof - days(last.transaction_date.reindex(g.msno).to_numpy())).dt.days.to_numpy()
        f = f.join(agg)
        out.append(f.reset_index(drop=True).set_index(g.index))
    return pd.concat(out).loc[X.index].reset_index(drop=True)


info = {}
Xtr, Xte = raw(tr), raw(te)
if a.arm.startswith('ff'):
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    tr.assign(is_churn=ytr).to_csv(tmp / 'train.csv', index=False); te.to_csv(tmp / 'test.csv', index=False)
    tx.to_parquet(tmp / 'transactions.parquet'); mem.to_csv(tmp / 'members.csv', index=False)
    cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'is_churn', '--id', 'msno', '--task', 'binary', '--time', 'as_of_date',
           '--table', f'tx={tmp / "transactions.parquet"}:msno:transaction_date', '--table', f'members={tmp / "members.csv"}:msno',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True, cwd=a.repo)
    info['pipe_peak_gb'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20, 2)
    Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert (Ftr.msno.astype(str).to_numpy() == tr.msno.to_numpy()).all() and (Fte.msno.astype(str).to_numpy() == te.msno.to_numpy()).all()
    new = [c for c in Ftr.columns if c not in Xtr.columns and c not in ('msno', 'is_churn', 'as_of_date')]
    Xtr = pd.concat([Xtr, Ftr[new].reset_index(drop=True)], axis=1); Xte = pd.concat([Xte, Fte[new].reset_index(drop=True)], axis=1)
    info['n_new'] = len(new)
    shutil.rmtree(tmp, ignore_errors=True)
if a.arm.endswith('hand'):
    Xtr = pd.concat([Xtr, hand(tr)], axis=1); Xte = pd.concat([Xte, hand(te)], axis=1)
fe_t = time.time() - t0
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
va = np.random.default_rng(a.judge_seed).random(len(Xtr)) < 0.15
b = lgb.train(dict(P, seed=a.judge_seed * 10), lgb.Dataset(Xtr[~va], ytr[~va]), 5000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=a.judge_seed * 10 + s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, judge_seed=a.judge_seed,
           logloss=float(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))), logloss_const=float(log_loss(yte, np.full(len(yte), ytr.mean()))),
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0),
           peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2), n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
