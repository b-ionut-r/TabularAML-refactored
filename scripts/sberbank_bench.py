"""Sberbank Russian Housing Market (Kaggle 2017, $25k): sale price of a Moscow flat. RMSLE, as the contest.

Data: a public copy of the contest files (``muhammadhanzlatahir/sberbank-russian-housing-market``: train.csv with the
contest's 30,471 rows and 292 columns, sales 2011-08-20 to 2015-06-30, and macro.csv, one row per day; Kaggle's own
download needs the rules accepted). The contest scored the sales after the training file (July 2015 - May 2016), so a
window holds out the last six months before a cut and trains on everything before: ``--win`` 0 holds out
2015-01-01..2015-06-30, 1 holds out 2014-07-01..2014-12-31. Dates stay as written (absolute). Held-out prices reach
only the scorer. Columns typed as Kaggle's CSV reads.

Arms (same bagged LightGBM on log1p(price): early stopping on the latest 15% of training sales, then 3 seeds):
  ``raw``  the file's columns (timestamp as year, month and day number) with macro.csv joined by date;
  ``ff``   blind: ``scripts/contest_features.py`` at its defaults (``--log-target``), macro.csv given as a table;
  ``hand`` raw + the top public kernels' features: floor over max floor, living / kitchen over full area, rooms per
           area, building age at sale, sales in the same month and week, month and day of week;
  ``ff_hand`` both. ``--shuffle`` permutes the training prices (leakage control). Reports runtime and peak memory.
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil, resource
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--tag', default=''); ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='/home/user/data/sberbank/'); ap.add_argument('--log', default='sberbank.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--judge-seed', type=int, default=0)
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
a = ap.parse_args()
D = Path(a.data)
t0 = time.time()
df = pd.read_csv(D / 'train.csv', low_memory=False)
macro = pd.read_csv(D / 'macro.csv', low_memory=False)
lo, hi = [('2015-01-01', '2015-06-30'), ('2014-07-01', '2014-12-31')][a.win]
tr = df[df.timestamp < lo].reset_index(drop=True)
te = df[(df.timestamp >= lo) & (df.timestamp <= hi)].reset_index(drop=True)
ytr, yte = tr.pop('price_doc').to_numpy(dtype=float), te.pop('price_doc').to_numpy(dtype=float)
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
print(f'train {len(tr)} sales to {tr.timestamp.max()}, held out {len(te)} ({lo}..{hi})', flush=True)


def raw(X):
    X = X.merge(macro, on='timestamp', how='left')
    t = pd.to_datetime(X.pop('timestamp'))
    X['year'], X['month'], X['day_number'] = t.dt.year, t.dt.month, (t - pd.Timestamp('2010-01-01')).dt.days
    return X.drop(columns=['id'])


def hand(X):
    t = pd.to_datetime(X.timestamp)
    both = pd.concat([tr.timestamp, te.timestamp])   # sale counts need no prices
    tb = pd.to_datetime(both)
    H = pd.DataFrame(index=X.index)
    H['floor_ratio'] = X.floor / X.max_floor.replace(0, np.nan)
    H['life_ratio'] = X.life_sq / X.full_sq.replace(0, np.nan)
    H['kitch_ratio'] = X.kitch_sq / X.full_sq.replace(0, np.nan)
    H['sq_per_room'] = X.full_sq / X.num_room.replace(0, np.nan)
    H['age_at_sale'] = t.dt.year - X.build_year
    H['month_sales'] = t.dt.to_period('M').map(tb.dt.to_period('M').value_counts())
    H['week_sales'] = (t.dt.year * 100 + t.dt.isocalendar().week.astype(int)).map(
        (tb.dt.year * 100 + tb.dt.isocalendar().week.astype(int)).value_counts())
    H['dow'] = t.dt.dayofweek
    return H.astype(float).reset_index(drop=True)


info = {}
Xtr, Xte = raw(tr), raw(te)
if a.arm.startswith('ff'):
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    tr.assign(price_doc=ytr).to_csv(tmp / 'train.csv', index=False); te.to_csv(tmp / 'test.csv', index=False)
    cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'price_doc', '--id', 'id', '--task', 'regression', '--log-target',
           '--table', f'macro={D / "macro.csv"}:timestamp', '--budget', str(a.budget), '--out-dir', str(tmp / 'out')] + \
          (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True, cwd=a.repo)
    info['pipe_peak_gb'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20, 2)
    Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert (Ftr['id'].to_numpy() == tr.id.to_numpy()).all() and (Fte['id'].to_numpy() == te.id.to_numpy()).all()
    new = [c for c in Ftr.columns if c not in tr.columns and c not in macro.columns and c not in ('price_doc',)]
    Xtr = pd.concat([Xtr, Ftr[new].reset_index(drop=True)], axis=1); Xte = pd.concat([Xte, Fte[new].reset_index(drop=True)], axis=1)
    info['n_new'] = len(new)
    shutil.rmtree(tmp, ignore_errors=True)
if a.arm.endswith('hand'):
    Xtr = pd.concat([Xtr, hand(tr)], axis=1); Xte = pd.concat([Xte, hand(te)], axis=1)
fe_t = time.time() - t0
Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]; Xte = Xte[Xtr.columns]
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
ly = np.log1p(ytr)
order = np.argsort(tr.timestamp.to_numpy(), kind='stable'); cut = int(len(order) * 0.85)
fit, va = order[:cut], order[cut:]
b = lgb.train(dict(P, seed=a.judge_seed * 10), lgb.Dataset(Xtr.iloc[fit], ly[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ly[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=a.judge_seed * 10 + s), lgb.Dataset(Xtr, ly), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
rmsle = float(np.sqrt(np.mean((p - np.log1p(yte)) ** 2)))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, judge_seed=a.judge_seed, rmsle=rmsle,
           rmsle_const=float(np.sqrt(np.mean((np.median(np.log1p(ytr)) - np.log1p(yte)) ** 2))), best_it=b.best_iteration,
           n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0),
           peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2), n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
