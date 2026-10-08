"""Recruit Restaurant Visitor Forecasting (Kaggle 2018, $25k): daily visitors per restaurant.

Data: the contest files (GitHub ``MengenL-ds/Forecasting-Restaurant-Visitor-Demand-with-Machine-Learning``
data/raw): air_visit_data (store, date, visitors), air_store_info, hpg_store_info, store_id_relation,
date_info, air_reserve and hpg_reserve (reservations with visit and booking times).

Holdout built like the contest's test file: the last 39 days of the data (``--win 0``) or the 39 days
before (``--win 1``); training rows are the visits before it. The test file lists every store x day of
the window (open or not), so the unlabeled rows are that full grid; only days with recorded visits are
scored (the contest ignored the rest). Reservations are cut at the holdout start (the contest's
reservation files end with its training period). Metric: RMSLE of visitors.

Arms: ``raw`` (store id, date as a day number, store info and date info joined, day of week);
``pipe`` (the contest files given to ``scripts/contest_features.py`` at its defaults, nothing else);
``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/recruit/'); ap.add_argument('--log', default='recruit.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = Path(a.data)
v = pd.read_csv(D / 'air_visit_data.csv')
v['visit_date'] = pd.to_datetime(v['visit_date'])
end = v.visit_date.max() - pd.Timedelta(days=39 * a.win)
start = end - pd.Timedelta(days=38)
v = v[v.visit_date <= end]
trv = v[v.visit_date < start].reset_index(drop=True)
hov = v[v.visit_date >= start].reset_index(drop=True)
stores = trv.air_store_id.unique()
grid = pd.MultiIndex.from_product([stores, pd.date_range(start, end)], names=['air_store_id', 'visit_date']).to_frame(index=False)
hov = hov[hov.air_store_id.isin(stores)]
ytr = np.log1p(trv.visitors.to_numpy(dtype=float))
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time(); info = {}
fmt = lambda df: df.assign(visit_date=df.visit_date.dt.strftime('%Y-%m-%d'))
if a.arm == 'raw':
    si = pd.read_csv(D / 'air_store_info.csv'); di = pd.read_csv(D / 'date_info.csv').rename(columns={'calendar_date': 'visit_date'})
    def raw(df):
        df = fmt(df[['air_store_id', 'visit_date']]).merge(si, on='air_store_id', how='left').merge(di, on='visit_date', how='left')
        df['visit_date'] = (pd.to_datetime(df.visit_date) - pd.Timestamp('2016-01-01')).dt.days.astype(float)
        return df
    Xtr, Xg = raw(trv), raw(grid)
else:
    tmp = Path(tempfile.mkdtemp())
    tr_csv = fmt(trv[['air_store_id', 'visit_date']]).assign(visitors=np.expm1(ytr))
    tr_csv.to_csv(tmp / 'train.csv', index=False); fmt(grid).to_csv(tmp / 'test.csv', index=False)
    tabs = []
    for f in ['air_store_info', 'hpg_store_info', 'store_id_relation', 'date_info']:
        tabs += ['--table', f'{f}={D / (f + ".csv")}']
    for f in ['air_reserve', 'hpg_reserve']:
        r = pd.read_csv(D / (f + '.csv'))
        r = r[pd.to_datetime(r.reserve_datetime) < start]
        r.to_csv(tmp / (f + '.csv'), index=False); tabs += ['--table', f'{f}={tmp / (f + ".csv")}']
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'visitors', '--task', 'regression', '--log-target',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out')] + tabs
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['visitors'])
    Xg = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    info = dict(n_cols=Xtr.shape[1])
fe_t = time.time() - t0
for X in (Xtr, Xg):
    if X['visit_date'].dtype != float:
        X['visit_date'] = (pd.to_datetime(X['visit_date'].astype(str)) - pd.Timestamp('2016-01-01')).dt.days.astype(float)
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xg[c] = pd.Categorical(Xg[c].astype(str), categories=cats)
P = dict(objective='regression', learning_rate=0.03, num_leaves=63, min_child_samples=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
d = Xtr['visit_date'].to_numpy(); va = d > d.max() - 39
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 10000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
pg = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xg)
# Score the grid rows with recorded visits.
key = grid.assign(p=pg).merge(hov[['air_store_id', 'visit_date', 'visitors']], on=['air_store_id', 'visit_date'])
rmsle = float(np.sqrt(np.mean((np.clip(key.p, 0, None) - np.log1p(key.visitors)) ** 2)))
const = float(np.sqrt(np.mean((np.mean(ytr) - np.log1p(key.visitors)) ** 2)))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, rmsle=rmsle, rmsle_const=const, best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_grid=len(Xg), n_scored=len(key), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
