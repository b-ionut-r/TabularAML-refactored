"""Rossmann Store Sales (Kaggle 2015, $35k) through scripts/contest_features.py, as a Kaggle entry is made.

The last 48 days of train.csv (the length of the contest's test period) are held out and written in
test.csv's format (Id, Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday; no Sales,
no Customers); the earlier rows are the training file, store.csv the extra table. Metric: RMSPE over
held-out days with sales, scored with one LightGBM on log1p(Sales) fitted on open days with sales.

``--rows all`` gives contest_features.py every training row (closed days with zero sales included),
``--rows open`` only open days with sales (what contest kernels trained on). ``--log-target`` is
passed through. ``--arm raw`` scores the training file's columns with store.csv joined.
"""
import sys, time, json, warnings, argparse, subprocess, tempfile
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--rows', default='all')
ap.add_argument('--log-target', action='store_true'); ap.add_argument('--budget', type=float, default=900)
ap.add_argument('--tag', default=''); ap.add_argument('--data', default='data/rossmann/'); ap.add_argument('--log', default='rossmann.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--keep', default='', help='keep the output directory here')
ap.add_argument('--forge-kw', default='{}')
a = ap.parse_args()
D = Path(a.data)
tr = pd.read_csv(D / 'train.csv', dtype={'StateHoliday': str}).drop(columns=['Customers'])
st = pd.read_csv(D / 'store.csv')
d = pd.to_datetime(tr.Date)
start = d.max() - pd.Timedelta(days=47)
ho = tr[d >= start].reset_index(drop=True)
tr = tr[d < start].reset_index(drop=True)
if a.rows == 'open':
    tr = tr[(tr.Open == 1) & (tr.Sales > 0)].reset_index(drop=True)
if a.shuffle:
    tr['Sales'] = np.random.default_rng(0).permutation(tr.Sales.to_numpy())
te = ho.drop(columns=['Sales']).assign(Id=np.arange(1, len(ho) + 1))[['Id', 'Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']]
t0 = time.time(); info = {}
if a.arm == 'raw':
    Xtr, Xte = tr.merge(st, on='Store', how='left'), te.merge(st, on='Store', how='left')
    ytr_all = Xtr.pop('Sales').to_numpy(dtype=float)
else:
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    tr.to_csv(tmp / 'train.csv', index=False); te.to_csv(tmp / 'test.csv', index=False); st.to_csv(tmp / 'store.csv', index=False)
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'Sales', '--task', 'regression', '--budget', str(a.budget),
           '--out-dir', str(tmp / 'out'), '--table', f'store={tmp / "store.csv"}'] + (['--log-target'] if a.log_target else []) + ['--forge-kw', a.forge_kw]
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet')
    ytr_all = Xtr.pop('Sales').to_numpy(dtype=float)
    Xtr = Xtr.drop(columns=['Id'], errors='ignore')
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    info = dict(n_cols=Xtr.shape[1])
fe_t = time.time() - t0
for X in (Xtr, Xte):
    X['Date'] = (pd.to_datetime(X['Date'].astype(str)) - pd.Timestamp('2013-01-01')).dt.days.astype(float)
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
fit = (Xtr['Open'].to_numpy() == 1) & (ytr_all > 0)
Xf, yf = Xtr[fit], np.log1p(ytr_all[fit])
P = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
dd = Xf['Date'].to_numpy(); va = dd > dd.max() - 48
b = lgb.train(P, lgb.Dataset(Xf[~va], yf[~va]), 10000, valid_sets=[lgb.Dataset(Xf[va], yf[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.expm1(lgb.train(P, lgb.Dataset(Xf, yf), int(b.best_iteration * 1.1) + 1).predict(Xte))
s = ho.Sales.to_numpy(dtype=float); m = s > 0
rmspe = float(np.sqrt(np.mean(((s[m] - p[m]) / s[m]) ** 2)))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), rows=a.rows, log_target=a.log_target, rmspe=rmspe, best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
