"""Walmart Recruiting - Store Sales Forecasting (Kaggle 2014): weekly sales per store x department.

Data: the contest files (Hugging Face mirror ``large-traversaal/Walmart-sales``): train.csv (Store, Dept,
Date, Weekly_Sales, IsHoliday), stores.csv (Type, Size), features.csv (temperature, fuel price, markdowns,
CPI, unemployment per store and week, covering the test weeks too).

Holdout built like the contest's test file: the last 39 weeks of train.csv (the test file's length,
``--win 0``) or the 39 weeks before (``--win 1``), as rows of Store, Dept, Date, IsHoliday with stores.csv
and features.csv joined; training rows are the weeks before it. Metric: WMAE (holiday weeks weigh 5).
Judge: one LightGBM with an L1 objective and the same weights, early-stopped on the latest 39 training
weeks. Arms: ``raw`` (joined columns, date as a day number), ``fc`` (+ tabularaml/generate/forecast.py),
``pipe`` (the files given to ``scripts/contest_features.py`` at its defaults). ``--shuffle`` permutes the
training labels (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/walmart/'); ap.add_argument('--log', default='walmart.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--kw', default='{}')
a = ap.parse_args()
D = Path(a.data)
tr = pd.read_csv(D / 'train.csv')
st = pd.read_csv(D / 'stores.csv')
fe = pd.read_csv(D / 'features.csv').drop(columns=['IsHoliday'])
d = pd.to_datetime(tr.Date)
end = d.max() - pd.Timedelta(weeks=39 * a.win)
start = end - pd.Timedelta(weeks=38)
tr = tr[d <= end]
d = pd.to_datetime(tr.Date)
ho = tr[d >= start].reset_index(drop=True)
tr = tr[d < start].reset_index(drop=True)
ytr = tr.pop('Weekly_Sales').to_numpy(dtype=float)
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
te = ho.drop(columns=['Weekly_Sales'])
join = lambda df: df.merge(st, on='Store', how='left').merge(fe, on=['Store', 'Date'], how='left')
t0 = time.time(); info = {}
if a.arm in ('raw', 'fc'):
    Xtr, Xte = join(tr), join(te)
    if a.arm == 'fc':
        from tabularaml.generate.forecast import forecast_features
        Ftr, Fte, ff = forecast_features(Xtr, ytr, Xte, **json.loads(a.kw))
        Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1)
        info = dict(n_fc=Ftr.shape[1])
else:
    tmp = Path(tempfile.mkdtemp())
    tr.assign(Weekly_Sales=ytr).to_csv(tmp / 'train.csv', index=False); te.to_csv(tmp / 'test.csv', index=False)
    fe.to_csv(tmp / 'features.csv', index=False)
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'Weekly_Sales', '--task', 'regression', '--budget', str(a.budget),
           '--out-dir', str(tmp / 'out'), '--table', f'stores={D / "stores.csv"}', '--table', f'features={tmp / "features.csv"}']
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['Weekly_Sales'])
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    info = dict(n_cols=Xtr.shape[1])
fe_t = time.time() - t0
for X in (Xtr, Xte):
    X['Date'] = (pd.to_datetime(X['Date'].astype(str)) - pd.Timestamp('2010-01-01')).dt.days.astype(float)
    X['IsHoliday'] = X['IsHoliday'].astype(str).str.upper().eq('TRUE').astype(float)
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
wtr = np.where(Xtr['IsHoliday'].to_numpy() == 1, 5.0, 1.0)
P = dict(objective='l1', learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
dd = Xtr['Date'].to_numpy(); va = dd > dd.max() - 7 * 39
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va], weight=wtr[~va]), 10000,
              valid_sets=[lgb.Dataset(Xtr[va], ytr[va], weight=wtr[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr, weight=wtr), int(b.best_iteration * 1.1) + 1).predict(Xte)
s = ho.Weekly_Sales.to_numpy(dtype=float)
w = np.where(Xte['IsHoliday'].to_numpy() == 1, 5.0, 1.0)
wmae = float(np.sum(w * np.abs(s - p)) / np.sum(w))
const = float(np.sum(w * np.abs(s - np.median(ytr))) / np.sum(w))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, wmae=wmae, wmae_const=const, best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
