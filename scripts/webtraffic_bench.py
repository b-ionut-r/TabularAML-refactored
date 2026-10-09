"""Web Traffic Time Series Forecasting (Kaggle 2017, $25k): daily views of Wikipedia pages.

Data: the contest's series (Monash archive, Zenodo record 4656080, ``kaggle_web_traffic_dataset_with_missing_values.tsf``:
145,063 pages, 2015-07-01 .. 2017-09-10; page names are replaced by T1..). A random sample of ``--pages`` pages.

Holdout built like the contest's second stage: data up to a cut, then a 2-day gap, then 62 days to
forecast (``--win 0`` ends on the last day; ``--win 1`` 64 days earlier). The test file lists every page
for every day of the window, so the unlabeled rows are that full grid; days without a recorded value are
not scored (the contest scored them as missing). Training rows are the last ``--days`` days before the cut
with a value. Metric: SMAPE. Judge: one LightGBM with an L1 objective on log1p(views), early-stopped on the
latest 62 training days. Arms: ``raw`` (page id, date as a day number, weekday), ``fc`` (+ forecast.py).
``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--pages', type=int, default=5000); ap.add_argument('--days', type=int, default=500)
ap.add_argument('--tag', default=''); ap.add_argument('--kw', default='{}')
ap.add_argument('--data', default='data/wt/kaggle_web_traffic_dataset_with_missing_values.tsf')
ap.add_argument('--log', default='webtraffic.jsonl'); ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
rows = []
with open(a.data) as f:
    for line in f:
        if line.startswith(('#', '@')) or not line.strip():
            continue
        name, start, vals = line.rstrip('\n').split(':', 2)
        rows.append((name, vals))
rng = np.random.default_rng(0)
pick = rng.choice(len(rows), a.pages, replace=False)
names = [rows[i][0] for i in pick]
V = np.array([[np.nan if v == '?' else float(v) for v in rows[i][1].split(',')] for i in pick])
del rows
dates = pd.date_range('2015-07-01', periods=V.shape[1])
last = V.shape[1] - 1 - 64 * a.win                    # last day of the holdout window
cut = last - 64                                       # last labelled day; then 2 days' gap
ho_days = np.arange(cut + 3, last + 1)
tr_days = np.arange(max(0, cut - a.days + 1), cut + 1)
long = lambda days: pd.DataFrame({'page': np.repeat(names, len(days)), 'date': np.tile(dates[days].strftime('%Y-%m-%d'), len(names)),
                                  'y': V[:, days].ravel()})
tr, ho = long(tr_days), long(ho_days)
tr = tr[tr.y.notna()].reset_index(drop=True)
ytr = np.log1p(tr.pop('y').to_numpy())
if a.shuffle:
    ytr = rng.permutation(ytr)
yho = ho.pop('y').to_numpy()
print('train', len(tr), 'holdout grid', len(ho), flush=True)
t0 = time.time(); info = {}
Xtr, Xho = tr.copy(), ho.copy()
if a.arm == 'fc':
    from tabularaml.generate.forecast import forecast_features
    Ftr, Fho, ff = forecast_features(Xtr, ytr, Xho, **json.loads(a.kw))
    Xtr, Xho = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xho, Fho], axis=1)
    info = dict(n_fc=Ftr.shape[1])
fe_t = time.time() - t0
for X in (Xtr, Xho):
    d = pd.to_datetime(X['date'])
    X['date'] = (d - pd.Timestamp('2015-07-01')).dt.days.astype(float)
    X['dow'] = d.dt.dayofweek.astype(float)
    X['page'] = pd.Categorical(X['page'], categories=names)
P = dict(objective='l1', learning_rate=0.05, num_leaves=127, min_child_samples=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0, max_cat_to_onehot=4, cat_smooth=10)
dd = Xtr['date'].to_numpy(); va = dd > dd.max() - 62
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 5000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.expm1(np.clip(lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho), 0, None))
m = np.isfinite(yho)
def smape(y, p):
    den = np.abs(y) + np.abs(p)
    return float(200 * np.mean(np.where(den > 0, np.abs(y - p) / np.where(den > 0, den, 1), 0.0)))
med = np.expm1(np.median(ytr))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, smape=smape(yho[m], np.round(p[m])),
           smape_const=smape(yho[m], np.full(m.sum(), np.round(med))), best_it=b.best_iteration, fe_s=round(fe_t),
           total_s=round(time.time() - t0), n_tr=len(Xtr), n_ho=int(m.sum()), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
