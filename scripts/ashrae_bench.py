"""ASHRAE Great Energy Predictor III (Kaggle 2019, $25k): hourly meter readings per building.

Data: the contest's source, the Building Data Genome 2 project (github.com/buds-lab/building-data-genome-project-2:
data/meters/cleaned/electricity_cleaned.csv, data/weather/weather.csv, data/metadata/metadata.csv), 2016-2017.
The contest trained on 2016 and scored the following years for the same buildings, with the weather known.

Holdout built that way: ``--buildings`` random buildings with electricity readings in both years (``--seed``
picks the sample); training rows are their 2016 hours with a reading, the unlabeled rows every 2017 hour of
the same buildings (the test file lists all of them), scored where a reading exists. Columns: building id,
site, primary use, floor area, year built, timestamp, and the site's weather that hour. Metric: RMSLE.
Judge: one LightGBM on log1p(reading), early-stopped on December 2016. Arms: ``raw``, ``fc``
(+ tabularaml/generate/forecast.py at its defaults). ``--shuffle`` permutes the training labels.
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--buildings', type=int, default=300); ap.add_argument('--tag', default=''); ap.add_argument('--kw', default='{}')
ap.add_argument('--data', default='data/bdg2/'); ap.add_argument('--log', default='ashrae.jsonl'); ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--budget', type=float, default=900)
a = ap.parse_args()
D = Path(a.data)
el = pd.read_csv(D / 'electricity_cleaned.csv', parse_dates=['timestamp'])
md = pd.read_csv(D / 'metadata.csv')[['building_id', 'site_id', 'primaryspaceusage', 'sqm', 'yearbuilt']]
we = pd.read_csv(D / 'weather.csv', parse_dates=['timestamp'])
y16, y17 = el.timestamp.dt.year == 2016, el.timestamp.dt.year == 2017
ok = [c for c in el.columns[1:] if el.loc[y16, c].notna().mean() > 0.5 and el.loc[y17, c].notna().mean() > 0.5]
rng = np.random.default_rng(a.seed)
bs = list(rng.choice(ok, a.buildings, replace=False))
L = el[['timestamp'] + bs].melt(id_vars='timestamp', var_name='building_id', value_name='y')
L = L.merge(md, on='building_id', how='left').merge(we, on=['site_id', 'timestamp'], how='left')
L['timestamp'] = L.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S')
tr = L[L.timestamp < '2017'].dropna(subset=['y']).reset_index(drop=True)
te = L[L.timestamp >= '2017'].reset_index(drop=True)
ytr = np.log1p(tr.pop('y').clip(lower=0).to_numpy())
if a.shuffle:
    ytr = rng.permutation(ytr)
yte = te.pop('y').to_numpy()
print('train', len(tr), 'test grid', len(te), flush=True)
t0 = time.time(); info = {}
Xtr, Xte = tr, te
if a.arm == 'fc':
    from tabularaml.generate.forecast import forecast_features
    Ftr, Fte, ff = forecast_features(Xtr, ytr, Xte, **json.loads(a.kw))
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1)
    info = dict(n_fc=Ftr.shape[1], on=ff.active_)
if a.arm == 'pipe':
    sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
    from _pipe import pipe_features
    Xtr, Xte, info = pipe_features(Xtr, ytr, Xte, budget=a.budget)
fe_t = time.time() - t0
for X in (Xtr, Xte):
    d = pd.to_datetime(X['timestamp'])
    X['timestamp'] = (d - pd.Timestamp('2016-01-01')).dt.total_seconds().astype(float) / 3600
    X['hour'], X['dow'] = d.dt.hour.astype(float), d.dt.dayofweek.astype(float)
for c in ['building_id', 'site_id', 'primaryspaceusage']:
    cats = pd.Index(pd.unique(Xtr[c].astype(str)))
    Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_child_samples=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
dd = Xtr['timestamp'].to_numpy(); va = dd >= (pd.Timestamp('2016-12-01') - pd.Timestamp('2016-01-01')).days * 24
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 5000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.clip(lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte), 0, None)
m = np.isfinite(yte)
yl = np.log1p(np.clip(yte[m], 0, None))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, rmsle=float(np.sqrt(np.mean((p[m] - yl) ** 2))),
           rmsle_const=float(np.sqrt(np.mean((np.mean(ytr) - yl) ** 2))), best_it=b.best_iteration, fe_s=round(fe_t),
           total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=int(m.sum()), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
