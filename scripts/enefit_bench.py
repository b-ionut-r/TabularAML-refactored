"""Enefit: Predict Energy Behavior of Prosumers (Kaggle 2024, $50k): hourly production and consumption per
prediction unit (county x business x product) of Estonian solar prosumers.

Data: the contest files (a public Kaggle copy, ``artisusxiren/predict-energy-behavior-of-prosumers``): train.csv,
client.csv, forecast_weather.csv, weather_station_to_county_mapping.csv; 2021-09-01 .. 2023-05-31.

The contest served one day at a time with that day's rows, the day-ahead weather forecast (made the day before),
client counts and capacity from two days earlier, and targets known up to two days earlier. Holdout built that
way: ``--days`` test days (``--win 0`` ends on the last day; ``--win k`` ``--days * k`` days earlier); training
labels end two days before the first test day, so the test horizons run from 25 hours on (the day in between is in
neither). Each row carries the contest's per-day columns: unit keys, hour, installed capacity and client count of
its day's block, and the county mean of the day-ahead forecast weather for that hour (also on training rows, as
the contest gave it). Metric: MAE. Judge: one LightGBM with an L1 objective on the target, early-stopped on the last
training week. Arms: ``raw``, ``fc`` (+ forecast.py at its defaults). ``--shuffle`` permutes the training labels.
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--days', type=int, default=7); ap.add_argument('--tag', default=''); ap.add_argument('--kw', default='{}')
ap.add_argument('--data', default='data/enefit/'); ap.add_argument('--log', default='enefit.jsonl')
ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--budget', type=float, default=900)
a = ap.parse_args()
D = Path(a.data)
W = ['temperature', 'dewpoint', 'cloudcover_total', 'cloudcover_low', '10_metre_u_wind_component', '10_metre_v_wind_component',
     'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall', 'total_precipitation']
cache = D / 'fw_county.parquet'
if not cache.exists():
    st = pd.read_csv(D / 'weather_station_to_county_mapping.csv').dropna(subset=['county'])
    st[['latitude', 'longitude']] = st[['latitude', 'longitude']].round(1)
    fw = pd.read_csv(D / 'forecast_weather.csv', usecols=['latitude', 'longitude', 'data_block_id', 'forecast_datetime'] + W)
    fw[['latitude', 'longitude']] = fw[['latitude', 'longitude']].round(1)
    fw = fw.merge(st[['latitude', 'longitude', 'county']], on=['latitude', 'longitude'])
    fw['datetime'] = pd.to_datetime(fw.forecast_datetime, utc=True).dt.tz_convert('Europe/Tallinn').dt.tz_localize(None)
    fw = fw.groupby(['county', 'data_block_id', 'datetime'], as_index=False)[W].mean()
    fw['county'] = fw.county.astype(int)
    fw.to_parquet(cache)
fw = pd.read_parquet(cache)
tr0 = pd.read_csv(D / 'train.csv', parse_dates=['datetime']).dropna(subset=['target'])
cl = pd.read_csv(D / 'client.csv')[['county', 'is_business', 'product_type', 'data_block_id', 'eic_count', 'installed_capacity']]
A = tr0.merge(cl, on=['county', 'is_business', 'product_type', 'data_block_id'], how='left')
A = A.merge(fw, on=['county', 'data_block_id', 'datetime'], how='left')
day = A.datetime.dt.normalize()
last = day.max() - pd.Timedelta(days=a.days * a.win)
first = last - pd.Timedelta(days=a.days - 1)
cut = first - pd.Timedelta(days=1)                      # labels known up to two days before the first test day
tr = A[day < cut].reset_index(drop=True)
te = A[(day >= first) & (day <= last)].reset_index(drop=True)
drop = ['row_id', 'data_block_id', 'target']
ytr, yte = tr.target.to_numpy(), te.target.to_numpy()
tr, te = tr.drop(columns=drop), te.drop(columns=drop)
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
for X in (tr, te):
    X['datetime'] = X.datetime.dt.strftime('%Y-%m-%d %H:%M:%S')
print('train', len(tr), 'test', len(te), 'days', first.date(), '..', last.date(), flush=True)
t0 = time.time(); info = {}
Xtr, Xte = tr.copy(), te.copy()
if a.arm == 'fc':
    from tabularaml.generate.forecast import forecast_features
    Ftr, Fte, ff = forecast_features(Xtr, ytr, Xte, **json.loads(a.kw))
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1)
    info = dict(n_fc=Ftr.shape[1], on=ff.active_, why=getattr(ff, 'reason_', ''))
if a.arm == 'pipe':
    sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
    from _pipe import pipe_features
    Xtr, Xte, info = pipe_features(Xtr, ytr, Xte, budget=a.budget)
fe_t = time.time() - t0
for X in (Xtr, Xte):
    d = pd.to_datetime(X['datetime'])
    X['datetime'] = (d - pd.Timestamp('2021-09-01')).dt.total_seconds().astype(float) / 3600
    X['hour'], X['dow'], X['doy'] = d.dt.hour.astype(float), d.dt.dayofweek.astype(float), d.dt.dayofyear.astype(float)
P = dict(objective='l1', learning_rate=0.05, num_leaves=255, min_child_samples=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
tt = Xtr['datetime'].to_numpy(); va = tt > tt.max() - 168
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 5000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte).clip(0)
uid = tr.prediction_unit_id.astype(str) + '_' + tr.is_consumption.astype(str)
med = pd.Series(ytr).groupby(uid.to_numpy()).median()
c = (te.prediction_unit_id.astype(str) + '_' + te.is_consumption.astype(str)).map(med).fillna(np.median(ytr)).to_numpy()
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, mae=float(np.mean(np.abs(yte - p))),
           mae_unit_median=float(np.mean(np.abs(yte - c))), best_it=b.best_iteration, fe_s=round(fe_t),
           total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
