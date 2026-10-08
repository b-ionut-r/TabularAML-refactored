"""GEFCom2012 load track (Kaggle 2012, $7.5k): hourly load of 20 US utility zones, forecasting part.

Data: the contest files as published by its organiser (Tao Hong; GEFCom2012.zip: Load_history.csv,
Load_solution.csv, Holiday_List.csv). The contest asked for the week after the history (2008-07-01 .. 07)
without temperatures for that week, and for 8 earlier missing weeks (backcasting, not covered here: those weeks
lie inside the history).

``--win 0`` is that contest week, scored with the published solution; ``--win k`` the full week ending 7(k-1) days before 2008-06-30 (the history's last full day is 06-29), with
the history up to it. Rows: zone, timestamp, holiday flag (the holiday list is known in advance); temperatures
are left out because the test week had none. Metric: the contest's WRMSE over the 20 zones (weight 1 each) and
their sum, the system (weight 20). Judge: one LightGBM on log1p(load), early-stopped on the last training week.
Arms: ``raw``, ``fc`` (+ forecast.py at its defaults). ``--shuffle`` permutes the training labels.
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--tag', default=''); ap.add_argument('--kw', default='{}')
ap.add_argument('--data', default='data/gefcom12/GEFCOM2012_Data/Load/')
ap.add_argument('--log', default='gefcom12.jsonl'); ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = Path(a.data)
H = [f'h{i}' for i in range(1, 25)]
def long(df):
    df = df.melt(id_vars=['zone_id', 'year', 'month', 'day'], value_vars=H, var_name='h', value_name='y')
    ts = pd.to_datetime(df[['year', 'month', 'day']]) + pd.to_timedelta(df.h.str[1:].astype(int) - 1, unit='h')
    return pd.DataFrame({'zone_id': df.zone_id.to_numpy(), 'ts': ts.to_numpy(), 'y': pd.to_numeric(df.y, errors='coerce').to_numpy()})
hist = long(pd.read_csv(D / 'Load_history.csv', thousands=','))
sol = pd.read_csv(D / 'Load_solution.csv')
sol = long(sol[(sol.zone_id <= 20) & (sol.year == 2008) & (sol.month == 7)].drop(columns=['id', 'weight']))
A = pd.concat([hist.dropna(subset=['y']), sol]).sort_values(['ts', 'zone_id']).reset_index(drop=True)
hol = pd.read_csv(D / 'Holiday_List.csv', index_col=0)
days = {pd.Timestamp(f"{v.split(', ', 1)[1]}, {y}").normalize() for y in hol.columns for v in hol[y].dropna()}
A['holiday'] = A.ts.dt.normalize().isin(days).astype(int)
start = pd.Timestamp('2008-07-01') if a.win == 0 else pd.Timestamp('2008-06-30') - pd.Timedelta(days=7 * a.win)
end = start + pd.Timedelta(days=7)
tr, te = A[A.ts < start].reset_index(drop=True), A[(A.ts >= start) & (A.ts < end)].reset_index(drop=True)
assert te.zone_id.nunique() == 20 and len(te) == 20 * 168, len(te)
ytr = np.log1p(tr.pop('y').to_numpy()); yte = te.pop('y').to_numpy()
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
for X in (tr, te):
    X['ts'] = X.ts.dt.strftime('%Y-%m-%d %H:%M:%S')
print('train', len(tr), 'test', len(te), 'week', start.date(), flush=True)
t0 = time.time(); info = {}
Xtr, Xte = tr.copy(), te.copy()
if a.arm == 'fc':
    from tabularaml.generate.forecast import forecast_features
    Ftr, Fte, ff = forecast_features(Xtr, ytr, Xte, **json.loads(a.kw))
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1)
    info = dict(n_fc=Ftr.shape[1], on=ff.active_, why=getattr(ff, 'reason_', ''))
fe_t = time.time() - t0
for X in (Xtr, Xte):
    d = pd.to_datetime(X['ts'])
    X['ts'] = (d - pd.Timestamp('2004-01-01')).dt.total_seconds().astype(float) / 3600
    X['hour'], X['dow'], X['doy'] = d.dt.hour.astype(float), d.dt.dayofweek.astype(float), d.dt.dayofyear.astype(float)
    X['zone_id'] = pd.Categorical(X['zone_id'], categories=range(1, 21))
P = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_child_samples=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
tt = Xtr['ts'].to_numpy(); va = tt > tt.max() - 168
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 5000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.expm1(lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte))
def wrmse(y, p):
    sy, sp = pd.Series(y).groupby(te.ts.to_numpy()).sum(), pd.Series(p).groupby(te.ts.to_numpy()).sum()
    return float(np.sqrt((np.sum((y - p) ** 2) + 20 * np.sum((sy - sp) ** 2)) / (len(y) + 20 * len(sy))))
c = np.full(len(yte), 0.0)
for z in range(1, 21):
    c[te.zone_id.to_numpy() == z] = np.expm1(np.mean(ytr[tr.zone_id.to_numpy() == z]))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, wrmse=wrmse(yte, p), wrmse_zone_mean=wrmse(yte, c),
           best_it=b.best_iteration, fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
