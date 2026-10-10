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
ap.add_argument('--forge-kw', default='{}'); ap.add_argument('--win', type=int, default=0, help='holdout ends 48 x win days before the last day')
ap.add_argument('--customers', action='store_true', help='keep the training-only Customers column for the forecasting family')
ap.add_argument('--stores', type=int, default=0, help='random sample of stores (quick runs)')
a = ap.parse_args()
D = Path(a.data)
tr = pd.read_csv(D / 'train.csv', dtype={'StateHoliday': str})
if not a.customers:
    tr = tr.drop(columns=['Customers'])
st = pd.read_csv(D / 'store.csv')
if a.stores:
    keep = np.random.default_rng(0).choice(st.Store.to_numpy(), a.stores, replace=False)
    tr, st = tr[tr.Store.isin(keep)].reset_index(drop=True), st[st.Store.isin(keep)].reset_index(drop=True)
d = pd.to_datetime(tr.Date)
if a.win:
    tr = tr[d <= d.max() - pd.Timedelta(days=48 * a.win)].reset_index(drop=True); d = pd.to_datetime(tr.Date)
start = d.max() - pd.Timedelta(days=47)
ho = tr[d >= start].reset_index(drop=True)
tr = tr[d < start].reset_index(drop=True)
if a.rows == 'open':
    tr = tr[(tr.Open == 1) & (tr.Sales > 0)].reset_index(drop=True)
if a.shuffle:
    tr['Sales'] = np.random.default_rng(0).permutation(tr.Sales.to_numpy())
te = ho.drop(columns=['Sales', 'Customers'], errors='ignore').assign(Id=np.arange(1, len(ho) + 1))[['Id', 'Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']]


def hand(Xtr, ytr, Xte):
    """Features from the winners' public write-ups and top public kernels: per-store sales levels over the last
    quarter / half-year / year of training (overall, by weekday, by promo), sales per customer, competition and
    Promo2 age, Promo2 month, and days since / until state and school holidays and promotions per store."""
    cust = pd.read_csv(D / 'train.csv', usecols=['Store', 'Date', 'Customers'])
    both = pd.concat([Xtr.assign(_tr=1, _y=ytr), Xte.assign(_tr=0, _y=np.nan)], ignore_index=True)
    both['_d'] = pd.to_datetime(both.Date)
    both = both.merge(cust.assign(_d=pd.to_datetime(cust.Date)).drop(columns='Date'), on=['Store', '_d'], how='left')
    both.loc[both._tr == 0, 'Customers'] = np.nan                         # unknown on test days
    end = both.loc[both._tr == 1, '_d'].max()
    L = both[(both._tr == 1) & (both.Open == 1) & (both._y > 0)].copy(); L['ly'] = np.log1p(L._y)
    out = {}
    for days in (90, 180, 365):
        R = L[L._d > end - pd.Timedelta(days=days)]
        out[f'h_store_mean{days}'] = both.Store.map(R.groupby('Store').ly.mean())
        k = R.groupby(['Store', 'DayOfWeek']).ly.mean()
        out[f'h_store_dow_mean{days}'] = pd.Series(k.reindex(pd.MultiIndex.from_frame(both[['Store', 'DayOfWeek']])).to_numpy())
        k = R.groupby(['Store', 'Promo']).ly.mean()
        out[f'h_store_promo_mean{days}'] = pd.Series(k.reindex(pd.MultiIndex.from_frame(both[['Store', 'Promo']])).to_numpy())
    out['h_store_cust_mean'] = both.Store.map(L.groupby('Store').Customers.mean())
    out['h_store_spc'] = both.Store.map((L.groupby('Store')._y.sum() / L.groupby('Store').Customers.sum()))
    y, m = both._d.dt.year, both._d.dt.month
    out['h_comp_months'] = (12 * (y - both.CompetitionOpenSinceYear) + m - both.CompetitionOpenSinceMonth).clip(lower=0)
    wk = both._d.dt.isocalendar().week.astype(float)
    out['h_promo2_weeks'] = (52 * (y - both.Promo2SinceYear) + wk - both.Promo2SinceWeek).clip(lower=0) * both.Promo2
    mon = both._d.dt.strftime('%b').replace('Sep', 'Sept')
    out['h_promo2_month'] = [int(isinstance(iv, str) and mm in iv.split(',')) for iv, mm in zip(both.PromoInterval, mon)]
    both = both.sort_values(['Store', '_d'])
    for name, flag in (('state', both.StateHoliday.astype(str) != '0'), ('school', both.SchoolHoliday == 1), ('promo', both.Promo == 1)):
        t = both._d.where(flag)
        g = both.Store
        last = t.groupby(g).ffill(); nxt = t.groupby(g).bfill()
        out[f'h_since_{name}'] = ((both._d - last).dt.days).reindex(both.index)
        out[f'h_until_{name}'] = ((nxt - both._d).dt.days).reindex(both.index)
    H = pd.DataFrame({k: pd.Series(np.asarray(v, dtype=float)) if not isinstance(v, pd.Series) else v.astype(float) for k, v in out.items()})
    H = H.sort_index()
    n = len(Xtr)
    return (pd.concat([Xtr.reset_index(drop=True), H.iloc[:n].reset_index(drop=True)], axis=1),
            pd.concat([Xte.reset_index(drop=True), H.iloc[n:].reset_index(drop=True)], axis=1))

t0 = time.time(); info = {}
if a.arm in ('raw', 'fc', 'hand', 'hand_fc'):
    Xtr, Xte = tr.merge(st, on='Store', how='left'), te.drop(columns=['Id']).merge(st, on='Store', how='left')
    ytr_all = Xtr.pop('Sales').to_numpy(dtype=float)
    if a.arm.startswith('hand'):
        Xtr, Xte = hand(Xtr, ytr_all, Xte)
    if a.arm.endswith('fc'):
        # Forecasting family alone (tabularaml/generate/forecast.py) on top of the raw columns.
        from tabularaml.generate.forecast import forecast_features
        Ftr, Fte, ff = forecast_features(Xtr, ytr_all, Xte, **json.loads(a.forge_kw))
        Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1)
    Xtr = Xtr.drop(columns=['Customers'], errors='ignore')   # not in the test file
    info = dict(n_cols=Xtr.shape[1])
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
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, rows=a.rows, log_target=a.log_target, rmspe=rmspe, best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
