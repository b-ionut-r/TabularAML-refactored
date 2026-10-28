"""M5 Forecasting - Accuracy ($50k, 2020): daily unit sales per item at Walmart stores. Slice: one store (CA_1),
the last 150 days before the holdout. Holdout: the last 28 days (the contest horizon, --win 0) or the 28 before.
Calendar (events, SNAP) and weekly prices joined. Metric: RMSSE averaged over items (scale = mean squared
day-to-day change over each item's history) and RMSE. Raw = the joined columns, date as a date. ``hand`` adds
the classic hand-made M5 features: each item's sales means over 7 / 28 / 56 / 112 days and std over 28 days,
all ending 28 days before the row (the horizon, so they exist for every test day), relative price and price
momentum. ``hand_forge`` runs FeatureForge on top of them."""
import sys, time, json, warnings, argparse
warnings.filterwarnings('ignore'); from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--store', default='CA_1'); ap.add_argument('--days', type=int, default=150)
ap.add_argument('--budget', type=float, default=1200); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--hist', type=int, default=0, help='extra labelled days before the training window that only the forecasting family reads')
ap.add_argument('--obj', default='tweedie')
ap.add_argument('--data', default='data/m5/'); ap.add_argument('--log', default='m5.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = a.data
s = pd.read_csv(D + 'sales_train_evaluation.csv'); s = s[s.store_id == a.store]
cal = pd.read_csv(D + 'calendar.csv'); cal['date'] = pd.to_datetime(cal['date'])
dcols = [c for c in s.columns if c.startswith('d_')]
last = len(dcols) - 28 * a.win               # last day index of the holdout
ho_days = dcols[last - 28:last]; tr_days = dcols[last - 28 - a.days - a.hist:last - 28]
hist = s[dcols[:last - 28]].to_numpy(dtype=float)
scale = pd.Series(np.nanmean(np.diff(np.where(np.cumsum(hist, 1) > 0, hist, np.nan), axis=1) ** 2, axis=1), index=s.id.values)
ids = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
L = s[ids + tr_days + ho_days].melt(id_vars=ids, var_name='d', value_name='sales')
L = L.merge(cal[['d', 'date', 'wm_yr_wk', 'event_name_1', 'event_type_1', 'snap_CA']], on='d').drop(columns=['d'])
pr = pd.read_csv(D + 'sell_prices.csv'); pr = pr[pr.store_id == a.store]
L = L.merge(pr, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
L = L[L.sell_price.notna()].reset_index(drop=True)     # not yet on sale
for c in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1']:
    L[c] = L[c].fillna('none').astype('category')
y_all = L.pop('sales').to_numpy(dtype=float); rid = L.pop('id').to_numpy()
L = L.drop(columns=['store_id', 'state_id'])
start = cal.loc[cal.d == ho_days[0], 'date'].iloc[0]
tr = (L.date < start).to_numpy(); ho = ~tr
if a.shuffle:
    y_all[tr] = np.random.default_rng(0).permutation(y_all[tr])   # leakage control
def raw(df):
    df = df.copy(); df['date'] = (df['date'] - pd.Timestamp('2011-01-01')).dt.days.astype(float); return df
def hand(df):
    df = df.copy()
    dix = cal.set_index('date').loc[df['date'], 'd'].str[2:].astype(int).to_numpy() - 1   # 0-based day index
    row = pd.Index(s.id.values).get_indexer(rid_l)
    H = np.nan_to_num(s[dcols[:last - 28]].to_numpy(dtype=float))      # sales known before the holdout
    C = np.concatenate([np.zeros((len(H), 1)), np.cumsum(H, 1)], 1); C2 = np.concatenate([np.zeros((len(H), 1)), np.cumsum(H ** 2, 1)], 1)
    end = np.minimum(dix - 28 + 1, H.shape[1])                            # window ends at day t-28 inclusive
    for w in (7, 28, 56, 112):
        st = np.maximum(end - w, 0); n_w = np.maximum(end - st, 1)
        df[f'lag28_mean{w}'] = (C[row, end] - C[row, st]) / n_w
        if w == 28:
            m2 = (C2[row, end] - C2[row, st]) / n_w
            df['lag28_std28'] = np.sqrt(np.maximum(m2 - df['lag28_mean28'] ** 2, 0))
    g = df.groupby('item_id', observed=True)['sell_price']
    df['price_rel'] = df['sell_price'] / g.transform('mean')
    df['price_norm'] = df['sell_price'] / g.transform('max')
    wk = pr[['item_id', 'wm_yr_wk', 'sell_price']].sort_values(['item_id', 'wm_yr_wk'])
    wk['prev'] = wk.groupby('item_id')['sell_price'].shift(1)
    prev = df[['item_id', 'wm_yr_wk']].astype({'item_id': str}).merge(wk.astype({'item_id': str}), on=['item_id', 'wm_yr_wk'], how='left')['prev'].to_numpy()
    df['price_momentum'] = df['sell_price'].to_numpy() / prev
    return df
if a.arm.startswith('hand'):
    rid_l = rid; L = hand(L)
Xtr, Xho, ytr, yho = L[tr].reset_index(drop=True), L[ho].reset_index(drop=True), y_all[tr], y_all[ho]
first_day = cal.loc[cal.d == dcols[last - 28 - a.days], 'date'].iloc[0]
recent = (Xtr.date >= first_day).to_numpy()            # the judge's training rows; earlier rows only feed the family
Hx, Hy = Xtr, ytr
Xtr, ytr = Xtr[recent].reset_index(drop=True), ytr[recent]
print('rows', len(L), 'train', len(Xtr), 'holdout', len(Xho), flush=True)
t0 = time.time(); info = {}
if a.arm in ('raw', 'hand'):
    Xtr, Xho = raw(Xtr), raw(Xho)
elif a.arm in ('fc', 'hand_fc'):
    # Forecasting family alone (tabularaml/generate/forecast.py) on top of the raw (or hand) columns.
    from tabularaml.generate.forecast import forecast_features
    from tabularaml.generate.forecast import ForecastFeatures
    ff = ForecastFeatures(**json.loads(a.kw)).fit(Hx, Hy, Xho)
    Ftr, Fho = ff.transform(Xtr).reset_index(drop=True), ff.transform(Xho).reset_index(drop=True)
    Xtr, Xho = pd.concat([raw(Xtr), Ftr], axis=1), pd.concat([raw(Xho), Fho], axis=1)
    info = dict(n_fc=Ftr.shape[1])
elif a.arm in ('forge', 'hand_forge'):
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='regression', time_budget=a.budget, random_state=0, n_jobs=4, verbose=True, **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xho)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, time_col=f.time_col_, feats=f.new_columns_[:60])
    Xtr, Xho = raw(f.transform_train(Xtr)), raw(f.transform(Xho))
if a.arm == 'pipe':
    sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
    from _pipe import pipe_features
    Ftr, Xho, info = pipe_features(Hx, Hy, Xho, budget=a.budget)
    Xtr = Ftr[recent].reset_index(drop=True)
    Xtr, Xho = raw(Xtr), raw(Xho)
fe_t = time.time() - t0
P = dict(objective=a.obj, tweedie_variance_power=1.1, learning_rate=0.05, num_leaves=63, min_child_samples=100,
         feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
d = Xtr['date'].to_numpy(); cut = d.max() - 28
i_tr, i_va = np.flatnonzero(d <= cut), np.flatnonzero(d > cut)
b = lgb.train(P, lgb.Dataset(Xtr.iloc[i_tr], ytr[i_tr]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[i_va], ytr[i_va])], callbacks=[lgb.early_stopping(100, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho)
e = pd.DataFrame({'id': rid[ho], 'se': (p - yho) ** 2}).groupby('id')['se'].mean()
sc = scale.reindex(e.index).to_numpy(); ok = np.isfinite(sc) & (sc > 0)
rmsse = float(np.mean(np.sqrt(e.to_numpy()[ok] / sc[ok])))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, rmsse=rmsse, rmse=float(np.sqrt(np.mean((p - yho) ** 2))), best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
