"""Corporación Favorita Grocery Sales Forecasting ($30k, 2018): daily unit sales per store x item.
Slice: a few stores, 2017-05-01 .. 2017-08-15, zero-filled store x item x day grid. Holdout: the last 16 days
(the contest's test horizon) or the 16 days before (--win 1). Target log1p(sales clipped at 0); metric NWRMSLE
(perishable items weight 1.25).

Test-file parity: the contest's train file lists only store x item x days with a recorded sale (and their
promotion flag), while its test file lists every pair with its promotion flag. So the holdout is the full
grid of the store x item pairs seen before the holdout (a pair first sold inside the holdout would reveal
that sale), and ``onpromotion`` is left out: it is known only on days with a sale, so on the grid
"promotion" would mean "sold". ``--shuffle`` permutes the training labels (a leakage control: every arm
must then score like a constant)."""
import sys, time, json, warnings, argparse
warnings.filterwarnings('ignore'); from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq, pyarrow.compute as pc
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--stores', default='44,45,47'); ap.add_argument('--start', default='2017-05-01')
ap.add_argument('--budget', type=float, default=1200); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/favorita/'); ap.add_argument('--log', default='favorita.jsonl'); ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = a.data
stores = [int(s) for s in a.stores.split(',')]
t = pq.read_table(D + 'train.parquet', columns=['date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion'],
                  filters=[('date', '>=', a.start), ('store_nbr', 'in', stores)]).to_pandas()
t['date'] = pd.to_datetime(t['date'])
end = t.date.max() - pd.Timedelta(days=16 * a.win)
start = end - pd.Timedelta(days=15)
t = t[t.date <= end]
# Zero-filled grid over the store x item pairs sold before the holdout.
pairs = t.loc[t.date < start, ['store_nbr', 'item_nbr']].drop_duplicates()
days = pd.date_range(t.date.min(), t.date.max())
grid = pairs.merge(pd.DataFrame({'date': days}), how='cross')
g = grid.merge(t, on=['date', 'store_nbr', 'item_nbr'], how='left')
g['unit_sales'] = g['unit_sales'].fillna(0).clip(lower=0)
g = g.drop(columns=['onpromotion'])
g = g.merge(pd.read_parquet(D + 'items.parquet'), on='item_nbr', how='left').merge(pd.read_parquet(D + 'stores.parquet'), on='store_nbr', how='left')
for c in ['family', 'city', 'state', 'type']:
    g[c] = g[c].astype('category')
y_all = np.log1p(g.pop('unit_sales').to_numpy())
tr = (g.date < start).to_numpy(); ho = ((g.date >= start) & (g.date <= end)).to_numpy()
print('rows', len(g), 'train', tr.sum(), 'holdout', ho.sum(), 'zero share', float((y_all == 0).mean()), flush=True)
def raw(df):
    df = df.copy(); df['date'] = (df['date'] - pd.Timestamp('2017-01-01')).dt.days.astype(float); return df
Xtr, Xho, ytr, yho = g[tr].reset_index(drop=True), g[ho].reset_index(drop=True), y_all[tr], y_all[ho]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
U = Xho  # the holdout's features (the test file)
t0 = time.time(); info = {}
if a.arm == 'raw':
    Xtr, Xho = raw(Xtr), raw(Xho)
elif a.arm.startswith('hand'):
    # Winners' core features: per store x item mean log sales over the last 7 / 14 / 28 / 56 days before the
    # holdout (one anchor), per weekday means, promo counts over the horizon.
    Xtr, Xho = raw(Xtr), raw(Xho)
    from tabularaml.generate.forge import LaggedTargetMean
    for w in (7, 14, 28, 56):
        for keys in (['store_nbr', 'item_nbr'],):
            sp = LaggedTargetMean(keys, 'date', w, 16).fit(Xtr, ytr, None)
            Xtr[sp.name], Xho[sp.name] = sp.fit_transform_oof(Xtr, ytr, None, None), sp.transform(Xho, None)
elif a.arm == 'fc':
    # Forecasting family alone (tabularaml/generate/forecast.py) on top of the raw columns.
    from tabularaml.generate.forecast import forecast_features
    Ftr, Fho, ff = forecast_features(Xtr, ytr, Xho, **json.loads(a.kw))
    Xtr, Xho = pd.concat([raw(Xtr), Ftr], axis=1), pd.concat([raw(Xho), Fho], axis=1)
    info = dict(n_fc=Ftr.shape[1])
elif a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    kw = dict(time_col='date'); kw.update(json.loads(a.kw))
    f = FeatureForge(task='regression', time_budget=a.budget, random_state=0, n_jobs=4, verbose=True, **kw).fit(Xtr, ytr, X_unlabeled=U)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, feats=f.new_columns_[:80])
    Xtr, Xho = raw(f.transform_train(Xtr)), raw(f.transform(Xho))
fe_t = time.time() - t0
P = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
d = Xtr['date'].to_numpy(); cut = d.max() - 16
i_tr, i_va = np.flatnonzero(d <= cut), np.flatnonzero(d > cut)
b = lgb.train(P, lgb.Dataset(Xtr.iloc[i_tr], ytr[i_tr]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[i_va], ytr[i_va])], callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.clip(lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho), 0, None)
w = np.where(Xho['perishable'].to_numpy() == 1, 1.25, 1.0)
nwrmsle = float(np.sqrt(np.sum(w * (p - yho) ** 2) / np.sum(w)))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, nwrmsle=nwrmsle,
           nwrmsle_const=float(np.sqrt(np.sum(w * (np.mean(ytr) - yho) ** 2) / np.sum(w))), best_it=b.best_iteration, fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
