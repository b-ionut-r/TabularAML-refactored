"""GoDaddy Microbusiness Density Forecasting (Kaggle 2023, $60k): monthly density per US county.

Data: the contest's train and revealed-test months (2019-08 .. 2022-12, 3,135 counties) with its census
columns joined by year, as one public file (Kaggle dataset ``jjithin/business-density``,
``microbusiness_density.csv``). The contest forecast the next months of every county from these.

Holdout built that way: training rows are every county-month up to a cut, the unlabeled rows every county for
the next ``--months`` months (``--win 0`` ends on 2022-12; ``--win 1`` ends ``--months`` months earlier).
Columns: county id, county, state, month, and the census columns (broadband, college, foreign born, IT
workers, median income) of that year; ``active`` (the count behind the target) is not in the test file and is
left out. Metric: SMAPE of the density. Judge: one LightGBM with an L1 objective on log1p(density),
early-stopped on the latest ``--months`` training months. Arms: ``raw``, ``fc`` (+ forecast.py at its
defaults). ``--shuffle`` permutes the training labels. ``last`` is the last known value, for reference.
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--months', type=int, default=6); ap.add_argument('--tag', default=''); ap.add_argument('--kw', default='{}')
ap.add_argument('--data', default='data/godaddy/microbusiness_density.csv')
ap.add_argument('--log', default='godaddy.jsonl'); ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--base', action='store_true', help='judge starts from the latest known value (fc arm): init_score')
a = ap.parse_args()
D = pd.read_csv(a.data).drop(columns=['active', 'year', 'month'])
months = np.sort(D.first_day_of_month.unique())
last = len(months) - 1 - a.months * a.win
cut = months[last - a.months]
ho_m = months[last - a.months + 1: last + 1]
tr = D[D.first_day_of_month <= cut].reset_index(drop=True)
ho = D[D.first_day_of_month.isin(ho_m)].reset_index(drop=True)
ytr = np.log1p(tr.pop('microbusiness_density').to_numpy())
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
yho = ho.pop('microbusiness_density').to_numpy()
print('train', len(tr), 'holdout', len(ho), 'cut', cut, 'months', ho_m[0], '..', ho_m[-1], flush=True)
t0 = time.time(); info = {}
Xtr, Xho = tr.copy(), ho.copy()
if a.arm == 'fc':
    from tabularaml.generate.forecast import forecast_features
    Ftr, Fho, ff = forecast_features(Xtr, ytr, Xho, **json.loads(a.kw))
    Xtr, Xho = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xho, Fho], axis=1)
    info = dict(n_fc=Ftr.shape[1], on=ff.active_, why=getattr(ff, 'reason_', ''))
fe_t = time.time() - t0
for X in (Xtr, Xho):
    d = pd.to_datetime(X['first_day_of_month'])
    X['first_day_of_month'] = (d.dt.year * 12 + d.dt.month).astype(float)
    X['month'] = d.dt.month.astype(float)
    for c in ('county', 'state'):
        X[c] = X[c].astype('category')
cats = {c: Xtr[c].cat.categories for c in ('county', 'state')}
for c in cats:
    Xho[c] = pd.Categorical(Xho[c].astype(str), categories=cats[c])
P = dict(objective='l1', learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=0)
mm = Xtr['first_day_of_month'].to_numpy(); va = mm > mm.max() - a.months
btr = bho = None
if a.base:   # the latest value known at each row's forecast origin (as the family reads it), as the starting score
    inv = (lambda v: np.expm1(v)) if ff.log_ else (lambda v: v)
    btr, bho = (np.nan_to_num(inv(X['fc_mean1'].to_numpy(float)), nan=float(np.median(ytr))) for X in (Xtr, Xho))
ds = lambda m: lgb.Dataset(Xtr[m], ytr[m], init_score=None if btr is None else btr[m])
b = lgb.train(P, ds(~va), 5000, valid_sets=[ds(va)], callbacks=[lgb.early_stopping(100, verbose=False)])
p = lgb.train(P, ds(np.ones(len(Xtr), bool)), int(b.best_iteration * 1.1) + 1).predict(Xho)
p = np.expm1(p + (0 if bho is None else bho)).clip(0)
def smape(y, p):
    den = np.abs(y) + np.abs(p)
    return float(200 * np.mean(np.where(den > 0, np.abs(y - p) / np.where(den > 0, den, 1), 0.0)))
lastv = tr.assign(y=np.expm1(ytr)).groupby('cfips').y.last()
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, smape=smape(yho, p),
           smape_last=smape(yho, ho.cfips.map(lastv).to_numpy()), smape_const=smape(yho, np.full(len(yho), np.expm1(np.median(ytr)))),
           best_it=b.best_iteration, fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_ho=len(Xho), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
