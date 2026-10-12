"""Expedia Hotel Recommendations (Kaggle 2016): which of 100 hotel clusters a user books. MAP@5.
The contest's train file holds every search and booking of 2013-2014 (user_id, second-level date_time); its test file
holds only the 2015 bookings of the same users, without is_booking, cnt and hotel_cluster.

Built the same way here, on 3% of users (``expedia_prep``: the contest train.csv from the public copy
``monishc/expediatrainingset``): training rows = every event before ``--cut`` (default 2014-07-01); held out = the
bookings from the cut to the end of 2014 of users with an earlier event, is_booking / cnt dropped.
Arms (same LightGBM, multiclass over 100 clusters: early stopping on the latest 15% of training rows, then one refit
on all at 1.1x the best iteration; scored on the held-out bookings):
  ``raw``   the event's own columns (dates as days since 1970);
  ``hand``  raw + the winners' label-free staples: stay length, days from search to check-in, check-in month and
            weekday, the search's hour and weekday;
  ``ff``    blind: ``scripts/contest_features.py`` at its defaults (``--repo``).
``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil, resource
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--data', default='/home/user/data/expedia/events_u3.parquet')
ap.add_argument('--cut', default='2014-07-01'); ap.add_argument('--end', default='2015-01-01'); ap.add_argument('--log', default='expedia.jsonl')
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1])); ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--shuffle-seed', type=int, default=0); ap.add_argument('--tag', default='')
ap.add_argument('--budget', default=None, help="contest_features --budget (work-clock seconds; default: its own)")
a = ap.parse_args()
d = pd.read_parquet(a.data)
d['date_time'] = pd.to_datetime(d.date_time)
d = d.sort_values('date_time', kind='stable').reset_index(drop=True)
tr = d[d.date_time < a.cut].reset_index(drop=True)
te = d[(d.date_time >= a.cut) & (d.date_time < a.end) & (d.is_booking == 1) & d.user_id.isin(set(tr.user_id))].reset_index(drop=True)
ytr, yte = tr.pop('hotel_cluster').to_numpy(), te.pop('hotel_cluster').to_numpy()
tr, te = tr.drop(columns=['is_booking', 'cnt']), te.drop(columns=['is_booking', 'cnt'])
if a.shuffle:
    ytr = np.random.default_rng(a.shuffle_seed).permutation(ytr)
print(f'train {len(tr)} events, held out {len(te)} bookings, {tr.user_id.nunique()} users', flush=True)
t0 = time.time(); info = {}


def days(s):
    return pd.to_datetime(s, errors='coerce').to_numpy('datetime64[s]').astype('int64') / 86400.0


def raw(X):
    X = X.copy()
    for c in ('date_time', 'srch_ci', 'srch_co'):
        X[c] = days(X[c])
        X.loc[~np.isfinite(X[c]) | (X[c] < 0), c] = np.nan
    return X


if a.arm in ('raw', 'hand'):
    Xtr, Xte = raw(tr), raw(te)
    if a.arm == 'hand':
        for X, S in ((Xtr, tr), (Xte, te)):
            X['stay'] = X.srch_co - X.srch_ci
            X['ahead'] = X.srch_ci - X.date_time
            ci = pd.to_datetime(S.srch_ci, errors='coerce')
            X['ci_month'], X['ci_dow'] = ci.dt.month, ci.dt.dayofweek
            X['hour'], X['dow'] = S.date_time.dt.hour, S.date_time.dt.dayofweek
else:
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    try:
        tr.assign(hotel_cluster=ytr).to_parquet(tmp / 'train.parquet'); te.to_parquet(tmp / 'test.parquet')
        subprocess.run([sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
                        '--test', str(tmp / 'test.parquet'), '--target', 'hotel_cluster', '--task', 'multiclass',
                        '--out-dir', str(tmp / 'out')] + (['--budget', a.budget] if a.budget else []), check=True, cwd=a.repo)
        info['peak_mb'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
        Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['hotel_cluster'])
        Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    info['commit'] = subprocess.run(['git', '-C', a.repo, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = days(X[c])
        elif not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype('category')
P = dict(objective='multiclass', num_class=100, learning_rate=0.1, num_leaves=31, min_child_samples=100,
         feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1, seed=0)
n = len(Xtr); fit, va = np.arange(int(0.85 * n)), np.arange(int(0.85 * n), n)
b = lgb.train(P, lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 2000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte)
top = np.argsort(-p, axis=1)[:, :5]
hit = top == yte[:, None]
map5 = float((hit / np.arange(1, 6)).sum(1).mean())
res = dict(budget=a.budget, arm=a.arm + a.tag + (f'_shuffled{a.shuffle_seed}' if a.shuffle else ''), map5=map5, n_tr=n, n_te=len(Xte),
           n_cols=Xtr.shape[1], best_it=b.best_iteration, fe_s=round(fe_t), total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res)); open(a.log, 'a').write(json.dumps(res) + '\n')
