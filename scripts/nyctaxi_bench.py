"""New York City Taxi Trip Duration (Kaggle 2017): duration of a yellow-cab trip from its pickup time, pickup and
dropoff coordinates, vendor and passenger count. RMSLE.

Data: a public Kaggle copy of the contest's train.csv (``yasserh/nyc-taxi-trip-duration``: 1,458,644 trips,
January-June 2016). The contest's test file was a random sample of trips from the same months with
``dropoff_datetime`` removed, so the holdout is a random 20% of the trips (``--seed``) and ``dropoff_datetime`` is
dropped from every arm; held-out durations reach only the scorer. Columns are typed the way Kaggle's CSV reads
(times as strings, the store-and-forward flag as text).
Arms (same bagged LightGBM on log1p(duration): early stopping on 15% of the training rows, then 3 seeds):
  ``raw``    the file's columns, the pickup time as seconds;
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults (``--log-target``, the metric is RMSLE) on the file;
  ``hand``   raw + the features of the winners' write-ups and top public kernels that need no outside data (no OSRM
             routes, no weather): haversine / Manhattan distance and bearing, midpoint, PCA-rotated coordinates,
             k-means coordinate clusters (train + test points), pickup time parts, trip counts per hour and per
             cluster-hour (label-free), and out-of-fold average log duration and speed per hour, pickup / dropoff
             cluster, cluster pair and coordinate cell (5 folds over the training rows);
  ``ff_hand`` both.
``--groups`` keeps only some hand groups (dist, pca, clust, time, cnt, speed); ``--shuffle`` permutes the training
labels (leakage control). Reports runtime and peak memory (this process and the pipeline).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil, resource, os
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default=''); ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='/home/user/data/nyctaxi/'); ap.add_argument('--log', default='nyctaxi.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--groups', default='')
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]), help='checkout whose contest_features.py runs')
ap.add_argument('--ff-cache', action='store_true', help='reuse the pipeline output of an earlier run with the same seed and tag')
a = ap.parse_args()
D = Path(a.data)
df = pd.read_csv(D / 'NYC.csv').drop(columns=['dropoff_datetime'])
df['y'] = df.pop('trip_duration').astype(float)
rng = np.random.default_rng(a.seed)
is_ho = np.zeros(len(df), bool); is_ho[rng.choice(len(df), len(df) // 5, replace=False)] = True
tr, te = df[~is_ho].reset_index(drop=True), df[is_ho].reset_index(drop=True)
ytr, yte = np.log1p(tr.pop('y').to_numpy()), np.log1p(te.pop('y').to_numpy())
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
T0 = pd.Timestamp('2016-01-01')


def raw(X):
    X = X.drop(columns=['id']).copy()
    X['pickup_datetime'] = (pd.to_datetime(X['pickup_datetime']) - T0).dt.total_seconds()
    X['store_and_fwd_flag'] = (X['store_and_fwd_flag'] == 'Y').astype(float)
    return X


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def hand(Xtr, ytr, Xte):
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    A = pd.concat([Xtr, Xte], ignore_index=True); n = len(Xtr); H = {}
    la1, lo1, la2, lo2 = (A[c].to_numpy(float) for c in ('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'))
    g = {}
    H['dist_hav'] = hav(la1, lo1, la2, lo2)
    H['dist_manh'] = hav(la1, lo1, la1, lo2) + hav(la1, lo1, la2, lo1)
    y_, x_ = np.radians(lo2 - lo1), None
    H['bearing'] = np.degrees(np.arctan2(np.sin(y_) * np.cos(np.radians(la2)),
                              np.cos(np.radians(la1)) * np.sin(np.radians(la2)) - np.sin(np.radians(la1)) * np.cos(np.radians(la2)) * np.cos(y_)))
    H['center_lat'], H['center_lon'] = (la1 + la2) / 2, (lo1 + lo2) / 2
    g['dist'] = list(H)
    P = np.vstack([np.c_[la1, lo1], np.c_[la2, lo2]])
    pca = PCA().fit(P)
    p1, p2 = pca.transform(np.c_[la1, lo1]), pca.transform(np.c_[la2, lo2])
    H['pickup_pca0'], H['pickup_pca1'], H['dropoff_pca0'], H['dropoff_pca1'] = p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]
    H['pca_manh'] = np.abs(p2[:, 0] - p1[:, 0]) + np.abs(p2[:, 1] - p1[:, 1])
    g['pca'] = [c for c in H if 'pca' in c]
    km = MiniBatchKMeans(100, batch_size=10000, random_state=0, n_init=3).fit(P[np.random.default_rng(0).choice(len(P), 500_000, replace=False)])
    cp, cd = km.predict(np.c_[la1, lo1]), km.predict(np.c_[la2, lo2])
    H['pickup_cluster'], H['dropoff_cluster'] = cp.astype(float), cd.astype(float)
    g['clust'] = ['pickup_cluster', 'dropoff_cluster']
    t = pd.to_datetime(A.pickup_datetime)
    H['hour'], H['weekday'], H['dayofyear'], H['week'] = t.dt.hour, t.dt.weekday, t.dt.dayofyear, t.dt.isocalendar().week.astype(int)
    H['week_hour'] = t.dt.weekday * 24 + t.dt.hour
    H['minute_of_day'] = t.dt.hour * 60 + t.dt.minute
    g['time'] = ['hour', 'weekday', 'dayofyear', 'week', 'week_hour', 'minute_of_day']
    hb = t.dt.floor('h')
    H['cnt_hour'] = hb.map(hb.value_counts()).to_numpy(float)
    k = hb.astype(str) + '_' + pd.Series(cp).astype(str)
    H['cnt_pickup_cluster_hour'] = k.map(k.value_counts()).to_numpy(float)
    k = hb.astype(str) + '_' + pd.Series(cd).astype(str)
    H['cnt_dropoff_cluster_hour'] = k.map(k.value_counts()).to_numpy(float)
    g['cnt'] = ['cnt_hour', 'cnt_pickup_cluster_hour', 'cnt_dropoff_cluster_hour']
    # Out-of-fold averages of the log duration and the speed (5 folds over the training rows; test rows use all).
    speed = H['dist_hav'][:n] / np.expm1(ytr).clip(1) * 3600
    keys = {'hour': H['hour'].to_numpy(), 'pclust': cp, 'dclust': cd, 'pair': cp * 100 + cd,
            'cell': np.round(H['center_lat'], 2) * 1e4 + np.round(H['center_lon'], 2),
            'clust_hour': cp * 100 + H['hour'].to_numpy()}
    fold = np.random.default_rng(1).permutation(n) % 5
    g['speed'] = []
    for name, key in keys.items():
        for tgt_name, tgt in (('logdur', ytr), ('speed', speed)):
            out = np.full(len(A), np.nan)
            prior = tgt.mean()
            for f in range(6):
                m = fold != f if f < 5 else np.ones(n, bool)
                s = pd.DataFrame({'k': key[:n][m], 'v': tgt[m]}).groupby('k').v.agg(['sum', 'count'])
                enc = (s['sum'] + 20 * prior) / (s['count'] + 20)
                idx = np.flatnonzero(fold == f) if f < 5 else np.arange(n, len(A))
                out[idx] = pd.Series(key[idx]).map(enc).to_numpy()
            H[f'avg_{tgt_name}_{name}'] = out; g['speed'].append(f'avg_{tgt_name}_{name}')
    H = pd.DataFrame(H)
    if a.groups:
        H = H[[c for gg in a.groups.split(',') for c in g[gg]]]
    return H.iloc[:n].reset_index(drop=True), H.iloc[n:].reset_index(drop=True)


print('train', len(tr), 'held out', len(te), flush=True)
t0 = time.time(); info = {}
Xtr, Xte = raw(tr), raw(te)
cache = D / f'ff_cache_{a.seed}{a.tag}{"_shuf" if a.shuffle else ""}'
if a.arm.startswith('ff') and a.ff_cache and (cache / 'tr.parquet').exists():
    Ftr, Fte = pd.read_parquet(cache / 'tr.parquet'), pd.read_parquet(cache / 'te.parquet')
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1); info['n_new'] = Ftr.shape[1]
elif a.arm.startswith('ff'):
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    tr.assign(trip_duration=np.expm1(ytr)).to_parquet(tmp / 'train.parquet'); te.to_parquet(tmp / 'test.parquet')
    cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
           '--test', str(tmp / 'test.parquet'), '--target', 'trip_duration', '--id', 'id', '--task', 'regression', '--log-target',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True, cwd=a.repo)
    info['pipe_peak_gb'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20, 2)
    Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert (Ftr['id'].astype(str).to_numpy() == tr['id'].to_numpy()).all() and (Fte['id'].astype(str).to_numpy() == te['id'].to_numpy()).all()
    new = [c for c in Ftr.columns if c not in tr.columns and c != 'trip_duration']
    Ftr, Fte = Ftr[new].reset_index(drop=True), Fte[new].reset_index(drop=True)
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1)
    info['n_new'] = len(new)
    cache.mkdir(exist_ok=True); Ftr.to_parquet(cache / 'tr.parquet'); Fte.to_parquet(cache / 'te.parquet')
    shutil.rmtree(tmp, ignore_errors=True)
if a.arm.endswith('hand'):
    Htr, Hte = hand(tr, ytr, te)
    Xtr, Xte = pd.concat([Xtr, Htr], axis=1), pd.concat([Xte, Hte], axis=1)
fe_t = time.time() - t0
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='regression', learning_rate=0.1, num_leaves=255, min_child_samples=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1, max_bin=255)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); k = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:k]), np.sort(perm[k:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 3000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, rmsle=float(np.sqrt(np.mean((p - yte) ** 2))),
           rmsle_const=float(np.sqrt(np.mean((ytr.mean() - yte) ** 2))), best_it=b.best_iteration, n_cols=Xtr.shape[1],
           fe_s=round(fe_t), total_s=round(time.time() - t0), peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2),
           n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
