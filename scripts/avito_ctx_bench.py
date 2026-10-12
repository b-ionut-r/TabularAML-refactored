"""Avito Context Ad Clicks (Kaggle 2015, $20k): will a user click a contextual ad shown on a search results page.
Log loss, as the contest.

Data: RelBench's copy of the contest database (``rel-avito``: SearchStream, SearchInfo, UserInfo, AdsInfo; 2015-04-25
to 2015-05-20; Kaggle's own download needs the rules accepted). The contest trained on these weeks and scored the
next days' searches of the same users, so a window holds out the last ``--days`` days before a cut and trains on all
earlier rows: ``--win`` 0 holds out 2015-05-15..05-20, 1 holds out 2015-05-09..05-14. Rows: contextual ads
(ObjectType 3) with a label; ``--sample`` of the users (by a hash of UserID, ``--seed``) keeps it in memory. Each row
carries its search (time, user, IP, logged on, query, location, category), the user's device record and the ad's
category, location and price, all known when the page is shown. Held-out labels reach only the scorer.

Arms (same bagged LightGBM: early stopping on the latest 15% of training rows, then 3 seeds on all of them):
  ``raw``  the columns above (search time as seconds);
  ``ff``   blind: ``scripts/contest_features.py`` at its defaults (``--extra`` passes flags, e.g. event gaps off);
  ``hand`` raw + the winners' features: the user's searches in all rows (train and test, no labels), that day and
           that hour, seconds since the user's previous search and until the next, the user's rank among their
           searches, the ad's and the user-ad pair's rows, ads on the page;
  ``ff_hand`` both. ``--shuffle`` permutes the training labels (leakage control). Reports runtime and peak memory.
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil, resource, zlib
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--sample', type=float, default=0.25); ap.add_argument('--tag', default='')
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='/home/user/data/ravito/dl/db/'); ap.add_argument('--log', default='avito.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
a = ap.parse_args()
D = Path(a.data)
t0 = time.time()
cache = D.parent / f'win{a.win}_s{a.seed}_{a.sample}.parquet'
if not cache.exists():
    ss = pd.read_parquet(D / 'SearchStream.parquet', columns=['SearchID', 'AdID', 'Position', 'ObjectType', 'HistCTR', 'IsClick'])
    ss = ss[(ss.ObjectType == 3) & ss.IsClick.notna()].drop(columns='ObjectType')
    si = pd.read_parquet(D / 'SearchInfo.parquet')
    si = si[si.UserID.notna()]
    keep = np.array([zlib.crc32(f'{a.seed}:{u}'.encode()) % 10_000 < a.sample * 10_000 for u in si.UserID.astype(int)])
    si = si[keep]
    df = ss.merge(si, on='SearchID', how='inner')
    ads = pd.read_parquet(D / 'AdsInfo.parquet', columns=['AdID', 'LocationID', 'CategoryID', 'Price'])
    ads = ads[ads.AdID.isin(set(df.AdID))].rename(columns={'LocationID': 'AdLocationID', 'CategoryID': 'AdCategoryID', 'Price': 'AdPrice'})
    df = df.merge(ads, on='AdID', how='left').merge(pd.read_parquet(D / 'UserInfo.parquet'), on='UserID', how='left')
    df = df.sort_values(['SearchDate', 'SearchID', 'Position']).reset_index(drop=True)
    for c in df.columns:
        if str(df[c].dtype) in ('Int64', 'Float64'):
            df[c] = df[c].astype(float)
    df.to_parquet(cache)
df = pd.read_parquet(cache)
cut, end = [('2015-05-15', '2015-05-21'), ('2015-05-09', '2015-05-15')][a.win]
tr = df[df.SearchDate < cut].reset_index(drop=True)
te = df[(df.SearchDate >= cut) & (df.SearchDate < end)].reset_index(drop=True)
ytr, yte = tr.pop('IsClick').to_numpy().astype(int), te.pop('IsClick').to_numpy().astype(int)
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
print(f'train {len(tr)} rows to {tr.SearchDate.max()} (clicks {ytr.mean():.4f}), held out {len(te)} ({cut}..{end}, clicks {yte.mean():.4f}), '
      f'users {tr.UserID.nunique()} / {te.UserID.nunique()} ({te.UserID.isin(set(tr.UserID)).mean():.2f} of test rows seen)', flush=True)


def raw(X):
    X = X.drop(columns=['SearchID']).copy()
    X['SearchDate'] = (X.SearchDate - pd.Timestamp('2015-04-25')).dt.total_seconds()
    return X


def hand():
    both = pd.concat([tr, te], ignore_index=True)
    t = (both.SearchDate - pd.Timestamp('2015-04-25')).dt.total_seconds().to_numpy()
    s = both.drop_duplicates('SearchID')[['SearchID', 'UserID', 'SearchDate']].copy()
    s['t'] = (s.SearchDate - pd.Timestamp('2015-04-25')).dt.total_seconds()
    s = s.sort_values(['UserID', 't', 'SearchID'])
    g = s.groupby('UserID').t
    s['prev_gap'] = s.t - g.shift(1); s['next_gap'] = g.shift(-1) - s.t; s['user_rank'] = s.groupby('UserID').cumcount()
    H = pd.DataFrame(index=both.index)
    m = both[['SearchID']].merge(s[['SearchID', 'prev_gap', 'next_gap', 'user_rank']], on='SearchID', how='left')
    for c in ('prev_gap', 'next_gap', 'user_rank'):
        H[c] = m[c].to_numpy()
    H['user_searches'] = both.UserID.map(s.UserID.value_counts())
    day = both.SearchDate.dt.floor('D'); hour = both.SearchDate.dt.floor('h')
    sd = s.assign(day=s.SearchDate.dt.floor('D'), hour=s.SearchDate.dt.floor('h'))
    H['user_day_searches'] = pd.MultiIndex.from_arrays([both.UserID, day]).map(sd.groupby(['UserID', 'day']).size()).to_numpy()
    H['user_hour_searches'] = pd.MultiIndex.from_arrays([both.UserID, hour]).map(sd.groupby(['UserID', 'hour']).size()).to_numpy()
    H['ad_rows'] = both.AdID.map(both.AdID.value_counts())
    H['user_ad_rows'] = both.groupby(['UserID', 'AdID']).AdID.transform('size')
    H['page_ads'] = both.groupby('SearchID').AdID.transform('size')
    H = H.astype(float)
    return H.iloc[:len(tr)].reset_index(drop=True), H.iloc[len(tr):].reset_index(drop=True)


info = {}
Xtr, Xte = raw(tr), raw(te)
if a.arm.startswith('ff'):
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    tr.drop(columns=['SearchID']).assign(IsClick=ytr).to_parquet(tmp / 'train.parquet'); te.drop(columns=['SearchID']).to_parquet(tmp / 'test.parquet')
    cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
           '--test', str(tmp / 'test.parquet'), '--target', 'IsClick', '--task', 'binary', '--budget', str(a.budget),
           '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True, cwd=a.repo)
    info['pipe_peak_gb'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20, 2)
    Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert len(Ftr) == len(tr) and len(Fte) == len(te)
    new = [c for c in Ftr.columns if c not in tr.columns and c != 'IsClick']
    Xtr = pd.concat([Xtr, Ftr[new].reset_index(drop=True)], axis=1); Xte = pd.concat([Xte, Fte[new].reset_index(drop=True)], axis=1)
    info['n_new'] = len(new); info['new'] = new[:40]
    shutil.rmtree(tmp, ignore_errors=True)
if a.arm.endswith('hand'):
    Htr, Hte = hand()
    Xtr = pd.concat([Xtr, Htr], axis=1); Xte = pd.concat([Xte, Hte], axis=1)
fe_t = time.time() - t0
Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]; Xte = Xte[Xtr.columns]
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=200, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, cat_smooth=50, num_threads=4, verbose=-1)
order = np.argsort(tr.SearchDate.to_numpy(), kind='stable'); cut_i = int(len(order) * 0.85)
fit, va = np.sort(order[:cut_i]), np.sort(order[cut_i:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, seed=a.seed, logloss=float(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))),
           auc=float(roc_auc_score(yte, p)), logloss_const=float(log_loss(yte, np.full(len(yte), ytr.mean()))), best_it=b.best_iteration,
           n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0),
           peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2), n_tr=len(Xtr), n_te=len(Xte), **info)
np.save(f'pred_{res["arm"]}_w{a.win}_s{a.seed}.npy', p); np.save(f'y_w{a.win}_s{a.seed}.npy', yte); np.save(f'u_w{a.win}_s{a.seed}.npy', te.UserID.to_numpy())
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
