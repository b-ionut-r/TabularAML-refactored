"""Two Sigma Connect: Rental Listing Inquiries (Kaggle 2017, $25k): interest level (low / medium / high) of a
RenHop listing in New York.

Data: a public Kaggle copy of the contest's train.json (``logan1997/two-sigma-challenge``: 49,352 listings,
April-June 2016). The contest's test file was a random sample of listings from the same months (same managers and
buildings), so the holdout is a random 20% of the listings (``--seed``); its labels reach only the scorer.
The pipeline gets the list columns (``features``, ``photos``) as lists, as the file holds them; the other arms
read them joined into strings (``features`` with " ; ", ``photos`` the URLs with " ").
Metric: multiclass log loss. Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds):
  ``raw``   the file's columns: numbers, ids and addresses as categories, ``created`` as a timestamp (text dropped);
  ``ff``    blind: ``scripts/contest_features.py`` at its defaults on the file as given;
  ``hand``  raw + the features of the winners' write-ups and top public kernels: photo / feature / description-word
            counts, created hour / weekday / day, price per bedroom / bathroom / room, manager and building listing
            counts (train + test rows, label-free) and out-of-fold manager / building interest rates, the 100 most
            common amenity tokens, rounded location counts, distance to the centre (the listing id is already a raw column);
  ``ff_hand`` both.
``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import log_loss
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default=''); ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='data/twosigma/'); ap.add_argument('--log', default='twosigma.jsonl'); ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--groups', default='', help='hand arms: only these hand groups (comma list of cnt,time,price,ids,geo,feat,rate)')
ap.add_argument('--ff-drop', default='', help='regex of pipeline columns to leave out (diagnostics)')
ap.add_argument('--ff-cache', action='store_true', help='reuse the pipeline output of an earlier run with the same seed')
a = ap.parse_args()
D = Path(a.data)
df = pd.read_json(D / 'train.json').reset_index(drop=True)
LISTS = df[['features', 'photos']].copy()   # the pipeline gets them as the contest file holds them (lists)
df['features'] = df['features'].map(lambda v: ' ; '.join(v))
df['photos'] = df['photos'].map(lambda v: ' '.join(v))
df['y'] = df.interest_level.map({'low': 0, 'medium': 1, 'high': 2}); df = df.drop(columns=['interest_level'])
rng = np.random.default_rng(a.seed)
is_ho = np.zeros(len(df), bool); is_ho[rng.choice(len(df), len(df) // 5, replace=False)] = True
tr, te = df[~is_ho].reset_index(drop=True), df[is_ho].reset_index(drop=True)
ytr, yte = tr.pop('y').to_numpy(), te.pop('y').to_numpy()
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
TEXT = ['description', 'features', 'photos']
IDS = ['building_id', 'manager_id', 'display_address', 'street_address']


def raw(X):
    X = X.drop(columns=TEXT).copy()
    X['created'] = (pd.to_datetime(X['created']) - pd.Timestamp('2016-01-01')).dt.total_seconds() / 86400
    return X


def hand(Xtr, ytr, Xte):
    """Winners' / top kernels' features, label-free ones over train + test rows, target rates out of fold."""
    A = pd.concat([Xtr, Xte], ignore_index=True); n = len(Xtr); H = pd.DataFrame(index=A.index)
    H['n_photos'] = A.photos.str.split().str.len().fillna(0)
    H['n_features'] = A.features.map(lambda s: 0 if not s else len(s.split(' ; ')))
    H['n_desc_words'] = A.description.str.split().str.len().fillna(0)
    H['desc_len'] = A.description.str.len()
    c = pd.to_datetime(A.created)
    H['hour'], H['weekday'], H['day'], H['month'] = c.dt.hour, c.dt.weekday, c.dt.day, c.dt.month
    H['price_per_bed'] = A.price / A.bedrooms.replace(0, np.nan)
    H['price_per_bath'] = A.price / A.bathrooms.replace(0, np.nan)
    H['price_per_room'] = A.price / (A.bedrooms + A.bathrooms).replace(0, np.nan)
    H['building_zero'] = (A.building_id == '0').astype(float)
    for k in ('manager_id', 'building_id', 'display_address'):
        H[f'{k}_count'] = A[k].map(A[k].value_counts())
    loc = A.latitude.round(3).astype(str) + '_' + A.longitude.round(3).astype(str)
    H['loc_count'] = loc.map(loc.value_counts())
    H['dist_centre'] = np.hypot(A.latitude - 40.7527, A.longitude + 73.9772)
    tok = A.features.str.lower().str.split(' ; ').explode()
    for t in tok[tok.str.len() > 0].value_counts().index[:100]:
        H['feat_' + ''.join(ch if ch.isalnum() else '_' for ch in t)[:40]] = A.features.str.lower().str.split(' ; ').map(lambda v, t=t: float(t in v))
    # Out-of-fold interest rates of the manager and the building (5 folds over the training rows).
    Y = np.eye(3)[ytr]
    fold = np.random.default_rng(1).permutation(n) % 5
    for k in ('manager_id', 'building_id'):
        out = np.full((len(A), 3), np.nan)
        for f in range(5):
            m = fold != f
            g = pd.DataFrame(Y[m], index=Xtr[k].to_numpy()[m]).groupby(level=0).agg(['sum', 'count'])
            s, cnt = g.xs('sum', axis=1, level=1), g.xs('count', axis=1, level=1)
            rate = ((s + 3 * Y.mean(0)) / (cnt + 3)).reindex(Xtr[k].to_numpy()[fold == f]).to_numpy()
            out[np.flatnonzero(fold == f)] = rate
        g = pd.DataFrame(Y, index=Xtr[k].to_numpy()).groupby(level=0).agg(['sum', 'count'])
        s, cnt = g.xs('sum', axis=1, level=1), g.xs('count', axis=1, level=1)
        out[n:] = ((s + 3 * Y.mean(0)) / (cnt + 3)).reindex(Xte[k].to_numpy()).to_numpy()
        for j, lv in enumerate(('low', 'medium', 'high')):
            H[f'{k}_rate_{lv}'] = out[:, j]
    H = H.loc[:, ~H.columns.duplicated()]
    if a.groups:
        grp = {'cnt': ['n_photos', 'n_features', 'n_desc_words', 'desc_len'], 'time': ['hour', 'weekday', 'day', 'month'],
               'price': ['price_per_bed', 'price_per_bath', 'price_per_room'],
               'ids': ['building_zero', 'manager_id_count', 'building_id_count', 'display_address_count', 'loc_count'],
               'geo': ['dist_centre'], 'feat': [c for c in H.columns if c.startswith('feat_')],
               'rate': [c for c in H.columns if '_rate_' in c]}
        H = H[[c for g in a.groups.split(',') for c in grp[g]]]
    return H.iloc[:n].reset_index(drop=True), H.iloc[n:].reset_index(drop=True)


print('train', len(tr), 'held out', len(te), flush=True)
t0 = time.time(); info = {}
Xtr, Xte = raw(tr), raw(te)
cache = D / f'ff_cache_{a.seed}{"_shuf" if a.shuffle else ""}'
if a.arm.startswith('ff') and a.ff_cache and (cache / 'tr.parquet').exists():
    Ftr, Fte = pd.read_parquet(cache / 'tr.parquet'), pd.read_parquet(cache / 'te.parquet')
    if a.ff_drop:
        import re
        keep = [c for c in Ftr.columns if not re.match(a.ff_drop, c)]; Ftr, Fte = Ftr[keep], Fte[keep]
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1); info['n_new'] = Ftr.shape[1]
elif a.arm.startswith('ff'):
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    Ltr, Lte = LISTS[~is_ho].reset_index(drop=True), LISTS[is_ho].reset_index(drop=True)
    tr.assign(y=ytr, features=Ltr.features, photos=Ltr.photos).to_parquet(tmp / 'train.parquet')
    te.assign(features=Lte.features, photos=Lte.photos).to_parquet(tmp / 'test.parquet')
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
           '--test', str(tmp / 'test.parquet'), '--target', 'y', '--id', 'listing_id', '--task', 'multiclass',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True)
    Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert (Ftr.listing_id.to_numpy() == tr.listing_id.to_numpy()).all() and (Fte.listing_id.to_numpy() == te.listing_id.to_numpy()).all()
    new = [c for c in Ftr.columns if c not in tr.columns and c != 'y']
    Xtr, Xte = pd.concat([Xtr, Ftr[new]], axis=1), pd.concat([Xte, Fte[new]], axis=1)
    info['n_new'] = len(new)
    cache.mkdir(exist_ok=True); Ftr[new].to_parquet(cache / 'tr.parquet'); Fte[new].reset_index(drop=True).to_parquet(cache / 'te.parquet')
    shutil.rmtree(tmp, ignore_errors=True)
if a.arm.endswith('hand'):
    Htr, Hte = hand(tr, ytr, te)
    Xtr, Xte = pd.concat([Xtr, Htr], axis=1), pd.concat([Xte, Hte], axis=1)
fe_t = time.time() - t0
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='multiclass', num_class=3, learning_rate=0.03, num_leaves=31, min_child_samples=30, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1, max_cat_to_onehot=8, cat_smooth=20)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); k = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:k]), np.sort(perm[k:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros((len(Xte), 3))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
prior = np.bincount(ytr, minlength=3) / len(ytr)
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, logloss=float(log_loss(yte, p, labels=[0, 1, 2])),
           logloss_prior=float(log_loss(yte, np.tile(prior, (len(yte), 1)), labels=[0, 1, 2])), best_it=b.best_iteration,
           n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
