"""H&M Personalized Fashion Recommendations (Kaggle 2022, $50k): which articles a customer buys in the next week.
Here the repeat part, as a binary task: rows = (customer, article) pairs the customer bought before the label week;
label = bought again in that week.

Data: a public Kaggle copy of the contest files (``derrickmwiti/hm-personalized-fashion-recommendations``:
transactions_train.parquet, 31,788,324 rows = the contest's; articles.parquet, 105,542 articles). The contest's own
download needs the rules accepted.

Window 0 labels the last week of the data (2020-09-16 to 22), window 1 the week before; the history is every
earlier transaction of ``--customers`` random customers who bought before the label week (``--seed``). Held out:
20% of those customers, whole (their history is known, as the contest's customers' was; their label week reaches
only the scorer). Metric: AUC.
Arms (same bagged LightGBM: early stopping on 15% of the training customers, then 3 seeds):
  ``raw``  the pair's own columns (customer, article and its product / type / department / garment / colour codes);
  ``ff``   blind: ``scripts/contest_features.py`` at its defaults, the history as an event log
           (``log=...:customer_id:week``), so its as-of aggregation switches on.
``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--customers', type=int, default=30000); ap.add_argument('--window', type=int, default=0)
ap.add_argument('--data', default='data/hm/'); ap.add_argument('--log', default='hm.jsonl'); ap.add_argument('--tag', default='')
ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--keep', default='all', help="diagnosis: 'agg' keeps the raw columns plus the log's aggregates only")
a = ap.parse_args()
D = Path(a.data)
T = pd.read_parquet(D / 'transactions_train.parquet')
T['t_dat'] = pd.to_datetime(T['t_dat'])
end = T.t_dat.max() + pd.Timedelta(days=1)
start = end - pd.Timedelta(days=7 * (a.window + 1))
stop = start + pd.Timedelta(days=7)
T = T[T.t_dat < stop]
T['week'] = ((T.t_dat - T.t_dat.min()).dt.days // 7).astype(np.int16)
hist_c = T.loc[T.t_dat < start, 'customer_id'].unique()
pick = set(np.random.default_rng(a.seed).choice(hist_c, a.customers, replace=False))
T = T[T.customer_id.isin(pick)].reset_index(drop=True)
art = pd.read_parquet(D / 'articles.parquet', columns=['article_id', 'product_code', 'product_type_no', 'department_no',
                                                         'garment_group_no', 'colour_group_code'])
T = T.merge(art, on='article_id', how='left')
H, L = T[T.t_dat < start], T[T.t_dat >= start]
label_week = int(L.week.min()) if len(L) else int(H.week.max()) + 1
pairs = H[['customer_id', 'article_id']].drop_duplicates().merge(art, on='article_id', how='left')
pairs['week'] = label_week
bought = set(zip(L.customer_id, L.article_id))
y = np.array([(c, i) in bought for c, i in zip(pairs.customer_id, pairs.article_id)], dtype=int)
custs = np.array(sorted(pick))
ho_c = set(np.random.default_rng(a.seed + 7).choice(custs, len(custs) // 5, replace=False))
ho = pairs.customer_id.isin(ho_c).to_numpy()
log = H[['customer_id', 'article_id', 'week', 'price', 'sales_channel_id', 'product_code', 'product_type_no',
         'department_no', 'garment_group_no', 'colour_group_code']].reset_index(drop=True)
print('pairs', pairs.shape, 'positive rate', round(y.mean(), 4), 'history rows', len(log), 'label week', label_week, flush=True)
tr, te = pairs[~ho].reset_index(drop=True), pairs[ho].reset_index(drop=True)
ytr, yte = y[~ho], y[ho]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time(); info = {}
if a.arm == 'raw':
    Xtr, Xte = tr.drop(columns=['week']), te.drop(columns=['week'])
else:
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    try:
        tr.assign(bought_again=ytr).to_parquet(tmp / 'train.parquet'); te.to_parquet(tmp / 'test.parquet')
        log.to_parquet(tmp / 'log.parquet')
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
                        '--test', str(tmp / 'test.parquet'), '--target', 'bought_again', '--task', 'binary', '--out-dir', str(tmp / 'out'),
                        '--table', f'log={tmp / "log.parquet"}:customer_id:week'], check=True)
        import resource
        info['peak_mb'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
        Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['bought_again'])
        Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if a.keep == 'agg':
        agg = [c for c in Xtr.columns if c.startswith('log__')]
        Xtr = pd.concat([tr.drop(columns=['week']), Xtr[agg]], axis=1)
        Xte = pd.concat([te.drop(columns=['week']), Xte[agg]], axis=1)
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype('category')
cid = tr.customer_id.to_numpy()
va = np.isin(cid, np.random.default_rng(a.seed + 1).choice(np.unique(cid), int(0.15 * len(np.unique(cid))), replace=False))
for X in (Xtr, Xte):
    X.drop(columns=[c for c in ('customer_id',) if c in X.columns], inplace=True)
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr[~va], ytr[~va]), 5000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else '') + ('' if a.keep == 'all' else '_' + a.keep), seed=a.seed, window=a.window, customers=a.customers,
           auc=float(roc_auc_score(yte, p)), n_cols=Xtr.shape[1], best_it=b.best_iteration, fe_s=round(fe_t),
           total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res)); open(a.log, 'a').write(json.dumps(res) + '\n')
