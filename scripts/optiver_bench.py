"""Optiver Realized Volatility Prediction (Kaggle 2021, $100k): realized volatility of a stock over the next ten
minutes from the previous ten minutes of its order book (two levels, one row per book change) and its trades.

Data: a public Kaggle copy of the contest's files (``akshaymairal/optiver-realized-volatility-prediction``):
train.csv (stock_id, time_id, target: 112 stocks x 3,830 ten-minute buckets), book_train.parquet and
trade_train.parquet partitioned by stock. A sample of ``--stocks`` stocks (``--stock-seed``) keeps the child tables in
memory. The contest's test was later buckets of the same stocks, all stocks of a bucket together; time_ids are
shuffled in the files (their order is not given), so the holdout is a random 20% of the time_ids (``--seed``) with
every sampled stock of those buckets; held-out targets reach only the scorer.
Main rows carry ``row_id`` = stock_id * 100000 + time_id (the contest's test.csv key "stock-time" as an integer, to keep the
child tables small); the book and trade tables get the same key
(their stock_id comes from the partition directory) and are given to the pipeline as child tables.
Metric: RMSPE. Judge: the same bagged LightGBM on log(target) (early stopping on 15% of the training rows, then 3
seeds).
Arms:
  ``raw``   stock_id (the file's only column besides the bucket id);
  ``ff``    blind: ``scripts/contest_features.py`` at its defaults (``--log-target``, a ratio metric) with the book and
            trade tables;
  ``hand``  raw + the features of the top public kernels and winners' write-ups: weighted average prices (levels 1, 2),
            log returns and realized volatility over the whole bucket and from seconds 150 / 300 / 450 on, price
            spreads, bid / ask spreads, volume imbalance, total volume, trade realized volatility, trade counts, sizes
            and order counts, and per-bucket means of the main ones over the sampled stocks (label-free cross-stock
            features of the same time_id, test buckets included);
  ``ff_hand`` both.
``--shuffle`` permutes the training targets (leakage control). Reports runtime and peak memory.
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil, resource
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--stocks', type=int, default=24); ap.add_argument('--stock-seed', type=int, default=0)
ap.add_argument('--tag', default=''); ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='/home/user/data/optiver/'); ap.add_argument('--log', default='optiver.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--groups', default='')
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
ap.add_argument('--ff-cache', action='store_true')
ap.add_argument('--add-paths', action='store_true', help='diagnostic: append tabularaml.generate.returns features of both tables')
a = ap.parse_args()
D = Path(a.data)
allst = sorted(int(p.name.split('=')[1]) for p in (D / 'book_train.parquet').iterdir())
stocks = sorted(np.random.default_rng(a.stock_seed).choice(allst, min(a.stocks, len(allst)), replace=False).tolist())
main = pd.read_csv(D / 'train.csv')
main = main[main.stock_id.isin(stocks)].reset_index(drop=True)
main.insert(0, 'row_id', main.stock_id.astype(np.int64) * 100_000 + main.time_id)


def stock_table(kind, s):
    t = pd.read_parquet(D / f'{kind}_train.parquet' / f'stock_id={s}')
    t.insert(0, 'row_id', s * 100_000 + t.time_id.astype(np.int64))
    return t.drop(columns=['time_id'])


def child_file(kind):
    """The sampled stocks' rows of one table with the row_id key, written a stock at a time (never all in memory)."""
    import pyarrow as pa, pyarrow.parquet as pq
    f = D / f'prep_{kind}_{len(stocks)}_{a.stock_seed}.parquet'
    if not f.exists():
        w = None
        for s in stocks:
            t = pa.Table.from_pandas(stock_table(kind, s), preserve_index=False)
            w = w or pq.ParquetWriter(str(f) + '.tmp', t.schema)
            w.write_table(t, row_group_size=1_000_000)
        w.close(); Path(str(f) + '.tmp').rename(f)
    return f


book_f, trade_f = child_file('book'), child_file('trade')
tids = np.sort(main.time_id.unique())
rng = np.random.default_rng(a.seed)
ho_t = set(rng.choice(tids, len(tids) // 5, replace=False).tolist())
is_ho = main.time_id.isin(ho_t).to_numpy()
tr, te = main[~is_ho].reset_index(drop=True), main[is_ho].reset_index(drop=True)
ytr, yte = tr.pop('target').to_numpy(), te.pop('target').to_numpy()
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
print(f'stocks {len(stocks)} train {len(tr)} held out {len(te)}', flush=True)


def raw(X):
    return X[['stock_id']].astype(str).astype('category')


def hand_stock(book, trade):
    b = book
    b['wap1'] = (b.bid_price1 * b.ask_size1 + b.ask_price1 * b.bid_size1) / (b.bid_size1 + b.ask_size1)
    b['wap2'] = (b.bid_price2 * b.ask_size2 + b.ask_price2 * b.bid_size2) / (b.bid_size2 + b.ask_size2)
    g = b.groupby('row_id', sort=False)
    b['lr1'] = np.log(b.wap1).groupby(b.row_id).diff()
    b['lr2'] = np.log(b.wap2).groupby(b.row_id).diff()
    b['price_spread'] = (b.ask_price1 - b.bid_price1) / ((b.ask_price1 + b.bid_price1) / 2)
    b['bid_spread'] = b.bid_price1 - b.bid_price2
    b['ask_spread'] = b.ask_price1 - b.ask_price2
    b['total_volume'] = b.ask_size1 + b.ask_size2 + b.bid_size1 + b.bid_size2
    b['volume_imbalance'] = ((b.ask_size1 + b.ask_size2) - (b.bid_size1 + b.bid_size2)).abs()
    rv = lambda x: np.sqrt(np.nansum(x ** 2))
    H = []
    for start in (0, 150, 300, 450):
        s = b[b.seconds_in_bucket >= start]
        g = s.groupby('row_id')
        f = pd.DataFrame({'rv1': g.lr1.apply(rv), 'rv2': g.lr2.apply(rv)})
        if start == 0:
            f = f.join(g[['wap1', 'wap2', 'price_spread', 'bid_spread', 'ask_spread', 'total_volume', 'volume_imbalance']].agg(['mean', 'std', 'sum']).set_axis(
                [f'{c}_{m}' for c in ['wap1', 'wap2', 'price_spread', 'bid_spread', 'ask_spread', 'total_volume', 'volume_imbalance'] for m in ['mean', 'std', 'sum']], axis=1))
            f['n_book'] = g.size()
        H.append(f.add_suffix(f'_{start}'))
    t = trade
    t['lr'] = np.log(t.price).groupby(t.row_id).diff()
    for start in (0, 300):
        s = t[t.seconds_in_bucket >= start]; g = s.groupby('row_id')
        f = pd.DataFrame({'trade_rv': g.lr.apply(rv), 'trade_n': g.size(), 'trade_size': g['size'].sum(),
                          'trade_orders': g.order_count.sum(), 'trade_size_mean': g['size'].mean()})
        H.append(f.add_suffix(f'_{start}'))
    return pd.concat(H, axis=1)


def hand(tr, te):
    H = pd.concat([hand_stock(stock_table('book', s), stock_table('trade', s)) for s in stocks])
    A = pd.concat([tr, te], ignore_index=True)
    F = H.reindex(A.row_id.to_numpy()).reset_index(drop=True)
    # Cross-stock: the mean of the main ones over the sampled stocks of the same bucket.
    for c in ['rv1_0', 'rv2_0', 'rv1_300', 'trade_rv_0', 'price_spread_mean_0', 'total_volume_mean_0', 'trade_size_0']:
        F[f'{c}_time_mean'] = F[c].groupby(A.time_id.to_numpy()).transform('mean').to_numpy()
    n = len(tr)
    return F.iloc[:n].reset_index(drop=True), F.iloc[n:].reset_index(drop=True)


t0 = time.time(); info = {}
Xtr, Xte = raw(tr), raw(te)
cache = D / f'ff_cache_{a.seed}{a.tag}{"_shuf" if a.shuffle else ""}'
if a.arm.startswith('ff') and a.ff_cache and (cache / 'tr.parquet').exists():
    Ftr, Fte = pd.read_parquet(cache / 'tr.parquet'), pd.read_parquet(cache / 'te.parquet')
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1); info['n_new'] = Ftr.shape[1]
elif a.arm.startswith('ff'):
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    tr.assign(target=ytr).to_parquet(tmp / 'train.parquet'); te.to_parquet(tmp / 'test.parquet')
    cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
           '--test', str(tmp / 'test.parquet'), '--target', 'target', '--id', 'row_id', '--task', 'regression', '--log-target',
           '--table', f'book={book_f}:row_id', '--table', f'trade={trade_f}:row_id',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True, cwd=a.repo)
    info['pipe_peak_gb'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20, 2)
    Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert (Ftr['row_id'].to_numpy() == tr.row_id.to_numpy()).all() and (Fte['row_id'].to_numpy() == te.row_id.to_numpy()).all()
    new = [c for c in Ftr.columns if c not in tr.columns and c != 'target']
    Ftr, Fte = Ftr[new].reset_index(drop=True), Fte[new].reset_index(drop=True)
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1)
    info['n_new'] = len(new)
    cache.mkdir(exist_ok=True); Ftr.to_parquet(cache / 'tr.parquet'); Fte.to_parquet(cache / 'te.parquet')
    shutil.rmtree(tmp, ignore_errors=True)
if a.arm.endswith('hand'):
    Htr, Hte = hand(tr, te)
    Xtr, Xte = pd.concat([Xtr, Htr], axis=1), pd.concat([Xte, Hte], axis=1)
if a.add_paths:
    from tabularaml.generate.returns import return_features
    book, trade = pd.read_parquet(book_f), pd.read_parquet(trade_f)
    R = pd.concat([return_features(book, 'row_id', 'book'), return_features(trade, 'row_id', 'trade')], axis=1)
    if a.groups == 'wap':
        b = book[['row_id', 'seconds_in_bucket']].copy()
        b['wap1'] = (book.bid_price1 * book.ask_size1 + book.ask_price1 * book.bid_size1) / (book.bid_size1 + book.ask_size1)
        b['wap2'] = (book.bid_price2 * book.ask_size2 + book.ask_price2 * book.bid_size2) / (book.bid_size2 + book.ask_size2)
        W = return_features(b, 'row_id', 'bookw')
        R = pd.concat([R, W[[c for c in W.columns if 'wap' in c]]], axis=1)
    if a.groups == 'tmean':
        tid = pd.Series(R.index.to_numpy() % 100_000, index=R.index)
        for c in [c for c in R.columns if c.endswith('__rv') or c.endswith('__rv_late')]:
            R[c + '_time_mean'] = R[c].groupby(tid).transform('mean')
    Xtr = pd.concat([Xtr, R.reindex(tr.row_id.to_numpy()).reset_index(drop=True)], axis=1)
    Xte = pd.concat([Xte, R.reindex(te.row_id.to_numpy()).reset_index(drop=True)], axis=1)
fe_t = time.time() - t0
for c in Xtr.columns:
    if not (pd.api.types.is_numeric_dtype(Xtr[c]) or isinstance(Xtr[c].dtype, pd.CategoricalDtype)):
        Xtr[c], Xte[c] = Xtr[c].astype(str), Xte[c].astype(str)
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
lt = np.log(ytr)
P = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); k = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:k]), np.sort(perm[k:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], lt[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], lt[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, lt), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
pred = np.exp(p)
rmspe = lambda q: float(np.sqrt(np.mean(((yte - q) / yte) ** 2)))
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, stocks=len(stocks), rmspe=rmspe(pred),
           rmspe_const=rmspe(np.full(len(yte), np.exp(lt.mean()))), best_it=b.best_iteration, n_cols=Xtr.shape[1],
           fe_s=round(fe_t), total_s=round(time.time() - t0), peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2),
           n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
