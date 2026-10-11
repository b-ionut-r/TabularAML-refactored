"""Santander Product Recommendation (Kaggle 2016, $60k): which products a bank customer adds next month. MAP@7 over
every customer of the scored month (customers adding nothing score 0), as the contest.

Data: a public copy of the contest files (``xhk300/santander-product-recommendation-extract``, train_ver2.csv with the
contest's 13,647,309 rows; Kaggle's own download needs the rules accepted), read as Kaggle's CSV reads and saved to
``train_ver2_k.parquet``. The contest gave monthly snapshots up to 2016-05 (demographics and 24 product flags) and
scored the products customers added in 2016-06, with the June rows holding demographics only. A window does the same
one month earlier: ``--win`` 0 scores 2016-05, 1 scores 2016-04. Training rows are the additions of the two months
before (one row per customer and added product, the month's demographics, label = the product); held-out rows are a
``--test-frac`` sample of the scored month's customers, demographics only. The snapshots before the scored month are
the history file, cut to the main tables' customers (as kernels did, to fit memory). Held-out additions reach only
the scorer. Products owned the month before are never recommended (the same for every arm).

Arms (same bagged LightGBM, multiclass: early stopping on 15% of training rows, then 3 seeds):
  ``raw``  the month's own columns;
  ``ff``   blind: ``scripts/contest_features.py`` at its defaults, history file given as a related table
           (``hist=...:ncodpers:fecha_dato``);
  ``hand`` raw + the winners' and top kernels' history features: each product's flag 1-5 months back, months owned
           of the last 5, added / dropped last month, products held last month, and the previous month's
           demographics that change (activity, segment, relation type);
  ``ff_hand`` both. ``--shuffle`` permutes the training labels (leakage control). Reports runtime and peak memory.
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil, resource
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--tag', default=''); ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='/home/user/data/spr/'); ap.add_argument('--log', default='spr.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--test-frac', type=float, default=0.3)
ap.add_argument('--train-months', type=int, default=2); ap.add_argument('--prep-only', action='store_true')
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1])); ap.add_argument('--judge-seed', type=int, default=0)
a = ap.parse_args()
D = Path(a.data)
t0 = time.time()
C = D / f'win{a.win}_m{a.train_months}_f{a.test_frac}'


def prep():
    """Builds the window's tables once (the 13.6M-row file needs ~13 GB) in a child process."""
    import pyarrow.parquet as pq
    # Text columns as categories (13.6M rows of Python strings would not fit in memory).
    df = pq.read_table(D / 'train_ver2_k.parquet').to_pandas(strings_to_categorical=True)
    df['fecha_dato'] = df.fecha_dato.cat.as_ordered()
    months = sorted(df.fecha_dato.cat.categories)
    H = months[-1 - a.win]
    T = months[months.index(H) - a.train_months:months.index(H)]
    PROD = [c for c in df.columns if c.startswith('ind_') and c.endswith('ult1')]
    DEMO = [c for c in df.columns if c not in PROD]

    def added(m):
        """(customer, product) pairs added in month m: held in m, not held (or not a customer) the month before."""
        prev = months[months.index(m) - 1]
        cur = df[df.fecha_dato == m].set_index('ncodpers')[PROD]
        pv = df[df.fecha_dato == prev].set_index('ncodpers')[PROD].reindex(cur.index).fillna(0)
        new = (cur.fillna(0).to_numpy() == 1) & (pv.to_numpy() != 1)
        r, c = np.nonzero(new)
        return pd.DataFrame({'ncodpers': cur.index.to_numpy()[r], 'product': np.array(PROD)[c]})

    rows = []
    for m in T:
        demo = df[df.fecha_dato == m][DEMO].drop_duplicates('ncodpers')
        rows.append(added(m).merge(demo, on='ncodpers', how='left'))
    trm = pd.concat(rows, ignore_index=True)
    tem = df[df.fecha_dato == H][DEMO]
    tem = tem[np.random.default_rng(1).random(len(tem)) < a.test_frac].reset_index(drop=True)
    truth = added(H)
    truth = truth[truth.ncodpers.isin(set(tem.ncodpers))]
    owned = df[df.fecha_dato == months[months.index(H) - 1]][['ncodpers'] + PROD]
    hist = df[(df.fecha_dato < H) & df.ncodpers.isin(set(trm.ncodpers) | set(tem.ncodpers))].reset_index(drop=True)
    hist['fecha_dato'] = hist.fecha_dato.astype(str)
    C.mkdir(exist_ok=True)
    for n, t in dict(trm=trm, tem=tem, truth=truth, owned=owned, hist=hist).items():
        t.to_parquet(C / f'{n}.parquet')
    json.dump(dict(months=months, H=H, T=T, PROD=PROD), open(C / 'meta.json', 'w'))


if '--prep-only' in sys.argv:
    prep(); sys.exit()
if not (C / 'meta.json').exists():
    subprocess.run([sys.executable] + sys.argv + ['--prep-only'], check=True)
M = json.load(open(C / 'meta.json')); months, H, T, PROD = M['months'], M['H'], M['T'], M['PROD']
trm, tem, owned_prev = (pd.read_parquet(C / f'{n}.parquet') for n in ('trm', 'tem', 'owned'))
for X in (trm, tem):
    X['fecha_dato'] = X.fecha_dato.astype(str)
owned_prev = owned_prev.set_index('ncodpers')
truth = pd.read_parquet(C / 'truth.parquet').groupby('ncodpers')['product'].apply(set)
hist = pd.read_parquet(C / 'hist.parquet')
ytr = trm.pop('product').astype(str).to_numpy()
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
print(f'scored month {H}, training months {T}: {len(trm)} additions, {len(tem)} held-out customers '
      f'({len(truth)} adding), history {len(hist)} rows', flush=True)

# Main tables round-trip through CSV, as the contest's files read.
tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
trm.assign(product=ytr).to_csv(tmp / 'train.csv', index=False); tem.to_csv(tmp / 'test.csv', index=False)
Xtr = pd.read_csv(tmp / 'train.csv', low_memory=False).drop(columns=['product'])
Xte = pd.read_csv(tmp / 'test.csv', low_memory=False)
info = {}


def hand(X):
    """Product flags 1-5 months back and their changes, from the history file (strictly earlier months)."""
    h = hist[['ncodpers', 'fecha_dato'] + PROD + ['ind_actividad_cliente', 'segmento', 'tiprel_1mes']]
    mi = {m: i for i, m in enumerate(months)}
    X = X[['ncodpers', 'fecha_dato']].copy(); X['_i'] = X.fecha_dato.map(mi)
    hh = h.assign(_i=h.fecha_dato.map(mi)).drop(columns=['fecha_dato'])
    out = {}
    lag = {}
    for k in range(1, 6):
        j = X[['ncodpers', '_i']].assign(_i=X._i - k).merge(hh, on=['ncodpers', '_i'], how='left')
        lag[k] = j
        for p in PROD:
            out[f'{p}_lag{k}'] = j[p].to_numpy(dtype=float)
    for p in PROD:
        L = np.column_stack([out[f'{p}_lag{k}'] for k in range(1, 6)])
        out[f'{p}_months5'] = np.nansum(L, 1)
        out[f'{p}_added1'] = ((L[:, 0] == 1) & (L[:, 1] != 1)).astype(float)
        out[f'{p}_dropped1'] = ((L[:, 0] != 1) & (L[:, 1] == 1)).astype(float)
    out['n_products_lag1'] = lag[1][PROD].sum(1, min_count=1).to_numpy(dtype=float)
    out['n_products_lag2'] = lag[2][PROD].sum(1, min_count=1).to_numpy(dtype=float)
    out['actividad_lag1'] = lag[1]['ind_actividad_cliente'].to_numpy(dtype=float)
    H_ = pd.DataFrame(out)
    H_['segmento_lag1'] = pd.Categorical(lag[1]['segmento'].astype(str))
    H_['tiprel_lag1'] = pd.Categorical(lag[1]['tiprel_1mes'].astype(str))
    return H_


if a.arm.startswith('ff'):
    hp = tmp / 'hist.parquet'; hist.to_parquet(hp)
    cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'product', '--task', 'multiclass',
           '--table', f'hist={hp}:ncodpers:fecha_dato', '--budget', str(a.budget), '--out-dir', str(tmp / 'out')] + \
          (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True, cwd=a.repo)
    info['pipe_peak_gb'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20, 2)
    Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert len(Ftr) == len(Xtr) and len(Fte) == len(Xte)
    new = [c for c in Ftr.columns if c not in Xtr.columns and c != 'product']
    Xtr = pd.concat([Xtr, Ftr[new].reset_index(drop=True)], axis=1); Xte = pd.concat([Xte, Fte[new].reset_index(drop=True)], axis=1)
    info['n_new'] = len(new)
if a.arm.endswith('hand'):
    Xtr = pd.concat([Xtr, hand(Xtr)], axis=1); Xte = pd.concat([Xte, hand(Xte)], axis=1)
shutil.rmtree(tmp, ignore_errors=True)
fe_t = time.time() - t0
ids = Xte.ncodpers.to_numpy()
for X in (Xtr, Xte):
    X.drop(columns=['ncodpers'], inplace=True)
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
classes = np.array(sorted(set(ytr)))
yi = np.searchsorted(classes, ytr)
P = dict(objective='multiclass', num_class=len(classes), learning_rate=0.05, num_leaves=63, min_child_samples=50,
         feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
va = np.random.default_rng(a.judge_seed).random(len(Xtr)) < 0.15
b = lgb.train(dict(P, seed=a.judge_seed * 10), lgb.Dataset(Xtr[~va], yi[~va]), 3000, valid_sets=[lgb.Dataset(Xtr[va], yi[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
p = np.zeros((len(Xte), len(classes)))
for s in range(3):
    p += lgb.train(dict(P, seed=a.judge_seed * 10 + s), lgb.Dataset(Xtr, yi), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
own = owned_prev.reindex(ids)[list(classes)].fillna(0).to_numpy() == 1
p[own] = -1
top = np.argsort(-p, 1)[:, :7]


def apk(actual, pred):
    hits, s = 0, 0.0
    for i, q in enumerate(pred):
        if q in actual:
            hits += 1; s += hits / (i + 1)
    return s / min(len(actual), 7)


tr_ = truth.to_dict()
ap7 = np.array([apk(tr_[c], classes[t]) if c in tr_ else 0.0 for c, t in zip(ids, top)])
adders = np.array([c in tr_ for c in ids])
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, month=str(H), judge_seed=a.judge_seed,
           map7=float(ap7.mean()), map7_adders=float(ap7[adders].mean()), best_it=b.best_iteration, n_cols=Xtr.shape[1],
           fe_s=round(fe_t), total_s=round(time.time() - t0), peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2),
           n_tr=len(Xtr), n_te=len(Xte), n_adders=int(adders.sum()), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
