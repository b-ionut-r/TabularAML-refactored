"""Instacart Market Basket Analysis ($25k): which previously bought products a user reorders in the next order.
Rows = (user, product) pairs from the user's prior orders; label = in the user's train order.
Holdout: 20% of users. AUC / logloss."""
import sys, time, json, warnings, argparse
warnings.filterwarnings('ignore'); from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--users', type=int, default=15000); ap.add_argument('--budget', type=float, default=1200)
ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/instacart/', help='folder with the competition files as parquet')
ap.add_argument('--log', default='instacart.jsonl'); ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = a.data
orders = pd.read_parquet(D + 'orders.parquet')
rng = np.random.default_rng(0)
tr_users = orders.loc[orders.eval_set == 'train', 'user_id'].unique()
users = rng.choice(tr_users, a.users, replace=False)
orders = orders[orders.user_id.isin(users)].copy()
orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)
orders['day'] = orders.groupby('user_id')['days_since_prior_order'].cumsum()
prior_o = orders[orders.eval_set == 'prior']
op = pd.read_parquet(D + 'order_products__prior.parquet')
op = op[op.order_id.isin(prior_o.order_id)].merge(prior_o[['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'day']], on='order_id')
prod = pd.read_parquet(D + 'products.parquet')[['product_id', 'aisle_id', 'department_id']]
train_o = orders[orders.eval_set == 'train']
opt = pd.read_parquet(D + 'order_products__train.parquet'); opt = opt[opt.order_id.isin(train_o.order_id)]
pairs = op[['user_id', 'product_id']].drop_duplicates()
pairs = pairs.merge(train_o[['user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'day']], on='user_id')
pairs = pairs.merge(prod, on='product_id', how='left')
lab = set(zip(opt.merge(train_o[['order_id', 'user_id']], on='order_id').user_id, opt.product_id))
y = np.array([(u, p) in lab for u, p in zip(pairs.user_id, pairs.product_id)], dtype=int)
pairs['up'] = pairs.user_id.astype(np.int64) * 1_000_000 + pairs.product_id
op['up'] = op.user_id.astype(np.int64) * 1_000_000 + op.product_id
print('pairs', pairs.shape, 'positive rate', y.mean(), 'prior rows', len(op), flush=True)
u_ho = np.random.default_rng(a.seed).choice(users, int(0.2 * len(users)), replace=False)
ho = pairs.user_id.isin(u_ho).to_numpy()
t0 = time.time(); info = {}
feat_cols = ['user_id', 'product_id', 'aisle_id', 'department_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']
X = pairs[feat_cols].copy()
if 'rel' in a.arm:
    from tabularaml.generate.relational import Child, RelatedTables
    ch = [Child('up', op[['up', 'order_number', 'add_to_cart_order', 'reordered', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'day']], key='up', time='order_number'),
          Child('user', prior_o[['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'day']].merge(op.groupby('order_id').agg(basket=('product_id', 'size'), basket_reorder=('reordered', 'mean')).reset_index(), on='order_id').drop(columns=['order_id']), key='user_id', time='order_number'),
          Child('prod', op[['product_id', 'add_to_cart_order', 'reordered', 'order_number', 'days_since_prior_order']], key='product_id')]
    F = RelatedTables(ch).features()
    print('related', F.shape, round(time.time() - t0), flush=True)
    for k, cols in (('up', [c for c in F.columns if c.startswith('up__')]), ('user_id', [c for c in F.columns if c.startswith('user__')]), ('product_id', [c for c in F.columns if c.startswith('prod__')])):
        X = X.join(F.loc[:, cols].reindex(pairs[k].to_numpy()).reset_index(drop=True))
if 'asof' in a.arm:
    # The same children aggregated as of each row's order (asof_features), as contest_features.py does
    # when the main key repeats; product-level history stays keyed (no comparable time).
    from tabularaml.generate.relational import Child, RelatedTables, asof_features
    M = pairs[['up', 'user_id', 'order_number']].reset_index(drop=True)
    uo = prior_o[['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'day']].merge(
        op.groupby('order_id').agg(basket=('product_id', 'size'), basket_reorder=('reordered', 'mean')).reset_index(), on='order_id').drop(columns=['order_id'])
    parts = [asof_features(M, 'up', 'order_number', Child('up', op[['up', 'order_number', 'add_to_cart_order', 'reordered', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'day']], key='up', time='order_number')),
             asof_features(M, 'user_id', 'order_number', Child('user', uo, key='user_id', time='order_number'))]
    Fp = RelatedTables([Child('prod', op[['product_id', 'add_to_cart_order', 'reordered', 'order_number', 'days_since_prior_order']], key='product_id')]).features()
    parts.append(Fp.reindex(pairs['product_id'].to_numpy()).reset_index(drop=True))
    X = pd.concat([X.reset_index(drop=True)] + parts, axis=1)
    print('asof', X.shape, round(time.time() - t0), flush=True)
if 'text' in a.arm:
    X['product_name'] = pairs['product_id'].map(pd.read_parquet(D + 'products.parquet').set_index('product_id')['product_name']).to_numpy()
Xtr, Xho, ytr, yho = X[~ho].reset_index(drop=True), X[ho].reset_index(drop=True), y[~ho], y[ho]
if a.shuffle:  # leakage control: permuted training labels
    ytr = np.random.default_rng(0).permutation(ytr)
if 'forge' in a.arm:
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='binary', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True, **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xho)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, feats=f.new_columns_[:60])
    Xtr, Xho = f.transform_train(Xtr), f.transform(Xho)
fe_t = time.time() - t0
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, num_threads=4, verbose=-1, seed=a.seed)
utr = pairs.user_id.to_numpy()[~ho]; uu = np.unique(utr); va_u = np.random.default_rng(a.seed + 1).choice(uu, int(0.15 * len(uu)), replace=False)
va = np.isin(utr, va_u)
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 5000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho)
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, auc=roc_auc_score(yho, p), logloss=log_loss(yho, p), n_cols=Xtr.shape[1], best_it=b.best_iteration, fe_s=round(fe_t), total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
