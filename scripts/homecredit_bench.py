"""Home Credit Default Risk ($70k): main application table, optional label-free aggregates of the 5 child
tables (a RelatedTables parquet indexed by SK_ID_CURR) and out-of-fold child-row models
(``--childmodel``). Stratified 80/20 holdout of loans; AUC. Float32 matrices, one copy (15 GB RAM).
Only training loans' labels reach the child-row models; ``--shuffle`` permutes the training labels
(leakage control: AUC must then land near 0.5); ``--sub`` subsamples loans for a quick control."""
import sys, json, warnings, gc, argparse; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
ap = argparse.ArgumentParser(); ap.add_argument('--relfile', default=''); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default=''); ap.add_argument('--childmodel', action='store_true')
ap.add_argument('--data', default='data/homecredit/'); ap.add_argument('--log', default='homecredit.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--sub', type=int, default=0)
a = ap.parse_args()
app = pd.read_parquet('/tmp/claude-0/data/hc/app.parquet')
y = app.pop('TARGET').to_numpy(); ids = app.pop('SK_ID_CURR').to_numpy()
cats = [c for c in app.columns if not pd.api.types.is_numeric_dtype(app[c])]
for c in cats: app[c] = app[c].cat.codes.astype(np.float32).replace(-1, np.nan)
names = list(app.columns)
M = app.to_numpy(np.float32); del app
if a.relfile:
    R = pd.read_parquet(a.data + a.relfile)
    R = R.reindex(ids); names += list(R.columns)
    cnt = [i for i, c in enumerate(R.columns) if c.endswith('__count')]
    RM = R.to_numpy(np.float32); del R; gc.collect()
    RM[:, cnt] = np.nan_to_num(RM[:, cnt])
    M = np.hstack([M, RM]); del RM; gc.collect()
itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, random_state=a.seed, stratify=y)
itr, ite = np.sort(itr), np.sort(ite)
if a.shuffle:
    y = y.copy(); y[itr] = np.random.default_rng(0).permutation(y[itr])
if a.childmodel:
    import time; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tabularaml.generate.relational import Child, child_model_features
    D = a.data; keep = set(ids); rd = lambda f: (lambda t: t[t.SK_ID_CURR.isin(keep)])(pd.read_parquet(D + f + '.parquet'))
    yk = pd.Series(y[itr], index=ids[itr])
    for ch in [Child('bureau', rd('bureau'), key='SK_ID_CURR', time='DAYS_CREDIT', drop=['SK_ID_BUREAU']),
               Child('prev', rd('previous_application'), key='SK_ID_CURR', time='DAYS_DECISION', drop=['SK_ID_PREV']),
               Child('inst', rd('installments_payments'), key='SK_ID_CURR', time='DAYS_INSTALMENT', drop=['SK_ID_PREV']),
               Child('pos', rd('POS_CASH_balance'), key='SK_ID_CURR', time='MONTHS_BALANCE', drop=['SK_ID_PREV']),
               Child('cc', rd('credit_card_balance'), key='SK_ID_CURR', time='MONTHS_BALANCE', drop=['SK_ID_PREV'])]:
        t = time.time()
        F = child_model_features(ch, yk, ids[ite], seed=a.seed).reindex(ids)
        print(ch.name, F.shape, round(time.time() - t), flush=True)
        names += list(F.columns); M = np.hstack([M, F.to_numpy(np.float32)]); del F, ch; gc.collect()
P = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_child_samples=100, feature_fraction=0.3,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=a.seed, max_bin=255)
cat_idx = [names.index(c) for c in cats]
perm = np.random.default_rng(a.seed).permutation(len(itr)); tr, va = itr[np.sort(perm[:int(.85*len(perm))])], itr[np.sort(perm[int(.85*len(perm)):])]
dtr = lgb.Dataset(M[tr], y[tr], categorical_feature=cat_idx, free_raw_data=True)
dva = lgb.Dataset(M[va], y[va], reference=dtr)
b = lgb.train(P, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
n_it = int(b.best_iteration * 1.1) + 1; del dtr, dva, b; gc.collect()
full = lgb.train(P, lgb.Dataset(M[itr], y[itr], categorical_feature=cat_idx), n_it)
p = full.predict(M[ite])
res = dict(arm=('cm' if a.childmodel else 'rel' if a.relfile else 'raw') + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, ncol=len(names), auc=roc_auc_score(y[ite], p), logloss=log_loss(y[ite], p))
print('RESULT', json.dumps(res)); open(a.log, 'a').write(json.dumps(res) + '\n')
