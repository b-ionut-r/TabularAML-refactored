"""Data Science Bowl 2019 (Kaggle, $160k): event logs per child (installation) and labelled assessments.

Data: Hugging Face ``pytorch-lifestream/datascience-bowl2019`` (train.csv.gz, train_labels.csv.gz).
Main rows: the labelled assessments (installation, title, start time); target accuracy_group 0-3.
Child: the 11.3M game events (without the event_data JSON). Holdout: 20% of installations (the
contest's test set is new children), two seeds. Metric: quadratic weighted kappa of the regression
output cut at thresholds matching the training class shares (the usual contest decoding), and RMSE.

Test-file parity: the contest's test file holds each new child's history up to one randomly chosen
assessment, and only that assessment is scored. So the held-out children enter FeatureForge (as unlabeled
rows) with one random assessment each, and QWK is the mean over ``--draws`` random draws of one assessment
per held-out child (``qwk``). Cut-points: ``qwk`` cuts at the training class shares of the test predictions
(label-free, as contest kernels did); ``qwk_trcut`` uses cut-points fit on training rows only (the early-
stopping model's predictions on its validation children, cut at the training class shares); ``qwk_all``
scores every held-out assessment (the earlier, optimistic protocol). ``--shuffle`` permutes the training
labels (a leakage control: every arm must then land near 0).

Arms: ``raw`` (assessment title, world, start time), ``asof`` (+ event-log aggregations as of each
assessment's start: only the child's earlier events), ``forge`` / ``asof_forge`` (+ FeatureForge).
"""
import sys, time, json, warnings, argparse, gc
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=1200); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/dsb/'); ap.add_argument('--log', default='dsb.jsonl')
ap.add_argument('--draws', type=int, default=20); ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--pick-transform', action='store_true',
                help="transform each draw's picked assessments alone, so no held-out child's later rows are visible (test parity)")
ap.add_argument('--shuffle-history', action='store_true',
                help="permute the event codes of assessment events (the attempts that make up the outcome), independently of the labels")
a = ap.parse_args()
D = Path(a.data)
lab = pd.read_csv(D / 'train_labels.csv.gz')
cache = D / 'events.parquet'
if not cache.exists():
    cols = ['event_id', 'game_session', 'timestamp', 'installation_id', 'event_count', 'event_code', 'game_time', 'title',
            'event_type', 'world']
    ids = set(lab.installation_id)
    parts = []
    for ch in pd.read_csv(D / 'train.csv.gz', usecols=cols, chunksize=2_000_000):
        ch = ch[ch.installation_id.isin(ids)]
        ch['timestamp'] = pd.to_datetime(ch['timestamp']).astype('int64') / 1e9
        for c in ('event_id', 'game_session', 'installation_id', 'title', 'event_type', 'world'):
            ch[c] = ch[c].astype('category')
        parts.append(ch)
    ev = pd.concat(parts, ignore_index=True)
    for c in ('event_id', 'game_session', 'installation_id', 'title', 'event_type', 'world'):
        ev[c] = ev[c].astype(str).astype('category')
    ev['event_code'] = ev['event_code'].astype(str).astype('category')
    ev.to_parquet(cache); del parts; gc.collect()
ev = pd.read_parquet(cache)
if a.shuffle_history:
    m = (ev['event_type'].astype(str) == 'Assessment').to_numpy()
    codes = ev['event_code'].astype(str).to_numpy().copy()
    codes[m] = np.random.default_rng(5).permutation(codes[m])
    ev['event_code'] = pd.Categorical(codes)
st = ev.groupby('game_session', observed=True).agg(start=('timestamp', 'min'), world=('world', 'first'))
main = lab[['installation_id', 'game_session', 'title', 'accuracy_group']].merge(st, left_on='game_session', right_index=True)
main = main.sort_values('start').reset_index(drop=True)
y = main.pop('accuracy_group').to_numpy(dtype=float)
main['world'] = main['world'].astype(str)
inst = main['installation_id'].to_numpy()
u = np.unique(inst); rng = np.random.default_rng(a.seed); ho_inst = set(rng.choice(u, int(0.2 * len(u)), replace=False))
ho = np.array([i in ho_inst for i in inst]); tr = ~ho
X = main.drop(columns=['game_session'])
info = {}
t0 = time.time()
if a.arm.startswith('asof'):
    from tabularaml.generate.relational import Child, asof_features
    evc = ev.drop(columns=['game_session'])
    F = asof_features(X, 'installation_id', 'start', Child('ev', evc, key='installation_id', time='timestamp'))
    X = pd.concat([X, F], axis=1); info['n_asof'] = F.shape[1]
    del evc; gc.collect()
del ev; gc.collect()
Xtr, Xho, ytr, yho = X[tr].reset_index(drop=True), X[ho].reset_index(drop=True), y[tr], y[ho]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
# One random assessment per held-out child, per draw (row positions within Xho).
ho_codes = pd.factorize(Xho['installation_id'].astype(str))[0]
def draw(k):
    r = np.random.default_rng(1000 * a.seed + k).random(len(Xho))
    return np.sort(pd.DataFrame({'c': ho_codes, 'r': r}).sort_values('r').drop_duplicates('c').index.to_numpy())
picks = [draw(k) for k in range(a.draws)]
if a.arm.endswith('forge'):
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='regression', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xho.iloc[picks[0]].reset_index(drop=True))
    info.update(n_added=len(f.new_columns_), gate=f.gate_passed_, group_col=f.group_col_, time_col=f.time_col_, feats=f.new_columns_[:30])
    Xho_raw = Xho
    Xtr, Xho = f.transform_train(Xtr), f.transform(Xho)
fe_t = time.time() - t0
g_tr = Xtr.pop('installation_id').astype(str).to_numpy(); Xho = Xho.drop(columns=['installation_id'])
catmap = {}
for c in Xtr.columns:
    if Xtr[c].dtype == object or isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        cats = pd.Index(pd.unique(Xtr[c].astype(str))); catmap[c] = cats
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xho[c] = pd.Categorical(Xho[c].astype(str), categories=cats)
def prep_ho(Xh):
    Xh = Xh.drop(columns=['installation_id'])[Xtr.columns]
    for c, cats in catmap.items():
        Xh[c] = pd.Categorical(Xh[c].astype(str), categories=cats)
    return Xh
P = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=a.seed)
gu = np.unique(g_tr); va_g = set(np.random.default_rng(a.seed + 1).choice(gu, int(0.15 * len(gu)), replace=False))
va = np.array([g in va_g for g in g_tr])
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 10000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho)
share = np.cumsum(np.bincount(ytr.astype(int), minlength=4) / len(ytr))[:3]
tr_cuts = np.quantile(b.predict(Xtr[va]), share)
kap = lambda yy, pp, cuts: cohen_kappa_score(yy.astype(int), np.digitize(pp, cuts), weights='quadratic')
if a.pick_transform and a.arm.endswith('forge'):
    model = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1)
    pk = [model.predict(prep_ho(f.transform(Xho_raw.iloc[i].reset_index(drop=True)))) for i in picks]
    qwk_draw = [kap(yho[i], q, np.quantile(q, share)) for i, q in zip(picks, pk)]
    qwk_trcut = [kap(yho[i], q, tr_cuts) for i, q in zip(picks, pk)]
else:
    qwk_draw = [kap(yho[i], p[i], np.quantile(p[i], share)) for i in picks]
if not (a.pick_transform and a.arm.endswith('forge')):
    qwk_trcut = [kap(yho[i], p[i], tr_cuts) for i in picks]
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else '') + ('_shufhist' if a.shuffle_history else '') + ('_pick' if a.pick_transform else ''), seed=a.seed, qwk=float(np.mean(qwk_draw)),
           qwk_sd=float(np.std(qwk_draw)), qwk_trcut=float(np.mean(qwk_trcut)), qwk_all=float(kap(yho, p, np.quantile(p, share))), rmse=float(np.sqrt(np.mean((p - yho) ** 2))), best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_ho=len(Xho), n_cols=Xtr.shape[1], **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
