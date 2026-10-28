"""Unseen recent contests ("sealed" set): raw vs FeatureForge blind, held out like each contest's test.

    python sealed_bench.py --contest icr --arm raw|ff --seed 0 [--shuffle] [--repo <worktree>]

Data: DATA/sealed/<contest>/ (see /mnt/project-files/generalization/sealed/MANIFEST.md for sources). Each
contest's ``prep`` returns the main rows, the target, the task, the held-out mask built like the contest's
test (new subjects / new groups / later dates), child tables keyed to the main rows, and its scorer.
``ff`` runs ``scripts/contest_features.py`` of ``--repo`` at its shipped defaults with the held-out rows'
features as the test file (and the child tables as ``--table``); held-out labels reach only the scorer.
``--shuffle`` permutes the training labels (leakage control: the score must land at chance).
Judge (identical for every arm): LightGBM lr 0.05, 63 leaves, early stopping on 15% of the training rows (the
latest 15% for time splits), then 3 seeds refit on all training rows at 1.1x the best iteration.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--contest', required=True); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--budget', default='900'); ap.add_argument('--extra', default='')
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
ap.add_argument('--data', default=os.environ.get('SEALED_DATA', '/home/user/data/new3'))
ap.add_argument('--log', default='/home/user/logs/new3.jsonl')
a = ap.parse_args()
D = Path(a.data) / a.contest


def group_holdout(groups, frac=0.2, seed=0):
    u = pd.unique(pd.Series(groups).astype(str))
    rng = np.random.default_rng(seed)
    ho = set(rng.choice(u, max(1, int(round(frac * len(u)))), replace=False))
    return pd.Series(groups).astype(str).isin(ho).to_numpy()


def strat_holdout(y, frac=0.2, seed=0):
    from sklearn.model_selection import train_test_split
    i = np.arange(len(y)); _, te = train_test_split(i, test_size=frac, random_state=seed, stratify=y)
    m = np.zeros(len(y), bool); m[te] = True; return m


# ------------------------------------------------------------------ scorers
def balanced_logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-(np.mean(np.log(p[y == 1])) + np.mean(np.log(1 - p[y == 0]))) / 2)


def stratified_cindex(groups):
    def f(y, p):  # y = (efs, efs_time); higher p = higher risk
        from itertools import combinations  # noqa
        efs, t = y
        out = []
        for g in np.unique(groups):
            m = groups == g
            out.append(cindex(t[m], -p[m], efs[m]))
        return float(np.mean(out) - np.std(out))
    return f


def cindex(t, s, e):
    """Harrell's C: s is the predicted survival score (higher = lives longer)."""
    o = np.argsort(t); t, s, e = t[o], s[o], e[o]
    conc = disc = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        later = t > t[i]
        conc += np.sum(s[later] > s[i]) + 0.5 * np.sum(s[later] == s[i]); disc += later.sum()
    return conc / disc if disc else 0.5


def jpx_sharpe(dates):
    def f(y, p):
        df = pd.DataFrame(dict(d=dates, y=y, p=p))
        w = np.linspace(2, 1, 200)
        def day(g):
            g = g.sort_values('p', ascending=False)
            if len(g) < 400:
                return np.nan
            return (g.y.to_numpy()[:200] * w).sum() / w.mean() - (g.y.to_numpy()[::-1][:200] * w).sum() / w.mean()
        r = df.groupby('d').apply(day).dropna()
        return float(r.mean() / r.std())
    return f


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - p) ** 2)))


# ------------------------------------------------------------------ contests
def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - p)))


def prep_polymer(target):
    """NeurIPS Open Polymer Prediction 2025: one SMILES string per polymer, five properties each measured on a
    subset. One run per property (polymer_Tg, polymer_FFV, ...), scored by MAE; combine.py forms the contest's wMAE.
    Held out: 20% of polymers at random (the hidden test is new polymers), the same polymers for every property."""
    d = pd.read_csv(D.parent / 'polymer' / 'train.csv')
    ho_all = np.zeros(len(d), bool)
    ho_all[np.random.default_rng(a.seed).permutation(len(d))[: int(round(0.2 * len(d)))]] = True
    m = d[target].notna().to_numpy()
    X = d.loc[m, ['SMILES']].reset_index(drop=True)
    return dict(X=X, y=d.loc[m, target].to_numpy(), task='regression', ho=ho_all[m], score=mae, metric='mae',
                higher=False, hand=polymer_hand)


def polymer_hand(Xtr, Xte, ytr):
    """The public recipe most top notebooks started from: every RDKit 2D descriptor plus Morgan count fingerprints
    (radius 2, 1024 bits) of the repeat unit, the polymer's '*' ends as dummy atoms."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    names = [n for n, _ in Descriptors.descList]
    def feats(smis):
        rows = []
        for s in smis:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                rows.append([np.nan] * (len(names) + 1024)); continue
            dv = []
            for _, f in Descriptors.descList:
                try:
                    dv.append(float(f(mol)))
                except Exception:
                    dv.append(np.nan)
            rows.append(dv + list(gen.GetCountFingerprintAsNumPy(mol).astype(float)))
        F = pd.DataFrame(rows, columns=names + [f'mfp{i}' for i in range(1024)])
        return F.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    Ftr, Fte = feats(Xtr['SMILES']), feats(Xte['SMILES'])
    keep = Ftr.columns[Ftr.nunique() > 1]
    return Ftr[keep], Fte[keep]


def hash_keep(key, pct):
    k = np.asarray(key).astype(np.uint64)
    with np.errstate(over='ignore'):
        return ((k * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)) % np.uint64(100) < np.uint64(pct)


def gini_stability(weeks):
    """Home Credit 2024's metric: mean weekly Gini + 88 x min(0, slope of the weekly Gini) - 0.5 x sd of its residuals."""
    from sklearn.metrics import roc_auc_score
    def f(y, p):
        w = np.unique(weeks); g = []
        for k in w:
            m = weeks == k
            g.append(2 * roc_auc_score(y[m], p[m]) - 1 if 0 < y[m].sum() < m.sum() else np.nan)
        g = np.array(g); ok = np.isfinite(g); x = np.arange(len(w))[ok]; g = g[ok]
        a, b = np.polyfit(x, g, 1)
        return float(g.mean() + 88 * min(0.0, a) - 0.5 * np.std(g - (a * x + b)))
    return f


HC_CHILD = ['applprev_1', 'person_1', 'credit_bureau_b_1', 'tax_registry_a_1', 'tax_registry_b_1', 'tax_registry_c_1',
            'debitcard_1', 'deposit_1', 'other_1']


def _hc_read(name, keep_ids):
    import glob
    fs = sorted(glob.glob(str(D.parent / 'hccrms' / f'train_{name}*.parquet')))
    fs = [f for f in fs if Path(f).stem in (f'train_{name}',) or Path(f).stem.startswith(f'train_{name}_')]
    t = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    t = t[t.case_id.isin(keep_ids)].reset_index(drop=True)
    for c in t.columns:  # D = dates, as the file's own column-name suffixes say
        if c.endswith('D') or c == 'date_decision':
            t[c] = pd.to_datetime(t[c], errors='coerce')
    return t


def prep_hccrms(_=None):
    """Home Credit - Credit Risk Model Stability (2024): depth-0 tables joined on case_id as the main rows, depth-1
    tables as child tables. 25% of cases (hash of case_id); held out: the latest 20% of weeks (WEEK_NUM >= 74), as the
    contest's test followed the training weeks. Score: the contest's gini stability over the held-out weeks."""
    base = pd.read_parquet(D.parent / 'hccrms' / 'train_base.parquet')
    base = base[hash_keep(base.case_id, 25)].reset_index(drop=True)
    ids = set(base.case_id)
    X = base.drop(columns=['MONTH'])
    X['date_decision'] = pd.to_datetime(X['date_decision'])
    for name in ['static_0', 'static_cb_0']:
        t = _hc_read(name, ids)
        X = X.merge(t, on='case_id', how='left')
    X = X.sort_values(['date_decision', 'case_id']).reset_index(drop=True)
    y = X.pop('target').to_numpy()
    ho = (X['WEEK_NUM'] >= 74).to_numpy()
    tables = {}
    for name in HC_CHILD:
        t = _hc_read(name, ids)
        tables[name] = (t, 'case_id', 'num_group1' if 'num_group1' in t.columns else None)
    weeks = X.loc[ho, 'WEEK_NUM'].to_numpy()
    return dict(X=X, y=y, task='binary', ho=ho, score=gini_stability(weeks), metric='gini_stability', higher=True,
                time=True, id='case_id', tables=tables, hand=lambda a_, b_, y_: hc_hand(a_, b_, tables))


def hc_hand(Xtr, Xte, tables):
    """The public starter most top teams extended: every date as days before the decision date, each depth-1
    table aggregated per case (max, mean, min of numbers; max of dates as days before the decision; max of
    each category, as the starter did; row count). Applied to train and held-out rows alike (label-free)."""
    def per_case(t):
        num = [c for c in t.columns if c not in ('case_id', 'num_group1', 'num_group2') and pd.api.types.is_numeric_dtype(t[c])]
        dts = [c for c in t.columns if pd.api.types.is_datetime64_any_dtype(t[c])]
        cat = [c for c in t.columns if c not in num + dts + ['case_id', 'num_group1', 'num_group2']]
        g = t.groupby('case_id')
        parts = [g.size().rename('n')]
        if num:
            parts.append(g[num].agg(['max', 'mean', 'min']).set_axis([f'{a}_{b}' for a, b in g[num].agg(['max', 'mean', 'min']).columns], axis=1))
        if dts:
            parts.append(g[dts].max().add_suffix('_maxdate'))
        if cat:  # the starter took max of the M (category) columns too
            parts.append(g[cat].max().add_suffix('_max'))
        return pd.concat(parts, axis=1)
    aggs = {n: per_case(t).add_prefix(f'{n}__') for n, (t, _, _) in tables.items()}
    def build(X):
        F = X.copy()
        for n, A in aggs.items():
            F = F.merge(A, left_on='case_id', right_index=True, how='left')
        dd = pd.to_datetime(F['date_decision'])
        for c in F.columns:
            if pd.api.types.is_datetime64_any_dtype(F[c]) and c != 'date_decision':
                F[c] = (dd - F[c]).dt.days.astype(float)
        return F
    return build(Xtr), build(Xte)


def pearson(y, p):
    return float(np.corrcoef(y, p)[0, 1])


def _window(n, w):
    """Row range and held-out mask for a time window: w0 trains on the first 80% and holds out the rest; w1 trains on
    the first 60% and holds out the next 20% (the last 20% unused)."""
    end = n if w in (None, 'w0') else int(0.8 * n)
    cut = int(0.8 * n) if w in (None, 'w0') else int(0.6 * n)
    ho = np.zeros(end, bool); ho[cut:] = True
    return end, ho


def prep_drw(w=None):
    """DRW Crypto Market Prediction (2025): 5 order-book / trade columns and 780 anonymized features per minute, label
    = an anonymized future return. Every 4th minute (memory); the timestamp is dropped (the contest's test rows came
    without it). Held out: the latest 20% of minutes, as the test followed the training year (``drw_w1``: the 20%
    before that, training on the first 60%). Score: Pearson."""
    d = pd.read_parquet(D.parent / 'drw' / 'train_every4min.parquet').sort_values('timestamp').reset_index(drop=True)
    end, ho = _window(len(d), w)
    d = d.iloc[:end]
    y = d.pop('label').to_numpy()
    X = d.drop(columns=['timestamp'])
    return dict(X=X, y=y, task='regression', ho=ho, score=pearson, metric='pearson', higher=True, time=True,
                hand=drw_hand)


def prep_hull(w=None):
    """Hull Tactical Market Prediction (2025): one row per trading day, 94 anonymized market, macro, price, sentiment
    and volatility features; label = the next day's S&P 500 return in excess of the risk-free rate. date_id is
    dropped (rows stay in day order), so the rows are a series without a timestamp, as DRW's test came. Held out: the
    latest 20% of days (``hull_w1``: the 20% before that). forward_returns and risk_free_rate are the label's own parts
    and are dropped. Score: Pearson."""
    d = pd.read_csv(D.parent / 'hull' / 'train.csv').sort_values('date_id').reset_index(drop=True)
    end, ho = _window(len(d), w)
    d = d.iloc[:end]
    y = d['market_forward_excess_returns'].to_numpy()
    X = d.drop(columns=['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns'])
    return dict(X=X, y=y, task='regression', ho=ho, score=pearson, metric='pearson', higher=True, time=True)


def drw_hand(Xtr, Xte, ytr):
    """The order-flow features of the public DRW notebooks: book and trade imbalances, ratios and log volume."""
    def f(X):
        F = X.copy(); e = 1e-9
        b, a_, bu, se, v = X.bid_qty, X.ask_qty, X.buy_qty, X.sell_qty, X.volume
        F['bid_ask_imb'] = (b - a_) / (b + a_ + e); F['buy_sell_imb'] = (bu - se) / (bu + se + e)
        F['order_flow'] = bu - se; F['flow_x_book'] = F.buy_sell_imb * F.bid_ask_imb
        F['liquidity'] = b + a_; F['buy_ratio'] = bu / (v + e); F['log_volume'] = np.log1p(v)
        F['bid_ask_ratio'] = b / (a_ + e); F['depth_to_volume'] = (b + a_) / (v + e)
        F['abs_flow_per_vol'] = (bu - se).abs() / (v + e)
        return F
    return f(Xtr), f(Xte)


P = (globals()[f'prep_{a.contest}']() if f'prep_{a.contest}' in globals() else
     globals()[f"prep_{a.contest.split('_')[0]}"](a.contest.split('_', 1)[1]))
if a.arm == 'prep':  # shapes only: no model, no label statistics
    print(dict(contest=a.contest, rows=len(P['X']), cols=P['X'].shape[1], held_out=int(P['ho'].sum()),
               tables={k: v[0].shape for k, v in P.get('tables', {}).items()}))
    sys.exit()
X, y, ho = P['X'], np.asarray(P['y'], float), P['ho']
Xtr, Xte = X[~ho].reset_index(drop=True), X[ho].reset_index(drop=True)
ytr, yte = y[~ho], y[ho]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time()
idc = P.get('id')
if a.arm == 'ff':
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    try:
        Xtr.assign(__y=ytr).to_parquet(tmp / 'train.parquet'); Xte.to_parquet(tmp / 'test.parquet')
        cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
               '--test', str(tmp / 'test.parquet'), '--target', '__y', '--task', P['task'], '--budget', a.budget,
               '--out-dir', str(tmp / 'out')] + (['--id', idc] if idc else [])
        for name, (tab, key, tcol) in P.get('tables', {}).items():
            tab.to_parquet(tmp / f'{name}.parquet'); cmd += ['--table', f'{name}={tmp / (name + ".parquet")}:{key}' + (f':{tcol}' if tcol else '')]
        subprocess.run(cmd + (a.extra.split() if a.extra else []), check=True, cwd=a.repo)
        import resource
        PEAK = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024  # MB, the feature build's peak
        Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['__y'])
        Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Ftr.columns]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    Xtr, Xte = Ftr, Fte
if a.arm == 'hand':
    Xtr, Xte = P['hand'](Xtr, Xte, ytr)
fe_t = time.time() - t0
if idc:
    Xtr, Xte = Xtr.drop(columns=[idc], errors='ignore'), Xte.drop(columns=[idc], errors='ignore')
del X; P.pop('X', None); P.pop('tables', None); P.pop('hand', None)  # the judge alone holds memory from here
import gc; gc.collect()
Xtr, Xte = Xtr.copy(), Xte.copy()
for c in Xtr.columns:
    if pd.api.types.is_float_dtype(Xtr[c]) and Xtr[c].dtype != np.float32:  # judge in float32 (DRW is float32 already)
        Xtr[c] = Xtr[c].astype(np.float32); Xte[c] = Xte[c].astype(np.float32)
for c in Xtr.columns:
    if pd.api.types.is_datetime64_any_dtype(Xtr[c]):
        Xtr[c] = (Xtr[c] - pd.Timestamp('2000-01-01')).dt.days.astype(float); Xte[c] = (Xte[c] - pd.Timestamp('2000-01-01')).dt.days.astype(float)
    elif not pd.api.types.is_numeric_dtype(Xtr[c]):
        _s = lambda v: v.astype(object).where(v.notna(), 'nan').astype(str)  # missing as 'nan' on pandas 2 and 3
        cats = pd.Index(pd.unique(_s(Xtr[c])))
        Xtr[c] = pd.Categorical(_s(Xtr[c]), categories=cats); Xte[c] = pd.Categorical(_s(Xte[c]), categories=cats)
import lightgbm as lgb
obj = 'binary' if P['task'] == 'binary' else 'regression'
Pm = dict(objective=obj, learning_rate=0.05, num_leaves=63, min_child_samples=20 if len(Xtr) < 5000 else 100,
          feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1,
          max_cat_to_onehot=8, cat_smooth=50)
n = len(Xtr)
if P.get('time'):
    fit, va = np.arange(int(0.85 * n)), np.arange(int(0.85 * n), n)
else:
    perm = np.random.default_rng(a.seed + 1).permutation(n); fit, va = np.sort(perm[:int(0.85 * n)]), np.sort(perm[int(0.85 * n):])
b = lgb.train(dict(Pm, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(Pm, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(contest=a.contest, arm=a.arm + ('_shuffled' if a.shuffle else ''), seed=a.seed, metric=P['metric'],
           score=P['score'](yte, p), n_tr=len(Xtr), n_te=len(Xte), n_cols=Xtr.shape[1], best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0),
           peak_mb=globals().get('PEAK'), commit=subprocess.run(['git', '-C', a.repo, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
print('RESULT', json.dumps(res)); open(a.log, 'a').write(json.dumps(res) + '\n')
