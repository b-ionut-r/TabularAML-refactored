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
ap.add_argument('--data', default=os.environ.get('SEALED_DATA', '/home/user/data/sealed'))
ap.add_argument('--log', default='/home/user/logs/sealed.jsonl')
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
def prep_icr():
    d = pd.read_csv(D / 'train.csv')
    y = d.pop('Class').to_numpy()
    d = d.drop(columns=['Id'])
    return dict(X=d, y=y, task='binary', ho=strat_holdout(y, 0.2, a.seed), score=balanced_logloss, metric='balanced_logloss',
                higher=False)


def prep_cibmtr():
    d = pd.read_csv(D / 'train.csv')
    efs, t = d.pop('efs').to_numpy(), d.pop('efs_time').to_numpy()
    race = d['race_group'].astype(str).to_numpy()
    ho = strat_holdout(race, 0.2, a.seed)
    # Training target: the Kaplan-Meier survival at the event time, shifted down for censored rows (the usual
    # public transform for this contest); fitted on training rows only. Scored as risk = -prediction.
    tt, ee = t[~ho], efs[~ho]
    u = np.sort(np.unique(tt)); S, s = {}, 1.0
    for v in u:
        n = np.sum(tt >= v); dd = np.sum((tt == v) & (ee == 1)); s *= 1 - dd / n; S[v] = s
    km = np.interp(t, u, [S[v] for v in u])
    y = km - 0.15 * (efs == 0)
    d = d.drop(columns=['ID'])
    sc = stratified_cindex(race[ho])
    return dict(X=d, y=y, task='regression', ho=ho, score=lambda yy, p: sc((efs[ho], t[ho]), -p), metric='strat_cindex',
                higher=True, y_score=None)


def prep_mcts():
    d = pd.read_parquet(D / 'train.parquet')
    y = d.pop('utility_agent1').to_numpy()
    d = d.drop(columns=['Id', 'num_wins_agent1', 'num_draws_agent1', 'num_losses_agent1'])
    d = d.loc[:, d.nunique(dropna=False) > 1]  # constant columns carry nothing (the test file has the same)
    ho = group_holdout(d['GameRulesetName'], 0.2, a.seed)
    return dict(X=d, y=y, task='regression', ho=ho, score=rmse, metric='rmse', higher=False)


def prep_writing_quality():
    sc = pd.read_csv(D / 'train_scores.csv')
    logs = pd.read_parquet(D / 'train_logs.parquet')
    last = logs.sort_values(['id', 'event_id']).groupby('id').agg(n_events=('event_id', 'size'), final_word_count=('word_count', 'last'))
    # Raw main rows: what one row per essay trivially carries (event count, final word count); the keystroke log
    # is the child table (the test file gives every test essay's full log).
    X = sc[['id']].merge(last, left_on='id', right_index=True, how='left')
    y = sc['score'].to_numpy()
    ho = strat_holdout((y * 2).astype(int), 0.2, a.seed)
    return dict(X=X, y=y, task='regression', ho=ho, score=rmse, metric='rmse', higher=False, id='id',
                tables={'logs': (logs, 'id', 'down_time')})


def prep_jpx():
    base = D / 'jpx-tokyo-stock-exchange-prediction'
    tr = pd.read_parquet(base / 'train_files' / 'stock_prices.parquet')
    te = pd.read_parquet(base / 'supplemental_files' / 'stock_prices.parquet')
    d = pd.concat([tr, te], ignore_index=True).drop(columns=['RowId'])
    d = d[d['Target'].notna()].reset_index(drop=True)
    for c in d.columns:
        if c not in ('Date', 'SecuritiesCode', 'SupervisionFlag') and not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d['Date'] = pd.to_datetime(d['Date'])
    d = d.sort_values(['Date', 'SecuritiesCode']).reset_index(drop=True)
    sl = pd.read_csv(base / 'stock_list.csv')[['SecuritiesCode', '33SectorCode', '17SectorCode', 'NewIndexSeriesSizeCode',
                                                'MarketCapitalization', 'IssuedShares']]
    d = d.merge(sl, on='SecuritiesCode', how='left')
    y = d.pop('Target').to_numpy()
    ho = (d['Date'] >= pd.Timestamp('2021-12-06')).to_numpy()
    dates = d.loc[ho, 'Date'].to_numpy()
    return dict(X=d, y=y, task='regression', ho=ho, score=jpx_sharpe(dates), metric='sharpe', higher=True, time=True)


# ---- sealed half: prepared, not to be run until the feature thread declares the final variant ----
def pauc80(y, p):
    """ISIC 2024 metric: partial AUC above 80% TPR, on the contest's [0, 0.2] scale."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(np.abs(np.asarray(y) - 1), -p)  # negatives become positives, as the contest's scorer
    m = fpr <= 0.2
    x = np.r_[fpr[m], 0.2]; t = np.r_[tpr[m], np.interp(0.2, fpr, tpr)]
    return float(0.2 - auc(x, t))


def prep_isic2024_meta():
    d = pd.read_parquet(D / 'train-metadata.parquet')
    y = d.pop('target').to_numpy()
    leak = ['lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm',
            'tbp_lv_dnn_lesion_confidence']
    d = d.drop(columns=[c for c in leak if c in d.columns] + ['isic_id'])
    for c in d.columns:  # columns the CSV conversion left as strings although numeric
        if not pd.api.types.is_numeric_dtype(d[c]):
            v = pd.to_numeric(d[c], errors='coerce')
            if v.notna().sum() >= 0.99 * d[c].notna().sum():
                d[c] = v
    ho = group_holdout(d['patient_id'], 0.2, a.seed)  # new patients, as the contest's test
    return dict(X=d, y=y, task='binary', ho=ho, score=pauc80, metric='pauc80', higher=True)


def qwk_cut(ytr):
    from sklearn.metrics import cohen_kappa_score
    def f(y, p):  # regression output cut at the training class shares (the usual decoding)
        q = np.cumsum(np.bincount(ytr.astype(int), minlength=4))[:-1] / len(ytr)
        th = np.quantile(p, q)
        return float(cohen_kappa_score(y.astype(int), np.digitize(p, th), weights='quadratic'))
    return f


def prep_cmi_piu():
    d = pd.read_csv(D / 'train.csv')
    d = d[d['sii'].notna()].reset_index(drop=True)
    y = d.pop('sii').to_numpy()
    d = d.drop(columns=[c for c in d.columns if c.startswith('PCIAT-')] + ['id'])  # PCIAT items make up sii
    ho = strat_holdout(y.astype(int), 0.2, a.seed)  # new participants
    # Actigraphy (series_train) covers only 100 of the downloaded ids, so it is not used (train.csv only).
    return dict(X=d, y=y, task='regression', ho=ho, score=qwk_cut(y[~ho]), metric='qwk', higher=True)


LG = {'0-4': 0, '5-12': 1, '13-22': 2}


def best_f1(y, p):
    from sklearn.metrics import f1_score
    return float(max(f1_score(y, p > t, average='macro') for t in np.arange(0.4, 0.81, 0.01)))


def prep_student_gameplay():
    ev = pd.read_parquet(D / 'train_sub15pct.parquet')
    lab = pd.read_csv(D / 'train_labels.csv')
    lab['session_id'] = lab.session_id.str.split('_q').str[0].astype(np.int64)
    lab['q'] = pd.read_csv(D / 'train_labels.csv').session_id.str.split('_q').str[1].astype(int)
    lab = lab[lab.session_id.isin(set(ev.session_id))].reset_index(drop=True)
    lab['lg'] = np.where(lab.q <= 3, 0, np.where(lab.q <= 13, 1, 2))
    # The contest served each level group's events before asking its questions: a question may use the events of its
    # own level group and earlier ones. So the child table is keyed by session x level group, cumulatively.
    ev['lg'] = ev.level_group.map(LG)
    parts = [ev[ev.lg <= k].assign(key=lambda e, k=k: e.session_id.astype(str) + f'_{k}') for k in range(3)]
    child = pd.concat(parts, ignore_index=True).drop(columns=['session_id', 'lg', 'level_group'])
    X = pd.DataFrame(dict(key=lab.session_id.astype(str) + '_' + lab.lg.astype(str), q=lab.q, lg=lab.lg))
    y = lab.correct.to_numpy()
    ho = group_holdout(lab.session_id, 0.2, a.seed)  # new sessions
    return dict(X=X, y=y, task='binary', ho=ho, score=best_f1, metric='f1_macro_best_t', higher=True, id='key',
                tables={'events': (child, 'key', 'index')})


def prep_otto():
    """Candidates: the items a session viewed before a random cut; label: the item comes back after the cut
    (any event). Training sessions are the week before the last training week's; held-out sessions the last week
    (the contest's test was the following week). 60k sessions per week, cut points from --seed."""
    tr = pd.read_parquet(D / 'train_sub15pct.parquet')
    end = tr.ts.max(); wk = 7 * 86400
    rng = np.random.default_rng(a.seed)
    out, child = [], []
    for name, lo, hi in [('train', end - 2 * wk, end - wk), ('hold', end - wk, end + 1)]:
        e = tr[(tr.ts > lo) & (tr.ts <= hi)]
        first = e.groupby('session').ts.min(); keep = first[first <= lo + wk].index  # sessions starting in the week
        n = e.groupby('session').size(); keep = n.index[(n >= 2)].intersection(keep)
        keep = rng.choice(keep.to_numpy(), min(60_000, len(keep)), replace=False)
        e = e[e.session.isin(set(keep))].sort_values(['session', 'ts']).reset_index(drop=True)
        pos = e.groupby('session').cumcount(); cnt = e.groupby('session').session.transform('size')
        cut = (rng.random(len(keep)) * 1.0)
        cutpos = pd.Series(np.maximum(1, (cut * pd.Series(cnt.groupby(e.session).first()).to_numpy()).astype(int)), index=np.sort(keep))
        vis = pos < e.session.map(cutpos)
        v, f = e[vis], e[~vis]
        cand = v.groupby(['session', 'aid']).agg(n_seen=('ts', 'size'), last_ts=('ts', 'max'), last_type=('type', 'last')).reset_index()
        cand['since_last'] = cand.session.map(v.groupby('session').ts.max()) - cand.last_ts
        fut = set(zip(f.session, f.aid))
        cand['y'] = [int((s_, a_) in fut) for s_, a_ in zip(cand.session, cand.aid)]
        cand['part'] = name
        out.append(cand.drop(columns=['last_ts'])); child.append(v)
    X = pd.concat(out, ignore_index=True)
    y = X.pop('y').to_numpy(); ho = (X.pop('part') == 'hold').to_numpy()
    from sklearn.metrics import roc_auc_score
    return dict(X=X, y=y, task='binary', ho=ho, score=roc_auc_score, metric='auc', higher=True, time=True,
                tables={'events': (pd.concat(child, ignore_index=True), 'session', 'ts')})


def weighted_r2(w):
    return lambda y, p: float(1 - np.sum(w * (y - p) ** 2) / np.sum(w * y ** 2))


def prep_jane_street():
    """Shard 9 (date_id 1530-1698): train on dates 1600-1668, hold out the last 30 dates; every 4th time_id (memory).
    The previous day's responders at the same time_id and symbol are given, as the contest's lags file did."""
    f = D / 'train.parquet' / 'partition_id=9' / 'part-0.parquet'
    d = pd.read_parquet(f, filters=[('date_id', '>=', 1599)])
    d = d[d.time_id % 4 == 0].reset_index(drop=True)
    resp = [f'responder_{i}' for i in range(9)]
    lag = d[['date_id', 'time_id', 'symbol_id'] + resp].copy(); lag['date_id'] += 1
    d = d.merge(lag.rename(columns={r: r + '_lag_1' for r in resp}), on=['date_id', 'time_id', 'symbol_id'], how='left')
    d = d[d.date_id >= 1600].sort_values(['date_id', 'time_id', 'symbol_id']).reset_index(drop=True)
    y = d['responder_6'].to_numpy(); d = d.drop(columns=resp)
    ho = (d.date_id >= 1669).to_numpy()
    w = d.loc[ho, 'weight'].to_numpy()
    return dict(X=d, y=y, task='regression', ho=ho, score=weighted_r2(w), metric='weighted_r2', higher=True, time=True)


SEALED = {'isic2024_meta', 'cmi_piu', 'student_gameplay', 'otto', 'jane_street'}


if a.contest in SEALED and a.arm != 'prep' and not os.environ.get('UNSEAL'):
    sys.exit(f'{a.contest} is sealed until the final variant is declared (set UNSEAL=1 then)')
P = globals()[f'prep_{a.contest}']()
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
        Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['__y'])
        Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Ftr.columns]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    Xtr, Xte = Ftr, Fte
fe_t = time.time() - t0
if idc:
    Xtr, Xte = Xtr.drop(columns=[idc], errors='ignore'), Xte.drop(columns=[idc], errors='ignore')
Xtr, Xte = Xtr.copy(), Xte.copy()
for c in Xtr.columns:
    if pd.api.types.is_datetime64_any_dtype(Xtr[c]):
        Xtr[c] = (Xtr[c] - pd.Timestamp('2000-01-01')).dt.days.astype(float); Xte[c] = (Xte[c] - pd.Timestamp('2000-01-01')).dt.days.astype(float)
    elif not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
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
           fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res)); open(a.log, 'a').write(json.dumps(res) + '\n')
