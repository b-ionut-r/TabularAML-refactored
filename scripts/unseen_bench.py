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
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--perm', type=int, default=0); ap.add_argument('--budget', default='900'); ap.add_argument('--extra', default='')
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


def prep_svp():
    # Santander Value Prediction (2018): 4,459 rows x 4,991 anonymised, mostly-zero columns; the test is a random
    # draw of other customers. Scored as RMSLE: fit log1p(target), RMSE on that scale.
    d = pd.read_csv(D / 'train.csv')
    y = np.log1p(d.pop('target').to_numpy())
    d = d.drop(columns=['ID'])
    d = d.loc[:, d.nunique(dropna=False) > 1]
    ho = np.zeros(len(d), bool); ho[np.random.default_rng(a.seed).permutation(len(d))[:int(0.2 * len(d))]] = True
    return dict(X=d, y=y, task='regression', ho=ho, score=rmse, metric='rmsle', higher=False)


def prep_talkingdata():
    # TalkingData AdTracking (2018): click log, every click of 2% of IPs (tdata_prep.py). The real test is the next
    # day's hours 4, 5, 9, 10, 13, 14 (UTC) about 12 h after train ends; here train = clicks before 2017-11-08 16:00,
    # holdout = 2017-11-09 at those hours, the rows between are dropped. attributed_time is never loaded (leak).
    d = pd.read_parquet(D / 'clicks.parquet').sort_values('click_time', kind='stable').reset_index(drop=True)
    t = d['click_time']
    tr = t < pd.Timestamp('2017-11-08 16:00')
    te = (t.dt.normalize() == pd.Timestamp('2017-11-09')) & t.dt.hour.isin([4, 5, 9, 10, 13, 14])
    d = d[tr | te].reset_index(drop=True); ho = te[tr | te].to_numpy()
    y = d.pop('is_attributed').to_numpy()
    for c in ['ip', 'app', 'device', 'os', 'channel']:
        d[c] = d[c].astype('int64')
    from sklearn.metrics import roc_auc_score
    return dict(X=d, y=y, task='binary', ho=ho, score=roc_auc_score, metric='auc', higher=True, time=True)



def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def prep_allstate():
    # Allstate Claims Severity (2016): 188k claims, 116 categorical + 14 continuous columns, heavy right-skewed
    # loss scored by MAE; the test is a random draw of other claims. Fit log(loss + 200), score MAE on the loss.
    d = pd.read_csv(D / 'train.csv')
    y = np.log(d.pop('loss').to_numpy() + 200)
    d = d.drop(columns=['id'])
    ho = np.zeros(len(d), bool); ho[np.random.default_rng(a.seed).permutation(len(d))[:int(0.2 * len(d))]] = True
    return dict(X=d, y=y, task='regression', ho=ho, score=lambda yt, p: float(np.mean(np.abs(np.exp(yt) - np.exp(p)))),
                metric='mae', higher=False)


def prep_bnp():
    # BNP Paribas Cardif (2016): 114k claims, 131 anonymised columns (19 categorical, one with 18k levels), a third
    # of the values missing; binary, log loss; the test is a random draw of other claims.
    d = pd.read_csv(D / 'train.csv')
    y = d.pop('target').to_numpy()
    d = d.drop(columns=['ID'])
    return dict(X=d, y=y, task='binary', ho=strat_holdout(y, 0.2, a.seed), score=logloss, metric='logloss', higher=False)


def prep_homesite():
    # Homesite Quote Conversion (2015-16): 261k quotes, 297 mixed columns (coded categoricals, personal, property
    # and geographic fields) and the quote date; binary, AUC; the test is other quotes from the same period.
    d = pd.read_csv(D / 'train.csv', low_memory=False)
    y = d.pop('QuoteConversion_Flag').to_numpy()
    d = d.drop(columns=['QuoteNumber'])
    d['Original_Quote_Date'] = pd.to_datetime(d['Original_Quote_Date'])
    return dict(X=d, y=y, task='binary', ho=strat_holdout(y, 0.2, a.seed), score=auc, metric='auc', higher=True)


def prep_ashrae():
    # ASHRAE Great Energy Predictor III (2019): hourly meter readings of 1,449 buildings over 2016, with building
    # metadata and site weather tables; the real test is the next two years of the same buildings, scored RMSLE.
    # A fixed 10% of buildings (145, seed 0, joined to both tables by subset10.parquet); held out: Sep-Dec 2016.
    d = pd.read_parquet(D / 'subset10.parquet')
    y = np.log1p(d.pop('meter_reading').to_numpy())
    ho = (d['timestamp'] >= pd.Timestamp('2016-09-01')).to_numpy()
    return dict(X=d, y=y, task='regression', ho=ho, score=rmse, metric='rmsle', higher=False, time=True)


def prep_storesales():
    # Store Sales - Time Series Forecasting (Favorita successor, 2021-): daily sales of 33 product families in
    # 54 stores, 2013-2017, with stores, oil price and national holiday tables (joined by joined.parquet; the
    # transactions table is left out, as it does not cover the test days). The real test is the next 16 days,
    # scored RMSLE; held out here: the last 16 days of train (2017-07-31 .. 2017-08-15).
    d = pd.read_parquet(D / 'joined.parquet')
    y = np.log1p(d.pop('sales').to_numpy())
    ho = (d['date'] >= pd.Timestamp('2017-07-31')).to_numpy()
    return dict(X=d, y=y, task='regression', ho=ho, score=rmse, metric='rmsle', higher=False, time=True)


def prep_pfs():
    # Predict Future Sales (1C, Coursera contest): monthly sales of each item in each shop; the test is next
    # month's count for every shop x item pair of the test grid, clipped to [0, 20], scored RMSE. Built like
    # that: each month's rows are every (shop with sales that month) x (item with sales that month), months
    # 20..33 (the usual kernels' grid, kept to 14 months for memory); held out: month 33 (Oct 2015).
    s = pd.read_csv(D / 'raw' / 'sales_train.csv')
    s = s[s.date_block_num >= 20]
    cnt = s.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)['item_cnt_day'].sum()
    parts = []
    for m, g in s.groupby('date_block_num'):
        sh, it = np.unique(g.shop_id), np.unique(g.item_id)
        parts.append(pd.DataFrame({'date_block_num': m, 'shop_id': np.repeat(sh, len(it)), 'item_id': np.tile(it, len(sh))}))
    d = pd.concat(parts, ignore_index=True).merge(cnt, how='left', on=['date_block_num', 'shop_id', 'item_id'])
    items = pd.read_csv(D / 'raw' / 'items.csv')[['item_id', 'item_category_id']]
    d = d.merge(items, how='left', on='item_id')
    d['date'] = pd.to_datetime({'year': 2013 + d.date_block_num // 12, 'month': d.date_block_num % 12 + 1, 'day': 1})
    d = d.sort_values(['date_block_num', 'shop_id', 'item_id'], kind='stable').reset_index(drop=True)
    y = d.pop('item_cnt_day').fillna(0).clip(0, 20).to_numpy()
    ho = (d['date_block_num'] == 33).to_numpy()
    return dict(X=d, y=y, task='regression', ho=ho, score=rmse, metric='rmse_clip20', higher=False, time=True)


def prep_rohlik():
    # Rohlik Sales Forecasting Challenge v2 (2024-25): daily sales of each product in each of 7 warehouses, with
    # the inventory (product, name, 4 category levels) and calendar (holidays, closures per warehouse) tables.
    # The test is the next 14 days, scored weighted MAE (test_weights per unique_id). Held out: the last 14 days
    # of train (2024-05-20 .. 2024-06-02); availability is left out (absent from the real test file).
    d = pd.read_parquet(D / 'raw' / 'sales_train.parquet').drop(columns=['availability'])
    d = d.merge(pd.read_parquet(D / 'raw' / 'inventory.parquet'), how='left', on=['unique_id', 'warehouse'])
    d = d.merge(pd.read_parquet(D / 'raw' / 'calendar.parquet'), how='left', on=['date', 'warehouse'])
    d['date'] = pd.to_datetime(d['date'])
    d = d[d['sales'].notna()].sort_values(['date', 'unique_id'], kind='stable').reset_index(drop=True)
    w = d['unique_id'].map(pd.read_parquet(D / 'raw' / 'test_weights.parquet').set_index('unique_id')['weight']).fillna(1.0)
    y = np.log1p(d.pop('sales').to_numpy())
    ho = (d['date'] >= pd.Timestamp('2024-05-20')).to_numpy(); wh = w.to_numpy()[ho]
    wmae = lambda yy, p: float(np.sum(wh * np.abs(np.expm1(yy) - np.expm1(p))) / np.sum(wh))
    return dict(X=d, y=y, task='regression', ho=ho, score=wmae, metric='wmae', higher=False, time=True)


def prep_gefw():
    # GEFCom2014 wind track (IEEE, prizes): hourly power of 10 wind farms (normalized 0..1) with the farms'
    # wind forecasts (u/v at 10 m and 100 m). Each task gave history to the end of a month and scored the next
    # month with its forecasts known. Final task (15) history, held out: its last month (Nov 2013). The contest
    # scored quantiles (pinball); here the point forecast, RMSE.
    d = pd.concat([pd.read_csv(D / 'raw' / f'Task15_W_Zone{z}.csv') for z in range(1, 11)], ignore_index=True)
    d['TIMESTAMP'] = pd.to_datetime(d['TIMESTAMP'], format='%Y%m%d %H:%M')
    d = d[d['TARGETVAR'].notna()].sort_values(['TIMESTAMP', 'ZONEID'], kind='stable').reset_index(drop=True)
    y = d.pop('TARGETVAR').to_numpy()
    ho = (d['TIMESTAMP'] > pd.Timestamp('2013-11-01')).to_numpy()
    return dict(X=d, y=y, task='regression', ho=ho, score=rmse, metric='rmse', higher=False, time=True)


def _stormy_rows():
    d = pd.read_csv(D.parent / 'stormy' / 'raw' / 'train.csv', parse_dates=['date'])
    d = d.merge(pd.read_csv(D.parent / 'stormy' / 'raw' / 'key.csv'), how='left', on='store_nbr')
    w = pd.read_csv(D.parent / 'stormy' / 'raw' / 'weather.csv', parse_dates=['date'], dtype=str).assign(
        station_nbr=lambda t: t.station_nbr.astype(int), date=lambda t: pd.to_datetime(t.date))
    d = d.merge(w, how='left', on=['station_nbr', 'date'])
    return d.sort_values(['date', 'store_nbr', 'item_nbr'], kind='stable').reset_index(drop=True), w


def _stormy_finish(d, ho):
    # Store x item pairs that never sell in the training rows are dropped (their test answer is 0; the usual prep).
    keep = d.groupby(['store_nbr', 'item_nbr'])['units'].transform(lambda u: u[~ho[u.index]].max() > 0).to_numpy()
    d, ho = d[keep].reset_index(drop=True), ho[keep]
    y = np.log1p(d.pop('units').to_numpy())
    return dict(X=d, y=y, task='regression', ho=ho, score=rmse, metric='rmsle', higher=False, time=True)


def prep_stormy():
    # Walmart Recruiting II: Sales in Stormy Weather (2015): daily units of 111 weather-sensitive items in 45
    # stores, with the stores' weather station table (raw strings, as shipped). The real test is the days within
    # 3 of a weather event (snowfall >= 2 in or rain >= 1 in at the store's station), interleaved with training
    # days; RMSLE. Held out that way: a random half (seed) of the training period's events, every row of the
    # station's stores within 3 days of the event.
    d, w = _stormy_rows()
    num = lambda c: pd.to_numeric(w[c].str.strip().replace({'T': '0.001', 'M': None}), errors='coerce')
    ev = w.loc[(num('snowfall') >= 2) | (num('preciptotal') >= 1), ['station_nbr', 'date']].reset_index(drop=True)
    rng = np.random.default_rng(a.seed); ev = ev[rng.random(len(ev)) < 0.5]
    win = pd.concat([ev.assign(date=ev.date + pd.Timedelta(days=k)) for k in range(-3, 4)]).drop_duplicates()
    ho = d[['station_nbr', 'date']].merge(win.assign(_h=1), how='left', on=['station_nbr', 'date'])['_h'].notna().to_numpy()
    return _stormy_finish(d, ho)


def prep_stormy_last():
    # Same rows, held out as a forecast instead (NOT how this contest tested): the last 28 days of train
    # (2014-10-04 .. 2014-10-31). Only to exercise the forecasting family's key search on a key whose low
    # cardinality part (store_nbr, 45 levels) is outranked by the joined weather columns.
    d, w = _stormy_rows()
    ho = (d['date'] >= pd.Timestamp('2014-10-04')).to_numpy()
    return _stormy_finish(d, ho)


def hand_svp(X):
    # Public kernels' row statistics over the non-zero cells (count of non-zeros, log-scale mean/max/min/std/median, sum).
    v = X.to_numpy(float); nz = v != 0; lv = np.where(nz, np.log1p(np.abs(v)), np.nan)
    import warnings; warnings.simplefilter('ignore')
    H = pd.DataFrame({'nz_count': nz.sum(1), 'nz_logmean': np.nanmean(lv, 1), 'nz_logmax': np.nanmax(lv, 1),
                      'nz_logmin': np.nanmin(lv, 1), 'nz_logstd': np.nanstd(lv, 1), 'nz_logmed': np.nanmedian(lv, 1),
                      'logsum': np.log1p(np.abs(v).sum(1))}, index=X.index)
    return pd.concat([X, H], axis=1)


def hand_talkingdata(X):
    # Winners' staples: group counts, unique counts and time to the next click of the same user/app (on all rows,
    # unlabeled holdout included, as the real test was used).
    X = X.copy(); t = X['click_time']; X['hour'] = t.dt.hour; X['day'] = t.dt.day
    ts = t.astype('int64') // 10**9
    for g in [['ip', 'day', 'hour'], ['ip', 'app'], ['ip', 'app', 'os'], ['ip', 'device'], ['app', 'channel'], ['ip']]:
        X['cnt_' + '_'.join(g)] = X.groupby(g)['app'].transform('size').astype(float)
    for g, c in [(['ip'], 'channel'), (['ip'], 'app'), (['ip', 'day'], 'hour'), (['ip', 'app'], 'os'), (['ip'], 'device'),
                 (['app'], 'channel'), (['ip', 'device', 'os'], 'app')]:
        X['nu_' + '_'.join(g) + '_' + c] = X.groupby(g)[c].transform('nunique').astype(float)
    for g in [['ip', 'app', 'device', 'os'], ['ip', 'os', 'device'], ['ip', 'app', 'device', 'os', 'channel']]:
        X['next_' + '_'.join(g)] = (X.groupby(g)['click_time'].shift(-1).astype('int64') // 10**9 - ts).where(
            X.groupby(g)['click_time'].shift(-1).notna()).astype(float)
        X['prev_' + '_'.join(g)] = (ts - X.groupby(g)['click_time'].shift(1).astype('int64') // 10**9).where(
            X.groupby(g)['click_time'].shift(1).notna()).astype(float)
    X['cum_ip_device_os'] = X.groupby(['ip', 'device', 'os']).cumcount().astype(float)
    return X


# Held-back validation cases (kept out of this repo until each is run blind): preps live in heldback_preps.py
# next to this script when present; their FF arm runs only with HELDBACK_FF=1.
HELDBACK = set()
_hb = Path(__file__).with_name('heldback_preps.py')
if _hb.exists():
    exec(_hb.read_text())

SEALED = {'isic2024_meta', 'cmi_piu', 'student_gameplay', 'otto', 'jane_street'}


if a.contest in SEALED and a.arm != 'prep' and not os.environ.get('UNSEAL'):
    sys.exit(f'{a.contest} is sealed until the final variant is declared (set UNSEAL=1 then)')
if a.contest in HELDBACK and a.arm == 'ff' and not os.environ.get('HELDBACK_FF'):
    sys.exit(f'{a.contest} is held back for a family hash: set HELDBACK_FF=1 to run FeatureForge on it')
P = globals()[f'prep_{a.contest}']()
if a.arm == 'prep':  # shapes only: no model, no label statistics
    print(dict(contest=a.contest, rows=len(P['X']), cols=P['X'].shape[1], held_out=int(P['ho'].sum()),
               tables={k: v[0].shape for k, v in P.get('tables', {}).items()}))
    sys.exit()
X, y, ho = P['X'], np.asarray(P['y'], float), P['ho']
Xtr, Xte = X[~ho].reset_index(drop=True), X[ho].reset_index(drop=True)
ytr, yte = y[~ho], y[ho]
if a.shuffle:
    ytr = np.random.default_rng(a.perm).permutation(ytr)  # read several perms (3-5) on rare-positive contests
t0 = time.time()
idc = P.get('id')
if a.arm == 'hand':
    _hf = globals()[f'hand_{a.contest}']; del X, P['X']
    if getattr(_hf, 'rowwise', False):  # per-row features: no joint copy of a wide frame
        Xtr, Xte = _hf(Xtr), _hf(Xte)
    else:
        _H = _hf(pd.concat([Xtr, Xte], ignore_index=True))
        Xtr, Xte = _H.iloc[:len(Xtr)].reset_index(drop=True), _H.iloc[len(Xtr):].reset_index(drop=True); del _H
if a.arm == 'ff':
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    try:
        Xtr.assign(__y=ytr).to_parquet(tmp / 'train.parquet'); Xte.to_parquet(tmp / 'test.parquet')
        cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
               '--test', str(tmp / 'test.parquet'), '--target', '__y', '--task', P['task'], '--budget', a.budget,
               '--out-dir', str(tmp / 'out')] + (['--id', idc] if idc else [])
        for name, (tab, key, tcol) in P.get('tables', {}).items():
            tab.to_parquet(tmp / f'{name}.parquet'); cmd += ['--table', f'{name}={tmp / (name + ".parquet")}:{key}' + (f':{tcol}' if tcol else '')]
        # The frames are on disk now: free this process's copies so FeatureForge gets the box.
        del X, Xtr, Xte; P['X'] = None; P.pop('tables', None); tab = None; import gc; gc.collect()
        try: import ctypes; ctypes.CDLL('libc.so.6').malloc_trim(0)  # hand freed pages back to the OS
        except OSError: pass
        subprocess.run(cmd + (a.extra.split() if a.extra else []), check=True, cwd=a.repo)
        import resource; ff_peak_gb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)
        Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['__y'])
        Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Ftr.columns]
    finally:
        if os.environ.get('KEEP_OUT'): shutil.copytree(tmp / 'out', os.environ['KEEP_OUT'], dirs_exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
    Xtr, Xte = Ftr, Fte
fe_t = time.time() - t0
if os.environ.get('BUILD_ONLY'):  # outputs kept via KEEP_OUT; no judge
    print('BUILT', a.contest, round(fe_t), globals().get('ff_peak_gb')); sys.exit()
if idc:
    Xtr, Xte = Xtr.drop(columns=[idc], errors='ignore'), Xte.drop(columns=[idc], errors='ignore')
Xtr, Xte = Xtr.copy(), Xte.copy()
for c in Xtr.columns:
    if pd.api.types.is_datetime64_any_dtype(Xtr[c]):
        # fractional days: time of day kept (whole days for date-only columns, as before)
        Xtr[c] = (Xtr[c] - pd.Timestamp('2000-01-01')).dt.total_seconds() / 86400; Xte[c] = (Xte[c] - pd.Timestamp('2000-01-01')).dt.total_seconds() / 86400
    elif not pd.api.types.is_numeric_dtype(Xtr[c]):
        # object -> str per value: astype(str) on a category of long rule texts makes a fixed-width array (118 GiB on MCTS)
        _s = lambda v: v.astype(object).map(str)
        cats = pd.Index(pd.unique(_s(Xtr[c])))
        Xtr[c] = pd.Categorical(_s(Xtr[c]), categories=cats); Xte[c] = pd.Categorical(_s(Xte[c]), categories=cats)
if os.environ.get('CAST32'):  # check: the judge on float32 numerics (as contest_features writes them)
    for c in Xtr.columns:
        if Xtr[c].dtype == np.float64:
            Xtr[c], Xte[c] = Xtr[c].astype(np.float32), Xte[c].astype(np.float32)
import lightgbm as lgb
obj = {'binary': 'binary', 'multiclass': 'multiclass'}.get(P['task'], 'regression')
Pm = dict(objective=obj, **({'num_class': P['n_class']} if obj == 'multiclass' else {}), learning_rate=0.05, num_leaves=63, min_child_samples=20 if len(Xtr) < 5000 else 100,
          feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1,
          max_cat_to_onehot=8, cat_smooth=50)
n = len(Xtr)
if P.get('time'):
    fit, va = np.arange(int(0.85 * n)), np.arange(int(0.85 * n), n)
else:
    perm = np.random.default_rng(a.seed + 1).permutation(n); fit, va = np.sort(perm[:int(0.85 * n)]), np.sort(perm[int(0.85 * n):])
b = lgb.train(dict(Pm, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros((len(Xte), P['n_class']) if obj == 'multiclass' else len(Xte))
for s in range(3):
    p += lgb.train(dict(Pm, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(contest=a.contest, arm=a.arm + (f'_shuffled{a.perm or ""}' if a.shuffle else ''), seed=a.seed, metric=P['metric'],
           score=P['score'](yte, p), n_tr=len(Xtr), n_te=len(Xte), n_cols=Xtr.shape[1], best_it=b.best_iteration,
           extra=a.extra, fe_s=round(fe_t), total_s=round(time.time() - t0), ff_peak_gb=globals().get('ff_peak_gb'))
print('RESULT', json.dumps(res)); open(a.log, 'a').write(json.dumps(res) + '\n')
