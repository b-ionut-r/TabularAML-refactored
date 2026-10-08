"""Late Kaggle submissions: raw columns vs FeatureForge, same judge model, real private leaderboard.

    python scripts/kaggle_late.py prep  --contest wnv
    python scripts/kaggle_late.py feats --contest wnv            # scripts/contest_features.py at shipped defaults
    python scripts/kaggle_late.py fit   --contest wnv --arm raw  # judge -> submission csv
    python scripts/kaggle_late.py fit   --contest wnv --arm forge
    python scripts/kaggle_late.py submit --contest wnv --arm forge

Data come from the Kaggle API (proxy-authenticated). ``prep`` turns each contest's files into one
main train/test table (plus child tables where the contest has them), with no features beyond the
joins the contest's own files define (e.g. West Nile's weather by date). Both arms are scored by the
identical judge: bagged LightGBM with fixed parameters (5 folds, or for time-ordered contests early
stopping on the latest block and 3 refitted seeds), test predictions averaged. Features only: no tuning.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/tmp/claude-0/kg")
REPO = Path(__file__).resolve().parents[1]
# The egress proxy injects the real Kaggle credentials; the CLI only needs some to be set.
KENV = {"KAGGLE_USERNAME": "via-proxy", "KAGGLE_KEY": "via-proxy", **__import__("os").environ}

CONTESTS = {
    "wnv": dict(slug="predict-west-nile-virus", target="WnvPresent", id="Id", task="binary", metric="auc",
                folds="group:year", dates=["Date"]),
    # West Nile plus rows per (date, trap, species) in each file: the file-construction signal, labelled as such.
    "wnvdup": dict(slug="predict-west-nile-virus", target="WnvPresent", id="Id", task="binary", metric="auc",
                   folds="group:year", dates=["Date"]),
    "porto": dict(slug="porto-seguro-safe-driver-prediction", target="target", id="id", task="binary",
                  metric="gini", folds="kfold"),
    "ross": dict(slug="rossmann-store-sales", target="Sales", id="Id", task="regression", metric="rmspe",
                 folds="time", dates=["Date"], log_target=True, closed_zero="Open"),
    # Rossmann again, FeatureForge with only the gate's family fallback (gate_families) on; same prepared files.
    "ross_gf": dict(slug="rossmann-store-sales", target="Sales", id="Id", task="regression", metric="rmspe",
                    folds="time", dates=["Date"], log_target=True, closed_zero="Open", raw="ross"),
    # Rossmann with the forecasting family (branch claude/forecasting-features-9ob67a, worktree): features are
    # built on every training day (closed days too, for recency); the judge then trains on open days with sales.
    "ross_fc": dict(slug="rossmann-store-sales", target="Sales", id="Id", task="regression", metric="rmspe",
                    folds="time", dates=["Date"], log_target=True, closed_zero="Open", raw="ross",
                    repo="/tmp/claude-0/wt_fc", fit_pos=True),
    "ieee": dict(slug="ieee-fraud-detection", target="isFraud", id="TransactionID", task="binary", metric="auc",
                 folds="time"),
    # Full IEEE-CIS (590k x 434) runs FeatureForge out of memory on 15 GB: latest 40% of training rows, whole test.
    "ieee40": dict(slug="ieee-fraud-detection", target="isFraud", id="TransactionID", task="binary", metric="auc",
                   folds="time", raw="ieee"),
    # Forecasting contests: a recent window of training days (both arms), the whole test horizon.
    "m5": dict(slug="m5-forecasting-accuracy", target="sales", id="Id", task="regression", metric="rmse",
               folds="time", dates=["date"], writer="m5", lr=0.1, repo="/tmp/claude-0/wt_fc"),
    "fav": dict(slug="favorita-grocery-sales-forecasting", target="unit_sales", id="id", task="regression",
                metric="rmse_log", folds="time", dates=["date"], log_target=True, lr=0.1, max_rounds=1500, repo="/tmp/claude-0/wt_fc"),
    # Full IEEE-CIS with the memory-lean FeatureForge (branch claude/faster-featureforge-e7d8eu, worktree).
    "ieee_big": dict(slug="ieee-fraud-detection", target="isFraud", id="TransactionID", task="binary", metric="auc",
                     folds="time", raw="ieee", repo="/tmp/claude-0/wt_big"),
    "hc": dict(slug="home-credit-default-risk", target="TARGET", id="SK_ID_CURR", task="binary", metric="auc",
               folds="kfold", child_models=True),
    "sct": dict(slug="santander-customer-transaction-prediction", target="target", id="ID_code", task="binary",
                metric="auc", folds="kfold"),
}


def download(c: str) -> Path:
    d = ROOT / CONTESTS[c].get("raw", c) / "raw"
    if (d / ".done").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    slug = CONTESTS[c]["slug"]
    subprocess.run(["curl", "-sS", "-L", "--fail", "-o", str(d / "all.zip"),
                    f"https://www.kaggle.com/api/v1/competitions/data/download-all/{slug}"], check=True)
    zipfile.ZipFile(d / "all.zip").extractall(d)
    for z in d.glob("*.zip"):
        if z.name != "all.zip":
            try:
                zipfile.ZipFile(z).extractall(d)
            except zipfile.BadZipFile:
                pass
    (d / ".done").touch()
    return d


# ---------------------------------------------------------------- prep (joins only)

def prep_wnv(d: Path):
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    tr = tr.drop(columns=["NumMosquitos"])  # not in the test file
    w = pd.read_csv(d / "weather.csv")
    w = w[w.Station == 1].drop(columns=["Station"])
    for c in w.columns:
        if c not in ("Date", "CodeSum"):
            w[c] = pd.to_numeric(w[c].replace({"T": "0.005", "M": np.nan, "-": np.nan}), errors="coerce")
    return tr.merge(w, on="Date", how="left"), te.merge(w, on="Date", how="left"), {}


def prep_csv(d: Path):
    return pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv"), {}


def prep_ross(d: Path):
    st = pd.read_csv(d / "store.csv")
    tr = pd.read_csv(d / "train.csv", dtype={"StateHoliday": str})
    te = pd.read_csv(d / "test.csv", dtype={"StateHoliday": str})
    # Closed days and zero-sales days are not scored (RMSPE skips zero sales); Customers is not in test.
    tr = tr[(tr.Open == 1) & (tr.Sales > 0)].drop(columns=["Customers"])
    te["Open"] = te["Open"].fillna(1)
    tr = tr.merge(st, on="Store", how="left").sort_values(["Date", "Store"]).reset_index(drop=True)
    te = te.merge(st, on="Store", how="left")
    return tr, te, {}


def prep_ieee(d: Path):
    def side(s):
        t, i = pd.read_csv(d / f"{s}_transaction.csv"), pd.read_csv(d / f"{s}_identity.csv")
        i.columns = [c.replace("-", "_") for c in i.columns]  # test identity spells id-01
        return t.merge(i, on="TransactionID", how="left")
    tr, te = side("train"), side("test")
    return tr.sort_values("TransactionDT").reset_index(drop=True), te, {}


def prep_hc(d: Path):
    tr, te = pd.read_csv(d / "application_train.csv"), pd.read_csv(d / "application_test.csv")
    tables = {n: pd.read_csv(d / f) for n, f in [("bureau", "bureau.csv"), ("prev", "previous_application.csv"),
                                                ("pos", "POS_CASH_balance.csv"), ("inst", "installments_payments.csv"),
                                                ("cc", "credit_card_balance.csv")]}
    return tr, te, tables


def prep_ieee40(d: Path):
    tr, te, t = prep_ieee(d)
    return tr.iloc[int(0.6 * len(tr)):].reset_index(drop=True), te, t


M5_DAYS = 120  # training window: the last 120 days before the 28-day evaluation horizon


def prep_m5(d: Path, n_days: int = M5_DAYS):
    s = pd.read_csv(d / "sales_train_evaluation.csv")
    cal = pd.read_csv(d / "calendar.csv")
    pr = pd.read_csv(d / "sell_prices.csv")
    keys = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    days = [f"d_{i}" for i in range(1942 - n_days, 1942)]
    tr = s[keys + days].melt(id_vars=keys, var_name="d", value_name="sales")
    te = s[keys].merge(pd.DataFrame({"d": [f"d_{i}" for i in range(1942, 1970)]}), how="cross")
    calc = ["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"]
    out = []
    for df in (tr, te):
        df = df.merge(cal[calc], on="d", how="left").merge(pr, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        df["F"] = df["d"].str[2:].astype(int) - 1941
        out.append(df)
    tr, te = out
    tr = tr[tr.sell_price.notna()]  # not on sale yet
    te["Id"] = te["id"] + "|" + te["F"].astype(str)
    tr = tr.drop(columns=["id", "d", "F"]).sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)
    te = te.drop(columns=["id", "d", "F"])
    return tr, te, {}


FAV_DAYS = 14  # training window: the last 14 days before the test's 16 days, as a full store x item grid


def prep_fav(d: Path, n_days: int = FAV_DAYS):
    it, st = pd.read_csv(d / "items.csv"), pd.read_csv(d / "stores.csv")
    oil = pd.read_csv(d / "oil.csv")
    te = pd.read_csv(d / "test.csv").drop(columns=["onpromotion"])
    start = (pd.Timestamp(te.date.min()) - pd.Timedelta(days=n_days)).strftime("%Y-%m-%d")
    parts = []
    for ch in pd.read_csv(d / "train.csv", chunksize=5_000_000, usecols=["date", "store_nbr", "item_nbr", "unit_sales"]):
        ch = ch[ch.date >= start]
        if len(ch):
            parts.append(ch)
    sales = pd.concat(parts, ignore_index=True)
    # The train file lists only rows with sales: rebuild the grid of the test's store x item pairs, zero-filled.
    # onpromotion is dropped: in train it is recorded only on days with sales (the audit's finding).
    pairs = te[["store_nbr", "item_nbr"]].drop_duplicates()
    dates = pd.DataFrame({"date": sorted(sales.date.unique())})
    tr = pairs.merge(dates, how="cross").merge(sales, on=["date", "store_nbr", "item_nbr"], how="left")
    tr["unit_sales"] = tr["unit_sales"].fillna(0).clip(lower=0)
    out = []
    for df in (tr, te):
        df = df.merge(it, on="item_nbr", how="left").merge(st, on="store_nbr", how="left").merge(oil, on="date", how="left")
        out.append(df)
    tr, te = out
    tr = tr.sort_values(["date", "store_nbr", "item_nbr"]).reset_index(drop=True)
    return tr, te, {}


def prep_ross_all(d: Path):
    st = pd.read_csv(d / "store.csv")
    tr = pd.read_csv(d / "train.csv", dtype={"StateHoliday": str}).drop(columns=["Customers"])
    te = pd.read_csv(d / "test.csv", dtype={"StateHoliday": str})
    tr = tr.merge(st, on="Store", how="left").sort_values(["Date", "Store"]).reset_index(drop=True)
    return tr, te.merge(st, on="Store", how="left"), {}


PREP = {"ross_fc": prep_ross_all, "m5": prep_m5, "fav": prep_fav, "ieee40": prep_ieee40, "wnv": prep_wnv, "porto": prep_csv, "sct": prep_csv, "ross": prep_ross, "ieee": prep_ieee, "hc": prep_hc}


def to_compact(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def cmd_prep(c: str):
    d = download(c)
    tr, te, tables = PREP[c](d)
    if CONTESTS[c]["id"] not in tr.columns:  # contest_features.py wants the id column on both sides
        tr.insert(0, CONTESTS[c]["id"], -np.arange(1, len(tr) + 1))
    p = ROOT / c / "prep"
    p.mkdir(parents=True, exist_ok=True)
    to_compact(tr).to_parquet(p / "train.parquet")
    to_compact(te).to_parquet(p / "test.parquet")
    for name, df in tables.items():
        to_compact(df).to_parquet(p / f"{name}.parquet")
    json.dump(sorted(tables), open(p / "tables.json", "w"))
    print(f"prep {c}: train {tr.shape} test {te.shape} tables {[ (k, v.shape) for k, v in tables.items()]}")


FC_HISTORY = {"m5": 450, "fav": 56}  # fav: 112 days ran out of memory (23.6M rows)  # days of labelled history the forecasting family reads (memory-bound)


def cmd_fcfeats(c: str):
    """Forecasting family only (PR #3's ForecastFeatures), for panels too large for FeatureForge's search:
    features are computed from a long labelled history, then the training rows are cut to the raw arm's
    window, so both arms train on the same rows."""
    cfg = CONTESTS[c]
    sys.path.insert(0, cfg["repo"])
    from tabularaml.generate.forecast import ForecastFeatures
    t0 = time.time()
    tr, te, _ = PREP[c](download(c), FC_HISTORY[c])
    if cfg["id"] not in tr.columns:
        tr.insert(0, cfg["id"], -np.arange(1, len(tr) + 1))
    to_compact(tr), to_compact(te)
    y = tr.pop(cfg["target"]).to_numpy()
    raw_tr = pd.read_parquet(ROOT / c / "prep" / "train.parquet", columns=[cfg["dates"][0]])
    first = raw_tr[cfg["dates"][0]].min()
    win = (tr[cfg["dates"][0]] >= first).to_numpy()
    feat_cols = [x for x in tr.columns if x != cfg["id"]]
    ff = ForecastFeatures().fit(tr[feat_cols], np.log1p(y) if cfg.get("log_target") else y, te[feat_cols])
    assert ff.active_, "forecasting family did not switch on"
    Ftr = ff.transform(tr.loc[win, feat_cols].reset_index(drop=True))
    Fte = ff.transform(te[feat_cols])
    out_tr = pd.concat([tr[win].reset_index(drop=True), Ftr.reset_index(drop=True)], axis=1)
    out_tr[cfg["target"]] = y[win]
    out_te = pd.concat([te.reset_index(drop=True), Fte.reset_index(drop=True)], axis=1)
    o = ROOT / c / "fcfeats"
    o.mkdir(parents=True, exist_ok=True)
    out_tr.to_parquet(o / "train_features.parquet")
    out_te.to_parquet(o / "test_features.parquet")
    print(f"forecast family: {Ftr.shape[1]} columns, {win.sum()} of {len(tr)} history rows kept, {time.time() - t0:.0f}s")


def cmd_feats(c: str, budget: float, extra: list[str]):
    cfg, p = CONTESTS[c], ROOT / c / "prep"
    tables = json.load(open(p / "tables.json"))
    repo = Path(cfg.get("repo", REPO))
    cmd = [sys.executable, str(repo / "scripts" / "contest_features.py"), "--train", str(p / "train.parquet"),
           "--test", str(p / "test.parquet"), "--target", cfg["target"], "--id", cfg["id"],
           "--task", cfg["task"], "--budget", str(budget), "--out-dir", str(ROOT / c / "feats")]
    if cfg.get("log_target"):
        cmd.append("--log-target")
    if cfg.get("child_models"):
        cmd.append("--child-models")
    for t in tables:
        cmd += ["--table", f"{t}={p / (t + '.parquet')}"]
    t0 = time.time()
    subprocess.run(cmd + extra, check=True, cwd=repo)
    json.dump(dict(feats_s=round(time.time() - t0)), open(ROOT / c / "feats" / "time.json", "w"))


# ---------------------------------------------------------------- judge

def encode(Xtr: pd.DataFrame, Xte: pd.DataFrame, dates: list[str]):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in Xtr.columns:
        if c in dates:
            for X in (Xtr, Xte):
                X[c] = (pd.to_datetime(X[c].astype(str)) - pd.Timestamp("2000-01-01")).dt.days.astype(np.float32)
        elif not pd.api.types.is_numeric_dtype(Xtr[c]):
            u = pd.Categorical(pd.concat([Xtr[c], Xte[c]]).astype(str)).categories
            Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=u)
            Xte[c] = pd.Categorical(Xte[c].astype(str), categories=u)
    return Xtr, Xte


def judge(Xtr, y, Xte, cfg, groups=None, seeds=(0, 1, 2)):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
    obj = "binary" if cfg["task"] == "binary" else "regression"
    # learning rate 0.1 on the multi-million-row forecasting tables (both arms alike)
    P = dict(objective=obj, learning_rate=cfg.get("lr", 0.03), num_leaves=63, min_child_samples=50, feature_fraction=0.6,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20, num_threads=4, verbose=-1)
    yt = np.log1p(y) if cfg.get("log_target") else y
    oof, pte = np.zeros(len(Xtr)), np.zeros(len(Xte))
    if cfg["folds"] == "time":  # early stop on the latest 15% of rows (rows sorted by time), refit per seed
        n = int(0.85 * len(Xtr))
        b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[:n], yt[:n]), cfg.get("max_rounds", 10000),
                      valid_sets=[lgb.Dataset(Xtr.iloc[n:], yt[n:])], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[n:] = b.predict(Xtr.iloc[n:], num_iteration=b.best_iteration)
        for s in seeds:
            pte += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, yt), int(b.best_iteration * 1.1) + 1).predict(Xte) / len(seeds)
        return pte, oof, np.arange(len(Xtr)) >= n
    if groups is not None:
        splits = GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(Xtr, y, groups)
    elif obj == "binary":
        splits = StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, y)
    else:
        splits = KFold(5, shuffle=True, random_state=0).split(Xtr)
    splits = list(splits)
    for k, (a, v) in enumerate(splits):
        b = lgb.train(dict(P, seed=k), lgb.Dataset(Xtr.iloc[a], yt[a]), 10000,
                      valid_sets=[lgb.Dataset(Xtr.iloc[v], yt[v])], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[v] = b.predict(Xtr.iloc[v], num_iteration=b.best_iteration)
        pte += b.predict(Xte, num_iteration=b.best_iteration) / len(splits)
    return pte, oof, np.ones(len(Xtr), bool)


def score(metric, y, p):
    from sklearn.metrics import roc_auc_score
    if metric == "auc":
        return roc_auc_score(y, p)
    if metric == "gini":
        return 2 * roc_auc_score(y, p) - 1
    if metric == "rmspe":  # y, p on the log1p scale (log_target)
        y, p = np.expm1(y), np.expm1(p)
        return float(np.sqrt(np.mean(((y - p) / y) ** 2)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - p) ** 2)))
    if metric == "rmse_log":  # y, p already on the log1p scale
        return float(np.sqrt(np.mean((y - p) ** 2)))
    raise ValueError(metric)


def cmd_fit(c: str, arm: str, drop: list[str]):
    cfg = CONTESTS[c]
    if arm == "raw":
        tr, te = pd.read_parquet(ROOT / c / "prep" / "train.parquet"), pd.read_parquet(ROOT / c / "prep" / "test.parquet")
    else:
        sub = "fcfeats" if arm == "fc" else "feats"
        tr = pd.read_parquet(ROOT / c / sub / "train_features.parquet")
        te = pd.read_parquet(ROOT / c / sub / "test_features.parquet")
    y = tr.pop(cfg["target"]).to_numpy()
    ids = te[cfg["id"]].to_numpy()
    Xtr = tr.drop(columns=[cfg["id"]], errors="ignore")
    Xte = te.drop(columns=[cfg["id"]], errors="ignore")
    gone = [x for x in Xtr.columns if any(s in x for s in drop)] if drop else []
    Xtr, Xte = Xtr.drop(columns=gone), Xte.drop(columns=gone)
    if gone:
        print(f"dropped {len(gone)} columns: {gone}")
    Xte = Xte[Xtr.columns]
    if cfg.get("fit_pos"):  # score-relevant training rows only (RMSPE skips zero sales)
        keep = y > 0
        Xtr, y = Xtr[keep].reset_index(drop=True), y[keep]
    groups = None
    if cfg["folds"] == "group:year":
        groups = pd.to_datetime(Xtr["Date"].astype(str)).dt.year.to_numpy()
    Xtr, Xte = encode(Xtr, Xte, cfg.get("dates", []))
    t0 = time.time()
    pte, oof, m = judge(Xtr, y, Xte, cfg, groups)
    cv = score(cfg["metric"], np.log1p(y[m]) if cfg.get("log_target") else y[m], oof[m])
    if cfg.get("log_target"):
        pte = np.expm1(pte).clip(0)
    if cfg["task"] == "regression" and y.min() >= 0:  # non-negative target: no negative forecasts
        pte = pte.clip(0)
    if cfg.get("closed_zero"):
        pte[te[cfg["closed_zero"]].to_numpy() == 0] = 0
    tag = arm + ("_" + "-".join(drop) if drop else "")
    out = ROOT / c / "subs"
    out.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({cfg["id"]: ids, cfg["target"]: pte})
    if cfg.get("writer") == "m5":  # wide: id, F1..F28; validation rows (public board) left at 0
        sub[["id", "F"]] = sub["Id"].str.split("|", expand=True)
        w = sub.pivot(index="id", columns="F", values=cfg["target"])
        w = w[[str(k) for k in range(1, 29)]]
        w.columns = [f"F{k}" for k in range(1, 29)]
        v = w.copy() * 0
        v.index = v.index.str.replace("_evaluation", "_validation")
        sub = pd.concat([v, w]).reset_index()
    sub.to_csv(out / f"{tag}.csv", index=False)
    res = dict(contest=c, arm=tag, cv=round(cv, 5), n_feat=Xtr.shape[1], judge_s=round(time.time() - t0))
    print("RESULT", json.dumps(res))
    with open(ROOT / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


def cmd_submit(c: str, arm: str, msg: str):
    f = ROOT / c / "subs" / f"{arm}.csv"
    subprocess.run(["kaggle", "competitions", "submit", "-c", CONTESTS[c]["slug"], "-f", str(f), "-m", msg or arm],
                   check=True, env=KENV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "feats", "fcfeats", "fit", "submit"])
    ap.add_argument("--contest", required=True, choices=sorted(CONTESTS))
    ap.add_argument("--arm", default="raw")
    ap.add_argument("--budget", type=float, default=900)
    ap.add_argument("--drop", nargs="*", default=[], help="drop feature columns whose names contain any of these")
    ap.add_argument("--msg", default="")
    a, extra = ap.parse_known_args()
    if a.cmd == "prep":
        cmd_prep(a.contest)
    elif a.cmd == "fcfeats":
        cmd_fcfeats(a.contest)
    elif a.cmd == "feats":
        cmd_feats(a.contest, a.budget, extra)
    elif a.cmd == "fit":
        cmd_fit(a.contest, a.arm, a.drop)
    else:
        cmd_submit(a.contest, a.arm, a.msg)


if __name__ == "__main__":
    main()
