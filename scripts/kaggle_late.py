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
    "ieee": dict(slug="ieee-fraud-detection", target="isFraud", id="TransactionID", task="binary", metric="auc",
                 folds="time"),
    "hc": dict(slug="home-credit-default-risk", target="TARGET", id="SK_ID_CURR", task="binary", metric="auc",
               folds="kfold", child_models=True),
    "sct": dict(slug="santander-customer-transaction-prediction", target="target", id="ID_code", task="binary",
                metric="auc", folds="kfold"),
}


def download(c: str) -> Path:
    d = ROOT / c / "raw"
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


PREP = {"wnv": prep_wnv, "porto": prep_csv, "sct": prep_csv, "ross": prep_ross, "ieee": prep_ieee, "hc": prep_hc}


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


def cmd_feats(c: str, budget: float, extra: list[str]):
    cfg, p = CONTESTS[c], ROOT / c / "prep"
    tables = json.load(open(p / "tables.json"))
    cmd = [sys.executable, str(REPO / "scripts" / "contest_features.py"), "--train", str(p / "train.parquet"),
           "--test", str(p / "test.parquet"), "--target", cfg["target"], "--id", cfg["id"],
           "--task", cfg["task"], "--budget", str(budget), "--out-dir", str(ROOT / c / "feats")]
    if cfg.get("log_target"):
        cmd.append("--log-target")
    if cfg.get("child_models"):
        cmd.append("--child-models")
    for t in tables:
        cmd += ["--table", f"{t}={p / (t + '.parquet')}"]
    t0 = time.time()
    subprocess.run(cmd + extra, check=True, cwd=REPO)
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
    P = dict(objective=obj, learning_rate=0.03, num_leaves=63, min_child_samples=50, feature_fraction=0.6,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20, num_threads=4, verbose=-1)
    yt = np.log1p(y) if cfg.get("log_target") else y
    oof, pte = np.zeros(len(Xtr)), np.zeros(len(Xte))
    if cfg["folds"] == "time":  # early stop on the latest 15% of rows (rows sorted by time), refit per seed
        n = int(0.85 * len(Xtr))
        b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[:n], yt[:n]), 10000,
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
    raise ValueError(metric)


def cmd_fit(c: str, arm: str, drop: list[str]):
    cfg = CONTESTS[c]
    if arm == "raw":
        tr, te = pd.read_parquet(ROOT / c / "prep" / "train.parquet"), pd.read_parquet(ROOT / c / "prep" / "test.parquet")
    else:
        tr = pd.read_parquet(ROOT / c / "feats" / "train_features.parquet")
        te = pd.read_parquet(ROOT / c / "feats" / "test_features.parquet")
    y = tr.pop(cfg["target"]).to_numpy()
    ids = te[cfg["id"]].to_numpy()
    Xtr = tr.drop(columns=[cfg["id"]], errors="ignore")
    Xte = te.drop(columns=[cfg["id"]], errors="ignore")
    gone = [x for x in Xtr.columns if any(s in x for s in drop)] if drop else []
    Xtr, Xte = Xtr.drop(columns=gone), Xte.drop(columns=gone)
    if gone:
        print(f"dropped {len(gone)} columns: {gone}")
    Xte = Xte[Xtr.columns]
    groups = None
    if cfg["folds"] == "group:year":
        groups = pd.to_datetime(Xtr["Date"].astype(str)).dt.year.to_numpy()
    Xtr, Xte = encode(Xtr, Xte, cfg.get("dates", []))
    t0 = time.time()
    pte, oof, m = judge(Xtr, y, Xte, cfg, groups)
    cv = score(cfg["metric"], np.log1p(y[m]) if cfg.get("log_target") else y[m], oof[m])
    if cfg.get("log_target"):
        pte = np.expm1(pte).clip(0)
    if cfg.get("closed_zero"):
        pte[te[cfg["closed_zero"]].to_numpy() == 0] = 0
    tag = arm + ("_" + "-".join(drop) if drop else "")
    out = ROOT / c / "subs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({cfg["id"]: ids, cfg["target"]: pte}).to_csv(out / f"{tag}.csv", index=False)
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
    ap.add_argument("cmd", choices=["prep", "feats", "fit", "submit"])
    ap.add_argument("--contest", required=True, choices=sorted(CONTESTS))
    ap.add_argument("--arm", default="raw")
    ap.add_argument("--budget", type=float, default=900)
    ap.add_argument("--drop", nargs="*", default=[], help="drop feature columns whose names contain any of these")
    ap.add_argument("--msg", default="")
    a, extra = ap.parse_known_args()
    if a.cmd == "prep":
        cmd_prep(a.contest)
    elif a.cmd == "feats":
        cmd_feats(a.contest, a.budget, extra)
    elif a.cmd == "fit":
        cmd_fit(a.contest, a.arm, a.drop)
    else:
        cmd_submit(a.contest, a.arm, a.msg)


if __name__ == "__main__":
    main()
