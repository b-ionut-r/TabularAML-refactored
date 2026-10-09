"""IEEE-CIS Fraud Detection: the winners' hand-made features (Chris Deotte / Konstantin Yakovlev's
1st-place recipe) on the same time-ordered holdout and judge as ``ieee_fraud.py``.

    python scripts/ieee_hand.py --frac 1.0 --arm hand

Arms: ``raw``; ``hand`` = client id uid = card1 + addr1 + (transaction day - D1), D columns
re-based to the day they point to, cents, frequency encodings, and per-uid aggregates of
TransactionAmt, D, C, M, dist and V columns (mean / std) and of categoricals (nunique); uid
itself is dropped, as the winners did (the test period has new clients). Label-free statistics
are computed over training and held-out rows together, as the winners computed them over
train + test. ``--shuffle`` permutes the training labels.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ieee_fraud import judge, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--arm", default="hand")
ap.add_argument("--shuffle", action="store_true"); ap.add_argument("--log", default="ieee_hand.jsonl")
ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "ieee_fraud")
ap.add_argument("--forge-kw", default=None, help="also run FeatureForge on top (JSON kwargs)")
ap.add_argument("--budget", type=float, default=900)
ap.add_argument("--ablate", default="", help="comma list: noagg, nofe, keepids, nodnorm")
a = ap.parse_args()
df = load(a.cache)
df = df.iloc[int(len(df) * (1 - a.frac)):].reset_index(drop=True)
y = df.pop("isFraud"); df = df.drop(columns=["TransactionID"])
n = int(0.8 * len(df))
t0 = time.time()
abl = set(filter(None, a.ablate.split(",")))
if a.arm == "hand":
    X = df.copy()
    day = np.floor(X.TransactionDT / 86400)
    for c in ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D10", "D11", "D12", "D13", "D14", "D15"]:
        X[c + "n"] = day - X[c]
    X["cents"] = (X.TransactionAmt - np.floor(X.TransactionAmt)).astype(np.float32)
    s = lambda c: X[c].astype(str)
    X["uid"] = s("card1") + "_" + s("addr1") + "_" + X["D1n"].astype(str)
    X["card1_addr1"] = s("card1") + "_" + s("addr1")
    X["card1_addr1_P"] = X["card1_addr1"] + "_" + s("P_emaildomain")
    for c in [] if "nofe" in abl else ["addr1", "card1", "card2", "card3", "P_emaildomain", "R_emaildomain", "card1_addr1",
              "card1_addr1_P", "uid", "cents", "dist1", "D1n", "D11n"]:
        X[c + "_FE"] = X[c].map(X[c].value_counts(dropna=False)).astype(np.float32)
    g = X.groupby("uid")
    nums = (["TransactionAmt", "dist1", "cents"] + [f"D{i}n" for i in (1, 4, 10, 11, 15)] + ["D9", "D3", "D5", "D6"]
            + [f"C{i}" for i in range(1, 15)] + [f"V{i}" for i in (127, 130, 136, 258, 294, 307, 308, 310, 312, 313, 314, 315, 317)])
    for c in [f"M{i}" for i in range(1, 10)]:
        X[c] = X[c].map({"T": 1.0, "F": 0.0, "M0": 0.0, "M1": 1.0, "M2": 2.0}).astype(np.float32)
    nums += [f"M{i}" for i in range(1, 10)]
    new = {}
    for c in [] if "noagg" in abl else nums:
        new[f"{c}_uid_mean"] = g[c].transform("mean").astype(np.float32)
        new[f"{c}_uid_std"] = g[c].transform("std").astype(np.float32)
    for c in [] if "noagg" in abl else ["P_emaildomain", "R_emaildomain", "dist1", "id_02", "id_13", "id_17", "id_19", "id_20", "card4", "addr2", "cents", "C13", "V314"]:
        new[f"{c}_uid_nunique"] = g[c].transform("nunique").astype(np.float32)
    X = pd.concat([X, pd.DataFrame(new)], axis=1)
    X = X.drop(columns=["uid", "card1_addr1", "card1_addr1_P"] + ([] if "keepids" in abl else ["D1n", "TransactionDT"]))
    if "nodnorm" in abl:
        X = X.drop(columns=[c for c in X.columns if c.endswith("n") and c[:-1] in df.columns and c.startswith("D") and c != "D1n"])
    df = X
if a.arm == "profile":
    # The generic block alone, with the key FeatureForge discovers (card1, addr1, day - D1) and
    # numerics in a raw model's importance order: how much of the hand gain it carries.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tabularaml.generate.profile import ClientProfile
    import lightgbm as lgb
    X = df.copy()
    X["anchor"] = np.floor(X.TransactionDT / 86400) - X.D1
    Xm = X.iloc[:n].copy()
    for c in Xm.columns:
        if not pd.api.types.is_numeric_dtype(Xm[c]):
            Xm[c] = Xm[c].astype("category")
    b = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, verbose=-1, num_threads=4, feature_fraction=0.5),
                  lgb.Dataset(Xm, y.iloc[:n]), 200)
    imp = pd.Series(b.feature_importance("gain"), index=Xm.columns).sort_values(ascending=False)
    num = [c for c in imp.index if pd.api.types.is_numeric_dtype(X[c]) and c not in ("card1", "addr1", "anchor", "TransactionDT")
           and X[c].nunique() > 2][:120]
    cat = [c for c in imp.index if not pd.api.types.is_numeric_dtype(X[c]) and 2 < X[c].nunique()][:16]
    reb = [("TransactionDT", 86400.0, d) for d in ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D10", "D11", "D12", "D13", "D14", "D15"]]
    keys = [["card1", "addr1", "anchor"]] + ([["card1", "anchor"]] if "two" in abl else [])
    blocks = []
    for k in keys:
        sp = ClientProfile(k, num, cat, reb).fit(X)
        blocks.append(pd.DataFrame(sp.transform(X), columns=sp.out_names()))
    df = pd.concat([X] + blocks, axis=1)
    if "dropids" in abl:
        df = df.drop(columns=["anchor", "TransactionDT"])
fe_s = time.time() - t0
Xtr, Xte = df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)
ytr, yte = y.iloc[:n].reset_index(drop=True), y.iloc[n:].reset_index(drop=True)
if a.shuffle:
    ytr = pd.Series(np.random.default_rng(0).permutation(ytr.to_numpy()))
info = {}
if a.forge_kw is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task="binary", time_budget=a.budget, n_jobs=4, **json.loads(a.forge_kw)).fit(Xtr, ytr, X_unlabeled=Xte)
    Xtr, Xte = f.transform_train(Xtr), f.transform(Xte)
    info = dict(n_added=len(f.new_columns_)); fe_s = time.time() - t0
p = judge(Xtr.copy(), ytr, Xte.copy())
res = dict(arm=a.arm + ("-" + a.ablate if a.ablate else "") + ("+forge" if a.forge_kw is not None else "") + ("_shuffled" if a.shuffle else ""), frac=a.frac,
           ncol=Xtr.shape[1], auc=roc_auc_score(yte, p), fe_s=round(fe_s), total_s=round(time.time() - t0), **info)
print("RESULT", json.dumps(res), flush=True); open(a.log, "a").write(json.dumps(res) + "\n")
