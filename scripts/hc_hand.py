"""Home Credit: the winners' public hand features on top of a contest_features.py output.

Adds the well-known public-kernel features (application ratios; installment lateness and
payment shortfall; previous-application credit ratio and approved/refused splits; bureau
active/closed splits and bureau_balance status counts; recent-window installment aggregates)
to a feature directory, so ``score_holdout.py`` can judge them next to FeatureForge with the
same LightGBM. ``--groups`` picks which groups are added (ablations).

    python scripts/hc_hand.py --base feats/hc3_noscan --data data/hc_dl --out feats/hc_hand --groups app,inst
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

K = "SK_ID_CURR"


def agg(df: pd.DataFrame, cols, stats, prefix: str, key: str = K) -> pd.DataFrame:
    g = df.groupby(key)[cols].agg(stats)
    g.columns = [f"{prefix}__{c}__{s}" for c, s in g.columns]
    return g.astype(np.float32)


def app_feats(A: pd.DataFrame) -> pd.DataFrame:
    A = A.copy()
    A["DAYS_EMPLOYED"] = A["DAYS_EMPLOYED"].replace(365243, np.nan)
    ext = A[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    F = pd.DataFrame(index=A.index)
    F["h_days_employed_perc"] = A["DAYS_EMPLOYED"] / A["DAYS_BIRTH"]
    F["h_income_credit"] = A["AMT_INCOME_TOTAL"] / A["AMT_CREDIT"]
    F["h_income_per_person"] = A["AMT_INCOME_TOTAL"] / A["CNT_FAM_MEMBERS"]
    F["h_annuity_income"] = A["AMT_ANNUITY"] / A["AMT_INCOME_TOTAL"]
    F["h_payment_rate"] = A["AMT_ANNUITY"] / A["AMT_CREDIT"]
    F["h_credit_goods"] = A["AMT_CREDIT"] / A["AMT_GOODS_PRICE"]
    F["h_ext_mean"] = ext.mean(1)
    F["h_ext_std"] = ext.std(1)
    F["h_ext_prod"] = ext.prod(1, min_count=3)
    F["h_ext_weighted"] = A["EXT_SOURCE_1"] * 2 + A["EXT_SOURCE_2"] + A["EXT_SOURCE_3"] * 3
    F["h_phone_age"] = A["DAYS_LAST_PHONE_CHANGE"] / A["DAYS_BIRTH"]
    F[K] = A[K].to_numpy()
    return F.set_index(K)


def inst_feats(d: Path, recent: bool) -> pd.DataFrame:
    I = pd.read_csv(d / "installments_payments.csv")
    I["PAYMENT_PERC"] = I["AMT_PAYMENT"] / I["AMT_INSTALMENT"]
    I["PAYMENT_DIFF"] = I["AMT_INSTALMENT"] - I["AMT_PAYMENT"]
    I["DPD"] = (I["DAYS_ENTRY_PAYMENT"] - I["DAYS_INSTALMENT"]).clip(lower=0)
    I["DBD"] = (I["DAYS_INSTALMENT"] - I["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    I["LATE"] = (I["DPD"] > 0).astype(float)
    cols = ["PAYMENT_PERC", "PAYMENT_DIFF", "DPD", "DBD", "LATE"]
    out = [agg(I, cols, ["mean", "max", "sum", "std"], "h_inst")]
    if recent:
        for w in (365, 90):
            R = I[I["DAYS_INSTALMENT"] > -w]
            out.append(agg(R, cols, ["mean", "max", "sum"], f"h_inst{w}"))
    return pd.concat(out, axis=1)


def prev_feats(d: Path) -> pd.DataFrame:
    P = pd.read_csv(d / "previous_application.csv")
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        P[c] = P[c].replace(365243, np.nan)
    P["APP_CREDIT_PERC"] = P["AMT_APPLICATION"] / P["AMT_CREDIT"]
    P["CREDIT_GOODS"] = P["AMT_CREDIT"] / P["AMT_GOODS_PRICE"]
    nums = ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "APP_CREDIT_PERC", "AMT_DOWN_PAYMENT",
            "DAYS_DECISION", "CNT_PAYMENT", "CREDIT_GOODS"]
    out = [agg(P, ["APP_CREDIT_PERC", "CREDIT_GOODS"], ["mean", "max", "min", "var"], "h_prev")]
    for st in ("Approved", "Refused"):
        out.append(agg(P[P["NAME_CONTRACT_STATUS"] == st], nums, ["mean", "max", "min"], f"h_prev_{st[:3].lower()}"))
    return pd.concat(out, axis=1)


def bureau_feats(d: Path) -> pd.DataFrame:
    B = pd.read_csv(d / "bureau.csv")
    BB = pd.read_csv(d / "bureau_balance.csv")
    bb = BB.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].agg(["min", "max", "size"])
    bb.columns = [f"BB_MONTHS_{c}" for c in bb.columns]
    st = pd.crosstab(BB["SK_ID_BUREAU"], BB["STATUS"], normalize="index")
    st.columns = [f"BB_STATUS_{c}" for c in st.columns]
    B = B.merge(bb, left_on="SK_ID_BUREAU", right_index=True, how="left")
    B = B.merge(st, left_on="SK_ID_BUREAU", right_index=True, how="left")
    B["DEBT_CREDIT"] = B["AMT_CREDIT_SUM_DEBT"] / B["AMT_CREDIT_SUM"]
    B["ENDDATE_DIFF"] = B["DAYS_CREDIT_ENDDATE"] - B["DAYS_ENDDATE_FACT"]
    bbc = [c for c in B.columns if c.startswith("BB_")]
    nums = ["DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "DAYS_CREDIT_UPDATE", "CREDIT_DAY_OVERDUE",
            "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
            "AMT_CREDIT_SUM_LIMIT", "AMT_ANNUITY", "CNT_CREDIT_PROLONG", "DEBT_CREDIT"]
    out = [agg(B, bbc + ["DEBT_CREDIT", "ENDDATE_DIFF"], ["mean", "max", "min"], "h_bur")]
    for s in ("Active", "Closed"):
        out.append(agg(B[B["CREDIT_ACTIVE"] == s], nums, ["mean", "max", "min", "sum"], f"h_bur_{s.lower()}"))
    return pd.concat(out, axis=1)


def knn_feats(tr_raw: pd.DataFrame, te_raw: pd.DataFrame, y: pd.Series, k: int = 500) -> pd.DataFrame:
    """1st place's neighbours feature: mean label of the 500 nearest training applicants on
    EXT_SOURCE_1..3 and annuity / credit; out-of-fold on the training rows."""
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NearestNeighbors
    def X(A):
        Z = A[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].copy()
        Z["pr"] = A["AMT_ANNUITY"] / A["AMT_CREDIT"]
        return Z
    Ztr, Zte = X(tr_raw), X(te_raw)
    mu, sd = Ztr.mean(), Ztr.std()
    Ztr, Zte = ((Ztr - mu) / sd).fillna(0).to_numpy(), ((Zte - mu) / sd).fillna(0).to_numpy()
    yv = y.to_numpy(dtype=float)
    f_tr = np.zeros(len(Ztr))
    for a, b in KFold(5, shuffle=True, random_state=0).split(Ztr):
        nn = NearestNeighbors(n_neighbors=k, n_jobs=4).fit(Ztr[a])
        f_tr[b] = yv[a][nn.kneighbors(Ztr[b], return_distance=False)].mean(1)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=4).fit(Ztr)
    f_te = yv[nn.kneighbors(Zte, return_distance=False)].mean(1)
    ids = np.concatenate([tr_raw[K].to_numpy(), te_raw[K].to_numpy()])
    return pd.DataFrame({"h_knn500": np.concatenate([f_tr, f_te]).astype(np.float32)}, index=pd.Index(ids, name=K))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw-train", required=True, help="holdout train.csv (application columns)")
    ap.add_argument("--raw-test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="app,inst,recent,prev,bureau")
    a = ap.parse_args()
    t0, d, groups = time.time(), Path(a.data), set(a.groups.split(","))
    tr = pd.read_parquet(Path(a.base) / "train_features.parquet")
    te = pd.read_parquet(Path(a.base) / "test_features.parquet")
    parts = []
    if "app" in groups:
        A = pd.concat([pd.read_csv(a.raw_train), pd.read_csv(a.raw_test)], ignore_index=True)
        parts.append(app_feats(A))
    if "knn" in groups:
        R, T = pd.read_csv(a.raw_train), pd.read_csv(a.raw_test)
        parts.append(knn_feats(R, T, R["TARGET"]))
    if "inst" in groups or "recent" in groups:
        parts.append(inst_feats(d, "recent" in groups))
    if "prev" in groups:
        parts.append(prev_feats(d))
    if "bureau" in groups:
        parts.append(bureau_feats(d))
    H = pd.concat(parts, axis=1)
    H = H[~H.index.duplicated()]
    print(f"{H.shape[1]} hand columns in {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for df, name in ((tr, "train"), (te, "test")):
        h = H.reindex(df[K].to_numpy()).reset_index(drop=True)
        pd.concat([df.reset_index(drop=True), h], axis=1).to_parquet(out / f"{name}_features.parquet")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
