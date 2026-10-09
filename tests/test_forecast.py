"""ForecastFeatures: switches on from structure, and no row's features read its own or later labels."""
import numpy as np
import pandas as pd

from tabularaml.generate.forecast import ForecastFeatures


def _panel(n_ent=30, days=400, test_days=28, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days + test_days)
    df = pd.DataFrame([(e, d) for e in range(n_ent) for d in dates], columns=["store", "date"])
    df["kind"] = np.where(df.store % 3 == 0, "a", "b")
    df["promo"] = (rng.random(len(df)) < 0.2).astype(int)
    df["price"] = np.round(rng.uniform(1, 5, len(df)), 2)
    df["date"] = df.date.dt.strftime("%Y-%m-%d")
    lvl = rng.gamma(5, 100, n_ent)
    y = lvl[df.store] * (1 + 0.3 * df.promo) * rng.lognormal(0, 0.2, len(df))
    tr = (pd.to_datetime(df.date) < dates[days]).to_numpy()
    return df[tr].reset_index(drop=True), y[tr], df[~tr].reset_index(drop=True)


def test_switches_on_and_off():
    X, y, U = _panel()
    f = ForecastFeatures(verbose=False).fit(X, y, U)
    assert f.active_ and f.entity_ == ["store"] and f.hmin_ == 1 and f.hmax_ == 28
    assert ["kind"] in f.groups_ and [c for c, _ in f.covariates_] == ["promo"] and f.numeric_ == ["price"]
    # Unlabeled rows inside the training period (a random split): off.
    g = ForecastFeatures(verbose=False).fit(X.iloc[::2], y[::2], X.iloc[1::2])
    assert not g.active_


def test_no_own_or_later_labels():
    X, y, U = _panel()
    X["visits"] = np.round(y / 7)   # a training-only companion of the target
    f = ForecastFeatures(verbose=False, medians=True, event_counts=True, numeric_covariates=True, companions=True).fit(X, y, U)
    F = f.transform(X)
    p = f._periods(X)
    assert np.all(F["fc_h"].to_numpy() >= 1) and "fc_cmp_mean7__visits" in F and "fc_num_rel__price" in F and "fc_ev_next7__promo" in F
    rng = np.random.default_rng(1)
    for i in rng.choice(len(X), 20, replace=False):
        y2 = y.copy()
        later = (p >= p[i]) & (X.store.to_numpy() == X.store[i])
        y2[later] = y2[later] * 10 + 1000          # own and later labels of the entity changed
        y2[p >= p[i]] = y2[p >= p[i]] * 3          # and every later label of any entity
        X2 = X.copy(); X2.loc[p >= p[i], "visits"] *= 5
        F2 = ForecastFeatures(verbose=False, medians=True, event_counts=True, numeric_covariates=True, companions=True).fit(X2, y2, U).transform(X2.iloc[[i]])
        a, b = F.iloc[[i]].to_numpy(), F2.to_numpy()
        assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1), rtol=1e-5), X.iloc[i]


def test_test_rows_use_training_labels_only():
    X, y, U = _panel()
    f = ForecastFeatures(verbose=False).fit(X, y, U)
    Fu = f.transform(U)
    assert np.all(Fu["fc_h"].to_numpy() == f._periods(U) - f.T_)
    assert Fu["fc_mean7"].notna().all()


def test_hourly_no_own_or_later_labels():
    rng = np.random.default_rng(2)
    ts = pd.date_range("2020-01-01", periods=24 * 60 + 24 * 14, freq="h")
    df = pd.DataFrame([(e, t) for e in range(8) for t in ts], columns=["meter", "ts"])
    y = 10 + 5 * np.sin(df.ts.dt.hour / 24 * 2 * np.pi) + df.meter + rng.normal(0, 1, len(df))
    df["ts"] = df.ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    tr = (df.ts < str(ts[24 * 60])).to_numpy()
    X, U, y = df[tr].reset_index(drop=True), df[~tr].reset_index(drop=True), y[tr].to_numpy()
    f = ForecastFeatures(verbose=False, log_target=False).fit(X, y, U)
    assert f.active_ and f.step_ == 1 / 24 and f.hmax_ == 24 * 14
    F = f.transform(X)
    p = f._periods(X)
    for i in rng.choice(len(X), 10, replace=False):
        y2 = y.copy()
        y2[p >= p[i]] = y2[p >= p[i]] * 5 + 100
        F2 = ForecastFeatures(verbose=False, log_target=False).fit(X, y2, U).transform(X.iloc[[i]])
        assert np.allclose(np.nan_to_num(F.iloc[[i]].to_numpy(), nan=-1), np.nan_to_num(F2.to_numpy(), nan=-1), rtol=1e-5)
    assert f.transform(U)["fc_sameday1"].notna().mean() > 0.9


def test_monthly_periods_and_no_own_or_later_labels():
    rng = np.random.default_rng(3)
    ms = pd.date_range("2018-01-01", periods=48, freq="MS")
    df = pd.DataFrame([(e, t) for e in range(30) for t in ms], columns=["county", "month"])
    y = 5 + df.county * 0.1 + np.arange(len(df)) % 48 * 0.05 + rng.normal(0, 0.1, len(df))
    df["month"] = df.month.dt.strftime("%Y-%m-%d")
    tr = (df.month < "2021-07-01").to_numpy()
    X, U, y = df[tr].reset_index(drop=True), df[~tr].reset_index(drop=True), y[tr].to_numpy()
    f = ForecastFeatures(verbose=False, log_target=False).fit(X, y, U)
    assert f.active_ and f.monthly_ and (f.hmin_, f.hmax_) == (1, 6)
    assert np.array_equal(np.unique(np.diff(np.unique(f._periods(df)))), [1])
    F = f.transform(X)
    p = f._periods(X)
    for i in rng.choice(len(X), 10, replace=False):
        y2 = y.copy()
        y2[p >= p[i]] = y2[p >= p[i]] * 5 + 100
        F2 = ForecastFeatures(verbose=False, log_target=False).fit(X, y2, U).transform(X.iloc[[i]])
        assert np.allclose(np.nan_to_num(F.iloc[[i]].to_numpy(), nan=-1), np.nan_to_num(F2.to_numpy(), nan=-1), rtol=1e-5)
