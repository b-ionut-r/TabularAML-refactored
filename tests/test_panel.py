import numpy as np
import pandas as pd

from tabularaml.generate.panel import find_panel, panel_features


def _auction(days=range(10), stocks=30, steps=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        for s in range(steps):
            for k in range(stocks):
                rows.append((k, d, s * 10, d * steps + s))
    df = pd.DataFrame(rows, columns=["stock_id", "date_id", "seconds_in_bucket", "time_id"])
    df["wap"] = 1 + rng.normal(0, 0.001, len(df)).cumsum() / 100 + 0.01 * df.stock_id
    df["imbalance_size"] = rng.lognormal(10, 1, len(df))
    return df


def test_finds_the_auction_panel_and_uses_earlier_steps_only():
    tr, te = _auction(range(8)), _auction(range(8, 10), seed=1)
    P = find_panel(tr, te)
    assert P == {"entity": "stock_id", "moment": ["time_id"], "day": "date_id"}
    F, G, _ = panel_features(tr, te)
    assert len(F) == len(tr) and len(G) == len(te)
    x = tr[(tr.stock_id == 3) & (tr.date_id == 2)].sort_values("time_id")
    f = F.loc[x.index, "wap__step_change1"].to_numpy()
    assert np.isnan(f[0]) and np.allclose(f[1:], x.wap.to_numpy()[1:] / x.wap.to_numpy()[:-1] - 1)
    m = tr.time_id == 5
    assert np.isclose(F.loc[m, "wap__vs_moment"].mean(), 0, atol=1e-9)


def test_off_on_daily_panels_and_random_splits():
    # A store-by-date panel: one moment per day.
    rng = np.random.default_rng(0)
    df = pd.DataFrame([(s, d) for d in range(200) for s in range(50)], columns=["store", "date"])
    df["promo"] = rng.random(len(df))
    assert find_panel(df[df.date < 150], df[df.date >= 150]) is None
    # The auction split at random: test days are not new.
    a = _auction(range(10))
    idx = np.random.default_rng(0).permutation(len(a))
    assert find_panel(a.iloc[np.sort(idx[:3000])], a.iloc[np.sort(idx[3000:])]) is None
