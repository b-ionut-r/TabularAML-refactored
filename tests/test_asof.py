import numpy as np
import pandas as pd

from tabularaml.generate.relational import Child, asof_features


def _tx(shift_days=0):
    rng = np.random.default_rng(0)
    rows = []
    for u in range(50):
        for k in range(6):
            d = pd.Timestamp("2016-09-01") + pd.Timedelta(days=30 * k + int(rng.integers(0, 5)) + shift_days)
            rows.append((u, int(d.strftime("%Y%m%d")), int((d + pd.Timedelta(days=30)).strftime("%Y%m%d")), float(rng.integers(99, 150))))
    return pd.DataFrame(rows, columns=["user", "transaction_date", "expire_date", "paid"])


def test_dates_as_numbers_are_read_relative_to_the_row():
    # Shifting every date (a later month's users) leaves the as-of features unchanged.
    main = pd.DataFrame({"user": range(50), "as_of": 20170201})
    F = asof_features(main, "user", "as_of", Child("tx", _tx(), key="user", time="transaction_date"))
    G = asof_features(main.assign(as_of=20170303), "user", "as_of", Child("tx", _tx(30), key="user", time="transaction_date"))
    assert "tx__expire_date_max" in F.columns and "tx__expire_date_sum" not in F.columns
    assert F["tx__n"].equals(G["tx__n"])
    np.testing.assert_allclose(F["tx__expire_date_max"], G["tx__expire_date_max"], atol=1.5)
    np.testing.assert_allclose(F["tx__paid_mean"], G["tx__paid_mean"])
