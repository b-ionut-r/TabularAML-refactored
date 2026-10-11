import numpy as np
import pandas as pd

from tabularaml.generate.lists import as_lists, list_features, unit_ratio_features


def _listings(n=600, seed=0):
    rng = np.random.default_rng(seed)
    amen = ["Doorman", "Elevator", "Dishwasher", "Hardwood Floors", "Cats Allowed", "Pre-War"]
    feats = [[a for a in amen if rng.random() < 0.4] for _ in range(n)]
    photos = [[f"https://p/{i}_{j}.jpg" for j in range(rng.integers(0, 8))] for i in range(n)]
    beds = rng.integers(0, 5, n)
    baths = rng.integers(1, 4, n).astype(float)
    price = np.round((900 * (beds + 1) + 900 * baths) * rng.lognormal(0, 0.6, n))
    return pd.DataFrame({"features": feats, "photos": photos, "bedrooms": beds, "bathrooms": baths, "price": price,
                         "wall": rng.choice(["Stone, brick", "Panel", "Wooden"], n)})


def test_lists_from_cells_and_delimited_strings():
    df = _listings()
    Ftr, Fte, found = list_features(df.iloc[:400], df.iloc[400:])
    assert found == ["features", "photos"]   # the comma inside a category name is not a list
    assert len(Ftr) == 400 and len(Fte) == 200
    assert (Ftr["photos__n_items"].to_numpy() == df.photos.iloc[:400].map(len).to_numpy()).all()
    assert (Ftr["features__has__doorman"].to_numpy() == df.features.iloc[:400].map(lambda v: "Doorman" in v)).all()
    assert not any(c.startswith("photos__has__") for c in Ftr.columns)   # each URL appears once
    joined = df.features.map(" | ".join)
    assert as_lists(joined).map(len).equals(df.features.map(len))


def test_unit_ratios_need_an_amount_and_two_counts():
    df = _listings().drop(columns=["features", "photos"])
    Utr, Ute = unit_ratio_features(df.iloc[:400], df.iloc[400:])
    assert set(Utr.columns) == {"price__per__bedrooms", "price__per__bathrooms", "price__per__bedrooms_bathrooms"}
    b = df.bedrooms.iloc[:400].to_numpy()
    r = Utr["price__per__bedrooms"].to_numpy()
    assert np.isnan(r[b == 0]).all()
    assert np.allclose(r[b > 0], df.price.iloc[:400].to_numpy()[b > 0] / b[b > 0], rtol=1e-5)
    Utr, _ = unit_ratio_features(df.iloc[:400].drop(columns=["bathrooms"]), df.iloc[400:].drop(columns=["bathrooms"]))
    assert Utr.shape[1] == 0


def test_addresses_are_not_lists_but_tag_sets_are():
    rng = np.random.default_rng(0)
    streets = [f"{n} North {s} Avenue" for n, s in zip(rng.integers(100, 9999, 150), rng.choice(list("ABCDEFGH"), 150))]
    addr = pd.Series([f"{rng.choice(streets)}, Chicago, IL {rng.integers(60601, 60660)}, USA" for _ in range(3000)])
    assert as_lists(addr) is None
    voc = [f"tag{i}" for i in range(30)]
    tags = pd.Series([",".join(rng.choice(voc, rng.integers(1, 6), replace=False)) for _ in range(3000)])
    assert as_lists(tags) is not None


def test_categorical_columns_are_read_not_crashed_on():
    s = pd.Series([f"code{i % 300}" for i in range(1000)] + [None] * 20, dtype="category")
    assert as_lists(s) is None
    rng = np.random.default_rng(0)
    tags = pd.Series([", ".join(f"tag{t}" for t in rng.choice(30, 3, replace=False)) for _ in range(400)],
                     dtype="category")
    assert as_lists(tags) is not None
