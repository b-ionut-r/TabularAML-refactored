import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.forge import Context
from tabularaml.generate.seqstr import CharGrams, CharSVD, is_seq_string, is_smiles, seq_specs


def _smiles(n, seed=0):
    rng = np.random.default_rng(seed)
    parts = ["C", "CC", "c1ccccc1", "C(=O)O", "N", "O", "C(C)C", "C(Cl)", "C(C#N)", "OC"]
    return pd.Series(["*" + "".join(rng.choice(parts, rng.integers(2, 8))) + "*" for _ in range(n)])


def test_detects_structured_strings_not_text_or_categories():
    s = _smiles(400)
    assert is_seq_string(s)
    assert not is_seq_string(pd.Series(["red", "green", "blue"] * 200))
    assert not is_seq_string(pd.Series([f"the quick brown fox number {i}" for i in range(400)]))
    rng = np.random.default_rng(1)
    assert not is_seq_string(pd.Series([f"{x:08x}" for x in rng.integers(0, 2**31, 400)]))  # hashed ids
    assert not is_seq_string(pd.Series([f"AB-{i:05d}-X{i % 7}" for i in range(400)]))  # fixed-width codes


def test_char_views_shapes_and_unseen_values():
    s = _smiles(300)
    df, new = pd.DataFrame({"s": s[:200]}), pd.DataFrame({"s": s[200:]})
    ctx = Context("regression", 0, 0)
    for sp in (CharGrams("s"), CharSVD("s")):
        sp.fit(df, None, ctx)
        a, b = sp.transform(df, ctx), sp.transform(new, ctx)
        assert a.shape == (200, sp.n_out) and b.shape == (100, sp.n_out)
        assert np.isfinite(a).all()
        assert len(sp.out_names()) == sp.n_out


def test_molecule_specs_when_rdkit_present():
    pytest.importorskip("rdkit")
    s = _smiles(300)
    assert is_smiles(s)
    assert not is_smiles(pd.Series([f"AB-{i:05d}-X" for i in range(300)]))
    specs = seq_specs(["s"], ["s"], 0)
    names = [type(sp).__name__ for sp in specs]
    assert "MolDescriptors" in names and "MolFingerprint" in names
    ctx = Context("regression", 0, 0)
    df = pd.DataFrame({"s": s})
    for sp in specs:
        if type(sp).__name__ in ("MolDescriptors", "MolFingerprint"):
            v = sp.fit(df, None, ctx).transform(df, ctx)
            assert v.shape == (300, sp.n_out) and len(sp.out_names()) == sp.n_out
