import numpy as np

from tabularaml.generate.forge import _later_half, _winsor


def test_winsor_within_class_keeps_rare_positive_rows():
    # 0.4% positives: globally every positive row sits in the clipped tails.
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(9960, int), np.ones(40, int)]
    d = np.r_[rng.normal(0.0001, 0.0001, 9960), rng.normal(-0.5, 0.1, 40)]
    glob, per = _winsor(d), _winsor(d, y)
    assert d.mean() < 0 and per.mean() < 0          # per-class clipping keeps the sign of the mean
    assert glob[y == 1].min() > per[y == 1].min()   # global clipping squashed the positives
    assert np.allclose(_winsor(d, None), glob)


def test_later_half_sees_a_reversed_gain():
    t = np.arange(100.0)
    d = np.r_[np.full(50, 1.0), np.full(50, -0.2)]
    assert d.mean() > 0 and _later_half(d, t) < 0
    assert _later_half(d[::-1], t[::-1]) < 0        # ordered by time, not by position
    assert _later_half(np.ones(10), np.zeros(10)) > 0
