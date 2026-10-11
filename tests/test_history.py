"""Latest-state history features read rows in time order, per key, and only the child table."""
import numpy as np
import pandas as pd

from tabularaml.generate.history import history_features, repeats
from tabularaml.generate.relational import Child


def test_last_mean_prev_in_time_order():
    df = pd.DataFrame({"k": [1, 1, 1, 2, 2, 3], "t": [3, 1, 2, 1, 2, 1], "x": [30.0, 10, 20, 5, 6, 7]})
    ch = Child("s", df, key="k", time="t")
    assert repeats(ch)
    H = history_features(ch)
    assert H.loc[1, "s__x_last"] == 30 and H.loc[1, "s__x_last_mean"] == 10 and H.loc[1, "s__x_last_prev"] == 10
    assert H.loc[2, "s__x_last_prev"] == 1 and np.isnan(H.loc[3, "s__x_last_prev"])
    assert not repeats(Child("s", df.drop_duplicates("k"), key="k", time="t"))


def test_event_log_strictly_earlier():
    from tabularaml.generate.history import event_log_features
    log = pd.DataFrame({"u": [1, 1, 1, 1, 2], "t": [0, 0, 5, 9, 0], "q": [7, 8, 7, 8, 7], "ok": [1, 0, 1, -1, 1]})
    main = pd.DataFrame({"u": [1, 1, 1, 2], "t": [0, 5, 9, 3], "q": [7, 7, 8, 7]})
    F = event_log_features(main, Child("log", log, key="u", time="t"), "ok", "t", item="q", outcome_values=[0, 1])
    # time 0: nothing earlier (its own bundle hidden); time 5: two answers at time 0; time 9: three answers.
    assert np.isnan(F.loc[0, "log__ok__hist_mean"]) and F.loc[1, "log__ok__hist_n"] == 2 and F.loc[1, "log__ok__hist_mean"] == 0.5
    assert F.loc[2, "log__ok__hist_n"] == 3 and F.loc[2, "log__ok__hist_lag1"] == 1 and F.loc[2, "log__ok__hist_since1"] == 4
    assert F.loc[1, "log__ok__hist_item_n"] == 1 and F.loc[1, "log__ok__hist_item_mean"] == 1 and F.loc[2, "log__ok__hist_item_mean"] == 0
    assert F.loc[3, "log__ok__hist_n"] == 1 and F.loc[3, "log__ok__hist_item_n"] == 1
