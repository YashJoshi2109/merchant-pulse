import numpy as np
import pandas as pd

from merchant_pulse import config, synth
from merchant_pulse.features import FEATURE_COLS


def test_hierarchy_shape(tables):
    h = tables["hierarchy"]
    assert h.store_id.nunique() == 10 and h.cat_id.nunique() == 3 and h.dept_id.nunique() == 7
    assert len(h) == 280 and h.id.is_unique


def test_sales_complete_and_nonnegative(tables):
    s = tables["sales"]
    assert len(s) == 280 * 731
    assert (s.units >= 0).all() and (s.sell_price > 0).all()
    clean = tables["truth_sales_clean"]
    xmas = clean[clean.date == pd.Timestamp("2025-12-25")]
    assert xmas.units.sum() == 0, "store closed on Christmas"


def test_snap_lifts_foods(tables):
    s = tables["truth_sales_clean"].merge(tables["hierarchy"], on="id").merge(tables["calendar"][["date", "snap_CA", "event_name"]], on="date")
    ca = s[(s.state_id == "CA") & (s.cat_id == "FOODS") & (s.event_name == "")]
    lift = ca[ca.snap_CA == 1].units.mean() / ca[ca.snap_CA == 0].units.mean()
    assert 1.1 < lift < 1.4


def test_inject_anomalies_labels_match_data(tables):
    rng = np.random.default_rng(0)
    clean = tables["truth_sales_clean"]
    ids = tables["hierarchy"].id.tolist()
    inj, labels = synth.inject_anomalies(clean, ids, pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-20"), rng, n_per_type=2)
    assert set(labels.anomaly_type) == set(config.ANOMALY_TYPES)
    so = labels[labels.anomaly_type == "stockout"].iloc[0]
    seg = inj[(inj.id == so.id) & (inj.date >= so.start) & (inj.date <= so.end)]
    assert (seg.units == 0).all()
    pe = labels[labels.anomaly_type == "price_error"].iloc[0]
    seg = inj[(inj.id == pe.id) & (inj.date >= pe.start) & (inj.date <= pe.end)]
    ref = clean[(clean.id == pe.id) & (clean.date >= pe.start) & (clean.date <= pe.end)]
    assert np.allclose(seg.sell_price.values, ref.sell_price.values * 10, atol=0.02)
    # untouched rows are identical
    untouched = inj[~inj.id.isin(labels.id)].units.values
    assert (untouched == clean[~clean.id.isin(labels.id)].units.values).all()


def test_features_no_leakage(feats):
    g = feats[feats.id == feats.id.iloc[0]].sort_values("date")
    # rolling mean shifted by one: rm7 at t equals mean of units t-7..t-1
    t = 100
    assert abs(g.rm7.iloc[t] - g.units.iloc[t - 7:t].mean()) < 1e-9
    assert set(FEATURE_COLS).issubset(feats.columns)
    assert feats[FEATURE_COLS].isna().sum().sum() == 0
