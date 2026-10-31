import numpy as np
import pandas as pd
import pytest

from merchant_pulse import synth
from merchant_pulse.detectors import Ensemble, hw_one_step
from merchant_pulse.episodes import to_episodes
from merchant_pulse.features import build_features


def test_holt_winters_tracks_weekly_pattern():
    T, N = 200, 3
    base = np.array([1.0, 2.0, 3.0])
    week = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = base[None, :] + week[np.arange(T) % 7][:, None]
    yh = hw_one_step(y, np.full(N, 0.2), np.full(N, 0.0), np.full(N, 0.2))
    assert np.abs(y[50:] - yh[50:]).mean() < 0.02


@pytest.fixture(scope="module")
def fitted(feats, dates):
    return Ensemble().fit(feats, dates["holdout_start"], dates["calib_start"])


def test_calibration_respects_budget(fitted):
    cal = fitted.calibration
    assert cal["flag_rate"]["residual"] <= fitted.fa_budget * 1.5
    assert cal["flag_rate"]["iforest"] <= fitted.fa_budget * 1.5
    # per-day flag rate on the calibration slice; episodes (what the gate counts) are fewer than flagged days
    assert cal["ensemble_fa_per_1k"] <= 12.0


def test_price_error_and_stockout_are_caught(fitted, tables, dates):
    rng = np.random.default_rng(3)
    clean = tables["truth_sales_clean"]
    ids = tables["hierarchy"][tables["hierarchy"].store_id == "CA_3"].id.tolist()
    inj, labels = synth.inject_anomalies(clean, ids, dates["live_start"], dates["last"] - pd.Timedelta(days=5), rng,
                                         n_per_type=2, types=["price_error", "stockout"])
    f2 = build_features(inj.merge(tables["hierarchy"], on="id"), tables["calendar"])
    sc = fitted.score(f2, dates["live_start"], dates["last"])
    ep = to_episodes(sc[sc.is_alert])
    for _, lab in labels.iterrows():
        m = ep[(ep.id == lab.id) & (ep.end >= lab.start) & (ep.start <= lab.end + pd.Timedelta(days=2))]
        assert len(m) > 0, f"{lab.anomaly_type} on {lab.id} not detected"
        assert lab.anomaly_type in set(m.anomaly_type)


def test_episodes_merge_consecutive_days(fitted, feats, dates):
    sc = fitted.score(feats, dates["live_start"], dates["last"])
    al = sc[sc.is_alert]
    ep = to_episodes(al)
    assert len(ep) <= len(al)
    for _, e in ep.iterrows():
        assert e.end >= e.start and e.days >= 1
