"""Feature engineering. Stand-in for Dataform models on BigQuery.

One row per series-day. Every rolling statistic is shifted by one day so a feature never contains
the value it is meant to judge. All detectors read this table and nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

FEATURE_COLS = ["log_units", "z28", "ratio_7", "lag7_ratio", "price_pct", "promo", "snap", "event", "weekend",
                "zero_streak", "price_ratio", "drift_ratio"]
# log_units_adj (log units minus known holiday / SNAP / price / yearly effects) is added by the ensemble after the
# residual model has estimated those effects; the autoencoder and isolation forest read it.


def build_features(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"])
    cal = calendar.copy(); cal["date"] = pd.to_datetime(cal["date"])
    s = s.merge(cal[["date", "dow", "wm_yr_wk", "event_name", "snap_CA", "snap_TX", "snap_WI"]], on="date", how="left")
    s = s.sort_values(["id", "date"]).reset_index(drop=True)
    g = s.groupby("id", sort=False)
    s["log_units"] = np.log1p(s["units"])
    s["lag_1"] = g["units"].shift(1)
    s["lag_7"] = g["units"].shift(7)
    s["lag_28"] = g["units"].shift(28)
    s["rm7"] = g["units"].transform(lambda x: x.shift(1).rolling(7, min_periods=4).mean())
    s["rm28"] = g["units"].transform(lambda x: x.shift(1).rolling(28, min_periods=14).mean())
    s["rm28_log"] = g["log_units"].transform(lambda x: x.shift(1).rolling(28, min_periods=14).mean())
    s["rs28_log"] = g["log_units"].transform(lambda x: x.shift(1).rolling(28, min_periods=14).std())
    s["z28"] = (s["log_units"] - s["rm28_log"]) / (s["rs28_log"] + 0.05)
    s["ratio_7"] = s["units"] / (s["rm7"] + 1.0)
    # slow baseline: last 7 days incl. today vs the 28 days ending 8 days ago (7-day sums cancel weekday effects)
    s["rm7_incl"] = g["units"].transform(lambda x: x.rolling(7, min_periods=4).mean())
    s["rm28_prev"] = g["units"].transform(lambda x: x.shift(8).rolling(28, min_periods=14).mean())
    s["drift_ratio"] = s["rm7_incl"] / (s["rm28_prev"] + 0.5) - 1.0
    s["lag7_ratio"] = s["units"] / (s["lag_7"] + 1.0)
    s["price_ref"] = g["sell_price"].transform(lambda x: x.shift(1).rolling(91, min_periods=7).median())
    s["price_ref"] = s["price_ref"].fillna(s["sell_price"])
    s["price_pct"] = s["sell_price"] / s["price_ref"] - 1.0
    s["price_ratio"] = s["sell_price"] / s["price_ref"]
    s["promo"] = (s["price_pct"] < -0.05).astype(int)
    snap = np.select([s.state_id == "CA", s.state_id == "TX", s.state_id == "WI"],
                     [s.snap_CA, s.snap_TX, s.snap_WI], 0)
    s["snap"] = snap.astype(int)
    s["event"] = (s["event_name"].fillna("") != "").astype(int)
    s["weekend"] = (s["dow"] >= 5).astype(int)
    zero = (s["units"] == 0).astype(int)
    # consecutive zero-day run length including today
    grp = (zero == 0).groupby(s["id"]).cumsum()
    s["zero_streak"] = zero.groupby([s["id"], grp]).cumsum()
    s = s.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    s[["z28", "ratio_7", "lag7_ratio", "price_pct", "drift_ratio"]] = s[["z28", "ratio_7", "lag7_ratio", "price_pct", "drift_ratio"]].fillna(0.0)
    s["event_name"] = s["event_name"].fillna("")
    return s


def split_dates(features: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Train / calibration / holdout boundaries derived from the data's last date."""
    last = pd.to_datetime(features["date"]).max()
    holdout_start = last - pd.Timedelta(days=config.HOLDOUT_DAYS - 1)
    calib_start = holdout_start - pd.Timedelta(days=config.CALIB_DAYS)
    live_start = last - pd.Timedelta(days=config.LIVE_WINDOW_DAYS - 1)
    return {"first": pd.to_datetime(features["date"]).min(), "last": last, "holdout_start": holdout_start,
            "calib_start": calib_start, "live_start": live_start}
