"""Collapse consecutive alert days on one series into episodes.

A five-day stockout is one thing a merchant acts on, not five alerts. Alert days on the same series
with a gap of at most `max_gap` days merge; the episode carries the first day, the last day, the dominant
anomaly type, the strongest score and the summed revenue at risk.
"""
from __future__ import annotations

import hashlib

import pandas as pd


def _episode_id(sid: str, start: pd.Timestamp) -> str:
    return hashlib.sha1(f"{sid}|{start.date()}".encode()).hexdigest()[:10]


def to_episodes(alert_days: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    if alert_days.empty:
        return pd.DataFrame()
    a = alert_days.sort_values(["id", "date"]).copy()
    a["date"] = pd.to_datetime(a["date"])
    gap = a.groupby("id").date.diff().dt.days.fillna(999)
    a["ep"] = (gap > max_gap).groupby(a["id"]).cumsum()
    rows = []
    for (sid, _), g in a.groupby(["id", "ep"], sort=False):
        strongest = g.loc[g.z_resid.abs().idxmax()]
        ty = g.anomaly_type.mode()
        ty = ty.iloc[0] if len(ty) else strongest.anomaly_type
        if (g.anomaly_type == "price_error").any():
            ty = "price_error"
        elif (g.anomaly_type == "stockout").any():
            ty = "stockout"
        first = g.iloc[0]
        rows.append({
            "alert_id": _episode_id(sid, first.date), "id": sid, "start": first.date, "end": g.date.max(),
            "days": int(len(g)), "store_id": first.store_id, "state_id": first.state_id, "item_id": first.item_id,
            "dept_id": first.dept_id, "cat_id": first.cat_id, "anomaly_type": ty,
            "direction": strongest.direction, "z_resid": float(strongest.z_resid), "z_scale": strongest.z_scale,
            "votes": int(g.votes.max()), "rule": bool(g.f_rule.any()),
            "detectors": ",".join([d for d, c in (("residual", "f_resid"), ("iforest", "f_if"), ("autoencoder", "f_ae")) if g[c].any()]),
            "units_actual": float(g.units.sum()), "units_expected": float(g.yhat_units.sum()),
            "price": float(strongest.sell_price), "price_ref": float(strongest.price_ref),
            "revenue_at_risk": float(g.revenue_at_risk.max()), "event_name": ",".join(sorted({e for e in g.event_name if e})),
            "peak_date": strongest.date,
        })
    return pd.DataFrame(rows).sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)
