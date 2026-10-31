"""Production monitoring (figure 2). Stand-in for Vertex AI Model Monitoring + Cloud Monitoring.

Three signals decide whether to retrain or roll back: feature drift (PSI of the last 7 days against the
training reference), alert volume per store per day, and precision computed from merchant feedback.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, store


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref = ref[np.isfinite(ref)]; cur = cur[np.isfinite(cur)]
    if len(ref) < 50 or len(cur) < 20:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    r = np.histogram(ref, edges)[0] / len(ref); c = np.histogram(cur, edges)[0] / len(cur)
    r = np.clip(r, 1e-4, None); c = np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feedback_precision(alerts: pd.DataFrame | None, feedback: pd.DataFrame | None, days: int = 28) -> dict:
    if alerts is None or feedback is None or feedback.empty:
        return {"confirmed": 0, "dismissed": 0, "precision": float("nan")}
    fb = feedback.sort_values("ts").drop_duplicates("alert_id", keep="last")
    cutoff = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=days)
    fb = fb[pd.to_datetime(fb.ts).dt.tz_localize(None) >= cutoff] if "ts" in fb else fb
    c = int((fb.verdict == "confirmed").sum()); d = int((fb.verdict == "dismissed").sum())
    return {"confirmed": c, "dismissed": d, "precision": c / (c + d) if c + d else float("nan")}


def snapshot(feats: pd.DataFrame, alerts: pd.DataFrame | None, feedback: pd.DataFrame | None, train_end: pd.Timestamp,
             last: pd.Timestamp, version: str | None) -> dict:
    ref = feats[(feats.date >= train_end - pd.Timedelta(days=90)) & (feats.date < train_end)]
    cur = feats[feats.date > last - pd.Timedelta(days=7)]
    drift = {c: psi(ref[c].values.astype(float), cur[c].values.astype(float)) for c in config.MONITOR_FEATURES}
    n_series = feats.id.nunique()
    recent_alerts = alerts[(pd.to_datetime(alerts.peak_date) > last - pd.Timedelta(days=7)) & (alerts.level == "item")] if alerts is not None and len(alerts) else pd.DataFrame()
    alert_rate = len(recent_alerts) / (n_series * 7) * 1000
    per_store = recent_alerts.groupby("store_id").size().div(7) if len(recent_alerts) else pd.Series(dtype=float)
    fbp = feedback_precision(alerts, feedback)
    reasons = [f"PSI({c})={v:.2f} > {config.PSI_RETRAIN}" for c, v in drift.items() if np.isfinite(v) and v > config.PSI_RETRAIN]
    if np.isfinite(fbp["precision"]) and fbp["precision"] < config.SLO["precision_min"] and fbp["confirmed"] + fbp["dismissed"] >= 10:
        reasons.append(f"feedback precision {fbp['precision']:.2f} < {config.SLO['precision_min']}")
    if alert_rate > config.GATE["fa_per_1k_max"] * 3:
        reasons.append(f"alert rate {alert_rate:.1f}/1k > 3x budget")
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "as_of": str(last.date()), "model_version": version,
           **{f"psi_{c}": v for c, v in drift.items()}, "alert_episodes_per_1k_series_days_7d": alert_rate,
           "max_alerts_per_store_per_day": float(per_store.max()) if len(per_store) else 0.0,
           "feedback_confirmed": fbp["confirmed"], "feedback_dismissed": fbp["dismissed"], "feedback_precision": fbp["precision"],
           "retrain_trigger": len(reasons) > 0, "trigger_reasons": "; ".join(reasons)}
    store.append("monitoring", pd.DataFrame([row]))
    return row
