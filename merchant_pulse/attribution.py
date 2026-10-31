"""Root-cause attribution (figure 3): rules first, then model contributions.

Rules name the things a feature-importance method cannot: a price change on the same day, a SNAP issuance
day, a named holiday, a zero-sales run. After the rules, the top feature contributions for the Isolation
Forest row are added, via SHAP TreeExplainer when the `shap` package is importable and a z-score contrast
against the series' own 28-day history otherwise. Output is always exactly three drivers, strongest first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLS

try:  # optional
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:  # pragma: no cover
    _HAS_SHAP = False

_EXPLAINERS: dict = {}


def _explainer(model):
    key = id(model)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(model)
    return _EXPLAINERS[key]


FEATURE_LABEL = {
    "log_units": "sales level", "z28": "deviation vs 28-day norm", "ratio_7": "vs trailing 7-day mean",
    "lag7_ratio": "vs same weekday last week", "price_pct": "price vs reference", "promo": "promotion flag",
    "snap": "SNAP day", "event": "holiday / event", "weekend": "weekend", "zero_streak": "zero-sales run",
    "price_ratio": "price ratio", "drift_ratio": "7-day vs prior 28-day trend", "log_units_adj": "adjusted sales level",
}


def rule_drivers(row: pd.Series) -> list[dict]:
    d = []
    pr = float(row.get("price_ratio", 1.0))
    if pr >= 3 or pr <= 1 / 3:
        d.append({"driver": f"price file {pr:.1f}x reference (${row.price_ref:.2f} → ${row.sell_price:.2f})", "kind": "rule", "weight": 1.0})
    elif row.get("price_pct", 0) <= -0.05:
        d.append({"driver": f"price cut {abs(row.price_pct) * 100:.0f}% in effect", "kind": "rule", "weight": 0.8})
    elif row.get("price_pct", 0) >= 0.05:
        d.append({"driver": f"price up {row.price_pct * 100:.0f}% vs reference", "kind": "rule", "weight": 0.8})
    if int(row.get("zero_streak", 0)) >= 2:
        d.append({"driver": f"{int(row.zero_streak)}-day zero-sales run, likely stockout", "kind": "rule", "weight": 0.9})
    if str(row.get("event_name", "")):
        d.append({"driver": f"event: {row.event_name}", "kind": "rule", "weight": 0.6})
    if int(row.get("snap", 0)) == 1:
        d.append({"driver": f"SNAP issuance day ({row.state_id})", "kind": "rule", "weight": 0.4})
    return d


def model_drivers(row: pd.Series, history: pd.DataFrame, iforest=None, cols: list[str] | None = None, transform=None) -> list[dict]:
    cols = [c for c in (cols or FEATURE_COLS) if c in row.index and c in history.columns]
    out = []
    if _HAS_SHAP and iforest is not None and len(cols) == iforest.n_features_in_:
        try:
            expl = _explainer(iforest)
            x = transform(row.to_frame().T) if transform is not None else row[cols].astype(float).values.reshape(1, -1)
            sv = expl.shap_values(x)[0]
            order = np.argsort(sv)[:4]        # most negative shap = pushes toward anomalous for IsolationForest
            for i in order:
                out.append({"driver": FEATURE_LABEL.get(cols[i], cols[i]), "kind": "shap", "weight": float(abs(sv[i]))})
            return out
        except Exception:
            pass
    # fallback: contrast against the series' own recent history
    h = history[cols].astype(float)
    mu, sd = h.mean(), h.std().replace(0, np.nan)
    z = ((row[cols].astype(float) - mu) / sd).abs().fillna(0).sort_values(ascending=False)
    for c, v in z.head(4).items():
        out.append({"driver": f"{FEATURE_LABEL.get(c, c)} ({v:.1f}σ from series norm)", "kind": "contrast", "weight": float(v)})
    return out


def attribute(row: pd.Series, history: pd.DataFrame, iforest=None, cols: list[str] | None = None, k: int = 3, transform=None) -> list[dict]:
    drivers = rule_drivers(row)
    seen = {d["driver"] for d in drivers}
    for d in model_drivers(row, history, iforest, cols, transform):
        if len(drivers) >= k:
            break
        if d["driver"] not in seen:
            drivers.append(d); seen.add(d["driver"])
    return drivers[:k]
