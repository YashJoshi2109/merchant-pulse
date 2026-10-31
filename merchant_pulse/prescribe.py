"""Prescriptive layer: what should the merchant do next, and what is it worth.

Uplift is learned from the natural experiments already in the data: every historical price cut is a treated
series-day, every non-promo day a control. A T-learner (two gradient-boosted regressors, treated and control)
predicts baseline-relative lift; quantile models on the treated arm give an interval, and confidence is the
probability the incremental effect is positive. Actions are chosen per anomaly type from a small catalog and
ranked by expected incremental revenue over the horizon.

Production equivalent: EconML causal forest on BigQuery features, served from Vertex AI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor

from .detectors import HORIZON_DAYS

X_COLS = ["cat_FOODS", "cat_HOBBIES", "cat_HOUSEHOLD", "state_CA", "state_TX", "state_WI", "weekend", "snap", "event",
          "log_base", "price_pct"]

ACTIONS = {
    "drop": [("price_cut_10", "Price cut 10% for 7 days", -0.10), ("price_cut_20", "Price cut 20% for 7 days", -0.20),
             ("promo_display", "Promo display + 15% cut", -0.15)],
    "drift": [("price_cut_10", "Price cut 10% for 14 days", -0.10), ("price_cut_20", "Price cut 20% for 14 days", -0.20),
              ("promo_display", "Promo display + 15% cut", -0.15)],
}


def design(df: pd.DataFrame, price_pct: np.ndarray | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in ("FOODS", "HOBBIES", "HOUSEHOLD"):
        X[f"cat_{c}"] = (df.cat_id == c).astype(int)
    for s in ("CA", "TX", "WI"):
        X[f"state_{s}"] = (df.state_id == s).astype(int)
    num = lambda s: pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
    X["weekend"] = num(df.weekend).astype(int); X["snap"] = num(df.snap).astype(int); X["event"] = num(df.event).astype(int)
    X["log_base"] = np.log1p(num(df.rm28).clip(lower=0))
    X["price_pct"] = num(df.price_pct).values if price_pct is None else price_pct
    return X[X_COLS].astype(float)


@dataclass
class UpliftModel:
    control: GradientBoostingRegressor | None = None
    treated: GradientBoostingRegressor | None = None
    treated_lo: GradientBoostingRegressor | None = None
    treated_hi: GradientBoostingRegressor | None = None
    meta: dict = field(default_factory=dict)

    def fit(self, feats: pd.DataFrame, train_end: pd.Timestamp, seed: int = 42, max_rows: int = 40000) -> "UpliftModel":
        tr = feats[(feats.date < train_end) & feats.rm28.notna() & (feats.rm28 > 0.5) & (feats.event == 0)].copy()
        tr["lift"] = tr.units / (tr.rm28 + 0.5) - 1.0
        tr["lift"] = tr.lift.clip(-1, 5)
        treated = tr[tr.promo == 1]; control = tr[tr.promo == 0]
        if len(control) > max_rows:
            control = control.sample(max_rows, random_state=seed)
        if len(treated) > max_rows:
            treated = treated.sample(max_rows, random_state=seed)
        kw = dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
        self.control = GradientBoostingRegressor(**kw).fit(design(control), control.lift)
        self.treated = GradientBoostingRegressor(**kw).fit(design(treated), treated.lift)
        self.treated_lo = GradientBoostingRegressor(loss="quantile", alpha=0.1, **kw).fit(design(treated), treated.lift)
        self.treated_hi = GradientBoostingRegressor(loss="quantile", alpha=0.9, **kw).fit(design(treated), treated.lift)
        self.meta = {"n_treated": int(len(treated)), "n_control": int(len(control)), "train_end": str(train_end.date())}
        return self

    def predict_lift(self, rows: pd.DataFrame, price_pct: float) -> pd.DataFrame:
        """Incremental baseline-relative lift of applying price_pct vs doing nothing, with 80% interval."""
        Xc = design(rows, np.zeros(len(rows)))
        Xt = design(rows, np.full(len(rows), price_pct))
        c = self.control.predict(Xc)
        t = self.treated.predict(Xt); lo = self.treated_lo.predict(Xt); hi = self.treated_hi.predict(Xt)
        eff = t - c; sd = np.maximum((hi - lo) / 2.56, 0.05)
        return pd.DataFrame({"lift": eff, "lift_lo": lo - c, "lift_hi": hi - c, "p_positive": norm.cdf(eff / sd)}, index=rows.index)


def recommend(ep: pd.Series, feats_row: pd.Series, uplift: UpliftModel | None) -> dict:
    """One action per episode. Returns action, expected incremental units and revenue over the horizon, confidence."""
    base = float(max(feats_row.rm28 if pd.notna(feats_row.rm28) else ep.units_expected / max(ep.days, 1), 0.1))
    price = float(ep.price_ref if ep.price_ref > 0 else ep.price)
    horizon = HORIZON_DAYS.get(ep.anomaly_type, 3)
    t = ep.anomaly_type
    if t == "stockout":
        lost = base * horizon
        return {"action_code": "replenish", "action": "Replenish now; expedite if lead time > 2 days",
                "expected_units": lost, "expected_revenue": lost * price, "confidence": 0.9,
                "rationale": f"Series sells ~{base:.1f}/day; every day out of stock forgoes ~${base * price:.0f}."}
    if t == "price_error":
        gap = max(base - ep.units_actual / max(ep.days, 1), 0) * horizon
        return {"action_code": "fix_price", "action": f"Correct price file to ${price:.2f} and verify at register",
                "expected_units": gap, "expected_revenue": gap * price, "confidence": 0.95,
                "rationale": f"Shelf price is {ep.price / price:.1f}x the reference; volume collapses until it is fixed."}
    if t == "spike":
        cover = base * 7
        return {"action_code": "protect_stock", "action": "Check inventory cover for 7 days; hold price; confirm demand source",
                "expected_units": cover, "expected_revenue": cover * price, "confidence": 0.7,
                "rationale": "Unexplained demand spike; the risk is a stockout next week, not the spike itself."}
    # drop / drift: pick the action with the highest expected incremental revenue
    best = None
    if uplift is not None:
        for code, label, cut in ACTIONS.get(t, ACTIONS["drop"]):
            pred = uplift.predict_lift(feats_row.to_frame().T, cut).iloc[0]
            inc_units = base * pred.lift * horizon
            inc_rev = base * (1 + pred.lift) * price * (1 + cut) * horizon - base * price * horizon
            cand = {"action_code": code, "action": label, "expected_units": float(inc_units), "expected_revenue": float(inc_rev),
                    "confidence": float(pred.p_positive), "lift_pct": float(pred.lift * 100),
                    "rationale": f"Uplift model: {pred.lift * 100:+.0f}% units (80% interval {pred.lift_lo * 100:+.0f}% to {pred.lift_hi * 100:+.0f}%) at {abs(cut) * 100:.0f}% off."}
            if best is None or cand["expected_revenue"] > best["expected_revenue"]:
                best = cand
    if best is None or best["expected_revenue"] <= 0:
        return {"action_code": "hold", "action": "Hold price; review assortment and shelf placement; re-check in 7 days",
                "expected_units": 0.0, "expected_revenue": 0.0, "confidence": 0.5,
                "rationale": "No price action clears the margin bar; investigate non-price causes first."}
    return best
