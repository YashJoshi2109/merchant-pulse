"""Evaluation harness (figure 4) and the promotion gate (figure 2).

Detection track: take the clean holdout for a share of stores, inject labeled anomalies, run the production
ensemble with production thresholds, score episodes against labels. Recommendation track: every historical
price promotion in the holdout is a natural experiment; compare the uplift model's predicted lift with the
realized lift net of a matched control. Both reports are attached to the model version; the gate reads them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, synth
from .episodes import to_episodes
from .features import build_features

GRACE_DAYS = {"spike": 2, "drop": 3, "drift": 7, "stockout": 3, "price_error": 2}


def _match(ep: pd.DataFrame, lab: pd.Series) -> pd.DataFrame:
    g = GRACE_DAYS.get(lab.anomaly_type, 3)
    m = (ep.id == lab.id) & (ep.end >= lab.start - pd.Timedelta(days=1)) & (ep.start <= lab.end + pd.Timedelta(days=g))
    return ep[m]


def detection_harness(sales_clean: pd.DataFrame, hierarchy: pd.DataFrame, calendar: pd.DataFrame, ensemble,
                      holdout_start: pd.Timestamp, last: pd.Timestamp, seed: int = config.SEED, store_frac: float = 0.5,
                      n_per_type: int = 6) -> dict:
    rng = np.random.default_rng(seed + 7)
    stores = sorted(hierarchy.store_id.unique())
    hold = sorted(rng.choice(stores, max(2, int(round(len(stores) * store_frac))), replace=False).tolist())
    ids_hold = hierarchy[hierarchy.store_id.isin(hold)].id.tolist()
    inj_end = last - pd.Timedelta(days=7)
    s_inj, labels = synth.inject_anomalies(sales_clean, ids_hold, holdout_start, inj_end, rng, n_per_type=n_per_type)
    feats = build_features(s_inj.merge(hierarchy, on="id"), calendar)
    sc = ensemble.score(feats, holdout_start, last)
    sc = sc[sc.store_id.isin(hold)]
    ep = to_episodes(sc[sc.is_alert])
    n_series_days = int(sc.shape[0])
    rows, matched_ids = [], set()
    for _, lab in labels.iterrows():
        m = _match(ep, lab) if len(ep) else ep
        hit = len(m) > 0
        if hit:
            matched_ids.update(m.alert_id.tolist())
        rows.append({"id": lab.id, "anomaly_type": lab.anomaly_type, "start": lab.start, "end": lab.end, "detected": hit,
                     "ttd_days": int(max(0, (m.start.min() - lab.start).days)) if hit else None,
                     "detected_as": ",".join(sorted(set(m.anomaly_type))) if hit else "",
                     "type_correct": bool(hit and (lab.anomaly_type in set(m.anomaly_type)))})
    res = pd.DataFrame(rows)
    fp = ep[~ep.alert_id.isin(matched_ids)] if len(ep) else ep
    tp = int(res.detected.sum()); fn = int((~res.detected).sum()); n_fp = int(len(fp))
    precision = tp / (tp + n_fp) if tp + n_fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    by_type = res.groupby("anomaly_type").agg(n=("detected", "size"), recall=("detected", "mean"),
                                              ttd_median=("ttd_days", "median"), type_acc=("type_correct", "mean")).reset_index()
    # calibration: on clean series-days (outside any label window) the 1-day test should sit inside +-1.96 about 95% of the time
    in_label = np.zeros(len(sc), dtype=bool)
    for _, lab in labels.iterrows():
        in_label |= ((sc.id == lab.id) & (sc.date >= lab.start) & (sc.date <= lab.end + pd.Timedelta(days=3))).values
    clean = sc[~in_label & (sc.rm28 >= 1.0)]
    coverage = float((clean.z_1d.abs() < 1.96).mean()) if len(clean) else float("nan")
    return {
        "holdout_stores": hold, "n_series": len(ids_hold), "n_series_days": n_series_days, "n_labels": int(len(labels)),
        "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": n_fp, "fn": fn,
        "fa_per_1k": n_fp / n_series_days * 1000 if n_series_days else float("nan"),
        "ttd_median": float(res[res.detected].ttd_days.median()) if tp else float("nan"),
        "pi_coverage_95": coverage, "by_type": by_type, "labels": res, "false_episodes": fp, "episodes": ep,
    }


def recommendation_harness(feats: pd.DataFrame, uplift, holdout_start: pd.Timestamp, truth_elasticity: pd.DataFrame | None = None) -> dict:
    """Score the uplift model on holdout promotions against matched controls."""
    f = feats[feats.date >= holdout_start].sort_values(["id", "date"]).copy()
    f["promo_run"] = (f.promo.diff().fillna(0) != 0).groupby(f.id).cumsum()
    events = []
    truth = truth_elasticity.set_index("id").elasticity.to_dict() if truth_elasticity is not None else {}
    for (sid, run), g in f[f.promo == 1].groupby(["id", "promo_run"]):
        if len(g) < 3:
            continue
        start = g.date.min(); end = g.date.max()
        first = g.iloc[0]
        if pd.isna(first.rm28) or first.rm28 < 1.0:
            continue
        cut = float(g.price_pct.mean())
        realized = float(g.units.mean() / (first.rm28 + 0.5) - 1.0)
        ctrl = f[(f.cat_id == first.cat_id) & (f.store_id == first.store_id) & (f.id != sid) & (f.date >= start) & (f.date <= end)]
        ctrl = ctrl[ctrl.promo == 0]
        ctrl_change = float((ctrl.units / (ctrl.rm28 + 0.5) - 1.0).mean()) if len(ctrl) >= 5 else 0.0
        pred = uplift.predict_lift(first.to_frame().T, cut).iloc[0]
        events.append({"id": sid, "start": start, "end": end, "days": int(len(g)), "price_cut_pct": cut * 100,
                       "predicted_lift": float(pred.lift), "realized_lift": realized, "control_change": ctrl_change,
                       "realized_lift_adj": realized - ctrl_change,
                       "true_elasticity_lift": float((1 + cut) ** truth[sid] - 1) if sid in truth else np.nan})
    ev = pd.DataFrame(events)
    if ev.empty:
        return {"n_events": 0, "sign_accuracy": float("nan"), "mae": float("nan"), "corr": float("nan"), "events": ev}
    sign_acc = float(((ev.predicted_lift > 0) == (ev.realized_lift_adj > 0)).mean())
    mae = float((ev.predicted_lift - ev.realized_lift_adj).abs().mean())
    corr = float(ev[["predicted_lift", "realized_lift_adj"]].corr().iloc[0, 1]) if len(ev) > 3 else float("nan")
    corr_truth = float(ev[["predicted_lift", "true_elasticity_lift"]].corr().iloc[0, 1]) if ev.true_elasticity_lift.notna().sum() > 3 else float("nan")
    return {"n_events": int(len(ev)), "sign_accuracy": sign_acc, "mae": mae, "corr": corr, "corr_with_true_elasticity": corr_truth,
            "events": ev}


def gate(det: dict, rec: dict, thresholds: dict | None = None) -> dict:
    th = thresholds or config.GATE
    checks = [
        {"check": "F1 on injected anomalies", "value": det["f1"], "target": f">= {th['f1_min']}", "pass": det["f1"] >= th["f1_min"]},
        {"check": "False alerts per 1k series-days", "value": det["fa_per_1k"], "target": f"<= {th['fa_per_1k_max']}", "pass": det["fa_per_1k"] <= th["fa_per_1k_max"]},
        {"check": "Median time to detect (days)", "value": det["ttd_median"], "target": f"<= {th['ttd_median_max']}",
         "pass": bool(np.isfinite(det["ttd_median"]) and det["ttd_median"] <= th["ttd_median_max"])},
        {"check": "Recommendation sign accuracy", "value": rec["sign_accuracy"], "target": f">= {th['sign_acc_min']}",
         "pass": bool(np.isfinite(rec["sign_accuracy"]) and rec["sign_accuracy"] >= th["sign_acc_min"])},
    ]
    return {"pass": all(c["pass"] for c in checks), "checks": checks}
