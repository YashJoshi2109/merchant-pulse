"""Nightly run (figure 5): features → score → episodes → reconcile → attribute → prescribe → narrate → write.

Every step is keyed on the run date, so a rerun replaces that run's alerts instead of duplicating them.
If the narrative step fails the alert still ships with drivers and action; the note says so.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

from . import attribution, config, monitoring, narrative, prescribe, registry, store
from .episodes import to_episodes
from .features import build_features, split_dates
from .reconcile import reconcile

ALERT_COLS_JSON = ["drivers", "recommendation"]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_tables() -> dict:
    t = {k: store.read(k) for k in ("sales", "calendar", "hierarchy", "truth_sales_clean", "truth_elasticity")}
    if t["sales"] is None:
        raise RuntimeError("no data; run scripts/bootstrap.py")
    return t


def nightly(run_date: pd.Timestamp | None = None, backfill_days: int = 1, use_gemini: bool = False, log=print,
            version: str | None = None) -> dict:
    steps = []
    def step(name, fn):
        t = time.time(); out = fn(); dt = time.time() - t
        steps.append({"step": name, "seconds": round(dt, 2)}); log(f"  {name:28s} {dt:6.2f}s")
        return out

    tables = step("load tables", load_tables)
    feats = step("build features (Dataform)", lambda: build_features(tables["sales"], tables["calendar"]))
    d = split_dates(feats)
    run_date = pd.Timestamp(run_date) if run_date is not None else d["last"]
    start = run_date - pd.Timedelta(days=backfill_days - 1)
    v = version or registry.production()
    ens, uplift = step("load production models", lambda: registry.load_models(v))
    sc = step("score ensemble", lambda: ens.score(feats, start, run_date))
    ep = step("episodes", lambda: to_episodes(sc[sc.is_alert]))
    rec = step("hierarchical reconciliation", lambda: reconcile(ep))

    def enrich():
        if rec.empty:
            return rec
        sci = sc.set_index(["id", "date"])
        drivers_col, rec_col, note_col, src_col = [], [], [], []
        for _, e in rec.iterrows():
            if e.level == "item":
                row = sci.loc[(e.id, e.peak_date)]
                hist = ens._adjust(feats[(feats.id == e.id) & (feats.date < e.start)].tail(28))
                drivers = attribution.attribute(row, hist, ens.iforest.model, ens.iforest.cols, transform=ens.iforest._transform)
                r = prescribe.recommend(e, row, uplift)
            else:
                n = len(str(e.children).split(","))
                what = "category" if e.level == "category" else "store"
                drivers = [{"driver": f"{n} {'items' if e.level == 'category' else 'categories'} in this {what} moved {e.direction} together", "kind": "reconciliation", "weight": 1.0},
                           {"driver": "pattern is systemic, not item-specific", "kind": "reconciliation", "weight": 0.8}]
                if e.event_name:
                    drivers.append({"driver": f"event: {e.event_name}", "kind": "rule", "weight": 0.6})
                r = {"action_code": "investigate_systemic", "action": f"Review {what}-level cause: supply, planogram, local competitor, store hours",
                     "expected_units": float(abs(e.units_expected - e.units_actual)), "expected_revenue": float(e.revenue_at_risk),
                     "confidence": 0.6, "rationale": "Several series moved together; fixing one item will not fix this."}
            try:
                note, src = narrative.write_note(e, drivers, r, use_gemini=use_gemini)
            except Exception as ex:  # narrative must never block the alert
                note, src = f"Narrative pending ({type(ex).__name__}). Drivers: " + "; ".join(x["driver"] for x in drivers) + f". Action: {r['action']}.", "pending"
            drivers_col.append(json.dumps(drivers)); rec_col.append(json.dumps(r)); note_col.append(note); src_col.append(src)
        out = rec.copy()
        out["drivers"] = drivers_col; out["recommendation"] = rec_col; out["note"] = note_col; out["note_source"] = src_col
        out["action"] = [json.loads(x)["action"] for x in rec_col]
        out["expected_revenue"] = [json.loads(x).get("expected_revenue", 0.0) for x in rec_col]
        out["confidence"] = [json.loads(x).get("confidence", 0.0) for x in rec_col]
        out["model_version"] = v; out["run_date"] = run_date; out["written_at"] = _ts()
        return out
    alerts = step("attribute + prescribe + narrate", enrich)

    def write():
        cur = store.read("alerts")
        if cur is not None and len(cur):
            # idempotent: drop previous alerts whose peak falls inside this run's window, keep the rest
            keep = ~((pd.to_datetime(cur.peak_date) >= start) & (pd.to_datetime(cur.peak_date) <= run_date))
            # also drop stale copies of any alert_id we are rewriting
            if len(alerts):
                keep &= ~cur.alert_id.isin(alerts.alert_id)
            cur = cur[keep]
            out = pd.concat([cur, alerts], ignore_index=True) if len(alerts) else cur
        else:
            out = alerts
        store.write("alerts", out)
        store.upsert_by_date("scores", sc.assign(run_date=run_date), date_col="date")
        return len(alerts)
    n_written = step("write alerts table", write)
    snap = step("monitoring snapshot", lambda: monitoring.snapshot(feats, store.read("alerts"), store.read("feedback"), d["holdout_start"], run_date, v))
    run = {"run_date": run_date, "window_start": start, "model_version": v, "alerts_written": int(n_written),
           "alert_days": int(sc.is_alert.sum()), "series_scored": int(sc.id.nunique()), "rows_scored": int(len(sc)),
           "total_seconds": round(sum(s["seconds"] for s in steps), 2), "steps": json.dumps(steps), "ts": _ts(),
           "retrain_trigger": bool(snap["retrain_trigger"]), "trigger_reasons": snap["trigger_reasons"]}
    store.append("runs", pd.DataFrame([run]))
    log(f"nightly run {run_date.date()}: {n_written} alerts, {run['total_seconds']}s, model {v}")
    return {"run": run, "steps": steps, "alerts": alerts, "scores": sc}


def alerts_with_status() -> pd.DataFrame:
    a = store.read("alerts")
    if a is None or a.empty:
        return pd.DataFrame()
    fb = store.read("feedback")
    a = a.copy(); a["status"] = "open"
    if fb is not None and len(fb):
        last = fb.sort_values("ts").drop_duplicates("alert_id", keep="last").set_index("alert_id").verdict
        a["status"] = a.alert_id.map(last).fillna("open")
    return a


def record_feedback(alert_id: str, verdict: str, user: str = "merchant", comment: str = "") -> None:
    store.append("feedback", pd.DataFrame([{"alert_id": alert_id, "verdict": verdict, "user": user, "comment": comment, "ts": _ts()}]))
