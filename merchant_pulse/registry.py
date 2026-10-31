"""Model registry, promotion gate, canary and rollback (figure 2). Stand-in for Vertex AI Model Registry.

A version is a folder under registry/models/vN with the serialized ensemble, the uplift model, the eval report
and a lineage record. registry.json holds the version list, the production pointer and an event log. Rollback
is a pointer change, which is why it takes milliseconds here and minutes on Cloud Run.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd

from . import config, store
from .detectors import Ensemble
from .episodes import to_episodes
from .evaluate import detection_harness, gate, recommendation_harness
from .features import split_dates
from .prescribe import UpliftModel

REG = config.REGISTRY_DIR / "registry.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict:
    if REG.exists():
        return json.loads(REG.read_text())
    return {"versions": [], "production": None, "previous": None, "events": []}


def save(reg: dict) -> None:
    config.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(reg, indent=2, default=str))


def log_event(reg: dict, kind: str, version: str | None, detail: str) -> None:
    reg["events"].append({"ts": _now(), "event": kind, "version": version, "detail": detail})


def _file_hash(path) -> str:
    p = path if hasattr(path, "exists") else config.ROOT / path
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12] if p.exists() else "n/a"


def production() -> str | None:
    return load().get("production")


def get_version(v: str) -> dict | None:
    return next((x for x in load()["versions"] if x["version"] == v), None)


def load_models(v: str | None = None) -> tuple[Ensemble, UpliftModel]:
    v = v or production()
    if v is None:
        raise RuntimeError("no production model; run scripts/bootstrap.py")
    d = config.MODELS_DIR / v
    return joblib.load(d / "ensemble.joblib"), joblib.load(d / "uplift.joblib")


def _json_safe(report: dict) -> dict:
    out = {}
    for k, v in report.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.to_dict(orient="records")
        elif isinstance(v, (list, dict, str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def train_candidate(feats: pd.DataFrame, tables: dict, notes: str = "", fa_budget: float | None = None,
                    log=print) -> dict:
    """Validate → train → eval → gate. Returns the version record (status candidate|rejected)."""
    reg = load()
    n = len(reg["versions"]) + 1
    v = f"v{n}"
    d = split_dates(feats)
    t0 = time.time()
    # data validation
    daily = feats.groupby("date").units.sum()
    ref = daily.iloc[-35:-7]; last7 = daily.iloc[-7:]
    vol_z = float((last7.mean() - ref.mean()) / (ref.std() + 1e-9))
    nulls = int(feats[["units", "sell_price"]].isna().sum().sum())
    validation = {"rows": int(len(feats)), "series": int(feats.id.nunique()), "null_units_or_price": nulls,
                  "last7_volume_z_vs_28d": vol_z, "pass": nulls == 0 and abs(vol_z) < 3}
    log(f"[{v}] validate: {validation}")
    if not validation["pass"]:
        rec = {"version": v, "status": "rejected", "created": _now(), "validation": validation, "notes": notes}
        reg["versions"].append(rec); log_event(reg, "validation_failed", v, json.dumps(validation)); save(reg)
        return rec
    ens = Ensemble(fa_budget=fa_budget or config.PER_DETECTOR_FA).fit(feats, d["holdout_start"], d["calib_start"])
    log(f"[{v}] trained ensemble in {time.time() - t0:.1f}s; thresholds {ens.calibration['thresholds']}")
    up = UpliftModel().fit(feats, d["holdout_start"])
    log(f"[{v}] trained uplift: {up.meta}")
    t1 = time.time()
    det = detection_harness(tables["truth_sales_clean"], tables["hierarchy"], tables["calendar"], ens, d["holdout_start"], d["last"])
    rec_rep = recommendation_harness(feats, up, d["holdout_start"], tables.get("truth_elasticity"))
    g = gate(det, rec_rep)
    log(f"[{v}] eval in {time.time() - t1:.1f}s: F1={det['f1']:.2f} P={det['precision']:.2f} R={det['recall']:.2f} "
        f"FA/1k={det['fa_per_1k']:.2f} TTD={det['ttd_median']} sign_acc={rec_rep['sign_accuracy']:.2f} gate={'PASS' if g['pass'] else 'FAIL'}")
    vdir = config.MODELS_DIR / v
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ens, vdir / "ensemble.joblib"); joblib.dump(up, vdir / "uplift.joblib")
    (vdir / "eval.json").write_text(json.dumps({"detection": _json_safe(det), "recommendation": _json_safe(rec_rep), "gate": g},
                                               indent=2, default=str))
    lineage = {
        "data_snapshot": _file_hash(store.TABLES["sales"]), "feature_code": _file_hash(config.ROOT / "merchant_pulse" / "features.py"),
        "detector_code": _file_hash(config.ROOT / "merchant_pulse" / "detectors.py"), "train_end": str(d["holdout_start"].date()),
        "calibration": ens.calibration, "uplift": up.meta, "fa_budget": ens.fa_budget, "vote_min": ens.vote_min,
    }
    record = {
        "version": v, "status": "candidate" if g["pass"] else "rejected", "created": _now(), "notes": notes,
        "validation": validation, "lineage": lineage, "gate": g,
        "eval_summary": {"f1": det["f1"], "precision": det["precision"], "recall": det["recall"], "fa_per_1k": det["fa_per_1k"],
                         "ttd_median": det["ttd_median"], "sign_accuracy": rec_rep["sign_accuracy"], "pi_coverage_95": det["pi_coverage_95"],
                         "n_labels": det["n_labels"], "n_events": rec_rep["n_events"]},
        "train_seconds": round(time.time() - t0, 1),
    }
    reg["versions"].append(record)
    log_event(reg, "trained", v, f"gate {'passed' if g['pass'] else 'failed'}")
    if not g["pass"]:
        log_event(reg, "promotion_blocked", v, "; ".join(f"{c['check']}={c['value']:.2f} ({c['target']})" for c in g["checks"] if not c["pass"]))
    save(reg)
    return record


def register(v: str, approved_by: str = "demo-user", override: bool = False) -> dict:
    """Register a candidate. A failed gate can only be registered with override=True, and that is logged."""
    reg = load()
    rec = next(x for x in reg["versions"] if x["version"] == v)
    if not rec.get("gate", {}).get("pass") and not override:
        raise ValueError(f"{v} did not pass the gate; cannot register without an override")
    rec["status"] = "registered"; rec["approved_by"] = approved_by; rec["registered"] = _now(); rec["override"] = bool(override and not rec["gate"]["pass"])
    log_event(reg, "registered" if not rec["override"] else "registered_with_override", v, f"approved by {approved_by}")
    save(reg)
    return rec


def canary(v: str, feats: pd.DataFrame, days: int = 7, log=print) -> dict:
    """Shadow-score the last `days` with the candidate and the production model; compare alerts and latency."""
    reg = load()
    prod = reg.get("production")
    d = split_dates(feats)
    start = d["last"] - pd.Timedelta(days=days - 1)
    cand, _ = load_models(v)
    t = time.time(); sc_c = cand.score(feats, start, d["last"]); lat_c = (time.time() - t) * 1000
    ep_c = to_episodes(sc_c[sc_c.is_alert])
    report = {"version": v, "production": prod, "window_start": str(start.date()), "window_end": str(d["last"].date()),
              "candidate_alert_days": int(sc_c.is_alert.sum()), "candidate_episodes": int(len(ep_c)),
              "candidate_latency_ms_per_1k_rows": lat_c / max(len(sc_c), 1) * 1000}
    if prod and prod != v:
        pm, _ = load_models(prod)
        t = time.time(); sc_p = pm.score(feats, start, d["last"]); lat_p = (time.time() - t) * 1000
        ep_p = to_episodes(sc_p[sc_p.is_alert])
        a = set(zip(sc_c[sc_c.is_alert].id, sc_c[sc_c.is_alert].date)); b = set(zip(sc_p[sc_p.is_alert].id, sc_p[sc_p.is_alert].date))
        report.update({"production_alert_days": int(sc_p.is_alert.sum()), "production_episodes": int(len(ep_p)),
                       "alert_day_jaccard": len(a & b) / max(len(a | b), 1),
                       "production_latency_ms_per_1k_rows": lat_p / max(len(sc_p), 1) * 1000})
    report["pass"] = report["candidate_latency_ms_per_1k_rows"] < config.SLO["p95_latency_ms"] * 10 and \
        (report.get("alert_day_jaccard", 1.0) >= 0.3 or report.get("production_alert_days", 0) == 0)
    rec = next(x for x in reg["versions"] if x["version"] == v)
    rec["status"] = "canary"; rec["canary"] = report
    log_event(reg, "canary", v, f"jaccard={report.get('alert_day_jaccard', 'n/a')} pass={report['pass']}")
    save(reg)
    log(f"[{v}] canary: {report}")
    return report


def promote(v: str, reason: str = "canary passed") -> dict:
    reg = load()
    rec = next(x for x in reg["versions"] if x["version"] == v)
    if reg.get("production") and reg["production"] != v:
        prev = next(x for x in reg["versions"] if x["version"] == reg["production"])
        prev["status"] = "previous"
        reg["previous"] = reg["production"]
    reg["production"] = v; rec["status"] = "production"; rec["promoted"] = _now()
    log_event(reg, "promoted", v, reason)
    save(reg)
    return rec


def rollback(reason: str = "manual") -> dict:
    reg = load()
    t = time.time()
    cur, prev = reg.get("production"), reg.get("previous")
    if not prev:
        raise ValueError("no previous version to roll back to")
    for x in reg["versions"]:
        if x["version"] == cur:
            x["status"] = "rolled_back"
        if x["version"] == prev:
            x["status"] = "production"
    reg["production"], reg["previous"] = prev, cur
    ms = (time.time() - t) * 1000
    log_event(reg, "rollback", prev, f"from {cur} to {prev} in {ms:.0f} ms; reason: {reason}")
    save(reg)
    return {"from": cur, "to": prev, "ms": ms}


def versions_table() -> pd.DataFrame:
    reg = load()
    rows = []
    for x in reg["versions"]:
        es = x.get("eval_summary", {})
        rows.append({"version": x["version"], "status": x["status"], "created": x["created"][:19], "gate": "pass" if x.get("gate", {}).get("pass") else "fail",
                     "f1": es.get("f1"), "precision": es.get("precision"), "recall": es.get("recall"), "fa_per_1k": es.get("fa_per_1k"),
                     "ttd_median": es.get("ttd_median"), "sign_acc": es.get("sign_accuracy"), "approved_by": x.get("approved_by", ""),
                     "notes": x.get("notes", "")})
    return pd.DataFrame(rows)
