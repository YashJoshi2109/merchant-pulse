import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import annotate, features, read, setup
from merchant_pulse import config, pipeline, registry, store

setup("MLOps", "🔁")
st.title("6 · MLOps loop")
annotate("Train → gate → register → canary → promote → monitor → retrain, with rollback",
         "`registry.py` keeps versions under `registry/models/vN` (serialized ensemble + uplift model, eval report, lineage: data snapshot hash, "
         "feature and detector code hashes, thresholds, approver). **Train candidate** validates data, trains, runs the harness and the gate. "
         "**Register** needs a passed gate (or an explicit, logged override). **Canary** shadow-scores the last 7 days with candidate and production "
         "and compares alert overlap and latency. **Promote** moves the production pointer; **Rollback** moves it back, in milliseconds here. "
         "**Monitoring** computes PSI drift of key features (last 7 days vs training reference), alert volume, and precision from merchant feedback; "
         "any breach raises the retrain trigger.",
         "Cloud Build compiles the Vertex AI pipeline; the gate component blocks registration in Vertex AI Model Registry; Cloud Run serves two "
         "revisions with a 10/90 traffic split for 48 h; promotion is a traffic change; rollback re-points traffic to the previous revision in under "
         "5 minutes. Vertex AI Model Monitoring + Cloud Monitoring alert policies trigger the pipeline on drift or precision drop; a weekly Cloud "
         "Scheduler run is the floor.", expanded=True)

reg = registry.load()
vt = registry.versions_table()
st.subheader("Registry")
c1, c2 = st.columns([3, 1])
with c1:
    if len(vt):
        st.dataframe(vt.round(3), hide_index=True, width="stretch")
    else:
        st.info("Empty registry.")
with c2:
    st.metric("Production", reg.get("production") or "none"); st.metric("Previous", reg.get("previous") or "none")

st.subheader("Actions")
b1, b2, b3, b4, b5 = st.columns(5)
log_box = st.empty()
lines = []
def log(s):
    lines.append(s); log_box.code("\n".join(lines[-14:]))

with b1:
    fa = st.select_slider("Candidate FA budget", options=[0.005, 0.01, 0.02], value=config.PER_DETECTOR_FA, help="Per-detector flag rate on the calibration slice")
    if st.button("Train candidate", type="primary"):
        with st.spinner("training, evaluating, gating…"):
            tables = pipeline.load_tables()
            rec = registry.train_candidate(features(), tables, notes=f"trained from UI, fa_budget={fa}", fa_budget=fa, log=log)
        st.success(f"{rec['version']}: {rec['status']}"); st.rerun()
cands = [v["version"] for v in reg["versions"] if v["status"] in ("candidate", "rejected")]
regd = [v["version"] for v in reg["versions"] if v["status"] in ("registered", "canary")]
with b2:
    v_reg = st.selectbox("Register", cands or ["-"])
    override = st.checkbox("Override failed gate", help="Break-glass. Logged with your name; use only when there is no production baseline.")
    if st.button("Register") and v_reg != "-":
        try:
            registry.register(v_reg, approved_by="ui-approver", override=override); st.success(f"{v_reg} registered"); st.rerun()
        except ValueError as ex:
            st.error(str(ex))
with b3:
    v_can = st.selectbox("Canary", regd or ["-"])
    if st.button("Run canary (7 days shadow)") and v_can != "-":
        with st.spinner("shadow scoring…"):
            rep = registry.canary(v_can, features(), log=log)
        st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()}); 
with b4:
    v_pro = st.selectbox("Promote", [v["version"] for v in reg["versions"] if v["status"] in ("canary", "registered")] or ["-"])
    if st.button("Promote to production") and v_pro != "-":
        registry.promote(v_pro, reason="promoted from UI"); st.success(f"{v_pro} is production"); st.rerun()
with b5:
    st.write("")
    if st.button("⏪ Rollback to previous", help="Re-point production to the previous version"):
        try:
            r = registry.rollback(reason="manual rollback from UI"); st.success(f"rolled back {r['from']} → {r['to']} in {r['ms']:.0f} ms"); st.rerun()
        except ValueError as ex:
            st.error(str(ex))

prod = reg.get("production")
if prod:
    rec = registry.get_version(prod)
    with st.expander(f"Lineage of {prod}"):
        st.json({k: rec.get(k) for k in ("created", "approved_by", "override", "promoted", "notes", "validation", "lineage", "eval_summary")})

st.subheader("Monitoring")
mon = read("monitoring")
if mon is not None and len(mon):
    last = mon.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alert episodes / 1k series-days (7d)", f"{last.alert_episodes_per_1k_series_days_7d:.2f}", f"budget {config.GATE['fa_per_1k_max']}")
    m2.metric("Max alerts / store / day", f"{last.max_alerts_per_store_per_day:.1f}", f"cap {config.SLO['alerts_per_merchant_day']}")
    fp = last.feedback_precision
    m3.metric("Precision from feedback (28d)", f"{fp:.2f}" if fp == fp else "no feedback yet", f"{int(last.feedback_confirmed)} confirmed / {int(last.feedback_dismissed)} dismissed")
    m4.metric("Retrain trigger", "YES" if last.retrain_trigger else "no", last.trigger_reasons or "all signals inside limits", delta_color="inverse")
    psi_cols = [c for c in mon.columns if c.startswith("psi_")]
    fig = go.Figure(go.Bar(x=[c[4:] for c in psi_cols], y=[last[c] for c in psi_cols], marker_color=["#cf222e" if last[c] > config.PSI_RETRAIN else "#1f6feb" for c in psi_cols]))
    fig.add_hline(y=config.PSI_RETRAIN, line=dict(color="#cf222e", dash="dot"), annotation_text=f"retrain threshold {config.PSI_RETRAIN}")
    fig.update_layout(title="Feature drift: PSI, last 7 days vs training reference", height=280, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Monitoring snapshots"):
        st.dataframe(mon.round(3), hide_index=True, width="stretch")
    annotate("How to read the monitor",
             "PSI < 0.1 stable, 0.1–0.2 watch, > 0.2 retrain. Alert rate above 3× the false-alert budget or feedback precision under 0.75 "
             "(with at least 10 verdicts) also trips the trigger. The weekly schedule retrains regardless.",
             "Vertex AI Model Monitoring computes feature skew/drift against the training baseline; Cloud Monitoring alert policies on the custom "
             "metrics (alert rate, feedback precision) call the pipeline trigger via Pub/Sub → Cloud Function.")
else:
    st.info("No monitoring snapshots yet; run the nightly pipeline.")

st.subheader("Event log")
ev = pd.DataFrame(reg.get("events", []))
if len(ev):
    st.dataframe(ev.iloc[::-1], hide_index=True, width="stretch", height=260)
