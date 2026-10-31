import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import PALETTE, annotate, setup
from merchant_pulse import config, registry

setup("Eval Harness", "🧪")
st.title("5 · Evaluation harness and gate")
annotate("Two tracks, one gate",
         "**Detection quality**: take the clean holdout (last 90 days) for half the stores, inject labeled anomalies with material magnitudes "
         "(2–3 day +2× baseline spikes, 7–10 day 65% drops, 28-day −4%/day drifts, 3–10 day stockouts, 10× price errors) into series selling "
         "≥ 6 units/day, run the ensemble with production thresholds, and score episodes against labels: precision, recall, F1 by type, false alerts "
         "per 1k series-days, median time to detect, and 95% interval coverage on clean days. **Recommendation quality**: every holdout promotion "
         "is a natural experiment; predicted lift vs realized lift net of a matched control (same store and category, no promo). "
         "The gate reads both reports: F1 ≥ 0.85, FA ≤ 5/1k, TTD ≤ 2 days, sign accuracy ≥ 0.80.",
         "The same harness runs as the eval step of the Vertex AI pipeline; the report is attached to the Model Registry version and the gate "
         "component blocks registration. In production, precision is additionally measured from merchant feedback (see MLOps).", expanded=True)

reg = registry.load()
versions = [v["version"] for v in reg["versions"] if (config.MODELS_DIR / v["version"] / "eval.json").exists()]
if not versions:
    st.info("No evaluated versions yet."); st.stop()
v = st.selectbox("Model version", versions, index=len(versions) - 1)
rep = json.loads((config.MODELS_DIR / v / "eval.json").read_text())
det, rec, g = rep["detection"], rep["recommendation"], rep["gate"]

st.subheader("Gate")
cols = st.columns(len(g["checks"]))
for c, chk in zip(cols, g["checks"]):
    val = chk["value"]
    c.metric(chk["check"], f"{val:.2f}" if isinstance(val, (int, float)) and val == val else "n/a", ("PASS " if chk["pass"] else "FAIL ") + chk["target"],
             delta_color="normal" if chk["pass"] else "inverse")
st.markdown(f"**Gate result: {'PASS' if g['pass'] else 'FAIL'}**" + ("" if g["pass"] else " — promotion blocked (unless an approver logs an override)."))

st.subheader("Detection track")
a1, a2, a3, a4, a5, a6 = st.columns(6)
a1.metric("Labels injected", det["n_labels"]); a2.metric("Series-days", f"{det['n_series_days']:,}")
a3.metric("Precision", f"{det['precision']:.2f}"); a4.metric("Recall", f"{det['recall']:.2f}")
a5.metric("False alerts / 1k", f"{det['fa_per_1k']:.2f}"); a6.metric("95% PI coverage (clean days)", f"{det['pi_coverage_95']:.3f}")
by = pd.DataFrame(det["by_type"])
c1, c2 = st.columns([1, 1])
with c1:
    st.dataframe(by.rename(columns={"n": "injected", "recall": "recall", "ttd_median": "median TTD (days)", "type_acc": "type labeled correctly"}).round(2),
                 hide_index=True, width="stretch")
    st.caption(f"Holdout stores: {', '.join(det['holdout_stores'])} · {det['n_series']} series · TP {det['tp']} · FP {det['fp']} · FN {det['fn']}")
with c2:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=by.anomaly_type, y=by.recall, name="recall", marker_color=[PALETTE.get(t, "#1f6feb") for t in by.anomaly_type]))
    fig.add_trace(go.Scatter(x=by.anomaly_type, y=by.ttd_median, name="median TTD (days)", yaxis="y2", mode="markers+lines", line=dict(color="#8b949e")))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 1.05]), yaxis2=dict(overlaying="y", side="right"),
                      legend=dict(orientation="h", y=-0.25), title="Recall and time to detect by type")
    st.plotly_chart(fig, width="stretch")
with st.expander("Injected labels and outcome"):
    lab = pd.DataFrame(det["labels"]); st.dataframe(lab, hide_index=True, width="stretch", height=300)
with st.expander(f"False episodes ({det['fp']})"):
    fp = pd.DataFrame(det["false_episodes"])
    if len(fp):
        st.dataframe(fp[["start", "end", "id", "anomaly_type", "z_resid", "z_scale", "detectors", "units_actual", "units_expected", "price", "price_ref"]].round(2),
                     hide_index=True, width="stretch", height=300)
    st.caption("Most false episodes are statistically real deviations that no label covers: promotions on items more elastic than estimated, "
               "post-anomaly recovery, or heavy-tailed noise at the configured budget. They are listed, not hidden.")

st.subheader("Recommendation track")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Holdout promo events", rec["n_events"]); r2.metric("Sign accuracy", f"{rec['sign_accuracy']:.2f}")
r3.metric("MAE of lift", f"{rec['mae']:.2f}"); r4.metric("Corr. predicted vs realized", f"{rec['corr']:.2f}")
ev = pd.DataFrame(rec["events"])
if len(ev):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ev.predicted_lift, y=ev.realized_lift_adj, mode="markers", marker=dict(color="#1f6feb", opacity=0.6), name="event",
                             text=ev.id, hovertemplate="%{text}<br>pred %{x:.2f} realized %{y:.2f}"))
    lim = [min(ev.predicted_lift.min(), ev.realized_lift_adj.min()) - 0.1, max(ev.predicted_lift.max(), ev.realized_lift_adj.max()) + 0.1]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#8b949e", dash="dash"), name="y = x"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="predicted lift (uplift model)", yaxis_title="realized lift net of matched control",
                      title="Each holdout promotion: predicted vs realized")
    st.plotly_chart(fig, width="stretch")
    if "corr_with_true_elasticity" in rec and rec["corr_with_true_elasticity"] == rec["corr_with_true_elasticity"]:
        st.caption(f"Because the data is synthetic, the generator's true elasticity is known: correlation of predicted lift with the elasticity-implied lift = {rec['corr_with_true_elasticity']:.2f}. "
                   "This column does not exist with real data and is shown only as a check.")
