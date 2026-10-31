import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import PALETTE, TYPE_ORDER, annotate, hero, setup, style
from merchant_pulse import config, registry

setup("Eval Harness", "🧪")
hero("5 · Evaluation harness and gate", "Does it find real anomalies without crying wolf, and do the recommendations move sales? "
     "Injected labels give exact precision and recall; historical promotions give a matched-control read on lift. The gate reads both.",
     ["Injected anomalies", "Matched-control lift", "F1 · FA/1k · TTD · sign accuracy", "Blocks promotion"])
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
    by = by.set_index("anomaly_type").reindex([t for t in TYPE_ORDER if t in set(by.anomaly_type)]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=by.anomaly_type, y=by.recall, name="recall", marker_color=[PALETTE.get(t, "#4c9aff") for t in by.anomaly_type],
                         text=[f"{r:.0%}" for r in by.recall], textposition="outside"))
    fig.add_trace(go.Scatter(x=by.anomaly_type, y=by.ttd_median, name="median TTD (days)", yaxis="y2", mode="markers+lines",
                             line=dict(color="#c9d1d9"), marker=dict(size=9)))
    fig.update_layout(yaxis=dict(range=[0, 1.15]), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="days"))
    st.plotly_chart(style(fig, 300, "Recall and time to detect by type"), width="stretch")
e1, e2, e3 = st.columns(3)
with e1:
    fig = go.Figure(go.Bar(x=["precision", "recall", "F1"], y=[det["precision"], det["recall"], det["f1"]],
                           marker_color=[PALETTE["actual"], PALETTE["ok"], PALETTE["alert"]], text=[f"{det['precision']:.2f}", f"{det['recall']:.2f}", f"{det['f1']:.2f}"], textposition="outside"))
    fig.add_hline(y=config.GATE["f1_min"], line=dict(color="#8b949e", dash="dot"), annotation_text=f"F1 gate {config.GATE['f1_min']}", annotation_font_size=10)
    fig.update_layout(yaxis=dict(range=[0, 1.15]))
    st.plotly_chart(style(fig, 280, "Detection quality", legend=False), width="stretch")
with e2:
    lab = pd.DataFrame(det["labels"])
    hit = lab[lab.detected]
    fig = go.Figure(go.Histogram(x=hit.ttd_days, nbinsx=15, marker_color=PALETTE["actual"]))
    fig.add_vline(x=config.GATE["ttd_median_max"], line=dict(color=PALETTE["alert"], dash="dot"), annotation_text="TTD gate", annotation_font_size=10)
    fig.update_layout(xaxis_title="days to detect", yaxis_title="anomalies")
    st.plotly_chart(style(fig, 280, "Time to detect, all detected anomalies", legend=False), width="stretch")
with e3:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=det["fa_per_1k"], number=dict(suffix=" /1k"),
                                 gauge=dict(axis=dict(range=[0, 10]), bar=dict(color=PALETTE["alert"] if det["fa_per_1k"] > config.GATE["fa_per_1k_max"] else PALETTE["ok"]),
                                            steps=[dict(range=[0, config.GATE["fa_per_1k_max"]], color="rgba(63,185,80,0.15)"), dict(range=[config.GATE["fa_per_1k_max"], 10], color="rgba(255,107,107,0.15)")],
                                            threshold=dict(line=dict(color="#c9d1d9", width=2), value=config.GATE["fa_per_1k_max"]))))
    st.plotly_chart(style(fig, 280, "False alerts per 1k series-days (gate ≤ 5)", legend=False), width="stretch")
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
    s1, s2 = st.columns([3, 2])
    with s1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ev.predicted_lift, y=ev.realized_lift_adj, mode="markers",
                                 marker=dict(color=ev.price_cut_pct, colorscale=[[0, PALETTE["alert"]], [1, PALETTE["actual"]]], opacity=0.75, size=8,
                                             colorbar=dict(title="price cut %", thickness=10)), name="promo event",
                                 text=ev.id, hovertemplate="%{text}<br>pred %{x:.2f} realized %{y:.2f}"))
        lim = [min(ev.predicted_lift.min(), ev.realized_lift_adj.min()) - 0.1, max(ev.predicted_lift.max(), ev.realized_lift_adj.max()) + 0.1]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#8b949e", dash="dash"), name="y = x"))
        fig.update_layout(xaxis_title="predicted lift (uplift model)", yaxis_title="realized lift net of matched control")
        st.plotly_chart(style(fig, 360, "Each holdout promotion: predicted vs realized"), width="stretch")
    with s2:
        ev["cut_bin"] = pd.cut(-ev.price_cut_pct, bins=[0, 12, 18, 24, 35], labels=["10–12%", "12–18%", "18–24%", "24–30%"])
        bb = ev.groupby("cut_bin", observed=True).agg(pred=("predicted_lift", "mean"), real=("realized_lift_adj", "mean"), n=("id", "size")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bb.cut_bin.astype(str), y=bb.real * 100, name="realized lift %", marker_color=PALETTE["actual"]))
        fig.add_trace(go.Bar(x=bb.cut_bin.astype(str), y=bb.pred * 100, name="predicted lift %", marker_color=PALETTE["alert"]))
        fig.update_layout(barmode="group", xaxis_title="price cut", yaxis_title="% units lift")
        st.plotly_chart(style(fig, 360, "Lift by depth of price cut"), width="stretch")
    if "corr_with_true_elasticity" in rec and rec["corr_with_true_elasticity"] == rec["corr_with_true_elasticity"]:
        st.caption(f"Because the data is synthetic, the generator's true elasticity is known: correlation of predicted lift with the elasticity-implied lift = {rec['corr_with_true_elasticity']:.2f}. "
                   "This column does not exist with real data and is shown only as a check.")
