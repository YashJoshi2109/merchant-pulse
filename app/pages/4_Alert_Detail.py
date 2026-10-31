import json

import pandas as pd
import streamlit as st

from common import alerts, annotate, dates, features, fmt_money, hero, parse_json, read, series_chart, setup
from merchant_pulse import pipeline, store

setup("Alert Detail", "🧾")
hero("4 · Alert detail", "One alert the way a merchant sees it: what happened, why, what to do, what it is worth. Confirm or dismiss feeds the next training run.",
     ["3 drivers", "1 priced action", "3-sentence note", "Feedback = labels"])
annotate("From alert to action",
         "Each episode carries three drivers (rules first: same-day price change, SNAP day, holiday, zero-sales run; then SHAP on the Isolation "
         "Forest row), one recommended action from a catalog chosen by expected incremental revenue (a T-learner uplift model predicts the lift of a "
         "price cut from the data's own historical promotions, with an 80% interval and a probability the effect is positive), and a three-sentence "
         "note from a template (Gemini if `GEMINI_API_KEY` is set; it may only rephrase). Confirm/dismiss writes a row to `data/feedback.parquet`; "
         "those rows are the labels monitoring and retraining use.",
         "Alerts table in BigQuery, served by a Cloud Run FastAPI (`/alerts`, `/explain`, `/ask`) and a Looker Studio view. Gemini on Vertex AI "
         "writes the note from the structured record inside the project's IAM boundary. Feedback lands in BigQuery and joins the next training run.",
         expanded=False)

a = alerts()
if a.empty:
    st.info("No alerts yet."); st.stop()
f = features(); d = dates(); sc = read("scores"); sc["date"] = pd.to_datetime(sc["date"])

left, right = st.columns([1, 2])
with left:
    status_f = st.radio("Status", ["open", "all", "confirmed", "dismissed"], horizontal=True)
    v = a if status_f == "all" else a[a.status == status_f]
    v = v.sort_values(["child_explained", "revenue_at_risk"], ascending=[True, False])
    labels = {r.alert_id: f"{r.anomaly_type:11s} {r.store_id} {r.item_id if r.level == 'item' else r.cat_id:16s} {r.start.date()} ${r.revenue_at_risk:,.0f}" for _, r in v.iterrows()}
    if not labels:
        st.info("Nothing in this status."); st.stop()
    pick = st.radio("Alerts", list(labels.keys()), format_func=lambda k: labels[k], label_visibility="collapsed")
e = a[a.alert_id == pick].iloc[0]
rec = parse_json(e.recommendation); drivers = parse_json(e.drivers)
with right:
    st.markdown(f"### {e.anomaly_type.replace('_', ' ').title()} · {e.item_id if e.level == 'item' else e.cat_id} at {e.store_id}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Window", f"{e.start.date()} → {e.end.date()}", f"{int(e.days)} day(s)")
    m2.metric("Units actual vs expected", f"{e.units_actual:.0f} vs {e.units_expected:.0f}")
    m3.metric("Revenue at risk", fmt_money(e.revenue_at_risk))
    m4.metric("Detectors", e.detectors.replace(",", " + "), f"votes {int(e.votes)}" + (" + rule" if e.rule else ""))
    m5.metric("Status", e.status, f"model {e.model_version}")
    st.markdown("**Merchant note**")
    st.info(e.note + f"  \n*(source: {e.note_source})*")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Why (top-3 drivers)**")
        for drv in (drivers if isinstance(drivers, list) else []):
            st.markdown(f"- {drv['driver']}  <span style='color:#8b949e'>({drv['kind']})</span>", unsafe_allow_html=True)
    with c2:
        st.markdown("**What to do**")
        if isinstance(rec, dict):
            st.markdown(f"**{rec.get('action', '')}**")
            st.markdown(f"Expected impact **{fmt_money(rec.get('expected_revenue', 0))}** · confidence **{rec.get('confidence', 0) * 100:.0f}%**")
            st.caption(rec.get("rationale", ""))
    b1, b2, b3 = st.columns([1, 1, 4])
    if b1.button("✅ Confirm", type="primary"):
        pipeline.record_feedback(e.alert_id, "confirmed"); st.success("Recorded as confirmed (true positive)."); st.rerun()
    if b2.button("✖ Dismiss"):
        pipeline.record_feedback(e.alert_id, "dismissed"); st.warning("Recorded as dismissed (false positive)."); st.rerun()
    annotate("Why confirm/dismiss matters",
             "Precision in production has no labels except these. Confirmed = true positive, dismissed = false positive; monitoring computes a rolling "
             "precision and a drop below 0.75 triggers retraining. The next training run also receives them as hard examples.",
             "Same, with the feedback table in BigQuery and the precision metric in Vertex AI Model Monitoring / Cloud Monitoring with an alert policy.")

if e.level == "item":
    st.plotly_chart(series_chart(f, e.id, e.start - pd.Timedelta(days=60), min(d["last"], e.end + pd.Timedelta(days=14)), scores=sc,
                                 alerts_df=a[a.level == "item"], title=f"{e.id}: 60 days before and after"), width="stretch")
    row = sc[(sc.id == e.id) & (sc.date == e.peak_date)]
    if len(row):
        r = row.iloc[0]
        with st.expander("Scores on the peak day"):
            st.json({"date": str(r.date.date()), "units": int(r.units), "expected (fast model)": round(float(r.yhat_units), 2),
                     "z by scale": {k: round(float(r[f"z_{k}"]), 2) for k in ("1d", "3d", "7d", "14d", "28d")}, "picked scale": r.z_scale,
                     "isolation forest": round(float(r.if_score), 3), "autoencoder": round(float(r.ae_score), 3) if pd.notna(r.ae_score) else None,
                     "price vs reference": f"{r.sell_price:.2f} vs {r.price_ref:.2f}", "zero streak": int(r.zero_streak), "event": r.event_name or "-"})
else:
    kids = a[a.alert_id.isin(str(e.children).split(","))]
    st.markdown(f"**Children folded into this {e.level}-level alert:**")
    st.dataframe(kids[["start", "end", "item_id", "anomaly_type", "units_actual", "units_expected", "revenue_at_risk"]]
                 .assign(start=lambda x: x.start.dt.date, end=lambda x: x.end.dt.date), hide_index=True, width="stretch")

fb = store.read("feedback")
if fb is not None and len(fb):
    with st.expander(f"Feedback log ({len(fb)} rows)"):
        st.dataframe(fb.sort_values("ts", ascending=False), hide_index=True, width="stretch")
