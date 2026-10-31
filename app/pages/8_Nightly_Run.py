import json
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import annotate, dates, read, setup
from merchant_pulse import pipeline, registry

setup("Nightly Run", "🌙")
st.title("8 · Nightly run")
annotate("What happens at 02:00",
         "`pipeline.nightly(run_date)`: load tables → build features → load production models → score the window → episodes → reconciliation → "
         "attribution + prescription + narrative → write alerts (idempotent: alerts whose peak falls in the window are replaced) → monitoring "
         "snapshot → run record. If the narrative step fails the alert still ships with drivers and action; the note says 'pending'.",
         "Cloud Scheduler fires at 02:00; Dataform builds features in BigQuery; a Cloud Run job scores ~30k series in parallel tasks; attribution and "
         "uplift write the alerts table; Gemini writes notes; Looker refreshes and the digest goes out by 03:00. Each step is keyed on run date so a "
         "retry overwrites instead of duplicating.", expanded=True)

d = dates()
c1, c2, c3 = st.columns([1, 1, 2])
run_date = c1.date_input("Run date", value=d["last"].date(), min_value=(d["live_start"]).date(), max_value=d["last"].date())
backfill = c2.number_input("Backfill days", min_value=1, max_value=60, value=1)
gem = c3.checkbox("Use Gemini for notes (needs GEMINI_API_KEY)", value=False)
if st.button("Run nightly pipeline", type="primary"):
    box = st.empty(); lines = []
    def log(s):
        lines.append(s); box.code("\n".join(lines))
    t0 = time.time()
    out = pipeline.nightly(pd.Timestamp(run_date), backfill_days=int(backfill), use_gemini=gem, log=log)
    st.success(f"Done in {time.time() - t0:.1f}s: {out['run']['alerts_written']} alerts written for {run_date} (model {out['run']['model_version']}).")
    steps = pd.DataFrame(out["steps"])
    fig = go.Figure(go.Bar(x=steps.seconds, y=steps.step, orientation="h", marker_color="#1f6feb"))
    fig.update_layout(title="Seconds per step", height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    if len(out["alerts"]):
        st.dataframe(out["alerts"][["start", "end", "level", "store_id", "item_id", "cat_id", "anomaly_type", "revenue_at_risk", "action", "note_source"]]
                     .assign(start=lambda x: pd.to_datetime(x.start).dt.date, end=lambda x: pd.to_datetime(x.end).dt.date), hide_index=True, width="stretch")
    st.cache_data.clear()

st.subheader("Run history")
runs = read("runs")
if runs is not None and len(runs):
    show = runs.copy(); show["run_date"] = pd.to_datetime(show.run_date).dt.date
    st.dataframe(show[["ts", "run_date", "model_version", "alerts_written", "alert_days", "rows_scored", "total_seconds", "retrain_trigger", "trigger_reasons"]].iloc[::-1],
                 hide_index=True, width="stretch", height=260)
    last = runs.iloc[-1]
    steps = pd.DataFrame(json.loads(last.steps))
    st.caption("Last run, seconds per step: " + " · ".join(f"{s.step} {s.seconds}s" for s in steps.itertuples()))
else:
    st.info("No runs yet.")
