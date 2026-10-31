import pandas as pd
import streamlit as st

from common import alerts, annotate, dates, read, setup
from merchant_pulse import config, registry, store

setup("Overview", "🏠")
st.title("Merchant Pulse")
st.caption("Sales anomaly detection and prescriptive recommendations for merchants. Every box in the architecture is a page here; every page explains itself.")

if not store.exists("sales"):
    st.error("No data. From the project folder run: `.venv/bin/python scripts/bootstrap.py`")
    st.stop()

d = dates()
a = alerts()
reg = registry.load()
prod = reg.get("production")
runs = read("runs")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Series monitored", f"{read('hierarchy').shape[0]:,}")
c2.metric("Data through", str(d["last"].date()))
open_items = a[(a.level == "item") & (a.status == "open") & (~a.child_explained)] if len(a) else a
c3.metric("Open alerts (item level)", len(open_items))
c4.metric("Revenue at risk (open)", f"${open_items.revenue_at_risk.sum():,.0f}" if len(open_items) else "$0")
c5.metric("Production model", prod or "none", help="Vertex AI Model Registry stand-in: registry/registry.json")

annotate("What this system is",
         "A local, self-contained build of the Merchant Pulse design. Synthetic M5-style sales (3 states, 10 stores, 280 item×store series, 2 years) "
         "live as parquet tables under `data/`, queried through DuckDB. Three detectors vote on every series-day, episodes are reconciled up the "
         "hierarchy, attributed, given an action by an uplift model, and written as a merchant note. A registry gates, canaries and rolls back models. "
         "Merchant confirm/dismiss clicks are stored as labels.",
         "Pub/Sub + Dataflow → BigQuery raw → Dataform features → Vertex AI pipelines (train, eval, gate) → Model Registry → Cloud Run (canary, rollout) "
         "→ alerts table in BigQuery → Cloud Run API + Looker Studio → merchant. Gemini on Vertex AI writes the note and answers Text2SQL questions. "
         "Vertex AI Model Monitoring watches drift and feedback precision.", expanded=True)

st.subheader("Architecture, page by page")
rows = [
    ("1 · Data & Ingestion", "Sources, replay of streaming events, warehouse tables", "Pub/Sub, Dataflow, BigQuery raw"),
    ("2 · Features", "Per series-day feature table, shifted rolling stats", "Dataform on BigQuery"),
    ("3 · Detection", "Three detectors, 2-of-3 vote, episodes, reconciliation", "Vertex AI batch scoring / Cloud Run job"),
    ("4 · Alert Detail", "Drivers, recommended action, merchant note, confirm/dismiss", "Alerts table + Cloud Run API + Gemini"),
    ("5 · Eval Harness", "Injected anomalies, matched-control lift, the gate", "Vertex AI pipeline eval step"),
    ("6 · MLOps", "Registry, gate, canary, promote, rollback, monitoring, retrain trigger", "Model Registry, Cloud Run traffic split, Model Monitoring"),
    ("7 · Ask", "Natural language → SQL over the same tables", "Gemini + BigQuery"),
    ("8 · Nightly Run", "Run the whole pipeline with per-step timings", "Cloud Scheduler → Vertex AI pipeline"),
]
st.dataframe(pd.DataFrame(rows, columns=["page", "what it shows", "GCP equivalent"]), hide_index=True, width="stretch")

st.subheader("Latest run")
if runs is not None and len(runs):
    r = runs.iloc[-1]
    st.write(f"Run date **{pd.Timestamp(r.run_date).date()}** · model **{r.model_version}** · {int(r.alerts_written)} alerts written · "
             f"{int(r.rows_scored):,} series-days scored in **{r.total_seconds:.1f}s** · retrain trigger: **{'yes' if r.retrain_trigger else 'no'}**")
else:
    st.info("No runs yet. Open page 8 and run the nightly pipeline.")

if len(a):
    st.subheader("Top open alerts by revenue at risk")
    top = open_items.sort_values("revenue_at_risk", ascending=False).head(8)
    st.dataframe(top[["start", "end", "store_id", "item_id", "cat_id", "anomaly_type", "revenue_at_risk", "action", "confidence"]]
                 .assign(start=lambda x: x.start.dt.date, end=lambda x: x.end.dt.date), hide_index=True, width="stretch")
    st.page_link("pages/4_Alert_Detail.py", label="Open an alert →")

with st.expander("Recent registry events"):
    ev = pd.DataFrame(reg.get("events", []))
    if len(ev):
        st.dataframe(ev.tail(12), hide_index=True, width="stretch")
