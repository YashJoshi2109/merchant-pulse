import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import PALETTE, STATE_NAME, TYPE_ORDER, alerts, annotate, dates, hero, read, scores, setup, style
from merchant_pulse import config, registry, store

setup("Overview", "🏠")

if not store.exists("sales"):
    st.error("No data. From the project folder run: `.venv/bin/python scripts/bootstrap.py`")
    st.stop()

d = dates(); a = alerts(); reg = registry.load(); prod = reg.get("production"); runs = read("runs"); sc = scores()
hero("Merchant Pulse",
     "Finds unusual sales, explains why, and tells the merchant what to do next. Three detectors vote on every item × store × day, "
     "episodes roll up the hierarchy, drivers are attributed, an uplift model prices the action, and a note is written. A registry gates, "
     "canaries and rolls back models; merchant feedback closes the loop.",
     ["GCP-native design", "280 series · 2 years · daily", "3-detector ensemble", "Uplift-priced actions", "Gate · canary · rollback", f"Production {prod}"])

items = a[a.level == "item"] if len(a) else a
open_items = items[(items.status == "open") & (~items.child_explained)] if len(items) else items
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Series monitored", f"{read('hierarchy').shape[0]:,}")
c2.metric("Live window", f"{(d['last'] - d['live_start']).days + 1} days", str(d["last"].date()))
c3.metric("Open alerts", len(open_items), f"{len(items)} total episodes")
c4.metric("Revenue at risk (open)", f"${open_items.revenue_at_risk.sum():,.0f}" if len(open_items) else "$0")
conf = items[items.status == "confirmed"]; dism = items[items.status == "dismissed"]
c5.metric("Merchant feedback", f"{len(conf)} ✓ / {len(dism)} ✗", "precision from feedback" if len(conf) + len(dism) else "none yet")
es = registry.get_version(prod).get("eval_summary", {}) if prod else {}
c6.metric("Model recall (harness)", f"{es.get('recall', float('nan')):.2f}", f"F1 {es.get('f1', float('nan')):.2f} · FA {es.get('fa_per_1k', float('nan')):.1f}/1k")

annotate("What this system is",
         "A local, self-contained build of the Merchant Pulse design. Synthetic M5-style sales (3 states, 10 stores, 280 item×store series, 2 years) "
         "live as parquet tables under `data/`, queried through DuckDB. Three detectors vote on every series-day, episodes are reconciled up the "
         "hierarchy, attributed, given an action by an uplift model, and written as a merchant note. A registry gates, canaries and rolls back models. "
         "Merchant confirm/dismiss clicks are stored as labels.",
         "Pub/Sub + Dataflow → BigQuery raw → Dataform features → Vertex AI pipelines (train, eval, gate) → Model Registry → Cloud Run (canary, rollout) "
         "→ alerts table in BigQuery → Cloud Run API + Looker Studio → merchant. Gemini on Vertex AI writes the note and answers Text2SQL questions. "
         "Vertex AI Model Monitoring watches drift and feedback precision.")

if len(items):
    r1, r2 = st.columns([3, 2])
    with r1:
        daily = items.assign(day=items.peak_date.dt.normalize()).groupby(["day", "anomaly_type"]).size().reset_index(name="n")
        fig = px.bar(daily, x="day", y="n", color="anomaly_type", color_discrete_map=PALETTE, category_orders={"anomaly_type": TYPE_ORDER})
        fig.update_layout(barmode="stack", xaxis_title=None, yaxis_title="alert episodes")
        st.plotly_chart(style(fig, 300, "Alert episodes per day, by type (live window)"), width="stretch")
    with r2:
        bystate = items.groupby("state_id").revenue_at_risk.sum().reset_index()
        fig = go.Figure(go.Choropleth(locations=bystate.state_id, z=bystate.revenue_at_risk, locationmode="USA-states",
                                      colorscale=[[0, "#1a222d"], [1, PALETTE["alert"]]], marker_line_color="#0f141b", showscale=False,
                                      text=[f"{STATE_NAME[s]}: ${v:,.0f}" for s, v in zip(bystate.state_id, bystate.revenue_at_risk)], hoverinfo="text"))
        fig.update_geos(scope="usa", bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)", landcolor="#141b24", subunitcolor="#2a3441")
        st.plotly_chart(style(fig, 300, "Revenue at risk by state", legend=False), width="stretch")
    r3, r4, r5 = st.columns(3)
    with r3:
        bystore = items.groupby("store_id").revenue_at_risk.sum().sort_values(ascending=True).reset_index()
        fig = go.Figure(go.Bar(x=bystore.revenue_at_risk, y=bystore.store_id, orientation="h", marker_color=PALETTE["actual"],
                               text=[f"${v:,.0f}" for v in bystore.revenue_at_risk], textposition="outside"))
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(style(fig, 320, "Revenue at risk by store", legend=False), width="stretch")
    with r4:
        bt = items.groupby("anomaly_type").agg(n=("alert_id", "size"), rev=("revenue_at_risk", "sum")).reindex(TYPE_ORDER).dropna().reset_index()
        fig = go.Figure(go.Pie(labels=bt.anomaly_type, values=bt.n, hole=0.6, marker=dict(colors=[PALETTE[t] for t in bt.anomaly_type]),
                               textinfo="label+percent", sort=False))
        fig.update_layout(annotations=[dict(text=f"{int(bt.n.sum())}<br>episodes", x=0.5, y=0.5, showarrow=False, font_size=14)])
        st.plotly_chart(style(fig, 320, "Episodes by anomaly type", legend=False), width="stretch")
    with r5:
        tm = items.groupby(["cat_id", "dept_id"]).agg(rev=("revenue_at_risk", "sum"), n=("alert_id", "size")).reset_index()
        fig = px.treemap(tm, path=["cat_id", "dept_id"], values="rev", color="n", color_continuous_scale=["#1a222d", PALETTE["alert"]])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(texttemplate="%{label}<br>$%{value:,.0f}")
        st.plotly_chart(style(fig, 320, "Revenue at risk by category / department", legend=False), width="stretch")

    if sc is not None and len(sc):
        live = sc[sc.date >= d["live_start"]]
        tot = live.groupby("date").agg(actual=("units", "sum"), expected=("yhat_units", "sum")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tot.date, y=tot.expected, name="expected (sum of model forecasts)", line=dict(color=PALETTE["expected"], dash="dot")))
        fig.add_trace(go.Scatter(x=tot.date, y=tot.actual, name="actual units, all 280 series", line=dict(color=PALETTE["actual"], width=2.2), fill="tonexty",
                                 fillcolor="rgba(76,154,255,0.08)"))
        ad = live[live.is_alert].groupby("date").size().reset_index(name="n")
        fig.add_trace(go.Bar(x=ad.date, y=ad.n, name="alert days", marker_color=PALETTE["alert"], yaxis="y2", opacity=0.7))
        fig.update_layout(yaxis_title="units", yaxis2=dict(overlaying="y", side="right", title="alert days", showgrid=False), hovermode="x unified")
        st.plotly_chart(style(fig, 320, "Network-wide actual vs expected, with alert days"), width="stretch")

st.subheader("How it works, in one pass")
steps = [("Ingest", "POS, prices, promos, SNAP and holiday calendars land in one daily table."),
         ("Detect", "Residual (robust Holt-Winters, NB tails, frozen-origin tests) · Isolation Forest · autoencoder → 2-of-3 vote."),
         ("Reconcile", "Alert days become episodes; three items moving together in a store-category become one systemic alert."),
         ("Explain", "Rules first (price change, SNAP, holiday, zero run), then SHAP. Exactly three drivers."),
         ("Prescribe", "T-learner uplift from historical promotions picks the action with the highest expected incremental revenue."),
         ("Ship safely", "Eval harness → gate → registry → canary → promote; rollback is a pointer change. Monitoring triggers retraining.")]
cols = st.columns(3)
for i, (k, v) in enumerate(steps):
    cols[i % 3].markdown(f'<div class="mp-step"><b>{i + 1} · {k}</b><br><span style="color:#c9d1d9">{v}</span></div>', unsafe_allow_html=True)

st.subheader("Pages")
rows = [("0 · Architecture", "The five design diagrams, in-app", "—"),
        ("1 · Data & Ingestion", "Sources, replay of streaming events, warehouse tables, SNAP and holiday effects", "Pub/Sub, Dataflow, BigQuery raw"),
        ("2 · Features", "Per series-day feature table, shifted rolling stats", "Dataform on BigQuery"),
        ("3 · Detection", "Three detectors, 2-of-3 vote, episodes, reconciliation, agreement matrix", "Vertex AI batch scoring / Cloud Run job"),
        ("4 · Alert Detail", "Drivers, recommended action, merchant note, confirm/dismiss", "Alerts table + Cloud Run API + Gemini"),
        ("5 · Eval Harness", "Injected anomalies, matched-control lift, the gate", "Vertex AI pipeline eval step"),
        ("6 · MLOps", "Registry, gate, canary, promote, rollback, monitoring, retrain trigger", "Model Registry, Cloud Run traffic split, Model Monitoring"),
        ("7 · Ask", "Natural language → SQL over the same tables", "Gemini + BigQuery"),
        ("8 · Nightly Run", "Run the whole pipeline with per-step timings", "Cloud Scheduler → Vertex AI pipeline")]
st.dataframe(pd.DataFrame(rows, columns=["page", "what it shows", "GCP equivalent"]), hide_index=True, width="stretch")

if runs is not None and len(runs):
    r = runs.iloc[-1]
    st.caption(f"Latest run {pd.Timestamp(r.run_date).date()} · model {r.model_version} · {int(r.alerts_written)} alerts · "
               f"{int(r.rows_scored):,} series-days scored in {r.total_seconds:.1f}s · retrain trigger: {'yes' if r.retrain_trigger else 'no'}")
if len(a):
    st.subheader("Top open alerts by revenue at risk")
    top = open_items.sort_values("revenue_at_risk", ascending=False).head(8)
    st.dataframe(top[["start", "end", "store_id", "item_id", "cat_id", "anomaly_type", "revenue_at_risk", "action", "confidence"]]
                 .assign(start=lambda x: x.start.dt.date, end=lambda x: x.end.dt.date), hide_index=True, width="stretch")
    st.page_link("pages/4_Alert_Detail.py", label="Open an alert →")
