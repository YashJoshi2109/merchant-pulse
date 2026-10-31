import time

import pandas as pd
import plotly.express as px
import streamlit as st

import plotly.graph_objects as go

from common import PALETTE, annotate, dates, features, hero, read, series_chart, setup, style
from merchant_pulse import store

setup("Data & Ingestion", "📥")
hero("1 · Data & Ingestion", "Where sales data enters and rests. Two ingestion paths, one landing zone, ground truth kept apart from everything a model sees.",
     ["10 stores · 3 states", "280 series", "731 days", "SNAP + holidays + promos"])

annotate("Sources and ingestion",
         "`merchant_pulse/synth.py` generates M5-shaped data: 10 stores in CA/TX/WI, 3 categories, 7 departments, 28 items, daily units from "
         "2024-01-01 to 2025-12-31 with weekday and yearly seasonality, SNAP issuance days per state, named holidays (Christmas = closed), "
         "weekly prices with 10-30% promotions and item-level price elasticity, negative-binomial noise. Ground truth (clean sales, true elasticities, "
         "planted anomalies) sits in `data/truth` and is never read by a model. Tables are parquet; DuckDB exposes them as SQL views.",
         "POS events → Pub/Sub topic `sales-events` → Dataflow (Beam) streaming job: dedupe, late-data windows, exactly-once sink into a BigQuery raw "
         "table partitioned by date and clustered by store. Prices, promos, inventory arrive daily via Cloud Scheduler + BigQuery `MERGE`. "
         "Why both paths: intraday spikes are visible within minutes, while the nightly model reads one consistent daily table.", expanded=True)

stats = store.table_stats()
c1, c2 = st.columns([2, 3])
with c1:
    st.subheader("Warehouse tables")
    st.dataframe(stats, hide_index=True, width="stretch")
with c2:
    h = read("hierarchy")
    st.subheader("Hierarchy")
    fig = px.sunburst(h, path=["state_id", "store_id", "cat_id", "dept_id"], color="cat_id",
                      color_discrete_map={"FOODS": PALETTE["ok"], "HOUSEHOLD": PALETTE["actual"], "HOBBIES": PALETTE["drift"]})
    st.plotly_chart(style(fig, 380, legend=False), width="stretch")

f = features(); d = dates()
st.subheader("What the data looks like")
g1, g2 = st.columns([3, 2])
with g1:
    tot = f.groupby("date").units.sum().reset_index()
    fig = go.Figure(go.Scatter(x=tot.date, y=tot.units, line=dict(color=PALETTE["actual"], width=1.6), name="units, all series"))
    tot7 = tot.set_index("date").units.rolling(7).mean().reset_index()
    fig.add_trace(go.Scatter(x=tot7.date, y=tot7.units, line=dict(color=PALETTE["alert"], width=2.2), name="7-day mean"))
    cal = read("calendar"); cal["date"] = pd.to_datetime(cal["date"])
    for _, r in cal[cal.event_name.isin(["Christmas", "Thanksgiving", "BlackFriday", "SuperBowl", "IndependenceDay"])].iterrows():
        fig.add_vline(x=r.date, line=dict(color="#8b949e", width=1, dash="dot"))
    fig.update_layout(yaxis_title="units / day", hovermode="x unified")
    st.plotly_chart(style(fig, 320, "Network-wide daily units, two years (dotted: major events; Christmas = closed)"), width="stretch")
with g2:
    base = f[(f.event == 0)].groupby(["state_id", "cat_id", "snap"]).units.mean().unstack("snap")
    base["lift_pct"] = (base[1] / base[0] - 1) * 100
    base = base.reset_index()
    fig = px.bar(base, x="cat_id", y="lift_pct", color="state_id", barmode="group",
                 color_discrete_sequence=[PALETTE["actual"], PALETTE["alert"], PALETTE["drift"]], category_orders={"cat_id": ["FOODS", "HOUSEHOLD", "HOBBIES"]})
    fig.add_hline(y=0, line=dict(color="#8b949e", width=1))
    fig.update_layout(yaxis_title="SNAP day vs other days, % units", xaxis_title=None)
    st.plotly_chart(style(fig, 320, "SNAP issuance effect by category and state"), width="stretch")
w1, w2 = st.columns(2)
with w1:
    dow = f.groupby("dow").units.mean().reindex(range(7))
    fig = go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=dow.values, marker_color=PALETTE["actual"]))
    fig.update_layout(yaxis_title="mean units / series / day")
    st.plotly_chart(style(fig, 260, "Weekday pattern", legend=False), width="stretch")
with w2:
    pr = f.groupby("date").promo.mean().reset_index()
    fig = go.Figure(go.Scatter(x=pr.date, y=pr.promo * 100, fill="tozeroy", line=dict(color=PALETTE["promo"], width=1.5), fillcolor="rgba(63,185,80,0.15)"))
    fig.update_layout(yaxis_title="% of series on promotion")
    st.plotly_chart(style(fig, 260, "Share of series on promotion, by day", legend=False), width="stretch")

st.subheader("Replay a day of events (streaming path)")
annotate("What the replay shows",
         "The last day's rows are emitted as individual events in arrival order with a small random delay, de-duplicated by (id, date), and appended. "
         "It is a visual stand-in; no Pub/Sub involved.",
         "Dataflow consumes the topic, assigns event time, dedupes on a message key, handles late data with allowed lateness, and streams inserts into "
         "BigQuery. The nightly job then reads a complete day.")
col_a, col_b = st.columns([1, 3])
day = col_a.date_input("Day to replay", value=d["last"].date(), min_value=d["first"].date(), max_value=d["last"].date())
if col_a.button("Replay events"):
    rows = f[f.date == pd.Timestamp(day)][["id", "store_id", "item_id", "units", "sell_price"]].sample(frac=1.0, random_state=1)
    prog = col_b.progress(0, text="publishing to sales-events …")
    seen = set(); dupes = 0; shown = col_b.empty(); buf = []
    n = len(rows)
    for i, (_, r) in enumerate(rows.iterrows()):
        key = (r.id, day)
        if key in seen:
            dupes += 1
        seen.add(key)
        if i % 20 == 0 or i == n - 1:
            buf.append(f"{r.store_id:5s} {r.item_id:16s} units={int(r.units):3d} price={r.sell_price:6.2f}")
            shown.code("\n".join(buf[-8:]))
            prog.progress((i + 1) / n, text=f"{i + 1}/{n} events · {dupes} duplicates dropped")
            time.sleep(0.02)
    col_b.success(f"{n} events ingested for {day}; {f[f.date == pd.Timestamp(day)].units.sum():,.0f} units across {rows.store_id.nunique()} stores.")

st.subheader("Look at a series")
ids = sorted(f.id.unique())
sid = st.selectbox("Series (item_store)", ids, index=ids.index("FOODS_3_001_CA_3") if "FOODS_3_001_CA_3" in ids else 0)
rng = st.slider("Window", min_value=d["first"].date(), max_value=d["last"].date(), value=((d["last"] - pd.Timedelta(days=180)).date(), d["last"].date()))
st.plotly_chart(series_chart(f, sid, pd.Timestamp(rng[0]), pd.Timestamp(rng[1])), width="stretch")
cal = read("calendar")
with st.expander("Calendar: events and SNAP days"):
    ev = cal[cal.event_name != ""][["date", "event_name"]]
    st.dataframe(ev.assign(date=pd.to_datetime(ev.date).dt.date), hide_index=True, width="stretch", height=240)
    st.caption("SNAP schedule (days of month): CA 1-10 · TX 1,3,5,6,7,9,11,12,13,15 · WI 2,3,5,6,8,9,11,12,14,15 (M5 convention).")
