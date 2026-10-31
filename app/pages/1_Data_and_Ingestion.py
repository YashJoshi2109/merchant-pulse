import time

import pandas as pd
import plotly.express as px
import streamlit as st

from common import annotate, dates, features, read, series_chart, setup
from merchant_pulse import store

setup("Data & Ingestion", "📥")
st.title("1 · Data & Ingestion")

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
    fig = px.sunburst(h, path=["state_id", "store_id", "cat_id", "dept_id"], height=380)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

st.subheader("Replay a day of events (streaming path)")
annotate("What the replay shows",
         "The last day's rows are emitted as individual events in arrival order with a small random delay, de-duplicated by (id, date), and appended. "
         "It is a visual stand-in; no Pub/Sub involved.",
         "Dataflow consumes the topic, assigns event time, dedupes on a message key, handles late data with allowed lateness, and streams inserts into "
         "BigQuery. The nightly job then reads a complete day.")
f = features(); d = dates()
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
