import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import annotate, dates, features, hero, setup, style
from merchant_pulse.features import FEATURE_COLS

setup("Features", "🧮")
hero("2 · Features", "One row per series-day. Every rolling statistic is shifted by a day so a feature never contains the value it is meant to judge. "
     "All three detectors read this table and nothing else.", ["Lags · rolling z", "91-day price reference", "SNAP · events", "Zero-run length", "Drift ratio"])
annotate("Feature table",
         "`merchant_pulse/features.py` builds one row per series-day: lags (1, 7, 28), rolling means and std shifted by one day, a log z-score "
         "against the 28-day norm, a 91-day rolling median reference price with price % change and promo flag, SNAP and event flags, weekday, "
         "zero-sales run length, and `drift_ratio` (last 7 days vs the 28 days ending 8 days ago; 7-day sums cancel weekday effects). "
         "Everything rolling is shifted so a feature never contains the value it judges. All detectors read this table.",
         "Dataform models on BigQuery produce the same table incrementally by partition, with assertions (no nulls, no future dates, one row per "
         "series-day) and lineage in the Dataform graph. A feature bug is fixed in one SQL model.", expanded=True)

f = features(); d = dates()
ids = sorted(f.id.unique())
sid = st.selectbox("Series", ids, index=ids.index("FOODS_3_001_CA_3") if "FOODS_3_001_CA_3" in ids else 0)
g = f[(f.id == sid) & (f.date >= d["last"] - pd.Timedelta(days=120))].sort_values("date")

fig = go.Figure()
fig.add_trace(go.Scatter(x=g.date, y=g.units, name="units", line=dict(color="#1f6feb")))
fig.add_trace(go.Scatter(x=g.date, y=g.rm7, name="rolling mean 7 (shifted)", line=dict(color="#2da44e", dash="dot")))
fig.add_trace(go.Scatter(x=g.date, y=g.rm28, name="rolling mean 28 (shifted)", line=dict(color="#8b949e", dash="dash")))
fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2), title="Units and shifted rolling means")
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=g.date, y=g.z28, name="z28 (log units vs 28-day norm)", line=dict(color="#8250df")))
    fig2.add_trace(go.Scatter(x=g.date, y=g.drift_ratio, name="drift_ratio", line=dict(color="#d9480f")))
    fig2.add_hline(y=0, line=dict(color="#8b949e", width=1))
    fig2.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), title="Relative features")
    st.plotly_chart(fig2, width="stretch")
with c2:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=g.date, y=g.sell_price, name="sell price", line=dict(color="#1f6feb")))
    fig3.add_trace(go.Scatter(x=g.date, y=g.price_ref, name="reference price (91-day median)", line=dict(color="#8b949e", dash="dash")))
    fig3.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), title="Price and reference")
    st.plotly_chart(fig3, width="stretch")

st.subheader("Feature rows (last 14 days)")
show = ["date", "units", "sell_price", "price_ref", "price_pct", "promo", "snap", "event_name", "lag_7", "rm7", "rm28", "z28", "ratio_7", "drift_ratio", "zero_streak"]
st.dataframe(g[show].tail(14).assign(date=lambda x: x.date.dt.date).round(3), hide_index=True, width="stretch")
with st.expander("Feature definitions"):
    st.markdown("""
| feature | definition |
|---|---|
| `log_units` | log1p(units) |
| `lag_1, lag_7, lag_28` | units 1, 7, 28 days earlier |
| `rm7, rm28` | rolling mean of units over 7 / 28 days, shifted by 1 |
| `z28` | (log_units − rolling mean of log_units) / rolling std, 28 days, shifted |
| `ratio_7` | units / (rm7 + 1) |
| `lag7_ratio` | units / (lag_7 + 1) |
| `price_ref` | rolling 91-day median of sell_price, shifted |
| `price_pct`, `price_ratio` | sell_price / price_ref − 1, sell_price / price_ref |
| `promo` | price_pct < −5% |
| `snap` | SNAP issuance day for the store's state |
| `event`, `event_name` | named calendar event |
| `zero_streak` | consecutive zero-sales days including today |
| `drift_ratio` | mean(units, last 7 days incl. today) / mean(units, days t−35..t−8) − 1 |
| `log_units_adj` | log_units minus known holiday, SNAP, price-elasticity and yearly effects (added by the ensemble) |
""")
    st.caption(f"Detector inputs: {', '.join(FEATURE_COLS)} (+ log_units_adj)")
