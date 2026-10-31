import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import PALETTE, TYPE_ORDER, alerts, annotate, dates, features, hero, models, read, series_chart, setup, style

setup("Detection", "🎯")
hero("3 · Detection ensemble", "Three detectors with different blind spots vote on every series-day. Thresholds are calibrated to a false-alert budget; "
     "episodes are reconciled up the hierarchy so a merchant gets one alert per cause.",
     ["Residual · NB tails · frozen-origin", "Isolation Forest", "Autoencoder", "2-of-3 vote", "Effect-size + volume floors"])
annotate("Three detectors, one vote",
         "**Residual** (`ResidualDetector`): robust Holt-Winters in log space with holiday, SNAP, price-elasticity and yearly regressors; "
         "the 1-day test is an exact negative-binomial tail, the 3/7/14/28-day tests are frozen-origin cumulative shortfalls (forecast made before the "
         "window, so a drop cannot be absorbed). **Isolation Forest**: multivariate outlier score on per-series-standardized, holiday-adjusted features. "
         "**Autoencoder**: dense MLP on 28-day adjusted windows, reconstruction error on the last 7 days (LSTM in production). Each threshold is set on a "
         "clean calibration slice to a false-alert budget; an alert needs 2 of 3 votes, or the residual test at 1.25× its threshold, plus an effect-size "
         "floor on long windows and a 4 units/day volume floor. Rules short-circuit for price-file errors and improbable zero-sales runs. "
         "Alert days merge into episodes; three items moving together in a store-category become one systemic alert.",
         "Same code, run as a Cloud Run job (or Vertex AI batch prediction) over BigQuery features, ~30k series in parallel tasks. "
         "Prophet/SARIMAX refit weekly and predict daily; the LSTM autoencoder trains on Vertex AI. Thresholds and the vote rule are pipeline parameters "
         "versioned with the model.", expanded=True)

ens, _ = models()
f = features(); d = dates()
sc = read("scores")
a = alerts()
if ens is None or sc is None:
    st.warning("No production model or scores yet. Run `scripts/bootstrap.py`.")
    st.stop()
sc["date"] = pd.to_datetime(sc["date"])

st.subheader("Calibration (set on clean data before the holdout)")
cal = ens.calibration
c1, c2, c3 = st.columns(3)
c1.json({"residual thresholds by scale": {k: round(v, 2) for k, v in cal["thresholds"]["residual_by_scale"].items()},
         "isolation forest": round(cal["thresholds"]["iforest"], 3), "autoencoder": round(cal["thresholds"]["autoencoder"], 3)}, expanded=True)
c2.json({"flag rate on calibration slice": {k: f"{v * 100:.2f}%" for k, v in cal["flag_rate"].items()},
         "ensemble false alerts per 1k series-days": round(cal["ensemble_fa_per_1k"], 2)}, expanded=True)
c3.json({"vote": f">= {ens.vote_min} of 3 or residual at {ens.strong_ratio}x", "effect-size floor": ens.residual.min_effect,
         "volume floor (units/day)": ens.min_volume, "per-detector budget": f"{ens.fa_budget * 100:.1f}%",
         "regressors": {"category elasticity": {k: round(v, 2) for k, v in ens.residual.elasticity.items()},
                        "event effects": len(ens.residual.event_effect), "snap effects": len(ens.residual.snap_effect)}}, expanded=True)

st.subheader(f"Alert episodes in the live window ({d['live_start'].date()} to {d['last'].date()})")
items = a[a.level == "item"] if len(a) else a
if len(items):
    f1, f2, f3, f4 = st.columns(4)
    store_f = f1.multiselect("Store", sorted(items.store_id.unique()))
    cat_f = f2.multiselect("Category", sorted(items.cat_id.unique()))
    type_f = f3.multiselect("Type", sorted(items.anomaly_type.unique()))
    hide_child = f4.checkbox("Hide child-explained", value=True)
    v = items.copy()
    if store_f: v = v[v.store_id.isin(store_f)]
    if cat_f: v = v[v.cat_id.isin(cat_f)]
    if type_f: v = v[v.anomaly_type.isin(type_f)]
    if hide_child: v = v[~v.child_explained]
    st.dataframe(v[["start", "end", "days", "store_id", "item_id", "cat_id", "anomaly_type", "direction", "z_scale", "z_resid", "votes", "detectors", "rule",
                    "units_actual", "units_expected", "revenue_at_risk", "status"]]
                 .assign(start=lambda x: x.start.dt.date, end=lambda x: x.end.dt.date).round(2).sort_values("revenue_at_risk", ascending=False),
                 hide_index=True, width="stretch", height=320)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        vc = v.anomaly_type.value_counts().reindex(TYPE_ORDER).dropna()
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=[PALETTE.get(t, "#4c9aff") for t in vc.index], text=vc.values, textposition="outside"))
        st.plotly_chart(style(fig, 270, "Episodes by type", legend=False), width="stretch")
    with cc2:
        det = v.detectors.str.split(",").explode().value_counts()
        fig = go.Figure(go.Bar(x=det.index, y=det.values, marker_color=[PALETTE["actual"], PALETTE["drift"], PALETTE["alert"], PALETTE["ok"]][: len(det)],
                               text=det.values, textposition="outside"))
        st.plotly_chart(style(fig, 270, "Which detectors voted (episodes)", legend=False), width="stretch")
    with cc3:
        live = sc[sc.date >= d["live_start"]]
        flags = live[["f_resid", "f_if", "f_ae"]].astype(int)
        names = ["residual", "iforest", "autoencoder"]
        m = flags.T.values @ flags.values
        fig = go.Figure(go.Heatmap(z=m, x=names, y=names, colorscale=[[0, "#1a222d"], [1, PALETTE["alert"]]], text=m, texttemplate="%{text}", showscale=False))
        st.plotly_chart(style(fig, 270, "Detector agreement: co-flagged series-days", legend=False), width="stretch")
    tl = v.copy()
    fig = go.Figure()
    for t in TYPE_ORDER:
        g = tl[tl.anomaly_type == t]
        if len(g):
            fig.add_trace(go.Scatter(x=g.start, y=g.store_id, mode="markers", name=t, marker=dict(color=PALETTE[t], size=(g.revenue_at_risk.clip(lower=20) ** 0.5).clip(6, 28), opacity=0.85,
                                     line=dict(color="#0f141b", width=1)), text=[f"{r.item_id}<br>{r.start.date()} → {r.end.date()}<br>${r.revenue_at_risk:,.0f}" for _, r in g.iterrows()], hoverinfo="text"))
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(style(fig, 320, "Episode timeline by store (marker size = revenue at risk)"), width="stretch")
    sys_a = a[a.level != "item"]
    if len(sys_a):
        st.markdown(f"**Systemic alerts (reconciliation):** {len(sys_a)}")
        st.dataframe(sys_a[["start", "end", "level", "store_id", "cat_id", "anomaly_type", "direction", "units_actual", "units_expected", "revenue_at_risk", "children"]]
                     .assign(start=lambda x: x.start.dt.date, end=lambda x: x.end.dt.date), hide_index=True, width="stretch")
    else:
        st.caption("No systemic (category/store-level) alerts in this window: nothing moved three-items-wide in one store-category.")
else:
    st.info("No alerts in the live window.")

st.subheader("Inspect a series: detectors side by side")
ids = sorted(f.id.unique())
default = items.sort_values("revenue_at_risk", ascending=False).id.iloc[0] if len(items) else ids[0]
sid = st.selectbox("Series", ids, index=ids.index(default))
lo = d["live_start"] - pd.Timedelta(days=30)
st.plotly_chart(series_chart(f, sid, lo, d["last"], scores=sc, alerts_df=items if len(items) else None, title=f"{sid}: actual vs expected, alert days and episodes"),
                width="stretch")
s = sc[(sc.id == sid) & (sc.date >= lo)].sort_values("date")
if len(s):
    fig = go.Figure()
    thr = ens.residual.thresholds
    for col, name, color in [("z_1d", "1 day (exact NB tail)", "#4c9aff"), ("z_3d", "3 day", "#3fb950"), ("z_7d", "7 day", "#b083f0"),
                             ("z_14d", "14 day", "#e3b341"), ("z_28d", "28 day", "#ff6b6b")]:
        fig.add_trace(go.Scatter(x=s.date, y=s[col], name=name, line=dict(color=color, width=1.4)))
    fig.add_hline(y=thr["1d"], line=dict(color="#8b949e", dash="dot")); fig.add_hline(y=-thr["1d"], line=dict(color="#8b949e", dash="dot"))
    fig.update_layout(yaxis_title="standardized z", hovermode="x unified")
    st.plotly_chart(style(fig, 300, "Residual detector: five time scales (dotted = 1-day threshold)"), width="stretch")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=s.date, y=s.if_score, name="isolation forest score", line=dict(color="#4c9aff")))
    fig2.add_hline(y=ens.iforest.threshold, line=dict(color="#4c9aff", dash="dot"))
    fig2.add_trace(go.Scatter(x=s.date, y=s.ae_score, name="autoencoder error", line=dict(color="#ff8249"), yaxis="y2"))
    fig2.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False), hovermode="x unified")
    st.plotly_chart(style(fig2, 280, "Isolation Forest and autoencoder (dotted = IF threshold)"), width="stretch")
    with st.expander("Score rows"):
        st.dataframe(s[["date", "units", "yhat_units", "z_1d", "z_3d", "z_7d", "z_14d", "z_28d", "z_scale", "z_ratio", "if_score", "ae_score",
                        "f_resid", "f_if", "f_ae", "f_rule", "votes", "is_alert", "anomaly_type"]].assign(date=lambda x: x.date.dt.date).round(2),
                     hide_index=True, width="stretch", height=300)
