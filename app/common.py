"""Shared helpers for the Streamlit pages: data access, caching, and the self-describing annotations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from merchant_pulse import config, pipeline, registry, store  # noqa: E402
from merchant_pulse.features import split_dates  # noqa: E402

PALETTE = {"actual": "#1f6feb", "expected": "#8b949e", "alert": "#d9480f", "band": "rgba(139,148,158,0.18)",
           "spike": "#d9480f", "drop": "#1f6feb", "drift": "#8250df", "stockout": "#b08800", "price_error": "#cf222e"}


def ensure_bootstrapped() -> None:
    """First run on a fresh checkout: build data, train v1, backfill alerts. ~15 s. No-op when data exists."""
    if store.exists("sales") and registry.production():
        return
    with st.status("First run: generating data, training v1, backfilling alerts (about 15 s)…", expanded=True) as s:
        import importlib
        boot = importlib.import_module("scripts.bootstrap")
        lines = []
        boot.main(log=lambda m: (lines.append(m), s.write(m)))
        s.update(label="Bootstrap complete", state="complete", expanded=False)
    st.cache_data.clear(); st.cache_resource.clear()


def setup(title: str, icon: str = "📈") -> None:
    st.set_page_config(page_title=f"Merchant Pulse · {title}", page_icon=icon, layout="wide")
    ensure_bootstrapped()
    with st.sidebar:
        st.markdown("### Merchant Pulse")
        st.caption("GCP-style sales anomaly detection and prescriptive recommendations, demo build")
        st.session_state.setdefault("annot", True)
        st.toggle("Show annotations", key="annot", help="Explain each component: what runs here and what runs on GCP in production.")
        prod = registry.production()
        st.caption(f"Production model: **{prod or 'none'}**")
        if store.exists("sales"):
            d = dates()
            st.caption(f"Data through {d['last'].date()} · live window from {d['live_start'].date()}")
        else:
            st.warning("No data yet. Run `python scripts/bootstrap.py`.")


def annotate(title: str, here: str, prod: str, expanded: bool = False) -> None:
    """Self-describing panel: left column is what the demo does, right column is the production GCP equivalent."""
    if not st.session_state.get("annot", True):
        return
    with st.expander(f"ⓘ {title}", expanded=expanded):
        c1, c2 = st.columns(2)
        c1.markdown("**Here (demo)**\n\n" + here)
        c2.markdown("**In production (GCP)**\n\n" + prod)


@st.cache_data(show_spinner=False)
def table(name: str, _mtime: float = 0.0) -> pd.DataFrame | None:
    return store.read(name)


def mtime(name: str) -> float:
    p = store.TABLES[name]
    return p.stat().st_mtime if p.exists() else 0.0


def read(name: str) -> pd.DataFrame | None:
    return table(name, mtime(name))


@st.cache_data(show_spinner=False)
def _features(_mtime: float) -> pd.DataFrame:
    f = store.read("features")
    if f is None:
        from merchant_pulse.features import build_features
        f = build_features(store.read("sales"), store.read("calendar"))
    f["date"] = pd.to_datetime(f["date"])
    return f


def features() -> pd.DataFrame:
    return _features(mtime("features") + mtime("sales"))


def dates() -> dict:
    return split_dates(features())


@st.cache_resource(show_spinner=False)
def _models(version: str):
    return registry.load_models(version)


def models():
    v = registry.production()
    return _models(v) if v else (None, None)


def alerts() -> pd.DataFrame:
    a = pipeline.alerts_with_status()
    if a is None or a.empty:
        return pd.DataFrame()
    for c in ("start", "end", "peak_date", "run_date"):
        if c in a:
            a[c] = pd.to_datetime(a[c])
    return a


def parse_json(s):
    try:
        return json.loads(s)
    except Exception:
        return s


def series_chart(f: pd.DataFrame, sid: str, start, end, scores: pd.DataFrame | None = None, alerts_df: pd.DataFrame | None = None,
                 title: str | None = None) -> go.Figure:
    g = f[(f.id == sid) & (f.date >= start) & (f.date <= end)].sort_values("date")
    fig = go.Figure()
    if scores is not None and len(scores):
        s = scores[(scores.id == sid) & (scores.date >= start) & (scores.date <= end)].sort_values("date")
        if len(s):
            k = s.k.clip(lower=0.5) if "k" in s else 8.0
            sd = (s.yhat_units + s.yhat_units ** 2 / k) ** 0.5
            fig.add_trace(go.Scatter(x=pd.concat([s.date, s.date[::-1]]), y=pd.concat([(s.yhat_units + 1.96 * sd), (s.yhat_units - 1.96 * sd).clip(lower=0)[::-1]]),
                                     fill="toself", fillcolor=PALETTE["band"], line=dict(width=0), name="95% interval", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=s.date, y=s.yhat_units, name="expected (fast model)", line=dict(color=PALETTE["expected"], dash="dot")))
            al = s[s.is_alert]
            if len(al):
                fig.add_trace(go.Scatter(x=al.date, y=al.units, mode="markers", name="alert day",
                                         marker=dict(color=PALETTE["alert"], size=10, symbol="diamond")))
    fig.add_trace(go.Scatter(x=g.date, y=g.units, name="actual units", line=dict(color=PALETTE["actual"], width=2)))
    promo = g[g.promo == 1]
    if len(promo):
        fig.add_trace(go.Scatter(x=promo.date, y=[0] * len(promo), mode="markers", name="promo day", marker=dict(color="#2da44e", size=6, symbol="square")))
    ev = g[g.event_name != ""]
    for _, r in ev.iterrows():
        fig.add_vline(x=r.date, line=dict(color="#8b949e", dash="dash", width=1))
        fig.add_annotation(x=r.date, y=1, yref="paper", text=r.event_name, showarrow=False, font=dict(size=10, color="#8b949e"), yanchor="bottom")
    if alerts_df is not None and len(alerts_df):
        for _, a in alerts_df[alerts_df.id == sid].iterrows():
            fig.add_vrect(x0=a.start, x1=a.end + pd.Timedelta(days=1), fillcolor=PALETTE.get(a.anomaly_type, "#d9480f"), opacity=0.12, line_width=0,
                          annotation_text=a.anomaly_type, annotation_position="top left")
    fig.update_layout(title=title or sid, height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15),
                      yaxis_title="units / day", hovermode="x unified")
    return fig


def fmt_money(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "-"
