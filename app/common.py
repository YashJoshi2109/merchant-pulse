"""Shared helpers for the Streamlit pages: data access, caching, chart styling, branding and annotations."""
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

AUTHOR = "Yash Joshi"
AUTHOR_TITLE = "Senior Data Scientist · ML Engineer"
AUTHOR_EMAIL = "yash.joshi.careers@gmail.com"
REPO_URL = "https://github.com/YashJoshi2109/merchant-pulse"

PALETTE = {"actual": "#4c9aff", "expected": "#8b949e", "alert": "#ff8249", "band": "rgba(139,148,158,0.18)",
           "spike": "#ff8249", "drop": "#4c9aff", "drift": "#b083f0", "stockout": "#e3b341", "price_error": "#ff6b6b",
           "ok": "#3fb950", "muted": "#8b949e", "promo": "#3fb950"}
TYPE_ORDER = ["spike", "drop", "drift", "stockout", "price_error"]
STATE_NAME = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}

CSS = """
<style>
:root { --accent:#ff8249; --panel:#1a222d; --muted:#8b949e; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
div[data-testid="stMetric"] { background: var(--panel); border: 1px solid rgba(139,148,158,.18); border-radius: 10px; padding: 12px 14px; }
div[data-testid="stMetric"] label { color: var(--muted) !important; font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; }
div[data-testid="stMetricValue"] { font-size: 1.55rem; }
.mp-hero { background: linear-gradient(135deg, rgba(255,130,73,.18), rgba(76,154,255,.10)); border: 1px solid rgba(255,130,73,.35);
           border-radius: 14px; padding: 22px 26px; margin-bottom: 14px; }
.mp-hero h1 { margin: 0 0 6px 0; font-size: 2rem; letter-spacing: -.01em; }
.mp-hero p { margin: 0; color: #c9d1d9; max-width: 72ch; }
.mp-chip { display:inline-block; background: rgba(255,130,73,.15); border:1px solid rgba(255,130,73,.45); color:#ffb48f;
           border-radius: 999px; padding: 2px 10px; font-size: .78rem; margin-right: 6px; margin-top: 10px; }
.mp-author { border-top: 1px solid rgba(139,148,158,.25); margin-top: 14px; padding-top: 10px; font-size: .86rem; color:#c9d1d9; }
.mp-author b { color: #fff; }
.mp-step { background: var(--panel); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 6px 0; }
</style>
"""


def ensure_bootstrapped() -> None:
    """First run on a fresh checkout: build data, train v1, backfill alerts. ~15 s. No-op when data exists."""
    if store.exists("sales") and registry.production():
        return
    with st.status("First run: generating data, training v1, backfilling alerts (about 15 s)…", expanded=True) as s:
        import importlib
        boot = importlib.import_module("scripts.bootstrap")
        boot.main(log=lambda m: s.write(m))
        s.update(label="Bootstrap complete", state="complete", expanded=False)
    st.cache_data.clear(); st.cache_resource.clear()


def setup(title: str, icon: str = "📈") -> None:
    st.set_page_config(page_title=f"Merchant Pulse · {title}", page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_bootstrapped()
    with st.sidebar:
        st.markdown("### 📈 Merchant Pulse")
        st.caption("Sales anomaly detection and prescriptive recommendations for merchants. GCP-native design, runnable demo.")
        st.session_state.setdefault("annot", True)
        st.toggle("Explain each component", key="annot", help="Two columns per section: what runs here, and what runs on GCP in production.")
        prod = registry.production()
        st.caption(f"Production model **{prod or 'none'}**")
        if store.exists("sales"):
            d = dates()
            st.caption(f"Data through {d['last'].date()} · live window from {d['live_start'].date()}")
        st.markdown(f"""<div class="mp-author"><b>Built by {AUTHOR}</b><br>{AUTHOR_TITLE}<br>
<a href="{REPO_URL}" target="_blank">GitHub repo</a> · <a href="mailto:{AUTHOR_EMAIL}">{AUTHOR_EMAIL}</a><br>
<span style="color:#8b949e">Design, data engine, models, MLOps and app: all mine. Oct 2026.</span></div>""", unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = "".join(f'<span class="mp-chip">{c}</span>' for c in (chips or []))
    st.markdown(f'<div class="mp-hero"><h1>{title}</h1><p>{subtitle}</p>{chips_html}</div>', unsafe_allow_html=True)


def annotate(title: str, here: str, prod: str, expanded: bool = False) -> None:
    """Self-describing panel: left column is what the demo does, right column is the production GCP equivalent."""
    if not st.session_state.get("annot", True):
        return
    with st.expander(f"ⓘ {title}", expanded=expanded):
        c1, c2 = st.columns(2)
        c1.markdown("**Here (demo)**\n\n" + here)
        c2.markdown("**In production (GCP)**\n\n" + prod)


def style(fig: go.Figure, height: int | None = None, title: str | None = None, legend: bool = True) -> go.Figure:
    """One look for every chart: transparent panels, muted grid, horizontal legend."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12, color="#c9d1d9"),
                      margin=dict(l=10, r=10, t=44 if (title or (fig.layout.title and fig.layout.title.text)) else 14, b=10),
                      legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)) if legend else dict(),
                      showlegend=legend, hoverlabel=dict(bgcolor="#1a222d", font_size=12))
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=15)))
    fig.update_xaxes(gridcolor="rgba(139,148,158,0.14)", zeroline=False, showline=False)
    fig.update_yaxes(gridcolor="rgba(139,148,158,0.14)", zeroline=False, showline=False)
    return fig


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


def scores() -> pd.DataFrame | None:
    sc = read("scores")
    if sc is not None:
        sc = sc.copy(); sc["date"] = pd.to_datetime(sc["date"])
    return sc


def parse_json(s):
    try:
        return json.loads(s)
    except Exception:
        return s


def series_chart(f: pd.DataFrame, sid: str, start, end, scores: pd.DataFrame | None = None, alerts_df: pd.DataFrame | None = None,
                 title: str | None = None, height: int = 380) -> go.Figure:
    g = f[(f.id == sid) & (f.date >= start) & (f.date <= end)].sort_values("date")
    fig = go.Figure()
    if scores is not None and len(scores):
        s = scores[(scores.id == sid) & (scores.date >= start) & (scores.date <= end)].sort_values("date")
        if len(s):
            k = s.k.clip(lower=0.5) if "k" in s else 8.0
            sd = (s.yhat_units + s.yhat_units ** 2 / k) ** 0.5
            fig.add_trace(go.Scatter(x=pd.concat([s.date, s.date[::-1]]), y=pd.concat([(s.yhat_units + 1.96 * sd), (s.yhat_units - 1.96 * sd).clip(lower=0)[::-1]]),
                                     fill="toself", fillcolor=PALETTE["band"], line=dict(width=0), name="95% interval", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=s.date, y=s.yhat_units, name="expected", line=dict(color=PALETTE["expected"], dash="dot")))
            al = s[s.is_alert]
            if len(al):
                fig.add_trace(go.Scatter(x=al.date, y=al.units, mode="markers", name="alert day",
                                         marker=dict(color=PALETTE["alert"], size=11, symbol="diamond", line=dict(color="#fff", width=1))))
    fig.add_trace(go.Scatter(x=g.date, y=g.units, name="actual units", line=dict(color=PALETTE["actual"], width=2.2)))
    promo = g[g.promo == 1]
    if len(promo):
        fig.add_trace(go.Scatter(x=promo.date, y=[0] * len(promo), mode="markers", name="promo day", marker=dict(color=PALETTE["promo"], size=6, symbol="square")))
    ev = g[g.event_name != ""]
    for _, r in ev.iterrows():
        fig.add_vline(x=r.date, line=dict(color="#8b949e", dash="dash", width=1))
        fig.add_annotation(x=r.date, y=1, yref="paper", text=r.event_name, showarrow=False, font=dict(size=10, color="#8b949e"), yanchor="bottom")
    if alerts_df is not None and len(alerts_df):
        for _, a in alerts_df[alerts_df.id == sid].iterrows():
            fig.add_vrect(x0=a.start, x1=a.end + pd.Timedelta(days=1), fillcolor=PALETTE.get(a.anomaly_type, "#ff8249"), opacity=0.14, line_width=0,
                          annotation_text=a.anomaly_type, annotation_position="top left", annotation_font_size=10)
    fig.update_layout(yaxis_title="units / day", hovermode="x unified")
    return style(fig, height=height, title=title or sid)


def fmt_money(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "-"
