"""Ask box: natural language → SQL over the warehouse views.

A small pattern grammar covers the questions merchants actually ask (top stores, sales of an item, alerts by
type). Anything that already looks like SQL runs as-is, read-only. If GEMINI_API_KEY is set, unmatched
questions go to Gemini with the schema; the generated SQL still has to pass the read-only guard.
Production equivalent: Gemini on Vertex AI with BigQuery schema grounding, or BigQuery's own data canvas.
"""
from __future__ import annotations

import os
import re

from . import store

SCHEMA = """
sales(id, item_id, dept_id, cat_id, store_id, state_id, date, units, sell_price)
calendar(date, dow, wm_yr_wk, event_name, snap_CA, snap_TX, snap_WI)
alerts(alert_id, id, item_id, cat_id, store_id, state_id, level, anomaly_type, direction, start, end, days,
       units_actual, units_expected, revenue_at_risk, action, expected_revenue, confidence, note, model_version)
feedback(alert_id, verdict, user, ts)
monitoring(ts, as_of, model_version, psi_*, alert_episodes_per_1k_series_days_7d, feedback_precision, retrain_trigger)
"""

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|copy|pragma|install|load)\b", re.I)


def _days(q: str, default: int = 30) -> int:
    m = re.search(r"last\s+(\d+)\s+day", q)
    if m:
        return int(m.group(1))
    if "last week" in q or "past week" in q:
        return 7
    if "last month" in q or "past month" in q:
        return 30
    return default


def _n(q: str, default: int = 5) -> int:
    m = re.search(r"top\s+(\d+)", q)
    return int(m.group(1)) if m else default


def to_sql(question: str) -> tuple[str, str]:
    """Returns (sql, how) where how is 'passthrough' | 'pattern' | 'gemini' | 'none'."""
    q = question.strip()
    ql = q.lower()
    if ql.startswith(("select", "with")):
        return q, "passthrough"
    d = _days(ql)
    win = f"date > (SELECT max(date) FROM sales) - INTERVAL {d} DAY"
    metric = "sum(units * sell_price)" if any(w in ql for w in ("revenue", "sales $", "dollar")) else "sum(units)"
    mname = "revenue" if metric.startswith("sum(units *") else "units"
    m = re.search(r"top\s*\d*\s*(stores?|items?|categor(?:y|ies)|states?|departments?)", ql)
    if m:
        dim = {"store": "store_id", "item": "item_id", "categor": "cat_id", "state": "state_id", "department": "dept_id"}
        col = next(v for k, v in dim.items() if m.group(1).startswith(k))
        return f"SELECT {col}, {metric} AS {mname} FROM sales WHERE {win} GROUP BY 1 ORDER BY 2 DESC LIMIT {_n(ql)}", "pattern"
    m = re.search(r"(?:sales|units)\s+(?:of|for)\s+([A-Za-z]+_\d_\d{3})(?:\s+(?:in|at)\s+([A-Z]{2}_\d))?", q)
    if m:
        item, st = m.group(1), m.group(2)
        where = f"item_id = '{item}'" + (f" AND store_id = '{st}'" if st else "")
        return f"SELECT date, store_id, units, sell_price FROM sales WHERE {where} AND {win} ORDER BY date, store_id", "pattern"
    if "alert" in ql:
        if "most alerts" in ql or "which store" in ql:
            return "SELECT store_id, count(*) AS alerts, round(sum(revenue_at_risk), 0) AS revenue_at_risk FROM alerts WHERE level = 'item' GROUP BY 1 ORDER BY 2 DESC", "pattern"
        by = "anomaly_type" if "type" in ql else "cat_id" if "categor" in ql else "store_id" if "store" in ql else "anomaly_type"
        return f"SELECT {by}, count(*) AS alerts, round(sum(revenue_at_risk), 0) AS revenue_at_risk FROM alerts WHERE level = 'item' GROUP BY 1 ORDER BY 2 DESC", "pattern"
    for t in ("stockout", "price_error", "spike", "drop", "drift"):
        if t.replace("_", " ") in ql or t in ql:
            return f"SELECT start, \"end\", id, store_id, revenue_at_risk, action FROM alerts WHERE anomaly_type = '{t}' ORDER BY revenue_at_risk DESC LIMIT 50", "pattern"
    if "by category" in ql or "per category" in ql:
        return f"SELECT cat_id, {metric} AS {mname} FROM sales WHERE {win} GROUP BY 1 ORDER BY 2 DESC", "pattern"
    if "by store" in ql or "per store" in ql:
        return f"SELECT store_id, {metric} AS {mname} FROM sales WHERE {win} GROUP BY 1 ORDER BY 2 DESC", "pattern"
    if "by state" in ql:
        return f"SELECT state_id, {metric} AS {mname} FROM sales WHERE {win} GROUP BY 1 ORDER BY 2 DESC", "pattern"
    if "snap" in ql:
        return ("SELECT CASE WHEN (s.state_id='CA' AND c.snap_CA=1) OR (s.state_id='TX' AND c.snap_TX=1) OR (s.state_id='WI' AND c.snap_WI=1) THEN 'SNAP day' ELSE 'other' END AS day_type, "
                "s.cat_id, round(avg(s.units), 2) AS avg_units FROM sales s JOIN calendar c USING(date) GROUP BY 1, 2 ORDER BY 2, 1", "pattern")
    if "total" in ql or "how many units" in ql:
        return f"SELECT {metric} AS {mname}, count(DISTINCT id) AS series FROM sales WHERE {win}", "pattern"
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        try:
            from google import genai  # type: ignore
            client = genai.Client(api_key=key)
            prompt = f"DuckDB SQL only, no prose. Schema:\n{SCHEMA}\nQuestion: {q}\nSQL:"
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            sql = (resp.text or "").strip().strip("`").replace("sql\n", "", 1).strip()
            return sql, "gemini"
        except Exception:
            pass
    return "", "none"


def run(question: str):
    sql, how = to_sql(question)
    if not sql:
        return None, sql, how, "No pattern matched. Try: 'top 5 stores by revenue last 30 days', 'sales of FOODS_3_001 in CA_1 last 14 days', 'alerts by type', 'stockouts', or write SQL."
    if FORBIDDEN.search(sql):
        return None, sql, how, "Read-only: statement refused."
    try:
        return store.sql(sql), sql, how, ""
    except Exception as ex:
        return None, sql, how, f"SQL error: {ex}"
