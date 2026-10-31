"""Merchant note: three sentences, grounded in the alert record.

The template is the default and is fully deterministic. If GEMINI_API_KEY is set and the google-genai
package is importable, the same structured record is sent to Gemini with instructions to rephrase, never to
add facts. Production equivalent: Gemini on Vertex AI inside the project's IAM boundary.
"""
from __future__ import annotations

import os

import pandas as pd


def _fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%b %d")


def template_note(ep: pd.Series, drivers: list[dict], rec: dict) -> str:
    where = f"{ep.item_id} at {ep.store_id}" if ep.level == "item" else (
        f"{ep.cat_id} category at {ep.store_id}" if ep.level == "category" else f"store {ep.store_id}")
    span = _fmt_date(ep.start) if ep.start == ep.end else f"{_fmt_date(ep.start)} to {_fmt_date(ep.end)}"
    exp = max(ep.units_expected, 0.1)
    pct = (ep.units_actual / exp - 1) * 100
    what = {
        "spike": f"Sales for {where} ran {pct:+.0f}% above forecast over {span} ({ep.units_actual:.0f} vs {exp:.0f} units).",
        "drop": f"Sales for {where} fell {abs(pct):.0f}% below forecast over {span} ({ep.units_actual:.0f} vs {exp:.0f} units).",
        "drift": f"Sales for {where} have been sliding; the last {ep.days} days came in {abs(pct):.0f}% under what the trend predicted.",
        "stockout": f"{where} recorded zero sales over {span}, against an expected {exp:.0f} units.",
        "price_error": f"{where} is priced at ${ep.price:.2f} against a reference of ${ep.price_ref:.2f}; volume over {span} was {abs(pct):.0f}% below forecast.",
    }.get(ep.anomaly_type, f"Sales for {where} were anomalous over {span}.")
    why = "Likely drivers: " + "; ".join(d["driver"] for d in drivers) + "." if drivers else "No single driver stands out."
    do = f"Recommended: {rec['action']}."
    if rec.get("expected_revenue", 0) > 0:
        do += f" Expected impact about ${rec['expected_revenue']:,.0f} (confidence {rec['confidence'] * 100:.0f}%)."
    return " ".join([what, why, do])


def gemini_note(ep: pd.Series, drivers: list[dict], rec: dict, fallback: str) -> tuple[str, str]:
    """Returns (note, source). Source is 'gemini' when the call succeeded, otherwise 'template'."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return fallback, "template"
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=key)
        prompt = ("Rewrite this merchant alert as three plain sentences: what happened, why, what to do. "
                  "Use only the facts given; do not add numbers or causes.\n\n" + fallback)
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = (resp.text or "").strip()
        return (text or fallback), ("gemini" if text else "template")
    except Exception:
        return fallback, "template"


def write_note(ep: pd.Series, drivers: list[dict], rec: dict, use_gemini: bool = True) -> tuple[str, str]:
    base = template_note(ep, drivers, rec)
    if use_gemini:
        return gemini_note(ep, drivers, rec, base)
    return base, "template"
