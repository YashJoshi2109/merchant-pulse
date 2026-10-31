"""Synthetic M5-style retail data.

Why synthetic: the real M5 files need a Kaggle login. This generator reproduces M5's shape
(3 states, 10 stores, 3 categories, 7 departments, item x store daily unit sales, weekly prices,
SNAP days, holiday events) with known ground truth: clean sales, true price elasticities and a
table of planted anomalies. Ground truth lives in data/truth and is never read by the models.

Production equivalent: POS feeds -> Pub/Sub -> Dataflow -> BigQuery raw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

STATES = {"CA": ["CA_1", "CA_2", "CA_3", "CA_4"], "TX": ["TX_1", "TX_2", "TX_3"], "WI": ["WI_1", "WI_2", "WI_3"]}
DEPTS = {"HOBBIES": ["HOBBIES_1", "HOBBIES_2"], "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
         "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"]}
ITEMS_PER_DEPT = 4
BASE_DEMAND = {"HOBBIES": 1.6, "HOUSEHOLD": 2.6, "FOODS": 6.0}
BASE_PRICE = {"HOBBIES": 6.0, "HOUSEHOLD": 8.0, "FOODS": 3.5}
YEARLY_AMP = {"HOBBIES": 0.20, "HOUSEHOLD": 0.12, "FOODS": 0.10}
YEARLY_PHASE = {"HOBBIES": 330, "HOUSEHOLD": 300, "FOODS": 190}   # day-of-year of the peak
DOW_MULT = np.array([0.95, 0.90, 0.90, 0.92, 1.00, 1.25, 1.30])    # Mon..Sun
# M5 SNAP schedule: which days of the month SNAP benefits are issued, by state
SNAP_DAYS = {"CA": set(range(1, 11)), "TX": {1, 3, 5, 6, 7, 9, 11, 12, 13, 15}, "WI": {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}}
SNAP_MULT = {"FOODS": 1.25, "HOUSEHOLD": 1.03, "HOBBIES": 1.0}
NB_SHAPE = 12.0   # Poisson-gamma dispersion: var = mu + mu^2/NB_SHAPE
MIN_BASELINE = 6.0   # anomalies are planted only in series selling >= this many units/day (item-level alert scope)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    off = (weekday - d.weekday()) % 7
    return d + pd.Timedelta(days=off + 7 * (n - 1))


def _easter(year: int) -> pd.Timestamp:
    a = year % 19; b = year // 100; c = year % 100; d = b // 4; e = b % 4
    f = (b + 8) // 25; g = (b - f + 1) // 3; h = (19 * a + b - d - g + 15) % 30
    i = c // 4; k = c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31; day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def build_calendar(start: str = config.START, end: str = config.END) -> pd.DataFrame:
    """Calendar with weekday, M5-style week id, SNAP flags per state and named events with their effects."""
    dates = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["dow"] = cal.date.dt.weekday
    cal["wm_yr_wk"] = ((cal.date - dates[0]).dt.days // 7).astype(int)
    cal["doy"] = cal.date.dt.dayofyear
    for st, days in SNAP_DAYS.items():
        cal[f"snap_{st}"] = cal.date.dt.day.isin(days).astype(int)
    cal["event_name"] = ""
    events: dict[pd.Timestamp, str] = {}
    for y in sorted(set(dates.year)):
        tg = _nth_weekday(y, 11, 3, 4)
        events.update({
            pd.Timestamp(y, 12, 25): "Christmas", tg: "Thanksgiving", tg + pd.Timedelta(days=1): "BlackFriday",
            _nth_weekday(y, 2, 6, 2): "SuperBowl", pd.Timestamp(y, 7, 4): "IndependenceDay", _easter(y): "Easter",
            pd.Timestamp(y, 10, 31): "Halloween", _nth_weekday(y, 5, 0, 1) + pd.Timedelta(days=21): "MemorialDay",
            _nth_weekday(y, 9, 0, 1): "LaborDay", pd.Timestamp(y, 1, 1): "NewYear",
        })
    cal["event_name"] = cal.date.map(events).fillna("")
    return cal


# multiplicative effect of each event by category; Christmas = closed
EVENT_EFFECT = {
    "Christmas": {"HOBBIES": 0.0, "HOUSEHOLD": 0.0, "FOODS": 0.0},
    "Thanksgiving": {"HOBBIES": 0.5, "HOUSEHOLD": 0.5, "FOODS": 0.7},
    "BlackFriday": {"HOBBIES": 1.6, "HOUSEHOLD": 1.5, "FOODS": 1.05},
    "SuperBowl": {"HOBBIES": 1.0, "HOUSEHOLD": 1.05, "FOODS": 1.4},
    "IndependenceDay": {"HOBBIES": 1.05, "HOUSEHOLD": 1.0, "FOODS": 1.3},
    "Easter": {"HOBBIES": 1.1, "HOUSEHOLD": 1.0, "FOODS": 1.2},
    "Halloween": {"HOBBIES": 1.3, "HOUSEHOLD": 1.0, "FOODS": 1.15},
    "MemorialDay": {"HOBBIES": 1.0, "HOUSEHOLD": 1.05, "FOODS": 1.15},
    "LaborDay": {"HOBBIES": 1.0, "HOUSEHOLD": 1.05, "FOODS": 1.12},
    "NewYear": {"HOBBIES": 0.8, "HOUSEHOLD": 0.8, "FOODS": 0.85},
}


def build_hierarchy() -> pd.DataFrame:
    rows = []
    for state, stores in STATES.items():
        for store in stores:
            for cat, depts in DEPTS.items():
                for dept in depts:
                    for i in range(1, ITEMS_PER_DEPT + 1):
                        item = f"{dept}_{i:03d}"
                        rows.append({"id": f"{item}_{store}", "item_id": item, "dept_id": dept, "cat_id": cat,
                                     "store_id": store, "state_id": state})
    return pd.DataFrame(rows)


def build_prices(hier: pd.DataFrame, cal: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly sell prices per item x store with promo weeks (10-30% cuts) and true elasticities.

    Returns (prices long by id x wm_yr_wk, truth table of base price, elasticity, promo weeks).
    """
    weeks = np.sort(cal.wm_yr_wk.unique())
    items = hier.item_id.unique()
    base_price = {it: float(np.round(BASE_PRICE[it.split("_")[0]] * rng.lognormal(0, 0.35), 2)) for it in items}
    elasticity = {it: float(rng.uniform(-2.5, -1.2)) for it in items}
    rec, truth = [], []
    for _, r in hier.iterrows():
        store_adj = 1.0 + (0.03 if r.state_id == "CA" else -0.02 if r.state_id == "TX" else 0.0)
        p0 = round(base_price[r.item_id] * store_adj, 2)
        price = np.full(len(weeks), p0)
        w = 0
        promo_weeks = []
        while w < len(weeks):
            if rng.random() < 0.06:
                dur = int(rng.integers(1, 3)); cut = float(rng.uniform(0.10, 0.30))
                for k in range(w, min(w + dur, len(weeks))):
                    price[k] = round(p0 * (1 - cut), 2); promo_weeks.append((int(weeks[k]), cut))
                w += dur + 2
            else:
                w += 1
        for wk, p in zip(weeks, price):
            rec.append((r.id, int(wk), float(p)))
        truth.append({"id": r.id, "item_id": r.item_id, "base_price": p0, "elasticity": elasticity[r.item_id],
                      "n_promo_weeks": len(promo_weeks)})
    prices = pd.DataFrame(rec, columns=["id", "wm_yr_wk", "sell_price"])
    return prices, pd.DataFrame(truth)


def build_sales(hier: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame, truth_price: pd.DataFrame,
                rng: np.random.Generator) -> pd.DataFrame:
    """Daily units per series = base x dow x yearly x trend x SNAP x event x price-elasticity, NegBin noise."""
    T = len(cal)
    doy = cal.doy.values; dow = cal.dow.values
    tnorm = np.arange(T) / 365.0
    base_item = {it: BASE_DEMAND[it.split("_")[0]] * rng.lognormal(0, 0.5) for it in hier.item_id.unique()}
    store_size = {s: rng.uniform(0.7, 1.3) for st in STATES.values() for s in st}
    store_size["CA_3"] = 1.6
    trend_item = {it: rng.uniform(-0.05, 0.05) for it in hier.item_id.unique()}
    el = truth_price.set_index("id").elasticity.to_dict(); bp = truth_price.set_index("id").base_price.to_dict()
    wk_price = prices.pivot(index="wm_yr_wk", columns="id", values="sell_price")
    out = []
    for _, r in hier.iterrows():
        cat = r.cat_id
        lvl = base_item[r.item_id] * store_size[r.store_id]
        yearly = 1 + YEARLY_AMP[cat] * np.cos(2 * np.pi * (doy - YEARLY_PHASE[cat]) / 365.25)
        trend = 1 + trend_item[r.item_id] * tnorm
        snap = np.where(cal[f"snap_{r.state_id}"].values == 1, SNAP_MULT[cat], 1.0)
        ev = np.array([EVENT_EFFECT.get(e, {}).get(cat, 1.0) if e else 1.0 for e in cal.event_name.values])
        price = wk_price[r.id].reindex(cal.wm_yr_wk.values).values
        price_eff = (price / bp[r.id]) ** el[r.id]
        lam = lvl * DOW_MULT[dow] * yearly * trend * snap * ev * price_eff
        units = rng.poisson(rng.gamma(NB_SHAPE, np.maximum(lam, 1e-6) / NB_SHAPE))
        units = np.where(ev == 0.0, 0, units)
        out.append(pd.DataFrame({"id": r.id, "date": cal.date.values, "units": units.astype(int),
                                 "sell_price": np.round(price, 2)}))
    return pd.concat(out, ignore_index=True)


def inject_anomalies(sales: pd.DataFrame, series_ids: list[str], date_min: pd.Timestamp, date_max: pd.Timestamp,
                     rng: np.random.Generator, n_per_type: int = 3, types: list[str] | None = None,
                     min_gap_days: int = 35, min_baseline: float = MIN_BASELINE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plant labeled anomalies into a copy of `sales`. Shared by the demo (planted truth) and the eval harness.

    spike: 2-3 days +2x baseline (about x3) | drop: 7-10 days x0.35 | drift: 28 days, -4%/day compounding to -68%
    stockout: 3-10 days of zeros | price_error: 1-3 days price x10, units x0.2
    Magnitudes are deliberately material: at item x store level daily sales are heavy-tailed, and a one-day
    +150% blip on a 5-unit series is inside normal variation. The harness measures what a merchant would care about.
    Returns (modified sales, labels[id, anomaly_type, start, end, magnitude]).
    """
    types = types or config.ANOMALY_TYPES
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"])
    base = s[(s.date >= date_min - pd.Timedelta(days=56)) & (s.date < date_min)].groupby("id").units.mean()
    eligible = [i for i in series_ids if base.get(i, 0.0) >= min_baseline]
    if not eligible:
        raise ValueError("no series meets min_baseline for anomaly injection")
    series_ids = eligible
    idx = s.set_index(["id", "date"]).index
    labels = []
    used: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    span_days = (date_max - date_min).days
    u = s["units"].to_numpy(dtype=float, copy=True)
    p = s["sell_price"].to_numpy(dtype=float, copy=True)
    for t in types:
        for _ in range(n_per_type):
            for _try in range(50):
                sid = str(rng.choice(series_ids))
                dur = {"spike": rng.integers(2, 4), "drop": rng.integers(7, 11), "drift": 28,
                       "stockout": rng.integers(3, 11), "price_error": rng.integers(1, 4)}[t]
                dur = int(dur)
                if span_days - dur < 1:
                    continue
                start = date_min + pd.Timedelta(days=int(rng.integers(0, span_days - dur + 1)))
                end = start + pd.Timedelta(days=dur - 1)
                clash = any(not (end + pd.Timedelta(days=min_gap_days) < a or start - pd.Timedelta(days=min_gap_days) > b)
                            for a, b in used.get(sid, []))
                if not clash:
                    break
            else:
                continue
            used.setdefault(sid, []).append((start, end))
            days = pd.date_range(start, end, freq="D")
            loc = idx.get_indexer([(sid, d) for d in days])
            loc = loc[loc >= 0]
            if t == "spike":
                mag = 3.0; u[loc] = u[loc] + np.round(2.0 * base.get(sid, u[loc].mean()))
            elif t == "drop":
                mag = 0.35; u[loc] = np.round(u[loc] * mag).astype(int)
            elif t == "drift":
                mult = np.array([(1 - 0.04) ** (k + 1) for k in range(len(loc))]); mag = float(mult[-1])
                u[loc] = np.round(u[loc] * mult).astype(int)
            elif t == "stockout":
                mag = 0.0; u[loc] = 0
            elif t == "price_error":
                mag = 10.0; p[loc] = np.round(p[loc] * 10, 2); u[loc] = np.round(u[loc] * 0.2).astype(int)
            labels.append({"id": sid, "anomaly_type": t, "start": start, "end": end, "magnitude": float(mag)})
    s["units"] = u.astype(int); s["sell_price"] = np.round(p, 2)
    return s, pd.DataFrame(labels)


def generate(seed: int = config.SEED, plant_live_anomalies: bool = True) -> dict[str, pd.DataFrame]:
    """Build the full synthetic warehouse. Returns dict of tables; see store.write_all."""
    rng = np.random.default_rng(seed)
    cal = build_calendar()
    hier = build_hierarchy()
    prices, truth_price = build_prices(hier, cal, rng)
    clean = build_sales(hier, cal, prices, truth_price, rng)
    sales = clean
    planted = pd.DataFrame(columns=["id", "anomaly_type", "start", "end", "magnitude"])
    if plant_live_anomalies:
        end = pd.Timestamp(config.END)
        live_start = end - pd.Timedelta(days=config.LIVE_WINDOW_DAYS - 1)
        sales, planted = inject_anomalies(clean, hier.id.tolist(), live_start, end, rng, n_per_type=3)
    sales = sales.merge(hier, on="id")
    return {"calendar": cal, "hierarchy": hier, "prices": prices, "sales": sales,
            "truth_sales_clean": clean, "truth_planted": planted, "truth_elasticity": truth_price}
