"""Hierarchical reconciliation (figure 3, right column).

Item-level episodes that share a store, category, direction and overlapping dates are explained by one
systemic cause more often than by coincidence. When at least SYSTEMIC_MIN_CHILDREN of them line up, a single
category-level alert is emitted and the children are marked child-explained (kept, but not shown first).
Two or more systemic categories in one store roll up again to a store-level alert.

Low-volume series never alert on their own (materiality floor), so this is also how they get covered: the
category aggregate is scored like any other series by the same ensemble in the pipeline.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from . import config


def _hid(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def reconcile(episodes: pd.DataFrame, min_children: int = config.SYSTEMIC_MIN_CHILDREN) -> pd.DataFrame:
    if episodes.empty:
        return episodes
    ep = episodes.copy()
    ep["level"] = "item"
    ep["parent_id"] = ""
    ep["systemic"] = False
    ep["children"] = ""
    new_rows = []
    for (store, cat, direction), g in ep[ep.level == "item"].groupby(["store_id", "cat_id", "direction"]):
        if len(g) < min_children:
            continue
        # overlapping-window clusters: sort by start, chain while the next starts before the cluster ends + 1 day
        g = g.sort_values("start")
        cluster, cl_end = [], None
        clusters = []
        for _, r in g.iterrows():
            if cluster and r.start > cl_end + pd.Timedelta(days=1):
                clusters.append(cluster); cluster = []
            cluster.append(r); cl_end = r.end if cl_end is None or r.end > cl_end else cl_end
        if cluster:
            clusters.append(cluster)
        for cl in clusters:
            if len(cl) < min_children:
                continue
            ids = [r.alert_id for r in cl]
            start = min(r.start for r in cl); end = max(r.end for r in cl)
            pid = _hid("cat", store, cat, start)
            ep.loc[ep.alert_id.isin(ids), ["parent_id"]] = pid
            types = pd.Series([r.anomaly_type for r in cl]).value_counts()
            new_rows.append({
                "alert_id": pid, "id": f"{cat}_{store}", "start": start, "end": end, "days": int((end - start).days + 1),
                "store_id": store, "state_id": cl[0].state_id, "item_id": "*", "dept_id": "*", "cat_id": cat,
                "anomaly_type": types.index[0], "direction": direction,
                "z_resid": float(max((r.z_resid for r in cl), key=abs)), "z_scale": "n/a", "votes": int(max(r.votes for r in cl)),
                "rule": any(r.rule for r in cl), "detectors": "reconciliation",
                "units_actual": float(sum(r.units_actual for r in cl)), "units_expected": float(sum(r.units_expected for r in cl)),
                "price": float(pd.Series([r.price for r in cl]).mean()), "price_ref": float(pd.Series([r.price_ref for r in cl]).mean()),
                "revenue_at_risk": float(sum(r.revenue_at_risk for r in cl)), "event_name": cl[0].event_name,
                "peak_date": start, "level": "category", "parent_id": "", "systemic": True, "children": ",".join(ids),
            })
    if new_rows:
        cat_alerts = pd.DataFrame(new_rows)
        # store level: >= 2 systemic categories in the same store with overlapping windows and direction
        store_rows = []
        for (store, direction), g in cat_alerts.groupby(["store_id", "direction"]):
            if len(g) >= 2:
                start = g.start.min(); end = g.end.max()
                sid = _hid("store", store, start)
                cat_alerts.loc[g.index, "parent_id"] = sid
                store_rows.append({**g.iloc[0].to_dict(), "alert_id": sid, "id": store, "start": start, "end": end,
                                   "days": int((end - start).days + 1), "cat_id": "*", "anomaly_type": g.anomaly_type.mode().iloc[0],
                                   "units_actual": float(g.units_actual.sum()), "units_expected": float(g.units_expected.sum()),
                                   "revenue_at_risk": float(g.revenue_at_risk.sum()), "level": "store", "parent_id": "",
                                   "systemic": True, "children": ",".join(g.alert_id), "detectors": "reconciliation"})
        ep = pd.concat([ep, cat_alerts] + ([pd.DataFrame(store_rows)] if store_rows else []), ignore_index=True)
    ep["child_explained"] = ep.parent_id != ""
    return ep.sort_values(["child_explained", "revenue_at_risk"], ascending=[True, False]).reset_index(drop=True)
