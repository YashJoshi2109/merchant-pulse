import numpy as np
import pandas as pd

from merchant_pulse import config
from merchant_pulse.evaluate import gate
from merchant_pulse.monitoring import psi
from merchant_pulse.reconcile import reconcile
from merchant_pulse.text2sql import FORBIDDEN, to_sql


def test_gate_pass_and_fail():
    det_ok = {"f1": 0.9, "fa_per_1k": 1.0, "ttd_median": 1.0}
    rec_ok = {"sign_accuracy": 0.9}
    assert gate(det_ok, rec_ok)["pass"]
    det_bad = {**det_ok, "fa_per_1k": 9.0}
    g = gate(det_bad, rec_ok)
    assert not g["pass"]
    assert [c for c in g["checks"] if not c["pass"]][0]["check"].startswith("False alerts")


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 2000)) < 0.05
    assert psi(ref, rng.normal(1.5, 1, 2000)) > config.PSI_RETRAIN


def _ep(i, store, cat, start, days=3, direction="down"):
    s = pd.Timestamp(start)
    return {"alert_id": f"a{i}", "id": f"{cat}_1_00{i}_{store}", "start": s, "end": s + pd.Timedelta(days=days - 1), "days": days,
            "store_id": store, "state_id": store[:2], "item_id": f"{cat}_1_00{i}", "dept_id": f"{cat}_1", "cat_id": cat,
            "anomaly_type": "drop", "direction": direction, "z_resid": -3.5, "z_scale": "7d", "votes": 2, "rule": False,
            "detectors": "residual,iforest", "units_actual": 10.0, "units_expected": 30.0, "price": 3.0, "price_ref": 3.0,
            "revenue_at_risk": 60.0, "event_name": "", "peak_date": s}


def test_reconcile_rolls_up_three_items():
    ep = pd.DataFrame([_ep(1, "CA_1", "FOODS", "2025-12-10"), _ep(2, "CA_1", "FOODS", "2025-12-11"), _ep(3, "CA_1", "FOODS", "2025-12-10"),
                       _ep(4, "TX_1", "FOODS", "2025-12-10")])
    out = reconcile(ep, min_children=3)
    cat = out[out.level == "category"]
    assert len(cat) == 1 and cat.iloc[0].store_id == "CA_1" and cat.iloc[0].systemic
    assert out[(out.level == "item") & (out.store_id == "CA_1")].child_explained.all()
    assert not out[(out.level == "item") & (out.store_id == "TX_1")].child_explained.any()


def test_text2sql_patterns_are_read_only():
    for q in ["top 5 stores by revenue last 30 days", "sales of FOODS_3_001 in CA_1 last 14 days", "alerts by type", "stockouts",
              "units by category last week", "which store had the most alerts"]:
        sql, how = to_sql(q)
        assert sql and how == "pattern" and not FORBIDDEN.search(sql), q
    # raw DDL is refused by the read-only guard even if a pattern happens to match a word in it
    from merchant_pulse.text2sql import run
    df, sql, how, err = run("DELETE FROM sales")
    assert df is None and ("Read-only" in err or how == "none")
    assert FORBIDDEN.search("DROP TABLE sales")
