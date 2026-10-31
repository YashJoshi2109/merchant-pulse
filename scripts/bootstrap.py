"""One-shot setup: generate data, build features, train v1, gate, register, promote, backfill 30 days of alerts."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merchant_pulse import config, pipeline, registry, store, synth  # noqa: E402
from merchant_pulse.features import build_features, split_dates  # noqa: E402


def main(seed: int = config.SEED, log=print) -> None:
    print = log  # noqa: A001
    t0 = time.time()
    print("1/5 generating synthetic M5-style warehouse")
    tables = synth.generate(seed=seed)
    store.write_all(tables)
    print("    ", store.table_stats()[["table", "rows"]].to_string(index=False).replace("\n", "\n     "))
    print("2/5 building features")
    feats = build_features(tables["sales"], tables["calendar"])
    store.write("features", feats)
    d = split_dates(feats)
    print(f"     train < {d['holdout_start'].date()}, calibration from {d['calib_start'].date()}, live window from {d['live_start'].date()}")
    print("3/5 training candidate, running eval harness and gate")
    rec = registry.train_candidate(feats, tables, notes="bootstrap: initial model")
    for c in rec["gate"]["checks"]:
        print(f"       {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {c['value']:.3f} (target {c['target']})")
    if rec["status"] == "rejected":
        print("     gate FAILED. No production model exists yet, so bootstrap promotes with an explicit, logged override.")  # noqa
        registry.register(rec["version"], approved_by="bootstrap (break-glass override: no production baseline)", override=True)
    else:
        registry.register(rec["version"], approved_by="bootstrap")
    registry.promote(rec["version"], reason="bootstrap: first production model (no previous version to canary against)")
    print(f"     {rec['version']} registered and promoted")
    print(f"4/5 backfilling {config.LIVE_WINDOW_DAYS} days of alerts")
    out = pipeline.nightly(backfill_days=config.LIVE_WINDOW_DAYS, log=lambda s: log("    " + s))
    a = out["alerts"]
    if len(a):
        print(f"     {len(a)} alerts: {a.anomaly_type.value_counts().to_dict()} | levels {a.level.value_counts().to_dict()}")
    print("5/5 done in %.1fs. Launch: .venv/bin/streamlit run app/Home.py" % (time.time() - t0))


if __name__ == "__main__":
    main()
