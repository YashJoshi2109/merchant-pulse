"""Central configuration. In production these live in a config bucket or Vertex AI pipeline params."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRUTH_DIR = DATA_DIR / "truth"
REGISTRY_DIR = ROOT / "registry"
MODELS_DIR = REGISTRY_DIR / "models"

SEED = 42
START = "2024-01-01"
END = "2025-12-31"

# Time windows (days). Train = everything before the holdout. Live window = what the dashboard shows.
HOLDOUT_DAYS = 90
CALIB_DAYS = 60          # last N train days used to set detector thresholds
LIVE_WINDOW_DAYS = 30    # backfill scoring window for the demo
WINDOW = 28              # autoencoder window length

# Detection
PER_DETECTOR_FA = 0.01   # per-detector flag rate allowed on the clean calibration slice
VOTE_MIN = 2             # alerts need >= 2 of 3 detectors
SYSTEMIC_MIN_CHILDREN = 3

# Gate thresholds (figure 2) and SLOs
GATE = {"f1_min": 0.85, "fa_per_1k_max": 5.0, "ttd_median_max": 2.0, "sign_acc_min": 0.80}
SLO = {"p95_latency_ms": 300, "rollback_min": 5, "alerts_per_merchant_day": 10, "precision_min": 0.75}

# Monitoring
PSI_RETRAIN = 0.20
MONITOR_FEATURES = ["log_units", "z28", "price_pct", "ratio_7", "zero_streak"]

ANOMALY_TYPES = ["spike", "drop", "drift", "stockout", "price_error"]
