---
title: Merchant Pulse
emoji: 📈
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: __SDK__
app_file: app/Home.py
pinned: false
license: mit
short_description: Sales anomaly detection + prescriptive actions, GCP-style
---

# Merchant Pulse

Sales anomaly detection and prescriptive recommendations for merchants, built as a runnable demo of a GCP-native design.
Every component mirrors one box in the architecture (system HLD, MLOps loop, detection ensemble, eval harness, nightly run);
the Streamlit app explains each page in two columns: what runs here, and what runs on GCP in production.

## Run

```bash
uv venv --python 3.12 .venv            # or python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/bootstrap.py  # synthetic data, features, train v1, eval + gate, backfill 30 days of alerts
.venv/bin/streamlit run app/Home.py
.venv/bin/python -m pytest -q          # tests
```

Optional: `export GEMINI_API_KEY=...` and `pip install google-genai` to have Gemini write merchant notes and answer Text2SQL questions.
Without it, notes come from a deterministic template and the Ask box uses a pattern grammar.

## What is where

| component (diagram) | module | production equivalent |
|---|---|---|
| Sources, ingestion | `merchant_pulse/synth.py`, page 1 | POS → Pub/Sub → Dataflow → BigQuery raw |
| Warehouse | `merchant_pulse/store.py` (parquet + DuckDB views) | BigQuery |
| Features | `merchant_pulse/features.py` | Dataform |
| Detection ensemble | `merchant_pulse/detectors.py` | Vertex AI batch scoring / Cloud Run job |
| Episodes, reconciliation | `episodes.py`, `reconcile.py` | same code |
| Attribution | `attribution.py` (rules + SHAP) | same code |
| Prescriptive layer | `prescribe.py` (T-learner uplift) | EconML on Vertex AI |
| Narrative | `narrative.py` (template, optional Gemini) | Gemini on Vertex AI |
| Eval harness + gate | `evaluate.py` | Vertex AI pipeline eval step |
| Registry, canary, rollback | `registry.py` | Model Registry + Cloud Run traffic split |
| Monitoring | `monitoring.py` (PSI, alert rate, feedback precision) | Vertex AI Model Monitoring |
| Nightly run | `pipeline.py`, `scripts/bootstrap.py` | Cloud Scheduler → Vertex AI pipeline |
| Ask (Text2SQL) | `text2sql.py` | Gemini + BigQuery |

## Data

Synthetic, M5-shaped: 3 states, 10 stores, 3 categories, 7 departments, 28 items → 280 item×store daily series, 2024-01-01 to 2025-12-31.
Weekday and yearly seasonality, SNAP issuance days per state, named holidays (Christmas closed), weekly prices with 10–30% promotions and
item-level elasticity, negative-binomial noise. Ground truth (clean sales, elasticities, planted anomalies) lives in `data/truth` and is
never read by a model. The real M5 files need a Kaggle login; the generator reproduces their shape so the code runs anywhere.

## Detection, briefly

Three detectors vote per series-day. The residual detector is a robust Holt-Winters in log space with holiday, SNAP, price-elasticity and
yearly regressors; its 1-day test is an exact negative-binomial tail, its 3/7/14/28-day tests are frozen-origin cumulative shortfalls
(the forecast is made before the window, so a drop cannot be absorbed by adaptation). Isolation Forest and a dense autoencoder read
holiday-adjusted features standardized per series. Thresholds are set on a clean calibration slice to a false-alert budget; alerts need
2 of 3 votes or one calibrated test at 1.25× its threshold, plus an effect-size floor on long windows and a volume floor (series under
4 units/day roll up to category level). Deterministic rules catch price-file errors and improbable zero-sales runs. Alert days merge
into episodes; three or more items moving together in a store-category become one systemic alert.

## Honest limits

Planted anomalies are material by design (2–3 day +2× baseline spikes, week-long 65% drops, 28-day −4%/day drifts, 3–10 day stockouts,
10× price errors) on series selling at least 6 units/day. Smaller anomalies on noisier series are inside normal variation at this
granularity and are not claimed. The eval report (`registry/models/vN/eval.json`) states precision, recall, F1 by type, false alerts per
1k series-days, time to detect, and recommendation sign accuracy; the gate reads those numbers and blocks promotion when they miss.

## Deploy

- **Hugging Face Spaces** (Streamlit SDK): this README's front matter is the Space config. Push the repo to a Space and it builds.
- **Streamlit Community Cloud**: New app → this GitHub repo → main file `app/Home.py` → Python 3.12.
- Data and the v1 model are committed (~16 MB) so the app opens instantly; on an empty checkout the app bootstraps itself on first load (~15 s).
