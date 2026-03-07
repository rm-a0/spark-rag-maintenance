# Overview

A predictive maintenance pipeline for turbofan jet engines built to get familiar with **PySpark**, **XGBoost**, and **RAG pipelines** in a realistic ML engineering context.

Uses the [NASA C-MAPSS dataset](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) - the industry standard benchmark for remaining useful life prediction.

## What it does

Given multivariate sensor time-series from a turbofan engine, the system:

1. **Ingests** raw sensor data using PySpark and engineers rolling window features at scale
2. **Predicts** remaining useful life (regression) and failure probability within 30 cycles (classification) using XGBoost
3. **Diagnoses** high-risk engines by querying a maintenance manual PDF via a RAG pipeline

```
python main.py --pipeline          # Spark: raw CSV → Parquet with rolling features
python main.py --train             # XGBoost: train regressor + classifier
python main.py --predict --engine 42   # predict RUL + optional RAG diagnostic
```
## Stack

| Layer | Technology |
|---|---|
| Feature engineering | PySpark | 
| Storage | Apache Parquet | 
| ML | XGBoost | 
| RAG | LlamaIndex + HuggingFace embeddings | 
| Config | Pydantic Settings | 
| Environment | uv |

## Project structure

```
turbine-sentinel/
├── data/                     # data for training
│   ├── raw/                  # NASA C-MAPSS source files
│   └── processed/            # generated Parquet
├── manuals/                  # maintenance PDFs for RAG
├── artifacts/                # trained model files
├── notebooks/
│   └── 01_prototype.ipynb    # exploratory prototype
└── src/
    ├── config.py             # Pydantic Settings - single source of truth
    ├── pipeline.py           # Spark ingestion + feature engineering
    ├── train.py              # XGBoost training
    └── predict.py            # inference + RAG diagnostic
main.py                       # CLI entry point
```

## Setup

Requires Python ≥ 3.11 and Java 11 or 17 (for PySpark).

```bash
git clone https://github.com/your-username/turbine-sentinel
cd turbine-sentinel
uv sync
```

Download the C-MAPSS dataset and place the `FD001` files in `data/raw/`. Optionally add a turbine maintenance PDF to `manuals/` for RAG diagnostics.

## Usage

```bash
# Full run from scratch
python main.py --pipeline --train

# Diagnose a specific engine
python main.py --predict --engine 42

# Skip RAG (faster, no API call)
python main.py --predict --engine 42 --no-rag
```

Sample output:

```
==================================================
  TURBINE SENTINEL — ENGINE #42
==================================================
  Predicted RUL      : 18.4 cycles
  Failure probability: 91.2%
  Risk level         : HIGH

  Recommended action:
  Inspect high-pressure turbine seal for thermal
  fatigue. Check sensor_11 trend against manual
  section 4.3 — sustained elevation above 47.5
  indicates compressor stall precursor.
==================================================
```

## Features engineered

For each of the 13 informative sensors, three temporal features are added per cycle:

| Feature | Description |
|---|---|
| `sensor_N_roll_mean` | 5-cycle rolling average - smooths noise |
| `sensor_N_roll_std` | 5-cycle rolling std dev - captures volatility |
| `sensor_N_lag1` | Previous cycle value - encodes direction of change |

Sensors with near-zero variance across all engines (1, 5, 6, 9, 10, 16, 18, 19) are excluded.

## Model performance

Evaluated on a 80:20 split of the FD001 training set.

| Model | Metric | Result |
|---|---|---|
| RUL Regressor | RMSE | ~18 cycles |
| Failure Classifier | ROC-AUC | ~0.97 |

## Data

**Dataset**: [NASA C-MAPSS](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) — Saxena & Goebel (2008)