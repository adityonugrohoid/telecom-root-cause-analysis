<div align="center">

# Telecom Root Cause Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Rank causal candidates in telecom alarm cascades using XGBoost, achieving Accuracy@1 of 0.91 on incident-level test splits**

[Getting Started](#getting-started) | [Usage](#usage) | [Methodology](#methodology)

</div>

---

## Table of Contents

- [The Problem](#the-problem)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Data Engineering](#data-engineering)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Related Projects](#related-projects)
- [License](#license)
- [Author](#author)

## The Problem

### Alarm Storm Correlation

During network incidents, a single root fault triggers cascades of hundreds of downstream alarms. NOC engineers spend hours manually correlating events to identify the originating cause, slowing incident resolution and increasing MTTR.

### The Solution

Multi-class XGBoost classifier with per-incident probability ranking. Each alarm event is scored against engineered temporal and severity features; the highest-probability event across a cascade is surfaced as the root cause candidate.

## Features

- **Incident-level ranking** - per-incident GroupShuffleSplit prevents data leakage across alarm cascades
- **SHAP interpretability** - feature-level explanations for every root cause prediction
- **Domain-physics features** - time lag, severity encoding, cascade depth, and alarm co-occurrence signals
- **Custom ranking metrics** - Accuracy@1, Accuracy@3, Accuracy@5, and MRR evaluated per incident
- **Synthetic data generator** - reproducible `RCADataGenerator` producing 500 incidents x 20 events each
- **Pytest data-quality suite** - validates schema, value ranges, and one-root-cause-per-incident invariant

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| ML Model | XGBoost 1.7 (multi:softprob objective) |
| Interpretability | SHAP 0.42+ |
| Feature Engineering | pandas, NumPy, NetworkX |
| Notebook | Jupyter Lab |
| Packaging | uv + pyproject.toml |
| Testing | pytest + pytest-cov |

## Getting Started

### Prerequisites

- Python 3.11+
- uv package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/adityonugrohoid/telecom-root-cause-analysis.git
   cd telecom-root-cause-analysis
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Generate synthetic data:
   ```bash
   uv run python -m root_cause_analysis.data_generator
   ```

## Usage

Run the full analysis in Jupyter Lab (recommended):

```bash
uv run jupyter lab
```

Open `notebooks/02_root_cause_analysis.ipynb` and run all cells. The notebook executes data generation, feature engineering, XGBoost training, and SHAP analysis end-to-end.

To execute non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/02_root_cause_analysis.ipynb
```

Engineer features separately:

```bash
uv run python -m root_cause_analysis.features
```

## Methodology

### Problem Framing

| Attribute | Value |
|-----------|-------|
| Problem Type | Multi-class classification with per-incident ranking |
| Target Variable | `is_root_cause` (binary per event, ranked per incident) |
| Primary Metric | Accuracy@1 (top-ranked candidate = true root cause) |
| Key Challenge | 1 root cause per ~20 cascading events; temporal ordering not always deterministic |

### Training Approach

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost `multi:softprob`, max_depth=6, lr=0.1, n_estimators=200 |
| Features | 7 numerical + 3 categorical (ordinal-encoded) |
| Validation | Incident-level GroupShuffleSplit (100 test incidents) |
| Baseline | Always-predict-first-alarm heuristic |

Feature groups:
- `time_lag_seconds` - elapsed time from incident start
- `severity_encoded` - ordinal mapping from CRITICAL (4) to INFO (1)
- `throughput_delta`, `latency_delta`, `sinr_delta` - per-event KPI deltas
- `alarm_count`, `affected_cells` - cascade breadth signals
- `event_sequence_position` - topological order within cascade

## Results

### Key Findings

| Metric | Score | Notes |
|--------|-------|-------|
| Accuracy@1 | 0.91 | Top-ranked candidate = true root cause |
| Accuracy@3 | 1.00 | True root cause in top 3 candidates |
| MRR | 0.955 | Mean reciprocal rank across 100 test incidents |

### Top Predictors

1. `throughput_delta` - largest KPI deviation signals originating fault
2. `latency_delta` - latency spike precedes downstream alarm propagation
3. `time_lag_seconds` - root causes appear earliest in cascade (though not always first)

## Data Engineering

| Attribute | Value |
|-----------|-------|
| Data Source | Synthetic (`RCADataGenerator`, seed=42) |
| Records | 500 incidents x ~20 events = ~10,000 rows |
| Features | 7 numerical, 3 categorical, 1 datetime |
| Domain Physics | Root causes carry highest severity and lowest time lag; downstream alarms propagate with increasing lag and decreasing severity |

Event types: `hardware_failure`, `software_bug`, `config_error`, `overload`, `external`. Severity levels: `critical`, `major`, `minor`, `warning` with ordinal encoding.

## Project Structure

```
telecom-root-cause-analysis/
├── notebooks/
│   └── 02_root_cause_analysis.ipynb   # End-to-end analysis notebook
├── src/
│   └── root_cause_analysis/
│       ├── config.py                  # Centralized config (paths, model params)
│       ├── data_generator.py          # RCADataGenerator (synthetic alarm cascades)
│       ├── features.py                # Feature engineering pipeline
│       └── models.py                  # BaseModel + XGBoost training/evaluation
├── tests/
│   └── test_data_quality.py           # Schema, value range, and invariant tests
├── data/                              # Runtime artifacts (gitignored)
├── pyproject.toml                     # uv + hatchling build config
└── QUICKSTART.md                      # 5-minute setup reference
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src/root_cause_analysis
```

Tests cover data schema validation, value ranges, one-root-cause-per-incident invariant, and generator reproducibility.

## Related Projects

| Project | Description |
|---------|-------------|
| [telecom-ml-framework](https://github.com/adityonugrohoid/telecom-ml-framework) | Spec-first ML project templates and domain-informed data generators for 6 telecom use cases |
| [telecom-ml-portfolio](https://github.com/adityonugrohoid/telecom-ml-portfolio) | Index of 6 end-to-end telecom ML projects on synthetic network data |
| [telecom-churn-prediction](https://github.com/adityonugrohoid/telecom-churn-prediction) | Binary classification predicting subscriber churn (XGBoost, AUROC 0.86) |
| [telecom-anomaly-detection](https://github.com/adityonugrohoid/telecom-anomaly-detection) | Unsupervised cell-level anomaly detection on KPI time-series (Isolation Forest, F1 0.70) |
| [telecom-qoe-prediction](https://github.com/adityonugrohoid/telecom-qoe-prediction) | Session-level MOS regression from network KPIs (LightGBM, RMSE 0.45) |
| [telecom-capacity-forecasting](https://github.com/adityonugrohoid/telecom-capacity-forecasting) | Hourly per-cell traffic forecasting (LightGBM, MAPE 14.5%) |
| [telecom-network-optimization](https://github.com/adityonugrohoid/telecom-network-optimization) | RL-based RAN parameter tuning (Q-Learning, +61% vs random) |

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Adityo Nugroho** ([@adityonugrohoid](https://github.com/adityonugrohoid))
