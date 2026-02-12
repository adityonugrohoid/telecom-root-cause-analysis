# Telecom Root Cause Analysis

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Business Context

When network incidents occur, alarm storms generate hundreds of events. NOC teams waste hours correlating alarms manually. ML-driven RCA ranks causal candidates to identify the root cause within seconds.

## Problem Framing

Multi-class classification using XGBoost.

- **Target:** `is_root_cause`
- **Primary Metrics:** Accuracy@K, MRR (Mean Reciprocal Rank)
- **Challenges:**
  - Highly imbalanced (1 root cause per ~20 cascading events)
  - Temporal ordering is critical for causal inference
  - Alarm severity hierarchy must be encoded meaningfully

## Data Engineering

Incident-event cascading data structured around causal chains:

- Each incident contains **1 root cause event** (lag=0) and multiple **cascading downstream events** with increasing time lags and decreasing severity
- **Severity encoding** -- hierarchical mapping from CRITICAL to INFO
- **Temporal features** -- time lag from first alarm, inter-event gaps
- **Cascade depth** -- topological distance from the originating fault

Domain physics: root causes appear first (lowest time lag), carry the highest severity, and trigger predictable downstream alarm patterns.

## Methodology

- XGBoost multi-class classifier with severity-aware encoding
- **Feature groups:**
  - `time_lag_seconds` -- elapsed time from incident start
  - `severity_encoded` -- ordinal encoding of alarm severity
  - Cascade depth and branching factor
  - Alarm type co-occurrence within the incident window
- **Custom evaluation metrics:**
  - Accuracy@1 -- is the top-ranked candidate the true root cause?
  - Accuracy@3 -- is the true root cause in the top 3?
  - Accuracy@5 -- is the true root cause in the top 5?
  - MRR -- mean reciprocal rank across all incidents
- Per-incident ranking via predicted probability scores

## Key Findings

- **Accuracy@3:** ~0.70 on held-out test set
- **Top features:** `time_lag_seconds` and `severity_encoded` dominate feature importance
- Events with lag=0 and CRITICAL severity are correctly identified as root causes in the majority of incidents
- Cascade depth provides meaningful lift beyond time lag alone

## Quick Start

```bash
# Clone the repository
git clone https://github.com/adityonugrohoid/telecom-ml-portfolio.git
cd telecom-ml-portfolio/02-root-cause-analysis

# Install dependencies
uv sync

# Generate synthetic data
uv run python generate_data.py

# Run the notebook
uv run jupyter lab notebooks/
```

## Project Structure

```
02-root-cause-analysis/
├── README.md
├── pyproject.toml
├── notebooks/
│   └── 01_root_cause_analysis.ipynb
├── src/
│   └── root_cause_analysis/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       └── evaluate.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── generate_data.py
└── tests/
    └── .gitkeep
```

## Related Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | [Churn Prediction](../01-churn-prediction) | Binary classification to predict customer churn |
| 2 | **Root Cause Analysis** (this repo) | Multi-class classification for network alarm RCA |
| 3 | [Anomaly Detection](../03-anomaly-detection) | Unsupervised detection of network anomalies |
| 4 | [QoE Prediction](../04-qoe-prediction) | Regression to predict quality of experience |
| 5 | [Capacity Forecasting](../05-capacity-forecasting) | Time-series forecasting for network capacity planning |
| 6 | [Network Optimization](../06-network-optimization) | Optimization of network resource allocation |

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Author

**Adityo Nugroho**
