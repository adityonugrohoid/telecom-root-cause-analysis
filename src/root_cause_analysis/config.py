"""
Configuration management for Telecom Root Cause Analysis.

This module centralizes all configuration parameters, making it easy to
adjust settings without modifying core logic.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ============================================================================
# DATA GENERATION CONFIG
# ============================================================================

DATA_GEN_CONFIG = {
    "random_seed": 42,
    "n_samples": 10_000,
    "test_size": 0.2,
    "validation_size": 0.1,
    "use_case_params": {
        "n_incidents": 500,
        "events_per_incident": 20,
        "n_cells": 50,
        "alarm_severity_levels": ["critical", "major", "minor", "warning"],
        "event_types": ["hardware_failure", "software_bug", "config_error", "overload", "external"],
    },
}


# ============================================================================
# FEATURE ENGINEERING CONFIG
# ============================================================================

FEATURE_CONFIG = {
    "categorical_features": [
        "alarm_severity",
        "event_type",
        "network_element_type",
    ],
    "numerical_features": [
        "time_lag_seconds",
        "alarm_count",
        "affected_cells",
        "sinr_delta",
        "throughput_delta",
        "latency_delta",
        "event_sequence_position",
    ],
    "datetime_features": ["timestamp"],
    "rolling_windows": [7, 30],
    "create_features": True,
}


# ============================================================================
# MODEL TRAINING CONFIG
# ============================================================================

MODEL_CONFIG = {
    "algorithm": "xgboost",
    "cv_folds": 5,
    "cv_strategy": "stratified",
    "hyperparameters": {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
    },
    "early_stopping_rounds": 10,
    "verbose": True,
}


# ============================================================================
# EVALUATION CONFIG
# ============================================================================

EVAL_CONFIG = {
    "primary_metric": "accuracy_at_3",
    "threshold": 0.5,
    "compute_metrics": [
        "accuracy",
        "accuracy_at_3",
        "accuracy_at_5",
        "mean_reciprocal_rank",
        "f1",
    ],
    "top_k_values": [1, 3, 5],
}


# ============================================================================
# VISUALIZATION CONFIG
# ============================================================================

VIZ_CONFIG = {
    "style": "whitegrid",
    "palette": "husl",
    "context": "notebook",
    "figure_size": (12, 6),
    "dpi": 100,
}


# ============================================================================
# UTILITIES
# ============================================================================


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_custom_config(config_path: Path) -> Dict[str, Any]:
    """Load custom configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "data_gen": DATA_GEN_CONFIG,
        "features": FEATURE_CONFIG,
        "model": MODEL_CONFIG,
        "eval": EVAL_CONFIG,
        "viz": VIZ_CONFIG,
        "paths": {
            "root": PROJECT_ROOT,
            "data": DATA_DIR,
            "raw": RAW_DATA_DIR,
            "processed": PROCESSED_DATA_DIR,
            "notebooks": NOTEBOOKS_DIR,
        },
    }


if __name__ == "__main__":
    ensure_directories()
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Random seed: {DATA_GEN_CONFIG['random_seed']}")
    print(f"Algorithm: {MODEL_CONFIG['algorithm']}")
