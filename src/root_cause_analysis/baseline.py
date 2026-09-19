"""Baseline comparison for Telecom Root Cause Analysis.

The README names "Always-predict-first-alarm" as the baseline in its
Methodology table, but no code in this repository ever scored it against the
project's model. This module runs both on one shared, incident-level split
and reports the metrics side by side.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .config import DATA_GEN_CONFIG, PROJECT_ROOT
from .data_generator import RCADataGenerator
from .features import FeatureEngineer
from .models import XGBoostRCAClassifier

TARGET = "is_root_cause"
NON_FEATURE_COLUMNS = ["incident_id", "event_id", "timestamp", "cell_id"]

# The README's Methodology table reports Accuracy@1 as the primary metric.
# config.py's EVAL_CONFIG separately lists "accuracy_at_3" as the primary
# metric; that mismatch is a config/README inconsistency this module does not
# resolve. This module follows the README, since it is the number the
# baseline is supposed to be checked against.
PRIMARY_METRIC = "accuracy_at_1"


def accuracy_at_k(
    y_true: np.ndarray, y_score: np.ndarray, incident_ids: np.ndarray, k: int
) -> float:
    """Fraction of incidents where the true root cause is among the top-k scored events.

    XGBoostRCAClassifier.accuracy_at_k ranks the classes within a single
    row's probability vector (there are only 2 classes here, is_root_cause
    0 or 1), which is a different computation and cannot reproduce this
    incident-level ranking metric. This function reimplements the version
    notebooks/02_root_cause_analysis.ipynb computes by hand.
    """
    eval_df = pd.DataFrame({"incident_id": incident_ids, "y_true": y_true, "y_score": y_score})
    correct = 0
    total = 0
    for _, group in eval_df.groupby("incident_id"):
        top_k = group.nlargest(k, "y_score")
        if top_k["y_true"].sum() > 0:
            correct += 1
        total += 1
    return correct / total


def mean_reciprocal_rank(
    y_true: np.ndarray, y_score: np.ndarray, incident_ids: np.ndarray
) -> float:
    """Mean reciprocal rank of the true root cause within each incident.

    See accuracy_at_k for why this is reimplemented here rather than reused
    from XGBoostRCAClassifier.
    """
    eval_df = pd.DataFrame({"incident_id": incident_ids, "y_true": y_true, "y_score": y_score})
    reciprocal_rank_sum = 0.0
    total = 0
    for _, group in eval_df.groupby("incident_id"):
        ranked = group.sort_values("y_score", ascending=False).reset_index(drop=True)
        root_cause_positions = ranked[ranked["y_true"] == 1].index
        if len(root_cause_positions) > 0:
            reciprocal_rank_sum += 1.0 / (root_cause_positions[0] + 1)
        total += 1
    return reciprocal_rank_sum / total


def _first_alarm_baseline_predictions(X_test: pd.DataFrame) -> tuple:
    """Score and predicted label for the README's "always predict first alarm" heuristic.

    The heuristic ranks every event in an incident by closeness to the start
    of the cascade: the event at event_sequence_position 0 gets the highest
    score, and only that event is flagged as the root cause candidate.
    """
    positions = X_test["event_sequence_position"].to_numpy(dtype=float)
    score = -positions
    predicted_label = (positions == 0).astype(int)
    return score, predicted_label


def _classification_and_ranking_metrics(
    y_true: np.ndarray, predicted_label: np.ndarray, score: np.ndarray, incident_ids: np.ndarray
) -> dict:
    """The standard classification metrics plus the ranking metrics the README reports."""
    return {
        "accuracy": float(accuracy_score(y_true, predicted_label)),
        "precision": float(precision_score(y_true, predicted_label, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_label, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "accuracy_at_1": accuracy_at_k(y_true, score, incident_ids, 1),
        "accuracy_at_3": accuracy_at_k(y_true, score, incident_ids, 3),
        "mean_reciprocal_rank": mean_reciprocal_rank(y_true, score, incident_ids),
    }


def run_baseline_comparison(seed: int, n_incidents: int) -> dict:
    """Train the project's model and the README's baseline on one shared split.

    Generates data with RCADataGenerator, runs it through the project's
    feature pipeline, and makes a single incident-level GroupShuffleSplit,
    matching notebooks/02_root_cause_analysis.ipynb section 5.
    BaseModel.prepare_data (inherited by XGBoostRCAClassifier) instead does
    a plain random split with no incident grouping, which would leak events
    from the same cascade across train and test, so it is not used here.
    """
    generator = RCADataGenerator(seed=seed, n_incidents=n_incidents)
    raw = generator.generate()
    features = FeatureEngineer().pipeline(raw)

    incident_ids = features["incident_id"]
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in features.columns]
    df_model = features.drop(columns=drop_cols)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(df_model, groups=incident_ids))

    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_incident_ids = incident_ids.iloc[test_idx].to_numpy()
    y_test_array = y_test.to_numpy()

    model = XGBoostRCAClassifier()
    model.feature_names = X_train.columns.tolist()
    model.train(X_train, y_train)
    model_score = model.predict_proba(X_test)[:, 1]
    model_predicted_label = model.predict(X_test)
    model_metrics = _classification_and_ranking_metrics(
        y_test_array, model_predicted_label, model_score, test_incident_ids
    )

    baseline_score, baseline_predicted_label = _first_alarm_baseline_predictions(X_test)
    baseline_metrics = _classification_and_ranking_metrics(
        y_test_array, baseline_predicted_label, baseline_score, test_incident_ids
    )

    return {
        "seed": seed,
        "n_incidents": n_incidents,
        "n_train_incidents": int(incident_ids.iloc[train_idx].nunique()),
        "n_test_incidents": int(incident_ids.iloc[test_idx].nunique()),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "primary_metric": PRIMARY_METRIC,
        "model": model_metrics,
        "baseline": baseline_metrics,
    }


def main() -> None:
    """Run the baseline comparison at the README's reported scale and save it.

    Writes evidence/baseline_metrics.json. That directory is not covered by
    the "results/" rule in .gitignore, so the file is tracked by git.
    """
    seed = DATA_GEN_CONFIG["random_seed"]
    n_incidents = DATA_GEN_CONFIG["use_case_params"]["n_incidents"]
    result = run_baseline_comparison(seed=seed, n_incidents=n_incidents)

    output_dir = PROJECT_ROOT / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_metrics.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved baseline comparison to {output_path}")


if __name__ == "__main__":
    main()
