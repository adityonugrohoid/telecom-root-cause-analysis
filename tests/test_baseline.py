"""Tests for the README baseline comparison."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from root_cause_analysis.baseline import PRIMARY_METRIC, run_baseline_comparison

SEED = 42
N_INCIDENTS = 150

EXPECTED_METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "accuracy_at_1",
    "accuracy_at_3",
    "mean_reciprocal_rank",
}


@pytest.fixture(scope="module")
def comparison():
    return run_baseline_comparison(seed=SEED, n_incidents=N_INCIDENTS)


class TestBaselineComparison:
    def test_returns_both_blocks_with_expected_keys(self, comparison):
        assert set(comparison["model"]) == EXPECTED_METRIC_KEYS
        assert set(comparison["baseline"]) == EXPECTED_METRIC_KEYS

    def test_reports_seed_and_split_sizes(self, comparison):
        assert comparison["seed"] == SEED
        assert comparison["n_incidents"] == N_INCIDENTS
        assert comparison["n_train_incidents"] + comparison["n_test_incidents"] == N_INCIDENTS
        assert comparison["n_train_rows"] > 0
        assert comparison["n_test_rows"] > 0

    def test_model_at_least_as_good_as_baseline_on_primary_metric(self, comparison):
        assert comparison["model"][PRIMARY_METRIC] >= comparison["baseline"][PRIMARY_METRIC]
