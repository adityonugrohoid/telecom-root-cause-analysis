"""Tests for model training and evaluation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from root_cause_analysis.data_generator import RCADataGenerator
from root_cause_analysis.features import FeatureEngineer
from root_cause_analysis.models import BaseModel, XGBoostRCAClassifier

TARGET = "is_root_cause"
NON_FEATURES = ["incident_id", "event_id", "timestamp", "cell_id"]

SEED = 42
N_INCIDENTS = 150

# The README reports Accuracy@1 of 0.91 on the full 500-incident, seed-42
# split (measured here as 0.89, see the baseline PR notes). This floor is
# lower because the test trains on 150 incidents to stay fast (measured
# ~0.87 here); a drop below it means the signal in the generator or the
# feature pipeline has broken.
ACCURACY_AT_1_FLOOR = 0.75


def _accuracy_at_1(y_true: np.ndarray, y_score: np.ndarray, incident_ids: np.ndarray) -> float:
    """Fraction of incidents where the top-scored event is the true root cause.

    This is the ranking metric notebooks/02_root_cause_analysis.ipynb computes
    by hand (groupby incident_id, take the highest-scored event). It is not
    the same computation as XGBoostRCAClassifier.accuracy_at_k, which ranks
    the two classes within a single row's probability vector and cannot
    reproduce an incident-level ranking metric.
    """
    eval_df = pd.DataFrame({"incident_id": incident_ids, "y_true": y_true, "y_score": y_score})
    correct = 0
    total = 0
    for _, group in eval_df.groupby("incident_id"):
        top = group.nlargest(1, "y_score")
        if top["y_true"].sum() > 0:
            correct += 1
        total += 1
    return correct / total


@pytest.fixture(scope="module")
def features():
    """Seeded data run through the project's own feature pipeline."""
    raw = RCADataGenerator(seed=SEED, n_incidents=N_INCIDENTS).generate()
    return FeatureEngineer().pipeline(raw)


@pytest.fixture(scope="module")
def split(features):
    """Incident-level split, as the project's notebook performs it.

    BaseModel.prepare_data (inherited by XGBoostRCAClassifier) does a plain
    random train_test_split with no incident grouping, which would leak
    events from the same cascade across train and test. The notebook works
    around this with a manual GroupShuffleSplit on incident_id, so this
    fixture follows that path instead of the model's own prepare_data.
    """
    incident_ids = features["incident_id"]
    df_model = features.drop(columns=[c for c in NON_FEATURES if c in features.columns])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df_model, groups=incident_ids))

    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_incident_ids = incident_ids.iloc[test_idx].to_numpy()
    return X_train, X_test, y_train, y_test, test_incident_ids


@pytest.fixture(scope="module")
def trained(split):
    """The model as the project uses it: incident-level split, then train."""
    X_train, _, y_train, _, _ = split
    model = XGBoostRCAClassifier()
    model.feature_names = X_train.columns.tolist()
    model.train(X_train, y_train)
    return model


class TestTraining:
    def test_untrained_model_refuses_to_predict(self, split):
        _, X_test, _, _, _ = split
        with pytest.raises(ValueError):
            XGBoostRCAClassifier().predict(X_test)

    def test_base_model_has_no_training(self, split):
        X_train, _, y_train, _, _ = split
        with pytest.raises(NotImplementedError):
            BaseModel().train(X_train, y_train)

    def test_training_marks_the_model_trained(self, trained):
        assert trained.is_trained

    def test_training_is_reproducible(self, split, trained):
        X_train, X_test, y_train, _, _ = split
        again = XGBoostRCAClassifier()
        again.feature_names = X_train.columns.tolist()
        again.train(X_train, y_train)
        np.testing.assert_allclose(
            trained.predict_proba(X_test), again.predict_proba(X_test), rtol=0, atol=1e-7
        )


class TestEvaluation:
    def test_probabilities_are_valid(self, split, trained):
        _, X_test, _, _, _ = split
        proba = trained.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert ((proba >= 0) & (proba <= 1)).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_metrics_are_complete(self, split, trained):
        _, X_test, _, y_test, _ = split
        metrics = trained.evaluate(X_test, y_test, task_type="classification")
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert all(0 <= value <= 1 for value in metrics.values())

    def test_headline_accuracy_at_1_stays_above_the_floor(self, split, trained):
        _, X_test, _, y_test, test_incident_ids = split
        y_score = trained.predict_proba(X_test)[:, 1]
        acc_at_1 = _accuracy_at_1(y_test.to_numpy(), y_score, test_incident_ids)
        assert acc_at_1 >= ACCURACY_AT_1_FLOOR, f"Accuracy@1 fell to {acc_at_1:.3f}"

    def test_model_beats_the_first_alarm_heuristic(self, split, trained):
        _, X_test, _, y_test, test_incident_ids = split
        y_score = trained.predict_proba(X_test)[:, 1]
        model_acc_at_1 = _accuracy_at_1(y_test.to_numpy(), y_score, test_incident_ids)

        # Trivial predictor named in the README: always rank the first event
        # in the cascade highest.
        heuristic_score = -X_test["event_sequence_position"].to_numpy(dtype=float)
        heuristic_acc_at_1 = _accuracy_at_1(y_test.to_numpy(), heuristic_score, test_incident_ids)

        assert model_acc_at_1 > heuristic_acc_at_1

    def test_feature_importance_covers_every_feature(self, split, trained):
        X_train, _, _, _, _ = split
        importance = trained.get_feature_importance()
        assert sorted(importance["feature"]) == sorted(X_train.columns)
        assert importance["importance"].sum() == pytest.approx(1.0, abs=1e-5)


class TestPersistence:
    def test_saved_model_predicts_the_same(self, split, trained, tmp_path):
        _, X_test, _, _, _ = split
        path = tmp_path / "rca_model.pkl"
        trained.save(path)
        restored = XGBoostRCAClassifier()
        restored.load(path)
        np.testing.assert_array_equal(trained.predict(X_test), restored.predict(X_test))

    def test_untrained_model_refuses_to_save(self, tmp_path):
        with pytest.raises(ValueError):
            XGBoostRCAClassifier().save(tmp_path / "rca_model.pkl")
