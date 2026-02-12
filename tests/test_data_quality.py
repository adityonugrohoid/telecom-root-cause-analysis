"""Tests for data quality and validation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from root_cause_analysis.data_generator import RCADataGenerator


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    generator = RCADataGenerator(
        seed=42,
        n_samples=10000,
        n_incidents=50,
        events_per_incident=20,
        n_cells=10,
    )
    return generator.generate()


class TestDataQuality:
    def test_no_missing_values(self, sample_data):
        critical_cols = ["event_id", "incident_id", "alarm_severity", "is_root_cause"]
        for col in critical_cols:
            if col in sample_data.columns:
                assert sample_data[col].isna().sum() == 0, f"Missing values in {col}"

    def test_data_types(self, sample_data):
        assert pd.api.types.is_integer_dtype(sample_data["incident_id"])
        assert pd.api.types.is_numeric_dtype(sample_data["time_lag_seconds"])
        assert pd.api.types.is_numeric_dtype(sample_data["is_root_cause"])

    def test_value_ranges(self, sample_data):
        assert set(sample_data["is_root_cause"].unique()).issubset({0, 1})
        assert sample_data["time_lag_seconds"].min() >= 0
        assert sample_data["event_sequence_position"].min() >= 0

    def test_categorical_values(self, sample_data):
        assert set(sample_data["alarm_severity"].unique()).issubset(
            {"critical", "major", "minor", "warning"}
        )
        assert set(sample_data["event_type"].unique()).issubset(
            {"hardware_failure", "software_bug", "config_error", "overload", "external"}
        )

    def test_sample_size(self, sample_data):
        assert len(sample_data) > 500, (
            f"Expected more than 500 rows (n_incidents * events), got {len(sample_data)}"
        )

    def test_one_root_cause_per_incident(self, sample_data):
        root_causes_per_incident = sample_data.groupby("incident_id")["is_root_cause"].sum()
        for incident_id, count in root_causes_per_incident.items():
            assert count == 1, (
                f"Incident {incident_id} has {int(count)} root causes, expected exactly 1"
            )


class TestDataGenerator:
    def test_generator_reproducibility(self):
        gen1 = RCADataGenerator(
            seed=42, n_samples=1000, n_incidents=10, events_per_incident=5, n_cells=3
        )
        gen2 = RCADataGenerator(
            seed=42, n_samples=1000, n_incidents=10, events_per_incident=5, n_cells=3
        )
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_sinr_generation(self):
        gen = RCADataGenerator(seed=42, n_samples=100)
        sinr = gen.generate_sinr(1000)
        assert len(sinr) == 1000
        assert sinr.min() >= -5
        assert sinr.max() <= 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
