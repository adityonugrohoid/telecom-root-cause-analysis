"""
Domain-informed synthetic data generator for Telecom Root Cause Analysis.

This module generates realistic telecom data using domain knowledge rather than
off-the-shelf synthetic data tools.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_GEN_CONFIG, RAW_DATA_DIR, ensure_directories


class TelecomDataGenerator:
    """Base class for generating synthetic telecom data."""

    def __init__(self, seed: int = 42, n_samples: int = 10_000):
        self.seed = seed
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_sinr(self, n: int, base_sinr_db: float = 10.0, noise_std: float = 5.0) -> np.ndarray:
        sinr = self.rng.normal(base_sinr_db, noise_std, n)
        return np.clip(sinr, -5, 25)

    def sinr_to_throughput(self, sinr_db: np.ndarray, network_type: np.ndarray, noise_factor: float = 0.2) -> np.ndarray:
        sinr_linear = 10 ** (sinr_db / 10)
        capacity_factor = np.log2(1 + sinr_linear)
        max_throughput = np.where(network_type == "5G", 300, 50)
        throughput = capacity_factor * max_throughput / 5
        noise = self.rng.normal(1, noise_factor, len(throughput))
        throughput = throughput * noise
        return np.clip(throughput, 0.1, max_throughput)

    def generate_congestion_pattern(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        peak_morning = (hour >= 9) & (hour <= 11)
        peak_evening = (hour >= 18) & (hour <= 21)
        congestion = np.where(peak_morning | peak_evening, congestion * 1.3, congestion)
        is_weekend = day_of_week >= 5
        congestion = np.where(is_weekend, congestion * 0.8, congestion)
        noise = self.rng.normal(0, 0.1, len(congestion))
        congestion = congestion + noise
        return np.clip(congestion, 0, 1)

    def congestion_to_latency(self, congestion: np.ndarray, base_latency_ms: float = 20) -> np.ndarray:
        latency = base_latency_ms * (1 + 5 * congestion ** 2)
        jitter = self.rng.normal(0, 5, len(latency))
        latency = latency + jitter
        return np.clip(latency, 10, 300)

    def compute_qoe_mos(self, throughput_mbps: np.ndarray, latency_ms: np.ndarray, packet_loss_pct: np.ndarray, app_type: np.ndarray) -> np.ndarray:
        mos_throughput = 1 + 4 * (1 - np.exp(-throughput_mbps / 10))
        latency_penalty = np.clip(latency_ms / 100, 0, 2)
        loss_penalty = packet_loss_pct / 2
        mos = mos_throughput - latency_penalty - loss_penalty
        video_mask = app_type == "video_streaming"
        mos = np.where(video_mask, mos - packet_loss_pct * 0.5, mos)
        gaming_mask = app_type == "gaming"
        mos = np.where(gaming_mask, mos - latency_penalty * 0.5, mos)
        return np.clip(mos, 1, 5)

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        ensure_directories()
        output_path = RAW_DATA_DIR / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df):,} rows to {output_path}")
        return output_path


class RCADataGenerator(TelecomDataGenerator):
    """Generate synthetic incident-event data for telecom root cause analysis.

    Each incident consists of a cascade of correlated alarm events originating
    from a single root cause.  The root cause event has ``time_lag_seconds=0``
    and ``is_root_cause=1``; subsequent events receive increasing time lags,
    lower severity, and smaller KPI deltas.
    """

    # Mapping from event type to a "typical" severity for the root cause event
    _ROOT_CAUSE_SEVERITY = {
        "hardware_failure": "critical",
        "software_bug": "critical",
        "config_error": "major",
        "overload": "critical",
        "external": "major",
    }

    # Severity ordering (index 0 = highest)
    _SEVERITY_ORDER = ["critical", "major", "minor", "warning"]

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = 10_000,
        n_incidents: int = 500,
        events_per_incident: int = 20,
        n_cells: int = 50,
        event_types: Optional[list] = None,
        alarm_severity_levels: Optional[list] = None,
    ):
        super().__init__(seed=seed, n_samples=n_samples)
        self.n_incidents = n_incidents
        self.events_per_incident = events_per_incident
        self.n_cells = n_cells
        self.event_types = event_types or [
            "hardware_failure",
            "software_bug",
            "config_error",
            "overload",
            "external",
        ]
        self.alarm_severity_levels = alarm_severity_levels or self._SEVERITY_ORDER

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cascade_severity(self, root_severity: str, position: int) -> str:
        """Return a degraded severity for events further from the root cause."""
        idx = self._SEVERITY_ORDER.index(root_severity)
        # Each step has a chance to drop one severity level
        drop = min(position // 4, len(self._SEVERITY_ORDER) - 1 - idx)
        # Add some randomness -- occasionally stays the same or drops more
        jitter = self.rng.choice([-1, 0, 0, 1], p=[0.1, 0.4, 0.3, 0.2])
        new_idx = int(np.clip(idx + drop + jitter, 0, len(self._SEVERITY_ORDER) - 1))
        return self._SEVERITY_ORDER[new_idx]

    def _generate_incident(
        self, incident_id: int, base_timestamp: pd.Timestamp
    ) -> list:
        """Generate all events for a single incident.

        Returns a list of dicts, one per event.
        """
        n_events = self.events_per_incident
        root_cause_type = self.rng.choice(self.event_types)
        root_severity = self._ROOT_CAUSE_SEVERITY.get(root_cause_type, "major")
        primary_cell_id = self.rng.integers(1, self.n_cells + 1)

        # KPI deltas for the root cause (large impact)
        root_sinr_delta = float(self.rng.uniform(-15, -5))
        root_throughput_delta = float(self.rng.uniform(-80, -30))
        root_latency_delta = float(self.rng.uniform(30, 150))

        events = []

        for pos in range(n_events):
            if pos == 0:
                # --- Root cause event ---
                event = {
                    "incident_id": incident_id,
                    "event_sequence_position": 0,
                    "timestamp": base_timestamp,
                    "cell_id": primary_cell_id,
                    "event_type": root_cause_type,
                    "alarm_severity": root_severity,
                    "time_lag_seconds": 0,
                    "is_root_cause": 1,
                    "affected_cells": 1,
                    "sinr_delta": round(root_sinr_delta, 2),
                    "throughput_delta": round(root_throughput_delta, 2),
                    "latency_delta": round(root_latency_delta, 2),
                }
            else:
                # --- Cascading / secondary event ---
                # Time lag increases with position (5-300 seconds range)
                lag = int(
                    np.clip(
                        self.rng.exponential(scale=30) + pos * 5,
                        5,
                        300,
                    )
                )
                event_ts = base_timestamp + pd.Timedelta(seconds=lag)

                # Cascading events may be of a different type
                cascade_type = self.rng.choice(self.event_types)

                severity = self._cascade_severity(root_severity, pos)

                # KPI deltas decay with distance from root cause
                decay = np.exp(-0.15 * pos)
                sinr_delta = round(
                    root_sinr_delta * decay
                    + float(self.rng.normal(0, 1)),
                    2,
                )
                throughput_delta = round(
                    root_throughput_delta * decay
                    + float(self.rng.normal(0, 3)),
                    2,
                )
                latency_delta = round(
                    root_latency_delta * decay
                    + float(self.rng.normal(0, 5)),
                    2,
                )

                # Affected cells spread as the cascade propagates
                affected_cells = int(
                    np.clip(1 + self.rng.poisson(pos * 0.5), 1, self.n_cells)
                )

                # Cell may differ from the primary cell for later events
                if self.rng.random() < 0.3:
                    cell_id = self.rng.integers(1, self.n_cells + 1)
                else:
                    cell_id = primary_cell_id

                event = {
                    "incident_id": incident_id,
                    "event_sequence_position": pos,
                    "timestamp": event_ts,
                    "cell_id": cell_id,
                    "event_type": cascade_type,
                    "alarm_severity": severity,
                    "time_lag_seconds": lag,
                    "is_root_cause": 0,
                    "affected_cells": affected_cells,
                    "sinr_delta": sinr_delta,
                    "throughput_delta": throughput_delta,
                    "latency_delta": latency_delta,
                }

            events.append(event)

        return events

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate the full incident-event dataset.

        Returns:
            pd.DataFrame with one row per event.  Total rows are approximately
            ``n_incidents * events_per_incident``.
        """
        all_events: list = []

        # Spread incident base timestamps over the last 90 days
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.Timedelta(days=90)

        for inc_idx in range(self.n_incidents):
            incident_id = inc_idx + 1

            # Random base timestamp for the incident
            random_offset_s = int(self.rng.integers(0, 90 * 86400))
            base_ts = start_date + pd.Timedelta(seconds=random_offset_s)

            events = self._generate_incident(incident_id, base_ts)
            all_events.extend(events)

        df = pd.DataFrame(all_events)

        # Add a globally unique event_id
        df.insert(0, "event_id", range(1, len(df) + 1))

        # Sort by incident then sequence position for readability
        df = df.sort_values(
            ["incident_id", "event_sequence_position"]
        ).reset_index(drop=True)

        print(f"Generated {len(df):,} events across {self.n_incidents:,} incidents")
        print(
            f"Root cause events: {df['is_root_cause'].sum():,} "
            f"({df['is_root_cause'].mean():.2%})"
        )
        print(f"Event types: {df['event_type'].value_counts().to_dict()}")
        return df


def main() -> None:
    """Generate and save the root cause analysis dataset using project config."""
    config = DATA_GEN_CONFIG
    use_case = config.get("use_case_params", {})

    generator = RCADataGenerator(
        seed=config.get("random_seed", 42),
        n_samples=config.get("n_samples", 10_000),
        n_incidents=use_case.get("n_incidents", 500),
        events_per_incident=use_case.get("events_per_incident", 20),
        n_cells=use_case.get("n_cells", 50),
        event_types=use_case.get("event_types", None),
        alarm_severity_levels=use_case.get("alarm_severity_levels", None),
    )

    df = generator.generate()
    generator.save(df, "rca_data")


if __name__ == "__main__":
    main()
