"""Características causales y predictor por cuantiles para FOCuS.

Los estimadores viven solamente en memoria. ``metadata`` contiene la parte
serializable que puede incorporarse a manifiestos y snapshots.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import timezone
from itertools import groupby
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from app.modulos.estadistico.cobertura import reporting_policy

CONTINUOUS_METHOD = "residual_focus_v1"
FEATURE_SCHEMA_VERSION = "continuous-temporal-causal-v2-duration"
LEGACY_FEATURE_SCHEMA_VERSION = "continuous-temporal-causal-v1"
HISTORY_SIZE = 12
FEATURE_WIDTH = HISTORY_SIZE * 2 + 6 + 16
QUANTILES = (0.05, 0.50, 0.95)
PREDICTION_MODE = "recent_median_increment_v1"
MIN_TRAINING_SAMPLES = 500
MODEL_PARAMETERS = {
    "loss": "quantile", "learning_rate": 0.05, "max_iter": 100,
    "max_leaf_nodes": 15, "min_samples_leaf": 20,
    "l2_regularization": 1.0, "early_stopping": False, "random_state": 42,
}

def utc_timestamp(value: Any) -> pd.Timestamp | None:
    """Reuse normalized UTC timestamps; keep parsing for other input formats."""
    if isinstance(value, pd.Timestamp) and value.tzinfo is timezone.utc:
        return value
    return pd.to_datetime(value, errors="coerce", utc=True)


def causal_features(history: Iterable[tuple[Any, float]], at: Any, *, feature_schema_version=FEATURE_SCHEMA_VERSION) -> np.ndarray | None:
    """Return features made exclusively from the twelve earlier readings."""
    timestamp = utc_timestamp(at)
    rows = list(history)[-HISTORY_SIZE:]
    if pd.isna(timestamp) or len(rows) < HISTORY_SIZE:
        return None
    values = np.asarray([float(value) for _, value in rows], dtype=float)
    times = [utc_timestamp(item[0]) for item in rows]
    if not np.isfinite(values).all() or any(pd.isna(item) or item >= timestamp for item in times):
        return None
    ages = np.asarray([max(0.0, (timestamp - item).total_seconds()) for item in times])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600 + timestamp.microsecond / 3.6e9
    angle = 2 * math.pi * hour / 24
    base = np.concatenate([values, np.log1p(ages), [median, mad, values.min(), values.max(), math.sin(angle), math.cos(angle)]])
    if feature_schema_version == LEGACY_FEATURE_SCHEMA_VERSION:
        return base
    if feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("unknown_temporal_feature_schema")
    # State durations and backward time windows supplement report count.
    # Only the observed history is used; coverage shows when twelve reports
    # cannot describe the whole window. No synthetic reports enter FOCuS.
    durations = np.maximum(0., ages - np.r_[ages[1:], 0.])
    constant_age = ages[-1]
    for index in range(len(values) - 2, -1, -1):
        if values[index] != values[-1]:
            break
        constant_age = ages[index]
    temporal = [math.log1p(constant_age)]
    for seconds in (3600., 21600., 86400.):
        weights = np.maximum(0., np.minimum(ages, seconds) - np.minimum(np.r_[ages[1:], 0.], seconds))
        covered = float(weights.sum())
        mean = float(np.dot(values, weights) / covered) if covered else float(values[-1])
        variance = float(np.dot((values - mean) ** 2, weights) / covered) if covered else 0.
        temporal.extend([mean, math.sqrt(variance), min(1., covered / seconds),
                         math.log1p(min(constant_age, seconds)), math.log1p(float(durations.max()))])
    return np.concatenate([base, temporal])


def increment_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Anchor value features to the median of past readings; keep time and spread.

    The target is also an increment. A constant shift of the entire past then
    shifts the forecast by the same amount, including beyond the training range.
    Accept both feature schemas used by the comparison script.
    """
    vectors = np.asarray(features, dtype=float)
    anchor = vectors[..., 2 * HISTORY_SIZE].copy()
    relative = vectors.copy()
    value_indices = [*range(HISTORY_SIZE), 2 * HISTORY_SIZE,
                     2 * HISTORY_SIZE + 2, 2 * HISTORY_SIZE + 3]
    if vectors.shape[-1] == FEATURE_WIDTH:
        value_indices.extend(2 * HISTORY_SIZE + 7 + 5 * i for i in range(3))
    relative[..., value_indices] -= np.expand_dims(anchor, -1)
    return relative, anchor


@dataclass(frozen=True)
class TemporalSensorModel:
    models: tuple[Any, Any, Any]
    interval_margin: float
    training_value_floor: float
    metadata: dict[str, Any]

    def _inputs(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.metadata.get("prediction_mode") == PREDICTION_MODE:
            return increment_features(features)
        return features, np.zeros(np.asarray(features).shape[:-1])

    def predict_medians(self, features: np.ndarray) -> np.ndarray:
        inputs, anchor = self._inputs(features)
        return self.models[1].predict(inputs) + anchor

    def predict_median(self, features: np.ndarray) -> float:
        return float(self.predict_medians(features.reshape(1, -1))[0])

    def predict_raw(self, features: np.ndarray) -> tuple[float, float, float]:
        # Preserve quantile identities even when predictions cross: residuals
        # must always use the q=.50 estimator at position 1.
        inputs, anchor = self._inputs(features.reshape(1, -1))
        predicted = [float(model.predict(inputs)[0] + anchor[0]) for model in self.models]
        return predicted[0], predicted[1], predicted[2]

class FiltroContinuoTemporal:
    method = CONTINUOUS_METHOD

    def __init__(self, feature_schema_version=FEATURE_SCHEMA_VERSION):
        if feature_schema_version not in {FEATURE_SCHEMA_VERSION, LEGACY_FEATURE_SCHEMA_VERSION}:
            raise ValueError("unknown_temporal_feature_schema")
        self.feature_schema_version = feature_schema_version

    def _samples(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
        work = frame.copy()
        work["__order"] = range(len(work))
        work["timestamp_dt"] = pd.to_datetime(work["timestamp_dt"], errors="coerce", utc=True)
        work["value"] = pd.to_numeric(work["value"], errors="coerce")
        work = work[work["timestamp_dt"].notna()].sort_values(["timestamp_dt", "__order"], kind="stable")
        history: deque[tuple[pd.Timestamp, float]] = deque(maxlen=HISTORY_SIZE)
        policy = reporting_policy(frame)
        features, targets, timestamps = [], [], []
        numbers = work["value"].to_numpy(dtype=float, na_value=np.nan)
        valid_values = np.isfinite(numbers)
        if "is_valid_value" in work:
            valid_values &= work["is_valid_value"].fillna(False).to_numpy(dtype=bool)
        observations = zip(work["timestamp_dt"], numbers, valid_values)
        for timestamp, simultaneous in groupby(observations, key=lambda row: row[0]):
            rows = list(simultaneous)
            if history and policy["mode"] == "periodic":
                limit = policy["max_report_gap_seconds"]
                if limit is None or (timestamp - history[-1][0]).total_seconds() > limit:
                    history.clear()
            if not all(row[2] for row in rows):
                history.clear()
                continue
            vector = causal_features(history, timestamp, feature_schema_version=self.feature_schema_version)
            if vector is not None:
                for _, value, _ in rows:
                    features.append(vector.copy()); targets.append(float(value)); timestamps.append(timestamp)
            for _, value, _ in rows:
                history.append((timestamp, float(value)))
        width = HISTORY_SIZE * 2 + 6 if self.feature_schema_version == LEGACY_FEATURE_SCHEMA_VERSION else FEATURE_WIDTH
        return np.asarray(features, dtype=float).reshape((-1, width)), np.asarray(targets, dtype=float), timestamps
