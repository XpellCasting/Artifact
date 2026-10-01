"""Características causales y predictor por cuantiles para FOCuS.

Los estimadores viven solamente en memoria. ``metadata`` contiene la parte
serializable que puede incorporarse a manifiestos y snapshots.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

CONTINUOUS_METHOD = "residual_focus_v1"
FEATURE_SCHEMA_VERSION = "continuous-temporal-causal-v1"
HISTORY_SIZE = 12
QUANTILES = (0.05, 0.50, 0.95)
MIN_TRAINING_SAMPLES = 500
MODEL_PARAMETERS = {
    "loss": "quantile", "learning_rate": 0.05, "max_iter": 100,
    "max_leaf_nodes": 15, "min_samples_leaf": 20,
    "l2_regularization": 1.0, "early_stopping": False, "random_state": 42,
}

def causal_features(history: Iterable[tuple[Any, float]], at: Any) -> np.ndarray | None:
    """Return features made exclusively from the twelve earlier readings."""
    timestamp = pd.to_datetime(at, errors="coerce", utc=True)
    rows = list(history)[-HISTORY_SIZE:]
    if pd.isna(timestamp) or len(rows) < HISTORY_SIZE:
        return None
    values = np.asarray([float(value) for _, value in rows], dtype=float)
    times = [pd.to_datetime(item[0], errors="coerce", utc=True) for item in rows]
    if not np.isfinite(values).all() or any(pd.isna(item) or item >= timestamp for item in times):
        return None
    ages = np.asarray([max(0.0, (timestamp - item).total_seconds()) for item in times])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600 + timestamp.microsecond / 3.6e9
    angle = 2 * math.pi * hour / 24
    return np.concatenate([values, np.log1p(ages), [median, mad, values.min(), values.max(), math.sin(angle), math.cos(angle)]])

@dataclass(frozen=True)
class TemporalSensorModel:
    models: tuple[Any, Any, Any]
    interval_margin: float
    training_value_floor: float
    metadata: dict[str, Any]

    def predict_raw(self, features: np.ndarray) -> tuple[float, float, float]:
        predicted = sorted(float(model.predict(features.reshape(1, -1))[0]) for model in self.models)
        return predicted[0], predicted[1], predicted[2]

class FiltroContinuoTemporal:
    method = CONTINUOUS_METHOD

    def _samples(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
        work = frame.copy()
        work["__order"] = range(len(work))
        work["timestamp_dt"] = pd.to_datetime(work["timestamp_dt"], errors="coerce", utc=True)
        work["value"] = pd.to_numeric(work["value"], errors="coerce")
        if "is_valid_value" in work:
            work = work[work["is_valid_value"].astype(bool)]
        work = work[work["timestamp_dt"].notna() & work["value"].notna()]
        work = work[np.isfinite(work["value"].astype(float))].sort_values(["timestamp_dt", "__order"], kind="stable")
        history: deque[tuple[pd.Timestamp, float]] = deque(maxlen=HISTORY_SIZE)
        features, targets, timestamps = [], [], []
        for timestamp, simultaneous in work.groupby("timestamp_dt", sort=True):
            vector = causal_features(history, timestamp)
            if vector is not None:
                for row in simultaneous.to_dict("records"):
                    features.append(vector.copy()); targets.append(float(row["value"])); timestamps.append(timestamp)
            for row in simultaneous.to_dict("records"):
                history.append((timestamp, float(row["value"])))
        width = HISTORY_SIZE * 2 + 6
        return np.asarray(features, dtype=float).reshape((-1, width)), np.asarray(targets, dtype=float), timestamps

