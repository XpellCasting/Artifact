"""Modelo causal por cuantiles para sensores continuos.

Los estimadores viven solamente en memoria. ``metadata`` contiene la parte
serializable que puede incorporarse a manifiestos y snapshots.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor


CONTINUOUS_METHOD = "temporal_quantile_v1"
FEATURE_SCHEMA_VERSION = "continuous-temporal-causal-v1"
HISTORY_SIZE = 12
QUANTILES = (0.05, 0.50, 0.95)
TARGET_TAIL_FRACTION = 0.01
MIN_TRAINING_SAMPLES = 500
MIN_CALIBRATION_SAMPLES = 200
MODEL_PARAMETERS = {
    "loss": "quantile", "learning_rate": 0.05, "max_iter": 100,
    "max_leaf_nodes": 15, "min_samples_leaf": 20,
    "l2_regularization": 1.0, "early_stopping": False, "random_state": 42,
}


def _iso(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed.isoformat()


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


def calibrated_margin(errors: Iterable[float], tail_fraction: float = TARGET_TAIL_FRACTION) -> float:
    ordered = np.sort(np.asarray(list(errors), dtype=float))
    n = len(ordered)
    k = math.ceil((n + 1) * (1 - float(tail_fraction)))
    if not n or k > n:
        raise ValueError("insufficient_calibration")
    return float(ordered[k - 1])


@dataclass(frozen=True)
class TemporalSensorModel:
    models: tuple[Any, Any, Any]
    interval_margin: float
    training_value_floor: float
    metadata: dict[str, Any]

    def predict_raw(self, features: np.ndarray) -> tuple[float, float, float]:
        predicted = sorted(float(model.predict(features.reshape(1, -1))[0]) for model in self.models)
        return predicted[0], predicted[1], predicted[2]

    def predict(self, features: np.ndarray, observed: float) -> dict[str, Any]:
        lower_raw, expected, upper_raw = self.predict_raw(features)
        lower, upper = lower_raw - self.interval_margin, upper_raw + self.interval_margin
        exceedance = max(lower - observed, observed - upper, 0.0)
        scale = max(upper - lower, self.training_value_floor)
        normalized = exceedance / scale
        score = 0.0 if exceedance == 0 else 0.6 + 0.4 * normalized / (1 + normalized)
        return {
            "expected_value": expected, "lower_bound": lower, "upper_bound": upper,
            "observed_value": observed, "interval_margin": self.interval_margin,
            "normalized_exceedance": normalized, "suspicion_score": score,
            "is_interval_detection": exceedance > 0,
        }


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

    def train_sensor(self, frame: pd.DataFrame, entity_id: str) -> TemporalSensorModel | dict[str, Any]:
        x, y, timestamps = self._samples(frame)
        if len(y) < MIN_TRAINING_SAMPLES + MIN_CALIBRATION_SAMPLES:
            return self._ineligible(entity_id, len(y), "insufficient_training_data")
        split = int(math.floor(len(y) * 0.8))
        boundary = timestamps[split]
        while split > 0 and timestamps[split - 1] == boundary:
            split -= 1
        if split < MIN_TRAINING_SAMPLES or len(y) - split < MIN_CALIBRATION_SAMPLES:
            return self._ineligible(entity_id, len(y), "insufficient_training_data")
        models = []
        for quantile in QUANTILES:
            model = HistGradientBoostingRegressor(**MODEL_PARAMETERS, quantile=quantile)
            model.fit(x[:split], y[:split])
            models.append(model)
        calibration_predictions = np.column_stack([model.predict(x[split:]) for model in models])
        calibration_predictions.sort(axis=1)
        errors = np.maximum.reduce([calibration_predictions[:, 0] - y[split:], y[split:] - calibration_predictions[:, 2], np.zeros(len(y) - split)])
        try:
            margin = calibrated_margin(errors)
        except ValueError:
            return self._ineligible(entity_id, len(y), "insufficient_calibration")
        train_values = y[:split]
        median = float(np.median(train_values)); mad = float(np.median(np.abs(train_values - median)))
        value_floor = max(float(np.finfo(float).eps), 1.4826 * mad * 1e-6, float(np.ptp(train_values)) * 1e-9)
        fingerprint = sha256(np.asarray(calibration_predictions, dtype="<f8").tobytes()).hexdigest()
        metadata = {
            "entity_id": str(entity_id), "eligible": True, "reason": None,
            "continuous_method": CONTINUOUS_METHOD, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_parameters": {**MODEL_PARAMETERS, "quantiles": list(QUANTILES)},
            "training_partition": {"samples": split, "start": _iso(timestamps[0]), "end": _iso(timestamps[split - 1])},
            "calibration_partition": {"samples": len(y) - split, "start": _iso(timestamps[split]), "end": _iso(timestamps[-1])},
            "target_tail_fraction": TARGET_TAIL_FRACTION, "interval_margin": margin,
            "calibration_prediction_fingerprint": fingerprint, "scikit_learn_version": sklearn_version,
            "training_value_floor": value_floor,
        }
        return TemporalSensorModel(tuple(models), margin, value_floor, metadata)

    @staticmethod
    def _ineligible(entity_id: str, samples: int, reason: str) -> dict[str, Any]:
        return {"entity_id": str(entity_id), "eligible": False, "reason": reason, "samples": samples,
                "continuous_method": CONTINUOUS_METHOD, "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "model_parameters": {**MODEL_PARAMETERS, "quantiles": list(QUANTILES)},
                "target_tail_fraction": TARGET_TAIL_FRACTION, "scikit_learn_version": sklearn_version}

    @staticmethod
    def metadata(model_or_status: Any) -> dict[str, Any]:
        return dict(model_or_status.metadata if isinstance(model_or_status, TemporalSensorModel) else model_or_status)

    @staticmethod
    def runtime(frame: pd.DataFrame) -> dict[str, Any]:
        work = frame.copy()
        work["timestamp_dt"] = pd.to_datetime(work["timestamp_dt"], errors="coerce", utc=True)
        work["value"] = pd.to_numeric(work["value"], errors="coerce")
        if "is_valid_value" in work:
            work = work[work["is_valid_value"].astype(bool)]
        work = work[work["timestamp_dt"].notna() & work["value"].notna()]
        work = work[np.isfinite(work["value"].astype(float))].sort_values("timestamp_dt", kind="stable")
        return {"history": deque(((row.timestamp_dt, float(row.value)) for row in work.tail(HISTORY_SIZE).itertuples()), maxlen=HISTORY_SIZE)}


def serializable_manifest(models: dict[str, Any]) -> dict[str, Any]:
    payload = {key: FiltroContinuoTemporal.metadata(value) for key, value in sorted(models.items())}
    json.dumps(payload, allow_nan=False)
    return payload
