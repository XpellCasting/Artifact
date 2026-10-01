"""Detectores secuenciales sobre residuos del predictor temporal.

El predictor de cuantiles no se vuelve a ajustar aquí: se usa únicamente su
estimador central (q=.50).  Los objetos de este módulo contienen estado de una
ejecución y nunca se comparten entre nodos.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from app.modulos.estadistico.continuos_temporales import FEATURE_SCHEMA_VERSION
from app.modulos.estadistico.continuos_temporales import HISTORY_SIZE
from app.modulos.estadistico.continuos_temporales import MIN_TRAINING_SAMPLES
from app.modulos.estadistico.continuos_temporales import MODEL_PARAMETERS
from app.modulos.estadistico.continuos_temporales import QUANTILES
from app.modulos.estadistico.continuos_temporales import TemporalSensorModel

RESIDUAL_FOCUS_METHOD = "residual_focus_v1"
RESIDUAL_METHODS = {RESIDUAL_FOCUS_METHOD}
IMPLEMENTATION_VERSION = "residual-sequential-v1"
MIN_PARTITION_RESIDUALS = 200
MIN_THRESHOLD_DAYS = 7.0
DEFAULT_HALF_LIFE_SECONDS = 30 * 60.0
DEFAULT_EPISODE_GAP_SECONDS = 30 * 60.0


def _median_cadence_seconds(frame: pd.DataFrame) -> float:
    values = pd.to_datetime(frame["timestamp_dt"], errors="coerce", utc=True).sort_values().dropna()
    diffs = values.diff().dt.total_seconds().dropna()
    return float(diffs[diffs.gt(0)].median()) if not diffs.empty else 60.0


def _scale(values: list[float], floor: float) -> tuple[float, float, str]:
    centre = float(np.median(values))
    mad = float(np.median(np.abs(np.asarray(values) - centre)))
    if mad > 0:
        return centre, 1.4826 * mad, "mad"
    std = float(np.std(values))
    if std > 0:
        return centre, std, "standard_deviation"
    return centre, max(float(floor), 1e-12), "predictor_floor"


def _focus():
    try:
        from changepoint_online import Focus, Gaussian
    except Exception as exc:  # deliberately no EWMA fallback
        raise RuntimeError("focus_dependency_unavailable: changepoint-online==1.2.1") from exc
    return Focus(Gaussian(loc=0))


@dataclass
class ResidualRuntime:
    history: deque
    energy: float = 0.0
    last_timestamp: Any = None
    focus: Any = None
    focus_index: int = 0

    def __getstate__(self):
        """Persist runtime state without the third-party Focus instance.

        changepoint-online's Gaussian owns a local lambda and cannot be
        pickled.  A prepared historical baseline always starts a fresh
        sequential detector, so the instance can be recreated lazily on its
        first update after loading the cache.
        """
        state = self.__dict__.copy()
        state["focus"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.focus = None

    def reset_sequential(self, method: str) -> None:
        self.energy = 0.0
        self.focus_index = 0
        self.focus = _focus()


def new_runtime(history: deque, method: str) -> ResidualRuntime:
    runtime = ResidualRuntime(deque(history, maxlen=HISTORY_SIZE))
    runtime.reset_sequential(method)
    return runtime


def prepare_sensor(temporal, frame: pd.DataFrame, entity_id: str, method: str, *, half_life_seconds: float = DEFAULT_HALF_LIFE_SECONDS) -> dict[str, Any]:
    """Fit the existing q=.50 predictor and calibrate residual channels.

    The first 80% trains the immutable predictor.  The remaining 20% is split
    chronologically for robust normalization and threshold selection.
    """
    if method not in RESIDUAL_METHODS:
        raise ValueError("unknown_residual_method")
    if method == RESIDUAL_FOCUS_METHOD:
        _focus()  # fail clearly before an evaluation is started
    x, y, timestamps = temporal._samples(frame)
    if len(y) < MIN_TRAINING_SAMPLES + 2 * MIN_PARTITION_RESIDUALS:
        return {"entity_id": str(entity_id), "eligible": False, "reason": "insufficient_training_data", "continuous_method": method, "samples": int(len(y))}
    split = int(math.floor(len(y) * .8))
    while split > 0 and split < len(timestamps) and timestamps[split - 1] == timestamps[split]:
        split -= 1
    remaining = len(y) - split
    midpoint = split + remaining // 2
    while midpoint > split and midpoint < len(timestamps) and timestamps[midpoint - 1] == timestamps[midpoint]:
        midpoint -= 1
    if split < MIN_TRAINING_SAMPLES or midpoint - split < MIN_PARTITION_RESIDUALS or len(y) - midpoint < MIN_PARTITION_RESIDUALS:
        return {"entity_id": str(entity_id), "eligible": False, "reason": "insufficient_residual_partition", "continuous_method": method, "samples": int(len(y))}
    models = []
    for quantile in QUANTILES:
        model = HistGradientBoostingRegressor(**MODEL_PARAMETERS, quantile=quantile)
        model.fit(x[:split], y[:split]); models.append(model)
    train_values = y[:split]
    value_floor = max(float(np.finfo(float).eps), float(np.ptp(train_values)) * 1e-9, 1e-12)
    predictor = TemporalSensorModel(tuple(models), 0.0, value_floor, {"predictor": "temporal_quantile_q50", "feature_schema_version": FEATURE_SCHEMA_VERSION})
    residuals = [float(y[index] - predictor.predict_raw(x[index])[1]) for index in range(split, len(y))]
    normal_residuals = residuals[: midpoint - split]
    threshold_residuals = residuals[midpoint - split :]
    centre, scale, scale_source = _scale(normal_residuals, value_floor)
    normal_z = [(item - centre) / scale for item in normal_residuals]
    channels = _channel_sequence(normal_z, timestamps[split:midpoint], method, half_life_seconds)
    quantiles = {key: max(float(np.quantile([row[key] for row in channels], .99)), 1e-6) for key in ("P", "D", "N")}
    threshold_z = [(item - centre) / scale for item in threshold_residuals]
    threshold_channels = _channel_sequence(threshold_z, timestamps[midpoint:], method, half_life_seconds)
    threshold_scores = [max(row["P"] / quantiles["P"], row["D"] / quantiles["D"], row["N"] / quantiles["N"]) for row in threshold_channels]
    days = _effective_days(timestamps[midpoint:])
    if days < MIN_THRESHOLD_DAYS:
        return {"entity_id": str(entity_id), "eligible": False, "reason": "insufficient_threshold_days", "continuous_method": method, "threshold_days": days}
    threshold, candidates = _choose_threshold(threshold_scores, timestamps[midpoint:], days)
    if threshold is None:
        return {"entity_id": str(entity_id), "eligible": False, "reason": "calibration_budget_unmet", "continuous_method": method, "calibration_candidates": candidates}
    return {
        "entity_id": str(entity_id), "eligible": True, "continuous_method": method, "implementation_version": IMPLEMENTATION_VERSION,
        "predictor": predictor, "centre": centre, "scale": scale, "scale_source": scale_source, "channel_q99": quantiles,
        "threshold": threshold, "half_life_seconds": half_life_seconds, "episode_gap_seconds": DEFAULT_EPISODE_GAP_SECONDS,
        "median_cadence_seconds": _median_cadence_seconds(frame), "calibration_candidates": candidates,
        "partitions": {"predictor_fit": split, "normalization": midpoint - split, "threshold": len(y) - midpoint, "threshold_days": days},
        "runtime": new_runtime(deque(((row.timestamp_dt, float(row.value)) for row in frame.tail(HISTORY_SIZE).itertuples()), maxlen=HISTORY_SIZE), method),
    }


def _channel_sequence(zs: list[float], timestamps: list[Any], method: str, half_life: float) -> list[dict[str, float]]:
    runtime = new_runtime(deque(), method); rows = []
    for z, timestamp in zip(zs, timestamps):
        rows.append(update_channels(runtime, z, pd.to_datetime(timestamp, utc=True), method, half_life))
    return rows


def update_channels(runtime: ResidualRuntime, z: float, timestamp: Any, method: str, half_life_seconds: float) -> dict[str, float]:
    previous = runtime.last_timestamp
    dt = (timestamp - previous).total_seconds() if previous is not None else 0.0
    if dt <= 0: dt = max(1.0, half_life_seconds / 30.0)
    alpha = 1 - math.exp(-math.log(2) * dt / max(half_life_seconds, 1e-6))
    runtime.energy = (1 - alpha) * runtime.energy + alpha * z * z
    focus_stat, change_index = 0.0, None
    if method not in RESIDUAL_METHODS:
        raise ValueError("unknown_residual_method")
    if runtime.focus is None: runtime.focus = _focus()
    runtime.focus.update(z); runtime.focus_index += 1
    focus_stat = float(runtime.focus.statistic())
    point = runtime.focus.changepoint()
    change_index = point.get("changepoint") if isinstance(point, dict) else None
    runtime.last_timestamp = timestamp
    return {"P": abs(z), "D": focus_stat, "N": math.sqrt(max(0.0, runtime.energy)), "alpha": alpha, "focus_changepoint_index": change_index}


def _effective_days(timestamps: list[Any]) -> float:
    if len(timestamps) < 2: return 0.0
    values = [pd.to_datetime(item, utc=True) for item in timestamps]
    cadence = np.median([max(0.0, (b-a).total_seconds()) for a, b in zip(values, values[1:])]) or 60.0
    total = sum(min(max(0.0, (b-a).total_seconds()), cadence * 10) for a, b in zip(values, values[1:]))
    return float(total / 86400.0)


def _choose_threshold(scores: list[float], timestamps: list[Any], days: float) -> tuple[float | None, list[dict[str, Any]]]:
    candidates = list(np.geomspace(.5, 32, 61)) + [64., 128., 256., 512., 1024.]
    report = []
    for threshold in candidates:
        starts, last = 0, None
        for score, value in zip(scores, timestamps):
            timestamp = pd.to_datetime(value, utc=True)
            if score >= threshold and (last is None or (timestamp-last).total_seconds() > DEFAULT_EPISODE_GAP_SECONDS): starts += 1
            if score >= threshold: last = timestamp
        rate = starts / days if days else math.inf
        report.append({"threshold": float(threshold), "episodes": starts, "effective_days": days, "episodes_per_day": rate})
        if rate <= 1.0: return float(threshold), report
    return None, report
