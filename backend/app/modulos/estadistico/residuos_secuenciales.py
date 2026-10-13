"""Detectores secuenciales sobre residuos del predictor temporal.

El predictor de cuantiles no se vuelve a ajustar aquí: se usa únicamente su
estimador central (q=.50).  Los objetos de este módulo contienen estado de una
ejecución y nunca se comparten entre nodos.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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
from app.modulos.estadistico.continuos_temporales import PREDICTION_MODE, increment_features
from app.modulos.estadistico.cobertura import coverage, daily_partitions, reporting_policy

RESIDUAL_FOCUS_METHOD = "residual_focus_v1"
RESIDUAL_METHODS = {RESIDUAL_FOCUS_METHOD}
IMPLEMENTATION_VERSION = "residual-sequential-v5-increment-predictor"
MAX_CALIBRATION_ROW_RATE = .01
MAX_CALIBRATION_EPISODES_PER_DAY = 1.
CHANNELS = ("P", "D", "N")
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
        from focus_cpt import Detector
    except Exception as exc:  # deliberately no EWMA fallback
        raise RuntimeError("focus_dependency_unavailable: focus-cpt==0.1.10") from exc
    return Detector(type="univariate")


@dataclass
class ResidualRuntime:
    history: deque
    energy: float = 0.0
    last_timestamp: Any = None
    focus: Any = None
    focus_index: int = 0
    focus_values: list[float] = field(default_factory=list)

    def __getstate__(self):
        """Persist runtime state without the third-party Focus instance.

        The native detector is not included in the persisted baseline.
        Replay the stored standardized residuals when restoring an active
        stream so deepcopy, cache loading and batch boundaries preserve FOCuS.
        """
        state = self.__dict__.copy()
        state["focus"] = None
        return state

    def __setstate__(self, state):
        if state.get("focus_index", 0) and "focus_values" not in state:
            raise ValueError("Active FOCuS cache lacks residual history; rebuild reference")
        state.setdefault("focus_values", [])
        self.__dict__.update(state)
        self.focus = None

    def reset_sequential(self, method: str) -> None:
        self.energy = 0.0
        self.focus_index = 0
        self.focus_values = []
        self.focus = _focus()


def new_runtime(history: deque, method: str) -> ResidualRuntime:
    runtime = ResidualRuntime(deque(history, maxlen=HISTORY_SIZE))
    runtime.reset_sequential(method)
    return runtime


def prepare_sensor(temporal, frame: pd.DataFrame, entity_id: str, method: str, *, half_life_seconds: float = DEFAULT_HALF_LIFE_SECONDS) -> dict[str, Any]:
    """Fit causal increments with q=.50 and calibrate residual channels.

    Chronological 24-hour blocks separate fit, normalization and calibration.
    The final partial block remains in calibration, with its actual exposure.
    """
    if method not in RESIDUAL_METHODS:
        raise ValueError("unknown_residual_method")
    if method == RESIDUAL_FOCUS_METHOD:
        _focus()  # fail clearly before an evaluation is started
    x, y, timestamps = temporal._samples(frame)
    policy = reporting_policy(frame)
    ordered = frame.sort_values("timestamp_dt", kind="stable") if "timestamp_dt" in frame else frame.iloc[:0]
    latest = ordered.iloc[-1] if len(ordered) else None
    metadata = {"entity_id": str(entity_id), "continuous_method": method,
                "implementation_version": IMPLEMENTATION_VERSION, "reporting_policy": policy,
                "prediction_mode": PREDICTION_MODE,
                "samples": int(len(y)),
                "last_observation_at": latest["timestamp_dt"].isoformat() if latest is not None else None,
                "last_observed_value": str(latest.get("state", latest.get("value"))) if latest is not None else None}
    if latest is not None:
        _, _, _, partitions = daily_partitions(list(ordered["timestamp_dt"]))
        split = sum(t < pd.Timestamp(partitions["normalization_start"]) for t in timestamps)
        midpoint = sum(t < pd.Timestamp(partitions["threshold_start"]) for t in timestamps)
        partitions.update(predictor_fit=split, normalization=midpoint - split, threshold=len(y) - midpoint)
        exposure = coverage(ordered, pd.Timestamp(partitions["threshold_start"]),
                            pd.Timestamp(partitions["threshold_end"]), policy)
        days = exposure["exposure_days"]
        metadata.update(partitions={**partitions, "threshold_days": days}, coverage=exposure)
    if len(y) < MIN_TRAINING_SAMPLES + 2 * MIN_PARTITION_RESIDUALS:
        return {**metadata, "eligible": False, "reason": "insufficient_training_data"}
    if not partitions["enough_calendar_blocks"]:
        return {**metadata, "eligible": False, "reason": "insufficient_calendar_blocks"}
    if split < MIN_TRAINING_SAMPLES or midpoint - split < MIN_PARTITION_RESIDUALS or len(y) - midpoint < MIN_PARTITION_RESIDUALS:
        return {**metadata, "eligible": False, "reason": "insufficient_residual_partition"}
    if days + 1e-9 < MIN_THRESHOLD_DAYS:
        return {**metadata, "eligible": False, "reason": "insufficient_threshold_days", "threshold_days": days}
    inputs, anchors = increment_features(x)
    increments = y - anchors
    models = []
    for quantile in QUANTILES:
        model = HistGradientBoostingRegressor(**MODEL_PARAMETERS, quantile=quantile)
        model.fit(inputs[:split], increments[:split]); models.append(model)
    train_values = y[:split]
    value_floor = max(float(np.finfo(float).eps), float(np.ptp(train_values)) * 1e-9, 1e-12)
    predictor = TemporalSensorModel(tuple(models), 0.0, value_floor,
        {"predictor": "temporal_quantile_q50", "prediction_mode": PREDICTION_MODE,
         "feature_schema_version": getattr(temporal, "feature_schema_version", FEATURE_SCHEMA_VERSION)})
    # Calibration uses only q=.50. Batch prediction is identical and avoids
    # invoking three estimators separately for every historical observation.
    residuals = (y[split:] - predictor.predict_medians(x[split:])).astype(float).tolist()
    normal_residuals = residuals[: midpoint - split]
    threshold_residuals = residuals[midpoint - split :]
    centre, scale, scale_source = _scale(normal_residuals, value_floor)
    normal_z = [(item - centre) / scale for item in normal_residuals]
    channels = _channel_sequence(normal_z, timestamps[split:midpoint], method, half_life_seconds, frame=ordered, policy=policy)
    quantiles = {key: max(float(np.quantile([row[key] for row in channels], .99)), 1e-6) for key in ("P", "D", "N")}
    threshold_z = [(item - centre) / scale for item in threshold_residuals]
    threshold_channels = _channel_sequence(threshold_z, timestamps[midpoint:], method, half_life_seconds, frame=ordered, policy=policy)
    thresholds, candidates = {}, {}
    for channel in CHANNELS:
        scores = [row[channel] / quantiles[channel] for row in threshold_channels]
        thresholds[channel], candidates[channel] = _choose_threshold(
            scores, timestamps[midpoint:], days,
            max_row_rate=MAX_CALIBRATION_ROW_RATE / len(CHANNELS),
            max_episodes_per_day=MAX_CALIBRATION_EPISODES_PER_DAY / len(CHANNELS))
    if any(value is None for value in thresholds.values()):
        return {**metadata, "eligible": False, "reason": "calibration_budget_unmet", "calibration_candidates": candidates}
    times = np.asarray([pd.Timestamp(value).timestamp() for value in timestamps[midpoint:]])
    # Recheck candidate thresholds with the actual alert/reset policy. Resetting
    # D and N changes subsequent scores, so the uninterrupted proposal alone
    # cannot certify the calibration budgets.
    for attempt in range(61):
        threshold_channels = _channel_sequence(threshold_z, timestamps[midpoint:], method,
            half_life_seconds, frame=ordered, policy=policy,
            channel_q99=quantiles, channel_thresholds=thresholds)
        over_budget = []
        for channel in CHANNELS:
            flags = np.asarray([row[channel] / quantiles[channel] >= thresholds[channel]
                                for row in threshold_channels], dtype=bool)
            flagged = times[flags]
            count = int(bool(len(flagged))) + int(np.sum(np.diff(flagged) > DEFAULT_EPISODE_GAP_SECONDS))
            if flags.mean() > MAX_CALIBRATION_ROW_RATE / len(CHANNELS) or count / days > MAX_CALIBRATION_EPISODES_PER_DAY / len(CHANNELS):
                over_budget.append(channel)
        if not over_budget:
            break
        if attempt == 60:
            return {**metadata, "eligible": False, "reason": "calibration_budget_unmet",
                    "calibration_candidates": candidates}
        for channel in over_budget:
            thresholds[channel] *= 2 ** (1 / 6)
    joint_flags = [any(row[channel] / quantiles[channel] >= thresholds[channel] for channel in CHANNELS)
                   for row in threshold_channels]
    flagged_times = times[np.asarray(joint_flags, dtype=bool)]
    episodes = int(bool(len(flagged_times))) + int(np.sum(np.diff(flagged_times) > DEFAULT_EPISODE_GAP_SECONDS))
    history = deque(maxlen=HISTORY_SIZE)
    for row in ordered.tail(HISTORY_SIZE).to_dict("records"):
        value = row.get("value")
        if not bool(row.get("is_valid_value", True)) or pd.isna(value) or not math.isfinite(float(value)):
            history.clear()
        else:
            history.append((row["timestamp_dt"], float(value)))
    return {
        **metadata, "eligible": True,
        "predictor": predictor, "centre": centre, "scale": scale, "scale_source": scale_source, "channel_q99": quantiles,
        "threshold": 1., "channel_thresholds": thresholds,
        "calibration_policy": {"max_row_rate": MAX_CALIBRATION_ROW_RATE,
            "max_episodes_per_day": MAX_CALIBRATION_EPISODES_PER_DAY,
            "channel_budget_allocation": "equal_union_bound", "reset_after_detection": True,
            "reset_trigger": "residual_sequential_detection",
            "reset_on_invalid_state_or_declared_periodic_gap": True},
        "joint_calibration": {"row_count": len(joint_flags), "flagged_rows": sum(joint_flags),
            "flagged_row_rate": sum(joint_flags) / len(joint_flags), "episodes": episodes,
            "episodes_per_day": episodes / days},
        "half_life_seconds": half_life_seconds, "episode_gap_seconds": DEFAULT_EPISODE_GAP_SECONDS,
        "median_cadence_seconds": _median_cadence_seconds(frame), "calibration_candidates": candidates,
        "runtime": new_runtime(history, method),
    }


def _channel_sequence(zs: list[float], timestamps: list[Any], method: str, half_life: float, *, frame=None, policy=None, channel_q99=None, channel_thresholds=None) -> list[dict[str, float]]:
    runtime = new_runtime(deque(), method); rows = []
    barriers = []
    if frame is not None:
        values = pd.to_numeric(frame["value"], errors="coerce")
        valid = values.notna() & np.isfinite(values.astype(float))
        if "is_valid_value" in frame:
            valid &= frame["is_valid_value"].fillna(False).astype(bool)
        barriers = list(frame.loc[~valid, "timestamp_dt"])
    barrier_index, index = 0, 0
    while index < len(zs):
        timestamp = pd.to_datetime(timestamps[index], utc=True)
        end = index + 1
        while end < len(zs) and timestamps[end] == timestamps[index]:
            end += 1
        interrupted = False
        while barrier_index < len(barriers) and barriers[barrier_index] <= timestamp:
            interrupted |= runtime.last_timestamp is not None and barriers[barrier_index] > runtime.last_timestamp
            barrier_index += 1
        if policy and policy["mode"] == "periodic" and runtime.last_timestamp is not None:
            limit = policy["max_report_gap_seconds"]
            interrupted |= limit is None or (timestamp - runtime.last_timestamp).total_seconds() > limit
        if interrupted:
            runtime.reset_sequential(method)
            runtime.last_timestamp = None
        result = update_channels(runtime, float(np.mean(zs[index:end])), timestamp, method, half_life)
        group_rows = [{**result, "P": abs(z)} for z in zs[index:end]]
        rows.extend(group_rows)
        if channel_thresholds and any(row[channel] / channel_q99[channel] >= channel_thresholds[channel]
                                      for row in group_rows for channel in CHANNELS):
            runtime.reset_sequential(method)
        index = end
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
    if runtime.focus is None:
        runtime.focus = _focus()
        for previous_z in runtime.focus_values:
            runtime.focus.update(previous_z)
    runtime.focus.update(z); runtime.focus_index += 1
    runtime.focus_values.append(float(z))
    # Known zero pre-change mean, matching the standardized residual baseline.
    point = runtime.focus.get_statistics(family="gaussian", theta0=0.0)
    focus_stat = float(point["stat"])
    raw_index = point.get("changepoint")
    change_index = int(raw_index) if raw_index is not None and math.isfinite(float(raw_index)) else None
    runtime.last_timestamp = timestamp
    return {"P": abs(z), "D": focus_stat, "N": math.sqrt(max(0.0, runtime.energy)), "alpha": alpha, "focus_changepoint_index": change_index}


def _effective_days(timestamps: list[Any]) -> float:
    """Observed elapsed span; silence alone is not proof of missing reports."""
    if len(timestamps) < 2: return 0.0
    values = [pd.to_datetime(item, utc=True) for item in timestamps]
    total = sum(max(0.0, (b-a).total_seconds()) for a, b in zip(values, values[1:]))
    return float(total / 86400.0)


def _choose_threshold(scores: list[float], timestamps: list[Any], days: float, *, max_row_rate=MAX_CALIBRATION_ROW_RATE, max_episodes_per_day=MAX_CALIBRATION_EPISODES_PER_DAY) -> tuple[float | None, list[dict[str, Any]]]:
    values = np.asarray(scores, dtype=float)
    if not len(values) or not np.isfinite(values).all() or days <= 0:
        return None, []
    times = np.asarray([pd.Timestamp(value).timestamp() for value in timestamps])
    candidates = sorted(set([1., *np.geomspace(1., 1024., 61),
        max(1., float(np.quantile(values, 1 - max_row_rate))),
        max(1., float(np.nextafter(values.max(), np.inf)))]))
    report = []
    for threshold in candidates:
        flagged = times[values >= threshold]
        starts = int(bool(len(flagged))) + int(np.sum(np.diff(flagged) > DEFAULT_EPISODE_GAP_SECONDS))
        rate = starts / days if days else math.inf
        row_rate = len(flagged) / len(values)
        report.append({"threshold": float(threshold), "episodes": starts, "effective_days": days,
            "episodes_per_day": rate, "flagged_rows": len(flagged), "row_count": len(values), "flagged_row_rate": row_rate})
        if rate <= max_episodes_per_day and row_rate <= max_row_rate:
            return float(threshold), report
    return None, report
