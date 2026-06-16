from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, deque
from math import log
from time import perf_counter
from typing import Any

import pandas as pd

from .features import build_binary_temporal_features
from .stat_utils import EPSILON, mad, robust_z, stdev, transition_rarity


class StatisticalFilterStrategy(ABC):
    """Contrato comun para evaluar una entidad segun su tipo de dato."""

    sensor_kind: str

    @abstractmethod
    def evaluate(
        self,
        rows: pd.DataFrame,
        base: dict[str, Any],
        ground_truth: dict[tuple[str, str], dict[str, Any]],
        suspicion_threshold: float,
        feature_rows: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        """Devuelve sospechas estadisticas para una entidad."""


class ContinuousStatisticalStrategy(StatisticalFilterStrategy):
    sensor_kind = "continuous"

    def evaluate(
        self,
        rows: pd.DataFrame,
        base: dict[str, Any],
        ground_truth: dict[tuple[str, str], dict[str, Any]],
        suspicion_threshold: float,
        feature_rows: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        return run_continuous_filter(rows, base, ground_truth, suspicion_threshold)


class BinaryStatisticalStrategy(StatisticalFilterStrategy):
    sensor_kind = "binary"

    def evaluate(
        self,
        rows: pd.DataFrame,
        base: dict[str, Any],
        ground_truth: dict[tuple[str, str], dict[str, Any]],
        suspicion_threshold: float,
        feature_rows: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        return run_binary_filter(rows, base, ground_truth, feature_rows, suspicion_threshold)


class MultistateStatisticalStrategy(StatisticalFilterStrategy):
    sensor_kind = "multistate"

    def evaluate(
        self,
        rows: pd.DataFrame,
        base: dict[str, Any],
        ground_truth: dict[tuple[str, str], dict[str, Any]],
        suspicion_threshold: float,
        feature_rows: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        return run_multistate_filter(rows, base, ground_truth, suspicion_threshold)


class StatisticalStrategyFactory:
    """Factory Method para seleccionar la estrategia estadistica adecuada."""

    def __init__(self) -> None:
        self._strategies: dict[str, StatisticalFilterStrategy] = {
            "continuous": ContinuousStatisticalStrategy(),
            "binary": BinaryStatisticalStrategy(),
            "multistate": MultistateStatisticalStrategy(),
        }

    def create(self, sensor_kind: str) -> StatisticalFilterStrategy:
        return self._strategies.get(sensor_kind, self._strategies["multistate"])


def run_statistical_filter(
    normalized: pd.DataFrame,
    baseline: dict[str, dict[str, Any]],
    ground_truth: dict[tuple[str, str], dict[str, Any]],
    limit: int = 1000,
    feature_tables: dict[str, pd.DataFrame] | None = None,
    suspicion_threshold: float = 0.5,
) -> dict[str, Any]:
    started_at = perf_counter()
    results: list[dict[str, Any]] = []
    timings_by_kind: dict[str, float] = Counter()
    if normalized.empty:
        return {
            "results": 0,
            "suspicious": 0,
            "by_kind": {},
            "top_results": [],
            "all_results": [],
            "elapsed_ms": 0.0,
            "elapsed_seconds": 0.0,
            "timings_by_kind_ms": {},
        }

    # El pipeline entrega una tabla normalizada. Este modulo calcula los
    # componentes definidos en la tesis y los combina como puntaje S_i(t).
    strategy_factory = StatisticalStrategyFactory()
    for entity_id, rows in normalized.groupby("entity_id", sort=True):
        base = baseline.get(str(entity_id))
        if not base:
            continue
        train_rows = int(base["train_rows"])
        if base.get("evaluation_mode") == "new_events":
            evaluation_rows = rows.sort_values("timestamp_dt")
        else:
            evaluation_rows = rows.sort_values("timestamp_dt").iloc[train_rows:]
        if evaluation_rows.empty:
            evaluation_rows = rows.sort_values("timestamp_dt")
        kind = str(base["sensor_kind"])
        kind_started_at = perf_counter()
        feature_rows = feature_tables.get(str(entity_id)) if feature_tables else None
        strategy = strategy_factory.create(kind)
        results.extend(
            strategy.evaluate(
                evaluation_rows,
                base,
                ground_truth,
                suspicion_threshold,
                feature_rows,
            )
        )
        timings_by_kind[kind] += perf_counter() - kind_started_at

    results.sort(key=lambda item: item.get("suspicion_score", 0), reverse=True)
    limited_results = results[:limit]
    by_kind = Counter(item["sensor_kind"] for item in limited_results)
    suspicious = sum(1 for item in limited_results if item["is_suspicious"])
    elapsed_seconds = perf_counter() - started_at
    first_detection_delay = first_detection_delay_seconds(results, ground_truth)
    return {
        "results": len(limited_results),
        "suspicious": suspicious,
        "by_kind": {str(key): int(value) for key, value in by_kind.items()},
        "top_results": limited_results[:50],
        "all_results": limited_results,
        "elapsed_ms": round(elapsed_seconds * 1000, 3),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "timings_by_kind_ms": {
            str(kind): round(seconds * 1000, 3)
            for kind, seconds in timings_by_kind.items()
        },
        "first_detection_delay_seconds": first_detection_delay,
        "suspicion_threshold": clamp_unit_interval(suspicion_threshold),
    }


def run_continuous_filter(
    rows: pd.DataFrame,
    base: dict[str, Any],
    ground_truth: dict[tuple[str, str], dict[str, Any]],
    suspicion_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    summary = base["summary"]
    threshold = threshold_for_score(summary, suspicion_threshold)
    ordered = rows.sort_values("timestamp_dt").reset_index(drop=True)
    values = pd.to_numeric(ordered["value"], errors="coerce").astype(float)

    # Pandas calcula senales vectoriales. Las acumulaciones temporales quedan en
    # el bucle porque dependen del orden de llegada de las lecturas.
    minutes = ordered["timestamp_dt"].diff().dt.total_seconds() / 60.0
    rates = values.diff().abs() / minutes
    ewma = values.ewm(alpha=0.25, adjust=False).mean()
    local_median = values.rolling(12, min_periods=1).median()
    local_mad = values.rolling(12, min_periods=1).apply(lambda window: mad(window.tolist()), raw=False)
    local_std = values.rolling(12, min_periods=6).std()

    output = []
    drift_values: deque[tuple[pd.Timestamp, float]] = deque()
    value_values: deque[tuple[pd.Timestamp, float]] = deque()
    residual_ewma = 0.0
    cusum_positive = 0.0
    cusum_negative = 0.0
    previous_timestamp: pd.Timestamp | None = None
    for index, row in ordered.iterrows():
        value = float(values.iloc[index])
        reference = continuous_reference_for_hour(summary, row["timestamp_dt"].hour)
        previous_value = None if index == 0 else float(values.iloc[index - 1])
        delta_value = 0.0 if previous_value is None else value - previous_value
        delta_minutes = None if pd.isna(minutes.iloc[index]) else float(minutes.iloc[index])
        z = robust_z(value, reference["median"], reference["mad"])
        iqr_flag = 1.0 if value < reference["lower"] or value > reference["upper"] else 0.0
        rate_z = 0.0
        if pd.notna(rates.iloc[index]):
            rate_z = robust_z(float(rates.iloc[index]), summary["rate"]["median"], summary["rate"]["mad"])
        hampel = abs(value - float(local_median.iloc[index])) / (1.4826 * float(local_mad.iloc[index]) + EPSILON)
        ewma_z = robust_z(float(ewma.iloc[index]), reference["median"], reference["mad"])
        residual = value - reference["median"]
        residual_ewma = 0.30 * residual + 0.70 * residual_ewma

        # CUSUM se calcula sobre residuales con tolerancia K. Si hay una pausa
        # mayor a la ventana configurada se reinicia la acumulacion.
        current_timestamp = row["timestamp_dt"]
        cusum_window_minutes = int(summary.get("cusum_window_minutes", 120))
        if previous_timestamp is not None:
            gap_minutes = (current_timestamp - previous_timestamp).total_seconds() / 60.0
            if gap_minutes > cusum_window_minutes:
                cusum_positive = 0.0
                cusum_negative = 0.0
        previous_timestamp = current_timestamp
        cusum_k = max(float(reference["mad"]) * 0.50, EPSILON)
        cusum_positive = max(0.0, cusum_positive + residual - cusum_k)
        cusum_negative = max(0.0, cusum_negative - residual - cusum_k)
        cusum_raw = max(cusum_positive, cusum_negative)
        cusum_theta = max(float(reference["mad"]) * 8.0, EPSILON)

        drift_values.append((current_timestamp, residual))
        drift_keep_from = row["timestamp_dt"] - pd.Timedelta(minutes=int(summary.get("drift_window_minutes", 120)))
        while drift_values and drift_values[0][0] < drift_keep_from:
            drift_values.popleft()

        drift_slope_z = 0.0
        drift_mean_residual = 0.0
        if len(drift_values) >= 6:
            first_time = drift_values[0][0]
            x = [
                (timestamp - first_time).total_seconds() / 3600.0
                for timestamp, _ in drift_values
            ]
            y = [residual for _, residual in drift_values]
            x_mean = sum(x) / len(x)
            y_mean = sum(y) / len(y)
            denominator = sum((item - x_mean) ** 2 for item in x)
            if denominator > 0:
                slope = sum((x[item] - x_mean) * (y[item] - y_mean) for item in range(len(x))) / denominator
                drift_slope_z = robust_z(
                    abs(slope),
                    float(summary.get("residual_slope", {}).get("median", 0.0)),
                    float(summary.get("residual_slope", {}).get("mad", 0.0)),
                )
            drift_mean_residual = abs(y_mean)

        value_values.append((current_timestamp, value))
        while value_values and value_values[0][0] < drift_keep_from:
            value_values.popleft()
        value_slope_z = 0.0
        if len(value_values) >= 6:
            value_slope = abs(local_slope_per_hour(value_values))
            value_slope_z = robust_z(
                value_slope,
                float(summary.get("value_slope", {}).get("median", 0.0)),
                float(summary.get("value_slope", {}).get("mad", 0.0)),
            )

        variability_z = 0.0
        if pd.notna(local_std.iloc[index]):
            variability_z = robust_z(float(local_std.iloc[index]), summary["window_stdev"]["median"], summary["window_stdev"]["mad"])

        s_z = component_score(z, 3.5)
        s_iqr = iqr_flag
        s_rate = component_score(rate_z, 3.5)
        s_hampel = component_score(hampel, 3.5)
        s_cusum = component_score(cusum_raw, cusum_theta)
        s_variability = component_score(variability_z, 3.5)
        s_beta = component_score(value_slope_z, 3.5)
        s_drift = min(
            1.0,
            max(
                component_score(abs(residual_ewma), max(float(reference["mad"]) * 3.0, EPSILON)),
                component_score(drift_slope_z, 3.5),
                s_cusum,
            ),
        )

        score = weighted_component_score(
            {
                "s_z": s_z,
                "s_iqr": s_iqr,
                "s_rate": s_rate,
                "s_hampel": s_hampel,
                "s_cusum": s_cusum,
                "s_variability": s_variability,
                "s_beta": s_beta,
                "s_drift": s_drift,
            },
            {
                "s_z": 1.0,
                "s_iqr": 1.0,
                "s_rate": 1.0,
                "s_hampel": 1.0,
                "s_cusum": 1.0,
                "s_variability": 1.0,
                "s_beta": 1.0,
                "s_drift": 1.0,
            },
        )
        indicators = []
        if s_z >= 1.0:
            indicators.append("robust_z")
        if s_iqr:
            indicators.append("iqr")
        if s_rate >= 1.0:
            indicators.append("rate_change")
        if s_hampel >= 1.0:
            indicators.append("hampel")
        if ewma_z >= 3.5:
            indicators.append("ewma")
        if s_cusum >= 1.0:
            indicators.append("cusum")
        if s_variability >= 1.0:
            indicators.append("window_variability")
        if s_beta >= 1.0:
            indicators.append("local_slope")
        if s_drift >= 1.0 and drift_mean_residual >= reference["mad"] * 2:
            indicators.append("drift_slope")
        indicators, score = suppress_weak_persistent_shift(indicators, score, z, iqr_flag, rate_z, hampel, variability_z, threshold)
        is_suspicious = score >= threshold
        if is_suspicious:
            output.append(
                make_result(
                    row,
                    "continuous",
                    score,
                    is_suspicious,
                    indicators,
                    summary,
                    truth_for(ground_truth, row),
                    {
                        "value": round(value, 4),
                        "unit": "numeric",
                        "previous_value": None if previous_value is None else round(previous_value, 4),
                        "delta_value": round(float(delta_value), 4),
                        "delta_minutes": None if delta_minutes is None else round(float(delta_minutes), 4),
                        "rate_per_minute": None if pd.isna(rates.iloc[index]) else round(float(rates.iloc[index]), 6),
                        "return_to_normal_seconds": return_to_normal_seconds(ordered, values, index, reference),
                        "robust_z": round(float(z), 4),
                        "rate_z": round(float(rate_z), 4),
                        "ewma_z": round(float(ewma_z), 4),
                        "cusum_score": round(float(s_cusum), 4),
                        "cusum_raw": round(float(cusum_raw), 4),
                        "residual_ewma": round(float(residual_ewma), 4),
                        "value_slope_z": round(float(value_slope_z), 4),
                        "drift_slope_z": round(float(drift_slope_z), 4),
                        "drift_mean_residual": round(float(drift_mean_residual), 4),
                        "components": {
                            "s_Z": round(float(s_z), 4),
                            "s_I": round(float(s_iqr), 4),
                            "s_R": round(float(s_rate), 4),
                            "s_H": round(float(s_hampel), 4),
                            "s_C": round(float(s_cusum), 4),
                            "s_sigma": round(float(s_variability), 4),
                            "s_beta": round(float(s_beta), 4),
                            "S_drift": round(float(s_drift), 4),
                        },
                        "reference_hour": int(row["timestamp_dt"].hour),
                    },
                )
            )
    return output


def run_binary_filter(
    rows: pd.DataFrame,
    base: dict[str, Any],
    ground_truth: dict[tuple[str, str], dict[str, Any]],
    feature_rows: pd.DataFrame | None = None,
    suspicion_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    summary = base["summary"]
    threshold = threshold_for_score(summary, suspicion_threshold)
    ordered = rows.sort_values("timestamp_dt").reset_index(drop=True)
    metrics = feature_rows.copy() if feature_rows is not None else build_binary_temporal_features(
        ordered,
        window_seconds=int(summary.get("window_seconds", 60)),
        step_seconds=int(summary.get("step_seconds", 10)),
    )
    output = []
    emitted_keys: set[tuple[str, str]] = set()
    activation_burst_times: deque[pd.Timestamp] = deque()
    persistence_times: deque[pd.Timestamp] = deque()
    high_activity_times: deque[pd.Timestamp] = deque()
    absence_times: deque[pd.Timestamp] = deque()
    for _, metric in metrics.iterrows():
        index = min(int(metric["start"]), len(ordered) - 1)
        row = ordered.iloc[index]
        metric_time = pd.to_datetime(metric.get("window_start", row["timestamp_iso"]), errors="coerce")
        if pd.isna(metric_time):
            metric_time = row["timestamp_dt"]
        z_activations = upper_tail_score(float(metric["activations"]), summary["activations"], summary["percentiles"]["activations_p95"])
        z_changes = upper_tail_score(float(metric["state_changes"]), summary["state_changes"], summary["percentiles"]["state_changes_p95"])
        z_active_seconds = upper_tail_score(float(metric["active_seconds"]), summary["active_seconds"], summary["percentiles"]["active_seconds_p95"])
        z_ratio = upper_tail_score(float(metric["active_ratio"]), summary["active_ratio"], summary["percentiles"]["active_ratio_p95"])
        z_max_active = upper_tail_score(float(metric["max_active_seconds"]), summary["max_active_seconds"], summary["percentiles"]["max_active_seconds_p95"])
        z_gap = upper_tail_score(float(metric["gap"]), summary["gap"], summary["percentiles"]["gap_p95"])
        s_activation = component_score(z_activations, 3.5)
        s_ratio = max(component_score(z_ratio, 3.5), component_score(z_active_seconds, 3.5))
        s_changes = component_score(z_changes, 3.5)
        s_duration = component_score(z_max_active, 3.5)
        s_gap = component_score(z_gap, 3.5)

        # S_bin(W) combina A_W, P_on,W, C_W, D_on,max y G(t), todos ya
        # normalizados en 0..1 como indica la tesis.
        score = weighted_component_score(
            {
                "s_A": s_activation,
                "s_P": s_ratio,
                "s_C": s_changes,
                "s_D": s_duration,
                "s_G": s_gap,
            },
            {
                "s_A": 1.0,
                "s_P": 1.0,
                "s_C": 1.0,
                "s_D": 1.0,
                "s_G": 1.0,
            },
        )
        window_seconds = float(summary.get("window_seconds", 60))
        activation_limit = max(float(summary["percentiles"]["activations_p95"]) + 2.0, 3.0)
        changes_limit = max(float(summary["percentiles"]["state_changes_p95"]) + 3.0, 5.0)
        persistence_ratio_limit = max(min(float(summary["percentiles"]["active_ratio_p95"]) + 0.30, 1.0), 0.95)
        max_active_limit = max(float(summary["percentiles"]["max_active_seconds_p95"]) + window_seconds * 0.35, window_seconds * 0.90)
        gap_limit = binary_gap_limit_for_hour(summary, metric_time)
        active_ratio = float(metric["active_ratio"])
        active_seconds = float(metric["active_seconds"])
        high_activity_limit = max(min(float(summary["percentiles"]["active_ratio_p95"]) + 0.25, 0.85), 0.50)
        indicators = []
        if s_activation >= 1.0 and float(metric["activations"]) >= activation_limit:
            indicators.append("activation_frequency")
        if s_changes >= 1.0 and float(metric["state_changes"]) >= changes_limit:
            indicators.append("state_changes")
        if s_ratio >= 1.0 and float(metric["active_ratio"]) >= persistence_ratio_limit:
            indicators.append("state_persistence")
        if s_duration >= 1.0 and float(metric["max_active_seconds"]) >= max_active_limit:
            indicators.append("max_active_duration")
        if s_gap >= 1.0 and float(metric["gap"]) >= gap_limit:
            indicators.append("absence_gap")
        unexpected_time_score = unexpected_binary_activation_score(summary, metric, metric_time)
        if unexpected_time_score >= 1.0:
            indicators.append("unexpected_activation_time")
            score = max(score, min(1.0, 0.30 + component_score(unexpected_time_score, 3.0) * 0.55))
        current_activation_burst = "activation_frequency" in indicators
        current_persistence = "state_persistence" in indicators or "max_active_duration" in indicators
        current_high_activity = active_ratio >= high_activity_limit and active_seconds >= window_seconds * 0.50
        burst_count = update_recent_binary_pattern(
            activation_burst_times,
            metric_time,
            current_activation_burst,
            lookback_seconds=300,
        )
        persistence_count = update_recent_binary_pattern(
            persistence_times,
            metric_time,
            current_persistence,
            lookback_seconds=300,
        )
        high_activity_count = update_recent_binary_pattern(
            high_activity_times,
            metric_time,
            current_high_activity,
            lookback_seconds=300,
        )
        absence_count = update_recent_binary_pattern(
            absence_times,
            metric_time,
            "absence_gap" in indicators,
            lookback_seconds=600,
        )
        # La memoria de 5 minutos confirma que el patron es sostenido, pero no
        # debe disparar sospechas cuando la ventana actual ya volvio a normal.
        if current_activation_burst and burst_count >= 3:
            indicators.append("sustained_activation_burst")
            score = min(1.0, max(score, score + 0.10))
        if current_persistence and persistence_count >= 3:
            indicators.append("sustained_state_persistence")
            score = min(1.0, max(score, score + 0.10))
        if current_high_activity and high_activity_count >= 3:
            indicators.append("sustained_high_active_ratio")
            score = min(1.0, max(score, score + min(0.35, high_activity_count * 0.04)))
        if "absence_gap" in indicators and absence_count >= 3:
            indicators.append("sustained_absence_gap")
            score = min(1.0, max(score, score + min(0.30, absence_count * 0.04)))
        indicators, score = suppress_weak_binary_event(indicators, score, metric, threshold)
        # El indicador explica que variable se movio, pero la sospecha exige
        # que la combinacion ponderada cruce el umbral. Asi evitamos reportar
        # como falla una rafaga normal de movimiento con score < 0.5.
        is_suspicious = score >= threshold
        if not is_suspicious:
            continue
        event_key = (str(row["timestamp_iso"]), ",".join(indicators))
        if event_key in emitted_keys:
            continue
        emitted_keys.add(event_key)
        output.append(
            make_result(
                row,
                "binary",
                score,
                is_suspicious,
                    indicators,
                    summary,
                    truth_for(ground_truth, row),
                    {
                        "activations": int(metric["activations"]),
                        "state_changes": int(metric["state_changes"]),
                        "active_seconds": round(float(metric["active_seconds"]), 3),
                        "active_ratio": round(float(metric["active_ratio"]), 4),
                        "max_active_seconds": round(float(metric["max_active_seconds"]), 3),
                        "gap_seconds": round(float(metric["gap"]), 3),
                        "gap_limit_seconds": round(float(gap_limit), 3),
                        "sustained_windows_5m": int(max(burst_count, persistence_count, high_activity_count)),
                        "sustained_absence_windows_10m": int(absence_count),
                        "mean_activation_interval": None if pd.isna(metric["mean_activation_interval"]) else round(float(metric["mean_activation_interval"]), 3),
                        "components": {
                            "s_A": round(float(s_activation), 4),
                            "s_P": round(float(s_ratio), 4),
                            "s_C": round(float(s_changes), 4),
                            "s_D": round(float(s_duration), 4),
                            "s_G": round(float(s_gap), 4),
                        },
                    },
                )
            )
    return output


def run_multistate_filter(
    rows: pd.DataFrame,
    base: dict[str, Any],
    ground_truth: dict[tuple[str, str], dict[str, Any]],
    suspicion_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    summary = base["summary"]
    threshold = threshold_for_score(summary, suspicion_threshold)
    ordered = rows.sort_values("timestamp_dt").reset_index(drop=True)
    state_set = set(summary["state_set"])
    transitions = {key: Counter(value) for key, value in summary["transitions"].items()}
    output = []
    previous_state = None
    current_state = None
    state_started_at = None
    window_size = 30

    # En multivaloricos se busca cambio de distribucion, transiciones raras y
    # permanencias anormalmente largas/cortas para el estado actual.
    for index, row in ordered.iterrows():
        state = str(row["state"])
        indicators = []
        rarity = 0.0
        if previous_state is not None:
            rarity = transition_rarity(transitions, previous_state, state, state_set | {state})
            s_rarity = component_score(rarity, 3.0)
            if s_rarity >= 1.0:
                indicators.append("rare_transition")
        else:
            s_rarity = 0.0
        if state != current_state:
            previous_for_transition = current_state
            current_state = state
            state_started_at = row["timestamp_dt"]
        else:
            previous_for_transition = previous_state
        duration = max((row["timestamp_dt"] - state_started_at).total_seconds() / 60.0, 1.0) if state_started_at is not None else 1.0
        duration_summary = summary["durations"].get(state, {"median": 1.0, "mad": 0.0})
        duration_score = protected_multistate_duration_score(duration, duration_summary)
        # Para tiempo real, la primera lectura de un estado siempre tiene una
        # duracion corta. Aqui solo marcamos permanencias largas anormales, que
        # son las que representan stuck/state_stuck en la tesis.
        s_duration = component_score(duration_score, 3.5)
        if s_duration >= 1.0:
            indicators.append("state_duration")

        kl_score = 0.0
        if index >= window_size:
            window = ordered.iloc[index - window_size + 1 : index + 1]
            distribution = window["state"].value_counts(normalize=True)
            total_base = max(sum(summary["states"].values()), 1)
            for item_state, observed in distribution.items():
                expected = summary["states"].get(str(item_state), 0) / total_base
                kl_score += float(observed) * log((float(observed) + EPSILON) / (float(expected) + EPSILON))
            if component_score(kl_score, 0.35) >= 1.0:
                indicators.append("state_distribution")
        s_kl = component_score(kl_score, 0.35)

        score = weighted_component_score(
            {"s_KL": s_kl, "s_R": s_rarity, "s_d": s_duration},
            {"s_KL": 1.0, "s_R": 1.2, "s_d": 1.0},
        )
        is_suspicious = score >= threshold
        if is_suspicious:
            output.append(
                make_result(
                    row,
                    "multistate",
                    score,
                    is_suspicious,
                    indicators,
                    summary,
                    truth_for(ground_truth, row),
                    {
                        "state": state,
                        "previous_state": previous_for_transition,
                        "duration_seconds": round(float(duration * 60), 3),
                        "duration_score": round(float(duration_score), 4),
                        "transition_rarity": round(float(rarity), 4),
                        "distribution_score": round(float(kl_score), 4),
                        "components": {
                            "s_KL": round(float(s_kl), 4),
                            "s_R": round(float(s_rarity), 4),
                            "s_d": round(float(s_duration), 4),
                        },
                    },
                )
            )
        previous_state = state
    return output


def protected_multistate_duration_score(duration: float, duration_summary: dict[str, Any]) -> float:
    median_duration = float(duration_summary.get("median", 1.0) or 1.0)
    mad_duration = float(duration_summary.get("mad", 0.0) or 0.0)
    if duration <= median_duration:
        return 0.0
    if mad_duration == 0:
        return 0.0 if duration <= median_duration * 1.5 else 6.0
    return robust_z(duration, median_duration, mad_duration)


def truth_for(ground_truth: dict[tuple[str, str], dict[str, Any]], row: pd.Series) -> str | None:
    truth = ground_truth.get((str(row["timestamp_iso"]), str(row["entity_id"])))
    return truth.get("label") if truth else None


def make_result(
    row: pd.Series,
    sensor_kind: str,
    score: float,
    is_suspicious: bool,
    indicators: list[str],
    baseline_summary: dict[str, Any],
    ground_truth_label: str | None,
    reading: dict[str, Any],
) -> dict[str, Any]:
    threshold = float(baseline_summary.get("threshold", 1.0) or 1.0)
    pattern = pattern_from_indicators(sensor_kind, indicators)
    value = reading.get("value", reading.get("state", row["state"]))
    return {
        "entity_id": str(row["entity_id"]),
        "timestamp": str(row["timestamp_iso"]),
        "window_start": str(row["timestamp_iso"]),
        "window_end": str(row["timestamp_iso"]),
        "sensor_kind": sensor_kind,
        "tipo_interno": tipo_from_sensor_kind(sensor_kind),
        "observed_state": str(row["state"]),
        "value": value,
        "reading": reading,
        "raw_score": round(float(score), 4),
        "suspicion_score": round(score_to_unit_interval(float(score), threshold), 4),
        "is_suspicious": bool(is_suspicious),
        "pattern": pattern,
        "suspected_pattern": pattern,
        "pattern_detail": pattern_detail_from_indicators(sensor_kind, indicators, reading),
        "filter": "statistical_temporal",
        "passed_to_contextual_filter": bool(is_suspicious),
        "triggered_indicators": indicators,
        "baseline_summary": compact_baseline(baseline_summary),
        "ground_truth_label": ground_truth_label,
    }


def continuous_reference_for_hour(summary: dict[str, Any], hour: int) -> dict[str, Any]:
    hourly = summary.get("hourly", {})
    return hourly.get(hour) or hourly.get(str(hour)) or summary


def return_to_normal_seconds(
    ordered: pd.DataFrame,
    values: pd.Series,
    index: int,
    reference: dict[str, Any],
) -> float | None:
    """Tiempo hasta volver al rango robusto despues de una sospecha continua."""
    current_time = ordered["timestamp_dt"].iloc[index]
    lower = float(reference.get("lower", 0.0))
    upper = float(reference.get("upper", 0.0))
    for next_index in range(index + 1, len(ordered)):
        next_value = float(values.iloc[next_index])
        if lower <= next_value <= upper:
            next_time = ordered["timestamp_dt"].iloc[next_index]
            return round(float((next_time - current_time).total_seconds()), 3)
    return None


def tipo_from_sensor_kind(sensor_kind: str) -> str:
    if sensor_kind == "continuous":
        return "continua"
    if sensor_kind == "binary":
        return "estado_binario"
    return "estado_multivalor"


def suppress_weak_persistent_shift(
    indicators: list[str],
    score: float,
    z: float,
    iqr_flag: float,
    rate_z: float,
    hampel: float,
    variability_z: float,
    threshold: float,
) -> tuple[list[str], float]:
    if "drift_slope" in indicators:
        return indicators, score

    persistent_only = set(indicators).issubset({"ewma", "cusum"}) and bool(indicators)
    value_is_reasonable = z < 4 and not iqr_flag and rate_z < 4 and hampel < 4 and variability_z < 4
    if persistent_only and value_is_reasonable:
        return [], min(score, threshold * 0.45)
    return indicators, score


def pattern_detail_from_indicators(sensor_kind: str, indicators: list[str], reading: dict[str, Any]) -> str:
    if not indicators:
        return "score_threshold"
    if sensor_kind == "continuous":
        details = []
        if "ewma" in indicators:
            details.append(f"ewma_z={reading.get('ewma_z')}")
        if "cusum" in indicators:
            details.append(f"cusum_score={reading.get('cusum_score')}")
        if "robust_z" in indicators:
            details.append(f"robust_z={reading.get('robust_z')}")
        if "rate_change" in indicators:
            details.append(f"rate_z={reading.get('rate_z')}")
            details.append(f"delta={reading.get('delta_value')}")
            details.append(f"dt_min={reading.get('delta_minutes')}")
        if "drift_slope" in indicators:
            details.append(f"drift_slope_z={reading.get('drift_slope_z')}")
            details.append(f"drift_mean_residual={reading.get('drift_mean_residual')}")
        if reading.get("return_to_normal_seconds") is not None:
            details.append(f"return_s={reading.get('return_to_normal_seconds')}")
        return ", ".join(details) or ", ".join(indicators)
    return ", ".join(indicators)


def robust_or_percentile_score(value: float, summary: dict[str, Any], p95: float) -> float:
    spread = float(summary.get("mad", 0.0) or 0.0)
    if spread > EPSILON:
        return robust_z(value, float(summary["median"]), spread)
    if value <= p95:
        return 0.0
    return min(12.0, 4.0 + (value - p95))


def upper_tail_score(value: float, summary: dict[str, Any], p95: float) -> float:
    """Devuelve desviacion solo cuando el valor supera el comportamiento normal.

    En binarios, A_W, C_W, P_on,W, D_on,max y G(t) se interpretan como exceso
    respecto de la base. Un valor menor al centro no debe activar por si solo
    una sospecha de rafaga o permanencia.
    """
    center = float(summary.get("median", 0.0) or 0.0)
    spread = float(summary.get("mad", 0.0) or 0.0)
    if spread > EPSILON:
        return max(0.0, (value - center) / (1.4826 * spread + EPSILON))
    if value <= p95:
        return 0.0
    return min(12.0, 4.0 + (value - p95))


def component_score(value: float, tau: float) -> float:
    """Normaliza un indicador a 0..1: s=min(1, indicador/tau)."""
    safe_tau = max(float(tau), EPSILON)
    return max(0.0, min(1.0, float(value) / safe_tau))


def weighted_component_score(components: dict[str, float], weights: dict[str, float]) -> float:
    """Combina componentes normalizados como promedio ponderado acotado."""
    total_weight = sum(max(float(weight), 0.0) for weight in weights.values())
    if total_weight <= 0:
        return 0.0
    weighted_sum = 0.0
    for name, value in components.items():
        weighted_sum += max(float(weights.get(name, 0.0)), 0.0) * max(0.0, min(1.0, float(value)))
    return max(0.0, min(1.0, weighted_sum / total_weight))


def local_slope_per_hour(values: deque[tuple[pd.Timestamp, float]]) -> float:
    """Calcula beta(W_t) como pendiente local de la serie dentro de la ventana."""
    first_time = values[0][0]
    x = [(timestamp - first_time).total_seconds() / 3600.0 for timestamp, _ in values]
    y = [float(value) for _, value in values]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((item - x_mean) ** 2 for item in x)
    if denominator <= 0:
        return 0.0
    return sum((x[item] - x_mean) * (y[item] - y_mean) for item in range(len(x))) / denominator


def update_recent_binary_pattern(
    recent_times: deque[pd.Timestamp],
    timestamp: pd.Timestamp,
    active: bool,
    lookback_seconds: int,
) -> int:
    if active:
        recent_times.append(timestamp)
    keep_from = timestamp - pd.Timedelta(seconds=lookback_seconds)
    while recent_times and recent_times[0] < keep_from:
        recent_times.popleft()
    return len(recent_times)


def binary_gap_limit_for_hour(summary: dict[str, Any], timestamp: pd.Timestamp) -> float:
    global_p95 = float(summary["percentiles"].get("gap_p95", 0.0) or 0.0)
    hourly = summary.get("hourly", {})
    hour_summary = hourly.get(timestamp.hour) or hourly.get(str(timestamp.hour))
    if hour_summary:
        gap_summary = hour_summary.get("gap", {})
        hourly_p95 = float(gap_summary.get("p95", gap_summary.get("upper", global_p95)) or 0.0)
        return max(hourly_p95 * 1.6, 15 * 60.0)
    return max(global_p95 * 2.0, 60 * 60.0)


def unexpected_binary_activation_score(summary: dict[str, Any], metric: pd.Series, timestamp: pd.Timestamp) -> float:
    activations = float(metric["activations"])
    active_ratio = float(metric["active_ratio"])
    active_seconds = float(metric["active_seconds"])
    if activations <= 0 and active_ratio <= 0:
        return 0.0

    hourly = summary.get("hourly", {})
    hour_summary = hourly.get(timestamp.hour) or hourly.get(str(timestamp.hour))
    if not hour_summary:
        return 0.0

    activation_p95 = float(hour_summary.get("activations", {}).get("p95", 0.0) or 0.0)
    ratio_p95 = float(hour_summary.get("active_ratio", {}).get("p95", 0.0) or 0.0)
    seconds_p95 = float(hour_summary.get("active_seconds", {}).get("p95", 0.0) or 0.0)
    hour_is_usually_quiet = activation_p95 <= 0.0 and ratio_p95 <= 0.02 and seconds_p95 <= 1.0
    if not hour_is_usually_quiet:
        return 0.0

    # La hora suele ser silenciosa: una activacion breve entrega evidencia
    # inicial; multiples activaciones o mayor proporcion activa elevan el score.
    activation_component = min(3.0, activations)
    ratio_component = min(3.0, active_ratio / 0.20) if active_ratio > 0 else 0.0
    duration_component = min(2.0, active_seconds / 20.0) if active_seconds > 0 else 0.0
    return activation_component + ratio_component + duration_component


def suppress_weak_binary_event(
    indicators: list[str],
    score: float,
    metric: pd.Series,
    threshold: float,
) -> tuple[list[str], float]:
    activations = float(metric["activations"])
    changes = float(metric["state_changes"])
    active_ratio = float(metric["active_ratio"])
    active_seconds = float(metric["active_seconds"])
    gap = float(metric["gap"])

    # Una puerta o PIR puede tener una activacion normal aislada. No debe quedar
    # como sospecha si la evidencia completa es solo A=1, C=1 y una ventana corta
    # con P=1.0 producto de muestreo escaso.
    isolated_normal_event = activations <= 1 and changes <= 1 and active_seconds <= 60 and gap < 180
    weak_indicators = {"activation_frequency", "state_changes", "state_persistence", "max_active_duration"}
    if isolated_normal_event and set(indicators).issubset(weak_indicators):
        return [], min(score, threshold * 0.35)

    # Mantener persistencia solo cuando el estado activo dura varias ventanas o
    # aparece junto con otra senal fuerte; evita marcar una muestra "on" normal.
    if "state_persistence" in indicators and active_ratio >= 1 and activations <= 1 and changes <= 1:
        indicators = [item for item in indicators if item != "state_persistence"]
        score *= 0.65

    return indicators, score


def pattern_from_indicators(sensor_kind: str, indicators: list[str]) -> str:
    if not indicators:
        return "score_threshold"
    indicator_set = set(indicators)
    if sensor_kind == "continuous":
        if "drift_slope" in indicator_set:
            return "drift"
        if "iqr" in indicator_set or "hampel" in indicator_set:
            return "isolated_spike"
        if "rate_change" in indicator_set:
            return "abrupt_change"
        if "cusum" in indicator_set or "ewma" in indicator_set:
            return "persistent_shift"
        if "window_variability" in indicator_set:
            return "noise_or_stuck_value"
    if sensor_kind == "binary":
        if "sustained_high_active_ratio" in indicator_set:
            return "sustained_high_active_ratio"
        if "sustained_activation_burst" in indicator_set:
            return "sustained_activation_burst"
        if "sustained_state_persistence" in indicator_set:
            return "sustained_state_persistence"
        if "sustained_absence_gap" in indicator_set:
            return "sustained_absence_gap"
        if "unexpected_activation_time" in indicator_set:
            return "unexpected_activation_time"
        if "activation_frequency" in indicator_set or "state_changes" in indicator_set:
            return "activation_burst"
        if "absence_gap" in indicator_set:
            return "absence_gap"
        if "state_persistence" in indicator_set:
            return "state_persistence"
    if sensor_kind == "multistate":
        if "rare_transition" in indicator_set:
            return "rare_transition"
        if "state_duration" in indicator_set:
            return "state_duration"
        if "state_distribution" in indicator_set:
            return "state_distribution"
    return indicators[0]


def first_detection_delay_seconds(
    results: list[dict[str, Any]],
    ground_truth: dict[tuple[str, str], dict[str, Any]],
) -> float | None:
    fault_starts: dict[str, pd.Timestamp] = {}
    for (timestamp, entity_id), truth in ground_truth.items():
        if truth.get("label") == "normal":
            continue
        ts = pd.to_datetime(timestamp, errors="coerce")
        if pd.isna(ts):
            continue
        if entity_id not in fault_starts or ts < fault_starts[entity_id]:
            fault_starts[entity_id] = ts
    delays = []
    for result in sorted(results, key=lambda item: item["timestamp"]):
        if not result.get("is_suspicious"):
            continue
        entity_id = result["entity_id"]
        if entity_id not in fault_starts:
            continue
        detected_at = pd.to_datetime(result["timestamp"], errors="coerce")
        if pd.isna(detected_at):
            continue
        delay = (detected_at - fault_starts[entity_id]).total_seconds()
        if delay >= 0:
            delays.append(delay)
    return round(min(delays), 3) if delays else None


def score_to_unit_interval(raw_score: float, threshold: float) -> float:
    # La tesis define P_i^sos(t)=S_i(t)/(S_i(t)+tau_i). Aqui raw_score es S_i(t)
    # ya combinado desde componentes normalizados, y threshold representa tau_i.
    safe_threshold = max(threshold, EPSILON)
    return max(0.0, min(1.0, raw_score / (raw_score + safe_threshold)))


def threshold_for_score(summary: dict[str, Any], suspicion_threshold: float) -> float:
    base_threshold = float(summary.get("threshold", 1.0) or 1.0)
    unit_threshold = clamp_unit_interval(suspicion_threshold)
    # Inversa de score_to_unit_interval:
    # score = raw / (raw + base_threshold)
    # raw = score * base_threshold / (1 - score)
    return unit_threshold * base_threshold / max(1.0 - unit_threshold, EPSILON)


def clamp_unit_interval(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def compact_baseline(summary: dict[str, Any]) -> dict[str, Any]:
    allowed = {"count", "median", "mad", "q1", "q3", "lower", "upper", "threshold", "states", "state_set"}
    return {key: value for key, value in summary.items() if key in allowed}
