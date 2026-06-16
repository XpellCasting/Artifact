from __future__ import annotations

from bisect import bisect_left, bisect_right

import pandas as pd


BINARY_FEATURE_COLUMNS = [
    "activations",
    "state_changes",
    "active_seconds",
    "active_ratio",
    "max_active_seconds",
    "gap",
    "mean_activation_interval",
    "start",
    "window_start",
]


def build_binary_temporal_features(rows: pd.DataFrame, window_seconds: int = 60, step_seconds: int = 10) -> pd.DataFrame:
    """Calcula las variables binarias temporales antes del filtro estadistico.

    Esta fase transforma los estados crudos en las magnitudes de la tesis:
    A_W, C_W, T_on,W, P_on,W, D_on,max y gap desde la ultima activacion. El
    filtro luego compara estas features contra la base de normalidad, sin volver
    a estimar el comportamiento historico.
    """
    ordered = rows.sort_values("timestamp_dt").reset_index(drop=True)
    if ordered.empty:
        return pd.DataFrame(columns=BINARY_FEATURE_COLUMNS)

    values = pd.to_numeric(ordered["value"], errors="coerce").fillna(0).astype(int).tolist()
    timestamps = ordered["timestamp_dt"].tolist()
    first_timestamp = timestamps[0]
    times = [(timestamp - first_timestamp).total_seconds() for timestamp in timestamps]
    transition_times = []
    activation_times_all = []
    cumulative_active = [0.0] * len(times)

    # Los acumulados permiten calcular T_on,W como integral de estado activo:
    # active(t2) - active(t1), evitando recorrer el DataFrame completo en cada
    # ventana.
    for index in range(1, len(times)):
        cumulative_active[index] = cumulative_active[index - 1] + (times[index] - times[index - 1]) * values[index - 1]
        if values[index] != values[index - 1]:
            transition_times.append(times[index])
            if values[index - 1] == 0 and values[index] == 1:
                activation_times_all.append(times[index])

    metrics = []
    current_start = 0.0
    end_time = times[-1]

    while current_start <= end_time:
        current_end = current_start + float(window_seconds)
        start_pos = bisect_left(times, current_start)
        end_pos = bisect_left(times, current_end)
        if start_pos >= end_pos:
            current_start += float(step_seconds)
            continue

        change_left = bisect_left(transition_times, current_start)
        change_right = bisect_left(transition_times, current_end)
        activation_left = bisect_left(activation_times_all, current_start)
        activation_right = bisect_left(activation_times_all, current_end)
        activation_times = activation_times_all[activation_left:activation_right]
        active_seconds = active_integral_at(current_end, times, values, cumulative_active) - active_integral_at(current_start, times, values, cumulative_active)
        max_active_seconds = longest_active_run(current_start, current_end, times, values)
        window_last_time = times[end_pos - 1]
        last_activation_index = bisect_right(activation_times_all, window_last_time) - 1
        gap = window_last_time - activation_times_all[last_activation_index] if last_activation_index >= 0 else window_last_time - times[0]
        intervals = [
            current - previous_activation
            for previous_activation, current in zip(activation_times, activation_times[1:])
        ]

        metrics.append(
            {
                "activations": float(activation_right - activation_left),
                "state_changes": float(change_right - change_left),
                "active_seconds": float(active_seconds),
                "active_ratio": float(active_seconds / window_seconds),
                "max_active_seconds": float(max_active_seconds),
                "gap": float(gap),
                "mean_activation_interval": float(sum(intervals) / len(intervals)) if intervals else float("nan"),
                "start": float(start_pos),
                "window_start": (first_timestamp + pd.Timedelta(seconds=current_start)).isoformat(),
            }
        )
        current_start += float(step_seconds)

    return pd.DataFrame(metrics, columns=BINARY_FEATURE_COLUMNS)


def active_integral_at(moment: float, times: list[float], values: list[int], cumulative_active: list[float]) -> float:
    if moment <= times[0]:
        return 0.0

    index = min(max(bisect_right(times, moment) - 1, 0), len(times) - 1)
    return cumulative_active[index] + max(moment - times[index], 0.0) * values[index]


def longest_active_run(window_start: float, window_end: float, times: list[float], values: list[int]) -> float:
    """Devuelve D_on,max dentro de la ventana ya compactada."""
    max_active = 0.0
    current_active_run = 0.0
    index = max(bisect_right(times, window_start) - 1, 0)

    while index < len(times) and times[index] < window_end:
        segment_start = max(times[index], window_start)
        next_time = times[index + 1] if index + 1 < len(times) else window_end
        segment_end = min(next_time, window_end)

        if segment_end > segment_start and values[index] == 1:
            seconds = max(segment_end - segment_start, 0.0)
            current_active_run += seconds
            max_active = max(max_active, current_active_run)
        else:
            current_active_run = 0.0
        index += 1

    return max_active
