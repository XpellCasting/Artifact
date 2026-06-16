from __future__ import annotations

from collections import Counter, defaultdict
from math import log, sqrt
from statistics import median
from typing import Iterable


EPSILON = 1e-9


def to_float(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mad(values: list[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return median([abs(value - center) for value in values])


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def robust_z(value: float, center: float, spread: float) -> float:
    # Definicion robusta usada en la tesis: 1.4826 escala el MAD para hacerlo
    # comparable con la desviacion estandar bajo una distribucion normal.
    return abs(value - center) / (1.4826 * spread + EPSILON)


def summarize_numeric(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "median": 0.0,
            "mad": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "stdev": 0.0,
        }
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    iqr = q3 - q1
    return {
        "count": len(values),
        "median": float(median(values)),
        "mad": float(mad(values)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower": float(q1 - 1.5 * iqr),
        "upper": float(q3 + 1.5 * iqr),
        "stdev": float(stdev(values)),
    }


def transition_counts(states: Iterable[str]) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    previous: str | None = None
    for state in states:
        if previous is not None:
            counts[previous][state] += 1
        previous = state
    return counts


def transition_rarity(
    counts: dict[str, Counter],
    previous: str,
    current: str,
    all_states: set[str],
    smoothing: float = 1.0,
) -> float:
    options = max(len(all_states), 1)
    total = sum(counts.get(previous, Counter()).values())
    observed = counts.get(previous, Counter()).get(current, 0)
    probability = (observed + smoothing) / (total + smoothing * options)
    return -log(probability + EPSILON)
