"""Transformaciones causales congeladas; no dependen de PyTorch."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import math
from statistics import median

from ..Estados_Validos import valid_state

MISSING, INVALID, OOV = 0, 1, 2


def entropy(values, alphabet_size: int) -> tuple[float, bool]:
    if len(values) < 3:
        return 0.0, False
    counts = Counter(values)
    total = len(values)
    h = -sum((count / total) * math.log(count / total) for count in counts.values())
    return h / math.log(alphabet_size) if alphabet_size > 1 else 0.0, True


def dtw_distance(left: list[float], right: list[float], band_fraction: float = .1) -> float | None:
    """DTW absoluto con extremos fijados; distancia por largo de camino."""
    if not left or not right or any(not math.isfinite(x) for x in left + right):
        return None
    n, m = len(left), len(right)
    band = math.ceil(band_fraction * max(n, m))
    if abs(n - m) > band:
        return None
    prev = {}
    for i in range(n):
        curr = {}
        for j in range(max(0, i - band), min(m, i + band + 1)):
            candidates = []
            if i == j == 0:
                candidates.append((0.0, 0, 0))
            if i and j and j - 1 in prev:
                cost, length = prev[j - 1]
                candidates.append((cost, length, 0))
            if i and j in prev:
                cost, length = prev[j]
                candidates.append((cost, length, 1))
            if j and j - 1 in curr:
                cost, length = curr[j - 1]
                candidates.append((cost, length, 2))
            if candidates:
                cost, length, _ = min(candidates)
                curr[j] = (cost + abs(left[i] - right[j]), length + 1)
        prev = curr
    return prev[m - 1][0] / prev[m - 1][1] if m - 1 in prev else None


def _quantiles(values):
    values = sorted(values)
    def q(p):
        index = (len(values) - 1) * p
        lower, upper = math.floor(index), math.ceil(index)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)
    return q(.25), q(.5), q(.75)


def fit_preprocessor(train_fit_events: list[dict], annotations: list[dict], config,
                     entity_kinds: dict[str, str] | None = None) -> dict:
    """Ajusta solo los eventos entregados; el llamador debe certificar train_fit."""
    if entity_kinds is None:
        raise ValueError("Se requieren tipos de entidad declarados por la fuente")
    if len(entity_kinds) > config.max_nodes:
        raise ValueError("Demasiadas entidades")
    result = {"schema_version": config.schema_version, "entity_ids": sorted(entity_kinds),
              "entity_kinds": dict(entity_kinds), "entities": {},
              "feature_names": ["value", "delta", "mean", "min", "max", "std", "report_count",
                                "valid_value", "new_report", "invalid_report", "age_seconds",
                                "hour_sin", "hour_cos", "entropy", "entropy_available",
                                "dtw_min", "dtw_median", "dtw_similarity", "dtw_available"]}
    grouped = {entity: [] for entity in entity_kinds}
    for row in train_fit_events:
        entity = row["entity_id"]
        if entity in grouped and valid_state(str(row["state"])):
            grouped[entity].append(str(row["state"]))
    for entity in result["entity_ids"]:
        kind = entity_kinds[entity]
        values = grouped[entity]
        info = {"kind": kind, "constant": False, "normal_references": [],
                "max_age_seconds": config.feature_window_seconds}
        if kind == "continuous":
            nums = [float(x) for x in values if _finite_number(x)]
            if nums:
                q1, center, q3 = _quantiles(nums)
                info.update(median=center, scale=(q3 - q1) or 1.0,
                            constant=q3 == q1, histogram_edges=([] if min(nums) == max(nums) else [
                                min(nums) + (max(nums) - min(nums)) * i / config.entropy_bins
                                for i in range(1, config.entropy_bins)]))
            else:
                info.update(median=0.0, scale=1.0, histogram_edges=[])
        else:
            counts = Counter(values)
            vocab = sorted(counts, key=lambda key: (-counts[key], key))[:64]
            info.update(vocab={value: i + 3 for i, value in enumerate(vocab)},
                        overflow=sum(counts[value] for value in counts if value not in vocab))
        result["entities"][entity] = info
    # DTW references are selected only from reviewed normal windows supplied
    # with this fit call. The caller must pass train_fit annotations only.
    for entity in result["entity_ids"]:
        info = result["entities"][entity]
        if info["kind"] != "continuous":
            continue
        normal = sorted((a for a in annotations if a.get("entity_id") == entity
                         and a.get("label") == "normal"
                         and a.get("status") in {"reviewed", "verified"}),
                        key=lambda a: (a["start"], a.get("annotation_id", "")))
        events = sorted((e for e in train_fit_events if e["entity_id"] == entity),
                        key=lambda e: e["timestamp"])
        eligible, previous_end = [], ""
        for ann in normal:
            if ann["start"] <= previous_end:
                continue
            values = [float(e["state"]) for e in events
                      if ann["start"] <= e["timestamp"] <= ann["end"]
                      and valid_state(str(e["state"])) and _finite_number(e["state"])]
            if len(values) >= 3:
                eligible.append([(v - info["median"]) / info["scale"] for v in values])
                previous_end = ann["end"]
        if eligible:
            picks = [eligible[0], eligible[len(eligible) // 2], eligible[-1]]
            info["normal_references"] = list(dict.fromkeys(tuple(x) for x in picks))[:config.dtw_reference_count]
            info["normal_references"] = [list(x) for x in info["normal_references"]]
    return result


def _finite_number(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def state_id(value: str | None, info: dict) -> int:
    if value is None:
        return MISSING
    if not valid_state(value):
        return INVALID
    return info.get("vocab", {}).get(value, OOV)


def histogram_id(value: float, edges: list[float]) -> int:
    return bisect_right(edges, value)


def dtw_summary(sequence: list[float], references: list[list[float]], band_fraction=.1):
    distances = [dtw_distance(sequence, ref, band_fraction) for ref in references]
    distances = sorted(d for d in distances if d is not None)
    if not distances:
        return (0.0, 0.0, 0.0, False)
    return (distances[0], median(distances), 1 / (1 + distances[0]), True)
