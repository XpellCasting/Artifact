"""Construcción de ventanas causales para preparación e inferencia."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import math
from statistics import mean, pstdev

from .config import parse_utc
from .features import dtw_summary, entropy, histogram_id, state_id, _finite_number
from ..Estados_Validos import valid_state


def build_causal_sample(event_provider, target_time, preprocessor: dict, graph: dict, config,
                        labels: dict[str, int] | None = None) -> dict:
    """Construye X[T,N,F] con eventos <= cada paso, incluso en objetivos no redondeados.

    event_provider(start, end, entity_ids) retorna dicts con timestamp, entity_id,
    state y opcional source_row. La misma función sirve para lotes y sesión online.
    """
    target = parse_utc(target_time) if isinstance(target_time, str) else target_time
    if target.tzinfo is None:
        raise ValueError("El objetivo necesita zona temporal")
    entities = preprocessor["entity_ids"]
    first = target - timedelta(seconds=config.grid_seconds * (config.sequence_steps - 1))
    start = first - timedelta(seconds=config.feature_window_seconds)
    rows = event_provider(start, target, entities)
    grouped = defaultdict(list)
    for row in rows:
        entity = row.get("entity_id")
        if entity not in preprocessor["entities"]:
            continue
        at = parse_utc(row["timestamp"])
        if start <= at <= target:
            grouped[entity].append((at, int(row.get("source_row") or 0), str(row["state"])))
    for entity in entities:
        grouped[entity].sort(key=lambda value: (value[0], value[1]))
    x, states, present = [], [], []
    for step in range(config.sequence_steps):
        at = first + timedelta(seconds=config.grid_seconds * step)
        tick_x, tick_states, tick_present = [], [], []
        hour = (at.hour * 3600 + at.minute * 60 + at.second + at.microsecond / 1e6) / 86400
        for entity in entities:
            info = preprocessor["entities"][entity]
            window_start = at - timedelta(seconds=config.feature_window_seconds)
            past = [r for r in grouped[entity] if r[0] <= at]
            recent = [r for r in past if r[0] > window_start]
            latest = past[-1] if past else None
            if latest and (at - latest[0]).total_seconds() > info["max_age_seconds"]:
                latest = None
            valid_recent = [r for r in recent if valid_state(r[2])]
            kind = info["kind"]
            values = [float(r[2]) for r in valid_recent if _finite_number(r[2])] if kind == "continuous" else []
            current_valid = bool(latest and valid_state(latest[2]) and
                                 (kind != "continuous" or _finite_number(latest[2])))
            scale = info.get("scale", 1.0)
            center = info.get("median", 0.0)
            current = (float(latest[2]) - center) / scale if current_valid and kind == "continuous" else 0.0
            # An invalid report breaks the valid-value chain.
            prior_valid = None
            if kind == "continuous" and current_valid and len(past) > 1:
                previous = past[-2]
                if valid_state(previous[2]) and _finite_number(previous[2]):
                    prior_valid = float(previous[2])
            delta = (float(latest[2]) - prior_valid) / scale if prior_valid is not None else 0.0
            scaled = [(v - center) / scale for v in values]
            if config.use_entropy:
                if kind == "continuous":
                    bins = [histogram_id(v, info["histogram_edges"]) for v in values]
                    ent, ent_ok = entropy(bins, max(1, len(info["histogram_edges"]) + 1))
                else:
                    bins = [state_id(r[2], info) for r in valid_recent]
                    ent, ent_ok = entropy(bins, max(1, len(info.get("vocab", {})) + 1))
            else:
                ent, ent_ok = 0.0, False
            complete_dtw_window = len(valid_recent) == len(recent)
            if config.use_dtw and kind == "continuous" and len(scaled) >= 3 and complete_dtw_window:
                dmin, dmed, sim, dtw_ok = dtw_summary(scaled, info.get("normal_references", []), config.dtw_band_fraction)
            else:
                dmin, dmed, sim, dtw_ok = 0.0, 0.0, 0.0, False
            previous_at = at - timedelta(seconds=config.grid_seconds)
            tick_x.append([current, delta, mean(scaled) if scaled else 0.0,
                           min(scaled) if scaled else 0.0, max(scaled) if scaled else 0.0,
                           pstdev(scaled) if len(scaled) > 1 else 0.0,
                           float(len(recent)), float(current_valid),
                           float(any(previous_at < r[0] <= at for r in recent)),
                           float(any(not valid_state(r[2]) for r in recent)),
                           (at - latest[0]).total_seconds() if latest else 0.0,
                           math.sin(2 * math.pi * hour), math.cos(2 * math.pi * hour),
                           ent, float(ent_ok), dmin, dmed, sim, float(dtw_ok)])
            tick_states.append(state_id(latest[2] if latest else None, info) if kind != "continuous" else 0)
            # An explicit invalid report is informative. Silent absence is not.
            tick_present.append(bool(latest))
        x.append(tick_x); states.append(tick_states); present.append(tick_present)
    y = [labels.get(entity, -1) if labels else -1 for entity in entities]
    return {"X": x, "state_ids": states, "node_present": present,
            "adjacency": graph["adjacency"], "y": y,
            "label_mask": [value >= 0 for value in y],
            "metadata": {"target_time": target.isoformat(), "context_start": start.isoformat(),
                         "entity_ids": entities}}
