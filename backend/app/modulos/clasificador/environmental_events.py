"""Explicit mixed-sensor environmental interventions on immutable observations.

Relations and state meanings are supplied by the experiment author, never inferred
from labels or arbitrary category IDs. Only changed observations become targets.
"""
from bisect import bisect_left, bisect_right
from datetime import timedelta
import math

from app.modulos.clasificador.config import parse_utc


def validate_events(events, entities=None):
    if not isinstance(events, list):
        raise ValueError("environmental_events debe ser una lista")
    names = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Cada evento debe ser un objeto")
        name = event.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Cada evento requiere un nombre único")
        names.add(name)
        if not isinstance(event.get("relation_basis"), str) or not event["relation_basis"].strip():
            raise ValueError(f"{name}: documenta la relación física o funcional")
        if type(event.get("episodes", 1)) is not int or event.get("episodes", 1) < 1:
            raise ValueError(f"{name}: episodes debe ser positivo")
        for key in ("durations_seconds", "intensities"):
            values = event.get(key, [1.] if key == "intensities" else [])
            if not isinstance(values, list) or not values or any(
                type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values
            ):
                raise ValueError(f"{name}: {key} debe contener valores positivos finitos")
        responses = event.get("responses", [])
        if not isinstance(responses, list) or any(not isinstance(r, dict) or not isinstance(r.get("entity"), str) for r in responses) or len(responses) < 2 or len({r.get("entity") for r in responses}) != len(responses):
            raise ValueError(f"{name}: declara al menos dos entidades distintas")
        for response in responses:
            entity, kind = response.get("entity"), response.get("kind")
            if not isinstance(entity, str) or not entity or kind not in {"continuous", "binary", "multistate"}:
                raise ValueError(f"{name}: entidad o tipo inválido")
            delay = response.get("delay_seconds", 0)
            if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
                raise ValueError(f"{name}: retraso inválido")
            pattern = response.get("pattern", [])
            if not isinstance(pattern, list) or len(pattern) < 2 or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in pattern):
                raise ValueError(f"{name}: cada patrón necesita al menos dos puntos")
            positions = [p[0] for p in pattern]
            if any(type(p) not in (float, int) or not math.isfinite(p) for p in positions) or positions[0] != 0 or positions[-1] != 1 or any(a >= b for a, b in zip(positions, positions[1:])):
                raise ValueError(f"{name}: el patrón debe avanzar de 0 a 1")
            values = [p[1] for p in pattern]
            if kind == "continuous":
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or not any(values):
                    raise ValueError(f"{name}: desplazamientos continuos inválidos")
            elif any(not isinstance(v, str) for v in values):
                raise ValueError(f"{name}: los estados deben ser cadenas explícitas")
            if entities is not None:
                info = entities.get(entity)
                if info is None or info["kind"] != kind:
                    raise ValueError(f"{name}: {entity} ausente o tipo incompatible con el ajuste")
                if kind != "continuous" and (any(v not in info.get("vocab", {}) for v in values) or kind == "binary" and len(info.get("vocab", {})) != 2):
                    raise ValueError(f"{name}: estados fuera del vocabulario de {entity}")


def response_value(response, state, progress, intensity, limits):
    pattern = response["pattern"]
    index = max(i for i, point in enumerate(pattern) if point[0] <= progress)
    if response["kind"] != "continuous":
        return pattern[index][1]
    left, value = pattern[index]
    if index + 1 < len(pattern):
        right, following = pattern[index + 1]
        value += (following - value) * (progress - left) / (right - left)
    value = float(state) + intensity * value
    low, high = limits
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    if not math.isfinite(value):
        raise ValueError("Intervención continua no finita")
    return state if value == float(state) else format(value, ".12g")


def generate_events(config, prep, grouped, part, name, rng, reserved, earliest, digest):
    events = config.get("environmental_events", [])
    validate_events(events, prep["entities"])
    episodes, overlays, deficits = [], [], []
    context = timedelta(seconds=config["context_seconds"])
    end_partition = parse_utc(config["partitions"][part][1])
    for event in events:
        responses = event["responses"]
        times = {r["entity"]: [parse_utc(row["t"]) for row in grouped[r["entity"]]] for r in responses}
        generated = 0
        for repetition in range(event.get("episodes", 1)):
            duration = event["durations_seconds"][repetition % len(event["durations_seconds"])]
            intensities = event.get("intensities", [1.])
            intensity = intensities[repetition % len(intensities)]
            anchors = sorted({t for t in times[responses[0]["entity"]] if t >= earliest})
            rng.shuffle(anchors)
            for start in anchors:
                end = start + timedelta(seconds=duration + max(r.get("delay_seconds", 0) for r in responses))
                if end >= end_partition:
                    continue
                chosen, changes = [], []
                for response in responses:
                    entity = response["entity"]
                    ts, rows = times[entity], grouped[entity]
                    if not ts or start - ts[0] < context or any(start < b + context and end > a - context for a, b in reserved[entity]):
                        break
                    active = start + timedelta(seconds=response.get("delay_seconds", 0))
                    stop = active + timedelta(seconds=duration)
                    left, right = bisect_left(ts, active), bisect_right(ts, stop)
                    context_left, context_right = bisect_left(ts, start-context), bisect_right(ts, end)
                    if right-left < 3 or context_right >= len(rows) or not rows[context_right]["eligible"] or any(not row["eligible"] for row in rows[context_left:context_right]):
                        break
                    if (ts[right-1]-ts[left]).total_seconds() < duration*.5:
                        break
                    member_changes = []
                    for row, at in zip(rows[left:right], ts[left:right]):
                        after = response_value(response, row["state"], (at-active).total_seconds()/duration,
                                               intensity, config.get("sensors", {}).get(entity, {}).get("limits", [None, None]))
                        if after != row["state"]:
                            member_changes.append({"row_id": row["row_id"], "entity": entity, "timestamp": row["t"],
                                "before": row["state"], "after": after, "kind": "environmental_anomaly",
                                "label": "environmental_anomaly", "certainty": "synthetic_controlled",
                                "baseline_certainty": row["certainty"], "real_fault_confirmed": False})
                    if not member_changes:
                        break
                    chosen.append(entity)
                    changes.extend(member_changes)
                if len(chosen) != len(responses):
                    continue
                episode_id = digest([config["seed"], part, name, event, repetition, start.isoformat()])[:32]
                for change in changes:
                    change["intervention_id"] = episode_id
                overlays.extend(changes)
                for entity in chosen:
                    reserved[entity].append((start, end))
                episodes.append({"intervention_id": episode_id, "kind": "environmental_anomaly",
                    "event_name": event["name"], "relation_basis": event["relation_basis"], "entities": chosen,
                    "responses": responses, "start": start.isoformat(), "end": end.isoformat(),
                    "duration_seconds": duration, "intensity": intensity, "changed_rows": len(changes),
                    "identifiable_from": None, "identifiability_reason": "Intervención sintética explícita; no confirma un evento real."})
                generated += 1
                break
        if generated < event.get("episodes", 1):
            deficits.append({"kind": "environmental_anomaly", "event_name": event["name"],
                "requested": event.get("episodes", 1), "generated": generated,
                "missing": event.get("episodes", 1)-generated,
                "reason": "insufficient_eligible_context_coverage_or_effective_changes"})
    return episodes, overlays, deficits
