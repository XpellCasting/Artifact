"""Measurements of the output readings and their reference labels."""
from array import array
from collections import defaultdict, deque
from datetime import datetime, timezone
from math import floor, isclose, isfinite

from app.flow.metrics import CLASSES, DETECTION_STATES
from app.flow.result_spool import ALERT_DECISIONS
from app.modulos.clasificador import LABELS

MEASUREMENTS_VERSION = 1
MISSING_LATENCY = "La ejecución no registró latencia por lectura."
MISSING_DAYS = "No hay días con lecturas etiquetadas como normales o ambientales."
Brier_CLASSES = dict(zip(CLASSES, LABELS))


def distribution(values, reason):
    if not values:
        return {"count": 0, "median": None, "p95": None, "reason": reason}
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        left = floor(position)
        right = min(left + 1, len(ordered) - 1)
        return round(ordered[left] + (position - left) * (ordered[right] - ordered[left]), 6)

    return {"count": len(ordered), "median": percentile(.5), "p95": percentile(.95), "reason": None}


def _utc(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class RunMeasurements:
    def __init__(self, *, latency_limit=None):
        self.latencies = array("d") if latency_limit is None else deque(maxlen=latency_limit)
        self.delays = array("d")
        self.episodes = 0
        self.active_faults = {}
        self.timestamp = None
        self.pending = []
        self.episode_error = None
        self.day_error = None
        self.days = defaultdict(lambda: [0, 0, 0])
        self.brier = defaultdict(lambda: [0.0, 0])

    def observe(self, item, expected, fault_type):
        latency = item.get("latency_ms")
        if type(latency) in (float, int) and isfinite(latency) and latency >= 0:
            self.latencies.append(latency)

        entity = item.get("entity_id")
        timestamp = _utc(item.get("timestamp"))
        decision = item.get("final_decision")
        if timestamp is None or not entity:
            if expected == "Falla":
                self.episode_error = "Hay lecturas de falla sin sensor o fecha de referencia."
            if expected in ("normal", "anomalia"):
                self.day_error = "Hay lecturas normales o ambientales sin sensor o fecha."
        else:
            self._observe_episode(item, entity, timestamp, expected, fault_type, decision)
            if expected in ("normal", "anomalia"):
                day = self.days[(entity, timestamp.date())]
                day[0] += 1
                day[1] += (expected == "normal" and decision in ALERT_DECISIONS
                           or expected == "anomalia" and decision == "fault_compatible")
                day[2] += decision == "fault_compatible"

        if expected in Brier_CLASSES:
            self._observe_probabilities(item.get("evidence") or [], Brier_CLASSES[expected])

    def _observe_probabilities(self, evidence, target):
        if not isinstance(evidence, list):
            return
        seen = set()
        for entry in evidence:
            if not isinstance(entry, dict):
                continue
            node = entry.get("node_id")
            data = entry.get("data")
            if not isinstance(node, str) or not node or node in seen or entry.get("module_type") not in (None, "classifier"):
                continue
            scores = data.get("class_scores") if isinstance(data, dict) else None
            if not isinstance(scores, dict) or set(scores) != set(LABELS):
                continue
            if not all(type(value) in (float, int) and isfinite(value) and 0 <= value <= 1 for value in scores.values()):
                continue
            if not isclose(sum(scores.values()), 1, abs_tol=1e-4):
                continue
            seen.add(node)
            result = self.brier[node]
            result[0] += sum((scores[label] - (label == target)) ** 2 for label in LABELS)
            result[1] += 1

    def _observe_episode(self, item, entity, timestamp, expected, fault_type, decision):
        if self.episode_error:
            return
        if self.timestamp is not None and timestamp != self.timestamp:
            if timestamp < self.timestamp:
                self.episode_error = "El orden guardado de las lecturas no permite reconstruir sus episodios."
                return
            self._flush_episodes()
        self.timestamp = timestamp
        raw_index = item.get("source_raw_index")
        self.pending.append((raw_index if type(raw_index) is int else 0, entity, expected, fault_type, decision))

    def _flush_episodes(self):
        # Replay can append an unrouted reading after other rows at the same timestamp.
        for _, entity, expected, fault_type, decision in sorted(self.pending, key=lambda row: row[0]):
            if expected != "Falla":
                self.active_faults.pop(entity, None)
                continue
            active = self.active_faults.get(entity)
            if active is None or active[0] != fault_type:
                active = (fault_type, self.timestamp, False)
                self.active_faults[entity] = active
                self.episodes += 1
            if not active[2] and DETECTION_STATES.get(decision) == "detected":
                self.delays.append((self.timestamp - active[1]).total_seconds())
                self.active_faults[entity] = (active[0], active[1], True)
        self.pending.clear()

    def finalize(self, supervised_reason=None):
        self._flush_episodes()
        episode_error = supervised_reason or self.episode_error
        day_error = supervised_reason or self.day_error
        days = [] if day_error else [
            {"sensor": sensor, "day": day.isoformat(), "valid_readings": counts[0],
             "false_alerts": counts[1], "false_fault_alerts": counts[2]}
            for (sensor, day), counts in sorted(self.days.items())
        ]
        false_alerts = sum(row["false_alerts"] for row in days)
        detected = len(self.delays)
        return {
            "version": MEASUREMENTS_VERSION,
            "latency_ms": distribution(self.latencies, MISSING_LATENCY),
            "fault_delay_seconds": {
                **distribution([] if episode_error else self.delays, episode_error or (
                    "No se detectó ningún episodio de falla." if self.episodes else "No hay episodios de falla etiquetados."
                )),
                "episodes": None if episode_error else self.episodes,
                "detected": None if episode_error else detected,
                "missed": None if episode_error else self.episodes - detected,
            },
            "false_alerts_by_sensor_day": {
                "days": days, "count": None if day_error else false_alerts,
                "fault_count": None if day_error else sum(row["false_fault_alerts"] for row in days),
                "mean": false_alerts / len(days) if days else None,
                "reason": day_error or (None if days else MISSING_DAYS),
            },
            "brier": {
                "by_node": {node: {"value": total / count, "count": count}
                            for node, (total, count) in sorted(self.brier.items())} if not supervised_reason else {},
                "reason": supervised_reason or (None if self.brier else "No hay probabilidades de clasificador para lecturas etiquetadas en Salida."),
            },
        }
