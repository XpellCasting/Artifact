from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any

import pandas as pd

from .domain.entities import simulation_entities
from .domain.fault_types import DEFAULT_FAULTS, normalize_faults as normalize_domain_faults
from .fault_injection import FaultInjectorFactory


DEFAULT_ENTITIES = simulation_entities()


@dataclass(frozen=True)
class SimulationResult:
    entities: pd.DataFrame
    states: pd.DataFrame
    reference_states: pd.DataFrame
    event_states: pd.DataFrame
    ground_truth: pd.DataFrame
    config: dict[str, Any]
    faults: list[dict[str, Any]]


def generate_simulation(
    duration_hours: int = 72,
    sample_seconds: int = 10,
    seed: int = 20260614,
    faults: list[dict[str, Any]] | None = None,
    selected_entities: list[str] | None = None,
) -> SimulationResult:
    rng = random.Random(seed)
    duration_hours = max(72, int(duration_hours))
    sample_seconds = max(10, int(sample_seconds))
    faults = normalize_faults(faults if faults is not None else DEFAULT_FAULTS)
    entity_rows = [
        dict(entity)
        for entity in DEFAULT_ENTITIES
        if selected_entities is None or entity["entity_id"] in selected_entities
    ]
    entity_ids = {entity["entity_id"] for entity in entity_rows}
    active_faults = [fault for fault in faults if fault.get("enabled", True) and fault["entity_id"] in entity_ids]

    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=-3)))
    total_steps = int(duration_hours * 3600 / sample_seconds)
    timestamps = [start + timedelta(seconds=sample_seconds * index) for index in range(total_steps)]
    reference_hours = min(duration_hours - 24, max(48, int(duration_hours * 0.60)))
    reference_hours = max(24, (reference_hours // 24) * 24)
    reference_cutoff = int(reference_hours * 3600 / sample_seconds)
    fault_windows = materialize_fault_windows(active_faults, timestamps, reference_cutoff)

    state_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for entity in entity_rows:
        entity_id = entity["entity_id"]
        clean_values = generate_clean_series(entity, timestamps, rng)
        observed_values = list(clean_values)
        truth_by_index: dict[int, dict[str, Any]] = {}

        for fault in fault_windows:
            if fault["entity_id"] != entity_id:
                continue
            apply_fault(entity, observed_values, fault, rng)
            for index in range(fault["start_index"], fault["end_index"] + 1):
                truth_by_index[index] = {
                    "label": fault.get("label", "sensor_fault"),
                    "fault_type": fault["fault_type"],
                    "is_reliable": "false",
                }

        for index, timestamp in enumerate(timestamps):
            observed = observed_values[index]
            clean = clean_values[index]
            truth = truth_by_index.get(index, {"label": "normal", "fault_type": "", "is_reliable": "true"})
            timestamp_iso = timestamp.isoformat()
            row = {
                "timestamp": timestamp_iso,
                "entity_id": entity_id,
                "domain": entity["domain"],
                "state": format_state(observed),
                "attributes_json": "{}",
                "last_changed": timestamp_iso,
                "last_updated": timestamp_iso,
            }
            if observed is not None:
                state_rows.append(row)
                if index < reference_cutoff:
                    reference_rows.append(row)
                else:
                    event_rows.append(row)
            truth_rows.append(
                {
                    "timestamp": timestamp_iso,
                    "entity_id": entity_id,
                    "sensor_kind": entity["sensor_kind"],
                    "clean_state": format_state(clean),
                    "observed_state": "" if observed is None else format_state(observed),
                    "label": truth["label"],
                    "fault_type": truth["fault_type"],
                    "anomaly_type": "",
                    "root_cause": truth["fault_type"],
                    "is_reliable": truth["is_reliable"],
                }
            )

    config = {
        "duration_hours": duration_hours,
        "sample_seconds": sample_seconds,
        "seed": seed,
        "rows": len(state_rows),
        "reference_rows": len(reference_rows),
        "event_rows": len(event_rows),
        "entities": len(entity_rows),
        "reference_hours": reference_hours,
        "event_hours": duration_hours - reference_hours,
        "reference_cutoff_ratio": round(reference_cutoff / max(total_steps, 1), 4),
    }
    return SimulationResult(
        entities=pd.DataFrame(entity_rows),
        states=pd.DataFrame(state_rows),
        reference_states=pd.DataFrame(reference_rows),
        event_states=pd.DataFrame(event_rows),
        ground_truth=pd.DataFrame(truth_rows),
        config=config,
        faults=fault_windows,
    )


def normalize_faults(faults: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for fault in normalize_domain_faults(faults):
        if not fault.get("entity_id") or not fault.get("fault_type"):
            continue
        normalized.append(
            {
                "entity_id": str(fault["entity_id"]),
                "fault_type": str(fault["fault_type"]),
                "enabled": bool(fault.get("enabled", True)),
                "start_ratio": float(fault.get("start_ratio", 0.55)),
                "end_ratio": float(fault.get("end_ratio", 0.75)),
                "label": str(fault.get("label", "sensor_fault")),
                "stuck_value": fault.get("stuck_value"),
            }
        )
    return normalized


def materialize_fault_windows(faults: list[dict[str, Any]], timestamps: list[datetime], reference_cutoff: int) -> list[dict[str, Any]]:
    total = max(len(timestamps) - 1, 1)
    event_span = max(total - reference_cutoff, 1)
    windows = []
    for fault in faults:
        start_index = min(total, max(reference_cutoff, reference_cutoff + int(event_span * fault["start_ratio"])))
        end_index = min(total, max(start_index, reference_cutoff + int(event_span * fault["end_ratio"])))
        item = dict(fault)
        item["start_index"] = start_index
        item["end_index"] = end_index
        item["start"] = timestamps[start_index].isoformat()
        item["end"] = timestamps[end_index].isoformat()
        windows.append(item)
    return windows


def generate_clean_series(entity: dict[str, Any], timestamps: list[datetime], rng: random.Random) -> list[Any]:
    entity_id = entity["entity_id"]
    if entity_id == "sensor.living_temperature":
        return [round(21.5 + 1.6 * daily_wave(ts, -3) + rng.gauss(0, 0.18), 2) for ts in timestamps]
    if entity_id == "sensor.living_humidity":
        return [round(48 - 4.0 * daily_wave(ts, -2) + rng.gauss(0, 0.7), 2) for ts in timestamps]
    if entity_id == "binary_sensor.living_motion":
        return [binary_motion_state(ts, rng) for ts in timestamps]
    if entity_id == "binary_sensor.main_door_contact":
        return [door_state(ts, rng) for ts in timestamps]
    if entity_id == "climate.living":
        return [climate_state(ts) for ts in timestamps]
    return ["off" for _ in timestamps]


def apply_fault(entity: dict[str, Any], values: list[Any], fault: dict[str, Any], rng: random.Random) -> None:
    start = fault["start_index"]
    end = fault["end_index"]
    injector = FaultInjectorFactory().create(str(fault["fault_type"]))
    for index in range(start, end + 1):
        injector.apply_series(entity, values, index, start, end, fault, rng)


def daily_wave(timestamp: datetime, phase_hours: float = 0) -> float:
    hour = timestamp.hour + timestamp.minute / 60 + phase_hours
    return math.sin(2 * math.pi * hour / 24)


def binary_motion_state(timestamp: datetime, rng: random.Random) -> str:
    active_hours = timestamp.hour in {7, 8, 18, 19, 20, 21}
    probability = 0.18 if active_hours else 0.025
    return "on" if rng.random() < probability else "off"


def door_state(timestamp: datetime, rng: random.Random) -> str:
    active_hours = timestamp.hour in {7, 8, 17, 18, 19}
    probability = 0.05 if active_hours else 0.006
    return "on" if rng.random() < probability else "off"


def climate_state(timestamp: datetime) -> str:
    if timestamp.hour in range(0, 6):
        return "off"
    if timestamp.hour in range(6, 9):
        return "heat"
    if timestamp.hour in range(13, 17):
        return "cool"
    return "idle"

def format_state(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def frame_to_csv(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()
