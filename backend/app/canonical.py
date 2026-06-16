from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .domain.entities import (
    canonical_entities as domain_canonical_entities,
    canonical_entities_to_internal as domain_entities_to_internal,
)
from .domain.fault_types import normalize_fault
from .domain.schema import SCHEMA_VERSION
from .fault_injection import FaultInjectorFactory


DEFAULT_SEED = 20260614
DEFAULT_HISTORY_DAYS = 60
DEFAULT_SAMPLE_SECONDS = 60
START_TIMESTAMP = datetime(2026, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=-3)))
CANONICAL_ENTITIES = domain_canonical_entities()


@dataclass(frozen=True)
class CanonicalHistory:
    entities: pd.DataFrame
    events: pd.DataFrame
    ground_truth: pd.DataFrame
    metadata: pd.DataFrame
    created: bool


def ensure_static_history(
    directory: Path,
    duration_days: int = DEFAULT_HISTORY_DAYS,
    base_sample_seconds: int = DEFAULT_SAMPLE_SECONDS,
    seed: int = DEFAULT_SEED,
) -> CanonicalHistory:
    required = {
        "entities": directory / "entities.csv",
        "events": directory / "events.csv",
        "ground_truth": directory / "ground_truth.csv",
        "metadata": directory / "metadata.csv",
    }
    existing = [path.exists() for path in required.values()]
    if all(existing):
        history = load_static_history(directory, created=False, load_ground_truth=False)
        validate_static_history_domain(history, directory)
        return history
    if any(existing):
        missing = [name for name, path in required.items() if not path.exists()]
        raise ValueError(f"Static history is incomplete. Missing: {', '.join(missing)}")

    directory.mkdir(parents=True, exist_ok=True)
    history = generate_static_history(duration_days, base_sample_seconds, seed)
    history.entities.to_csv(required["entities"], index=False)
    history.events.to_csv(required["events"], index=False)
    history.ground_truth.to_csv(required["ground_truth"], index=False)
    history.metadata.to_csv(required["metadata"], index=False)
    return CanonicalHistory(history.entities, history.events, history.ground_truth, history.metadata, created=True)


def validate_static_history_domain(history: CanonicalHistory, directory: Path) -> None:
    metadata = history.metadata.to_dict("records")[0] if not history.metadata.empty else {}
    expected_entities = {str(entity["entity_id"]) for entity in CANONICAL_ENTITIES}
    actual_entities = set(history.entities["entity_id"].astype(str).tolist()) if "entity_id" in history.entities else set()
    if metadata.get("schema_version") == SCHEMA_VERSION and actual_entities == expected_entities:
        return
    raise ValueError(
        "Static history was generated with an older domain catalog. "
        f"Remove or regenerate CSV files in {directory} so schema {SCHEMA_VERSION} can be created."
    )


def load_static_history(directory: Path, created: bool = False, load_ground_truth: bool = True) -> CanonicalHistory:
    ground_truth = (
        pd.read_csv(directory / "ground_truth.csv", dtype=str, keep_default_na=False).fillna("")
        if load_ground_truth
        else pd.DataFrame()
    )
    return CanonicalHistory(
        entities=pd.read_csv(directory / "entities.csv", dtype=str, keep_default_na=False).fillna(""),
        events=pd.read_csv(directory / "events.csv", dtype=str, keep_default_na=False).fillna(""),
        ground_truth=ground_truth,
        metadata=pd.read_csv(directory / "metadata.csv", dtype=str, keep_default_na=False).fillna(""),
        created=created,
    )


def generate_static_history(duration_days: int, base_sample_seconds: int, seed: int) -> CanonicalHistory:
    duration_days = max(DEFAULT_HISTORY_DAYS, int(duration_days))
    base_sample_seconds = max(60, int(base_sample_seconds))
    entities = pd.DataFrame(CANONICAL_ENTITIES)
    event_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    end = START_TIMESTAMP + timedelta(days=duration_days)

    for entity in CANONICAL_ENTITIES:
        entity_seed = seed + stable_entity_offset(entity["entity_id"])
        rng = random.Random(entity_seed)
        timestamp = START_TIMESTAMP
        skipped_until: datetime | None = None
        while timestamp < end:
            if skipped_until and timestamp < skipped_until:
                timestamp += timedelta(seconds=base_sample_seconds)
                continue
            if rng.random() < 0.0008:
                skipped_until = timestamp + timedelta(minutes=rng.randint(3, 12))
                timestamp += timedelta(seconds=base_sample_seconds)
                continue
            if rng.random() < 0.0025:
                timestamp += timedelta(seconds=base_sample_seconds)
                continue

            jitter = rng.randint(-8, 8)
            observed_at = timestamp + timedelta(seconds=jitter)
            state = clean_state_for_entity(entity, observed_at, rng)
            row = canonical_event_row(entity, observed_at, state)
            event_rows.append(row)
            truth_rows.append(canonical_truth_row(row, state, state, "normal", "", "true"))
            timestamp += timedelta(seconds=base_sample_seconds)

    events = pd.DataFrame(event_rows).sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    ground_truth = pd.DataFrame(truth_rows).sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    metadata = pd.DataFrame(
        [
            {
                "seed": str(seed),
                "start_timestamp": START_TIMESTAMP.isoformat(),
                "end_timestamp": (end - timedelta(seconds=base_sample_seconds)).isoformat(),
                "duration_days": str(duration_days),
                "base_sample_seconds": str(base_sample_seconds),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            }
        ]
    )
    return CanonicalHistory(entities, events, ground_truth, metadata, created=True)


def next_realtime_events(
    entities: pd.DataFrame,
    timestamp: pd.Timestamp,
    selected_entities: list[str],
    faults: list[dict[str, Any]],
    tick_index: int,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    entity_lookup = {str(row["entity_id"]): row for row in entities.to_dict("records")}
    fault_lookup = {
        str(fault.get("entity_id")): normalize_fault(fault)
        for fault in faults
        if fault.get("enabled", True) and fault.get("entity_id")
    }
    event_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    active_faults: list[dict[str, Any]] = []
    py_timestamp = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp

    for entity_id in selected_entities:
        entity = entity_lookup.get(entity_id)
        if not entity:
            continue
        rng = random.Random(seed + stable_entity_offset(entity_id) + int(py_timestamp.timestamp()))
        clean_state = clean_state_for_entity(entity, py_timestamp, rng)
        fault = fault_lookup.get(entity_id)
        if fault:
            observed_state, missing = apply_tick_fault(entity, clean_state, fault, tick_index, rng)
            label = str(fault.get("label", "sensor_fault"))
            fault_type = str(fault.get("fault_type", ""))
            is_reliable = "false"
            active_faults.append(
                {
                    "entity_id": entity_id,
                    "fault_type": fault_type,
                    "enabled": True,
                    "start": py_timestamp.isoformat(),
                    "end": py_timestamp.isoformat(),
                }
            )
        else:
            observed_state = clean_state
            missing = False
            label = "normal"
            fault_type = ""
            is_reliable = "true"

        base_row = canonical_event_row(entity, py_timestamp, observed_state)
        if not missing:
            event_rows.append(base_row)
        truth_rows.append(canonical_truth_row(base_row, clean_state, observed_state if not missing else "", label, fault_type, is_reliable))

    return pd.DataFrame(event_rows), pd.DataFrame(truth_rows), active_faults


def canonical_event_row(entity: dict[str, Any], timestamp: datetime, state: Any) -> dict[str, Any]:
    return {
        "entity_id": str(entity["entity_id"]),
        "timestamp": timestamp.isoformat(),
        "state": format_state(state),
        "unit_of_measurement": str(entity.get("unit_of_measurement", "")),
        "device_class": str(entity.get("device_class", "")),
        "domain": str(entity.get("domain", "")),
        "tipo_interno": str(entity.get("tipo_interno", "")),
    }


def canonical_truth_row(
    event_row: dict[str, Any],
    clean_state: Any,
    observed_state: Any,
    label: str,
    fault_type: str,
    is_reliable: str,
) -> dict[str, Any]:
    return {
        "entity_id": str(event_row["entity_id"]),
        "timestamp": str(event_row["timestamp"]),
        "label": label,
        "fault_type": fault_type,
        "is_reliable": is_reliable,
        "clean_state": format_state(clean_state),
        "observed_state": format_state(observed_state),
    }


def clean_state_for_entity(entity: dict[str, Any], timestamp: datetime, rng: random.Random) -> Any:
    entity_id = str(entity["entity_id"])
    if entity_id == "sensor.living_temperature":
        return round(21.4 + 1.8 * daily_wave(timestamp, -3) + rng.gauss(0, 0.16), 2)
    if entity_id == "sensor.living_humidity":
        return round(48.5 - 4.2 * daily_wave(timestamp, -2) + rng.gauss(0, 0.65), 2)
    if entity_id == "binary_sensor.living_motion":
        active_hours = timestamp.hour in {7, 8, 18, 19, 20, 21}
        probability = 0.16 if active_hours else 0.012
        return "on" if rng.random() < probability else "off"
    if entity_id == "binary_sensor.main_door_contact":
        active_hours = timestamp.hour in {7, 8, 17, 18, 19}
        probability = 0.045 if active_hours else 0.004
        return "on" if rng.random() < probability else "off"
    if entity_id == "climate.living":
        if timestamp.hour in range(0, 6):
            return "off"
        if timestamp.hour in range(6, 9):
            return "heat"
        if timestamp.hour in range(13, 17):
            return "cool"
        return "idle"
    generic_state = generic_state_from_real_entity(entity, rng)
    if generic_state is not None:
        return generic_state
    return "off"


def generic_state_from_real_entity(entity: dict[str, Any], rng: random.Random) -> Any:
    tipo = str(entity.get("tipo_interno", ""))
    device_class = str(entity.get("device_class", ""))
    last_state = str(entity.get("last_state", "")).strip()
    if tipo == "continua":
        base_text = str(entity.get("numeric_median") or last_state or "0")
        try:
            base = float(base_text)
        except ValueError:
            return last_state or "0"
        try:
            spread = max(float(entity.get("numeric_mad") or 0.0), 0.1)
        except ValueError:
            spread = 0.1
        return round(base + rng.gauss(0, spread * 0.20), 2)

    if tipo == "estado_binario":
        if device_class == "motion":
            return "on" if rng.random() < 0.04 else "off"
        if last_state.lower() in {"on", "open"}:
            return "off" if rng.random() < 0.85 else "on"
        return "on" if rng.random() < 0.02 else "off"

    if tipo == "estado_multivalor" and last_state:
        return last_state
    return None


def apply_tick_fault(
    entity: dict[str, Any],
    clean_state: Any,
    fault: dict[str, Any],
    tick_index: int,
    rng: random.Random,
) -> tuple[Any, bool]:
    injector = FaultInjectorFactory().create(str(fault.get("fault_type", "")))
    observed_state, missing = injector.apply_tick(entity, clean_state, fault, tick_index, rng)
    return format_state(observed_state), missing


def canonical_entities_to_internal(entities: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return domain_entities_to_internal(entities.to_dict("records"))


def canonical_events_to_internal_states(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["timestamp", "entity_id", "domain", "state", "last_changed", "last_updated", "unit_of_measurement", "device_class", "tipo_interno"])
    states = events.copy()
    states["last_changed"] = states["timestamp"]
    states["last_updated"] = states["timestamp"]
    return states


def canonical_truth_to_lookup(ground_truth: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["timestamp"]), str(row["entity_id"])): row
        for row in ground_truth.to_dict("records")
    }


def stable_entity_offset(entity_id: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(entity_id))


def daily_wave(timestamp: datetime, phase_hours: float = 0) -> float:
    hour = timestamp.hour + timestamp.minute / 60 + phase_hours
    return math.sin(2 * math.pi * hour / 24)


def format_state(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
