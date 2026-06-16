from __future__ import annotations

from typing import Any


FAULT_RANDOM_SPIKE = "random/spike"
FAULT_RANDOM = "random"
FAULT_MALFUNCTION = "malfunction"
FAULT_BIAS = "bias"
FAULT_DRIFT = "drift"
FAULT_STUCK_AT = "stuck-at"
FAULT_MISSING = "missing"

CANONICAL_FAULT_TYPES = [
    FAULT_RANDOM_SPIKE,
    FAULT_RANDOM,
    FAULT_MALFUNCTION,
    FAULT_BIAS,
    FAULT_DRIFT,
    FAULT_STUCK_AT,
    FAULT_MISSING,
]

FAULT_ALIASES = {
    "spike": FAULT_RANDOM_SPIKE,
    "noise": FAULT_MALFUNCTION,
    "stuck": FAULT_STUCK_AT,
    "stuck_at": FAULT_STUCK_AT,
    "stuck-at": FAULT_STUCK_AT,
    "random_activation": FAULT_RANDOM,
    "stuck_on": FAULT_STUCK_AT,
    "stuck_off": FAULT_STUCK_AT,
    "chatter": FAULT_MALFUNCTION,
    "rare_state": FAULT_RANDOM,
    "rare_transition": FAULT_RANDOM,
    "state_stuck": FAULT_STUCK_AT,
    "rapid_switching": FAULT_MALFUNCTION,
}

DEFAULT_FAULTS = [
    {"entity_id": "sensor.living_temperature", "fault_type": FAULT_RANDOM_SPIKE, "enabled": True, "start_ratio": 0.55, "end_ratio": 0.57, "label": "sensor_fault"},
    {"entity_id": "sensor.living_humidity", "fault_type": FAULT_DRIFT, "enabled": True, "start_ratio": 0.60, "end_ratio": 0.90, "label": "sensor_fault"},
    {"entity_id": "binary_sensor.living_motion", "fault_type": FAULT_RANDOM, "enabled": True, "start_ratio": 0.58, "end_ratio": 0.70, "label": "sensor_fault"},
    {"entity_id": "binary_sensor.main_door_contact", "fault_type": FAULT_STUCK_AT, "enabled": True, "start_ratio": 0.62, "end_ratio": 0.78, "label": "sensor_fault"},
    {"entity_id": "climate.living", "fault_type": FAULT_RANDOM, "enabled": True, "start_ratio": 0.56, "end_ratio": 0.62, "label": "sensor_fault"},
]


def normalize_fault_type(fault_type: str) -> str:
    normalized = str(fault_type).strip().lower().replace("_", "-")
    legacy_key = str(fault_type).strip().lower()
    return FAULT_ALIASES.get(legacy_key) or FAULT_ALIASES.get(normalized) or normalized


def normalize_fault(fault: dict[str, Any]) -> dict[str, Any]:
    item = dict(fault)
    item["fault_type"] = normalize_fault_type(str(item.get("fault_type", "")))
    if str(fault.get("fault_type", "")).strip().lower() == "stuck_off":
        item.setdefault("stuck_value", "off")
    if str(fault.get("fault_type", "")).strip().lower() == "stuck_on":
        item.setdefault("stuck_value", "on")
    return item


def normalize_faults(faults: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_fault(fault) for fault in faults if fault.get("entity_id") and fault.get("fault_type")]
