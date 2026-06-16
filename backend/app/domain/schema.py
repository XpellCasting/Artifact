from __future__ import annotations


SCHEMA_VERSION = "canonical-v3"
CANONICAL_ENTITY_COLUMNS = [
    "entity_id",
    "unit_of_measurement",
    "device_class",
    "domain",
    "tipo_interno",
]
CANONICAL_EVENT_COLUMNS = [
    "entity_id",
    "timestamp",
    "state",
    "unit_of_measurement",
    "device_class",
    "domain",
    "tipo_interno",
]
GROUND_TRUTH_COLUMNS = [
    "entity_id",
    "timestamp",
    "label",
    "fault_type",
    "is_reliable",
    "clean_state",
    "observed_state",
]


def tipo_to_sensor_kind(tipo_interno: str) -> str:
    if tipo_interno == "continua":
        return "continuous"
    if tipo_interno == "estado_binario":
        return "binary"
    return "multistate"


def sensor_kind_to_tipo(sensor_kind: str) -> str:
    if sensor_kind == "continuous":
        return "continua"
    if sensor_kind == "binary":
        return "estado_binario"
    return "estado_multivalor"
