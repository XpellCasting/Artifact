from __future__ import annotations

from typing import Any

from .schema import CANONICAL_ENTITY_COLUMNS, sensor_kind_to_tipo, tipo_to_sensor_kind


DOMAIN_ENTITIES: list[dict[str, Any]] = [
    {
        "entity_id": "sensor.living_temperature",
        "domain": "sensor",
        "device_class": "temperature",
        "state_class": "measurement",
        "unit_of_measurement": "C",
        "area_id": "living",
        "device_id": "env_living",
        "integration": "simulation",
        "sensor_kind": "continuous",
    },
    {
        "entity_id": "sensor.living_humidity",
        "domain": "sensor",
        "device_class": "humidity",
        "state_class": "measurement",
        "unit_of_measurement": "%",
        "area_id": "living",
        "device_id": "env_living",
        "integration": "simulation",
        "sensor_kind": "continuous",
    },
    {
        "entity_id": "binary_sensor.living_motion",
        "domain": "binary_sensor",
        "device_class": "motion",
        "state_class": "",
        "unit_of_measurement": "",
        "area_id": "living",
        "device_id": "pir_living",
        "integration": "simulation",
        "sensor_kind": "binary",
    },
    {
        "entity_id": "binary_sensor.main_door_contact",
        "domain": "binary_sensor",
        "device_class": "door",
        "state_class": "",
        "unit_of_measurement": "",
        "area_id": "entrance",
        "device_id": "contact_main_door",
        "integration": "simulation",
        "sensor_kind": "binary",
    },
    {
        "entity_id": "climate.living",
        "domain": "climate",
        "device_class": "hvac_action",
        "state_class": "",
        "unit_of_measurement": "",
        "area_id": "living",
        "device_id": "climate_living",
        "integration": "simulation",
        "sensor_kind": "multistate",
    },
]


def canonical_entities() -> list[dict[str, Any]]:
    return [
        {
            **{column: str(entity.get(column, "")) for column in CANONICAL_ENTITY_COLUMNS},
            "tipo_interno": sensor_kind_to_tipo(str(entity["sensor_kind"])),
        }
        for entity in DOMAIN_ENTITIES
    ]


def simulation_entities() -> list[dict[str, Any]]:
    rows = []
    for entity in DOMAIN_ENTITIES:
        sensor_kind = str(entity["sensor_kind"])
        rows.append(
            {
                **entity,
                "unit": str(entity.get("unit_of_measurement", "")),
                "labels": f"{sensor_kind}|diagnostic_source",
                "tipo_interno": sensor_kind_to_tipo(sensor_kind),
            }
        )
    return rows


def canonical_entities_to_internal(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in entities:
        entity_id = str(row["entity_id"])
        sensor_kind = str(row.get("sensor_kind") or tipo_to_sensor_kind(str(row.get("tipo_interno", ""))))
        output[entity_id] = {
            **row,
            "unit": str(row.get("unit_of_measurement", row.get("unit", ""))),
            "sensor_kind": sensor_kind,
            "tipo_interno": str(row.get("tipo_interno") or sensor_kind_to_tipo(sensor_kind)),
            "labels": str(row.get("labels") or sensor_kind),
            "selected": True,
        }
    return output


def entity_by_id(entity_id: str) -> dict[str, Any] | None:
    for entity in DOMAIN_ENTITIES:
        if entity["entity_id"] == entity_id:
            return dict(entity)
    return None
