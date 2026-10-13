"""Clasificación por entidad: semántica declarada antes que apariencia numérica."""
import math
import re

KINDS = {"continuous", "binary", "multistate"}
BINARY_PAIRS = ({"on", "off"}, {"true", "false"}, {"open", "closed"},
                {"active", "inactive"}, {"detected", "clear"}, {"home", "not_home"})
CATEGORICAL_DOMAINS = {"select", "input_select", "climate", "person", "device_tracker",
                       "input_datetime", "time", "date", "datetime", "image", "counter"}
MEASUREMENTS = {"temperature", "humidity", "illuminance", "battery", "pressure",
                "power", "energy", "voltage", "current", "distance", "speed",
                "duration", "precipitation", "weight", "volume"}


def infer_kind(entity_id, states, *, declared="", domain="", unit="", device_class="",
               all_numeric=None, states_overflow=False):
    declared = str(declared).strip().lower()
    if declared in KINDS:
        return declared
    domain = str(domain).strip().lower() or entity_id.split(".")[0].lower()
    device_class = str(device_class).strip().lower()
    tokens = set(re.split(r"[^a-z0-9]+", entity_id.split(".", 1)[-1].lower()))
    states = {str(state).strip().lower() for state in states}
    if domain in {"binary_sensor", "switch", "input_boolean", "light"}:
        return "binary"
    if domain in CATEGORICAL_DOMAINS or device_class in {"enum", "timestamp", "date"}:
        return "multistate"
    if all_numeric is None:
        try:
            all_numeric = bool(states) and all(math.isfinite(float(value)) for value in states)
        except ValueError:
            all_numeric = False
    # Measurements stay numeric even when rounded to a handful of integers.
    categorical_hint = tokens & {"count", "counter", "conteo", "cantidad", "occupancy", "location",
                                 "locacion", "ubicacion", "schedule", "horario", "weekday", "status"}
    if all_numeric and (device_class in MEASUREMENTS or tokens & MEASUREMENTS
                        or (str(unit).strip() and not categorical_hint)):
        return "continuous"
    if categorical_hint:
        return "multistate"
    if all_numeric:
        return "continuous"
    if states and not states_overflow and any(states <= pair for pair in BINARY_PAIRS):
        return "binary"
    return "multistate"
