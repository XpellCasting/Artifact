"""Event-loop cache for confidence sensors received over the paired bridge."""
from copy import deepcopy
import math
import re

from homeassistant.core import callback


SOURCE_ID = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")


class ConfidenceHub:
    def __init__(self):
        self.entities = {}
        self.entity_ids = {}
        self.connected = self.synchronized = False
        self.store_id = None
        self.listeners = set()
        self.dashboard_update = None
        self._sync_entities = set()

    @callback
    def listen(self, listener):
        self.listeners.add(listener)
        return lambda: self.listeners.discard(listener)

    @callback
    def _notify(self):
        for listener in tuple(self.listeners):
            listener()

    @callback
    def set_connected(self, connected):
        self.connected = connected
        self.synchronized = False
        self._notify()

    @callback
    def apply(self, payload, *, sync=False):
        if (not isinstance(payload, dict) or payload.get("schema_version") != "confidence-entities-v1"
                or not isinstance(payload.get("store_id"), str) or not payload["store_id"]
                or not isinstance(payload.get("entities"), list) or len(payload["entities"]) > 500):
            raise ValueError("El catálogo de confianza no es válido.")
        rows = []
        for item in payload["entities"]:
            if not isinstance(item, dict):
                raise ValueError("La entidad de confianza no es válida.")
            source, value = item.get("source_entity_id"), item.get("value")
            mode = item.get("source_mode", "live")
            prefix = "test_" if mode == "test" else ""
            if (not isinstance(source, str) or not SOURCE_ID.fullmatch(source)
                    or source.startswith("sensor.confianza_")
                    or mode not in ("live", "test")
                    or item.get("entity_id") != f"confianza.{prefix}{source.replace('.', '_')}"
                    or item.get("store_id") != payload["store_id"]
                    or type(item.get("revision")) is not int or item["revision"] < 0
                    or type(item.get("available")) is not bool
                    or (value is not None and (type(value) not in (int, float)
                                              or not math.isfinite(value) or not 0 <= value <= 100))):
                raise ValueError("La entidad de confianza no es válida.")
            rows.append(item)
        if sync and (type(payload.get("revision")) is not int or payload["revision"] < 0):
            raise ValueError("La revisión del catálogo no es válida.")
        if payload["store_id"] != self.store_id:
            self.store_id = payload["store_id"]
            self.entities = {key: {**value, "available": False, "revision": -1} for key, value in self.entities.items()}
            self.synchronized = False
        if sync and payload.get("reset"):
            self._sync_entities = set()
        changed = set()
        for item in rows:
            entity = item["entity_id"]
            if sync:
                self._sync_entities.add(entity)
            previous = self.entities.get(entity)
            if previous is None or item["revision"] > previous["revision"]:
                if entity in changed:
                    # A timestamp group may include multiple readings of the same sensor.
                    self._notify()
                    changed.clear()
                self.entities[entity] = deepcopy(item)
                changed.add(entity)
        if sync and payload.get("complete"):
            for entity, item in self.entities.items():
                if entity not in self._sync_entities and item["revision"] <= payload["revision"]:
                    item["available"] = False
            self.synchronized = True
        self._notify()

    @callback
    def register_entity(self, key, entity_id):
        self.entity_ids[key] = entity_id
        if self.dashboard_update is not None:
            self.dashboard_update()
