"""Event-loop cache for confidence sensors received over the paired bridge."""
from copy import deepcopy
import math
import re

from homeassistant.core import callback
from homeassistant.helpers import entity_registry

from .const import DOMAIN

SOURCE_ID = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")


def confidence_unique_id(entry_id, source, mode):
    return f"{entry_id}_confidence_{'test_' if mode == 'test' else ''}{source}"


@callback
def remove_confidence_entity(hass, entry_id, unique_id):
    registry = entity_registry.async_get(hass)
    entity_id = registry.async_get_entity_id("sensor", DOMAIN, unique_id)
    if entity_id is not None and registry.async_get(entity_id).config_entry_id == entry_id:
        registry.async_remove(entity_id)


class ConfidenceHub:
    def __init__(self):
        self.entities = {}
        self.entity_ids = {}
        self.connected = self.synchronized = False
        self.store_id = None
        self.listeners = set()
        self.dashboard_update = None
        self._sync_entities = set()
        self._removed_revisions = {}

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
                    or ("deleted" in item and (item["deleted"] is not True or item["available"]))
                    or (value is not None and (type(value) not in (int, float)
                                              or not math.isfinite(value) or not 0 <= value <= 100))):
                raise ValueError("La entidad de confianza no es válida.")
            rows.append(item)
        if sync and (type(payload.get("revision")) is not int or payload["revision"] < 0):
            raise ValueError("La revisión del catálogo no es válida.")
        if payload["store_id"] != self.store_id:
            self.store_id = payload["store_id"]
            self.entities = {key: {**value, "available": False, "revision": -1} for key, value in self.entities.items()}
            self._removed_revisions.clear()
            self.synchronized = False
        if sync and payload.get("reset"):
            self._sync_entities = set()
        changed = set()
        removed = []
        dashboard_changed = False
        for item in rows:
            entity = item["entity_id"]
            revision = item["revision"]
            previous = self.entities.get(entity)
            removed_revision = self._removed_revisions.get(entity, -1)
            if item.get("deleted"):
                if revision >= removed_revision and (previous is None or revision > previous["revision"]):
                    self._removed_revisions[entity] = revision
                    self.entities.pop(entity, None)
                    if self.entity_ids.pop(entity, None) is not None:
                        dashboard_changed = True
                    removed.append(item)
                continue
            if sync:
                self._sync_entities.add(entity)
            if revision > removed_revision and (previous is None or revision > previous["revision"]):
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
        if dashboard_changed and self.dashboard_update is not None:
            self.dashboard_update()
        return removed

    @callback
    def register_entity(self, key, entity_id):
        self.entity_ids[key] = entity_id
        if self.dashboard_update is not None:
            self.dashboard_update()
