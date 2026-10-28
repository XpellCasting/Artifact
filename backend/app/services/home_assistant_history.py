"""Read recorded state changes from Home Assistant without storing credentials."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlencode


SUPPORTED_DOMAINS = {"sensor", "binary_sensor", "switch", "input_boolean", "select", "input_select"}
ENTITY_ID = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")


@dataclass
class HomeAssistantHistorySource:
    start: datetime
    end: datetime
    entity_ids: list[str] = field(default_factory=list)
    transport: Callable | None = field(default=None, repr=False)
    catalog_transport: Callable | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.transport is None:
            raise ValueError("La integración de Home Assistant es necesaria para consultar el historial.")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Las fechas deben incluir su zona horaria.")
        self.start = self.start.astimezone(timezone.utc)
        self.end = self.end.astimezone(timezone.utc)
        if self.start >= self.end:
            raise ValueError("La fecha final debe ser posterior a la inicial.")
        if self.end > datetime.now(timezone.utc) + timedelta(minutes=1):
            raise ValueError("El historial debe corresponder a fechas pasadas.")
        self.entity_ids = list(dict.fromkeys(item.strip() for item in self.entity_ids))
        if any(not ENTITY_ID.fullmatch(item) or item.split(".", 1)[0] not in SUPPORTED_DOMAINS
               for item in self.entity_ids):
            raise ValueError("Selecciona entidades sensor, binary_sensor, switch, input_boolean, select o input_select válidas.")

    def _get(self, path: str):
        return self.transport(path)

    def export_csv(self, destination: Path, report: Callable[[str, float, str], None]) -> int:
        report("connecting_home_assistant", 1, "Conectando con Home Assistant")
        # Explicit IDs also allow importing entities that are no longer in /states.
        entity_ids = self.entity_ids
        if not entity_ids:
            raise ValueError("Activa al menos un sensor antes de importar el historial.")
        batches = [entity_ids[index:index + 50] for index in range(0, len(entity_ids), 50)]
        days = max(1, (self.end - self.start + timedelta(days=1) - timedelta(microseconds=1)).days)
        total_requests = days * len(batches)
        completed = rows = 0
        last_seen = {}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["entity_id", "state", "last_changed",
                                    "unit_of_measurement", "device_class", "domain", "history_origin"])
            writer.writeheader()
            cursor = self.start
            while cursor < self.end:
                until = min(cursor + timedelta(days=1), self.end)
                for batch in batches:
                    # Import recorded changes, without artificial start states for each day.
                    query = urlencode({"filter_entity_id": ",".join(batch), "end_time": until.isoformat(),
                                       "minimal_response": "true", "skip_initial_state": "true",
                                       "significant_changes_only": "0"})
                    history = self._get(f"/history/period/{quote(cursor.isoformat(), safe='')}?{query}")
                    if not isinstance(history, list):
                        raise ValueError("Home Assistant no devolvió un historial válido.")
                    for group in history:
                        if not isinstance(group, list) or not group:
                            continue
                        entity_id = group[0].get("entity_id")
                        if entity_id not in batch:
                            continue
                        attributes = group[0].get("attributes") or {}
                        for event in group:
                            try:
                                at = datetime.fromisoformat(event["last_changed"].replace("Z", "+00:00"))
                                if at.tzinfo is None:
                                    continue
                                at = at.astimezone(timezone.utc)
                            except (KeyError, ValueError, TypeError, AttributeError):
                                continue
                            # Boundary states may repeat in the next daily response.
                            if not self.start <= at < self.end or at <= last_seen.get(entity_id, datetime.min.replace(tzinfo=timezone.utc)):
                                continue
                            attributes = event.get("attributes") or attributes
                            writer.writerow({"entity_id": entity_id, "state": event.get("state", ""),
                                             "last_changed": at.isoformat(), "domain": entity_id.split(".", 1)[0],
                                             "unit_of_measurement": attributes.get("unit_of_measurement", ""),
                                             "device_class": attributes.get("device_class", ""),
                                             "history_origin": "home_assistant"})
                            last_seen[entity_id] = at
                            rows += 1
                    completed += 1
                    report("fetching_home_assistant", 1 + 8 * completed / total_requests,
                           f"Importando historial: {completed}/{total_requests} consultas · {rows:,} lecturas")
                cursor = until
        if not rows:
            raise ValueError("No hay lecturas en el período seleccionado. Revisa las entidades y la retención del historial de Home Assistant.")
        return destination.stat().st_size

    def discover_entities(self):
        from app.services.tipos_entidad import infer_kind
        metadata = self.catalog_transport() if self.catalog_transport else {"entities": self._get("/states")}
        if not isinstance(metadata, dict):
            raise ValueError("Home Assistant no devolvió un catálogo válido.")
        states = metadata.get("entities")
        if not isinstance(states, list):
            raise ValueError("Home Assistant no devolvió una lista de entidades válida.")
        catalog = {}
        areas_available = metadata.get("areas_available") is True
        # A partial response cannot safely distinguish unknown and unassigned areas.
        areas_available = areas_available and all(
            not isinstance(item, dict) or "area_id" in item and "area_name" in item
            and (item["area_id"] is None and item["area_name"] is None
                 or isinstance(item["area_id"], str) and bool(item["area_id"])
                 and isinstance(item["area_name"], str) and bool(item["area_name"]))
            for item in states)
        for item in states:
            if not isinstance(item, dict):
                continue
            entity_id = item.get("entity_id", "")
            if not ENTITY_ID.fullmatch(entity_id) or entity_id.split(".", 1)[0] not in SUPPORTED_DOMAINS:
                continue
            attributes = item.get("attributes") or {}
            domain = entity_id.split(".", 1)[0]
            catalog[entity_id] = {"entity_id": entity_id, "friendly_name": attributes.get("friendly_name", ""),
                "domain": domain, "device_class": attributes.get("device_class", ""),
                "unit_of_measurement": attributes.get("unit_of_measurement", ""),
                "sensor_kind": infer_kind(entity_id, [item.get("state", "")], domain=domain,
                    unit=attributes.get("unit_of_measurement", ""), device_class=attributes.get("device_class", ""))}
            if areas_available:
                catalog[entity_id].update(area_id=item["area_id"], area_name=item["area_name"])
        if not catalog:
            raise ValueError("No hay sensores compatibles para seleccionar en Home Assistant.")
        return {"source": "home_assistant", "entities": [catalog[key] for key in sorted(catalog)],
                "entity_count": len(catalog), "areas_available": areas_available}
