"""Read recorded state changes from Home Assistant without storing credentials."""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SUPPORTED_DOMAINS = {"sensor", "binary_sensor", "switch", "input_boolean", "select", "input_select"}
ENTITY_ID = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")


def resolve_connection(url: str, token: str) -> tuple[str, str]:
    configured_url = os.environ.get("HOME_ASSISTANT_URL", "http://homeassistant.local:8123").strip().rstrip("/")
    url = url.strip().rstrip("/") or configured_url
    if not token and url != configured_url:
        raise ValueError("Introduce un token para esta URL de Home Assistant.")
    return url, token or os.environ.get("HOME_ASSISTANT_TOKEN", "")


def validate_connection(url: str, token: str) -> tuple[str, str]:
    url = url.strip().rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Introduce una URL HTTP(S) de Home Assistant sin credenciales ni parámetros.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("El puerto de Home Assistant no es válido.") from exc
    token = token.strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("Introduce un token de acceso de larga duración de Home Assistant.")
    return url, token


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A redirect must never forward the user's bearer token to another host.
        return None


@dataclass
class HomeAssistantHistorySource:
    url: str
    token: str = field(repr=False)
    start: datetime
    end: datetime
    entity_ids: list[str] = field(default_factory=list)
    transport: Callable | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.transport is None:
            self.url, self.token = validate_connection(self.url, self.token)
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
        if self.transport is not None:
            return self.transport(path)
        request = Request(f"{self.url}/api{path}", headers={
            "Authorization": f"Bearer {self.token}", "Accept": "application/json",
        })
        try:
            with build_opener(NoRedirect()).open(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise ValueError("Home Assistant rechazó el token. Revisa su vigencia y permisos.") from None
            if exc.code == 404:
                raise ValueError("No se encontró la API de historial. Revisa la URL y la integración History de Home Assistant.") from None
            raise ValueError(f"Home Assistant no pudo entregar el historial (HTTP {exc.code}).") from None
        except (URLError, TimeoutError, OSError):
            raise ValueError("No se pudo conectar con Home Assistant. Comprueba la URL y el acceso desde el servidor de Artifact.") from None
        except (ValueError, UnicodeError):
            raise ValueError("Home Assistant no devolvió una respuesta JSON válida.") from None

    def export_csv(self, destination: Path, report: Callable[[str, float, str], None]) -> int:
        report("connecting_home_assistant", 1, "Conectando con Home Assistant")
        # Explicit IDs also allow importing entities that are no longer in /states.
        entity_ids = self.entity_ids
        if not entity_ids:
            states = self._get("/states")
            if not isinstance(states, list):
                raise ValueError("Home Assistant no devolvió una lista de entidades válida.")
            entity_ids = sorted({item["entity_id"] for item in states
                                 if isinstance(item, dict) and ENTITY_ID.fullmatch(item.get("entity_id", ""))
                                 and item["entity_id"].split(".", 1)[0] in SUPPORTED_DOMAINS})
        if not entity_ids:
            raise ValueError("No hay entidades compatibles para importar desde Home Assistant.")
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
        states = self._get("/states")
        if not isinstance(states, list):
            raise ValueError("Home Assistant no devolvió una lista de entidades válida.")
        catalog = {}
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
        if not catalog:
            raise ValueError("No hay sensores compatibles para seleccionar en Home Assistant.")
        return {"source": "home_assistant", "entities": [catalog[key] for key in sorted(catalog)], "entity_count": len(catalog)}
