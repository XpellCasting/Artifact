"""Inspect candidate CSVs without replacing history or preparing any filter."""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
from time import time
from uuid import uuid4

from app.services.estados_validos import valid_state
from app.services.tipos_entidad import infer_kind, KINDS


def confirm_selection(path, entity_ids):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = path.with_suffix(path.suffix + ".selection.json")
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": 1, "sha256": digest,
        "entity_ids": sorted(set(entity_ids)), "confirmed_at": time()}), encoding="utf-8")
    temporary.replace(manifest)


@lru_cache(maxsize=32)
def _matches_selection(path, file_time, file_size, manifest_time, manifest_size):
    path = Path(path)
    try:
        payload = json.loads(path.with_suffix(path.suffix + ".selection.json").read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not payload.get("entity_ids"):
            return False
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest() == payload.get("sha256")
    except (OSError, ValueError):
        return False


def selection_confirmed(path):
    try:
        source = path.stat()
        manifest = path.with_suffix(path.suffix + ".selection.json").stat()
    except OSError:
        return False
    return _matches_selection(str(path), source.st_mtime_ns, source.st_size, manifest.st_mtime_ns, manifest.st_size)


class HistoryCandidates:
    ttl_seconds = 86400

    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, identifier, suffix=".csv"):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("La preselección de historial no es válida.")
        return self.directory / f"{identifier}{suffix}"

    def create(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cleanup()
        identifier = uuid4().hex
        return identifier, self.path(identifier)

    def discard(self, identifier):
        self.path(identifier).unlink(missing_ok=True)
        self.path(identifier, ".json").unlink(missing_ok=True)

    def cleanup(self):
        for path in self.directory.glob("*"):
            try:
                if (path.suffix in {".csv", ".json"} and re.fullmatch(r"[a-f0-9]{32}", path.stem)
                        and path.stat().st_mtime < time() - self.ttl_seconds):
                    path.unlink(missing_ok=True)
            except OSError:
                # Another upload/worker may still be using an expired candidate.
                continue

    def inspect(self, identifier, filename):
        entities = {}
        total = invalid = 0
        start = end = None
        try:
            with self.path(identifier).open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                missing = {"entity_id", "state", "last_changed"} - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"El CSV no contiene las columnas requeridas: {', '.join(sorted(missing))}.")
                for row in reader:
                    total += 1
                    entity_id = (row.get("entity_id") or "").strip()
                    try:
                        at = datetime.fromisoformat((row.get("last_changed") or "").strip().replace("Z", "+00:00"))
                        at = at.replace(tzinfo=timezone.utc) if at.tzinfo is None else at.astimezone(timezone.utc)
                    except (ValueError, TypeError):
                        invalid += 1
                        continue
                    if not entity_id:
                        invalid += 1
                        continue
                    entry = entities.setdefault(entity_id, {"entity_id": entity_id, "rows": 0, "_states": set(),
                        "_all_numeric": True, "_overflow": False, "_declared": set()})
                    entry["rows"] += 1
                    for key in ("friendly_name", "domain", "unit_of_measurement", "device_class"):
                        value = (row.get(key) or "").strip()
                        if value and not entry.get(key):
                            entry[key] = value
                    declared = (row.get("sensor_kind") or "").strip().lower()
                    if declared in KINDS:
                        entry["_declared"].add(declared)
                    value = (row.get("state") or "").strip()
                    if valid_state(value):
                        try:
                            numeric = math.isfinite(float(value))
                        except ValueError:
                            numeric = False
                        entry["_all_numeric"] &= numeric
                        if len(entry["_states"]) < 64:
                            entry["_states"].add(value)
                        elif value not in entry["_states"]:
                            entry["_overflow"] = True
                    start = at if start is None or at < start else start
                    end = at if end is None or at > end else end
        except (UnicodeError, csv.Error) as exc:
            raise ValueError("No se pudo leer el CSV. Usa un archivo UTF-8 separado por comas.") from exc
        if not entities:
            raise ValueError("El CSV no contiene sensores con entidad y fecha válidas.")
        catalog = []
        for entity_id, entry in sorted(entities.items()):
            if len(entry["_declared"]) > 1:
                raise ValueError(f"Hay tipos declarados incompatibles para {entity_id}.")
            entry["sensor_kind"] = infer_kind(entity_id, entry["_states"],
                declared=next(iter(entry["_declared"]), ""), domain=entry.get("domain", ""),
                unit=entry.get("unit_of_measurement", ""), device_class=entry.get("device_class", ""),
                all_numeric=bool(entry["_states"]) and entry["_all_numeric"], states_overflow=entry["_overflow"])
            entry.setdefault("domain", entity_id.split(".", 1)[0])
            catalog.append({key: value for key, value in entry.items() if not key.startswith("_")})
        payload = {"candidate_id": identifier, "source": "csv", "filename": filename[:200], "rows": total - invalid,
                   "invalid_rows": invalid, "entity_count": len(catalog), "entities": catalog,
                   "start": start.isoformat(), "end": end.isoformat(), "created_at": time()}
        self.path(identifier, ".json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def from_saved(self, path):
        if not path.is_file():
            raise FileNotFoundError("No hay un CSV guardado para revisar.")
        identifier, destination = self.create()
        try:
            shutil.copyfile(path, destination)
            return self.inspect(identifier, "CSV guardado en el servidor")
        except BaseException:
            self.discard(identifier)
            raise

    def selected(self, identifier, entity_ids):
        try:
            payload = json.loads(self.path(identifier, ".json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError("La preselección ya no está disponible. Selecciona de nuevo el CSV.") from None
        if payload["created_at"] < time() - self.ttl_seconds:
            raise ValueError("La preselección caducó. Selecciona de nuevo el CSV.")
        selected = set(entity_ids)
        if not selected:
            raise ValueError("Activa al menos un sensor antes de preparar los filtros.")
        if selected - {row["entity_id"] for row in payload["entities"]}:
            raise ValueError("La selección contiene sensores que no pertenecen a este CSV.")
        return selected

    def export_selected(self, identifier, entity_ids, destination, report):
        selected = self.selected(identifier, entity_ids)
        report("filtering_entities", 1, f"Seleccionando lecturas de {len(selected):,} sensores")
        rows = 0
        with self.path(identifier).open(encoding="utf-8-sig", newline="") as source, destination.open("w", encoding="utf-8", newline="") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if (row.get("entity_id") or "").strip() in selected:
                    writer.writerow(row)
                    rows += 1
        if not rows:
            raise ValueError("No hay lecturas de los sensores seleccionados.")
        return destination.stat().st_size
