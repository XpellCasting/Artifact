"""Independent, disk-backed workbench for human review of historical evidence.

Nothing here trains a filter, changes normality, or assigns a diagnosis. Raw
uploads are immutable; annotations and their revisions live in a separate DB.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import unquote
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from .Tipos_Entidad import infer_kind, KINDS
from .Estados_Validos import valid_state, VALIDITY_VERSION
from .Referencia_Revision import describe_annotation
from .CSV_Etiquetado import labelled_columns, read_label

UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
PARTITIONS = {"development", "validation", "test"}
DIAGNOSTIC_LABELS = {"normal", "environmental_anomaly", "sensor_fault"}
INDEPENDENT_EVIDENCE = {
    "maintenance_log", "controlled_action", "reference_sensor", "system_log",
}
INVALID_STATES = {"", "unknown", "unavailable", "none", "null", "nan", "inf", "-inf", "infinity", "-infinity"}
BINARY_PAIRS = ({"on", "off"}, {"true", "false"}, {"open", "closed"}, {"home", "not_home"})
BINARY_DOMAINS = {"binary_sensor", "switch", "input_boolean"}


class ReviewError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def parse_time(value: str) -> tuple[int, str]:
    """Require explicit offsets; do not guess the installation's local zone."""
    fraction = re.search(r"[Tt ]\d{2}:\d{2}:\d{2}[.,](\d+)", str(value))
    if fraction and any(digit != "0" for digit in fraction.group(1)[6:]):
        raise ReviewError("La precisión temporal máxima es de microsegundos; no se admiten fracciones significativas de más de seis dígitos.")
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError("missing timezone")
        parsed = parsed.astimezone(UTC)
        delta = parsed - EPOCH
        micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        return micros, parsed.isoformat()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ReviewError("Fecha inválida: use ISO 8601 con zona explícita (Z o ±HH:MM).") from exc


def iso_time(micros: int) -> str:
    return (EPOCH + timedelta(microseconds=micros)).isoformat()


def numeric_state(state: str) -> float | None:
    try:
        number = float(state.strip())
        return number if math.isfinite(number) else None
    except (ValueError, OverflowError):
        return None


class AnnotationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    entity_id: str
    source_row: int | None = Field(default=None, ge=2)
    context_entity_ids: list[str] = Field(default_factory=list, max_length=7)
    start: str
    end: str
    padding_minutes: float = Field(default=30, ge=0, le=10080)
    label: Literal["normal", "environmental_anomaly", "sensor_fault", "uncertain"] = "uncertain"
    fault_type: Literal["spike", "bias", "drift", "noise", "stuck", "unavailable", "state_replace"] | None = None
    status: Literal["pending", "reviewed", "draft", "verified"] = "pending"
    evidence_type: Literal[
        "maintenance_log", "controlled_action", "reference_sensor", "system_log",
        "manual_review", "other", "insufficient",
    ] = "manual_review"
    evidence: str = Field(default="", max_length=20000)
    reviewer: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=20000)


class ReviewStore:
    """SQLite indexes bound RAM to the upload buffer and the requested preview."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.sources = self.directory / "sources"
        self.temporary = self.directory / "temporary"
        self.sources.mkdir(parents=True, exist_ok=True)
        self.temporary.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "review.sqlite3"
        self._import_lock = Lock()
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL,
                    partition TEXT NOT NULL, filename TEXT NOT NULL,
                    metadata TEXT NOT NULL, source_name TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    dataset_id TEXT NOT NULL, source_row INTEGER NOT NULL,
                    entity_id TEXT NOT NULL, ts INTEGER NOT NULL,
                    timestamp TEXT NOT NULL, state TEXT NOT NULL,
                    numeric_value REAL, is_valid INTEGER NOT NULL,
                    raw_json TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, source_row),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                );
                CREATE INDEX IF NOT EXISTS events_entity_time
                    ON events(dataset_id, entity_id, ts, source_row);
                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
                );
                CREATE INDEX IF NOT EXISTS annotations_dataset ON annotations(dataset_id);
                CREATE TABLE IF NOT EXISTS annotation_revisions (
                    annotation_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (annotation_id, revision)
                );
            """)
            connection.execute("CREATE TABLE IF NOT EXISTS exploration_cache (dataset_id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL)")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=120)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def list_datasets(self, partition: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            datasets = [json.loads(row[0]) for row in connection.execute(
                "SELECT metadata FROM datasets ORDER BY rowid DESC")]
        if partition is not None:
            datasets = [item for item in datasets if item.get("partition") == partition]
        return [self._refresh_kinds(item) for item in datasets]

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT metadata FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
        if row is None:
            raise ReviewError("Histórico de revisión no encontrado.", 404)
        return self._refresh_kinds(json.loads(row[0]))

    def source_path(self, dataset_id: str) -> Path:
        """Resuelve la copia inmutable de una fuente ya importada."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT source_name FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
        if row is None:
            raise ReviewError("Histórico de revisión no encontrado.", 404)
        root = self.sources.resolve()
        path = (root / row["source_name"]).resolve()
        if path.parent != root:
            raise ReviewError("La ruta de la fuente importada no es válida.")
        return path

    def _refresh_kinds(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Actualiza catálogos anteriores sin tocar originales ni anotaciones."""
        if metadata.get("validity_version") != VALIDITY_VERSION:
            with self.connect() as connection:
                connection.create_function("valid_state", 1, lambda value: int(valid_state(value)))
                connection.execute("UPDATE events SET is_valid=valid_state(state) WHERE dataset_id=?", (metadata["dataset_id"],))
                connection.execute("DELETE FROM exploration_cache WHERE dataset_id=?", (metadata["dataset_id"],))
            metadata["validity_version"] = VALIDITY_VERSION
            metadata.pop("entity_kind_version", None)
        if metadata.get("entity_kind_version") == 1:
            return metadata
        with self.connect() as connection:
            for entity in metadata.get("entities", []):
                key = (metadata["dataset_id"], entity["entity_id"])
                states = [row[0] for row in connection.execute(
                    "SELECT DISTINCT state FROM events WHERE dataset_id=? AND entity_id=? AND is_valid=1 LIMIT 17", key)]
                counts = connection.execute(
                    "SELECT COUNT(*), COUNT(numeric_value) FROM events WHERE dataset_id=? AND entity_id=? AND is_valid=1", key).fetchone()
                first = connection.execute(
                    "SELECT raw_json FROM events WHERE dataset_id=? AND entity_id=? ORDER BY source_row LIMIT 1", key).fetchone()
                raw = dict(zip([column.strip().lower() for column in metadata["columns"]], json.loads(first[0]))) if first else {}
                entity["kind"] = infer_kind(
                    entity["entity_id"], states, unit=entity.get("unit", ""),
                    declared=raw.get("sensor_kind", ""), device_class=raw.get("device_class", ""),
                    all_numeric=bool(counts[0]) and counts[0] == counts[1], states_overflow=len(states) > 16)
            metadata["entity_kind_version"] = 1
            connection.execute("UPDATE datasets SET metadata=? WHERE dataset_id=?",
                               (json.dumps(metadata, ensure_ascii=False), metadata["dataset_id"]))
        return metadata

    def import_csv(self, path: Path, filename: str, partition: str) -> dict[str, Any]:
        if partition not in PARTITIONS:
            raise ReviewError("Partición desconocida: use development, validation o test.")
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
        with self._import_lock:
            with self.connect() as connection:
                existing = connection.execute(
                    "SELECT partition, metadata FROM datasets WHERE sha256=?", (sha,)
                ).fetchone()
                if existing:
                    if existing["partition"] != partition:
                        raise ReviewError("Esta misma fuente ya pertenece a otra partición; no se permite reutilizarla.", 409)
                    return self._refresh_kinds(json.loads(existing["metadata"]))
            dataset_id = hashlib.sha256(f"{sha}:{partition}".encode()).hexdigest()[:32]
            source_path = self.sources / f"{dataset_id}.csv"
            # Exclusive creation makes the source immutable even across processes.
            try:
                with Path(path).open("rb") as source, source_path.open("xb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            except FileExistsError as exc:
                raise ReviewError("Ya hay una importación de esta fuente en curso.", 409) from exc
            try:
                return self._index_source(source_path, dataset_id, sha, filename, partition)
            except BaseException:
                source_path.unlink(missing_ok=True)
                raise

    def _index_source(self, source_path: Path, dataset_id: str, sha: str,
                      filename: str, partition: str) -> dict[str, Any]:
        entities: dict[str, dict[str, Any]] = {}
        rejected: dict[str, int] = {}
        rejection_examples: list[dict[str, Any]] = []
        rows = indexed = 0
        start_us = end_us = None
        with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(8192)
            stream.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(stream, dialect=dialect)
            headers = next(reader, None)
            if not headers:
                raise ReviewError("CSV vacío o sin cabecera.")
            normalized = [header.strip().lower() for header in headers]
            if len(set(normalized)) != len(normalized):
                raise ReviewError("El CSV contiene columnas duplicadas.")
            timestamp_key = "last_changed" if "last_changed" in normalized else "timestamp"
            if not {"entity_id", "state", timestamp_key}.issubset(normalized):
                raise ReviewError("El CSV requiere entity_id, state y last_changed o timestamp.")
            try:
                label_columns = labelled_columns(headers)
            except ValueError as exc:
                raise ReviewError(str(exc)) from exc
            imported_labels = imported_synthetic = 0
            entity_col = normalized.index("entity_id")
            state_col = normalized.index("state")
            timestamp_col = normalized.index(timestamp_key)
            unit_col = next((normalized.index(key) for key in ("unit_of_measurement", "unit") if key in normalized), None)
            attributes_col = normalized.index("attributes") if "attributes" in normalized else None
            kind_col = normalized.index("sensor_kind") if "sensor_kind" in normalized else None
            class_col = normalized.index("device_class") if "device_class" in normalized else None
            with self.connect() as connection:
                connection.execute(
                    "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?)",
                    (dataset_id, sha, partition, filename, "{}", source_path.name),
                )
                batch: list[tuple[Any, ...]] = []
                for source_row, raw in enumerate(reader, start=2):
                    rows += 1
                    reason = None
                    if len(raw) != len(headers):
                        reason = "column_count"
                    elif not raw[entity_col].strip():
                        reason = "missing_entity_id"
                    elif raw[entity_col] != raw[entity_col].strip():
                        # The label ID must join exactly to the unchanged source ID.
                        reason = "entity_id_has_outer_whitespace"
                    else:
                        try:
                            ts, _ = parse_time(raw[timestamp_col])
                        except ReviewError:
                            reason = "invalid_or_timezone_missing_timestamp"
                    if reason:
                        rejected[reason] = rejected.get(reason, 0) + 1
                        if len(rejection_examples) < 12:
                            rejection_examples.append({"source_row": source_row, "reason": reason})
                        continue
                    if label_columns is not None:
                        try:
                            imported = read_label(raw, label_columns)
                        except ValueError as exc:
                            raise ReviewError(f"Fila {source_row}: {exc}") from exc
                        imported_labels += int(imported["evaluable"])
                        imported_synthetic += int(imported["es_sintetica"] == "true")
                    entity_id = raw[entity_col].strip()
                    state = raw[state_col]
                    value = numeric_state(state)
                    valid = valid_state(state)
                    # Numeric infinities are invalid even when spelt +Infinity.
                    try:
                        if not math.isfinite(float(state)):
                            valid = False
                    except (ValueError, OverflowError):
                        pass
                    unit = raw[unit_col] if unit_col is not None else ""
                    if not unit and attributes_col is not None:
                        try:
                            attrs = json.loads(raw[attributes_col])
                            if isinstance(attrs, dict):
                                unit = str(attrs.get("unit_of_measurement") or "")
                        except (ValueError, TypeError):
                            pass
                    stats = entities.setdefault(entity_id, {
                        "entity_id": entity_id, "rows": 0, "unit": unit,
                        "numeric": 0, "text": 0, "states": set(), "states_overflow": False,
                    })
                    stats["rows"] += 1
                    declaration = raw[kind_col].strip().lower() if kind_col is not None else ""
                    if declaration in KINDS:
                        if stats.get("declared") and stats["declared"] != declaration:
                            raise ReviewError(f"Tipos declarados incompatibles para {entity_id}")
                        stats["declared"] = declaration
                    if class_col is not None and raw[class_col].strip():
                        stats["device_class"] = raw[class_col].strip()
                    if unit and not stats["unit"]:
                        stats["unit"] = unit
                    if valid:
                        stats["numeric" if value is not None else "text"] += 1
                        if len(stats["states"]) < 16:
                            stats["states"].add(state.strip().lower())
                        elif state.strip().lower() not in stats["states"]:
                            stats["states_overflow"] = True
                    indexed += 1
                    start_us = ts if start_us is None else min(start_us, ts)
                    end_us = ts if end_us is None else max(end_us, ts)
                    batch.append((dataset_id, source_row, entity_id, ts, raw[timestamp_col], state,
                                  value, int(valid), json.dumps(raw, ensure_ascii=False)))
                    if len(batch) >= 1000:
                        connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                        batch.clear()
                if batch:
                    connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                if indexed == 0:
                    raise ReviewError("No hay filas indexables: revise las columnas y las zonas horarias explícitas.")
                entity_list = []
                for entity_id, stats in sorted(entities.items()):
                    kind = infer_kind(entity_id, stats["states"], declared=stats.get("declared", ""),
                                      unit=stats["unit"], device_class=stats.get("device_class", ""),
                                      all_numeric=bool(stats["numeric"]) and not stats["text"],
                                      states_overflow=stats["states_overflow"])
                    entity_list.append({"entity_id": entity_id, "rows": stats["rows"], "kind": kind, "unit": stats["unit"]})
                metadata = {
                    "labelled_import": label_columns is not None, "imported_labels": imported_labels, "imported_synthetic": imported_synthetic,
                    "validity_version": VALIDITY_VERSION,
                    "entity_kind_version": 1,
                    "dataset_id": dataset_id, "filename": Path(unquote(filename).replace("\\", "/")).name,
                    "partition": partition, "sha256": sha, "rows": rows, "indexed_rows": indexed,
                    "rejected_rows": rows - indexed, "start": iso_time(start_us), "end": iso_time(end_us),
                    "entities": entity_list, "columns": headers, "timestamp_column": headers[timestamp_col],
                    "delimiter": dialect.delimiter, "encoding": "utf-8-sig",
                    "created_at": datetime.now(UTC).isoformat(),
                    "rejection_summary": rejected, "rejection_examples": rejection_examples,
                    "warnings": ([f"{rows - indexed} filas conservadas en la fuente original, pero no indexables; se excluyen de fragmentos y exportaciones."] if rows != indexed else [])
                        + ["El tipo de entidad se infiere para visualizar; no establece su normalidad ni su diagnóstico."],
                    "timestamp_precision": "microseconds; meaningful submicrosecond timestamps are rejected",
                }
                connection.execute("UPDATE datasets SET filename=?, metadata=? WHERE dataset_id=?", (
                    metadata["filename"], json.dumps(metadata, ensure_ascii=False), dataset_id,
                ))
        return metadata

    def _window(self, dataset_id: str, entity_ids: list[str], start: str, end: str,
                padding_minutes: float) -> tuple[dict, list[str], int, int, int, int]:
        dataset = self.dataset(dataset_id)
        ids = list(dict.fromkeys(entity_ids))
        if not 1 <= len(ids) <= 8:
            raise ReviewError("Seleccione entre una y ocho entidades.")
        known = {entry["entity_id"] for entry in dataset["entities"]}
        if any(entity_id not in known for entity_id in ids):
            raise ReviewError("Alguna entidad no pertenece al histórico seleccionado.")
        start_us, _ = parse_time(start)
        end_us, _ = parse_time(end)
        dataset_start, _ = parse_time(dataset["start"])
        dataset_end, _ = parse_time(dataset["end"])
        if end_us < start_us or start_us < dataset_start or end_us > dataset_end:
            raise ReviewError("El intervalo debe estar ordenado y dentro del histórico seleccionado.")
        if not math.isfinite(padding_minutes) or not 0 <= padding_minutes <= 10080:
            raise ReviewError("El contexto debe estar entre 0 y 10080 minutos.")
        padding = round(padding_minutes * 60 * 1_000_000)
        # Clamp only to Python's representable dates, never silently to the dataset.
        lower = max(start_us - padding, -62135596800000000)
        upper = min(end_us + padding, 253402300799999999)
        return dataset, ids, start_us, end_us, lower, upper

    def fragment(self, dataset_id: str, entity_ids: list[str], start: str, end: str,
                 padding_minutes: float = 30, max_points: int = 600) -> dict[str, Any]:
        if not 10 <= max_points <= 5000:
            raise ReviewError("max_points debe estar entre 10 y 5000 por entidad.")
        dataset, ids, start_us, end_us, lower, upper = self._window(
            dataset_id, entity_ids, start, end, padding_minutes,
        )
        kinds = {entry["entity_id"]: entry for entry in dataset["entities"]}
        series = []
        total = 0
        with self.connect() as connection:
            for entity_id in ids:
                kind = kinds[entity_id]["kind"]
                count, invalid = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(1-is_valid),0) FROM events WHERE dataset_id=? AND entity_id=? AND ts BETWEEN ? AND ?",
                    (dataset_id, entity_id, lower, upper),
                ).fetchone()
                total += count
                sampled = count > max_points
                points = []
                buckets: dict[int, dict[str, sqlite3.Row]] = {}
                bucket_count = max(1, max_points // 5)
                for row in connection.execute(
                    "SELECT source_row, ts, timestamp, state, numeric_value, is_valid, raw_json FROM events "
                    "WHERE dataset_id=? AND entity_id=? AND ts BETWEEN ? AND ? ORDER BY ts, source_row",
                    (dataset_id, entity_id, lower, upper),
                ):
                    if not sampled:
                        points.append(row)
                        continue
                    index = min(bucket_count - 1, (row["ts"] - lower) * bucket_count // max(1, upper - lower + 1))
                    bucket = buckets.setdefault(index, {"first": row, "last": row})
                    bucket["last"] = row
                    if not row["is_valid"]:
                        bucket.setdefault("invalid", row)
                    else:
                        def key(item):
                            return item["numeric_value"] if kind == "continuous" else item["state"]
                        if "min" not in bucket or key(row) < key(bucket["min"]):
                            bucket["min"] = row
                        if "max" not in bucket or key(row) > key(bucket["max"]):
                            bucket["max"] = row
                if sampled:
                    unique = {row["source_row"]: row for bucket in buckets.values() for row in bucket.values()}
                    points = sorted(unique.values(), key=lambda row: (row["ts"], row["source_row"]))
                states = sorted({row["state"] for row in points if row["is_valid"]}) if kind != "continuous" else []
                state_map = {state: index for index, state in enumerate(states)}
                series.append({
                    "entity_id": entity_id, "kind": kind, "unit": kinds[entity_id]["unit"],
                    "rows": count, "invalid_rows": invalid, "sampled": sampled, "states": states,
                    "points": [{"source_row": row["source_row"], "timestamp": row["timestamp"], "state": row["state"],
                                "value": (row["numeric_value"] if kind == "continuous" else state_map.get(row["state"])) if row["is_valid"] else None,
                                "is_valid": bool(row["is_valid"]),
                                "imported_label": read_label(json.loads(row["raw_json"]), labelled_columns(dataset["columns"]))} for row in points],
                })
            placeholders = ",".join("?" for _ in ids)
            preview = [dict(row) for row in connection.execute(
                f"SELECT source_row, entity_id, timestamp, state FROM events WHERE dataset_id=? AND entity_id IN ({placeholders}) AND ts BETWEEN ? AND ? ORDER BY ts, source_row LIMIT 150",
                [dataset_id, *ids, lower, upper],
            )]
        return {"dataset_id": dataset_id, "start": iso_time(start_us), "end": iso_time(end_us),
                "view_start": iso_time(lower), "view_end": iso_time(upper), "total_rows": total,
                "series": series, "rows": preview, "preview_truncated": total > len(preview),
                "sampling_method": "time_buckets_first_min_max_last_invalid",
                "sampling_note": "La vista reducida conserva extremos y un hueco inválido por bloque; puede omitir transiciones o huecos adicionales. La exportación conserva todas las filas indexables."}

    def list_annotations(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        if dataset_id:
            self.dataset(dataset_id)
        with self.connect() as connection:
            query = "SELECT payload FROM annotations"
            parameters = []
            if dataset_id:
                query += " WHERE dataset_id=?"
                parameters.append(dataset_id)
            return [describe_annotation(json.loads(row[0])) for row in connection.execute(query + " ORDER BY rowid DESC", parameters)]

    def annotation(self, annotation_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM annotations WHERE annotation_id=?", (annotation_id,)).fetchone()
        if row is None:
            raise ReviewError("Anotación no encontrada.", 404)
        return describe_annotation(json.loads(row[0]))

    def save_annotation(self, payload: dict[str, Any] | AnnotationInput,
                        annotation_id: str | None = None) -> dict[str, Any]:
        try:
            annotation = payload if isinstance(payload, AnnotationInput) else AnnotationInput.model_validate(payload)
        except ValidationError as exc:
            raise ReviewError(str(exc)) from exc
        if annotation.label != "sensor_fault" and annotation.fault_type is not None:
            raise ReviewError("El tipo de falla solo corresponde a una etiqueta de falla de sensor.")
        if annotation.status == "reviewed" and (len(annotation.reviewer.strip()) < 2 or len(annotation.evidence.strip()) < 12):
            raise ReviewError("Marcar Revisado requiere responsable y una descripción de lo observado de al menos doce caracteres. No requiere evidencia externa.")
        context_ids = [entity_id for entity_id in dict.fromkeys(annotation.context_entity_ids) if entity_id != annotation.entity_id]
        dataset, ids, start_us, end_us, _, _ = self._window(
            annotation.dataset_id, [annotation.entity_id, *context_ids], annotation.start,
            annotation.end, annotation.padding_minutes,
        )
        if annotation.source_row is not None:
            with self.connect() as connection:
                point = connection.execute("SELECT entity_id,ts FROM events WHERE dataset_id=? AND source_row=?", (annotation.dataset_id, annotation.source_row)).fetchone()
            if point is None or point["entity_id"] != annotation.entity_id or start_us != end_us or point["ts"] != start_us:
                raise ReviewError("La etiqueta puntual debe corresponder a la fila, entidad y fecha exactas del registro.")
            kind = next(item["kind"] for item in dataset["entities"] if item["entity_id"] == annotation.entity_id)
            allowed = {"spike", "bias", "drift", "noise", "stuck", "unavailable"} if kind == "continuous" else {"stuck", "unavailable", "state_replace"}
            if annotation.label == "sensor_fault" and annotation.fault_type not in allowed:
                raise ReviewError("Selecciona un tipo de falla compatible con esta entidad.")
        if annotation.status == "verified" and (
            annotation.label not in DIAGNOSTIC_LABELS
            or annotation.evidence_type not in INDEPENDENT_EVIDENCE
            or len(annotation.reviewer.strip()) < 2
            or len(annotation.evidence.strip()) < 12
        ):
            raise ReviewError("Verificar exige una clase diagnóstica, revisor y evidencia concreta independiente (registro de mantenimiento/sistema, acción controlada o sensor de referencia). La inspección visual sola queda como borrador.")
        with self.connect() as connection:
            # Serializes revision increments across concurrent requests.
            connection.execute("BEGIN IMMEDIATE")
            previous = None
            if annotation_id:
                record = connection.execute("SELECT payload FROM annotations WHERE annotation_id=?", (annotation_id,)).fetchone()
                if record is None:
                    raise ReviewError("Anotación no encontrada.", 404)
                previous = json.loads(record[0])
                if previous["dataset_id"] != annotation.dataset_id:
                    raise ReviewError("Una anotación no puede cambiar de histórico.", 409)
            if annotation.status in {"reviewed", "verified"} and annotation.label in DIAGNOSTIC_LABELS:
                for record in connection.execute(
                    "SELECT payload FROM annotations WHERE dataset_id=?", (annotation.dataset_id,),
                ):
                    other = json.loads(record[0])
                    if (other["annotation_id"] != annotation_id
                            and other["entity_id"] == annotation.entity_id
                            and other.get("source_row") == annotation.source_row
                            and other["status"] in {"reviewed", "verified"}
                            and (other["label"] != annotation.label or
                                 (annotation.label == "sensor_fault" and other.get("fault_type") and annotation.fault_type
                                  and other["fault_type"] != annotation.fault_type))
                            and parse_time(other["start"])[0] <= end_us
                            and parse_time(other["end"])[0] >= start_us):
                        label = {"normal": "Normal", "sensor_fault": "Falla del sensor", "environmental_anomaly": "Anomalía ambiental"}.get(other["label"], other["label"])
                        raise ReviewError(
                            f"No se guardó: {annotation.entity_id} ya tiene una anotación revisada como {label} "
                            f"entre {other['start']} y {other['end']} (anotación {other['annotation_id']}). "
                            "El intervalo se solapa con la nueva interpretación. En Anotaciones del histórico, "
                            "usa Abrir y editar para corregir la existente, o selecciona otro intervalo. "
                            "Las fallas sintéticas se registran en Generar inyecciones; su base original puede seguir etiquetada como Normal.", 409)
            now = datetime.now(UTC).isoformat()
            identifier = annotation_id or uuid4().hex
            saved = {**annotation.model_dump(), "context_entity_ids": ids[1:],
                     "start": iso_time(start_us), "end": iso_time(end_us),
                     "annotation_id": identifier, "id": identifier,
                     "source_sha256": dataset["sha256"], "partition": dataset["partition"],
                     "created_at": previous["created_at"] if previous else now, "updated_at": now,
                     "revision": previous["revision"] + 1 if previous else 1,
                     "evaluable": annotation.status in {"reviewed", "verified"} and annotation.label in DIAGNOSTIC_LABELS}
            saved = describe_annotation(saved)
            serialized = json.dumps(saved, ensure_ascii=False)
            connection.execute("INSERT INTO annotations VALUES (?, ?, ?) ON CONFLICT(annotation_id) DO UPDATE SET payload=excluded.payload",
                               (identifier, annotation.dataset_id, serialized))
            connection.execute("INSERT INTO annotation_revisions VALUES (?, ?, ?)",
                               (identifier, saved["revision"], serialized))
        return saved

    def export_annotation(self, annotation_id: str) -> Path:
        annotation = self.annotation(annotation_id)
        dataset, ids, start_us, end_us, lower, upper = self._window(
            annotation["dataset_id"], [annotation["entity_id"], *annotation["context_entity_ids"]],
            annotation["start"], annotation["end"], annotation["padding_minutes"],
        )
        handle, archive_name = tempfile.mkstemp(prefix="review-export-", suffix=".zip", dir=self.temporary)
        os.close(handle)
        archive_path = Path(archive_name)
        exported = 0
        target_rows = 0
        try:
            with self.connect() as connection, zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                placeholders = ",".join("?" for _ in ids)
                with archive.open("events.csv", "w", force_zip64=True) as binary:
                    with io.TextIOWrapper(binary, encoding="utf-8", newline="") as output:
                        writer = csv.writer(output)
                        writer.writerow(dataset["columns"])
                        for row in connection.execute(
                            f"SELECT raw_json FROM events WHERE dataset_id=? AND entity_id IN ({placeholders}) AND ts BETWEEN ? AND ? ORDER BY ts, source_row",
                            [dataset["dataset_id"], *ids, lower, upper],
                        ):
                            writer.writerow(json.loads(row[0]))
                            exported += 1
                target_rows = connection.execute(
                    "SELECT COUNT(*) FROM events WHERE dataset_id=? AND entity_id=? AND ts BETWEEN ? AND ?",
                    (dataset["dataset_id"], annotation["entity_id"], start_us, end_us),
                ).fetchone()[0]
                truth_fields = ["annotation_id", "dataset_id", "source_sha256", "partition", "entity_id", "source_row", "start", "end", "label", "fault_type", "status", "review_status", "reference_origin", "externally_verified", "evaluation_scope", "normality_basis", "evaluable", "evidence_type", "evidence", "reviewer", "notes", "revision"]
                with archive.open("ground_truth.csv", "w") as binary:
                    with io.TextIOWrapper(binary, encoding="utf-8", newline="") as output:
                        writer = csv.DictWriter(output, fieldnames=truth_fields)
                        writer.writeheader()
                        writer.writerow({field: ("true" if annotation[field] else "false") if field in {"evaluable", "externally_verified"} else annotation.get(field) for field in truth_fields})
                manifest = {
                    "schema_version": "1.0", "exported_at": datetime.now(UTC).isoformat(),
                    "dataset": {key: dataset[key] for key in ("dataset_id", "filename", "partition", "sha256", "rows", "indexed_rows", "rejected_rows", "rejection_summary", "rejection_examples", "columns", "timestamp_column", "encoding", "delimiter")},
                    "annotation": annotation, "selected_entities": ids,
                    "label_interval": {"start": iso_time(start_us), "end": iso_time(end_us), "inclusive": True, "entity_id": annotation["entity_id"]},
                    "context_interval": {"start": iso_time(lower), "end": iso_time(upper), "inclusive": True},
                    "exported_rows": exported, "target_rows_in_label_interval": target_rows,
                    "raw_field_semantics": "events.csv preserves original column names and decoded field strings, including original state and timestamp strings. It is chronologically ordered by timestamp then source row. UTF-8 and comma CSV serialization may differ from the immutable original file. Source SHA256 refers to original uploaded bytes, not events.csv.",
                    "ground_truth_semantics": "One reference annotation for the target entity and interval only. reviewed means the researcher completed historical review, not external verification. Normal historical references assume normality after inspection. evaluable=true permits comparison against the assigned reference within evaluation_scope, never claims confirmed real-world faults. uncertain may be reviewed but is excluded from diagnostic metrics. Context and padding are not labelled. Legacy verified retains its original evidence-backed assertion.",
                    "rejected_rows_semantics": "Unindexable source rows remain in the immutable original; their time/entity membership cannot be established and they are excluded from events.csv. Source-wide counts and examples are recorded above.",
                }
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            return archive_path
        except BaseException:
            archive_path.unlink(missing_ok=True)
            raise


def crear_router_revision(directory: Path) -> APIRouter:
    store = ReviewStore(directory)
    router = APIRouter(prefix="/review", tags=["Revisión manual de fragmentos"])

    def translate(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except ReviewError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except (UnicodeError, csv.Error) as exc:
            raise HTTPException(status_code=400, detail="CSV ilegible: use UTF-8 y una estructura tabular válida.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets")
    def datasets():
        return {"datasets": store.list_datasets()}

    @router.post("/datasets/upload")
    async def upload(request: Request, partition: str = "development"):
        if partition not in PARTITIONS:
            raise HTTPException(status_code=400, detail="Partición desconocida.")
        filename = request.headers.get("x-filename", "history.csv")
        handle, temporary_name = tempfile.mkstemp(prefix="upload-", suffix=".csv", dir=store.temporary)
        path = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                async for chunk in request.stream():
                    await run_in_threadpool(stream.write, chunk)
            return await run_in_threadpool(translate, store.import_csv, path, filename, partition)
        finally:
            path.unlink(missing_ok=True)

    @router.get("/datasets/{dataset_id}/fragment")
    def fragment(dataset_id: str, entity_ids: str, start: str, end: str,
                 padding_minutes: float = 30, max_points: int = 600):
        return translate(store.fragment, dataset_id, [value.strip() for value in entity_ids.split(",") if value.strip()],
                         start, end, padding_minutes, max_points)

    from .Exploracion_Historico import overview, entity_detail

    @router.get("/datasets/{dataset_id}/exploration")
    def exploration(dataset_id: str):
        return translate(overview, store, dataset_id)

    @router.get("/datasets/{dataset_id}/exploration/entity")
    def exploration_entity(dataset_id: str, entity_id: str, start: str, end: str):
        return translate(entity_detail, store, dataset_id, entity_id, start, end)

    @router.get("/annotations")
    def annotations(dataset_id: str | None = None):
        return {"annotations": translate(store.list_annotations, dataset_id)}

    @router.post("/annotations")
    def create_annotation(payload: dict[str, Any]):
        return translate(store.save_annotation, payload)

    @router.put("/annotations/{annotation_id}")
    def update_annotation(annotation_id: str, payload: dict[str, Any]):
        return translate(store.save_annotation, payload, annotation_id)

    @router.get("/annotations/{annotation_id}/export")
    def export(annotation_id: str):
        path = translate(store.export_annotation, annotation_id)
        return FileResponse(path, media_type="application/zip", filename=f"fragment-{annotation_id}.zip",
                            background=BackgroundTask(path.unlink, missing_ok=True))

    from .Inyeccion_Fragmentos import crear_router_inyecciones

    router.include_router(crear_router_inyecciones(store))
    return router
