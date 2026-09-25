"""Research workbench and immutable runtime snapshots for fault-pattern diagnosis.

Review labels and controlled interventions supply supervision. F1/F2 decisions
are inputs to diagnosis and are never turned into training labels here.
"""
from __future__ import annotations
from .Referencia_Revision import reference_metadata

import csv
from bisect import bisect_left, bisect_right
import hashlib
import io
import json
from itertools import chain
import math
import os
import sqlite3
import tempfile
import time
import zipfile
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock, Thread
from typing import Any, Literal
from uuid import uuid4
import importlib.util

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.background import BackgroundTask

from .Revision_Fragmentos import ReviewStore, ReviewError, parse_time, iso_time, UTC
from .Inyeccion_Fragmentos import InjectionStore
from .Diagnostico_Patrones import CATALOG, ENGINE_VERSION, build_reference, extract_features, diagnose, fit_model, predict_features
from .GAT_LSTM.config import Config as GATConfig, historical_manifest_path
from .GAT_LSTM import LABELS as GAT_LABELS
from .GAT_LSTM.pipeline import (build_sequence_dataset, canonical_sources, development_annotations, development_sources,
                                load_partition, load_pretraining_partition)
from .GAT_LSTM.audit import sha256_file
from .GAT_LSTM.storage import sample_summaries
from .GAT_LSTM.evaluation import metrics as gat_metrics

MAX_ROWS = 250_000
MAX_EXAMPLES = 20_000
_FILE_LOCK = RLock()


def _now():
    return datetime.now(UTC).isoformat()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _write(path: Path, value):
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        # Windows disallows replacement while a reader lacks delete sharing.
        # Coordinate our readers; retry brief external indexing/antivirus locks.
        with _FILE_LOCK:
            for attempt in range(8):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path):
    with _FILE_LOCK:
        return json.loads(path.read_text(encoding="utf-8"))


class DiagnosticRunSession:
    """Immutable diagnostic snapshot plus causal, per-entity event indexes.

    It removes repeated snapshot reads, reference parsing and SQLite commits from
    the timestamp-group hot path.  The caller owns the session and must flush it.
    """
    def __init__(self, workbench, source, run_id):
        self.workbench, self.source, self.run_id = workbench, source, run_id
        self.snapshot = workbench._run_snapshot(source, run_id)
        self.gat_snapshot = None
        if self.snapshot.get("model", {}).get("method") == "gat_lstm":
            from .GAT_LSTM.runtime import load_snapshot
            try:
                self.gat_snapshot = load_snapshot(str(workbench.directory), self.snapshot["model"]["model_id"])
            except ImportError:
                self.gat_snapshot = None
            if self.gat_snapshot is not None:
                self.window_seconds = self.gat_snapshot[3]["config"]["lookback_seconds"] + self.gat_snapshot[3]["config"]["feature_window_seconds"]
            else:
                cfg = self.snapshot["model"]["config"]
                self.window_seconds = (cfg["sequence_steps"] - 1) * cfg["grid_seconds"] + cfg["feature_window_seconds"]
        else:
            self.window_seconds = self.snapshot.get("model", {}).get("window_seconds", 3600)
        self.references = {}
        for reference in self.snapshot["references"]:
            end, _ = parse_time(reference["end"])
            self.references.setdefault(str(reference["entity_id"]), []).append((end, reference))
        for values in self.references.values(): values.sort(key=lambda item: item[0])
        self.events, self.seen, self.pending = {}, set(), []
        self.last_flush = time.monotonic(); self.stats = {"snapshot_loads": 1, "event_rows": 0, "decision_writes": 0, "flushes": 0, "write_ms": 0.0}

    def add_events(self, frame):
        if not isinstance(frame, pd.DataFrame) or frame.empty: return
        time_column = "timestamp_dt" if "timestamp_dt" in frame else "timestamp" if "timestamp" in frame else "last_changed"
        order = [time_column, "entity_id"] + (["_raw_index"] if "_raw_index" in frame else [])
        for ordinal, row in enumerate(frame.sort_values(order, kind="stable").to_dict("records")):
            at = pd.to_datetime(row.get(time_column), utc=True, errors="coerce")
            if pd.isna(at): continue
            entity = str(row.get("entity_id", ""))
            raw_index = row.get("_raw_index")
            source_row = int(raw_index) if isinstance(raw_index, (int, float)) and not pd.isna(raw_index) else ordinal
            key = (entity, at.value, source_row)
            if key in self.seen: continue
            self.seen.add(key)
            self.events.setdefault(entity, []).append((at, {"timestamp": at.isoformat(), "state": str(row.get("state", "")), "source_row": source_row}))
            self.stats["event_rows"] += 1
        if self.snapshot.get("model", {}).get("method") == "gat_lstm":
            latest = max((items[-1][0] for items in self.events.values() if items), default=None)
            if latest is not None:
                cutoff = latest - pd.Timedelta(seconds=self.window_seconds)
                for entity, items in list(self.events.items()):
                    kept = [item for item in items if item[0] >= cutoff]
                    if kept:
                        self.events[entity] = kept
                    else:
                        self.events.pop(entity, None)
                active_keys = {(entity, at.value, row.get("source_row"))
                               for entity, rows in self.events.items() for at, row in rows}
                self.seen.intersection_update(active_keys)

    def reference_for(self, entity_id, at):
        values = self.references.get(str(entity_id), [])
        index = bisect_left(values, at, key=lambda item: item[0])
        return values[index - 1][1] if index else None

    def rows_for(self, entity_id, at, window_seconds):
        values = self.events.get(str(entity_id), [])
        start = at - pd.Timedelta(seconds=window_seconds)
        left = bisect_left(values, start, key=lambda item: item[0]); right = bisect_right(values, at, key=lambda item: item[0])
        return [row for _, row in values[left:right]]

    def events_between(self, start, end, entity_ids):
        result = []
        for entity in entity_ids:
            values = self.events.get(str(entity), [])
            left = bisect_left(values, start, key=lambda item: item[0])
            right = bisect_right(values, end, key=lambda item: item[0])
            result.extend(row for _, row in values[left:right])
        return result

    def queue(self, decisions):
        self.pending.extend(decisions)
        if len(self.pending) >= 500 or time.monotonic() - self.last_flush >= 2: self.flush()

    def flush(self):
        if not self.pending: return
        rows = [(result["diagnosis_id"], self.snapshot["run_id"], self.source, result["entity_id"], result["timestamp"], _canonical(result)) for result in self.pending]
        started = time.perf_counter()
        with self.workbench._connect() as connection:
            connection.executemany("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)", rows)
        self.stats["decision_writes"] += len(rows); self.stats["flushes"] += 1; self.stats["write_ms"] += round((time.perf_counter() - started) * 1000, 3); self.pending.clear(); self.last_flush = time.monotonic()


class BuildInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotation_ids: list[str] = Field(default_factory=list, max_length=200)
    injection_ids: list[str] = Field(default_factory=list, max_length=500)
    window_seconds: int = Field(default=3600, ge=60, le=604800)
    stride_seconds: int = Field(default=1800, ge=60, le=604800)
    include_drafts: bool = False
    method: Literal["robust_nearest_prototype", "gat_lstm"] | None = None
    source_ids: list[str] = Field(default_factory=list, max_length=10)
    gat_config: dict[str, Any] | None = None


class DiagnosticWorkbench:
    def __init__(self, directory: Path):
        self.review = ReviewStore(directory)
        self.injections = InjectionStore(self.review)
        self.directory = Path(directory) / "diagnostics"
        for folder in ("references", "datasets", "models", "jobs", "weights", "preprocessors",
                       "graphs", "sequences", "experiments", "encoders", "configurations"):
            (self.directory / folder).mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "diagnostics.sqlite3"
        self.lock = RLock()
        self.job_lock = Lock()
        self.running_jobs: set[str] = set()
        with self._connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions(decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    source TEXT NOT NULL, entity_id TEXT NOT NULL, timestamp TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS decisions_source_time ON decisions(source,timestamp);
            """)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=60)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _path(self, folder, identifier):
        if not identifier or any(char not in "0123456789abcdef" for char in identifier) or len(identifier) != 32:
            raise ReviewError("Identificador de diagnóstico no válido.", 404)
        return self.directory / folder / f"{identifier}.json"

    def _read(self, folder, identifier):
        try:
            return _read_json(self._path(folder, identifier))
        except FileNotFoundError as exc:
            raise ReviewError("No se encontró el recurso de diagnóstico.", 404) from exc

    def _list(self, folder):
        return [_read_json(path) for path in (self.directory / folder).glob("*.json")]

    @staticmethod
    def _dataset_summary(dataset):
        return {key: value for key, value in dataset.items() if key != "examples"}

    @staticmethod
    def _model_summary(model):
        return {key: value for key, value in model.items() if key not in {"model", "training_examples"}}

    def status(self):
        active = self._active()
        references = [{key: ref.get(key) for key in ("reference_id", "entity_id", "sensor_kind", "start", "end", "rows", "annotation_ids", "partition", "exploratory")} for ref in self._list("references")]
        gat_available = importlib.util.find_spec("torch") is not None
        return {"mode": "active_model" if active else "catalogue", "active_model_id": active,
                "engine_version": ENGINE_VERSION, "catalog": CATALOG, "references": references,
                "capabilities": {"robust_nearest_prototype": True,
                                 "gat_lstm": gat_available,
                                 "pytorch_available": gat_available},
                "gat_configurations": self.configurations(),
                "gat_encoders": self.encoders(),
                "datasets": sorted([self._dataset_summary(item) for item in self._list("datasets")], key=lambda item: item["created_at"], reverse=True),
                "models": sorted([dict(self._model_summary(item), active=item["model_id"] == active) for item in self._list("models")], key=lambda item: item["created_at"], reverse=True)}

    def sources(self, gat_only=False):
        manifest_path = historical_manifest_path()
        training = [{key: value for key, value in item.items() if key != "path"}
                    for item in development_sources(manifest_path, self.review)]
        if gat_only:
            hashes = {item["sha256"] for item in training}
            annotations = development_annotations(self.review, hashes)
            datasets = self.review.list_datasets(partition="development")
            injections = []
        else:
            annotations = self.review.list_annotations()
            datasets = self.review.list_datasets()
            injections = [{key: value for key, value in item.items()
                           if key not in {"preview", "baseline_snapshot"}}
                          for item in self.injections.list_injections()]
        return {"annotations": annotations, "datasets": datasets,
                "training_sources": training,
                # Kept as a compatibility alias; it now contains only Development.
                "canonical_sources": training,
                "injections": injections}

    def build_gat_dataset(self, source_ids, annotation_ids, gat_config=None,
                          progress=lambda *_: None):
        if importlib.util.find_spec("torch") is None:
            raise ReviewError("GAT-LSTM requiere la dependencia opcional PyTorch; el backend convencional sigue disponible.", 503)
        config = GATConfig(**(gat_config or {}))
        manifest_path = historical_manifest_path()
        allowed = {item["source_id"] for item in development_sources(manifest_path, self.review)}
        if len(source_ids) != 1 or source_ids[0] not in allowed:
            raise ReviewError("GAT-LSTM admite una sola fuente de Desarrollo. Validación y Prueba quedan reservadas para eventos futuros.")
        dataset = build_sequence_dataset(self.review, source_ids, annotation_ids, config,
                                         self.directory, manifest_path, progress)
        dataset["audit_id"] = sha256_file(manifest_path)
        dataset["evaluation_status"] = "insufficient_labels"
        dataset["label_support"] = dataset["label_counts"]
        evidence_origins = set()
        for source in development_sources(manifest_path, self.review):
            if source["source_id"] not in dataset["labelled_source_ids"]:
                continue
            if source.get("imported_labels", 0) > source.get("imported_synthetic", 0):
                evidence_origins.add("historical_review")
            if source.get("imported_synthetic", 0):
                evidence_origins.add("controlled_injection")
        evidence_origins.update(self.review.annotation(identifier).get("evidence_origin", "unspecified")
                                for identifier in dataset["annotation_ids"])
        dataset["evidence_origins"] = sorted(evidence_origins)
        _write(self._path("datasets", dataset["dataset_id"]), dataset)
        progress(100, "Dataset listo: secuencias y configuración guardadas")
        return self._dataset_summary(dataset)

    def configurations(self):
        return sorted([_read_json(path) for path in (self.directory / "configurations").glob("*.json")],
                      key=lambda item: item.get("created_at", ""), reverse=True)

    def encoders(self):
        rows = []
        for path in (self.directory / "encoders").glob("*.json"):
            item = _read_json(path)
            rows.append({key: item.get(key) for key in
                         ("encoder_id", "created_at", "schema_version", "metrics", "dataset_id",
                          "support_train_fit", "support_dev_tune", "can_activate")})
        return sorted(rows, key=lambda item: item.get("created_at", ""), reverse=True)

    def save_gat_configuration(self, payload):
        name = str(payload.get("name", "GAT-LSTM")).strip()[:100] or "GAT-LSTM"
        config_data = payload.get("config")
        if not isinstance(config_data, dict):
            raise ReviewError("La configuración debe ser un objeto JSON.")
        config = GATConfig(**config_data)
        identifier = uuid4().hex
        saved = {"configuration_id": identifier, "name": name, "config": config.as_dict(),
                 "created_at": _now(), "method": "gat_lstm"}
        _write(self._path("configurations", identifier), saved)
        return saved

    def _active(self):
        path = self.directory / "active.json"
        return _read_json(path).get("model_id") if path.exists() else None

    def job(self, identifier):
        result = self._read("jobs", identifier)
        if result["status"] in {"queued", "running"} and identifier not in self.running_jobs:
            result.update(status="failed", error="El servicio se reinició antes de terminar. Inicia de nuevo esta operación.")
        return result

    def start_job(self, action, *args):
        if not self.job_lock.acquire(blocking=False):
            raise ReviewError("Hay una preparación o entrenamiento en curso. Espera a que termine.", 409)
        identifier = uuid4().hex
        job = {"job_id": identifier, "status": "queued", "progress": 0, "message": "Operación preparada",
               "operation": action.__name__, "created_at": _now()}
        self.running_jobs.add(identifier)
        try:
            _write(self._path("jobs", identifier), job)
        except BaseException:
            self.running_jobs.discard(identifier)
            self.job_lock.release()
            raise

        def progress(value, message, **details):
            now = _now()
            next_phase = details.get("phase", job.get("phase"))
            if next_phase and next_phase != job.get("phase"):
                job["phase_started_at"] = now
            job.update(status="running", progress=value, message=message,
                       updated_at=now, **details)
            _write(self._path("jobs", identifier), job)

        def run():
            try:
                result = action(*args, progress=progress)
                message = {"build_gat_dataset": "Dataset listo: secuencias y configuración guardadas",
                           "train_gat": "Entrenamiento completado: candidato listo"}.get(action.__name__, "Operación completada")
                job.update(status="completed", progress=100, message=message, result=result)
            except Exception as exc:
                job.update(status="failed", message="No se pudo completar la operación", error=str(exc))
            finally:
                job["finished_at"] = job["updated_at"] = _now()
                try:
                    _write(self._path("jobs", identifier), job)
                finally:
                    self.running_jobs.discard(identifier)
                    self.job_lock.release()

        Thread(target=run, daemon=True, name=f"diagnostic-{identifier[:8]}").start()
        return dict(job)

    def _rows(self, annotation, padding=0):
        start, _ = parse_time(annotation["start"])
        end, _ = parse_time(annotation["end"])
        with self.review.connect() as connection:
            rows = connection.execute("SELECT source_row,timestamp,state FROM events WHERE dataset_id=? AND entity_id=? AND ts BETWEEN ? AND ? ORDER BY ts,source_row LIMIT ?",
                (annotation["dataset_id"], annotation["entity_id"], start - padding * 1_000_000, end, MAX_ROWS + 1)).fetchall()
        if len(rows) > MAX_ROWS:
            raise ReviewError("La referencia supera 250 000 lecturas. Divide la anotación en intervalos más pequeños.")
        return [dict(row) for row in rows]

    @staticmethod
    def _check_annotation(annotation, include_drafts):
        if len(annotation.get("reviewer", "").strip()) < 2 or len(annotation.get("evidence", "").strip()) < 12:
            raise ReviewError(f"Completa revisor y evidencia de la anotación {annotation['annotation_id']} antes de preparar ejemplos.")
        if not annotation.get("evaluable") and not include_drafts:
            raise ReviewError("La selección incluye referencias pendientes o inciertas. Completa la revisión del histórico o habilita explícitamente el uso exploratorio; no se exige evidencia externa.")

    def build_dataset(self, payload, progress=lambda *_: None):
        try:
            params = BuildInput.model_validate(payload)
        except ValidationError as exc:
            raise ReviewError(str(exc)) from exc
        if not params.annotation_ids and not params.injection_ids:
            raise ReviewError("Selecciona anotaciones o casos de inyección.")
        annotations = [self.review.annotation(identifier) for identifier in dict.fromkeys(params.annotation_ids)]
        cases = [self.injections.injection(identifier) for identifier in dict.fromkeys(params.injection_ids)]
        baselines = {(item["annotation_id"], item["revision"]): item for item in annotations if item["label"] == "normal"}
        for case in cases:
            baseline = case["baseline_snapshot"]
            baselines[(baseline["annotation_id"], baseline["revision"])] = baseline
        if not baselines:
            raise ReviewError("Selecciona al menos una referencia Normal o una inyección que conserve su referencia normal.")
        references, row_cache, examples, warnings = {}, {}, [], []
        for index, (key, annotation) in enumerate(baselines.items()):
            self._check_annotation(annotation, params.include_drafts)
            source = self.review.dataset(annotation["dataset_id"])
            kind = next(entity["kind"] for entity in source["entities"] if entity["entity_id"] == annotation["entity_id"])
            rows = self._rows(annotation)
            reference = build_reference(rows, annotation["entity_id"], kind)
            reference.update(entity_id=annotation["entity_id"], sensor_kind=kind, start=annotation["start"], end=annotation["end"],
                rows=len(rows), annotation_ids=[annotation["annotation_id"]], baseline_snapshot=annotation,
                partition=annotation["partition"], source_sha256=annotation["source_sha256"], exploratory=not annotation.get("evaluable", False))
            reference["reference_id"] = hashlib.sha256(_canonical(reference).encode()).hexdigest()[:32]
            references[key] = reference
            row_cache[key] = rows
            progress(5 + 20 * (index + 1) / len(baselines), "Construyendo referencias de funcionamiento normal")

        def append_examples(annotation, rows, label, provenance, reference, injection=None):
            lower, _ = parse_time(injection["start"] if injection else annotation["start"])
            upper, _ = parse_time(injection["end"] if injection else annotation["end"])
            window = params.window_seconds * 1_000_000
            stride = params.stride_seconds * 1_000_000
            candidates = [lower] if label == "spike" else list(range(min(lower + window, upper), upper + 1, stride))
            if not candidates or candidates[-1] != upper:
                candidates.append(upper)
            parsed = sorted([(parse_time(row["timestamp"])[0], row) for row in rows], key=lambda item: (item[0], item[1].get("source_row", 0)))
            # Binary searches avoid scanning the complete fragment for every window.
            from bisect import bisect_left, bisect_right
            times = [item[0] for item in parsed]
            for at in candidates:
                selected = [item[1] for item in parsed[bisect_left(times, at - window):bisect_right(times, at)]]
                if not selected:
                    continue
                if injection and injection["kind"] == "spike" and injection.get("spike_source_row") is not None:
                    selected = [row for row in selected if parse_time(row["timestamp"])[0] != at or row["source_row"] <= injection["spike_source_row"]]
                if len(examples) >= MAX_EXAMPLES:
                    raise ReviewError("El conjunto supera 20 000 ventanas. Aumenta el paso temporal o divide la selección.")
                features = extract_features(selected, reference, timestamp=iso_time(at), window_seconds=params.window_seconds,
                    allow_retrospective_reference=True)
                identifier = hashlib.sha256(_canonical([annotation["annotation_id"], annotation["revision"], injection and injection["injection_id"], at, params.window_seconds, label]).encode()).hexdigest()[:32]
                examples.append({"example_id": identifier, "group_id": annotation["annotation_id"],
                    "reference_origin": "controlled_injection" if injection else "historical_review",
                    "evaluation_scope": "controlled_injection" if injection else reference_metadata(annotation)["evaluation_scope"],
                    "baseline_normality_basis": reference_metadata(annotation)["normality_basis"],
                    "externally_verified": reference_metadata(annotation)["externally_verified"] if not injection else False,
                    "annotation_id": annotation["annotation_id"], "annotation_revision": annotation["revision"],
                    "injection_id": injection["injection_id"] if injection else None,
                    "entity_id": annotation["entity_id"], "sensor_kind": reference["sensor_kind"],
                    "partition": annotation["partition"], "label": label, "provenance": provenance,
                    "timestamp": iso_time(at), "window_start": selected[0]["timestamp"], "window_end": iso_time(at),
                    "baseline_start": annotation["start"], "baseline_end": annotation["end"],
                    "source_sha256": annotation["source_sha256"], "reference_id": reference["reference_id"],
                    "source_dataset_id": annotation["dataset_id"],
                    "cutoff_source_row": injection.get("spike_source_row") if injection and injection["kind"] == "spike" else None,
                    "evaluable": bool((injection or annotation).get("evaluable", False)),
                    "features": features})

        # Every selected synthetic case carries its original normal control. Controls
        # appear once per frozen baseline revision, never once per injected variant.
        for key, annotation in baselines.items():
            append_examples(annotation, row_cache[key], "normal", "reviewed_real", references[key])
        for annotation in annotations:
            if annotation["label"] == "normal":
                continue
            self._check_annotation(annotation, params.include_drafts)
            label = annotation.get("fault_type") if annotation["label"] == "sensor_fault" else annotation["label"]
            if label not in {item["fault_type"] for item in CATALOG} | {"environmental_anomaly"}:
                warnings.append(f"Anotación {annotation['annotation_id']} omitida: falta un tipo de falla conocido o una clase confirmable.")
                continue
            prior = [ref for ref in references.values() if ref["entity_id"] == annotation["entity_id"]
                and ref["partition"] == annotation["partition"] and parse_time(ref["end"])[0] < parse_time(annotation["start"])[0]]
            if not prior:
                raise ReviewError("Para un episodio real, selecciona también una referencia normal anterior de esa entidad.")
            reference = max(prior, key=lambda ref: parse_time(ref["end"])[0])
            append_examples(annotation, self._rows(annotation, params.window_seconds), label, "reviewed_real", reference)
        for index, case in enumerate(cases):
            annotation = case["baseline_snapshot"]
            rows = []
            # Zip rows are aligned with source order. Reuse indexed source-row IDs,
            # including duplicate timestamps, instead of inventing CSV line numbers.
            with self.review.connect() as connection:
                source_rows = connection.execute("SELECT source_row FROM events WHERE dataset_id=? AND entity_id IN (" + ",".join("?" for _ in [annotation["entity_id"], *annotation["context_entity_ids"]]) + ") AND ts BETWEEN ? AND ? ORDER BY ts,source_row",
                    [annotation["dataset_id"], annotation["entity_id"], *annotation["context_entity_ids"], parse_time(case["view_start"])[0], parse_time(case["view_end"])[0]])
                with zipfile.ZipFile(self.injections.export_injection(case["injection_id"])) as archive, archive.open("injected.csv") as binary:
                    reader = csv.reader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
                    columns = [name.strip().lower() for name in next(reader)]
                    entity_col, state_col = columns.index("entity_id"), columns.index("state")
                    timestamp_col = columns.index("last_changed" if "last_changed" in columns else "timestamp")
                    for raw, source_row in zip(reader, source_rows, strict=True):
                        if raw[entity_col] == annotation["entity_id"]:
                            rows.append({"source_row": source_row[0], "timestamp": raw[timestamp_col], "state": raw[state_col]})
            append_examples(annotation, rows, case["kind"], "semisynthetic", references[(annotation["annotation_id"], annotation["revision"])], case)
            progress(25 + 60 * (index + 1) / max(1, len(cases)), "Extrayendo ventanas etiquetadas de los casos")
        # Deduplicate overlapping selections of the exact same frozen example.
        examples = list({example["example_id"]: example for example in examples}.values())
        if not examples:
            raise ReviewError("La selección no produjo ventanas con observaciones.")
        label_counts, partition_counts = {}, {}
        for example in examples:
            label_counts[example["label"]] = label_counts.get(example["label"], 0) + 1
            partition_counts[example["partition"]] = partition_counts.get(example["partition"], 0) + 1
        identifier = uuid4().hex
        dataset = {"dataset_id": identifier, "created_at": _now(), "example_count": len(examples),
            "label_counts": label_counts, "partition_counts": partition_counts,
            "exploratory": any(not example["evaluable"] for example in examples),
            "reference_ids": list(dict.fromkeys(ref["reference_id"] for ref in references.values())),
            "configuration": params.model_dump(), "engine_version": ENGINE_VERSION,
            "warnings": warnings + ["Las ventanas de un episodio son dependientes; el grupo y la partición deben conservarse.",
                "Los controles y variantes sintéticas se describen respecto de la referencia normal revisada. Esta comparación es experimental, no una evaluación prospectiva del pipeline.",
                "Las características usan observaciones hasta cada decisión. Las etiquetas y los parámetros de inyección se exportan separados de las entradas."],
            "examples_preview": examples[:50], "examples": examples}
        for reference in references.values():
            _write(self._path("references", reference["reference_id"]), reference)
        _write(self._path("datasets", identifier), dataset)
        progress(100, "Referencias y conjunto guardados")
        return self._dataset_summary(dataset)

    def train(self, dataset_id, progress=lambda *_: None):
        dataset = self._read("datasets", dataset_id)
        examples = [example for example in dataset["examples"] if example["partition"] == "development"]
        if not examples:
            raise ReviewError("El conjunto no contiene ejemplos de Desarrollo para entrenar.")
        progress(20, "Aprendiendo prototipos únicamente con Desarrollo")
        model = fit_model(examples)
        identifier = uuid4().hex
        model["model_version"] = identifier
        trained = {"model_id": identifier, "created_at": _now(), "dataset_id": dataset_id,
            "engine_version": ENGINE_VERSION, "method": "robust_nearest_prototype",
            "classes": model["classes"], "training_count": sum(model["support"].values()),
            "excluded_count": len(examples) - sum(model["support"].values()),
            "excluded_examples": model["excluded_examples"],
            "training_groups": sorted({example["group_id"] for example in examples}),
            "training_end": max([example["baseline_end"] for example in examples] + [example["window_end"] for example in examples], key=lambda value: parse_time(value)[0]),
            "reference_ids": list(dict.fromkeys(example["reference_id"] for example in examples)),
            "window_seconds": dataset["configuration"]["window_seconds"],
            "exploratory": any(not example["evaluable"] for example in examples),
            "validated": False, "validation": None, "evaluations": [],
            "model": model, "training_examples": [{key: example[key] for key in ("group_id", "entity_id", "source_sha256", "baseline_start", "baseline_end", "window_start", "window_end")} for example in examples]}
        _write(self._path("models", identifier), trained)
        progress(100, "Modelo candidato guardado; todavía no está activo")
        return self._model_summary(trained)

    def train_gat(self, dataset_id, encoder_id=None, gat_config=None,
                  progress=lambda *_: None):
        if importlib.util.find_spec("torch") is None:
            raise ReviewError("GAT-LSTM no está disponible: falta PyTorch.", 503)
        dataset = self._read("datasets", dataset_id)
        if dataset.get("method") != "gat_lstm":
            raise ReviewError("El conjunto seleccionado no fue preparado para GAT-LSTM.")
        config = GATConfig(**(gat_config or dataset["config"]))
        built_config = dataset["config"]
        for key in ("grid_seconds", "sequence_steps", "feature_window_seconds", "graph_top_k",
                    "max_nodes", "use_entropy", "use_dtw"):
            if getattr(config, key) != built_config[key]:
                raise ReviewError(f"La configuración {key} cambió; prepara nuevamente el conjunto.")
        progress(1, "Cargando el dataset para entrenar el candidato", phase="loading")
        train_samples = load_partition(dataset, "train_fit")
        dev_samples = load_partition(dataset, "dev_tune")
        support = {part: dataset.get("label_counts", {}).get(part, {})
                   for part in ("train_fit", "dev_tune")}
        missing = {part: [label for label in GAT_LABELS if counts.get(label, 0) == 0]
                   for part, counts in support.items()}
        missing = {part: labels for part, labels in missing.items() if labels}
        episode_support = {part: dataset.get("episode_counts", {}).get(part, {})
                           for part in ("train_fit", "dev_tune")}
        missing_episodes = {part: [label for label in GAT_LABELS if counts.get(label, 0) == 0]
                            for part, counts in episode_support.items()}
        missing_episodes = {part: labels for part, labels in missing_episodes.items() if labels}
        if missing or missing_episodes:
            raise ReviewError(json.dumps({"code": "insufficient_labels", "missing_classes": missing,
                                          "support": support, "episode_support": episode_support,
                                          "missing_episode_classes": missing_episodes}, ensure_ascii=False))
        preprocessor = json.loads((Path(dataset["sequence_dir"]) / "preprocessor.json").read_text(encoding="utf-8"))
        graph = json.loads((Path(dataset["sequence_dir"]) / "graph.json").read_text(encoding="utf-8"))
        entity_ids = preprocessor["entity_ids"]
        category_sizes = [max(preprocessor["entities"][e].get("vocab", {}).values(), default=2) + 1
                          for e in entity_ids]
        entity_kinds = [preprocessor["entities"][e]["kind"] for e in entity_ids]
        encoder = None
        if encoder_id:
            encoder_path = self.directory / "encoders" / f"{encoder_id}.json"
            encoder_weights = self.directory / "encoders" / f"{encoder_id}.pt"
            if len(encoder_id) != 32 or any(ch not in "0123456789abcdef" for ch in encoder_id):
                raise ReviewError("ID de encoder inválido.")
            if not encoder_path.is_file() or not encoder_weights.is_file():
                raise ReviewError("No se encontró el encoder indicado.")
            encoder_meta = _read_json(encoder_path)
            if encoder_meta.get("can_activate") is not False or hashlib.sha256(encoder_weights.read_bytes()).hexdigest() != encoder_meta.get("weights_sha256"):
                raise ReviewError("El encoder no es compatible o su hash no coincide.")
            if (encoder_meta.get("schema_version") != config.schema_version
                    or encoder_meta.get("preprocessor", {}).get("entity_ids") != entity_ids
                    or encoder_meta.get("preprocessor", {}).get("entity_kinds") != preprocessor.get("entity_kinds")
                    or encoder_meta.get("graph", {}).get("graph_version") != graph.get("graph_version")):
                raise ReviewError("El encoder fue preentrenado con entidades o tipos distintos.")
            import torch
            encoder = torch.load(encoder_weights, map_location="cpu", weights_only=True)
        from .GAT_LSTM.training import train_classifier, save_candidate
        progress(5, "Iniciando entrenamiento GAT-LSTM solo con Desarrollo",
                 phase="training", epoch=0, max_epochs=config.max_epochs)
        def training_progress(row):
            percent = 5 + 80 * row["epoch"] / config.max_epochs
            message = (f"Época {row['epoch']}: pérdida train {row['train_loss']:.4f}, "
                       f"dev_tune {row['dev_loss']:.4f}")
            progress(min(85, percent), message, phase="training",
                     epoch=row["epoch"], max_epochs=config.max_epochs)
        def training_batch_progress(row):
            stage = "Ajustando pesos" if row["stage"] == "train" else "Evaluando en Desarrollo"
            percent = 5 + 80 * row["completed_batches"] / row["total_batches"]
            message = f"{stage} · época {row['epoch']} de {row['max_epochs']} · lote {row['batch']} de {row['batches']}"
            progress(percent, message, phase="training", **row)
        model, metrics = train_classifier(train_samples, dev_samples, config,
                                          category_sizes, entity_kinds, training_progress, encoder,
                                          batch_progress=training_batch_progress)
        progress(90, "Guardando pesos, grafo, preprocesador y política de rechazo", phase="saving")
        candidate = save_candidate(model, preprocessor, graph, config, self.directory, metrics)
        all_train_meta = [sample["metadata"] for sample in
                          chain(sample_summaries(train_samples), sample_summaries(dev_samples))]
        candidate.update({"dataset_id": dataset_id, "encoder_id": encoder_id,
            "entity_ids": entity_ids, "entity_kinds": entity_kinds, "classes": list(GAT_LABELS),
            "graph_version": graph["graph_version"],
            "training_count": sum(sum(sample["label_mask"]) for sample in sample_summaries(train_samples)),
            "training_contexts": [{key: m.get(key) for key in
                ("parent_episode_id", "context_start", "context_end", "context_entities")}
                for m in all_train_meta],
            "training_sample_ids": [m["sample_id"] for m in all_train_meta],
            "training_groups": sorted({m.get("parent_episode_id", "") for m in all_train_meta}),
            "training_end": max((m["context_end"] for m in all_train_meta), default=None),
            "source_hashes": dataset["source_hashes"], "validated": False,
            "evaluation_status": "candidate", "label_support": support,
            "evidence_origins": dataset.get("evidence_origins", []), "evaluations": [],
            "exploratory": dataset.get("exploratory", False)})
        _write(self._path("models", candidate["model_id"]), candidate)
        progress(100, "Entrenamiento completado: candidato listo", phase="completed")
        return self._model_summary(candidate)

    def pretrain_gat(self, dataset_id, gat_config=None, progress=lambda *_: None):
        if importlib.util.find_spec("torch") is None:
            raise ReviewError("GAT-LSTM no está disponible: falta PyTorch.", 503)
        dataset = self._read("datasets", dataset_id)
        if dataset.get("method") != "gat_lstm":
            raise ReviewError("El conjunto seleccionado no fue preparado para GAT-LSTM.")
        config = GATConfig(**(gat_config or dataset["config"]))
        if not config.pretrain_enabled:
            raise ReviewError("Activa preentrenamiento en la configuración y vuelve a preparar el conjunto.")
        for key in ("grid_seconds", "sequence_steps", "feature_window_seconds", "graph_top_k",
                    "max_nodes", "use_entropy", "use_dtw"):
            if getattr(config, key) != dataset["config"][key]:
                raise ReviewError(f"La configuración {key} cambió; prepara nuevamente el conjunto.")
        from .GAT_LSTM.pretraining import pretrain_encoder, save_encoder
        train_samples = load_pretraining_partition(dataset, "train_fit")
        dev_samples = load_pretraining_partition(dataset, "dev_tune")
        if not train_samples or not dev_samples:
            raise ReviewError(json.dumps({"code": "insufficient_predictive_targets",
                "train_fit": dataset.get("pretrain_support_by_entity", {}).get("train_fit", {}),
                "dev_tune": dataset.get("pretrain_support_by_entity", {}).get("dev_tune", {})}, ensure_ascii=False))
        preprocessor = json.loads((Path(dataset["sequence_dir"]) / "preprocessor.json").read_text(encoding="utf-8"))
        graph = json.loads((Path(dataset["sequence_dir"]) / "graph.json").read_text(encoding="utf-8"))
        categories = [max(preprocessor["entities"][e].get("vocab", {}).values(), default=2) + 1
                      for e in preprocessor["entity_ids"]]
        kinds = [preprocessor["entities"][e]["kind"] for e in preprocessor["entity_ids"]]
        progress(5, "Preentrenando solo con observaciones futuras originales")
        def pretraining_progress(row):
            percent = 5 + 80 * row["epoch"] / config.max_epochs
            message = (f"Época {row['epoch']}: pérdida predictiva train {row['train_loss']:.4f}, "
                       f"dev_tune {row['dev_loss']:.4f}")
            progress(min(85, percent), message)
        weights, metrics = pretrain_encoder(train_samples, dev_samples, config, categories, kinds,
                                            pretraining_progress)
        manifest = save_encoder(weights, preprocessor, graph, config, self.directory, metrics)
        manifest.update({"dataset_id": dataset_id, "source_hashes": dataset["source_hashes"],
                         "support_train_fit": dataset.get("pretrain_support_by_entity", {}).get("train_fit", {}),
                         "support_dev_tune": dataset.get("pretrain_support_by_entity", {}).get("dev_tune", {})})
        _write(self.directory / "encoders" / f"{manifest['encoder_id']}.json", manifest)
        progress(100, "Encoder predictivo guardado; requiere ajuste diagnóstico antes de activarse")
        return {key: value for key, value in manifest.items() if key not in {"preprocessor", "graph"}}

    def evaluate_gat(self, model_id, dataset_id, partition="validation",
                     progress=lambda *_: None):
        raise ReviewError("GAT-LSTM se prepara y ajusta solo con Desarrollo. Validación y Prueba quedan reservadas para generar eventos futuros.")
        trained, dataset = self._read("models", model_id), self._read("datasets", dataset_id)
        if trained.get("method") != "gat_lstm" or dataset.get("method") != "gat_lstm":
            raise ReviewError("El candidato o conjunto no corresponden a GAT-LSTM.")
        if trained.get("schema_version") != dataset.get("schema_version"):
            raise ReviewError("El esquema del conjunto no coincide con el del candidato.")
        if trained.get("entity_ids") != dataset.get("entity_ids"):
            raise ReviewError("Las entidades del conjunto no coinciden con el candidato.")
        if trained.get("graph_version") != dataset.get("graph_version"):
            raise ReviewError("El grafo congelado del conjunto no coincide con el candidato.")
        samples = load_partition(dataset, partition)
        if not samples:
            raise ReviewError(json.dumps({"code": "insufficient_evaluation_labels", "partition": partition,
                                          "support": dataset.get("label_counts", {}).get(partition, {})},
                                         ensure_ascii=False))
        trained_ids = set(trained.get("training_sample_ids", []))
        trained_groups = set(trained.get("training_groups", []))
        records, predictions = [], []
        snapshot_dir = self.directory
        from .GAT_LSTM.training import load_candidate
        import torch
        model, _, graph, manifest = load_candidate(model_id, snapshot_dir)
        threshold = manifest.get("rejection_policy", {}).get("threshold", .4)
        margin = manifest.get("rejection_policy", {}).get("margin", 0.0)
        progress(10, f"Evaluando objetivos de {partition}")
        with torch.no_grad():
            for index, sample in enumerate(samples):
                meta = sample["metadata"]
                if meta.get("sample_id") in trained_ids or meta.get("parent_episode_id") in trained_groups:
                    raise ReviewError("La evaluación comparte objetivo o episodio con el ajuste.")
                if trained.get("training_end") and meta.get("context_start", "") <= trained["training_end"]:
                    raise ReviewError("El contexto de evaluación no es posterior al último dato de ajuste.")
                x = torch.tensor([sample["X"]], dtype=torch.float32)
                states = torch.tensor([sample["state_ids"]], dtype=torch.long)
                present = torch.tensor([sample["node_present"]], dtype=torch.bool)
                adjacency = torch.tensor(graph["adjacency"], dtype=torch.bool)
                scores = torch.softmax(model(x, states, adjacency, present), dim=-1)[0].tolist()
                for node, active in enumerate(sample["label_mask"]):
                    if not active: continue
                    record = {"true_label": sample["y"][node], "scores": scores[node],
                              "episode_id": meta.get("parent_episode_id"), "sample_id": meta["sample_id"],
                              "evidence_origin": meta.get("evidence_origin", "unspecified")}
                    records.append(record)
                    predictions.append({**record, "entity_id": meta["entity_ids"][node],
                                        "timestamp": meta["target_time"]})
                if index % 25 == 0:
                    progress(10 + 75 * index / len(samples), "Calculando predicciones con política congelada")
        result = gat_metrics(records, threshold, margin)
        result.update({"evaluation_id": uuid4().hex, "dataset_id": dataset_id,
            "model_id": model_id, "partition": partition, "created_at": _now(),
            "scope": "classifier_windows_only_not_end_to_end", "threshold": threshold,
            "margin": margin, "label_support": result["per_class"],
            "evidence_origins": result["evidence_origins"], "predictions": predictions})
        experiment_dir = self.directory / "experiments" / model_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        _write(experiment_dir / f"{result['evaluation_id']}.json", result)
        trained.setdefault("evaluations", []).append(result)
        if partition == "validation":
            trained["validation"] = result
            trained["validated"] = bool(result["complete"] and trained.get("rejection_policy", {}).get("validated"))
            trained["evaluation_status"] = "validated" if trained["validated"] else "incomplete"
        else:
            trained["evaluation_status"] = "test_complete" if result["complete"] else "incomplete"
        _write(self._path("models", model_id), trained)
        progress(100, "Evaluación guardada")
        return self._model_summary(trained)

    @staticmethod
    def _leaks(example, training):
        if example["group_id"] == training["group_id"]:
            return True
        if example["entity_id"] != training["entity_id"]:
            return False
        # Source hashes can change through reserialization. Time/entity overlap is
        # therefore checked even for files with different byte identities.
        a = min(parse_time(example["baseline_start"])[0], parse_time(example["window_start"])[0])
        b = max(parse_time(example["baseline_end"])[0], parse_time(example["window_end"])[0])
        c = min(parse_time(training["baseline_start"])[0], parse_time(training["window_start"])[0])
        d = max(parse_time(training["baseline_end"])[0], parse_time(training["window_end"])[0])
        return a <= d and c <= b

    def evaluate(self, model_id, dataset_id, partition="validation", progress=lambda *_: None):
        if partition not in {"validation", "test"}:
            raise ReviewError("Evalúa con Validación o Prueba, nunca con Desarrollo.")
        trained = self._read("models", model_id)
        dataset = self._read("datasets", dataset_id)
        if dataset["configuration"]["window_seconds"] != trained["window_seconds"]:
            raise ReviewError("Usa la misma duración de ventana del entrenamiento al preparar la evaluación.")
        examples = [example for example in dataset["examples"] if example["partition"] == partition]
        if not examples:
            raise ReviewError("No hay ejemplos de la partición seleccionada en este conjunto.")
        if any(self._leaks(example, old) for example in examples for old in trained["training_examples"]):
            raise ReviewError("Hay episodios, variantes o períodos compartidos con el entrenamiento. Usa períodos independientes.")
        if any(parse_time(example["window_start"])[0] <= parse_time(trained["training_end"])[0] for example in examples):
            raise ReviewError("La evaluación debe ser posterior al último período utilizado para entrenar.")
        progress(20, "Evaluando el modelo fijo con etiquetas separadas")
        references = [self._read("references", identifier) for identifier in trained["reference_ids"]]
        injected_rows = {}
        matrix, correct, accepted = {}, 0, 0
        labelled_correct = labelled_total = 0
        by_origin = {}
        outputs = []
        for example in examples:
            # Evaluation uses only the fixed TRAINING reference. Never normalize
            # held-out examples with statistics fitted on their own normal period.
            eligible = [ref for ref in references if ref["entity_id"] == example["entity_id"]
                and parse_time(ref["end"])[0] < parse_time(example["window_start"])[0]]
            if not eligible:
                raise ReviewError("El modelo no contiene una referencia anterior para todas las entidades evaluadas.")
            reference = max(eligible, key=lambda ref: parse_time(ref["end"])[0])
            rows = self._example_rows(example, injected_rows)
            features = extract_features(rows, reference, example["timestamp"], trained["window_seconds"])
            prediction = predict_features(features, example["sensor_kind"], trained["model"])
            predicted = prediction.get("predicted_label") or prediction.get("fault_type") or "indeterminate"
            if prediction.get("status") == "indeterminate":
                predicted = "indeterminate"
            expected = example["label"]
            matrix.setdefault(expected, {})[predicted] = matrix.setdefault(expected, {}).get(predicted, 0) + 1
            correct += predicted == expected
            accepted += predicted != "indeterminate"
            if example["evaluable"]:
                labelled_total += 1
                labelled_correct += predicted == expected
            origin = example.get("reference_origin", "legacy_unspecified")
            group = by_origin.setdefault(origin, {"count": 0, "evaluable_count": 0, "evaluable_correct": 0})
            group["count"] += 1
            if example["evaluable"]:
                group["evaluable_count"] += 1
                group["evaluable_correct"] += predicted == expected
            outputs.append({"example_id": example["example_id"], "expected": expected, "predicted": predicted,
                            "reference_origin": origin})
        for group in by_origin.values():
            group["evaluable_accuracy"] = group.pop("evaluable_correct") / group["evaluable_count"] if group["evaluable_count"] else None
        labels = sorted(matrix)
        f1s = []
        for label in labels:
            tp = matrix.get(label, {}).get(label, 0)
            fp = sum(row.get(label, 0) for actual, row in matrix.items() if actual != label)
            fn = sum(matrix[label].values()) - tp
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
        evaluation = {"dataset_id": dataset_id, "partition": partition, "created_at": _now(),
            "count": len(examples), "accuracy": correct / len(examples), "macro_f1": sum(f1s) / len(f1s),
            "coverage": accepted / len(examples), "confusion_matrix": matrix,
            "by_reference_origin": by_origin,
            "evaluable_count": labelled_total, "evaluable_accuracy": labelled_correct / labelled_total if labelled_total else None,
            "scope": "classifier_windows_only_not_end_to_end", "reference_policy": "fixed_training_references",
            "exploratory": labelled_total != len(examples),
            "predictions": outputs[:100]}
        trained["evaluations"].append(evaluation)
        if partition == "validation":
            trained["validation"] = evaluation
            trained["validated"] = labelled_total == len(examples) and set(trained["classes"]).issubset(set(labels)) and not trained["exploratory"]
        _write(self._path("models", model_id), trained)
        return self._model_summary(trained)

    def _example_rows(self, example, injected_rows):
        lower, upper = parse_time(example["window_start"])[0], parse_time(example["window_end"])[0]
        if not example.get("injection_id"):
            with self.review.connect() as connection:
                return [dict(row) for row in connection.execute(
                    "SELECT timestamp,state,source_row FROM events WHERE dataset_id=? AND entity_id=? AND ts BETWEEN ? AND ? ORDER BY ts,source_row",
                    (example["source_dataset_id"], example["entity_id"], lower, upper))]
        identifier = example["injection_id"]
        if identifier not in injected_rows:
            case = self.injections.injection(identifier)
            annotation = case["baseline_snapshot"]
            entities = [annotation["entity_id"], *annotation["context_entity_ids"]]
            with self.review.connect() as connection:
                originals = connection.execute("SELECT source_row FROM events WHERE dataset_id=? AND entity_id IN (" + ",".join("?" for _ in entities) + ") AND ts BETWEEN ? AND ? ORDER BY ts,source_row",
                    [annotation["dataset_id"], *entities, parse_time(case["view_start"])[0], parse_time(case["view_end"])[0]])
                with zipfile.ZipFile(self.injections.export_injection(identifier)) as archive, archive.open("injected.csv") as binary:
                    reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
                    collected = []
                    for raw, original in zip(reader, originals, strict=True):
                        row = {key.strip().lower(): value for key, value in raw.items()}
                        if row["entity_id"] == example["entity_id"]:
                            collected.append({"timestamp": row.get("last_changed", row.get("timestamp")), "state": row["state"], "source_row": original[0]})
                    injected_rows[identifier] = collected
        return [row for row in injected_rows[identifier] if lower <= parse_time(row["timestamp"])[0] <= upper
            and (example.get("cutoff_source_row") is None or parse_time(row["timestamp"])[0] != upper or row["source_row"] <= example["cutoff_source_row"])]

    def activate(self, model_id, allow_exploratory=False):
        if (model_id and len(model_id) == 32 and
                (self.directory / "encoders" / f"{model_id}.json").is_file()):
            raise ReviewError("Un encoder predictivo requiere una cabeza diagnóstica ajustada antes de activarse.", 409)
        trained = self._read("models", model_id)
        if trained.get("kind") == "encoder_only" or trained.get("can_activate") is False:
            raise ReviewError("Un encoder predictivo requiere una cabeza diagnóstica ajustada antes de activarse.", 409)
        if trained.get("method") == "gat_lstm":
            if importlib.util.find_spec("torch") is None:
                raise ReviewError("GAT-LSTM no está disponible: falta PyTorch.", 503)
            from .GAT_LSTM.training import load_candidate
            load_candidate(model_id, self.directory)  # hash, esquema y dimensiones antes de publicar
        if not trained.get("validated", False) and not allow_exploratory:
            raise ReviewError("Esta versión no tiene validación completa. Evalúala primero o activa explícitamente el modo experimental.", 409)
        with self.lock:
            _write(self.directory / "active.json", {"model_id": model_id, "activated_at": _now(), "experimental": not trained.get("validated", False)})
        return dict(self._model_summary(trained), active=True)

    def deactivate(self, model_id):
        trained = self._read("models", model_id)
        with self.lock:
            active = self._active()
            if active is not None and active != model_id:
                raise ReviewError("Otra versión está activa. Actualiza la lista antes de desactivarla.", 409)
            if active == model_id:
                _write(self.directory / "active.json", {"model_id": None, "deactivated_at": _now(), "previous_model_id": model_id})
        return dict(self._model_summary(trained), active=False)

    def export_dataset(self, identifier):
        dataset = self._read("datasets", identifier)
        handle, name = tempfile.mkstemp(prefix="diagnostic-examples-", suffix=".zip", dir=self.review.temporary)
        os.close(handle)
        path = Path(name)
        try:
            features = sorted({key for example in dataset["examples"] for key in example["features"]})
            metadata_fields = [key for key in dataset["examples"][0] if key != "features"]
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                with archive.open("examples.csv", "w") as binary, io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=metadata_fields + [f"feature.{key}" for key in features])
                    writer.writeheader()
                    for example in dataset["examples"]:
                        writer.writerow({**{key: example[key] for key in metadata_fields}, **{f"feature.{key}": example["features"].get(key) for key in features}})
                archive.writestr("manifest.json", _canonical(dict(self._dataset_summary(dataset), semantics="Only feature.* columns are model inputs; label is the supervised target. IDs, partition and provenance are grouping/audit metadata.")))
            return path
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def _run_snapshot(self, source, run_id):
        key = f"{source}:{run_id}"
        with self.lock, self._connect() as connection:
            existing = connection.execute("SELECT payload FROM runs WHERE run_id=?", (key,)).fetchone()
            if existing:
                return json.loads(existing[0])
            active = self._active()
            trained = self._read("models", active) if active else None
            references = ([self._read("references", identifier) for identifier in trained.get("reference_ids", [])]
                          if trained and trained.get("method") != "gat_lstm"
                          else [] if trained else [ref for ref in self._list("references") if ref["partition"] == "development"])
            snapshot = {"run_id": key, "model": trained, "references": references, "created_at": _now()}
            connection.execute("INSERT INTO runs VALUES (?,?)", (key, _canonical(snapshot)))
            return snapshot

    def window_seconds_for_run(self, source, run_id):
        snapshot = self._run_snapshot(source, run_id)
        if snapshot.get("model", {}).get("method") == "gat_lstm":
            config = snapshot["model"]["config"]
            return (config["sequence_steps"] - 1) * config["grid_seconds"] + config["feature_window_seconds"]
        return snapshot["model"]["window_seconds"] if snapshot["model"] else 3600

    def create_run_session(self, source, run_id):
        return DiagnosticRunSession(self, source, run_id)

    def _enrich_gat_contextual(self, statistical_results, contextual_results, events,
                               source, run_id, session, snapshot):
        trained = snapshot["model"]
        from .GAT_LSTM.runtime import load_snapshot, predict_at
        gat_snapshot = (session.gat_snapshot if session is not None and session.gat_snapshot else None)
        if gat_snapshot is None and importlib.util.find_spec("torch") is not None:
            gat_snapshot = load_snapshot(str(self.directory), trained["model_id"])
        if gat_snapshot is None:
            enriched = []
            for context in contextual_results:
                item = deepcopy(context)
                if not item.get("routing", {}).get("terminal"):
                    item["diagnosis"] = {"entity_id": str(item.get("entity_id", "")),
                        "timestamp": str(item.get("timestamp", "")), "status": "indeterminate",
                        "predicted_label": None, "fault_type": None, "method": "gat_lstm",
                        "compatibility": None, "confidence": 0.0,
                        "confidence_kind": "uncalibrated_softmax", "class_scores": {},
                        "needs_review": True, "reasons": ["pytorch_dependency_missing"],
                        "model_version": trained["model_id"]}
                    item["diagnosis_mode"] = "active_model"
                enriched.append(item)
            return enriched
        preprocessor = gat_snapshot[1]
        entities = preprocessor["entity_ids"]
        event_index = {}
        if session is not None:
            session.add_events(events)
            provider = session.events_between
        else:
            rows_by_entity = {}
            if isinstance(events, pd.DataFrame) and not events.empty:
                time_column = "timestamp_dt" if "timestamp_dt" in events else "timestamp" if "timestamp" in events else "last_changed"
                selected = events[events["entity_id"].astype(str).isin(entities)].copy()
                selected["_diag_time"] = pd.to_datetime(selected[time_column], utc=True, errors="coerce")
                for entity, group in selected.dropna(subset=["_diag_time"]).groupby("entity_id", sort=False):
                    values = []
                    for ordinal, row in enumerate(group.sort_values(["_diag_time"], kind="stable").to_dict("records")):
                        values.append((row["_diag_time"], {"entity_id": str(entity),
                            "timestamp": row["_diag_time"].isoformat(), "state": str(row.get("state", "")),
                            "source_row": int(row.get("_raw_index", ordinal)) if isinstance(row.get("_raw_index", ordinal), int) else ordinal}))
                    rows_by_entity[str(entity)] = values
            def provider(start, end, requested):
                found = []
                for entity in requested:
                    values = rows_by_entity.get(entity, [])
                    left = bisect_left(values, start, key=lambda pair: pair[0])
                    right = bisect_right(values, end, key=lambda pair: pair[0])
                    found.extend(row for _, row in values[left:right])
                return found
        enriched, decisions = [], []
        for context in contextual_results:
            item = deepcopy(context)
            if item.get("routing", {}).get("terminal"):
                item.pop("diagnosis", None); item.pop("diagnosis_mode", None)
                enriched.append(item)
                continue
            entity_id, timestamp = str(item.get("entity_id", "")), str(item.get("timestamp", ""))
            try:
                result = predict_at(gat_snapshot, provider, entity_id, timestamp)
                target_us, _ = parse_time(timestamp)
                if source != "history" and trained.get("training_end"):
                    if target_us <= parse_time(trained["training_end"])[0]:
                        result.update(status="indeterminate", predicted_label=None, confidence=0.0,
                                      class_scores={}, needs_review=True,
                                      reasons=["target_not_after_training_period"])
            except (ValueError, KeyError, RuntimeError) as exc:
                result = {"entity_id": entity_id, "timestamp": timestamp, "status": "indeterminate",
                          "predicted_label": None, "fault_type": None, "method": "gat_lstm",
                          "compatibility": None, "confidence": 0.0,
                          "confidence_kind": "uncalibrated_softmax", "class_scores": {},
                          "needs_review": True, "reasons": [str(exc)],
                          "model_version": trained["model_id"]}
            result.update(source=source, run_id=run_id, mode="active_model",
                          causal=source != "history",
                          evaluation_scope="retrospective" if source == "history" else "causal_replay")
            result["diagnosis_id"] = hashlib.sha256(_canonical(
                [snapshot["run_id"], entity_id, timestamp, item.get("event_id"), trained["model_id"]]
                ).encode()).hexdigest()[:32]
            item["diagnosis"] = result
            item["diagnosis_mode"] = "active_model"
            enriched.append(item); decisions.append(result)
        if session is not None:
            session.queue(decisions)
        elif decisions:
            with self._connect() as connection:
                connection.executemany("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)",
                    [(r["diagnosis_id"], snapshot["run_id"], source, r["entity_id"], r["timestamp"], _canonical(r))
                     for r in decisions])
        return enriched

    def enrich_contextual(self, statistical_results, contextual_results, events=None, source="history", run_id="default", session=None):
        if session is not None and (session.source != source or session.run_id != run_id):
            raise ValueError("La sesión diagnóstica no corresponde a esta ejecución.")
        snapshot = session.snapshot if session is not None else self._run_snapshot(source, run_id)
        trained = snapshot["model"]
        if trained and trained.get("method") == "gat_lstm":
            return self._enrich_gat_contextual(statistical_results, contextual_results, events,
                                               source, run_id, session, snapshot)
        model = trained["model"] if trained else None
        window_seconds = trained["window_seconds"] if trained else 3600
        indexed = {}
        for statistical in statistical_results:
            try:
                key = statistical.get("observation_id") or (str(statistical.get("entity_id")), parse_time(statistical["timestamp"])[0])
                indexed[key] = statistical
            except (ReviewError, KeyError):
                continue
        event_groups = {}
        if session is not None:
            session.add_events(events)
        elif isinstance(events, pd.DataFrame) and not events.empty and "entity_id" in events:
            target_ids = {str(item.get("entity_id")) for item in contextual_results if not item.get("routing", {}).get("terminal")}
            selected = events[events["entity_id"].astype(str).isin(target_ids)].copy()
            time_column = "timestamp_dt" if "timestamp_dt" in selected else "timestamp" if "timestamp" in selected else "last_changed"
            selected["_diag_time"] = pd.to_datetime(selected[time_column], utc=True, errors="coerce")
            for entity, group in selected.groupby("entity_id", sort=False):
                ordered = group.dropna(subset=["_diag_time"]).sort_values("_diag_time", kind="stable")
                event_groups[str(entity)] = ordered.set_index("_diag_time", drop=False)
        enriched, decisions = [], []
        for context in contextual_results:
            item = deepcopy(context)
            if item.get("routing", {}).get("terminal"):
                item.pop("diagnosis", None)
                item.pop("diagnosis_mode", None)
                enriched.append(item)
                continue
            timestamp = str(item.get("timestamp", ""))
            try:
                at, _ = parse_time(timestamp)
            except ReviewError:
                enriched.append(item)
                continue
            entity_id = str(item.get("entity_id", ""))
            statistical = indexed.get(item.get("observation_id") or (entity_id, at), {"entity_id": entity_id, "timestamp": timestamp})
            eligible_refs = [ref for ref in snapshot["references"] if ref["entity_id"] == entity_id and parse_time(ref["end"])[0] < at] if session is None else None
            reference = (max(eligible_refs, key=lambda ref: parse_time(ref["end"])[0]) if eligible_refs else None) if session is None else session.reference_for(entity_id, at)
            rows = []
            group = event_groups.get(entity_id)
            if session is not None:
                # parse_time returns epoch microseconds for reference comparisons;
                # the event index and its window arithmetic use UTC Timestamps.
                rows = session.rows_for(entity_id, pd.to_datetime(timestamp, utc=True), window_seconds)
            elif group is not None:
                now = pd.to_datetime(timestamp, utc=True)
                window = group.loc[now - pd.Timedelta(seconds=window_seconds):now]
                rows = [{"timestamp": row["_diag_time"].isoformat(), "state": str(row.get("state", "")), "source_row": int(index)} for index, (_, row) in enumerate(window.iterrows())]
            statistical = dict(statistical, diagnostic_window_seconds=window_seconds)
            result = diagnose(statistical, item, reference=reference, rows=rows, model=model)
            if source != "history" and trained and parse_time(trained["training_end"])[0] >= at:
                result.update(status="indeterminate", fault_type=None, compatibility=None, needs_review=True,
                    reasons=["El modelo usa períodos posteriores o simultáneos a este evento. Se conserva pendiente para evitar fuga temporal."])
            if trained and reference is None:
                result.update(status="indeterminate", fault_type=None, compatibility=None, needs_review=True,
                    reasons=["No hay una referencia normal compatible y anterior para esta entidad en el modelo activo."])
            result.update(source=source, run_id=run_id, entity_id=entity_id, timestamp=timestamp,
                mode="active_model" if trained else "catalogue", causal=source != "history",
                evaluation_scope="retrospective" if source == "history" else "causal_replay",
                model_version=trained["model_id"] if trained else f"catalogue:{ENGINE_VERSION}")
            result["diagnosis_id"] = hashlib.sha256(_canonical([snapshot["run_id"], entity_id, timestamp, item.get("event_id"), result["model_version"]]).encode()).hexdigest()[:32]
            item["diagnosis"] = result
            item["diagnosis_mode"] = result["mode"]
            enriched.append(item)
            decisions.append(result)
        if session is not None:
            session.queue(decisions)
        else:
            with self._connect() as connection:
                connection.executemany("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)", [(result["diagnosis_id"], snapshot["run_id"], source, result["entity_id"], result["timestamp"], _canonical(result)) for result in decisions])
        return enriched

    def results(self, source=None, offset=0, limit=50):
        where, args = (" WHERE source=?", [source]) if source else ("", [])
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM decisions" + where, args).fetchone()[0]
            rows = connection.execute("SELECT payload FROM decisions" + where + " ORDER BY timestamp DESC,decision_id DESC LIMIT ? OFFSET ?", [*args, limit, offset]).fetchall()
        items = [json.loads(row[0]) for row in rows]
        return {"items": items, "total": total, "offset": offset, "next_offset": offset + len(items) if offset + len(items) < total else None}


def crear_router_diagnostico(workbench: DiagnosticWorkbench):
    router = APIRouter(prefix="/diagnostics", tags=["Diagnóstico por patrones y preparación de ejemplos"])

    def translate(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except ReviewError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/status")
    def status():
        return translate(workbench.status)

    @router.get("/sources")
    def sources(method: Literal["robust_nearest_prototype", "gat_lstm"] | None = Query(None)):
        return translate(workbench.sources, method == "gat_lstm")

    @router.get("/configurations")
    def configurations():
        return translate(workbench.configurations)

    @router.post("/configurations")
    def save_configuration(payload: dict):
        if set(payload) != {"name", "config"}:
            raise HTTPException(400, "Indica name y config para guardar una configuración GAT-LSTM.")
        return translate(workbench.save_gat_configuration, payload)

    @router.post("/build")
    def build(payload: BuildInput):
        if payload.method == "gat_lstm":
            if payload.injection_ids or payload.include_drafts:
                raise HTTPException(400, "La preparación GAT-LSTM admite una fuente de Desarrollo y etiquetas importadas o anotaciones revisadas; no incluye borradores ni IDs de inyección.")
            return translate(workbench.start_job, workbench.build_gat_dataset,
                             payload.source_ids, payload.annotation_ids, payload.gat_config)
        if payload.source_ids or payload.gat_config:
            raise HTTPException(400, "source_ids y gat_config requieren method=gat_lstm.")
        return translate(workbench.start_job, workbench.build_dataset, payload.model_dump())

    @router.get("/jobs/{job_id}")
    def job(job_id: str):
        return translate(workbench.job, job_id)

    @router.get("/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        return translate(lambda: workbench._dataset_summary(workbench._read("datasets", dataset_id)))

    @router.get("/datasets/{dataset_id}/export")
    def export(dataset_id: str):
        path = translate(workbench.export_dataset, dataset_id)
        return FileResponse(path, media_type="application/zip", filename=f"diagnostic-examples-{dataset_id}.zip", background=BackgroundTask(path.unlink, missing_ok=True))

    @router.post("/train")
    def train(payload: dict):
        allowed = {"dataset_id", "method", "encoder_id", "gat_config"}
        if set(payload) - allowed or "dataset_id" not in payload:
            raise HTTPException(400, "Indica dataset_id y, para GAT-LSTM, method, encoder_id y gat_config opcionales.")
        dataset = translate(workbench._read, "datasets", payload["dataset_id"])
        method = payload.get("method", dataset.get("method", "robust_nearest_prototype"))
        if method == "gat_lstm":
            return translate(workbench.start_job, workbench.train_gat, payload["dataset_id"],
                             payload.get("encoder_id"), payload.get("gat_config"))
        if set(payload) != {"dataset_id"}:
            raise HTTPException(400, "El motor convencional acepta únicamente dataset_id.")
        return translate(workbench.start_job, workbench.train, payload["dataset_id"])

    @router.post("/pretrain")
    def pretrain(payload: dict):
        if set(payload) - {"dataset_id", "gat_config"} or "dataset_id" not in payload:
            raise HTTPException(400, "Indica dataset_id y gat_config opcional para preentrenar.")
        return translate(workbench.start_job, workbench.pretrain_gat,
                         payload["dataset_id"], payload.get("gat_config"))

    @router.post("/models/{model_id}/evaluate")
    def evaluate(model_id: str, payload: dict):
        if set(payload) != {"dataset_id", "partition"}:
            raise HTTPException(400, "Indica dataset_id y partition para evaluar.")
        trained = translate(workbench._read, "models", model_id)
        if trained.get("method") == "gat_lstm":
            return translate(workbench.start_job, workbench.evaluate_gat, model_id,
                             payload["dataset_id"], payload["partition"])
        return translate(workbench.start_job, workbench.evaluate, model_id, payload["dataset_id"], payload["partition"])

    @router.post("/models/{model_id}/activate")
    def activate(model_id: str, payload: dict):
        if set(payload) - {"allow_exploratory"} or not isinstance(payload.get("allow_exploratory", False), bool):
            raise HTTPException(400, "allow_exploratory debe ser booleano.")
        return translate(workbench.activate, model_id, payload.get("allow_exploratory", False))

    @router.post("/models/{model_id}/deactivate")
    def deactivate(model_id: str):
        return translate(workbench.deactivate, model_id)

    @router.get("/results")
    def results(source: Literal["history", "validation_replay"] | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
        return translate(workbench.results, source, offset, limit)

    return router
