"""Research workbench and immutable runtime snapshots for fault-pattern diagnosis.

Review labels and controlled interventions supply supervision. F1/F2 decisions
are inputs to diagnosis and are never turned into training labels here.
"""
from __future__ import annotations
from app.database.diagnostics import DiagnosticRepository
from app.config import MODELS_DIR, REVIEW_DATA_DIR, database_path, artifact_directory

from bisect import bisect_left, bisect_right
import hashlib
import json
from itertools import chain
import os
import tempfile
import time
import zipfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock, Thread
from typing import Any, Literal
from uuid import uuid4
import importlib.util

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.database.review import ReviewStore
from app.database.review import ReviewError
from app.database.review import parse_time
from app.database.review import UTC
from app.services.inyecciones import InjectionStore
from app.modulos.clasificador.catalogo import CATALOG
from app.modulos.clasificador.disponibilidad import unavailable_results
from app.modulos.clasificador.disponibilidad import CLASSIFIER_MODEL_REQUIRED
ENGINE_VERSION = "gat-lstm-segments-v3"
from app.modulos.clasificador.config import Config as GATConfig
from app.modulos.clasificador.config import historical_manifest_path
from app.modulos.clasificador import LABELS as GAT_LABELS
from app.modulos.clasificador.pipeline import build_sequence_dataset
from app.modulos.clasificador.pipeline import development_annotations
from app.modulos.clasificador.pipeline import development_sources
from app.modulos.clasificador.pipeline import load_partition
from app.services.auditoria import sha256_file
from app.modulos.clasificador.storage import open_training_partitions
from app.modulos.clasificador.storage import sample_summaries
from app.modulos.clasificador.evaluation import metrics as gat_metrics

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
    def __init__(self, workbench, source, run_id, model_id=None):
        self.workbench, self.source, self.run_id = workbench, source, run_id
        snapshot_loader = getattr(workbench, "_run_snapshot_for_model", workbench._run_snapshot)
        self.snapshot = (snapshot_loader(source, run_id, model_id=model_id)
                        if model_id is not None and hasattr(workbench, "_run_snapshot_for_model")
                        else snapshot_loader(source, run_id))
        self.gat_snapshot = None
        if (self.snapshot.get("model") or {}).get("method") == "gat_lstm":
            from app.modulos.clasificador.runtime import load_snapshot
            try:
                self.gat_snapshot = load_snapshot(str(workbench.directory), self.snapshot["model"]["model_id"])
            except (ImportError, OSError, ValueError, KeyError, RuntimeError):
                self.gat_snapshot = None
            if self.gat_snapshot is not None:
                self.window_seconds = self.gat_snapshot[3]["config"]["lookback_seconds"] + self.gat_snapshot[3]["config"]["feature_window_seconds"]
            else:
                cfg = self.snapshot["model"]["config"]
                self.window_seconds = (cfg["sequence_steps"] - 1) * cfg["grid_seconds"] + cfg["feature_window_seconds"]
        else:
            self.window_seconds = (self.snapshot.get("model") or {}).get("window_seconds", 3600)
        self.references = {}
        if (self.snapshot.get("model") or {}).get("config", {}).get("architecture_version") == "gat_lstm_v3_segments_multiscale":
            self.window_seconds = 259200
        for reference in self.snapshot["references"]:
            end, _ = parse_time(reference["end"])
            self.references.setdefault(str(reference["entity_id"]), []).append((end, reference))
        for values in self.references.values(): values.sort(key=lambda item: item[0])
        self.events, self.event_heads, self.seen, self.pending = {}, {}, set(), []
        self.sensor_profiles = ({**workbench.sensor_profiles_for_run(self.snapshot["run_id"]),
                                 **{item["entity_id"]: item for item in workbench._list("sensors")
                                    if item.get("status") == "available"}}
                                if (self.snapshot.get("model") or {}).get("method") == "gat_lstm" else {})
        self.last_flush = time.monotonic(); self.stats = {
            "snapshot_loads": 1, "event_rows": 0, "buffered_event_rows": 0,
            "decision_writes": 0, "flushes": 0, "write_ms": 0.0,
            "gat_candidates": 0, "gat_samples": 0, "gat_sample_rejections": 0,
            "gat_sample_ms": 0.0, "gat_inferences": 0, "gat_inference_ms": 0.0,
        }

    def prepare_sensor_profiles(self, frame, before):
        """Freeze calibration from the selected history before replay starts."""
        if self.gat_snapshot is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return
        from app.modulos.clasificador.profiles import build_sensor_profile
        from app.modulos.clasificador.temporal import CausalInputError
        _, prep, _, manifest = self.gat_snapshot
        config = GATConfig(**manifest["config"])
        known = set(prep["entity_ids"]) | set(self.sensor_profiles)
        kinds = {info["kind"] for info in prep["entities"].values()}
        time_column = "timestamp_dt" if "timestamp_dt" in frame else "timestamp" if "timestamp" in frame else "last_changed"
        selected = frame[~frame["entity_id"].isin(known)].copy()
        selected["_profile_time"] = pd.to_datetime(selected[time_column], utc=True, errors="coerce")
        selected = selected[selected["_profile_time"] < before]
        for entity, group in selected.groupby("entity_id", sort=False):
            rows = [{"entity_id": str(entity), "timestamp": row["_profile_time"].isoformat(),
                     "state": str(row.get("state", "")), "sensor_kind": row.get("sensor_kind")}
                    for row in group.to_dict("records")]
            try:
                self.sensor_profiles[str(entity)] = build_sensor_profile(
                    rows, str(entity), before, config, kinds)
            except CausalInputError:
                # The runtime reports why this entity cannot yet be evaluated.
                continue
        self.workbench.save_sensor_profiles_for_run(self.snapshot["run_id"], self.sensor_profiles)

    def add_events(self, frame, *, events_sorted=False):
        if not isinstance(frame, pd.DataFrame) or frame.empty: return
        time_column = "timestamp_dt" if "timestamp_dt" in frame else "timestamp" if "timestamp" in frame else "last_changed"
        order = [time_column, "entity_id"] + (["_raw_index"] if "_raw_index" in frame else [])
        ordered = frame if events_sorted else frame.sort_values(order, kind="stable")
        for ordinal, row in enumerate(ordered.to_dict("records")):
            at = pd.to_datetime(row.get(time_column), utc=True, errors="coerce")
            if pd.isna(at): continue
            entity = str(row.get("entity_id", ""))
            raw_index = row.get("_raw_index")
            try:
                source_row = int(raw_index) if not pd.isna(raw_index) else ordinal
            except (TypeError, ValueError, OverflowError):
                source_row = ordinal
            key = (entity, at.value, source_row)
            if key in self.seen: continue
            self.seen.add(key)
            self.events.setdefault(entity, []).append((at, {
                "entity_id": entity, "timestamp": at.isoformat(),
                "state": str(row.get("state", "")), "source_row": source_row,
                "sensor_kind": row.get("sensor_kind"),
                "provenance": {"source": self.source, "run_id": self.run_id}}))
            self.stats["event_rows"] += 1
            self.stats["buffered_event_rows"] += 1
        latest = max((items[-1][0] for items in self.events.values() if items), default=None)
        if latest is not None:
            cutoff = latest - pd.Timedelta(seconds=self.window_seconds)
            for entity, items in list(self.events.items()):
                head = self.event_heads.get(entity, 0)
                boundary = bisect_left(items, cutoff, lo=head, key=lambda item: item[0])
                if (self.snapshot.get("model") or {}).get("config", {}).get("architecture_version") == "gat_lstm_v3_segments_multiscale":
                    boundary = max(head, boundary - 1)
                for at, row in items[head:boundary]:
                    self.seen.discard((entity, at.value, row.get("source_row")))
                self.stats["buffered_event_rows"] -= boundary - head
                if boundary >= len(items):
                    self.events.pop(entity, None)
                    self.event_heads.pop(entity, None)
                    continue
                if boundary >= 1024 and boundary * 2 >= len(items):
                    del items[:boundary]
                    boundary = 0
                self.event_heads[entity] = boundary

    def reference_for(self, entity_id, at):
        values = self.references.get(str(entity_id), [])
        index = bisect_left(values, at, key=lambda item: item[0])
        return values[index - 1][1] if index else None

    def rows_for(self, entity_id, at, window_seconds):
        values = self.events.get(str(entity_id), [])
        head = self.event_heads.get(str(entity_id), 0)
        start = at - pd.Timedelta(seconds=window_seconds)
        left = bisect_left(values, start, lo=head, key=lambda item: item[0]); right = bisect_right(values, at, lo=head, key=lambda item: item[0])
        return [row for _, row in values[left:right]]

    def events_between(self, start, end, entity_ids):
        result = []
        for entity in entity_ids:
            values = self.events.get(str(entity), [])
            head = self.event_heads.get(str(entity), 0)
            left = bisect_left(values, start, lo=head, key=lambda item: item[0])
            if (self.snapshot.get("model") or {}).get("config", {}).get("architecture_version") == "gat_lstm_v3_segments_multiscale":
                left = max(head, left - 1)
            right = bisect_right(values, end, lo=head, key=lambda item: item[0])
            result.extend(row for _, row in values[left:right])
        return result

    def queue(self, decisions):
        self.pending.extend(decisions)
        if len(self.pending) >= 500 or time.monotonic() - self.last_flush >= 2: self.flush()

    def flush(self):
        if not self.pending: return
        rows = [(result["diagnosis_id"], self.snapshot["run_id"], self.source, result["entity_id"], result["timestamp"], _canonical(result)) for result in self.pending]
        started = time.perf_counter()
        self.workbench.save_decisions(rows)
        self.stats["decision_writes"] += len(rows); self.stats["flushes"] += 1; self.stats["write_ms"] += round((time.perf_counter() - started) * 1000, 3); self.pending.clear(); self.last_flush = time.monotonic()




class DiagnosticWorkbench(DiagnosticRepository):
    def __init__(self, directory: Path):
        self.review = ReviewStore(directory)
        self.injections = InjectionStore(self.review)
        self.directory = MODELS_DIR if Path(directory).resolve() == REVIEW_DATA_DIR.resolve() else Path(directory) / "diagnostics"
        for folder in ("references", "datasets", "models", "jobs", "weights", "preprocessors",
                       "graphs", "sequences", "experiments", "configurations", "sensors"):
            (artifact_directory(self.directory, folder)).mkdir(parents=True, exist_ok=True)
        self.database = database_path("diagnostics.sqlite3", directory)
        self.lock = RLock()
        self.job_lock = Lock()
        self.running_jobs: set[str] = set()
        self.initialize_database()

    @staticmethod
    def _is_supported(item):
        return item.get("method", "gat_lstm") == "gat_lstm" and item.get("config", {}).get("architecture_version") in {"gat_lstm_v3_segments_multiscale"}

    @classmethod
    def _require_supported(cls, item):
        if not cls._is_supported(item):
            raise ReviewError("Artefacto incompatible: solo se admite SegmentGATLSTM (segmentos multiescala).", 409)

    def _path(self, folder, identifier):
        if not identifier or any(char not in "0123456789abcdef" for char in identifier) or len(identifier) != 32:
            raise ReviewError("Identificador de diagnóstico no válido.", 404)
        return artifact_directory(self.directory, folder) / f"{identifier}.json"

    def _read(self, folder, identifier):
        try:
            item = _read_json(self._path(folder, identifier))
            if folder in {"models", "datasets", "configurations"}:
                self._require_supported(item)
            if folder == "datasets":
                sequences = artifact_directory(self.directory, "sequences") / identifier
                if sequences.is_dir():
                    item["sequence_dir"] = str(sequences)
            return item
        except FileNotFoundError as exc:
            raise ReviewError("No se encontró el recurso de diagnóstico.", 404) from exc

    def _list(self, folder):
        items = [_read_json(path) for path in (artifact_directory(self.directory, folder)).glob("*.json")]
        return [item for item in items if folder not in {"models", "datasets", "configurations"} or self._is_supported(item)]

    @staticmethod
    def _dataset_summary(dataset):
        return {key: value for key, value in dataset.items() if key != "examples"}

    @staticmethod
    def _model_summary(model):
        return {key: value for key, value in model.items() if key not in {
            "model", "training_examples", "training_contexts", "training_sample_ids", "training_groups"}}

    def import_model(self, archive):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None:
            raise ReviewError("Importar el modelo requiere PyTorch y PyTorch Geometric en la API.", 503)
        from app.modulos.clasificador.transfer import import_model_zip
        try:
            with self.lock:
                return import_model_zip(archive, self.directory)
        except FileExistsError as exc:
            raise ReviewError(str(exc), 409) from exc

    def status(self):
        active = self._active()
        references = [{key: ref.get(key) for key in ("reference_id", "entity_id", "sensor_kind", "start", "end", "rows", "annotation_ids", "partition", "exploratory")} for ref in self._list("references")]
        gat_available = importlib.util.find_spec("torch") is not None
        return {"mode": "active_model" if active else "unavailable", "active_model_id": active,
                "engine_version": ENGINE_VERSION, "catalog": CATALOG, "references": references,
                "capabilities": {
                                 "gat_lstm": gat_available,
                                 "gat_lstm_segments": gat_available and importlib.util.find_spec("torch_geometric") is not None,
                                 "pyg_available": importlib.util.find_spec("torch_geometric") is not None,
                                 "pytorch_available": gat_available},
                "gat_configurations": self.configurations(),
                "sensor_profiles": [{key: item.get(key) for key in
                    ("profile_id", "entity_id", "kind", "unit", "status", "reference_end", "reference_rows", "reference_annotation_id")}
                    for item in self._list("sensors")],
                
                "datasets": sorted([self._dataset_summary(item) for item in self._list("datasets")], key=lambda item: item["created_at"], reverse=True),
                "models": sorted([dict(self._model_summary(item), active=item["model_id"] == active) for item in self._list("models")], key=lambda item: item["created_at"], reverse=True)}

    def sources(self, gat_only=True):
        manifest_path = historical_manifest_path()
        # A missing optional history manifest should not make the source-list
        # endpoint fail. Review datasets can still be displayed and the UI can
        # explain that no canonical Development history is currently available.
        training = ([{key: value for key, value in item.items() if key != "path"}
                    for item in development_sources(manifest_path, self.review)]
                    if manifest_path.exists() else [])
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
        config = GATConfig(**(gat_config if gat_config is not None else {
            "architecture_version": "gat_lstm_v3_segments_multiscale", "schema_version": "gat-lstm-segments-v1"}))
        manifest_path = historical_manifest_path()
        allowed = {item["source_id"] for item in development_sources(manifest_path, self.review)}
        if not source_ids or any(identifier not in allowed for identifier in source_ids):
            raise ReviewError("Selecciona fuentes de Desarrollo. Validación y Prueba quedan reservadas para eventos futuros.")
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

    def register_gat_sensor(self, payload):
        """Onboards a new shared-model sensor from a manually reviewed normal interval."""
        from collections import Counter
        from statistics import median
        from app.services.estados_validos import valid_state
        from app.modulos.clasificador.features import _finite_number
        from app.modulos.clasificador.features import _quantiles
        if set(payload) != {"entity_id", "reference_annotation_id"}:
            raise ReviewError("Indica entity_id y reference_annotation_id de una referencia normal revisada.")
        entity_id = str(payload["entity_id"]).strip()
        annotation = self.review.annotation(str(payload["reference_annotation_id"]))
        if annotation.get("entity_id") != entity_id or annotation.get("label") != "normal" or annotation.get("review_status") != "reviewed":
            raise ReviewError("La entidad necesita una referencia Normal revisada que coincida exactamente.")
        dataset = self.review.dataset(annotation["dataset_id"])
        if dataset.get("partition") != "development":
            raise ReviewError("La referencia de incorporación debe venir de Desarrollo.")
        kind_row = next((item for item in dataset.get("entities", []) if item["entity_id"] == entity_id), None)
        if not kind_row or kind_row.get("kind") not in {"continuous", "binary", "multistate"}:
            raise ReviewError("No se pudo determinar un tipo compatible para la entidad.")
        start_us, _ = parse_time(annotation["start"]); end_us, _ = parse_time(annotation["end"])
        with self.review.connect() as connection:
            rows = [dict(row) for row in connection.execute(
                "SELECT source_row, timestamp, state FROM events WHERE dataset_id=? AND entity_id=? AND ts BETWEEN ? AND ? AND is_valid=1 ORDER BY ts, source_row",
                (dataset["dataset_id"], entity_id, start_us, end_us))]
        if len(rows) < 3:
            raise ReviewError("La referencia normal debe contener al menos tres lecturas válidas.")
        states = [str(row["state"]) for row in rows if valid_state(str(row["state"]))]
        kind = kind_row["kind"]
        active = self._active()
        if active and self._read("models", active).get("architecture_version") == "gat_lstm_v3_segments_multiscale":
            kind = "continuous" if states and all(_finite_number(v) for v in states) else "multistate"
        profile_entity = {"kind": kind, "max_age_seconds": 900, "normal_references": []}
        if kind == "continuous":
            values = [float(value) for value in states if _finite_number(value)]
            if len(values) < 3:
                raise ReviewError("La referencia normal no contiene suficientes valores numéricos.")
            q1, center, q3 = _quantiles(values)
            profile_entity.update(median=center, scale=(q3 - q1) or 1.0,
                histogram_edges=[] if min(values) == max(values) else [min(values) + (max(values) - min(values)) * i / 16 for i in range(1, 16)])
            stride = max(1, len(values) // 256)
            profile_entity["normal_references"] = [[(value - center) / profile_entity["scale"]
                for value in values[::stride][:256]]]
        else:
            counts = Counter(states)
            profile_entity["vocab"] = {value: index + 3 for index, value in enumerate(sorted(counts, key=lambda value: (-counts[value], value))[:64])}
        profile_id = uuid4().hex
        profile = {"profile_id": profile_id, "entity_id": entity_id, "kind": kind,
            "unit": kind_row.get("unit") or "", "status": "available", "reference_dataset_id": dataset["dataset_id"],
            "reference_annotation_id": annotation["annotation_id"], "reference_start": annotation["start"],
            "reference_end": annotation["end"], "reference_rows": len(rows), "entity": profile_entity,
            "onboarding_policy": "reviewed_normal_reference_v1", "created_at": _now()}
        _write(self._path("sensors", profile_id), profile)
        return {key: profile[key] for key in profile if key != "entity"}

    def _active(self):
        path = self.directory / "active.json"
        if not path.exists():
            return None
        identifier = _read_json(path).get("model_id")
        if not identifier:
            return None
        try:
            model = self._read("models", identifier)
        except (ReviewError, ValueError):
            return None
        return identifier if (self.directory / "weights" / f"{identifier}.pt").is_file() else None

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


    def train_gat(self, dataset_id, encoder_id=None, gat_config=None,
                  progress=lambda *_, **__: None, *, checkpoint_dir=None, resume=False):
        if importlib.util.find_spec("torch") is None:
            raise ReviewError("GAT-LSTM no está disponible: falta PyTorch.", 503)
        dataset = self._read("datasets", dataset_id)
        if dataset.get("method") != "gat_lstm":
            raise ReviewError("El conjunto seleccionado no fue preparado para GAT-LSTM.")
        config = GATConfig(**(gat_config or dataset["config"]))
        if config.architecture_version == "gat_lstm_v3_segments_multiscale" and importlib.util.find_spec("torch_geometric") is None:
            raise ReviewError("GAT-LSTM por segmentos no está disponible: falta PyTorch Geometric.", 503)
        built_config = dataset["config"]
        for key in ("grid_seconds", "sequence_steps", "feature_window_seconds", "graph_top_k",
                    "max_nodes", "use_entropy", "use_dtw", "architecture_version", "local_timezone", "entity_value_modes", "active_scales"):
            built_value = built_config.get(key, "America/Santiago" if key == "local_timezone" else None)
            if key == "active_scales":
                built_value = tuple(built_config.get(key, (True, True, True)))
                if tuple(config.active_scales) == built_value:
                    continue
            if getattr(config, key) != built_value:
                raise ReviewError(f"La configuración {key} cambió; prepara nuevamente el conjunto.")
        progress(1, "Preparando almacenamiento de secuencias para entrenar", phase="cache_preparation")
        def cache_progress(ratio):
            if ratio is not None:
                progress(1 + 3 * ratio, "Preparando almacenamiento de secuencias para entrenar",
                         phase="cache_preparation", cache_progress=ratio * 100)
        with open_training_partitions(dataset, progress=cache_progress) as (train_samples, dev_samples, storage):
            if storage.get("cache_warning"):
                progress(4, storage["cache_warning"], phase="loading", **storage)
            else:
                progress(4, f"Dataset preparado desde {storage['source']}", phase="loading", **storage)
            return self._train_gat_candidate(dataset_id, encoder_id, dataset, config,
                train_samples, dev_samples, storage, progress,
                checkpoint_dir=checkpoint_dir, resume=resume)

    def _train_gat_candidate(self, dataset_id, encoder_id, dataset, config,
                             train_samples, dev_samples, storage, progress, *,
                             checkpoint_dir=None, resume=False):
        if encoder_id:
            raise ReviewError("SegmentGATLSTM usa su encoder de segmentos; no admite encoders predictivos v1.")
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
        from app.modulos.clasificador.training import train_classifier
        from app.modulos.clasificador.training import save_candidate
        progress(5, "Iniciando entrenamiento GAT-LSTM solo con Desarrollo",
                 phase="training", epoch=0, max_epochs=config.max_epochs)
        def training_progress(row):
            percent = 5 + 80 * row["epoch"] / config.max_epochs
            if row.get("threshold_trial"):
                percent = 5+80*((row["threshold_trial"]-1)+row["epoch"]/config.max_epochs)/(3 if config.segment_encoder == "gat" else 1)
            message = (f"Época {row['epoch']}: pérdida train {row['train_loss']:.4f}, "
                       f"dev_tune {row['dev_loss']:.4f}")
            progress(min(85, percent), message, phase="training",
                     epoch=row["epoch"], max_epochs=config.max_epochs)
        def training_batch_progress(row):
            stage = "Ajustando pesos" if row["stage"] == "train" else "Evaluando en Desarrollo"
            percent = 5 + 80 * row["completed_batches"] / row["total_batches"]
            if row.get("threshold_trial"):
                percent = 5+80*((row["threshold_trial"]-1)+row["completed_batches"]/row["total_batches"])/(3 if config.segment_encoder == "gat" else 1)
            message = f"{stage} · época {row['epoch']} de {row['max_epochs']} · lote {row['batch']} de {row['batches']}"
            progress(percent, message, phase="training", **row)
        checkpoint_identity = None
        if checkpoint_dir is not None:
            from app.modulos.clasificador.checkpoints import dataset_identity
            progress(5, "Verificando identidad del dataset para checkpoints", phase="checkpoint_preparation")
            checkpoint_identity = dataset_identity(dataset)
        model, metrics = train_classifier(train_samples, dev_samples, config,
                                          category_sizes, entity_kinds, training_progress, encoder,
                                          batch_progress=training_batch_progress,
                                          checkpoint_dir=checkpoint_dir, resume=resume,
                                          checkpoint_identity=checkpoint_identity)
        metrics["storage"] = storage
        progress(90, "Guardando pesos, grafo, preprocesador y política de rechazo", phase="saving")
        candidate = save_candidate(model, preprocessor, graph, config, self.directory, metrics)
        all_train_meta = [sample["metadata"] for sample in
                          chain(sample_summaries(train_samples), sample_summaries(dev_samples))]
        candidate.update({"dataset_id": dataset_id, "encoder_id": encoder_id,
            "entity_ids": entity_ids, "entity_kinds": entity_kinds, "classes": list(GAT_LABELS),
            "graph_version": candidate.get("graph_version", graph["graph_version"]),
            "training_count": sum(sum(sample["label_mask"]) for sample in sample_summaries(train_samples)),
            "training_contexts": [{key: m.get(key) for key in
                ("parent_episode_id", "context_start", "context_end", "context_entities")}
                for m in all_train_meta],
            "training_sample_ids": [m["sample_id"] for m in all_train_meta],
            "training_groups": sorted({m.get("parent_episode_id", "") for m in all_train_meta}),
            "training_end": max((m["context_end"] for m in all_train_meta), default=None),
            "source_hashes": dataset["source_hashes"],
            "source_inputs": dataset.get("source_inputs", []),
            "source_label_counts": dataset.get("source_label_counts", {}), "validated": False,
            "evaluation_status": "candidate", "label_support": support,
            "evidence_origins": dataset.get("evidence_origins", []), "evaluations": [],
            "exploratory": dataset.get("exploratory", False),
            "runtime": metrics.get("runtime"),
            "runtime_parameters": metrics.get("runtime_parameters"),
            "storage": storage})
        _write(self._path("models", candidate["model_id"]), candidate)
        progress(100, "Entrenamiento completado: candidato listo", phase="completed")
        return self._model_summary(candidate)


    def evaluate_gat(self, model_id, dataset_id, partition="validation",
                     progress=lambda *_: None):
        raise ReviewError("GAT-LSTM se prepara y ajusta solo con Desarrollo. Validación y Prueba quedan reservadas para generar eventos futuros.")


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
            if trained.get("architecture_version") == "gat_lstm_v3_segments_multiscale" and importlib.util.find_spec("torch_geometric") is None:
                raise ReviewError("GAT-LSTM por segmentos requiere PyTorch Geometric.", 503)
            from app.modulos.clasificador.training import load_candidate
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
        handle, name = tempfile.mkstemp(prefix="gat-dataset-", suffix=".zip", dir=self.review.temporary)
        os.close(handle)
        path = Path(name)
        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", _canonical(dataset))
                directory = Path(dataset["sequence_dir"])
                for source in sorted(directory.iterdir()):
                    if source.is_file():
                        archive.write(source, "sequences/" + source.name)
            return path
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def _run_snapshot(self, source, run_id):
        return self._run_snapshot_for_model(source, run_id, None)


    def window_seconds_for_run(self, source, run_id, model_id=None):
        snapshot = self._run_snapshot_for_model(source, run_id, model_id)
        if (snapshot.get("model") or {}).get("method") == "gat_lstm":
            config = snapshot["model"]["config"]
            if config.get("architecture_version") == "gat_lstm_v3_segments_multiscale":
                return 259200
            return (config["sequence_steps"] - 1) * config["grid_seconds"] + config["feature_window_seconds"]
        return snapshot["model"]["window_seconds"] if snapshot["model"] else 3600

    def create_run_session(self, source, run_id, model_id=None):
        return DiagnosticRunSession(self, source, run_id, model_id=model_id)

    def require_classifier_model(self, model_id=None):
        identifier = model_id or self._active()
        if not identifier:
            raise ValueError(CLASSIFIER_MODEL_REQUIRED)
        try:
            model = self._read("models", identifier)
            if not (self.directory / "weights" / f"{identifier}.pt").is_file():
                raise ValueError("Falta el archivo de pesos del modelo.")
            return model
        except ValueError as exc:
            raise ValueError(f"El modelo del filtro clasificador no está disponible. {exc} Selecciona otro modelo o desactiva ese filtro.") from exc

    def classifier_models(self):
        active = self._active()
        models = []
        for model in self._list("models"):
            models.append({key: model.get(key) for key in
                           ("model_id", "method", "architecture_version", "created_at", "validated", "exploratory", "window_seconds")})
            models[-1]["architecture_version"] = model.get("config", {}).get("architecture_version") or model.get("architecture_version")
            models[-1]["active"] = models[-1]["model_id"] == active
        return {"active_model_id": active,
                "models": sorted(models, key=lambda item: item.get("created_at") or "", reverse=True)}

    def _enrich_gat_contextual(self, statistical_results, contextual_results, events,
                               source, run_id, session, snapshot):
        trained = snapshot["model"]
        from app.modulos.clasificador.runtime import load_snapshot
        from app.modulos.clasificador.runtime import predict_at
        gat_snapshot = (session.gat_snapshot if session is not None and session.gat_snapshot else None)
        if gat_snapshot is None and importlib.util.find_spec("torch") is not None:
            try:
                gat_snapshot = load_snapshot(str(self.directory), trained["model_id"])
            except (ImportError, OSError, ValueError, KeyError, RuntimeError):
                return unavailable_results(contextual_results)
        if gat_snapshot is None:
            enriched = []
            for context in contextual_results:
                item = deepcopy(context)
                if not item.get("routing", {}).get("terminal"):
                    item["diagnosis"] = {"entity_id": str(item.get("entity_id", "")),
                        "timestamp": str(item.get("timestamp", "")), "status": "indeterminate",
                        "predicted_label": None, "fault_type": None, "method": "gat_lstm",
                        "compatibility": None, "confidence": 0.0,
                        "fault_type_status": "not_supported",
                        "preprocessor_version": trained.get("schema_version"),
                        "graph_version": trained.get("graph_version"),
                        "confidence_kind": "uncalibrated_softmax", "class_scores": {},
                        "needs_review": True, "reasons": ["pytorch_dependency_missing"],
                        "model_version": trained["model_id"]}
                    item["diagnosis_mode"] = "active_model"
                enriched.append(item)
            return enriched
        preprocessor = gat_snapshot[1]
        entities = preprocessor["entity_ids"]
        sensor_profiles = (session.sensor_profiles if session is not None else
                           {**self.sensor_profiles_for_run(snapshot["run_id"]),
                            **{item["entity_id"]: item for item in self._list("sensors")
                               if item.get("status") == "available"}})
        known_profile_ids = set(sensor_profiles)
        if session is not None:
            session.add_events(events)
            provider = session.events_between
        else:
            rows_by_entity = {}
            if isinstance(events, pd.DataFrame) and not events.empty:
                time_column = "timestamp_dt" if "timestamp_dt" in events else "timestamp" if "timestamp" in events else "last_changed"
                requested_entities = set(entities) | {str(item.get("entity_id", "")) for item in contextual_results}
                selected = events[events["entity_id"].astype(str).isin(requested_entities)].copy()
                selected["_diag_time"] = pd.to_datetime(selected[time_column], utc=True, errors="coerce")
                for entity, group in selected.dropna(subset=["_diag_time"]).groupby("entity_id", sort=False):
                    values = []
                    for ordinal, row in enumerate(group.sort_values(["_diag_time"], kind="stable").to_dict("records")):
                        try:
                            source_row = int(row.get("_raw_index", ordinal))
                        except (TypeError, ValueError, OverflowError):
                            source_row = ordinal
                        values.append((row["_diag_time"], {"entity_id": str(entity),
                            "timestamp": row["_diag_time"].isoformat(), "state": str(row.get("state", "")),
                            "sensor_kind": row.get("sensor_kind"),
                            "source_row": source_row, "provenance": {"source": source, "run_id": run_id}}))
                    rows_by_entity[str(entity)] = values
            def provider(start, end, requested):
                found = []
                for entity in requested:
                    values = rows_by_entity.get(entity, [])
                    left = bisect_left(values, start, key=lambda pair: pair[0])
                    if trained.get("config", {}).get("architecture_version") == "gat_lstm_v3_segments_multiscale":
                        left = max(0, left - 1)
                    right = bisect_right(values, end, key=lambda pair: pair[0])
                    found.extend(row for _, row in values[left:right])
                return found
        enriched, decisions = {}, []
        prediction_cache = {}
        def context_order(pair):
            at = pd.to_datetime(pair[1].get("timestamp"), utc=True, errors="coerce")
            return (pd.isna(at), at.value if not pd.isna(at) else 0, pair[0])
        for context_index, context in sorted(enumerate(contextual_results), key=context_order):
            item = deepcopy(context)
            if item.get("routing", {}).get("terminal"):
                item.pop("diagnosis", None); item.pop("diagnosis_mode", None)
                enriched[context_index] = item
                continue
            entity_id, timestamp = str(item.get("entity_id", "")), str(item.get("timestamp", ""))
            try:
                if session is not None:
                    session.stats["gat_candidates"] += 1
                result = predict_at(gat_snapshot, provider, entity_id, timestamp, sensor_profiles,
                                    metrics=session.stats if session is not None else None,
                                    prediction_cache=prediction_cache, sensor_kind=item.get("sensor_kind"))
                target_us, _ = parse_time(timestamp)
                if source != "history" and entity_id in entities and trained.get("training_end"):
                    if target_us <= parse_time(trained["training_end"])[0]:
                        result.update(status="indeterminate", predicted_label=None, confidence=0.0,
                                      class_scores={}, needs_review=True,
                                      reasons=["target_not_after_training_period"])
            except (ValueError, KeyError, TypeError, OverflowError, RuntimeError) as exc:
                reason = (exc.reason if hasattr(exc, "reason") else
                          "invalid_event_format" if isinstance(exc, (ValueError, KeyError, TypeError, OverflowError))
                          else "inference_error")
                result = {"entity_id": entity_id, "timestamp": timestamp, "status": "indeterminate",
                          "predicted_label": None, "fault_type": None, "method": "gat_lstm",
                          "compatibility": None, "confidence": 0.0,
                          "fault_type_status": "not_supported",
                          "confidence_kind": "uncalibrated_softmax", "class_scores": {},
                          "needs_review": True, "reasons": [reason],
                          "model_version": trained["model_id"]}
            result.update(source=source, run_id=run_id, mode="active_model",
                          causal=source != "history",
                          evaluation_scope="retrospective" if source == "history" else "causal_replay")
            result["diagnosis_id"] = hashlib.sha256(_canonical(
                [snapshot["run_id"], entity_id, timestamp, item.get("event_id"), trained["model_id"]]
                ).encode()).hexdigest()[:32]
            item["diagnosis"] = result
            item["diagnosis_mode"] = "active_model"
            enriched[context_index] = item
            decisions.append(result)
        self.save_sensor_profiles_for_run(snapshot["run_id"], {
            entity: profile for entity, profile in sensor_profiles.items() if entity not in known_profile_ids})
        if session is not None:
            session.queue(decisions)
        elif decisions:
            with self._connect() as connection:
                connection.executemany("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)",
                    [(r["diagnosis_id"], snapshot["run_id"], source, r["entity_id"], r["timestamp"], _canonical(r))
                     for r in decisions])
        return [enriched[index] for index in range(len(contextual_results))]

    def enrich_contextual(self, statistical_results, contextual_results, events=None, source="history", run_id="default", session=None):
        if session is not None and (session.source != source or session.run_id != run_id):
            raise ValueError("La sesión diagnóstica no corresponde a esta ejecución.")
        snapshot = session.snapshot if session is not None else self._run_snapshot(source, run_id)
        if not snapshot.get("model"):
            return unavailable_results(contextual_results)
        self._require_supported(snapshot["model"])
        return self._enrich_gat_contextual(statistical_results, contextual_results, events,
                                           source, run_id, session, snapshot)


