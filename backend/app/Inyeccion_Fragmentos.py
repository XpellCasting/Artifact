"""Reproducible interventions on a frozen, human-reviewed normal fragment.

Original records are immutable. A generated case is semisynthetic evidence,
never an observed real fault, and its partition comes from the source dataset.
"""
from __future__ import annotations
from .Estados_Validos import valid_state
from .Referencia_Revision import reference_metadata

import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .Revision_Fragmentos import (
    INDEPENDENT_EVIDENCE, INVALID_STATES, UTC, ReviewError, ReviewStore,
    iso_time, numeric_state, parse_time,
)

GENERATOR_VERSION = "1.1.0"
MAX_EXPORT_ROWS = 250_000
MAX_PREVIEW_POINTS = 400
NUMERIC_KINDS = {"bias", "drift", "noise", "spike"}


class InjectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    annotation_id: str
    annotation_revision: int = Field(ge=1)
    kind: Literal["bias", "drift", "noise", "spike", "stuck", "unavailable", "state_replace"]
    start: str
    end: str
    amount: float = 1.0
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    replacement_state: str = Field(default="", max_length=20000)


class _Preview:
    """Bound preview memory while retaining changed samples and both curves' extrema."""

    def __init__(self, count: int, lower: int, upper: int):
        self.sampled = count > MAX_PREVIEW_POINTS
        self.lower, self.upper = lower, upper
        self.points: list[dict] = []
        self.buckets: dict[int, dict[str, dict]] = {}

    def add(self, row, injected: str, changed: bool):
        point = {"source_row": row["source_row"], "timestamp": row["timestamp"],
                 "original_state": row["state"], "injected_state": injected,
                 "changed": changed, "_ts": row["ts"]}
        if not self.sampled:
            self.points.append(point)
            return
        # At most nine representatives in each of forty time buckets.
        index = min(39, (row["ts"] - self.lower) * 40 // max(1, self.upper - self.lower + 1))
        bucket = self.buckets.setdefault(index, {"first": point})
        bucket["last"] = point
        if changed:
            bucket.setdefault("first_changed", point)
            bucket["last_changed"] = point
        if not row["is_valid"]:
            bucket.setdefault("invalid", point)
        for field in ("original_state", "injected_state"):
            value = numeric_state(point[field])
            if value is None:
                continue
            for mode in ("min", "max"):
                key = f"{field}_{mode}"
                previous = numeric_state(bucket[key][field]) if key in bucket else None
                if previous is None or (value < previous if mode == "min" else value > previous):
                    bucket[key] = point

    def finish(self) -> list[dict]:
        if self.sampled:
            unique = {point["source_row"]: point for bucket in self.buckets.values() for point in bucket.values()}
            points = sorted(unique.values(), key=lambda point: (point["_ts"], point["source_row"]))
        else:
            points = self.points
        return [{key: value for key, value in point.items() if key != "_ts"} for point in points]


class InjectionStore:
    def __init__(self, store: ReviewStore):
        self.store = store
        self.directory = store.directory / "injections"
        self.directory.mkdir(parents=True, exist_ok=True)

    def _run_directory(self, injection_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", injection_id):
            raise ReviewError("Inyección no encontrada.", 404)
        return self.directory / injection_id

    def injection(self, injection_id: str) -> dict[str, Any]:
        directory = self._run_directory(injection_id)
        try:
            if not (directory / "fragment.zip").is_file():
                raise FileNotFoundError
            return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ReviewError("Inyección no encontrada.", 404) from exc
        except (ValueError, UnicodeError) as exc:
            raise ReviewError("Los metadatos de la inyección están dañados; revise el respaldo.", 409) from exc

    def list_injections(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        if dataset_id is not None:
            self.store.dataset(dataset_id)
        result = []
        for directory in self.directory.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            try:
                metadata = self.injection(directory.name)
            except ReviewError:
                continue
            if dataset_id is None or metadata["dataset_id"] == dataset_id:
                result.append(metadata)
        return sorted(result, key=lambda item: (item["created_at"], item["injection_id"]), reverse=True)

    def export_injection(self, injection_id: str) -> Path:
        self.injection(injection_id)
        return self._run_directory(injection_id) / "fragment.zip"

    def create(self, payload: dict[str, Any] | InjectionInput) -> dict[str, Any]:
        try:
            params = payload if isinstance(payload, InjectionInput) else InjectionInput.model_validate(payload)
        except ValidationError as exc:
            raise ReviewError(str(exc)) from exc
        annotation = self.store.annotation(params.annotation_id)
        self._check_baseline(annotation, params.annotation_revision)
        dataset, ids, baseline_start, baseline_end, lower, upper = self.store._window(
            annotation["dataset_id"], [annotation["entity_id"], *annotation["context_entity_ids"]],
            annotation["start"], annotation["end"], annotation["padding_minutes"],
        )
        start_us, _ = parse_time(params.start)
        end_us, _ = parse_time(params.end)
        if not baseline_start <= start_us <= end_us <= baseline_end:
            raise ReviewError("La intervención debe estar ordenada y contenida en el intervalo de la anotación normal.")
        entity_kind = next(entity["kind"] for entity in dataset["entities"] if entity["entity_id"] == annotation["entity_id"])
        if params.kind in NUMERIC_KINDS and entity_kind != "continuous":
            raise ReviewError("Este tipo de inyección requiere una entidad continua con estados numéricos.")
        if not math.isfinite(params.amount):
            raise ReviewError("La magnitud debe ser un número finito.")
        if params.kind == "noise" and params.amount <= 0:
            raise ReviewError("El ruido requiere una desviación estándar mayor que cero, en unidades de la entidad.")
        if params.kind in {"bias", "drift", "spike"} and params.amount == 0:
            raise ReviewError("La magnitud cero no altera los datos; seleccione una intervención efectiva.")
        if params.kind == "drift" and start_us == end_us:
            raise ReviewError("La deriva requiere un intervalo de duración positiva.")
        placeholders = ",".join("?" for _ in ids)
        where = f"dataset_id=? AND entity_id IN ({placeholders}) AND ts BETWEEN ? AND ?"
        selection = [dataset["dataset_id"], *ids, lower, upper]
        with self.store.connect() as connection:
            if params.kind == "state_replace":
                if entity_kind == "continuous":
                    raise ReviewError("El reemplazo de estado requiere una entidad discreta.")
                replacement = params.replacement_state
                if not valid_state(replacement):
                    raise ReviewError("Seleccione un estado válido observado; para indisponibilidad use unavailable.")
                observed = connection.execute(
                    "SELECT 1 FROM events WHERE dataset_id=? AND entity_id=? AND state=? AND is_valid=1 AND ts BETWEEN ? AND ? LIMIT 1",
                    (dataset["dataset_id"], annotation["entity_id"], replacement, baseline_start, baseline_end),
                ).fetchone()
                if observed is None:
                    raise ReviewError("El estado de reemplazo debe haberse observado como válido en esta entidad dentro del intervalo revisado.")
            count = connection.execute(f"SELECT COUNT(*) FROM events WHERE {where}", selection).fetchone()[0]
            if count > MAX_EXPORT_ROWS:
                raise ReviewError(f"El fragmento ampliado supera el límite de {MAX_EXPORT_ROWS:,} filas. Reduzca el intervalo o su contexto.", 413)
            target_count = connection.execute(
                "SELECT COUNT(*) FROM events WHERE dataset_id=? AND entity_id=? AND ts BETWEEN ? AND ?",
                (dataset["dataset_id"], annotation["entity_id"], lower, upper),
            ).fetchone()[0]
        injection_id = uuid4().hex
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.directory))
        try:
            metadata = self._generate(staging, injection_id, params, annotation, dataset,
                                      entity_kind, ids, start_us, end_us, lower, upper,
                                      target_count, where, selection)
            metadata_path = staging / "metadata.json"
            with metadata_path.open("w", encoding="utf-8") as output:
                json.dump(metadata, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.flush()
                os.fsync(output.fileno())
            # Freeze only the requested revision. An edit during generation invalidates
            # this run, and the write transaction excludes edits during publication.
            with self.store.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute("SELECT payload FROM annotations WHERE annotation_id=?", (params.annotation_id,)).fetchone()
                if row is None:
                    raise ReviewError("Anotación no encontrada.", 404)
                self._check_baseline(json.loads(row[0]), params.annotation_revision)
                staging.rename(self._run_directory(injection_id))
            return metadata
        finally:
            # Only this generator's flat temporary directory is eligible for cleanup.
            if staging.exists() and staging.resolve().parent == self.directory.resolve() and staging.name.startswith(".pending-"):
                for child in staging.iterdir():
                    if child.is_file():
                        child.unlink()
                staging.rmdir()

    @staticmethod
    def _check_baseline(annotation: dict, revision: int):
        if annotation["revision"] != revision:
            raise ReviewError("La anotación cambió de revisión. Ábrala de nuevo antes de generar la inyección.", 409)
        if annotation["label"] != "normal":
            raise ReviewError("La inyección requiere una anotación guardada como normal.")
        if annotation["status"] == "pending":
            raise ReviewError("Termine la revisión y guarde el fragmento como Revisado antes de generar una inyección.")
        if len(annotation["reviewer"].strip()) < 2 or len(annotation["evidence"].strip()) < 12:
            raise ReviewError("Documente el revisor y una evidencia de al menos doce caracteres para el fragmento normal.")

    def _generate(self, staging: Path, identifier: str, params: InjectionInput,
                  annotation: dict, dataset: dict, entity_kind: str, ids: list[str],
                  start_us: int, end_us: int, lower: int, upper: int, target_count: int,
                  where: str, selection: list) -> dict:
        rng = random.Random(params.seed)
        preview = _Preview(target_count, lower, upper)
        state_col = [column.strip().lower() for column in dataset["columns"]].index("state")
        changed_rows = exported_rows = skipped_invalid = 0
        first_state = None
        spike_ts = None
        spike_row = None
        files = {name: staging / name for name in ("original.csv", "injected.csv", "interventions.csv")}
        intervention_fields = ["source_row", "entity_id", "timestamp", "original_state", "injected_state"]
        with files["original.csv"].open("w", encoding="utf-8", newline="") as original_file, \
                files["injected.csv"].open("w", encoding="utf-8", newline="") as injected_file, \
                files["interventions.csv"].open("w", encoding="utf-8", newline="") as interventions_file, \
                self.store.connect() as connection:
            original_writer, injected_writer = csv.writer(original_file), csv.writer(injected_file)
            interventions_writer = csv.DictWriter(interventions_file, fieldnames=intervention_fields)
            original_writer.writerow(dataset["columns"])
            injected_writer.writerow(dataset["columns"])
            interventions_writer.writeheader()
            for row in connection.execute(f"SELECT * FROM events WHERE {where} ORDER BY ts, source_row", selection):
                raw = json.loads(row["raw_json"])
                original_writer.writerow(raw)
                output_state = row["state"]
                target = row["entity_id"] == annotation["entity_id"]
                in_interval = target and start_us <= row["ts"] <= end_us
                if in_interval and not row["is_valid"]:
                    skipped_invalid += 1
                if in_interval and row["is_valid"]:
                    if first_state is None:
                        first_state = row["state"]
                    if params.kind == "unavailable":
                        output_state = "unavailable"
                    elif params.kind == "state_replace":
                        output_state = params.replacement_state
                    elif params.kind == "stuck":
                        if entity_kind != "continuous" or numeric_state(first_state) != row["numeric_value"]:
                            output_state = first_state
                    elif params.kind != "spike" or spike_row is None:
                        value = row["numeric_value"]
                        if value is None or not math.isfinite(value):
                            raise ReviewError("La entidad continua contiene una lectura válida no numérica; revise su tipificación.")
                        if params.kind == "noise":
                            delta = rng.gauss(0.0, params.amount)
                        elif params.kind == "drift":
                            delta = params.amount * ((row["ts"] - start_us) / (end_us - start_us))
                        else:
                            delta = params.amount
                        result = value + delta
                        if not math.isfinite(result):
                            raise ReviewError("La intervención produce un número no finito. Reduzca la magnitud.")
                        if result != value:
                            output_state = repr(result)
                        if params.kind == "spike":
                            spike_ts, spike_row = row["ts"], row["source_row"]
                changed = output_state != row["state"]
                if changed:
                    raw[state_col] = output_state
                    changed_rows += 1
                    interventions_writer.writerow({"source_row": row["source_row"], "entity_id": row["entity_id"],
                                                   "timestamp": row["timestamp"], "original_state": row["state"],
                                                   "injected_state": output_state})
                injected_writer.writerow(raw)
                exported_rows += 1
                if target:
                    preview.add(row, output_state, changed)
        if not changed_rows:
            raise ReviewError("La intervención no modifica ninguna lectura válida. Revise el intervalo, la magnitud o el estado de reemplazo.")
        truth_start, truth_end = (spike_ts, spike_ts) if params.kind == "spike" else (start_us, end_us)
        basis = reference_metadata(annotation)
        evaluable = bool(annotation["status"] in {"reviewed", "verified"} and annotation["evaluable"])
        warnings = ["Caso semisintético: las alteraciones son controladas y no acreditan una falla real observada.",
                    "La etiqueta se limita al episodio intervenido de la entidad objetivo. El resto de filas y entidades no recibe una etiqueta normal."]
        if not basis["externally_verified"]:
            warnings.append("La normalidad del original se asume tras revisar el histórico; la intervención es conocida, pero no aporta verificación externa del original.")
        if not evaluable:
            warnings.append("Base normal en borrador: uso exploratorio; este caso queda fuera de las métricas principales (evaluable=false).")
        if skipped_invalid:
            warnings.append(f"Se conservaron sin modificar {skipped_invalid} estados inválidos dentro de la ventana de intervención.")
        if dataset["rejected_rows"]:
            warnings.append(f"La fuente contiene {dataset['rejected_rows']} filas no indexables, excluidas de ambos CSV.")
        if preview.sampled:
            warnings.append("La vista previa está muestreada: conserva cambios y extremos representativos, pero puede omitir transiciones. Los CSV contienen todas las filas del fragmento.")
        metadata = {
            "injection_id": identifier, "dataset_id": dataset["dataset_id"],
            "annotation_id": annotation["annotation_id"], "annotation_revision": annotation["revision"],
            "entity_id": annotation["entity_id"], "partition": dataset["partition"], "kind": params.kind,
            "start": iso_time(truth_start), "end": iso_time(truth_end),
            "requested_start": iso_time(start_us), "requested_end": iso_time(end_us),
            "amount": params.amount, "seed": params.seed, "replacement_state": params.replacement_state,
            "created_at": datetime.now(UTC).isoformat(), "evaluable": evaluable,
            "reference_origin": "controlled_injection", "intervention_known": True,
            "evaluation_scope": "controlled_injection", "baseline_normality_basis": basis["normality_basis"],
            "baseline_externally_verified": basis["externally_verified"],
            "changed_rows": changed_rows, "exported_rows": exported_rows,
            "preview": preview.finish(), "preview_sampled": preview.sampled, "warnings": warnings,
            "provenance": "semisynthetic", "generator_version": GENERATOR_VERSION,
            "source_sha256": dataset["sha256"], "baseline_snapshot": annotation,
            "view_start": iso_time(lower), "view_end": iso_time(upper),
            "spike_source_row": spike_row,
        }
        truth = {
            "reference_origin": "controlled_injection", "intervention_known": "true",
            "evaluation_scope": "controlled_injection", "baseline_normality_basis": basis["normality_basis"],
            "injection_id": identifier, "dataset_id": dataset["dataset_id"], "source_sha256": dataset["sha256"],
            "partition": dataset["partition"], "entity_id": annotation["entity_id"],
            "start": metadata["start"], "end": metadata["end"], "label": "sensor_fault",
            "provenance": "semisynthetic", "evaluable": "true" if evaluable else "false",
            "annotation_id": annotation["annotation_id"], "annotation_revision": annotation["revision"],
            "kind": params.kind, "changed_rows": changed_rows,
            "source_row": spike_row if params.kind == "spike" else "",
        }
        manifest = {
            "schema_version": "1.0", "generator_version": GENERATOR_VERSION, "provenance": "semisynthetic",
            "injection": {key: value for key, value in metadata.items() if key != "preview"},
            "baseline_snapshot": annotation, "source_sha256": dataset["sha256"],
            "dataset": dataset, "partition": dataset["partition"], "partition_policy": "inherited_from_source",
            "seed": params.seed, "rng": "Python random.Random(seed), gauss(0, amount)",
            "python_version": platform.python_version(),
            "parameters": params.model_dump(), "selected_entities": ids,
            "label_interval": {"start": metadata["start"], "end": metadata["end"], "entity_id": annotation["entity_id"], "inclusive": True},
            "context_interval": {"start": iso_time(lower), "end": iso_time(upper), "inclusive": True},
            "row_order": "timestamp in UTC, then original CSV source_row",
            "raw_field_semantics": "Both CSVs preserve original decoded field strings and column names; only state in interventions.csv rows changes in injected.csv. Original timestamp strings and all context rows are preserved. CSV serialization uses UTF-8 and commas, and may differ from uploaded bytes.",
            "ground_truth_semantics": "One controlled semisynthetic fault episode for the target interval only. For spike, source_row identifies the single observation. interventions.csv records the exact changes. Evaluable means usable for evaluating known injections; baseline_normality_basis distinguishes assumed normality from legacy independent evidence. Neither a reviewed baseline nor a known intervention confirms a real observed fault. Unchanged rows and context are not labelled normal.",
            "drift_semantics": "amount * (event_time - requested_start) / (requested_end - requested_start), in native units",
            "stuck_semantics": "Hold the first valid state observed within the requested intervention interval; preserve existing invalid states.",
            "noise_semantics": "Independent Gaussian noise with standard deviation amount in native units, drawn in chronological/source-row order for eligible target rows only.",
            "warnings": warnings,
        }
        truth_path = staging / "ground_truth.csv"
        with truth_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(truth))
            writer.writeheader()
            writer.writerow(truth)
        manifest["export_sha256"] = {}
        for name, path in {**files, "ground_truth.csv": truth_path}.items():
            with path.open("rb") as stream:
                manifest["export_sha256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
        archive_path = staging / "fragment.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for name, path in files.items():
                archive.write(path, name)
            archive.write(truth_path, "ground_truth.csv")
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        with archive_path.open("r+b") as archive_file:
            os.fsync(archive_file.fileno())
        for path in [*files.values(), truth_path]:
            path.unlink()
        return metadata


def crear_router_inyecciones(store: ReviewStore) -> APIRouter:
    from .Escenarios_Historico import ScenarioStore
    injections = InjectionStore(store)
    scenarios = ScenarioStore(injections)
    router = APIRouter(tags=["Inyección controlada de fragmentos"])

    def translate(call, *args):
        try:
            return call(*args)
        except ReviewError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @router.get("/injections")
    def list_injections(dataset_id: str | None = None):
        return {"injections": translate(injections.list_injections, dataset_id)}

    @router.post("/injections")
    def create_injection(payload: dict[str, Any]):
        return translate(injections.create, payload)

    @router.get("/injections/{injection_id}")
    def get_injection(injection_id: str):
        return translate(injections.injection, injection_id)

    @router.get("/injections/{injection_id}/export")
    def export_injection(injection_id: str):
        path = translate(injections.export_injection, injection_id)
        return FileResponse(path, media_type="application/zip", filename=f"injection-{injection_id}.zip")

    @router.get("/scenarios")
    def list_scenarios(dataset_id: str):
        return {"scenarios": translate(scenarios.list, dataset_id)}

    @router.post("/scenarios")
    def create_scenario(payload: dict[str, Any]):
        return translate(scenarios.create, payload)

    @router.get("/scenarios/{scenario_id}/export")
    def export_scenario(scenario_id: str):
        translate(scenarios.get, scenario_id)
        return FileResponse(scenarios.path(scenario_id) / "scenario.zip", media_type="application/zip", filename=f"historico-escenario-{scenario_id}.zip")

    @router.get("/scenarios/{scenario_id}/csv")
    def export_labelled_csv(scenario_id: str):
        snapshot = translate(scenarios.get, scenario_id)
        if not snapshot.get("labelled_csv"):
            raise HTTPException(409, "Este escenario es anterior al CSV etiquetado. Carga su selección y guarda una nueva versión.")
        return FileResponse(scenarios.path(scenario_id) / "labelled.csv", media_type="text/csv", filename=f"historico-etiquetado-{scenario_id}.csv")

    return router
