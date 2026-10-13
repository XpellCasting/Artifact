"""Compose immutable fragment interventions over every record of their source CSV."""
import csv
import hashlib
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

from app.database.review import ReviewError
from app.database.review import parse_time
from app.services.csv_etiquetado import labelled_columns
from app.services.csv_etiquetado import read_label
from app.services.verdad_escenario import compose_truth
from app.services.verdad_escenario import RowLabels
from app.services.verdad_escenario import TRUTH_FIELDS
from app.services.verdad_escenario import INTERVENTION_FIELDS


class ScenarioStore:
    def __init__(self, injections):
        self.injections = injections
        self.review = injections.store
        self.directory = self.review.sources.parent / "scenarios"
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, identifier):
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ReviewError("Identificador de escenario inválido.")
        return self.directory / identifier

    def get(self, identifier):
        try:
            return json.loads((self.path(identifier) / "metadata.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ReviewError("Escenario no encontrado.", 404) from exc
        except (OSError, ValueError) as exc:
            raise ReviewError("No se puede leer el escenario. Revisa sus permisos o restaura su respaldo.", 409) from exc

    def list(self, dataset_id):
        self.review.dataset(dataset_id)
        records = []
        for p in sorted(self.directory.iterdir()):
            if not re.fullmatch(r"[0-9a-f]{32}", p.name):
                continue
            try:
                records.append(self.get(p.name))
            except ReviewError:
                # One damaged/unreadable snapshot must not block other scenarios.
                continue
        return [record for record in records if record["dataset_id"] == dataset_id]

    def create(self, payload):
        dataset_id = payload.get("dataset_id", "")
        dataset = self.review.dataset(dataset_id)
        ids = payload.get("injection_ids", [])
        if not isinstance(ids, list) or len(ids) > 200 or any(not isinstance(i, str) for i in ids):
            raise ReviewError("Selecciona como máximo 200 inyecciones guardadas.")
        annotations = self.review.list_annotations(dataset_id)
        if not annotations and not ids and labelled_columns(dataset["columns"]) is None:
            raise ReviewError("Guarda una anotación o selecciona una inyección para exportar el escenario.")
        if len(set(ids)) != len(ids):
            raise ReviewError("Una inyección no puede repetirse en el escenario.")
        cases = [self.injections.injection(i) for i in ids]
        for index, case in enumerate(cases):
            if case["dataset_id"] != dataset_id or case["source_sha256"] != dataset["sha256"]:
                raise ReviewError("Todas las inyecciones deben pertenecer al mismo histórico y partición.")
            for other in cases[:index]:
                if case["entity_id"] == other["entity_id"] and max(parse_time(case["start"])[0], parse_time(other["start"])[0]) <= min(parse_time(case["end"])[0], parse_time(other["end"])[0]):
                    raise ReviewError(f"Hay intervalos superpuestos en {case['entity_id']}. Retira uno de los casos.", 409)
        changes, truths = {}, []
        for case in cases:
            with zipfile.ZipFile(self.injections.export_injection(case["injection_id"])) as archive:
                truths.extend(csv.DictReader(io.StringIO(archive.read("ground_truth.csv").decode("utf-8-sig"))))
                with archive.open("interventions.csv") as binary:
                    for row in csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")):
                        source_row = int(row["source_row"])
                        if source_row in changes:
                            raise ReviewError("Dos intervenciones modifican la misma fila.", 409)
                        changes[source_row] = {**row, "injection_id": case["injection_id"], "kind": case["kind"], "label": "sensor_fault"}
        truths = compose_truth(annotations, truths)
        labels = RowLabels(annotations)
        identifier = uuid4().hex
        source = self.review.sources / f"{dataset_id}.csv"
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != dataset["sha256"]:
                raise ReviewError("La fuente cambió respecto de las inyecciones guardadas.", 409)
        with tempfile.TemporaryDirectory(dir=self.directory, prefix="building-") as temporary:
            staging = Path(temporary)
            rows, applied = 0, 0
            with source.open("r", encoding="utf-8-sig", newline="") as stream, (staging / "injected.csv").open("w", encoding="utf-8", newline="") as output, (staging / "labelled.csv").open("w", encoding="utf-8", newline="") as labelled:
                sample = stream.read(8192)
                stream.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel
                reader, writer = csv.reader(stream, dialect=dialect), csv.writer(output)
                headers = next(reader)
                inherited_columns = labelled_columns(headers)
                state_index = [h.strip().lower() for h in headers].index("state")
                entity_index = [h.strip().lower() for h in headers].index("entity_id")
                normalized = [h.strip().lower() for h in headers]
                date_index = normalized.index("last_changed" if "last_changed" in normalized else "timestamp")
                labelled_writer = csv.writer(labelled)
                labelled_writer.writerow(["entity_id", "state", "last_changed", "es_sintetica", "tipo", "tipo_falla"])
                writer.writerow(headers)
                for number, raw in enumerate(reader, start=2):
                    rows += 1
                    change = changes.get(number)
                    if change:
                        if len(raw) != len(headers) or raw[state_index] != change["original_state"] or raw[entity_index] != change["entity_id"]:
                            raise ReviewError("Una intervención no coincide con su fila original.", 409)
                        raw[state_index] = change["injected_state"]
                        applied += 1
                    writer.writerow(raw)
                    entity, state, timestamp = [raw[i] if i < len(raw) else "" for i in (entity_index, state_index, date_index)]
                    labelled_writer.writerow([entity, state, timestamp, *labels.values(number, entity, timestamp, change, read_label(raw, inherited_columns) if len(raw) == len(headers) else None)])
            if applied != len(changes):
                raise ReviewError("No se encontraron todas las filas intervenidas.", 409)
            for name, records, defaults in [("ground_truth.csv", truths, TRUTH_FIELDS),
                    ("interventions.csv", list(changes.values()), INTERVENTION_FIELDS),
                    ("annotations.csv", annotations, ["annotation_id", "revision", "entity_id", "start", "end", "label", "status", "evaluable"])]:
                fields = list(dict.fromkeys([*defaults, *(key for row in records for key in row)]))
                with (staging / name).open("w", encoding="utf-8", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=fields)
                    writer.writeheader()
                    for record in records:
                        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, bool)) else value for key, value in record.items()})
            metadata = {"scenario_id": identifier, "dataset_id": dataset_id,
                "source_sha256": dataset["sha256"], "partition": dataset["partition"],
                "rows": rows, "changed_rows": applied, "entity_count": len(dataset["entities"]),
                "modified_entities": sorted({case["entity_id"] for case in cases}),
                "injection_ids": ids, "cases": cases, "annotation_count": len(annotations),
                "annotation_snapshots": annotations, "ground_truth_count": len(truths),
                "ground_truth_schema_version": "3.0", "labelled_csv": True, "ground_truth_preview": truths[:300],
                "ground_truth_preview_sampled": len(truths) > 300,
                "name": str(payload.get("name") or "Escenario del histórico")[:160]}
            manifest = {**metadata, "schema_version": "3.0", "row_order": "original source record order",
                "imported_ground_truth": bool(inherited_columns),
                "imported_ground_truth_semantics": "Inherited labels and synthetic flags are preserved from source columns. Current manual marks override inherited classes, retaining inherited synthetic flags. New interventions override modified rows. Prior interventions cannot be reconstructed from the six-column file; original.csv is the imported version.",
                "labelled_csv_semantics": "labelled.csv has one row per source record: entity_id,state,last_changed,es_sintetica,tipo,tipo_falla. es_sintetica=true only for records in interventions.csv whose state actually changed; these receive Falla and the injection kind. Other rows use the manual point label first, then the interval label. Unknown, pending, uncertain and unlabelled rows have NULL for tipo and tipo_falla and false for es_sintetica. Normal/anomalia have NULL for tipo_falla. No filtering or classifier is run. Raw source and annotations remain in the archive.",
                "ground_truth_semantics": "ground_truth.csv resolves labels for injected.csv with inclusive microsecond intervals. All current source annotations are frozen in annotations.csv and annotation_snapshots. Selected injections override manual labels only in their interval; point spikes use source_row, and remaining manual rows at that timestamp use excluded_source_rows. Overlapping manual references are merged into segments, retaining annotation_references; reviewed known labels take priority over pending/uncertain labels. Conflicting known labels become uncertain, non-evaluable. An injection overlapping a current reviewed non-normal interpretation is non-evaluable. Unreviewed or uncertain intervals remain non-evaluable. Unlabelled records are not automatically normal. Original CSV bytes and every source record are preserved; only registered intervention states change in injected.csv.",
                "export_sha256": {"original.csv": dataset["sha256"]}}
            for name in ("injected.csv", "labelled.csv", "ground_truth.csv", "interventions.csv", "annotations.csv"):
                with (staging / name).open("rb") as stream:
                    manifest["export_sha256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
            with zipfile.ZipFile(staging / "scenario.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(source, "original.csv")
                for name in ("injected.csv", "labelled.csv", "ground_truth.csv", "interventions.csv", "annotations.csv"):
                    archive.write(staging / name, name)
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            for name in ("injected.csv", "ground_truth.csv", "interventions.csv", "annotations.csv"):
                (staging / name).unlink()
            staging.rename(self.path(identifier))
        return metadata
