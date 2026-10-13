"""Preparación independiente de uno o varios CSV etiquetados de Desarrollo."""
from collections import Counter
import csv
import json
from pathlib import Path
import tempfile

from app.config import REVIEW_DATA_DIR, STORAGE_DIR
from app.modulos.clasificador import LABELS
from app.modulos.clasificador.config import Config, actual_partition, parse_utc, partition_bounds
from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.pipeline import build_sequence_dataset
from app.services.csv_etiquetado import labelled_columns, read_label


def inspect_csv(path, bounds):
    path = Path(path).resolve()
    labels, synthetic_labels, entities = Counter(), Counter(), set()
    partitions = {part: Counter() for part in ("train_fit", "dev_tune")}
    rows = synthetic_rows = 0
    start = end = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = [name.strip().lower() for name in reader.fieldnames or []]
        if (len(headers) != len(set(headers)) or not {"entity_id", "state", "last_changed"}.issubset(headers)
                or labelled_columns(headers) is None):
            raise ValueError(f"{path.name}: se requieren entity_id, state, last_changed, es_sintetica, tipo y tipo_falla")
        for number, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"{path.name}, fila {number}: número de columnas inválido")
            row = {key.strip().lower(): value for key, value in raw.items()}
            try:
                at = parse_utc(row["last_changed"])
                label = read_label([row[name] for name in ("es_sintetica", "tipo", "tipo_falla")], [0, 1, 2])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path.name}, fila {number}: {exc}") from exc
            if not row["entity_id"].strip():
                raise ValueError(f"{path.name}, fila {number}: entidad vacía")
            part = actual_partition(at, bounds)
            if part not in partitions:
                raise ValueError(f"{path.name}, fila {number}: fecha fuera de Desarrollo")
            entities.add(row["entity_id"])
            start, end = min(start, at) if start else at, max(end, at) if end else at
            rows += 1
            synthetic_rows += label["es_sintetica"] == "true"
            if label["evaluable"]:
                labels[label["label"]] += 1
                partitions[part][label["label"]] += 1
                if label["es_sintetica"] == "true":
                    synthetic_labels[label["label"]] += 1
    if not rows or not labels:
        raise ValueError(f"{path.name}: el CSV está vacío o no contiene etiquetas supervisadas")
    digest = sha256_file(path)
    return {"path": str(path), "dataset_id": digest[:32], "filename": path.name,
        "partition": "development", "sha256": digest, "rows": rows,
        "columns": headers, "labelled_import": True, "imported_labels": sum(labels.values()),
        "imported_synthetic": sum(synthetic_labels.values()), "synthetic_rows": synthetic_rows,
        "start": start.isoformat(), "end": end.isoformat(), "raw_label_counts": dict(labels),
        "raw_label_counts_by_partition": {part: dict(counts) for part, counts in partitions.items()},
        "entities": [{"entity_id": entity, "unit": ""} for entity in sorted(entities)]}


class CsvSources:
    """Fuentes inmutables por hash; no importa los CSV en la base de revisión."""
    def __init__(self, sources):
        self.sources = sources

    def list_datasets(self, partition=None):
        return self.sources if partition in (None, "development") else []

    def dataset(self, identifier):
        return next(source for source in self.sources if source["dataset_id"] == identifier)

    def source_path(self, identifier):
        return Path(self.dataset(identifier)["path"])


def prepare_csv_dataset(paths, *, config=None, review_dir=None, manifest_path=None,
                        development_start=None, development_end=None, inspect_only=False,
                        progress=lambda *_: None):
    config = config or Config()
    if not paths:
        raise ValueError("Indica al menos un --csv")
    if manifest_path is not None:
        if development_start or development_end:
            raise ValueError("Usa --manifest o los límites de Desarrollo, no ambos")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    else:
        if not development_start or not development_end:
            raise ValueError("Indica --development-start y --development-end, o --manifest")
        start, end = parse_utc(development_start), parse_utc(development_end)
        if start >= end:
            raise ValueError("El inicio de Desarrollo debe preceder a su fin exclusivo")
        # Only Development is supplied. Held-out partitions are empty and never loaded.
        manifest = {"boundaries": [start.isoformat(), end.isoformat(), end.isoformat(), end.isoformat()],
                    "partitions": {}}
    bounds = partition_bounds(manifest)
    sources = [inspect_csv(path, bounds) for path in paths]
    hashes, owners = set(), {}
    for source in sources:
        if source["sha256"] in hashes:
            raise ValueError("El mismo CSV no puede incorporarse dos veces")
        hashes.add(source["sha256"])
        for entity in source["entities"]:
            name = entity["entity_id"]
            if name in owners:
                raise ValueError(f"La entidad {name} aparece en dos CSV; usa identificadores distintos para históricos independientes")
            owners[name] = source["filename"]
    if len(owners) > config.max_nodes:
        raise ValueError(f"Los CSV suman {len(owners)} entidades; máximo configurado {config.max_nodes}")
    report = {"sources": [{key: value for key, value in source.items() if key != "path"} for source in sources],
              "rows": sum(source["rows"] for source in sources), "entities": len(owners),
              "split_bounds": {part: [start.isoformat(), end.isoformat() if end else None]
                               for part, (start, end) in bounds.items()}, "inspect_only": inspect_only}
    if inspect_only:
        return report
    from app.services.diagnostico import DiagnosticWorkbench, _write
    workbench = DiagnosticWorkbench(Path(review_dir) if review_dir else REVIEW_DATA_DIR)
    cache = (Path(review_dir) if review_dir else STORAGE_DIR) / "preparation"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gat-csv-", dir=cache) as temporary:
        split_path = Path(temporary) / "manifest.json"
        split_path.write_text(json.dumps(manifest), encoding="utf-8")
        dataset = build_sequence_dataset(CsvSources(sources),
            [f"review:{source['dataset_id']}" for source in sources], [], config,
            workbench.directory, split_path, progress)
        dataset["audit_id"] = sha256_file(split_path)
    evidence = set()
    if any(source["imported_synthetic"] for source in sources):
        evidence.add("controlled_injection")
    if any(source["imported_labels"] > source["imported_synthetic"] for source in sources):
        evidence.add("historical_review")
    ready = all(dataset["label_counts"][part].get(label, 0) > 0
                and dataset["episode_counts"][part].get(label, 0) > 0
                for part in ("train_fit", "dev_tune") for label in LABELS)
    ready = ready and any(item["windows"] for item in dataset["fault_type_inventory"]["train_fit"].values())
    dataset.update(evidence_origins=sorted(evidence), label_support=dataset["label_counts"],
                   evaluation_status="candidate" if ready else "insufficient_labels",
                   ready_for_training=ready, split_bounds=report["split_bounds"])
    _write(Path(dataset["sequence_dir"]) / "split_manifest.json", manifest)
    _write(workbench._path("datasets", dataset["dataset_id"]), dataset)
    progress(100, "Dataset conjunto listo; se conservaron las fuentes y sus episodios")
    return dataset
