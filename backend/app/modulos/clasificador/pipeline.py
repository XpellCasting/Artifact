"""Construcción persistente desde particiones canónicas con identidad verificable."""
from __future__ import annotations
from app.config import artifact_directory

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4


from app.modulos.clasificador import LABELS
from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.config import actual_partition
from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.config import partition_bounds
from app.modulos.clasificador.config import resolve_manifest_source
from app.modulos.clasificador.dataset import index_labelled_targets
from app.modulos.clasificador.features import fit_preprocessor
from app.modulos.clasificador.graph import build_graph
from app.modulos.clasificador.temporal import build_causal_sample
from app.modulos.clasificador.storage import DiskPartition
from app.modulos.clasificador.storage import PartitionWriter
from app.services.tipos_entidad import infer_kind
from app.services.csv_etiquetado import labelled_columns
from app.services.csv_etiquetado import read_label
from app.modulos.clasificador.catalogo import CATALOG


def canonical_sources(manifest_path: Path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    normalize = {"desarrollo": "development", "validacion": "validation", "prueba": "test"}
    return [{"source_id": f"canonical:{name}", "partition": normalize.get(name, name),
             "manifest_partition": name,
             "filename": resolve_manifest_source(manifest_path, info["path"]).name, "sha256": info["sha256"],
             "rows": info["rows"], "path": str(resolve_manifest_source(manifest_path, info["path"]))}
            for name, info in manifest["partitions"].items() if name != "reserva_agosto"]


def development_sources(manifest_path: Path, review_store):
    """Fuentes que el ajuste GAT puede leer: exclusivamente Desarrollo."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    partition_names = {"development": "development", "desarrollo": "development"}
    name = next((candidate for candidate in manifest["partitions"]
                 if candidate in partition_names and partition_names[candidate] == "development"), None)
    sources = []
    if name is not None:
        info = manifest["partitions"][name]
        path = resolve_manifest_source(manifest_path, info["path"])
        sources.append({"source_id": "canonical:desarrollo", "partition": "development",
            "manifest_partition": name, "filename": path.name, "sha256": info["sha256"],
            "rows": info["rows"], "path": str(path), "source_kind": "canonical",
            "labelled_import": False})
    for item in review_store.list_datasets(partition="development"):
        if item.get("partition") != "development" or not item.get("labelled_import"):
            continue
        try:
            columns = labelled_columns(item.get("columns", []))
        except ValueError:
            continue
        if columns is None or item.get("imported_labels", 0) <= 0:
            continue
        source_path = review_store.source_path(item["dataset_id"])
        sources.append({"source_id": f"review:{item['dataset_id']}",
                        "dataset_id": item["dataset_id"], "partition": "development",
                        "filename": item["filename"], "sha256": item["sha256"],
                        "rows": item["rows"], "path": str(source_path),
                        "source_kind": "labelled_csv", "labelled_import": True,
                        "imported_labels": item.get("imported_labels", 0),
                        "imported_synthetic": item.get("imported_synthetic", 0),
                        "start": item.get("start"), "end": item.get("end")})
    return sources


def development_annotations(review_store, source_hashes):
    """Carga anotaciones únicamente de datasets de Desarrollo con los hashes elegidos."""
    hashes = set(source_hashes)
    result, seen = [], set()
    for dataset in review_store.list_datasets(partition="development"):
        dataset_id = dataset.get("dataset_id")
        if dataset.get("sha256") not in hashes or dataset_id in seen:
            continue
        seen.add(dataset_id)
        result.extend({**annotation, "partition": "development"}
                      for annotation in review_store.list_annotations(dataset_id=dataset_id))
    return result


def _load_events(path: Path, *, read_row_labels=False, source_id=None):
    by_entity = defaultdict(list)
    kinds_states = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = labelled_columns(reader.fieldnames or []) if read_row_labels else None
        required = {"entity_id", "state"}
        if not required.issubset({name.strip().lower() for name in reader.fieldnames or []}):
            raise ValueError(f"Columnas faltantes en fuente canónica: {path.name}")
        for source_row, raw in enumerate(reader, 2):
            row = {str(k).strip().lower(): v for k, v in raw.items()}
            entity = str(row.get("entity_id", ""))
            timestamp = row.get("last_changed") or row.get("timestamp")
            if not entity or not timestamp:
                continue
            at = parse_utc(timestamp)
            value = {"entity_id": entity, "timestamp": at.isoformat(), "_time": at,
                     "state": str(row.get("state", "")), "source_row": source_row,
                     "provenance": source_id or path.name}
            if columns is not None:
                imported = read_label([raw.get(reader.fieldnames[index], "") for index in columns],
                                      [0, 1, 2])
                value.update({"label": imported["label"], "fault_type": imported["fault_type"],
                              "es_sintetica": imported["es_sintetica"] == "true"})
            by_entity[entity].append(value)
            kinds_states[entity].add(value["state"])
    for rows in by_entity.values():
        rows.sort(key=lambda item: (item["_time"], item["source_row"]))
    return by_entity, kinds_states


def _row_label_annotations(event_index, source, config):
    """Convierte etiquetas por fila en objetivos causales espaciados por episodio."""
    annotations = []
    gap_limit = max(config.labelled_stride_seconds * 3, config.feature_window_seconds)
    for entity, rows in event_index[source["source_id"]].items():
        previous_signature = None
        previous_time = None
        episode_id = None
        last_target = None
        for row in rows:
            label = row.get("label")
            if label not in LABELS:
                previous_signature = None
                previous_time = None
                episode_id = None
                last_target = None
                continue
            signature = (label, row.get("fault_type"), row.get("es_sintetica", False))
            gap = ((row["_time"] - previous_time).total_seconds()
                   if previous_time is not None else None)
            if signature != previous_signature or gap is None or gap > gap_limit:
                episode_id = hashlib.sha256(
                    f"{source['sha256']}:{entity}:{row['timestamp']}:{signature}".encode()).hexdigest()[:32]
                last_target = None
            if last_target is None or (row["_time"] - last_target).total_seconds() >= config.labelled_stride_seconds:
                identifier = hashlib.sha256(
                    f"row-label:{source['sha256']}:{row['source_row']}".encode()).hexdigest()[:32]
                annotations.append({"annotation_id": identifier, "dataset_id": source["dataset_id"],
                    "source_sha256": source["sha256"], "source_row": row["source_row"],
                    "entity_id": entity, "start": row["timestamp"], "end": row["timestamp"],
                    "label": label, "fault_type": row.get("fault_type"), "status": "reviewed",
                    "reference_origin": "controlled_injection" if row.get("es_sintetica") else "historical_review",
                    "parent_episode_id": episode_id, "es_sintetica": bool(row.get("es_sintetica")),
                    "label_source": source["filename"]})
                last_target = row["_time"]
            previous_signature, previous_time = signature, row["_time"]
    return annotations


def _provider(index):
    timelines = {entity: [row["_time"] for row in rows] for entity, rows in index.items()}
    def events(start, end, entities):
        result = []
        for entity in entities:
            rows = index.get(entity, [])
            times = timelines.get(entity, [])
            left, right = bisect_left(times, start), bisect_right(times, end)
            left = max(0, left - 1)
            result.extend({key: value for key, value in row.items() if key != "_time"}
                          for row in rows[left:right])
        return result
    return events


def _atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def build_sequence_dataset(review_store, source_ids: list[str], annotation_ids: list[str],
                           config: Config, diagnostics_dir: Path, manifest_path: Path,
                           progress=lambda *_: None) -> dict:
    """Builds reloadable NPY blocks; inputs are selected by stable source IDs only."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    bounds = partition_bounds(manifest)
    available = {item["source_id"]: item for item in development_sources(manifest_path, review_store)}
    selected = list(dict.fromkeys(source_ids))
    if not selected:
        raise ValueError("Selecciona una fuente etiquetada de Desarrollo")
    if len(selected) != 1:
        raise ValueError("Selecciona una sola fuente de Desarrollo para evitar mezclar versiones del histórico")
    sources, exclusions = {}, []
    for identifier in selected:
        if identifier not in available:
            raise ValueError(f"Fuente no permitida: GAT-LSTM solo admite fuentes de Desarrollo ({identifier})")
        item = available[identifier]
        if item.get("partition") != "development":
            raise ValueError("GAT-LSTM solo admite fuentes de Desarrollo; Validación y Prueba están reservadas")
        path = Path(item["path"])
        if item.get("start") and parse_utc(item["start"]) < bounds["train_fit"][0]:
            raise ValueError("La fuente contiene datos anteriores al límite de Desarrollo")
        if item.get("end") and (bounds["dev_tune"][1] is None
                                  or parse_utc(item["end"]) >= bounds["dev_tune"][1]):
            raise ValueError("La fuente rebasa Desarrollo; Validación y Prueba no se cargan en GAT-LSTM")
        actual_hash = sha256_file(path) if path.is_file() else None
        if actual_hash != item["sha256"]:
            exclusions.append({"source_id": identifier, "reason": "source_missing_or_hash_mismatch"})
            continue
        sources[identifier] = {**item, "actual_sha256": actual_hash}
    if not sources:
        raise ValueError("Ninguna fuente seleccionada pasó la verificación de identidad")
    progress(5, "Verificando identidad y leyendo fuentes seleccionadas")

    event_index, state_values = {}, {}
    for name, source in sources.items():
        event_index[name], state_values[name] = _load_events(
            Path(source["path"]), read_row_labels=bool(source.get("labelled_import")),
            source_id=source["source_id"])
    eligible_events = defaultdict(list)
    for rows_by_entity in event_index.values():
        for rows in rows_by_entity.values():
            for row in rows:
                part = actual_partition(row["_time"], bounds)
                if part in {"train_fit", "dev_tune"}:
                    eligible_events[part].append({key: value for key, value in row.items() if key != "_time"})
    train_fit = eligible_events["train_fit"]
    entities_train = defaultdict(set)
    for row in train_fit:
        entities_train[row["entity_id"]].add(row["state"])
    all_entities = sorted(entities_train)
    if len(all_entities) > config.max_nodes:
        raise ValueError(f"El conjunto contiene {len(all_entities)} entidades; máximo configurado {config.max_nodes}")
    entities = {}
    for entity in all_entities:
        domain = entity.split(".", 1)[0]
        if config.architecture_version == "gat_lstm_v3_segments_multiscale":
            from app.modulos.clasificador.features import _finite_number
            from app.services.estados_validos import valid_state
            values = [v for v in entities_train[entity] if valid_state(v)]
            entities[entity] = "continuous" if values and all(_finite_number(v) for v in values) else "multistate"
            mode = (config.entity_value_modes or {}).get(entity)
            if mode:
                entities[entity] = "continuous" if mode == "numeric" else "multistate"
        else:
            entities[entity] = infer_kind(entity, entities_train[entity], domain=domain)
    split_bounds = {"train_fit": bounds["train_fit"], "dev_tune": bounds["dev_tune"]}
    if not train_fit:
        raise ValueError("La fuente seleccionada no contiene eventos dentro de train_fit")
    progress(15, "Ajustando preprocesador y grafo solo con train_fit")

    selected_annotations = [review_store.annotation(identifier) for identifier in dict.fromkeys(annotation_ids)]
    source_sha = {item["sha256"] for item in sources.values()}
    linked_annotations = []
    for annotation in selected_annotations:
        if annotation.get("source_sha256") not in source_sha:
            exclusions.append({"annotation_id": annotation.get("annotation_id"),
                               "reason": "annotation_source_not_selected_or_not_identical"})
            continue
        if annotation.get("source_row") is not None:
            source = next(item for item in sources.values()
                          if item["sha256"] == annotation.get("source_sha256"))
            source_events = event_index[source["source_id"]].get(annotation["entity_id"], [])
            row = next((event for event in source_events
                        if event["source_row"] == annotation["source_row"]), None)
            if (row is None or parse_utc(row["timestamp"]) != parse_utc(annotation["start"])
                    or parse_utc(annotation["start"]) != parse_utc(annotation["end"])):
                exclusions.append({"annotation_id": annotation.get("annotation_id"),
                                   "reason": "source_row_identity_mismatch"})
                continue
        linked_annotations.append(annotation)
    row_annotations = []
    for source in sources.values():
        if source.get("labelled_import"):
            row_annotations.extend(_row_label_annotations(event_index, source, config))
    all_annotations = linked_annotations + row_annotations
    train_ann = [a for a in all_annotations
                 if actual_partition(parse_utc(a["start"]), bounds) == "train_fit"
                 and actual_partition(parse_utc(a["end"]), bounds) == "train_fit"]
    preprocessor = fit_preprocessor(train_fit, [] if config.architecture_version == "gat_lstm_v3_segments_multiscale" else train_ann, config, entities)
    for source in sources.values():
        if source.get("dataset_id"):
            source_dataset = review_store.dataset(source["dataset_id"])
            for entity in source_dataset.get("entities", []):
                if entity["entity_id"] in preprocessor["entities"]:
                    preprocessor["entities"][entity["entity_id"]]["unit"] = entity.get("unit") or ""
    shared_type_model = config.architecture_version == "gat_lstm_v2_shared_type"
    if shared_type_model:
        preprocessor["feature_names"].extend([
            "state_transition_ratio", "dominant_state_fraction", "state_diversity_ratio"])
        preprocessor["architecture_version"] = config.architecture_version
    catalog = [{"entity_id": entity, "sensor_kind": kind, "domain": entity.split(".", 1)[0]}
               for entity, kind in entities.items()]
    if config.architecture_version == "gat_lstm_v3_segments_multiscale":
        from app.modulos.clasificador.segments import fit_spec
        preprocessor["architecture_version"] = config.architecture_version
        graph = fit_spec(train_fit, preprocessor, config, bounds["train_fit"])
    else:
        graph = build_graph(train_fit, catalog, config)

    merged_index = defaultdict(list)
    for rows_by_entity in event_index.values():
        for entity, rows in rows_by_entity.items():
            merged_index[entity].extend(rows)
    for rows in merged_index.values():
        rows.sort(key=lambda item: (item["_time"], item["source_row"]))
    provider = _provider(merged_index)
    targets, target_exclusions = index_labelled_targets(all_annotations, provider,
                                                        split_bounds, config)
    exclusions.extend(target_exclusions)
    identifier = uuid4().hex
    root = artifact_directory(diagnostics_dir, "sequences")
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / identifier
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=root))
    fault_types = {item["fault_type"] for item in CATALOG}
    fault_type_windows = {part: Counter() for part in ("train_fit", "dev_tune")}
    fault_type_entity_windows = {part: defaultdict(Counter) for part in ("train_fit", "dev_tune")}
    fault_type_episodes = {part: defaultdict(lambda: defaultdict(set))
                           for part in ("train_fit", "dev_tune")}
    try:
        with ExitStack() as stack:
            supervised = {part: stack.enter_context(PartitionWriter(temp_dir, part))
                          for part in ("train_fit", "dev_tune")}
            annotations_by_id = {a["annotation_id"]: a for a in all_annotations}
            for index, target in enumerate(targets):
                target_time = parse_utc(target.timestamp)
                # Each sample supervises the annotated node only. Other nodes are context.
                sample = build_causal_sample(provider, target_time, preprocessor, graph, config,
                                             {target.entity_id: target.label},
                                             portable_features=shared_type_model)
                original = annotations_by_id[target.annotation_id]
                fault_type = original.get("fault_type")
                if (target.label == LABELS.index("sensor_fault") and fault_type in fault_types):
                    fault_type_windows[target.partition][fault_type] += 1
                    fault_type_entity_windows[target.partition][fault_type][target.entity_id] += 1
                    fault_type_episodes[target.partition][fault_type][target.entity_id].add(target.episode_id)
                sample["metadata"].update({"sample_id": hashlib.sha256(
                    f"{target.annotation_id}:{target.timestamp}:{target.entity_id}".encode()).hexdigest(),
                    "annotation_id": target.annotation_id, "parent_episode_id": target.episode_id,
                    "source_sha256": original.get("source_sha256"), "source_dataset_id": original.get("dataset_id"),
                    "partition": target.partition, "evidence_origin": original.get("reference_origin", "unspecified"),
                    "status": original.get("status"), "fault_subtype": original.get("fault_type"),
                    "context_entities": sample["metadata"]["entity_ids"],
                    "context_start": (target_time - timedelta(seconds=config.context_seconds)).isoformat(),
                    "context_end": target.timestamp, "source_row": target.source_row,
                    "target_label": target.label, "es_sintetica": bool(original.get("es_sintetica", False)),
                    "label_source": original.get("label_source")})
                supervised[target.partition].append(sample)
                if index % 25 == 0 or index + 1 == len(targets):
                    progress(35 + 30 * (index + 1) / max(1, len(targets)),
                             f"Construyendo secuencias etiquetadas: {index + 1:,} de {len(targets):,}")

            pretrain_support = {"train_fit": Counter(), "dev_tune": Counter()}

            progress(85, "Guardando las secuencias del dataset en disco")
            partition_counts, label_counts, episode_counts = {}, {}, {}
            for part, writer in supervised.items():
                partition_counts[part] = writer.count
                label_counts[part] = dict(writer.labels)
                episode_counts[part] = {label: len(groups) for label, groups in writer.episodes.items()}
                writer.finish()
        _atomic_json(temp_dir / "preprocessor.json", preprocessor)
        progress(95, "Guardando el grafo y la configuración del dataset")
        _atomic_json(temp_dir / "graph.json", graph)
        _atomic_json(temp_dir / "config.json", config.as_dict())
        os.replace(temp_dir, final_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    partition_counts.setdefault("validation", 0); partition_counts.setdefault("test", 0)
    label_counts.setdefault("validation", {}); label_counts.setdefault("test", {})
    episode_counts.setdefault("validation", {}); episode_counts.setdefault("test", {})
    fault_type_inventory = {}
    for part in ("train_fit", "dev_tune"):
        fault_type_inventory[part] = {}
        for fault_type in sorted(fault_types):
            by_entity = fault_type_episodes[part].get(fault_type, {})
            episodes = sum(len(ids) for ids in by_entity.values())
            fault_type_inventory[part][fault_type] = {
                "windows": fault_type_windows[part][fault_type],
                "independent_episodes": episodes,
                "entities": len(by_entity),
                "by_entity": {entity: {"windows": fault_type_entity_windows[part][fault_type][entity],
                                        "independent_episodes": len(ids)}
                              for entity, ids in sorted(by_entity.items())},
                "minimum_independent_episodes": 3,
                "support_status": "supported" if episodes >= 3 else "insufficient_support"}
    result = {"dataset_id": identifier, "method": "gat_lstm", "created_at": datetime.now(timezone.utc).isoformat(),
             "sequence_dir": str(final_dir), "source_ids": selected,
             "source_hashes": {name: source["actual_sha256"] for name, source in sources.items()},
             "annotation_ids": list(dict.fromkeys(a["annotation_id"] for a in linked_annotations)),
             "labelled_source_ids": [name for name, source in sources.items() if source.get("labelled_import")],
             "training_scope": "development_only",
             "entity_ids": preprocessor["entity_ids"], "entity_kinds": preprocessor["entity_kinds"],
             "context_seconds": config.context_seconds, "partition_counts": partition_counts,
             "label_counts": label_counts, "episode_counts": episode_counts, "exclusions": exclusions,
             "pretrain_counts": {part: sum(counts.values()) for part, counts in pretrain_support.items()},
             "pretrain_support_by_entity": {part: dict(counts) for part, counts in pretrain_support.items()},
             "fault_type_inventory": fault_type_inventory,
             "spatial_evidence": graph["spatial_evidence"], "graph_version": graph["graph_version"], "exploratory": False,
             "config": config.as_dict(), "schema_version": config.schema_version}
    progress(98, "Finalizando el registro del dataset")
    return result


def load_partition(dataset: dict, partition: str):
    """Abre arrays en disco sin convertir la partición completa en listas."""
    directory = Path(dataset["sequence_dir"])
    if not (directory / f"{partition}_metadata.jsonl").exists():
        return []
    return DiskPartition(directory, partition)


