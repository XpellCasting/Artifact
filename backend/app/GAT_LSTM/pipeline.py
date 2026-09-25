"""Construcción persistente desde particiones canónicas con identidad verificable."""
from __future__ import annotations

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

import numpy as np

from . import LABELS
from .audit import sha256_file
from .config import (Config, actual_partition, parse_utc, partition_bounds,
                     resolve_manifest_source)
from .dataset import index_labelled_targets
from .features import fit_preprocessor, state_id
from .graph import build_graph
from .temporal import build_causal_sample
from .storage import DiskPartition, PartitionWriter
from ..Tipos_Entidad import infer_kind
from ..Estados_Validos import valid_state
from ..CSV_Etiquetado import labelled_columns, read_label


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
        result.extend(review_store.list_annotations(dataset_id=dataset_id))
    return result


def _load_events(path: Path, *, read_row_labels=False):
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
                     "state": str(row.get("state", "")), "source_row": source_row}
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
            Path(source["path"]), read_row_labels=bool(source.get("labelled_import")))
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
    preprocessor = fit_preprocessor(train_fit, train_ann, config, entities)
    catalog = [{"entity_id": entity, "sensor_kind": kind, "domain": entity.split(".", 1)[0]}
               for entity, kind in entities.items()]
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
    root = Path(diagnostics_dir) / "sequences"
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / identifier
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=root))
    try:
        with ExitStack() as stack:
            supervised = {part: stack.enter_context(PartitionWriter(temp_dir, part))
                          for part in ("train_fit", "dev_tune")}
            predictive = {part: stack.enter_context(PartitionWriter(temp_dir, f"{part}_pretrain",
                                                                   predictive=True))
                          for part in ("train_fit", "dev_tune")}
            annotations_by_id = {a["annotation_id"]: a for a in all_annotations}
            for index, target in enumerate(targets):
                target_time = parse_utc(target.timestamp)
                # Each sample supervises the annotated node only. Other nodes are context.
                sample = build_causal_sample(provider, target_time, preprocessor, graph, config,
                                             {target.entity_id: target.label})
                original = annotations_by_id[target.annotation_id]
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
            if config.pretrain_enabled:
                progress(65, "Preparando las secuencias del preentrenamiento opcional")
                failure_annotations = [a for a in development_annotations(review_store, source_sha)
                    if a.get("source_sha256") in source_sha and a.get("label") == "sensor_fault"
                    and a.get("status") in {"reviewed", "verified"}]
                candidate_pools = {part: defaultdict(list) for part in ("train_fit", "dev_tune")}
                for entity in preprocessor["entity_ids"]:
                    previous_future = {"train_fit": None, "dev_tune": None}
                    for rows in merged_index.get(entity, []):
                        future_at = rows["_time"]
                        part = actual_partition(future_at, bounds)
                        if part not in candidate_pools:
                            continue
                        part_start, part_end = split_bounds[part]
                        target_at = future_at - timedelta(seconds=config.grid_seconds)
                        if target_at - timedelta(seconds=config.context_seconds) < part_start:
                            continue
                        if part_end is not None and future_at >= part_end:
                            continue
                        prior = previous_future[part]
                        if prior is not None and (future_at - prior).total_seconds() < config.pretrain_stride_seconds:
                            continue
                        if any(a["entity_id"] == entity and parse_utc(a["start"]) <= future_at <= parse_utc(a["end"])
                               for a in failure_annotations):
                            continue
                        info = preprocessor["entities"][entity]
                        if not valid_state(rows["state"]):
                            continue
                        if info["kind"] == "continuous":
                            try:
                                raw_value = float(rows["state"])
                            except (ValueError, TypeError, OverflowError):
                                continue
                            if not np.isfinite(raw_value): continue
                            target_value = (raw_value - info.get("median", 0.0)) / info.get("scale", 1.0)
                        else:
                            target_value = state_id(rows["state"], info)
                            if target_value == 2: continue
                        candidate_pools[part][entity].append((target_at, rows, target_value))
                        previous_future[part] = future_at
                for part, by_entity in candidate_pools.items():
                    cursors = {entity: 0 for entity in by_entity}
                    candidates = []
                    entities_round_robin = sorted(by_entity)
                    while len(candidates) < config.pretrain_max_samples:
                        progressed = False
                        for entity in entities_round_robin:
                            index = cursors[entity]
                            if index < len(by_entity[entity]):
                                candidates.append((entity, *by_entity[entity][index]))
                                cursors[entity] += 1; progressed = True
                                if len(candidates) >= config.pretrain_max_samples: break
                        if not progressed: break
                    for entity, target_at, future_row, target_value in candidates:
                        sample = build_causal_sample(provider, target_at, preprocessor, graph, config)
                        targets_vec = [0.0] * len(preprocessor["entity_ids"])
                        observed = [False] * len(preprocessor["entity_ids"])
                        node = preprocessor["entity_ids"].index(entity)
                        targets_vec[node] = target_value
                        observed[node] = True
                        sample["next_target"], sample["next_observed"] = targets_vec, observed
                        sample["metadata"].update({"sample_id": hashlib.sha256(
                            f"pretrain:{entity}:{future_row['timestamp']}".encode()).hexdigest(),
                            "partition": part, "next_entity": entity,
                            "target_time": target_at.isoformat(), "next_time": future_row["_time"].isoformat(),
                            "partition_start": split_bounds[part][0].isoformat(),
                            "partition_end": split_bounds[part][1].isoformat(),
                            "evidence_origin": "unlabelled_history"})
                        predictive[part].append(sample)
                        pretrain_support[part][entity] += 1

            progress(85, "Guardando las secuencias del dataset en disco")
            partition_counts, label_counts, episode_counts = {}, {}, {}
            for part, writer in supervised.items():
                partition_counts[part] = writer.count
                label_counts[part] = dict(writer.labels)
                episode_counts[part] = {label: len(groups) for label, groups in writer.episodes.items()}
                writer.finish()
            for writer in predictive.values():
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


def load_pretraining_partition(dataset: dict, partition: str):
    directory = Path(dataset["sequence_dir"])
    prefix = f"{partition}_pretrain"
    if not (directory / f"{prefix}_metadata.jsonl").exists():
        return []
    return DiskPartition(directory, prefix, predictive=True)
