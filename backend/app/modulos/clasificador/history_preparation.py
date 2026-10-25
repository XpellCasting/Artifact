"""Contrato portable de históricos independientes, con cobertura comprobada.

Los CSV y sus UTC son inmutables. El índice de objetivos conserva los ID de
inyección; las ventanas se construyen dentro de cada mundo y partición.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
import csv
from datetime import datetime, timedelta, timezone
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from app.modulos.clasificador import LABELS
from app.modulos.clasificador.config import Config, parse_utc
from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.model_constants import FAULT_TYPES
from app.services.csv_etiquetado import labelled_columns, read_label

PARTS = ("train_fit", "dev_tune")
TARGET_FIELDS = ("world_id", "source_row", "entity_id", "timestamp", "partition",
                 "label", "fault_subtype", "episode_id", "evidence_origin", "es_sintetica")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    entities = manifest["classifier_entities"]
    if len(entities) != 15 or len(set(entities)) != 15:
        raise ValueError("El contrato requiere exactamente los 15 sensores seleccionados")
    worlds = manifest["worlds"]
    if {w["world_id"] for w in worlds} != {"real", "synthetic"} or len(worlds) != 2:
        raise ValueError("Se requieren los mundos real y synthetic independientes")
    for world in worlds:
        lo, split, dev, hi = [parse_utc(world[k]) for k in
                             ("train_start", "train_end", "dev_start", "dev_end")]
        if not lo < split == dev < hi:
            raise ValueError("Cortes temporales inválidos")
    return path.parent, manifest


def source_path(root, manifest, world, key):
    return root / manifest["source_directory"] / world[key]


def world_bounds(world):
    return {"train_fit": (parse_utc(world["train_start"]), parse_utc(world["train_end"])),
            "dev_tune": (parse_utc(world["dev_start"]), parse_utc(world["dev_end"]))}


def partition(at, bounds):
    return next((part for part, (lo, hi) in bounds.items() if lo <= at < hi), None)


def labelled_rows(root, manifest, world):
    """Asocia filas por identidad completa, nunca mediante una unión aproximada."""
    with source_path(root, manifest, world, "path").open(encoding="utf-8-sig", newline="") as source, \
            source_path(root, manifest, world, "audit").open(encoding="utf-8-sig", newline="") as audit:
        reader, side = csv.DictReader(source), csv.DictReader(audit)
        fields = reader.fieldnames
        columns = labelled_columns(fields)
        if columns is None:
            raise ValueError("Faltan etiquetas tipo/tipo_falla/es_sintetica")
        for number, pair in enumerate(zip_longest(reader, side), 2):
            row, evidence = pair
            if row is None or evidence is None:
                raise ValueError("El CSV y la auditoría tienen distinta cantidad de filas")
            if (str(number) != evidence["source_row"] or evidence["world_id"] != world["world_id"]
                    or any(row[k] != evidence[k] for k in
                           ("entity_id", "state", "last_changed", "tipo", "tipo_falla"))):
                raise ValueError(f"Identidad de fila distinta en {world['world_id']}:{number}")
            label = read_label([row[k] for k in fields], columns)
            yield number, row, evidence, label


def evenly_spaced(rows, count):
    """Conserva extremos y elige instantes reales sin desplazar las etiquetas."""
    rows = sorted(rows, key=lambda r: (r["timestamp"], r["source_row"]))
    if len(rows) <= count:
        return rows
    return [rows[i] for i in sorted({round(i * (len(rows) - 1) / (count - 1))
                                     for i in range(count)})]


def validate_coverage(targets, episodes, entities, minimum=3):
    """Falla antes de preparar si se perdió un episodio, tipo o sensor."""
    by_id = {e["episode_id"]: e for e in episodes}
    pairs = Counter((e["entity_id"], e["fault_type"]) for e in episodes)
    if len(by_id) != len(episodes) or any(n != 1 for n in pairs.values()):
        raise ValueError("Se repite una inyección o una combinación sensor/tipo")
    observed, groups, counts = set(), defaultdict(set), Counter()
    sensor_counts = {part: Counter() for part in PARTS}
    class_counts = {part: Counter() for part in PARTS}
    group_parts = defaultdict(set)
    for target in targets:
        part, entity = target["partition"], target["entity_id"]
        if part not in PARTS or entity not in entities or target["label"] not in LABELS:
            raise ValueError("Objetivo fuera del contrato")
        sensor_counts[part][entity] += 1
        class_counts[part][target["label"]] += 1
        group_parts[target["episode_id"]].add(part)
        if target["label"] == "sensor_fault":
            episode = by_id.get(target["episode_id"])
            if not episode or any(target[k] != episode[v] for k, v in
                (("world_id", "world_id"), ("entity_id", "entity_id"),
                 ("partition", "partition"), ("fault_subtype", "fault_type"))):
                raise ValueError("Objetivo de falla sin correspondencia con el episodio auditado")
            observed.add(episode["episode_id"])
            groups[part, target["fault_subtype"]].add(target["episode_id"])
            counts[part, target["fault_subtype"]] += 1
    if observed != set(by_id):
        raise ValueError(f"Se perdieron episodios de inyección: {sorted(set(by_id) - observed)}")
    if any(len(parts) != 1 for parts in group_parts.values()):
        raise ValueError("Un episodio aparece en ajuste y validación")
    inventory = {}
    for part in PARTS:
        if set(sensor_counts[part]) != set(entities):
            raise ValueError(f"Faltan sensores supervisados en {part}")
        if set(class_counts[part]) != set(LABELS):
            raise ValueError(f"Faltan clases generales en {part}")
        inventory[part] = {}
        for kind in FAULT_TYPES:
            n = len(groups[part, kind])
            if n < minimum:
                raise ValueError(f"Cobertura insuficiente {part}/{kind}: {n} episodios; mínimo {minimum}")
            inventory[part][kind] = {"independent_episodes": n, "windows": counts[part, kind],
                                     "support_status": "supported"}
    return {"status": "VERIFIED", "injected_episodes": len(observed),
            "unique_entity_fault_pairs": len(pairs), "repeated_pairs": 0,
            "fault_type_inventory": inventory,
            "sensor_target_counts": {k: dict(v) for k, v in sensor_counts.items()},
            "class_target_counts": {k: dict(v) for k, v in class_counts.items()}}


def plan_histories(manifest_path, output, *, normal_stride_seconds=21600, episode_samples=12):
    """Auditoría completa + índice pequeño; no importa PyTorch ni entrena."""
    if normal_stride_seconds < 1 or episode_samples < 2:
        raise ValueError("Stride positivo y al menos dos muestras por episodio")
    root, manifest = read_manifest(manifest_path)
    output = Path(output)
    episodes = json.loads((root / manifest["episodes"]).read_text(encoding="utf-8-sig"))
    by_id = {e["episode_id"]: e for e in episodes}
    changed = set()
    with (root / manifest["source_directory"] / "intervenciones.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["state_changed"].lower() == "true":
                changed.add((row["world_id"], int(row["output_row"])))
    injected, anomalies, targets = defaultdict(list), defaultdict(list), []
    anomaly_parts, last_normal = defaultdict(set), {}
    injected_rows, actual_changed = Counter(), Counter()
    world_summary = {}
    context = timedelta(seconds=manifest["context_seconds"])
    if context.total_seconds() != Config().context_seconds:
        raise ValueError("Contexto incompatible con las escalas del modelo")
    for world in manifest["worlds"]:
        for name, checksum in (("path", "sha256"), ("audit", "audit_sha256")):
            if sha256_file(source_path(root, manifest, world, name)) != world[checksum]:
                raise ValueError(f"Hash de {world['world_id']}/{name} distinto del contrato")
        bounds = world_bounds(world)
        found, found_train, labels, ignored = set(), set(), Counter(), Counter()
        first_seen = {}
        rows_count = 0
        for number, row, evidence, label in labelled_rows(root, manifest, world):
            rows_count += 1
            entity, at = row["entity_id"], parse_utc(row["last_changed"])
            first_seen.setdefault(entity, at)
            found.add(entity)
            part = partition(at, bounds)
            if part is None:
                raise ValueError("El histórico contiene datos ajenos a Desarrollo")
            if part == "train_fit":
                found_train.add(entity)
            labels[label["label"]] += 1
            known = evidence["intervention_known"].lower() == "true"
            eid = evidence["episode_id"]
            if known:
                episode = by_id.get(eid)
                if (episode is None or episode["world_id"] != world["world_id"]
                    or episode["entity_id"] != entity or episode["partition"] != part
                    or label["label"] != "sensor_fault" or label["fault_type"] != episode["fault_type"]
                    or not parse_utc(episode["start"]) <= at <= parse_utc(episode["end"])):
                    raise ValueError("La inyección no coincide con su episodio y partición")
                injected_rows[eid] += 1
                if (world["world_id"], number) in changed:
                    actual_changed[eid] += 1
            if label["label"] == "environmental_anomaly":
                anomaly_parts[world["world_id"], eid].add(part)
            if at - context < bounds[part][0]:
                if known:
                    raise ValueError("La inyección cruza el contexto de su partición")
                ignored["partition_context_warmup"] += 1
                continue
            # La escala corta exige al menos 50 % de una hora observada.
            # Una entidad que aparece tarde no puede supervisarse en su primera lectura.
            if (at - first_seen[entity]).total_seconds() < 1800:
                if known:
                    raise ValueError("Inyección anterior al contexto mínimo del sensor")
                ignored["entity_context_warmup"] += 1
                continue
            target = dict(zip(TARGET_FIELDS, (world["world_id"], number, entity, at.isoformat(), part,
                          label["label"], label["fault_type"] or "", eid,
                          "controlled_injection" if known else "historical_review",
                          label["es_sintetica"])))
            if known:
                # No se supervisa como falla una lectura idéntica al valor limpio.
                if (world["world_id"], number) in changed:
                    injected[eid].append(target)
            elif label["label"] == "normal":
                key = (world["world_id"], part, entity)
                if key not in last_normal or (at - last_normal[key]).total_seconds() >= normal_stride_seconds:
                    last_normal[key] = at
                    target["episode_id"] = f"normal:{world['world_id']}:{part}:{entity}:{at.date()}"
                    targets.append(target)
            elif label["label"] == "environmental_anomaly":
                anomalies[world["world_id"], eid].append(target)
            else:
                ignored["uncertain_or_unverified_fault_target"] += 1
        if rows_count != world["rows"] or found != set(manifest["classifier_entities"]) or found_train != found:
            raise ValueError(f"Filas o sensores perdidos en {world['world_id']}")
        world_summary[world["world_id"]] = {"rows": rows_count, "entities": len(found),
             "train_entities": len(found_train), "labels": dict(labels), "context_only": dict(ignored)}
    for e in episodes:
        if injected_rows[e["episode_id"]] != e["rows"] or actual_changed[e["episode_id"]] != e["changed_rows"]:
            raise ValueError("El recuento de intervención no corresponde a su auditoría")
    for rows in injected.values():
        targets.extend(evenly_spaced(rows, episode_samples))
    for key, rows in anomalies.items():
        if len(anomaly_parts[key]) == 1:
            targets.extend(evenly_spaced(rows, episode_samples))
    targets.sort(key=lambda r: (r["world_id"], r["timestamp"], r["entity_id"], r["source_row"]))
    if len({(r['world_id'], r['source_row']) for r in targets}) != len(targets):
        raise ValueError("Objetivos duplicados")
    coverage = validate_coverage(targets, episodes, manifest["classifier_entities"])
    coverage.update(worlds=world_summary, targets=len(targets),
                    target_policy={"normal_stride_seconds": normal_stride_seconds,
                                   "max_samples_per_injected_or_anomaly_episode": episode_samples,
                                   "fault_supervision": "audited_injections_only",
                                   "reference_fit": "real_train_fit_only"}, training_executed=False)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "targets.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, TARGET_FIELDS)
        writer.writeheader(); writer.writerows(targets)
    coverage["targets_sha256"] = sha256_file(path)
    save_json(output / "coverage.json", coverage)
    return targets, coverage


def load_world_events(root, manifest, world, *, fit_only=False):
    """Un solo mundo por proveedor. Nunca concatena años para crear ventanas."""
    grouped = defaultdict(list)
    lo, hi = world_bounds(world)["train_fit"]
    with source_path(root, manifest, world, "path").open(encoding="utf-8-sig", newline="") as f:
        for number, row in enumerate(csv.DictReader(f), 2):
            at = parse_utc(row["last_changed"])
            if fit_only and not lo <= at < hi:
                continue
            grouped[row["entity_id"]].append({"entity_id": row["entity_id"], "timestamp": at.isoformat(),
                "state": row["state"], "source_row": number, "provenance": world["world_id"]})
    return grouped


def prepare_histories(manifest_path, output, config, progress=print, *, normal_stride_seconds=21600,
                      episode_samples=12, max_targets=None):
    """Construye las secuencias compatibles con gat_lstm_train, sin entrenar."""
    from app.modulos.clasificador.compressed_storage import CompressedPartitionWriter, StorageBudget
    from app.modulos.clasificador.dataset import Target
    from app.modulos.clasificador.features import fit_preprocessor
    from app.modulos.clasificador.preparation import prepared_targets
    from app.modulos.clasificador.segments import fit_spec
    output = Path(output).resolve()
    root, manifest = read_manifest(manifest_path)
    if config.max_nodes < 15:
        raise ValueError("max_nodes no permite conservar los 15 sensores")
    progress("Auditando ambos históricos y construyendo el índice de objetivos")
    targets, coverage = plan_histories(manifest_path, output, normal_stride_seconds=normal_stride_seconds,
                                      episode_samples=episode_samples)
    if max_targets is not None:
        # Prueba técnica representativa: no publicable ni utilizable para entrenar.
        selected = {}
        for r in targets:
            key = (r["partition"], r["label"], r["fault_subtype"] or r["entity_id"])
            selected.setdefault(key, r)
        targets = list(selected.values())[:max_targets]
    identifier = uuid4().hex
    seq_root = output / "review" / "diagnostics" / "sequences"
    seq_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".histories-", dir=seq_root))
    try:
        real = next(w for w in manifest["worlds"] if w["world_id"] == "real")
        progress("Ajustando referencias exclusivamente con real/train_fit; corte explícito del manifiesto")
        fit_index = load_world_events(root, manifest, real, fit_only=True)
        fit_rows = [row for rows in fit_index.values() for row in rows]
        kinds = {key: ("continuous" if (config.entity_value_modes or {}).get(key) == "numeric"
                       else "multistate") for key in manifest["classifier_entities"]}
        if set(config.entity_value_modes or {}) != set(kinds):
            raise ValueError("La configuración debe declarar el modo de los 15 sensores")
        preprocessor = fit_preprocessor(fit_rows, [], config, kinds)
        preprocessor["architecture_version"] = config.architecture_version
        graph = fit_spec(fit_rows, preprocessor, config, world_bounds(real)["train_fit"])
        del fit_rows, fit_index
        save_json(temporary / "preprocessor.json", preprocessor)
        save_json(temporary / "graph.json", graph)
        save_json(temporary / "config.json", config.as_dict())
        actual_targets, preparation_runtime = [], {}
        with ExitStack() as stack:
            budget = StorageBudget(temporary)
            writers = {part: stack.enter_context(CompressedPartitionWriter(temporary, part, budget=budget))
                       for part in PARTS}
            for world in manifest["worlds"]:
                wid = world["world_id"]
                progress(f"Construyendo ventanas de {wid}, sin mezclar proveedores")
                index = load_world_events(root, manifest, world)
                world_rows = [r for r in targets if r["world_id"] == wid]
                by_annotation, indexed = {}, []
                for r in world_rows:
                    key = f"{wid}:{r['source_row']}"
                    by_annotation[key] = r
                    indexed.append(Target(r["entity_id"], r["timestamp"], LABELS.index(r["label"]),
                                          key, r["partition"], r["episode_id"], int(r["source_row"])))
                indexed.sort(key=lambda t: (t.timestamp, t.entity_id))
                stats = preparation_runtime[wid] = {}
                with closing(prepared_targets(indexed, index, preprocessor, graph, config, temporary, stats)) as samples:
                    for i, (target, sample) in enumerate(samples, 1):
                        original = by_annotation[target.annotation_id]
                        node = preprocessor["entity_ids"].index(target.entity_id)
                        if not sample["label_mask"][node]:
                            raise ValueError(f"Objetivo sin cobertura efectiva: {target.annotation_id}")
                        # Identidad del episodio a través de planificación, almacenamiento y pérdida.
                        sample["metadata"].update(sample_id=target.annotation_id,
                            parent_episode_id=target.episode_id, annotation_id=target.annotation_id,
                            source_sha256=world["sha256"], source_dataset_id=wid, world_id=wid,
                            partition=target.partition, target_label=target.label, next_entity=target.entity_id,
                            fault_subtype=original["fault_subtype"] or None,
                            evidence_origin=original["evidence_origin"], status="reviewed",
                            es_sintetica=original["es_sintetica"] == "true",
                            context_start=(parse_utc(target.timestamp)-timedelta(seconds=config.context_seconds)).isoformat(),
                            context_end=target.timestamp, context_entities=preprocessor["entity_ids"],
                            source_row=target.source_row)
                        writers[target.partition].append(sample)
                        actual_targets.append(original)
                        if i % 100 == 0 or i == len(indexed):
                            progress(f"{wid}: {i}/{len(indexed)} objetivos escritos")
                del index
            for writer in writers.values():
                writer.finish()
            if max_targets is None:
                episodes = json.loads((root / manifest["episodes"]).read_text(encoding="utf-8-sig"))
                verified = validate_coverage(actual_targets, episodes, manifest["classifier_entities"])
            else:
                verified = {"status": "PROBE_ONLY"}
            result = {"dataset_id": identifier, "method": "gat_lstm", "config": config.as_dict(),
                "schema_version": config.schema_version, "sequence_dir": str(seq_root / identifier),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "entity_ids": preprocessor["entity_ids"], "entity_kinds": preprocessor["entity_kinds"],
                "source_ids": [w["world_id"] for w in manifest["worlds"]],
                "source_hashes": {w["world_id"]: w["sha256"] for w in manifest["worlds"]},
                "training_scope": "development_multiworld_explicit_split",
                "partition_counts": {part: writers[part].count for part in PARTS},
                "label_counts": {part: dict(writers[part].labels) for part in PARTS},
                "episode_counts": {part: {label: len(ids) for label, ids in writers[part].episodes.items()}
                                   for part in PARTS},
                "fault_type_inventory": coverage["fault_type_inventory"],
                "evidence_origins": ["controlled_injection", "historical_review"],
                "graph_version": graph["graph_version"], "exploratory": False,
                "preparation_runtime": preparation_runtime,
                "storage_runtime": {part: writers[part].stats() for part in PARTS},
                "preparation_verification": verified, "exclusions": [],
                "world_temporal_bounds": {w["world_id"]: {k: [a.isoformat(), b.isoformat()]
                    for k, (a, b) in world_bounds(w).items()} for w in manifest["worlds"]}}
        temporary.replace(seq_root / identifier)
        if max_targets is None:
            from app.services.diagnostico import DiagnosticWorkbench, _write
            workbench = DiagnosticWorkbench(output / "review")
            _write(workbench._path("datasets", identifier), result)
            save_json(output / "prepared.json", result)
        else:
            save_json(output / "probe.json", result)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
