"""Paquetes portables verificados; importar nunca activa ni reemplaza artefactos."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

import numpy as np

from app.config import MODELS_DIR, REVIEW_DATA_DIR, artifact_file
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.model_constants import KIND_TO_ID
from app.modulos.clasificador.segments import FEATURE_NAMES, MAX_SEGMENTS, SCALES, _validate_segment_steps
from app.modulos.clasificador.storage import FIELDS


FORMAT = "artifact-gat-lstm-transfer-v1"
MANIFEST = "transfer_manifest.json"
COPY_BYTES = 8 * 1024 ** 2
MAX_JSON_BYTES = 16 * 1024 ** 2
MAX_FILE_BYTES = 128 * 1024 ** 3
MAX_TOTAL_BYTES = 256 * 1024 ** 3
MAX_FILES = 10000
PARTITIONS = ("train_fit", "dev_tune", "validation", "test", "reserve")


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Identificador GAT-LSTM inválido")
    return value


def _directory(review_dir):
    review = Path(review_dir) if review_dir is not None else REVIEW_DATA_DIR
    return MODELS_DIR if review.resolve() == REVIEW_DATA_DIR.resolve() else review / "diagnostics"


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_json(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON demasiado grande: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Se requiere un objeto JSON: {path.name}")
    return value


def _digest(path):
    return sha256_file(path)


def _config(value):
    try:
        return Config(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Configuración incompatible; prepara nuevamente el dataset con SegmentGATLSTM: {exc}") from exc


def _safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Ruta inválida en el paquete")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or any(part in {".", ".."} for part in path.parts):
        raise ValueError("Ruta inválida en el paquete")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(any(char in part for char in ':*?"<>|') or any(ord(char) < 32 for char in part)
           or part.endswith((".", " ")) or part.split(".")[0].upper() in reserved
           for part in path.parts):
        raise ValueError("Ruta inválida en el paquete")
    return path


def _allowed_name(name, dataset_id, model_id):
    path = _safe_name(name)
    if name == f"datasets/{dataset_id}.json":
        return
    if len(path.parts) >= 3 and path.parts[:2] == ("sequences", dataset_id):
        return
    if model_id and name in {f"{folder}/{model_id}{suffix}" for folder, suffix in
                            (("models", ".json"), ("weights", ".pt"), ("graphs", ".json"), ("preprocessors", ".json"))}:
        return
    raise ValueError(f"Artefacto no permitido en el paquete: {name}")


def _contract(preprocessor, graph, config):
    entities = preprocessor.get("entity_ids")
    if (not isinstance(entities, list) or not entities or len(entities) > config.max_nodes
            or any(not isinstance(entity, str) for entity in entities) or len(set(entities)) != len(entities)
            or set(preprocessor.get("entities", {})) != set(entities)):
        raise ValueError("Entidades incompatibles en el preprocesador")
    if (preprocessor.get("schema_version") != config.schema_version
            or preprocessor.get("architecture_version", config.architecture_version) != config.architecture_version
            or preprocessor.get("feature_names") != FEATURE_NAMES
            or any(preprocessor["entities"][entity].get("kind") not in KIND_TO_ID for entity in entities)):
        raise ValueError("Esquema o características incompatibles en el preprocesador")
    if (graph.get("entity_ids") != entities or graph.get("schema_version") != config.schema_version
            or graph.get("method") != "within_entity_segment_dtw"
            or graph.get("scales") != [list(scale[:2]) for scale in SCALES]
            or graph.get("local_timezone") != config.local_timezone):
        raise ValueError("Contrato del grafo incompatible")
    adjacency = graph.get("adjacency")
    if (not isinstance(adjacency, list) or len(adjacency) != len(entities)
            or any(not isinstance(row, list) or len(row) != len(entities)
                   or any(type(value) is not bool for value in row) for row in adjacency)):
        raise ValueError("Dimensiones del grafo incompatibles")
    if adjacency != [[left == right for right in range(len(entities))] for left in range(len(entities))]:
        raise ValueError("El grafo de segmentos requiere entidades independientes")
    lengths = graph.get("segment_steps")
    if not isinstance(lengths, list) or len(lengths) != len(SCALES):
        raise ValueError("Escalas del grafo incompatibles")
    for (horizon, resolution, _), steps in zip(SCALES, lengths):
        _validate_segment_steps(horizon, resolution, steps)
    if graph.get("threshold") != config.segment_threshold:
        raise ValueError("Umbral del grafo incompatible")
    signature = sha256(json.dumps({key: value for key, value in graph.items() if key != "graph_version"},
                                 sort_keys=True).encode()).hexdigest()
    if graph.get("graph_version") != signature:
        raise ValueError("Firma graph_version incompatible")
    return entities


def _validate_dataset(dataset, sequence_dir, dataset_id):
    if not isinstance(dataset, dict) or dataset.get("dataset_id") != dataset_id or dataset.get("method") != "gat_lstm":
        raise ValueError("Manifiesto de dataset incompatible")
    config = _config(dataset["config"])
    if dataset.get("schema_version") != config.schema_version:
        raise ValueError("Esquema de dataset incompatible")
    if _config(_read_json(sequence_dir / "config.json")) != config:
        raise ValueError("La configuración del dataset y las secuencias no coincide")
    preprocessor = _read_json(sequence_dir / "preprocessor.json")
    graph = _read_json(sequence_dir / "graph.json")
    entities = _contract(preprocessor, graph, config)
    if dataset.get("entity_ids") != entities or dataset.get("graph_version") != graph["graph_version"]:
        raise ValueError("Entidades o firma del dataset incompatibles")
    expected_steps, nodes = 3 * MAX_SEGMENTS, len(entities)
    for partition in PARTITIONS:
        count = dataset.get("partition_counts", {}).get(partition, 0)
        if type(count) is not int or count < 0:
            raise ValueError("Recuento de partición inválido")
        metadata = sequence_dir / f"{partition}_metadata.jsonl"
        if count == 0:
            if metadata.exists() or any((sequence_dir / f"{partition}_{field}.npy").exists() for field in FIELDS):
                raise ValueError(f"Partición vacía con archivos: {partition}")
            continue
        shapes = {"X": (count, expected_steps, nodes, len(FEATURE_NAMES)),
                  "state_ids": (count, expected_steps, nodes), "node_present": (count, expected_steps, nodes),
                  "y": (count, nodes), "label_mask": (count, nodes)}
        for field, dtype in FIELDS.items():
            path = sequence_dir / f"{partition}_{field}.npy"
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if values.shape != shapes[field] or values.dtype != np.dtype(dtype) or not values.flags.c_contiguous:
                    raise ValueError(f"Forma o dtype incompatible: {path.name}")
                if values.offset + values.nbytes != path.stat().st_size:
                    raise ValueError(f"Tamaño NPY incompatible: {path.name}")
            finally:
                values._mmap.close()
        observed = 0
        with metadata.open("rb") as reader:
            while line := reader.readline(MAX_JSON_BYTES + 1):
                if len(line) > MAX_JSON_BYTES:
                    raise ValueError("Línea de metadatos demasiado grande")
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("entity_ids") != entities:
                    raise ValueError(f"Entidades de ventana incompatibles: {partition}")
                observed += 1
        if observed != count:
            raise ValueError(f"Recuento de ventanas incompatible: {partition}")
    if not all(dataset.get("partition_counts", {}).get(partition, 0) for partition in ("train_fit", "dev_tune")):
        raise ValueError("El dataset requiere train_fit y dev_tune")
    return preprocessor


def _validate_model(directory, model_id, dataset, sequence_dir=None):
    manifest = _read_json(directory / "models" / f"{model_id}.json")
    config = _config(manifest["config"])
    if (manifest.get("model_id") != model_id or manifest.get("method") != "gat_lstm"
            or manifest.get("schema_version") != config.schema_version
            or manifest.get("architecture_version") != config.architecture_version
            or manifest.get("dataset_id") != dataset["dataset_id"]
            or manifest.get("weights_file") != f"{model_id}.pt"
            or manifest.get("source_hashes") != dataset.get("source_hashes")):
        raise ValueError("Manifiesto de modelo incompatible con el dataset")
    built = _config(dataset["config"])
    for key in ("grid_seconds", "sequence_steps", "feature_window_seconds", "graph_top_k", "max_nodes",
                "use_entropy", "use_dtw", "architecture_version", "local_timezone", "entity_value_modes", "active_scales"):
        if getattr(config, key) != getattr(built, key):
            raise ValueError(f"Configuración de modelo incompatible: {key}")
    prep = _read_json(directory / "preprocessors" / f"{model_id}.json")
    graph = _read_json(directory / "graphs" / f"{model_id}.json")
    entities = _contract(prep, graph, config)
    sequence_dir = sequence_dir if sequence_dir is not None else directory / "sequences" / dataset["dataset_id"]
    if prep != _read_json(sequence_dir / "preprocessor.json"):
        raise ValueError("El preprocesador del modelo no pertenece al dataset")
    if manifest.get("graph_version") != graph["graph_version"]:
        raise ValueError("Firma del modelo y del grafo incompatible")
    policy = manifest.get("rejection_policy", {})
    if any(type(policy.get(key)) not in (int, float) or not math.isfinite(policy[key])
           or not 0 <= policy[key] <= 1 for key in ("threshold", "margin")):
        raise ValueError("Política de rechazo inválida")
    weights = directory / "weights" / f"{model_id}.pt"
    if _digest(weights) != manifest.get("weights_sha256"):
        raise ValueError("Hash de pesos incompatible")
    import torch
    from app.modulos.clasificador.network_segments import SegmentGATLSTM
    # mmap avoids retaining a second full copy of the serialized weights in RAM.
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    model = SegmentGATLSTM(len(FEATURE_NAMES), [prep["entities"][key]["kind"] for key in entities], config)
    model.load_state_dict(state, strict=True)
    expected_kinds = torch.tensor([KIND_TO_ID[prep["entities"][key]["kind"]] for key in entities])
    if not torch.equal(model.kind_ids, expected_kinds):
        raise ValueError("Tipos de entidades incompatibles con los pesos")


def export_bundle(dataset_id, output, *, model_id=None, review_dir=None):
    """Exporta secuencias y, opcionalmente, un candidato; no incluye active.json."""
    dataset_id = _identifier(dataset_id)
    if model_id is not None:
        model_id = _identifier(model_id)
    directory, output = _directory(review_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"El paquete ya existe: {output}")
    dataset = _read_json(artifact_file(directory, f"datasets/{dataset_id}.json"))
    sequence_dir = artifact_file(directory, f"sequences/{dataset_id}")
    if not sequence_dir.is_dir():
        sequence_dir = Path(dataset["sequence_dir"])
    _regular_target(sequence_dir)
    _validate_dataset(dataset, sequence_dir, dataset_id)
    portable = deepcopy(dataset)
    portable["sequence_dir"] = f"sequences/{dataset_id}"
    payloads = {f"datasets/{dataset_id}.json": _json_bytes(portable)}
    for path in sorted(sequence_dir.rglob("*")):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("No se admiten enlaces simbólicos en secuencias")
        if path.is_file():
            payloads[f"sequences/{dataset_id}/{path.relative_to(sequence_dir).as_posix()}"] = path
    if model_id:
        for folder, suffix in (("models", ".json"), ("weights", ".pt"), ("graphs", ".json"), ("preprocessors", ".json")):
            relative = f"{folder}/{model_id}{suffix}"
            path = artifact_file(directory, relative)
            _regular_target(path)
            payloads[relative] = path
        _validate_model(directory, model_id, dataset, sequence_dir)
    inventory = []
    seen = set()
    for name, value in payloads.items():
        _allowed_name(name, dataset_id, model_id)
        if name.casefold() in seen:
            raise ValueError("Rutas duplicadas en el paquete")
        seen.add(name.casefold())
        if isinstance(value, Path) and value.is_symlink():
            raise ValueError("No se admiten enlaces simbólicos")
        size = value.stat().st_size if isinstance(value, Path) else len(value)
        if size > MAX_FILE_BYTES:
            raise ValueError("Archivo demasiado grande para transferir")
        inventory.append({"path": name, "size": size, "sha256": _digest(value) if isinstance(value, Path) else sha256(value).hexdigest()})
    if len(inventory) > MAX_FILES or sum(row["size"] for row in inventory) > MAX_TOTAL_BYTES:
        raise ValueError("Paquete demasiado grande")
    manifest = {"format": FORMAT, "dataset_id": dataset_id, "model_id": model_id, "files": inventory}
    manifest_data = _json_bytes(manifest)
    if len(manifest_data) > MAX_JSON_BYTES:
        raise ValueError("El manifiesto de transferencia es demasiado grande")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".gat-transfer-", suffix=".tmp", dir=output.parent, delete=False) as temp:
        temporary_output = Path(temp.name)
    try:
        indexed = {item["path"]: item for item in inventory}
        with tarfile.open(temporary_output, "w:gz") as archive:
            for name, value in [(MANIFEST, manifest_data), *payloads.items()]:
                source_context = value.open("rb") if isinstance(value, Path) else io.BytesIO(value)
                with source_context as source:
                    size = value.stat().st_size if isinstance(value, Path) else len(value)
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = size, 0o600
                    digest = sha256()
                    class Reader:
                        def read(self, count):
                            block = source.read(count)
                            digest.update(block)
                            return block
                    archive.addfile(info, Reader())
                if name != MANIFEST:
                    expected = indexed[name]
                    if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                        raise ValueError(f"El archivo cambió durante la exportación: {name}")
        # Atomic publication without replacing another process's completed output.
        _publish(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    return {"archive": str(output.resolve()), "dataset_id": dataset_id, "model_id": model_id,
            "files": len(inventory), "sha256": _digest(output)}


def _unpack(archive_path, stage):
    with tarfile.open(archive_path, "r|*") as archive:
        members = iter(archive)
        first = next(members, None)
        if first is None or first.name != MANIFEST or not first.isfile() or not 0 < first.size <= MAX_JSON_BYTES:
            raise ValueError("Falta el manifiesto portable al inicio del paquete")
        manifest = json.loads(archive.extractfile(first).read(MAX_JSON_BYTES + 1))
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise ValueError("Formato de paquete incompatible")
        dataset_id = _identifier(manifest.get("dataset_id"))
        model_id = manifest.get("model_id")
        if model_id is not None:
            _identifier(model_id)
        inventory = manifest.get("files")
        if not isinstance(inventory, list) or not 1 <= len(inventory) <= MAX_FILES:
            raise ValueError("Inventario de paquete inválido")
        expected, folded, total = {}, set(), 0
        for item in inventory:
            if not isinstance(item, dict):
                raise ValueError("Entrada de inventario inválida")
            name, size, digest = item.get("path"), item.get("size"), item.get("sha256")
            _allowed_name(name, dataset_id, model_id)
            if name.casefold() in folded or type(size) is not int or not 0 <= size <= MAX_FILE_BYTES or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError("Inventario duplicado o inválido")
            expected[name], total = item, total + size
            folded.add(name.casefold())
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Paquete demasiado grande")
        seen = set()
        for member in members:
            name = member.name
            _allowed_name(name, dataset_id, model_id)
            if not member.isfile() or name not in expected or name in seen or member.size != expected[name]["size"]:
                raise ValueError(f"Entrada TAR inválida: {name}")
            seen.add(name)
            path = stage.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            digest, copied = sha256(), 0
            with archive.extractfile(member) as source, path.open("xb") as destination:
                while block := source.read(COPY_BYTES):
                    copied += len(block)
                    destination.write(block)
                    digest.update(block)
            if copied != member.size or digest.hexdigest() != expected[name]["sha256"]:
                raise ValueError(f"Archivo corrupto: {name}")
        if seen != set(expected):
            raise ValueError("Faltan archivos del inventario")
    return manifest


def _regular_target(path):
    if any(parent.is_symlink() or getattr(parent, "is_junction", lambda: False)()
           for parent in (path, *path.parents)):
        raise ValueError(f"El destino contiene un enlace simbólico: {path}")


def _publish(source, target):
    """Publica un archivo completo de forma exclusiva en el mismo filesystem."""
    if os.name == "nt":
        # Windows rename refuses an existing target, including on FAT/exFAT.
        os.rename(source, target)
    else:
        os.link(source, target)


def import_bundle(archive, *, review_dir=None):
    """Valida todo en staging, rechaza conflictos y registra manifiestos al final."""
    directory = _directory(review_dir)
    with tempfile.TemporaryDirectory(prefix="gat-transfer-import-") as temporary:
        stage = Path(temporary)
        manifest = _unpack(Path(archive), stage)
        dataset_id, model_id = manifest["dataset_id"], manifest["model_id"]
        dataset_path = stage / "datasets" / f"{dataset_id}.json"
        dataset = _read_json(dataset_path)
        if dataset.get("sequence_dir") != f"sequences/{dataset_id}":
            raise ValueError("La ruta del dataset debe ser portable")
        _validate_dataset(dataset, stage / "sequences" / dataset_id, dataset_id)
        if model_id:
            _validate_model(stage, model_id, dataset)
        dataset["sequence_dir"] = str(artifact_file(directory, f"sequences/{dataset_id}").resolve())
        dataset_path.write_bytes(_json_bytes(dataset))
        planned, existing = [], []
        for item in manifest["files"]:
            relative = item["path"]
            source, target = stage.joinpath(*PurePosixPath(relative).parts), artifact_file(directory, relative)
            _regular_target(target)
            if target.exists():
                if not target.is_file():
                    raise FileExistsError(f"Artefacto en conflicto: {target}")
                if relative == f"datasets/{dataset_id}.json":
                    local = _read_json(target)
                    local["sequence_dir"] = dataset["sequence_dir"]
                    identical = local == dataset
                else:
                    identical = target.stat().st_size == source.stat().st_size and _digest(target) == _digest(source)
                if not identical:
                    raise FileExistsError(f"Artefacto en conflicto: {target}")
                existing.append(relative)
            else:
                planned.append((relative, source, target))
        sequence_target = artifact_file(directory, f"sequences/{dataset_id}")
        if sequence_target.exists():
            actual = {f"sequences/{dataset_id}/{path.relative_to(sequence_target).as_posix()}"
                      for path in sequence_target.rglob("*") if path.is_file()}
            expected = {row["path"] for row in manifest["files"] if row["path"].startswith(f"sequences/{dataset_id}/")}
            if actual - expected:
                raise FileExistsError("El directorio de secuencias contiene artefactos en conflicto")
        # Publish registration manifests last. Roll back only newly created files on failure.
        planned.sort(key=lambda row: (row[0].startswith("datasets/"), row[0].startswith("models/"), row[0]))
        created = []
        try:
            for _, source, target in planned:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(prefix=".gat-import-", dir=target.parent, delete=False) as destination:
                    staged_target = Path(destination.name)
                    try:
                        with source.open("rb") as reader:
                            shutil.copyfileobj(reader, destination, length=COPY_BYTES)
                    except BaseException:
                        destination.close()
                        staged_target.unlink(missing_ok=True)
                        raise
                try:
                    _publish(staged_target, target)
                    created.append(target)
                finally:
                    staged_target.unlink(missing_ok=True)
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
    return {"dataset_id": dataset_id, "model_id": model_id, "imported_files": len(planned),
            "existing_files": len(existing), "activated": False}
