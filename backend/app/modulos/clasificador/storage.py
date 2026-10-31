"""Persistencia incremental y acceso bajo demanda a ventanas numéricas."""
from app.config import setting
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from hashlib import sha256
import json
import operator
import os
from pathlib import Path
import queue
import shutil
import threading
import time
from uuid import uuid4

import numpy as np

from app.modulos.clasificador import LABELS


FIELDS = {"X": np.float32, "state_ids": np.int64, "node_present": np.bool_,
          "y": np.int64, "label_mask": np.bool_}



class PartitionWriter:
    """Escribe una ventana cada vez, sin retener sus listas de Python."""

    def __init__(self, directory, prefix):
        self.directory, self.prefix = Path(directory), prefix
        self.fields = FIELDS
        self.count = 0
        self.labels = Counter()
        self.episodes = defaultdict(set)
        self.shapes = {}
        self.streams = {}
        self.stack = ExitStack()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stack.close()

    def append(self, sample):
        if not self.streams:
            for field in self.fields:
                self.streams[field] = self.stack.enter_context(
                    (self.directory / f"{self.prefix}_{field}.raw").open("wb"))
            self.metadata = self.stack.enter_context(
                (self.directory / f"{self.prefix}_metadata.jsonl").open("w", encoding="utf-8"))
        for field, dtype in self.fields.items():
            values = np.asarray(sample[field], dtype=dtype)
            if values.shape != self.shapes.setdefault(field, values.shape):
                raise ValueError(f"Forma inconsistente en {field}")
            self.streams[field].write(values.tobytes(order="C"))
        self.metadata.write(json.dumps(sample["metadata"], ensure_ascii=False) + "\n")
        if "y" in self.fields:
            group = sample["metadata"].get("parent_episode_id") or sample["metadata"]["sample_id"]
            for label, active in zip(sample["y"], sample["label_mask"]):
                if active:
                    self.labels[LABELS[label]] += 1
                    self.episodes[LABELS[label]].add(group)
        self.count += 1

    def finish(self):
        self.stack.close()
        if not self.count:
            return
        for field, dtype in self.fields.items():
            raw = self.directory / f"{self.prefix}_{field}.raw"
            with (self.directory / f"{self.prefix}_{field}.npy").open("wb") as output:
                np.lib.format.write_array_header_1_0(output, {
                    "descr": np.dtype(dtype).str, "fortran_order": False,
                    "shape": (self.count, *self.shapes[field])})
                with raw.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            raw.unlink()


class FileArray:
    """Lee filas NPY sin mantener sus páginas mapeadas en el proceso."""

    def __init__(self, path):
        self.path = Path(path)
        # Open only to inspect the header, then release the mapping immediately.
        mapped = np.load(self.path, mmap_mode="r", allow_pickle=False)
        try:
            if not mapped.flags.c_contiguous:
                raise ValueError("La partición requiere arrays NPY en orden C")
            self.shape, self.dtype, self.offset = mapped.shape, mapped.dtype, mapped.offset
        finally:
            mapped._mmap.close()
        self.row_items = int(np.prod(self.shape[1:]))
        self.row_bytes = self.row_items * self.dtype.itemsize
        self._file = self.path.open("rb")
        self._lock = threading.Lock()

    def read_rows(self, indices):
        """Reads contiguous row runs into one batch, retaining requested order."""
        indices = [operator.index(index) for index in indices]
        for index in indices:
            if index < 0 or index >= len(self):
                raise IndexError(index)
        output = np.empty((len(indices), *self.shape[1:]), dtype=self.dtype)
        positions = defaultdict(list)
        for position, index in enumerate(indices):
            positions[index].append(position)
        ordered = sorted(positions)
        runs = []
        for index in ordered:
            if runs and index == runs[-1][-1] + 1:
                runs[-1].append(index)
            else:
                runs.append([index])
        for run in runs:
            count = len(run)
            offset = self.offset + run[0] * self.row_bytes
            with self._lock:
                if hasattr(os, "pread"):
                    raw = os.pread(self._file.fileno(), count * self.row_bytes, offset)
                else:
                    self._file.seek(offset)
                    raw = self._file.read(count * self.row_bytes)
            if len(raw) != count * self.row_bytes:
                raise ValueError(f"Array NPY truncado: {self.path.name}")
            values = np.frombuffer(raw, dtype=self.dtype).reshape((count, *self.shape[1:]))
            for relative, index in enumerate(run):
                for position in positions[index]:
                    output[position] = values[relative]
        return output

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.read_rows([index])[0]

    def close(self):
        if not self._file.closed:
            self._file.close()


def sample_summaries(samples):
    """Accede a etiquetas y metadatos sin leer las ventanas de características."""
    if hasattr(samples, "summaries"):
        return samples.summaries()
    return iter(samples)


def iter_partition_batches(partition, groups, *, batch_bytes, max_buffer_bytes=256 * 1024 ** 2):
    """Reads batches on one thread with at most two bounded lookahead buffers."""
    # Consumer + queued batches + producer output and one temporary read buffer.
    depth = min(2, max(0, max_buffer_bytes // max(1, batch_bytes) - 3))
    if depth == 0:
        for indices in groups:
            started = time.monotonic()
            yield partition.read_batch(indices), time.monotonic() - started
        return
    channel = queue.Queue(maxsize=depth)
    stopped = threading.Event()
    sentinel = object()

    def put(value):
        while not stopped.is_set():
            try:
                channel.put(value, timeout=.1)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            for indices in groups:
                if stopped.is_set() or not put(partition.read_batch(indices)):
                    return
            put(sentinel)
        except BaseException as exc:
            put(exc)

    worker = threading.Thread(target=produce, name="gat-sequence-reader", daemon=True)
    worker.start()
    try:
        while True:
            started = time.monotonic()
            value = channel.get()
            waited = time.monotonic() - started
            if value is sentinel:
                break
            if isinstance(value, BaseException):
                raise value
            yield value, waited
    finally:
        stopped.set()
        worker.join()


class DiskPartition(Sequence):
    """Conserva metadatos; las ventanas se leen del archivo sin mapas persistentes."""

    def __init__(self, directory, prefix):
        directory = Path(directory)
        fields = FIELDS
        # Labels are small (N per window); keeping them avoids reopening files
        # for every episode count. Feature windows (T*N*F) remain on disk.
        self.arrays = {
            name: (np.load(directory / f"{prefix}_{name}.npy", allow_pickle=False)
                   if name in {"y", "label_mask"}
                   else FileArray(directory / f"{prefix}_{name}.npy"))
            for name in fields}
        with (directory / f"{prefix}_metadata.jsonl").open(encoding="utf-8") as stream:
            self.metadata = [json.loads(line) for line in stream]
        if any(len(array) != len(self.metadata) for array in self.arrays.values()):
            self.close()
            raise ValueError("Los arrays y metadatos de la partición no coinciden")
        self.adjacency = json.loads((directory / "graph.json").read_text(encoding="utf-8"))["adjacency"]

    def __len__(self):
        return len(self.metadata)

    def close(self):
        for array in self.arrays.values():
            if isinstance(array, FileArray):
                array.close()
        self.arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def summaries(self):
        for index, metadata in enumerate(self.metadata):
            yield {"metadata": metadata,
                   "y": self.arrays["y"][index].tolist(),
                   "label_mask": self.arrays["label_mask"][index].tolist()}

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._legacy_sample(index)

    def read_batch(self, indices):
        """Returns arrays directly for training, without a dict per sample."""
        indices = [operator.index(index) for index in indices]
        if any(index < 0 or index >= len(self) for index in indices):
            raise IndexError("Índice fuera de la partición")
        batch = {name: (array.read_rows(indices) if isinstance(array, FileArray)
                        else array[indices]) for name, array in self.arrays.items()}
        batch["metadata"] = [self.metadata[index] for index in indices]
        batch["adjacency"] = self.adjacency
        return batch

    def batch_nbytes(self):
        return sum(int(np.prod(array.shape[1:])) * array.dtype.itemsize
                   for array in self.arrays.values())

    def read_stats(self):
        return {"source": getattr(self, "source", "original"),
                "cache_key": getattr(self, "cache_key", None),
                "cache_warning": getattr(self, "cache_warning", None)}

    def _legacy_sample(self, index):
        batch = self.read_batch([index])
        sample = {name: values[0] for name, values in batch.items()
                  if name not in {"metadata", "adjacency"}}
        # Scalar labels remain ordinary Python values for metrics and JSON manifests.
        for name in ("y", "label_mask"):
            if name in sample:
                sample[name] = sample[name].tolist()
        return {**sample, "metadata": batch["metadata"][0], "adjacency": self.adjacency}


_CACHE_LOCK = threading.RLock()
_CACHE_IN_USE = set()
_CACHE_LIMIT = 12 * 1024 ** 3
_COPY_CHUNK = 8 * 1024 ** 2


def _tree_files(directory):
    return sorted((path for path in Path(directory).rglob("*") if path.is_file()),
                  key=lambda path: path.relative_to(directory).as_posix())


def _cache_identity(dataset, directory):
    files = _tree_files(directory)
    inventory = [{"name": path.relative_to(directory).as_posix(),
                  "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                 for path in files]
    payload = {"version": 1, "dataset_id": dataset.get("dataset_id"),
               "config": dataset.get("config"), "source_hashes": dataset.get("source_hashes"),
               "files": inventory}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest(), files, inventory


def _directory_size(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())


def _cache_valid(path, key, inventory):
    marker = path / "cache_manifest.json"
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        if manifest.get("key") != key or manifest.get("inventory") != inventory:
            return False
        expected = {item["name"]: item["size"] for item in inventory}
        actual = {item.relative_to(path).as_posix(): item.stat().st_size
                  for item in path.rglob("*") if item.is_file() and item.name != "cache_manifest.json"}
        if expected != actual:
            return False
        for item in path.rglob("*.npy"):
            array = np.load(item, mmap_mode="r", allow_pickle=False)
            try:
                if not array.flags.c_contiguous:
                    return False
                shape = array.shape
            finally:
                array._mmap.close()
            if not shape or shape[0] <= 0:
                return False
        graph = json.loads((path / "graph.json").read_text(encoding="utf-8"))
        node_count = len(graph["adjacency"])
        if not node_count or any(len(row) != node_count for row in graph["adjacency"]):
            return False
        for metadata in path.glob("*_metadata.jsonl"):
            count = sum(1 for _ in metadata.open(encoding="utf-8"))
            prefix = metadata.name.removesuffix("_metadata.jsonl")
            shapes = {}
            for field in ("X", "state_ids", "node_present", "y", "label_mask",
                          "next_target", "next_observed"):
                array_path = path / f"{prefix}_{field}.npy"
                if array_path.exists():
                    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
                    try:
                        if array.shape[0] != count:
                            return False
                        shapes[field] = array.shape
                    finally:
                        array._mmap.close()
            if "X" not in shapes or len(shapes["X"]) != 4:
                return False
            _, steps, nodes, _ = shapes["X"]
            if nodes != node_count or steps <= 0:
                return False
            for field in ("state_ids", "node_present"):
                if shapes.get(field) != (count, steps, nodes):
                    return False
            for field in ("y", "label_mask", "next_target", "next_observed"):
                if field in shapes and shapes[field] != (count, nodes):
                    return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _evict_for_space(root, needed, protected):
    root.mkdir(parents=True, exist_ok=True)
    entries = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".tmp-")]
    temporary = sum(_directory_size(path) for path in root.iterdir()
                    if path.is_dir() and path.name.startswith(".tmp-"))
    used = sum(_directory_size(path) for path in entries) + temporary
    for path in sorted(entries, key=lambda item: item.stat().st_atime):
        if used + needed <= _CACHE_LIMIT:
            break
        if path == protected or path in _CACHE_IN_USE:
            continue
        size = _directory_size(path)
        shutil.rmtree(path)
        used -= size
    if used + needed > _CACHE_LIMIT:
        raise OSError("El almacenamiento de secuencias superaría el límite de 12 GiB")


def _copy_to_cache(source, target, key, inventory, progress):
    source = Path(source)
    total = sum(item["size"] for item in inventory)
    copied = 0
    hashes = {}
    for item in inventory:
        relative = Path(item["name"])
        src, dst = source / relative, target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256()
        with src.open("rb") as reader, dst.open("wb") as writer:
            while True:
                chunk = reader.read(_COPY_CHUNK)
                if not chunk:
                    break
                writer.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                if progress:
                    progress(copied / max(1, total))
        if dst.stat().st_size != item["size"]:
            raise OSError(f"Copia incompleta: {relative.name}")
        hashes[item["name"]] = digest.hexdigest()
    (target / "cache_manifest.json").write_text(json.dumps(
        {"key": key, "inventory": inventory, "sha256": hashes},
        sort_keys=True), encoding="utf-8")


@contextmanager
def open_training_partitions(dataset, progress=None):
    """Prepares/reuses the bounded cache, then keeps descriptors open for training."""
    original = Path(dataset["sequence_dir"])
    cache_root_value = setting('GAT_SEQUENCE_CACHE_DIR')
    cache_warning = None
    selected = original
    key = None
    if cache_root_value:
        root = Path(cache_root_value)
        try:
            key, files, inventory = _cache_identity(dataset, original)
            target = root / key
            with _CACHE_LOCK:
                root.mkdir(parents=True, exist_ok=True)
                if _cache_valid(target, key, inventory):
                    os.utime(target, None)
                    selected = target
                else:
                    if target.exists() and target not in _CACHE_IN_USE:
                        shutil.rmtree(target)
                    needed = sum(item["size"] for item in inventory) + 1024 * 1024
                    _evict_for_space(root, needed, target)
                    temporary = root / f".tmp-{key}-{uuid4().hex}"
                    temporary.mkdir()
                    try:
                        _copy_to_cache(original, temporary, key, inventory,
                                       lambda ratio: progress(ratio) if progress else None)
                        if not _cache_valid(temporary, key, inventory):
                            raise OSError("La copia de secuencias no superó la validación")
                        os.replace(temporary, target)
                        selected = target
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                _CACHE_IN_USE.add(selected)
        except Exception as exc:
            cache_warning = f"No se pudo preparar la caché; se leerá el dataset original: {exc}"
            selected, key = original, None
            if progress:
                progress(None)
    elif progress:
        progress(None)

    prefixes = ("train_fit", "dev_tune")
    partitions = []
    try:
        for prefix in prefixes:
            metadata_path = selected / f"{prefix}_metadata.jsonl"
            if not metadata_path.exists():
                partitions.append([])
                continue
            partition = DiskPartition(selected, prefix)
            partition.source = "cache" if selected != original else "original"
            partition.cache_key = key
            partition.cache_warning = cache_warning
            partitions.append(partition)
        yield partitions[0], partitions[1], {"source": "cache" if selected != original else "original",
             "cache_key": key, "cache_warning": cache_warning}
    finally:
        for partition in partitions:
            if isinstance(partition, DiskPartition):
                partition.close()
        with _CACHE_LOCK:
            _CACHE_IN_USE.discard(selected)
