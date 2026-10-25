"""Ventanas únicas en HDF5 comprimido; etiquetas y orden independientes."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import threading

import numpy as np

from app.modulos.clasificador import LABELS

FORMAT = "gat-hdf5-shared-v1"
SHARED = {"X": np.float32, "state_ids": np.int64, "node_present": np.bool_}
LABELS_DTYPES = {"y": np.int64, "label_mask": np.bool_}


def hdf5():
    try:
        import h5py
    except ImportError as exc:
        raise ValueError("El almacenamiento comprimido requiere h5py; instala requirements-gat*.txt") from exc
    return h5py


class StorageBudget:
    """Límite conjunto de particiones y reserva física, también en el host WSL."""

    def __init__(self, directory, max_bytes=None, reserve_bytes=None):
        self.directory = Path(directory)
        self.max_bytes = int(max_bytes if max_bytes is not None else
                             os.environ.get("GAT_STORAGE_MAX_BYTES", 4 * 1024**3))
        self.reserve_bytes = int(reserve_bytes if reserve_bytes is not None else
                                 os.environ.get("GAT_STORAGE_RESERVE_BYTES", 5 * 1024**3))
        if self.max_bytes <= 0 or self.reserve_bytes < 0:
            raise ValueError("Presupuesto de almacenamiento inválido")

    def used(self):
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def check(self, additional):
        if self.used() + additional > self.max_bytes:
            raise OSError("La preparación alcanzaría el límite de almacenamiento; no se continúa escribiendo")
        locations = [self.directory]
        # The free space advertised by ext4 is virtual. On this Windows/WSL
        # installation the VHD lives on C:, so also honor the physical host.
        if os.name != "nt" and Path("/mnt/c").is_mount():
            locations.append(Path("/mnt/c"))
        if min(shutil.disk_usage(path).free for path in locations) < self.reserve_bytes + additional:
            raise OSError("Espacio físico insuficiente: se conserva la reserva libre de almacenamiento")


class CompressedPartitionWriter:
    """Escribe bloques limitados en RAM directamente comprimidos, sin archivo raw."""

    def __init__(self, directory, prefix, *, budget=None, buffer_rows=32):
        self.directory, self.prefix = Path(directory), prefix
        self.budget = budget or StorageBudget(directory)
        self.buffer_rows = buffer_rows
        if buffer_rows < 1:
            raise ValueError("El bloque debe contener al menos una muestra")
        self.count = self.unique_count = self.raw_bytes = 0
        self.labels, self.episodes = Counter(), defaultdict(set)
        self.shapes, self.seen, self.pending = {}, {}, []
        self.unique_pending = []
        self.stack, self.file = ExitStack(), None
        self.finished = False

    def __enter__(self):
        hdf5()  # Fail before any expensive preparation if dependency is absent.
        return self

    def __exit__(self, *args):
        self.stack.close()

    def _open(self, values):
        self.budget.check(1024**2)
        path = self.directory / f"{self.prefix}.h5"
        metadata = self.directory / f"{self.prefix}_metadata.jsonl.gz"
        self.file = self.stack.enter_context(hdf5().File(path, "x"))
        self.file.attrs.update(format=FORMAT, complete=False)
        self.metadata = self.stack.enter_context(gzip.open(metadata, "xt", encoding="utf-8", compresslevel=1))
        for name, dtype in {**SHARED, **LABELS_DTYPES}.items():
            shape = values[name].shape
            row_bytes = max(1, values[name].nbytes)
            rows = max(1, min(self.buffer_rows, 512 * 1024 // row_bytes))
            self.file.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape),
                dtype=dtype, chunks=(rows, *shape), compression="gzip", compression_opts=1, shuffle=True, fletcher32=True)
        self.file.create_dataset("window_index", shape=(0,), maxshape=(None,), dtype="i8",
                                 chunks=(self.buffer_rows,), compression="gzip", compression_opts=1, fletcher32=True)

    def append(self, sample):
        if self.finished:
            raise ValueError("La partición ya está finalizada")
        values = {}
        for name, dtype in {**SHARED, **LABELS_DTYPES}.items():
            array = np.ascontiguousarray(sample[name], dtype=dtype)
            if array.shape != self.shapes.setdefault(name, array.shape):
                raise ValueError(f"Forma inconsistente en {name}")
            values[name] = array
        if self.file is None:
            self._open(values)
        stamp = sample["metadata"]["target_time"]
        digest = sha256()
        for name in SHARED:
            digest.update(values[name].tobytes())
        fingerprint = digest.digest()
        if stamp in self.seen:
            window, previous = self.seen[stamp]
            if previous != fingerprint:
                raise ValueError("El mismo instante contiene ventanas distintas; no se pueden compartir")
        else:
            window = self.unique_count
            self.seen[stamp] = (window, fingerprint)
            self.unique_pending.append({name: values[name].copy() for name in SHARED})
            self.unique_count += 1
        metadata = json.dumps(sample["metadata"], ensure_ascii=False) + "\n"
        self.pending.append((window, {name: values[name].copy() for name in LABELS_DTYPES}, metadata))
        group = sample["metadata"].get("parent_episode_id") or sample["metadata"]["sample_id"]
        for label, active in zip(values["y"], values["label_mask"]):
            if active:
                self.labels[LABELS[int(label)]] += 1
                self.episodes[LABELS[int(label)]].add(group)
        self.count += 1
        self.raw_bytes += sum(array.nbytes for array in values.values())
        if len(self.pending) >= self.buffer_rows:
            self._flush()

    def _flush(self):
        if not self.pending:
            return
        # Conservative bound for one block, including HDF5 allocation overhead.
        additional = (sum(sum(v.nbytes for v in row.values()) for row in self.unique_pending)
                      + sum(sum(v.nbytes for v in row.values()) + len(meta.encode("utf-8"))
                            for _, row, meta in self.pending) + 1024**2)
        self.budget.check(additional)
        for fields, rows in ((SHARED, self.unique_pending), (LABELS_DTYPES, [row for _, row, _ in self.pending])):
            if not rows:
                continue
            for name in fields:
                dataset = self.file[name]
                start = len(dataset)
                dataset.resize(start + len(rows), axis=0)
                dataset[start:] = np.stack([row[name] for row in rows])
        indices = self.file["window_index"]
        start = len(indices)
        indices.resize(start + len(self.pending), axis=0)
        indices[start:] = [index for index, _, _ in self.pending]
        for _, _, meta in self.pending:
            self.metadata.write(meta)
        self.file.flush()
        self.metadata.flush()
        self.unique_pending.clear()
        self.pending.clear()

    def finish(self):
        if self.finished:
            return
        self._flush()
        if self.file is not None:
            self.file.attrs.update(complete=True, samples=self.count, unique_windows=self.unique_count,
                                   uncompressed_sample_bytes=self.raw_bytes)
            self.file.flush()
        self.stack.close()
        self.finished = True

    def stats(self):
        paths = [self.directory / f"{self.prefix}.h5",
                 self.directory / f"{self.prefix}_metadata.jsonl.gz"]
        return {"format": FORMAT, "compression": "gzip-1+shuffle", "samples": self.count,
                "unique_windows": self.unique_count, "uncompressed_array_bytes": self.raw_bytes,
                "stored_bytes": sum(p.stat().st_size for p in paths if p.exists())}


class SharedArray:
    """Vista lógica por etiqueta sobre ventanas físicas únicas; lectura por lote."""

    def __init__(self, dataset, indices, lock):
        self.dataset, self.indices, self.lock = dataset, indices, lock
        self.shape = (len(indices), *dataset.shape[1:])
        self.dtype = dataset.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return self.read_rows([index])[0]

    def read_rows(self, indices):
        if not indices:
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)
        physical, inverse = np.unique(self.indices[indices], return_inverse=True)
        with self.lock:
            rows = self.dataset[physical]
        return rows[inverse]


def open_compressed(directory, prefix):
    file = hdf5().File(Path(directory) / f"{prefix}.h5", "r")
    try:
        if file.attrs.get("format") != FORMAT or not file.attrs.get("complete"):
            raise ValueError("Partición comprimida incompleta o incompatible")
        indices = file["window_index"][:]
        if indices.size and (indices.min() < 0 or indices.max() >= len(file["X"])):
            raise ValueError("Índice de ventanas inválido")
        if any(len(file[name]) != len(file["X"]) for name in SHARED):
            raise ValueError("Las ventanas compartidas no coinciden")
        x_shape = file["X"].shape
        if len(x_shape) != 4 or any(file[name].shape != x_shape[:3] for name in ("state_ids", "node_present")):
            raise ValueError("Dimensiones de ventanas comprimidas incompatibles")
        if any(file[name].shape != (len(indices), x_shape[2]) for name in LABELS_DTYPES):
            raise ValueError("Dimensiones de etiquetas incompatibles")
        lock = threading.Lock()
        arrays = {name: SharedArray(file[name], indices, lock) for name in SHARED}
        arrays.update({name: file[name][:] for name in LABELS_DTYPES})
        with gzip.open(Path(directory) / f"{prefix}_metadata.jsonl.gz", "rt", encoding="utf-8") as stream:
            metadata = [json.loads(line) for line in stream]
        if any(len(array) != len(metadata) for array in arrays.values()):
            raise ValueError("Los arrays y metadatos de la partición no coinciden")
        return file, arrays, metadata
    except BaseException:
        file.close()
        raise
