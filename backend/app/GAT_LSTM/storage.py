"""Persistencia incremental y acceso bajo demanda a ventanas numéricas."""
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
import json
import operator
from pathlib import Path
import shutil

import numpy as np

from . import LABELS


FIELDS = {"X": np.float32, "state_ids": np.int64, "node_present": np.bool_,
          "y": np.int64, "label_mask": np.bool_}
PREDICTIVE_FIELDS = {key: value for key, value in FIELDS.items()
                     if key not in {"y", "label_mask"}}
PREDICTIVE_FIELDS.update(next_target=np.float32, next_observed=np.bool_)


class PartitionWriter:
    """Escribe una ventana cada vez, sin retener sus listas de Python."""

    def __init__(self, directory, prefix, *, predictive=False):
        self.directory, self.prefix = Path(directory), prefix
        self.fields = PREDICTIVE_FIELDS if predictive else FIELDS
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

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        values = np.fromfile(self.path, dtype=self.dtype, count=self.row_items,
                             offset=self.offset + index * self.row_bytes)
        if values.size != self.row_items:
            raise ValueError(f"Array NPY truncado: {self.path.name}")
        return values.reshape(self.shape[1:])


def sample_summaries(samples):
    """Accede a etiquetas y metadatos sin leer las ventanas de características."""
    if isinstance(samples, DiskPartition):
        return samples.summaries()
    return iter(samples)


class DiskPartition(Sequence):
    """Conserva metadatos; las ventanas se leen del archivo sin mapas persistentes."""

    def __init__(self, directory, prefix, *, predictive=False):
        directory = Path(directory)
        fields = PREDICTIVE_FIELDS if predictive else FIELDS
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
            raise ValueError("Los arrays y metadatos de la partición no coinciden")
        self.adjacency = json.loads((directory / "graph.json").read_text(encoding="utf-8"))["adjacency"]
        self.predictive = predictive

    def __len__(self):
        return len(self.metadata)

    def close(self):
        self.arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def summaries(self):
        for index, metadata in enumerate(self.metadata):
            yield {"metadata": metadata,
                   "y": ([-1] * len(self.adjacency) if self.predictive
                         else self.arrays["y"][index].tolist()),
                   "label_mask": ([False] * len(self.adjacency) if self.predictive
                                  else self.arrays["label_mask"][index].tolist())}

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample = {name: array[index] for name, array in self.arrays.items()}
        # Scalar labels remain ordinary Python values for metrics and JSON manifests.
        for name in ("y", "label_mask"):
            if name in sample:
                sample[name] = sample[name].tolist()
        if self.predictive:
            sample.update(y=[-1] * len(self.adjacency), label_mask=[False] * len(self.adjacency))
        return {**sample, "metadata": self.metadata[index], "adjacency": self.adjacency}
