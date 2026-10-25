"""Índice temporal compartible y caché acotada de intervalos causales."""
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from app.modulos.clasificador.config import parse_utc

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
DTYPE = np.dtype([("time", "<i8"), ("row", "<i8"), ("state", "<i4"), ("origin", "<i4")])


def microseconds(at):
    delta = at - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


class IndexedRows:
    def __init__(self, array, states, origins, cache_size=2048):
        self.array, self.states, self.origins = array, states, origins
        self.times = array["time"]
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.signature = None

    def bounds(self, start, end, *, interval=False):
        left = bisect_right(self.times, microseconds(start)) if interval else bisect_left(self.times, microseconds(start))
        return max(0, left - 1), bisect_right(self.times, microseconds(end))

    def parsed(self, start, end):
        left, right = self.bounds(start, end, interval=True)
        return [(EPOCH + timedelta(microseconds=int(row["time"])), int(row["row"]),
                 self.states[int(row["state"])]) for row in self.array[left:right]]

    def intervals(self, start, end, resolution, info):
        from app.modulos.clasificador.segments import _intervals_from_parsed
        signature = (info["kind"], info.get("median"), info.get("scale"), tuple(info.get("vocab", {}).items()))
        if signature != self.signature:
            self.cache.clear()
            self.signature = signature
        count = int((end - start).total_seconds() // resolution)
        keys = [(resolution, start + timedelta(seconds=(step + 1) * resolution)) for step in range(count)]
        result = [None] * count
        step = 0
        while step < count:
            if keys[step] in self.cache:
                result[step] = self.cache[keys[step]]
                self.cache.move_to_end(keys[step])
                step += 1
                continue
            first = step
            while step < count and keys[step] not in self.cache:
                step += 1
            a = start + timedelta(seconds=first * resolution)
            b = start + timedelta(seconds=step * resolution)
            ticks = _intervals_from_parsed(self.parsed(a, b), a, b, resolution, info)
            for offset, tick in enumerate(ticks, first):
                result[offset] = tick
                self.cache[keys[offset]] = tick
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result


class IndexedEventProvider:
    def __init__(self, entities):
        self.entities = entities

    @classmethod
    def from_events(cls, index):
        entities = {}
        for entity, rows in index.items():
            rows = sorted(rows, key=lambda row: (row.get("_time") or parse_utc(row["timestamp"]), int(row.get("source_row") or 0)))
            states = list(dict.fromkeys(str(row["state"]) for row in rows))
            origins = list(dict.fromkeys(str(row.get("provenance", "")) for row in rows))
            state_ids, origin_ids = {v: i for i, v in enumerate(states)}, {v: i for i, v in enumerate(origins)}
            array = np.empty(len(rows), dtype=DTYPE)
            for i, row in enumerate(rows):
                array[i] = (microseconds(row.get("_time") or parse_utc(row["timestamp"])),
                            int(row.get("source_row") or 0), state_ids[str(row["state"])],
                            origin_ids[str(row.get("provenance", ""))])
            entities[entity] = IndexedRows(array, states, origins)
        return cls(entities)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for number, (entity, rows) in enumerate(self.entities.items()):
            name = f"{number}.npy"
            np.save(directory / name, rows.array, allow_pickle=False)
            manifest[entity] = {"file": name, "states": rows.states, "origins": rows.origins}
        (directory / "index.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        return cls({entity: IndexedRows(np.load(directory / row["file"], mmap_mode="r", allow_pickle=False),
                                       row["states"], row["origins"]) for entity, row in manifest.items()})

    def provenance(self, start, end, entities):
        result = set()
        for entity in entities:
            rows = self.entities.get(entity)
            if rows is not None:
                left, right = rows.bounds(start, end)
                result.update(rows.origins[int(i)] for i in np.unique(rows.array["origin"][left:right]))
        return sorted(result)

    def __call__(self, start, end, entities):
        result = []
        for entity in entities:
            rows = self.entities.get(entity)
            if rows is not None:
                left, right = rows.bounds(start, end)
                for row in rows.array[left:right]:
                    result.append({"entity_id": entity, "timestamp": (EPOCH + timedelta(microseconds=int(row["time"]))).isoformat(),
                                   "source_row": int(row["row"]), "state": rows.states[int(row["state"])],
                                   "provenance": rows.origins[int(row["origin"])]})
        return result

    def close(self):
        for rows in self.entities.values():
            mapping = getattr(rows.array, "_mmap", None)
            if mapping is not None:
                mapping.close()
