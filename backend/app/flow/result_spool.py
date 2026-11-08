"""Immutable result archives: one observation/evidence pool, many node views.

The scratch database is unpublished and disposable until finish + fsync. Only a
sealed archive may be published by FlowStore's metadata transaction. RAM is
bounded by the current timestamp group, rather than the complete evaluation.
"""
from collections import Counter, OrderedDict
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from time import perf_counter
import zlib

from app.flow.definitions import _json

FORMAT = "flow-flyweight-v1"
ALERT_DECISIONS = {"fault_compatible", "environmental", "suspicion"}
OBSERVATION_FIELDS = {
    "observation_id", "entity_id", "timestamp", "sensor_kind", "state", "value",
    "source_raw_index", "is_valid_value",
}


def _same(left, right):
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def encode_observation(records):
    """Factor common fields and evidence without mutating the live graph."""
    base = {key: value for key, value in records[0]["payload"].items() if key in OBSERVATION_FIELDS}
    evidence, buckets, nodes = [], {}, []
    for record in records:
        item = record["payload"]
        fields = {key: value for key, value in item.items() if key not in base or not _same(value, base[key])}
        node = {"node": record["node_id"], "sequence": record["sequence"], "fields": fields}
        missing = [key for key in base if key not in item]
        if missing:
            node["missing"] = missing
        entries = item.get("evidence")
        if isinstance(entries, list) and all(isinstance(entry, dict) for entry in entries):
            fields.pop("evidence", None)
            references = []
            for entry in entries:
                # An ID identifies an immutable assessment. Retain variants too:
                # legacy adapters can add module metadata to an earlier envelope.
                identifier = entry.get("evidence_id") or _json(entry)
                candidates = buckets.setdefault(identifier, [])
                index = next((index for index in candidates if _same(evidence[index], entry)), None)
                if index is None:
                    index = len(evidence)
                    candidates.append(index)
                    evidence.append(entry)
                references.append(index)
            node["evidence"] = references
        nodes.append(node)
    raw = _json({"base": base, "evidence": evidence, "nodes": nodes}).encode("utf-8")
    return zlib.compress(raw, level=1), len(raw), len(evidence)


def decode_observation(blob):
    return json.loads(zlib.decompress(blob))


def materialize(body, slot, *, owned=False):
    node = body["nodes"][slot]
    item = dict(body["base"]) if owned else deepcopy(body["base"])
    for key in node.get("missing", []):
        item.pop(key, None)
    item.update(node["fields"] if owned else deepcopy(node["fields"]))
    if "evidence" in node:
        seen = set()
        entries = []
        for index in node["evidence"]:
            entries.append(body["evidence"][index] if owned and index not in seen else deepcopy(body["evidence"][index]))
            seen.add(index)
        item["evidence"] = entries
    return item


def archive_path(database, filename):
    directory = Path(database).with_suffix(".results").resolve()
    if not filename or "/" in filename or "\\" in filename:
        raise ValueError("Nombre de archivo de resultados inválido")
    target = (directory / filename).resolve()
    if target.parent != directory:
        raise ValueError("El archivo de resultados está fuera de su directorio")
    return target


def sync_directory(directory):
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def archive_for_run(db, database, run_id):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='flow_run_archives'").fetchone():
        return None
    row = db.execute("SELECT filename,format_version FROM flow_run_archives WHERE run_id=?", (run_id,)).fetchone()
    if row and row[1] != FORMAT:
        raise ValueError("Versión de archivo de resultados no compatible")
    return archive_path(database, row[0]) if row else None


def iter_run_results(db, database, run_id, node_id):
    archive = archive_for_run(db, database, run_id)
    if archive is not None:
        yield from ResultArchive(archive).records(node_id)
    else:
        for identity, sequence, payload in db.execute(
                "SELECT observation_id,sequence,payload FROM flow_results WHERE run_id=? AND node_id=? ORDER BY sequence", (run_id, node_id)):
            yield {"node_id": node_id, "observation_id": identity, "sequence": sequence, "payload": json.loads(payload)}


class ResultArchive:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"No se encuentra el archivo de resultados: {self.path.name}")

    def connect(self):
        # Published files are sealed: no WAL/shared-memory files or writer locks.
        return sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)

    def page(self, node_id, offset, limit):
        db = self.connect()
        try:
            counts = json.loads(db.execute("SELECT value FROM metadata WHERE key='node_counts'").fetchone()[0])
            # Apply the page boundary to the small reference index first. Joining
            # before OFFSET would seek every earlier observation on deep pages.
            rows = db.execute("""WITH page AS MATERIALIZED (
                    SELECT observation,slot,sequence FROM results
                    WHERE node_id=? ORDER BY sequence LIMIT ? OFFSET ?
                ) SELECT observation.payload,page.slot FROM page
                JOIN observations AS observation ON observation.id=page.observation
                ORDER BY page.sequence""", (node_id, limit, offset))
            return counts.get(node_id, 0), [materialize(decode_observation(blob), slot, owned=True) for blob, slot in rows]
        finally:
            db.close()

    def diagnostic_snapshots(self, default=None):
        db = self.connect()
        try:
            row = db.execute("SELECT value FROM metadata WHERE key='diagnostic_snapshots'").fetchone()
            return json.loads(zlib.decompress(row[0])) if row else default
        finally:
            db.close()

    def records(self, node_id=None, *, alerts_only=False):
        db = self.connect()
        cache = OrderedDict()
        try:
            conditions, args = [], []
            if node_id is not None:
                conditions.append("node_id=?")
                args.append(node_id)
            if alerts_only:
                conditions.append("alert=1")
            clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            order = "sequence" if node_id is not None else "ordinal"
            rows = db.execute("SELECT node_id,sequence,observation,slot FROM results" + clause + " ORDER BY " + order, args)
            for node, sequence, observation, slot in rows:
                if observation not in cache:
                    blob = db.execute("SELECT payload FROM observations WHERE id=?", (observation,)).fetchone()[0]
                    cache[observation] = decode_observation(blob)
                    if len(cache) > 256:
                        cache.popitem(last=False)
                cache.move_to_end(observation)
                item = materialize(cache[observation], slot)
                yield {"node_id": node, "sequence": sequence,
                       "observation_id": str(item["observation_id"]), "payload": item}
        finally:
            db.close()


class SpoolRecords:
    """Iterator compatibility plus a sealed-file fast path for FlowStore."""
    def __init__(self, spool):
        self.spool = spool
        self.on_progress = None

    def __iter__(self):
        yield from ResultArchive(self.spool.path).records()

    def progress(self, stage, detail):
        if self.on_progress:
            self.on_progress(stage, detail)


class FlowResultSpool:
    def __init__(self, job_id, directory=None):
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f"flow-evaluation-{job_id}-", suffix=".sqlite3", dir=directory)
        os.close(descriptor)
        self.path = Path(name)
        self.closed = self.published = False
        self.failed = False
        self.has_audit_snapshot = False
        self.sequences = Counter()
        self.stats = {"observations": 0, "records": 0, "unique_evidence": 0,
                      "evidence_references": 0, "uncompressed_bytes": 0, "spooling_ms": 0., "finalize_ms": 0.}
        self.unflushed_inputs = 0
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        # Only this disposable scratch file uses relaxed durability. It is never
        # exposed as a completed run before finish verifies and fsyncs the file.
        self.db.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE observations (id INTEGER PRIMARY KEY,identity TEXT NOT NULL UNIQUE,payload BLOB NOT NULL);
            CREATE TABLE results (node_id TEXT NOT NULL,sequence INTEGER NOT NULL,observation INTEGER NOT NULL,
                                  slot INTEGER NOT NULL,ordinal INTEGER NOT NULL,alert INTEGER NOT NULL,
                                  PRIMARY KEY(node_id,sequence)) WITHOUT ROWID;
            CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        """)

    def write_block(self, results):
        if self.closed:
            raise ValueError("El archivo de resultados ya está cerrado")
        try:
            self._write_block(results)
        except BaseException:
            self.failed = True
            raise

    def _write_block(self, results):
        started = perf_counter()
        grouped = {}
        ordinal = self.stats["records"]
        for node, items in results.items():
            for item in items:
                identity = str(item["observation_id"])
                grouped.setdefault(identity, []).append({"node_id": node, "sequence": self.sequences[node],
                                                         "payload": item, "ordinal": ordinal})
                self.sequences[node] += 1
                ordinal += 1
        for identity, records in grouped.items():
            if len({record["node_id"] for record in records}) != len(records):
                raise sqlite3.IntegrityError("Resultado duplicado para la misma observación y nodo")
            blob, raw_bytes, unique = encode_observation(records)
            observation = self.db.execute("INSERT INTO observations(identity,payload) VALUES (?,?)", (identity, blob)).lastrowid
            self.db.executemany("INSERT INTO results VALUES (?,?,?,?,?,?)",
                               [(record["node_id"], record["sequence"], observation, slot, record["ordinal"],
                                 int(record["node_id"] == "output" and record["payload"].get("final_decision") in ALERT_DECISIONS))
                                for slot, record in enumerate(records)])
            self.stats["unique_evidence"] += unique
            self.stats["uncompressed_bytes"] += raw_bytes
            self.stats["evidence_references"] += sum(len(record["payload"].get("evidence", []))
                                                     for record in records if isinstance(record["payload"].get("evidence"), list))
        self.stats["observations"] += len(grouped)
        self.stats["records"] = ordinal
        self.unflushed_inputs += len(grouped)
        if self.unflushed_inputs >= 500:
            self.db.commit()
            self.unflushed_inputs = 0
        self.stats["spooling_ms"] += (perf_counter() - started) * 1000

    def finish(self, snapshot=None):
        if self.closed:
            return
        if self.failed:
            raise ValueError("La escritura de resultados falló; el archivo no se puede publicar")
        started = perf_counter()
        if snapshot is not None and "diagnostic_snapshots" in snapshot:
            raw = _json(snapshot["diagnostic_snapshots"]).encode("utf-8")
            blob = zlib.compress(raw, 1)
            self.db.execute("INSERT INTO metadata VALUES ('diagnostic_snapshots',?)", (blob,))
            self.has_audit_snapshot = True
            self.stats["audit_uncompressed_bytes"] = len(raw)
            self.stats["audit_compressed_bytes"] = len(blob)
        self.db.execute("INSERT INTO metadata VALUES ('format',?)", (FORMAT,))
        self.db.execute("INSERT INTO metadata VALUES ('node_counts',?)", (_json(dict(self.sequences)),))
        self.db.commit()
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("El archivo de resultados no superó la comprobación de integridad")
        self.db.close()
        with self.path.open("r+b") as stream:
            os.fsync(stream.fileno())
        self.closed = True
        self.stats["archive_bytes"] = self.path.stat().st_size
        self.stats["finalize_ms"] = (perf_counter() - started) * 1000

    def records(self):
        self.finish()
        return SpoolRecords(self)

    def stage_publication(self, destination):
        self.finish()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        moved = True
        try:
            os.replace(self.path, destination)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            moved = False
            try:
                with self.path.open("rb") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
        sync_directory(destination.parent)
        return moved

    def cleanup(self):
        if not self.closed:
            self.db.close()
            self.closed = True
        if not self.published:
            self.path.unlink(missing_ok=True)
