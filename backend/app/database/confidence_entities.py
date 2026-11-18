"""Persistent catalog and current states for live and test confidence sensors."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from app.flow.definitions import _json, _now, _time
from app.flow.confidence_change import explain_confidence_change


VERSION = "confidence-entities-v1"
SOURCE_ID = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")
CATALOG_SCHEMA = """
    CREATE TABLE IF NOT EXISTS confidence_catalog (
        source_entity_id TEXT NOT NULL, source_mode TEXT NOT NULL, session_id TEXT,
        revision INTEGER NOT NULL, payload TEXT NOT NULL,
        PRIMARY KEY (source_entity_id, source_mode)
    )
"""


def confidence_entity_id(source, *, test=False):
    if not isinstance(source, str) or not SOURCE_ID.fullmatch(source) or source.startswith("sensor.confianza_"):
        raise ValueError("La entidad de origen no es un sensor compatible para publicar confianza.")
    prefix = "test_" if test else ""
    return f"confianza.{prefix}{source.replace('.', '_')}"


class ConfidenceEntityStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(CATALOG_SCHEMA)
            columns = {column[1] for column in db.execute("PRAGMA table_info(confidence_catalog)")}
            if "source_mode" not in columns:
                db.execute("ALTER TABLE confidence_catalog RENAME TO confidence_catalog_live_v1")
                db.execute(CATALOG_SCHEMA)
                db.execute("""
                    INSERT INTO confidence_catalog
                    SELECT source_entity_id,'live',session_id,revision,payload FROM confidence_catalog_live_v1
                """)
                db.execute("DROP TABLE confidence_catalog_live_v1")
            db.execute("""
                CREATE TABLE IF NOT EXISTS confidence_clock (
                    id INTEGER PRIMARY KEY CHECK(id=1),store_id TEXT NOT NULL,revision INTEGER NOT NULL
                )
            """)
            db.execute("INSERT OR IGNORE INTO confidence_clock VALUES (1,?,0)", (uuid4().hex,))
            self.store_id = db.execute("SELECT store_id FROM confidence_clock WHERE id=1").fetchone()[0]
            # Evaluation and reception sessions end when the backend restarts.
            for payload, in db.execute("SELECT payload FROM confidence_catalog WHERE session_id IS NOT NULL").fetchall():
                item = json.loads(payload)
                if item.get("available"):
                    item.update(available=False, activity="stopped")
                    self._write(db, None, item)

    @contextmanager
    def _database(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _write(self, db, session_id, item):
        db.execute("UPDATE confidence_clock SET revision=revision+1 WHERE id=1")
        revision = db.execute("SELECT revision FROM confidence_clock WHERE id=1").fetchone()[0]
        item.update(store_id=self.store_id, revision=revision)
        db.execute("INSERT OR REPLACE INTO confidence_catalog VALUES (?,?,?,?,?)",
                   (item["source_entity_id"], item.get("source_mode", "live"), session_id, revision, _json(item)))
        return {key: value for key, value in item.items() if key not in ("_scores", "_facts")}

    def start(self, session_id, flow_id, sources, *, test=False):
        sources = sorted(set(sources))
        identifiers = {}
        for source in sources:
            try:
                identifiers[source] = confidence_entity_id(source, test=test)
            except ValueError:
                if not test:
                    raise
                # CSVs can also contain IDs that have no Home Assistant equivalent.
        changed = []
        with self._database() as db:
            for source, identifier in identifiers.items():
                item = {"entity_id": identifier, "source_entity_id": source, "source_mode": "test" if test else "live",
                        "flow_id": flow_id, "session_id": session_id, "available": True,
                        "activity": "waiting", "confidence": None, "value": None,
                        "confidence_kind": "operational_index", "confidence_calibrated": False,
                        "evaluated_at": None, "source_timestamp": None, "observation_id": None,
                        "last_changed": _now()}
                changed.append(self._write(db, session_id, item))
        return changed

    def record(self, session_id, readings):
        changed = []
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            for reading in readings:
                source = reading.get("entity_id")
                row = db.execute("SELECT payload FROM confidence_catalog WHERE source_entity_id=? AND session_id=?",
                                 (source, session_id)).fetchone()
                if row is None:
                    continue
                previous = json.loads(row[0])
                if (previous.get("observation_id") == reading.get("observation_id")
                        and previous.get("execution_id") == reading.get("execution_id")
                        and previous.get("evaluated_at") is not None):
                    continue
                confidence = reading.get("confidence")
                if confidence is not None and (type(confidence) not in (float, int)
                                               or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                    raise ValueError("La confianza debe estar entre 0 y 1 o ser nula.")
                now = _now()
                value = round(confidence * 100, 6) if confidence is not None else None
                fusion = reading.get("confidence_fusion") or {}
                item = {**previous, "value": value, "confidence": confidence,
                        "activity": "receiving", "available": True, "evaluated_at": now,
                        "source_timestamp": _time(reading.get("timestamp")), "source_value": reading.get("value"),
                        "source_state": reading.get("state"), "observation_id": reading.get("observation_id"),
                        "execution_id": reading.get("execution_id"), "config_hash": reading.get("config_hash"),
                        "evidence_coverage": reading.get("evidence_coverage"), "method": fusion.get("method"),
                        "active_filters": fusion.get("active_filters"), "available_filters": fusion.get("available_filters"),
                        "reason": fusion.get("reason"), "confidence_kind": reading.get("confidence_kind", "operational_index"),
                        "confidence_calibrated": reading.get("confidence_calibrated") is True,
                        "last_changed": now if value != previous.get("value") else previous["last_changed"]}
                explain_confidence_change(previous, item, reading)
                changed.append(self._write(db, session_id, item))
        return changed

    def stop(self, session_id):
        changed = []
        with self._database() as db:
            for payload, in db.execute("SELECT payload FROM confidence_catalog WHERE session_id=?", (session_id,)).fetchall():
                item = json.loads(payload)
                item.update(available=False, activity="stopped", last_changed=_now())
                changed.append(self._write(db, None, item))
        return changed

    def snapshot(self):
        with self._database() as db:
            db.execute("BEGIN")
            revision = db.execute("SELECT revision FROM confidence_clock WHERE id=1").fetchone()[0]
            items = [json.loads(row[0]) for row in db.execute("SELECT payload FROM confidence_catalog ORDER BY source_entity_id,source_mode")]
            for item in items:
                item.pop("_scores", None)
                item.pop("_facts", None)
        return {"schema_version": VERSION, "store_id": self.store_id, "revision": revision, "entities": items}
