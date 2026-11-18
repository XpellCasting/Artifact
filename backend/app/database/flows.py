from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from threading import Lock
import unicodedata
from time import perf_counter
from typing import Any
from uuid import uuid4
import zlib
from app.flow.definitions import _json, _now, validate_flow
from app.flow.measurements import MEASUREMENTS_VERSION, RunMeasurements
from app.flow.metrics import METRICS_VERSION, reconstruct_metrics, run_timings, saved_support, unavailable_metrics
from app.flow.result_spool import FORMAT, ALERT_DECISIONS, SpoolRecords, ResultArchive, archive_path, archive_for_run, iter_run_results

ALERT_SEVERITIES = {"fault": ("fault_compatible",), "anomaly": ("environmental", "suspicion")}
METRICS_CACHE_SIZE = 32


def _flow_name(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "Flujo sin nombre")).split()) or "Flujo sin nombre"


def _flow_name_key(value):
    return _flow_name(value).casefold()


class FlowStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._metrics_cache = OrderedDict()
        self._metrics_lock = Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as db:
            # Keep status, saved runs and alerts readable while a large spool is saved.
            db.execute("PRAGMA journal_mode=WAL")
            import_alerts = not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='flow_alerts'").fetchone()
            db.executescript("""
                CREATE TABLE IF NOT EXISTS flows (
                    flow_id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS flow_runs (
                    run_id TEXT PRIMARY KEY, flow_id TEXT NOT NULL, snapshot TEXT NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS flow_run_display (
                    run_id TEXT PRIMARY KEY, flow_id TEXT NOT NULL, snapshot TEXT NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS flow_run_display_date ON flow_run_display(created_at DESC);
                CREATE INDEX IF NOT EXISTS flow_run_display_flow ON flow_run_display(flow_id,created_at DESC);
                CREATE TRIGGER IF NOT EXISTS flow_run_display_insert AFTER INSERT ON flow_runs BEGIN
                    INSERT INTO flow_run_display VALUES
                        (NEW.run_id,NEW.flow_id,json_remove(NEW.snapshot,'$.diagnostic_snapshots'),NEW.summary,NEW.created_at);
                END;
                CREATE TRIGGER IF NOT EXISTS flow_run_display_update AFTER UPDATE ON flow_runs BEGIN
                    DELETE FROM flow_run_display WHERE run_id=OLD.run_id;
                    INSERT INTO flow_run_display VALUES
                        (NEW.run_id,NEW.flow_id,json_remove(NEW.snapshot,'$.diagnostic_snapshots'),NEW.summary,NEW.created_at);
                END;
                CREATE TRIGGER IF NOT EXISTS flow_run_display_delete AFTER DELETE ON flow_runs BEGIN
                    DELETE FROM flow_run_display WHERE run_id=OLD.run_id;
                END;
                CREATE TABLE IF NOT EXISTS flow_results (
                    run_id TEXT NOT NULL, node_id TEXT NOT NULL, observation_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (run_id, node_id, observation_id)
                );
                CREATE INDEX IF NOT EXISTS flow_results_page ON flow_results(run_id, node_id, sequence);
                CREATE TABLE IF NOT EXISTS flow_run_archives (
                    run_id TEXT PRIMARY KEY,filename TEXT NOT NULL,format_version TEXT NOT NULL,
                    counts TEXT NOT NULL,timings TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS flow_run_archive_delete AFTER DELETE ON flow_runs BEGIN
                    DELETE FROM flow_run_archives WHERE run_id=OLD.run_id;
                END;
                CREATE TABLE IF NOT EXISTS flow_alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL, observation_id TEXT NOT NULL,
                    flow_id TEXT NOT NULL, run_id TEXT, created_at TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE (execution_id, observation_id)
                );
                CREATE INDEX IF NOT EXISTS flow_alerts_execution ON flow_alerts(execution_id,alert_id);
                CREATE TABLE IF NOT EXISTS retained_live_runs (
                    run_id TEXT PRIMARY KEY, execution_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS expired_live_archives (filename TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS flow_alert_state (
                    id INTEGER PRIMARY KEY CHECK (id=1), execution_id TEXT NOT NULL
                );
            """)
            # Backfill once for existing installations; full audit snapshots remain in flow_runs.
            db.execute("""INSERT INTO flow_run_display
                SELECT run_id,flow_id,json_remove(snapshot,'$.diagnostic_snapshots'),summary,created_at
                FROM flow_runs WHERE run_id NOT IN (SELECT run_id FROM flow_run_display)""")
            if import_alerts:
                db.execute("""INSERT OR IGNORE INTO flow_alerts
                    (execution_id,observation_id,flow_id,run_id,created_at,payload)
                    SELECT result.run_id,result.observation_id,run.flow_id,result.run_id,run.created_at,result.payload
                    FROM flow_results AS result JOIN flow_runs AS run ON run.run_id=result.run_id
                    WHERE result.node_id='output'
                      AND json_extract(result.payload,'$.final_decision') IN ('fault_compatible','environmental','suspicion')
                    ORDER BY run.created_at,result.sequence""")
            if not db.execute("SELECT 1 FROM flow_alert_state").fetchone():
                # Include saved runs without alerts and legacy unfinished executions.
                db.execute("""INSERT INTO flow_alert_state(id,execution_id)
                    SELECT 1,execution_id FROM (
                        SELECT COALESCE((SELECT execution_id FROM flow_alerts WHERE run_id=run.run_id LIMIT 1),run.run_id) AS execution_id,
                               run.created_at AS created_at,run.rowid AS sequence
                        FROM flow_runs AS run
                        UNION ALL
                        SELECT execution_id,MAX(created_at),MAX(alert_id) FROM flow_alerts
                        WHERE run_id IS NULL GROUP BY execution_id
                    ) ORDER BY created_at DESC,sequence DESC LIMIT 1""")

        self._remove_expired_archives()

    def _remove_expired_archives(self):
        with self._database() as db:
            for filename, in db.execute("SELECT filename FROM expired_live_archives").fetchall():
                try:
                    archive_path(self.path, filename).unlink(missing_ok=True)
                except OSError:
                    logging.getLogger(__name__).warning("No se pudo retirar el archivo caducado %s", filename)
                    continue
                db.execute("DELETE FROM expired_live_archives WHERE filename=?", (filename,))

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    @contextmanager
    def _database(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        checked = validate_flow(payload)
        if not checked["valid"]:
            raise ValueError("; ".join(checked["errors"]))
        flow_id = uuid4().hex
        item = {"flow_id": flow_id, "name": _flow_name(payload.get("name")), "nodes": payload["nodes"], "edges": payload["edges"]}
        with self._database() as db:
            # Serialize lookup and insert across devices, including separate API processes.
            db.execute("BEGIN IMMEDIATE")
            existing = self._find_name(db, item["name"])
            if existing:
                return {**existing, "loaded_existing": True}
            db.execute("INSERT INTO flows VALUES (?,?,?,?)", (flow_id, item["name"], _json(item), _now()))
        return item

    @staticmethod
    def _find_name(db, name):
        key = _flow_name_key(name)
        for flow_id, saved_name in db.execute("SELECT flow_id,name FROM flows ORDER BY updated_at DESC,rowid DESC"):
            if _flow_name_key(saved_name) == key:
                payload, updated = db.execute("SELECT payload,updated_at FROM flows WHERE flow_id=?", (flow_id,)).fetchone()
                return {**json.loads(payload), "updated_at": updated}
        return None

    def list(self) -> list[dict[str, Any]]:
        with self._database() as db:
            rows = db.execute("SELECT payload,updated_at FROM flows ORDER BY updated_at DESC,rowid DESC").fetchall()
        # Older duplicate IDs remain readable by saved runs and previously opened browsers.
        items, names = [], set()
        for payload, updated in rows:
            item = json.loads(payload)
            key = _flow_name_key(item.get("name"))
            if key not in names:
                names.add(key)
                items.append({**item, "updated_at": updated})
        return items

    def get(self, flow_id: str) -> dict[str, Any]:
        with self._database() as db:
            row = db.execute("SELECT payload,updated_at FROM flows WHERE flow_id=?", (flow_id,)).fetchone()
        if not row:
            raise KeyError(flow_id)
        return {**json.loads(row[0]), "updated_at": row[1]}

    def update(self, flow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.get(flow_id)
        checked = validate_flow(payload)
        if not checked["valid"]:
            raise ValueError("; ".join(checked["errors"]))
        item = {"flow_id": flow_id, "name": _flow_name(payload.get("name")), "nodes": payload["nodes"], "edges": payload["edges"]}
        updated_at = _now()
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM flows WHERE flow_id=?", (flow_id,)).fetchone():
                raise KeyError(flow_id)
            existing = self._find_name(db, item["name"])
            if existing and existing["flow_id"] != flow_id:
                return {**existing, "loaded_existing": True}
            db.execute("UPDATE flows SET name=?,payload=?,updated_at=? WHERE flow_id=?", (item["name"], _json(item), updated_at, flow_id))
        return {**item, "updated_at": updated_at}

    def save_run(self, flow_id: str, snapshot: dict[str, Any], summary: dict[str, Any], results: dict[str, list[dict[str, Any]]]) -> str:
        run_id = uuid4().hex
        with self._database() as db:
            self._activate_alert_execution(db, run_id)
            db.execute("INSERT INTO flow_runs VALUES (?,?,?,?,?)", (run_id, flow_id, _json(snapshot), _json(summary), _now()))
            records = []
            for node_id, items in results.items():
                for sequence, item in enumerate(items):
                    records.append((run_id, node_id, str(item["observation_id"]), sequence, _json(item)))
            db.executemany("INSERT INTO flow_results VALUES (?,?,?,?,?)", records)
            self._insert_alerts(db, run_id, flow_id, results.get("output", []), run_id)
        return run_id

    def save_run_stream(self, flow_id: str, snapshot: dict[str, Any], summary: dict[str, Any], records, execution_id: str | None = None) -> tuple[str, float]:
        """Persist a completed spool atomically without materializing its rows."""
        if isinstance(records, SpoolRecords):
            return self._save_archive(flow_id, snapshot, summary, records, execution_id)
        run_id = uuid4().hex
        started = perf_counter()
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            if execution_id is None:
                self._activate_alert_execution(db, run_id)
            else:
                db.execute("INSERT OR IGNORE INTO flow_alert_state VALUES (1,?)", (execution_id,))
            def values():
                for record in records:
                    if record["node_id"] == "output" and record["payload"].get("final_decision") in ALERT_DECISIONS:
                        self._insert_alerts(db, execution_id or run_id, flow_id, [record["payload"]], run_id)
                    yield (run_id, record["node_id"], record["observation_id"],
                           record["sequence"], _json(record["payload"]))
            db.executemany("INSERT INTO flow_results VALUES (?,?,?,?,?)", values())
            insert_ms = round((perf_counter() - started) * 1000, 3)
            performance = summary.setdefault("performance", {})
            phases = performance.setdefault("phases_ms", {})
            phases["result_persistence_before_commit"] = insert_ms
            phases["total_with_result_inserts"] = round(phases.get("total", 0) + insert_ms, 3)
            included = performance.setdefault("included_in", {})
            included["result_persistence_before_commit"] = "total_with_result_inserts"
            included["total_with_result_inserts"] = "evaluation and result insertion; excludes commit latency"
            db.execute("INSERT INTO flow_runs VALUES (?,?,?,?,?)",
                       (run_id, flow_id, _json(snapshot), _json(summary), _now()))
            if execution_id:
                db.execute("UPDATE flow_alerts SET run_id=? WHERE execution_id=?", (run_id, execution_id))
            db.commit()
            return run_id, round((perf_counter() - started) * 1000, 3)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _archive_performance(summary, counts, timings):
        performance = summary.setdefault("performance", {})
        phases = performance.setdefault("phases_ms", {})
        phases.update({key: round(value, 3) for key, value in timings.items()})
        phases["result_spooling"] = round(counts["spooling_ms"], 3)
        phases["result_spool_finalize"] = round(counts["finalize_ms"], 3)
        phases["total_with_result_inserts"] = round(phases.get("total", 0) + phases.get("result_persistence_before_commit", 0), 3)
        phases["total_with_persistence"] = round(phases["total_with_result_inserts"] + phases["result_spool_finalize"] + phases.get("result_commit", 0), 3)
        performance.setdefault("counts", {})["result_storage"] = {
            "format": FORMAT, **{key: value for key, value in counts.items() if not key.endswith("_ms")}}
        performance.setdefault("included_in", {}).update({
            "result_spooling": "causal_evaluation", "result_spool_finalize": "total_with_persistence",
            "result_archive_publication": "result_persistence_before_commit", "result_commit": "total_with_persistence",
            "result_persistence_before_commit": "total_with_result_inserts",
            "total_with_result_inserts": "evaluation and publication; excludes archive finalization and commit",
        })

    def _save_archive(self, flow_id, snapshot, summary, records, execution_id):
        started = perf_counter()
        spool = records.spool
        spool.finish()
        scratch_path = spool.path
        run_id = uuid4().hex
        destination = archive_path(self.path, run_id + ".sqlite3")
        db, committed = None, False
        timings = {}
        try:
            records.progress("publication", f"Guardando {spool.stats['records']:,} resultados de la ejecución")
            spool.stage_publication(destination)
            timings["result_archive_publication"] = (perf_counter() - started) * 1000
            db = self._connect()
            db.execute("BEGIN IMMEDIATE")
            if execution_id is None:
                self._activate_alert_execution(db, run_id)
                self._insert_alerts(db, run_id, flow_id,
                                    (record["payload"] for record in ResultArchive(destination).records("output", alerts_only=True)), run_id)
            else:
                db.execute("INSERT OR IGNORE INTO flow_alert_state VALUES (1,?)", (execution_id,))
                db.execute("UPDATE flow_alerts SET run_id=? WHERE execution_id=?", (run_id, execution_id))
            records.progress("commit", "Confirmando el archivo de resultados y las métricas")
            timings["result_persistence_before_commit"] = (perf_counter() - started) * 1000
            self._archive_performance(summary, spool.stats, timings)
            stored_snapshot = snapshot
            if spool.has_audit_snapshot:
                # Keep identities/configuration accessible to older SQL tools;
                # the full immutable training manifest lives once in the archive.
                snapshots = {}
                diagnostic = snapshot.get("diagnostic_snapshots")
                for node, value in (diagnostic if isinstance(diagnostic, dict) else {}).items():
                    value = value if isinstance(value, dict) else {}
                    model = value.get("model")
                    model = model if isinstance(model, dict) else {}
                    snapshots[node] = {"run_id": value.get("run_id"), "model": {
                        key: model[key] for key in ("model_id", "method", "config", "architecture_version",
                                                  "schema_version", "graph_version", "window_seconds", "training_end") if key in model}}
                stored_snapshot = {**snapshot, "diagnostic_snapshots": snapshots}
            db.execute("INSERT INTO flow_runs VALUES (?,?,?,?,?)", (run_id, flow_id, _json(stored_snapshot), _json(summary), _now()))
            db.execute("INSERT INTO flow_run_archives VALUES (?,?,?,?,?)",
                       (run_id, destination.name, FORMAT, _json(spool.stats), _json(timings)))
            if spool.observation_limit is not None and execution_id is not None:
                session_limit = summary["live_retention"]["sessions"]
                db.execute("INSERT INTO retained_live_runs VALUES (?,?)", (run_id, execution_id))
                expired = db.execute("""SELECT r.run_id,r.execution_id,a.filename
                    FROM retained_live_runs r JOIN flow_run_archives a ON a.run_id=r.run_id
                    ORDER BY r.rowid DESC LIMIT -1 OFFSET ?""", (session_limit,)).fetchall()
                for old_run, old_execution, filename in expired:
                    db.execute("INSERT INTO expired_live_archives VALUES (?)", (filename,))
                    db.execute("DELETE FROM flow_alerts WHERE execution_id=?", (old_execution,))
                    db.execute("DELETE FROM flow_runs WHERE run_id=?", (old_run,))
                    db.execute("DELETE FROM retained_live_runs WHERE run_id=?", (old_run,))
            commit_started = perf_counter()
            timings["result_persistence_before_commit"] = (commit_started - started) * 1000
            db.commit()
            timings["result_commit"] = (perf_counter() - commit_started) * 1000
            committed = True
            spool.path, spool.published = destination, True
            if scratch_path.exists():
                try:
                    scratch_path.unlink()
                except OSError:
                    logging.getLogger(__name__).warning("No se pudo retirar el archivo temporal %s", scratch_path.name, exc_info=True)
            self._archive_performance(summary, spool.stats, timings)
            # Record commit latency without rewriting the large audit snapshot.
            # A telemetry failure must not turn a committed evaluation into a failure.
            try:
                db.execute("UPDATE flow_run_archives SET timings=? WHERE run_id=?", (_json(timings), run_id))
                db.commit()
            except sqlite3.Error:
                db.rollback()
                logging.getLogger(__name__).warning("No se pudo guardar el tiempo de confirmación de %s", run_id, exc_info=True)
            if spool.observation_limit is not None:
                try:
                    self._remove_expired_archives()
                except sqlite3.Error:
                    logging.getLogger(__name__).warning("Limpieza de archivos caducados pendiente", exc_info=True)
            return run_id, round((perf_counter() - started) * 1000, 3)
        except BaseException:
            if not committed:
                if db is not None:
                    db.rollback()
                if destination.exists():
                    if spool.path.exists():
                        destination.unlink()
                    else:
                        os.replace(destination, spool.path)
            raise
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def _activate_alert_execution(db, execution_id):
        db.execute("INSERT INTO flow_alert_state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET execution_id=excluded.execution_id",
                   (execution_id,))

    def begin_alert_execution(self, execution_id: str) -> str | None:
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT execution_id FROM flow_alert_state WHERE id=1").fetchone()
            self._activate_alert_execution(db, execution_id)
            return previous[0] if previous is not None else None

    def discard_alert_execution(self, execution_id: str, previous: str | None) -> None:
        with self._database() as db:
            db.execute("DELETE FROM flow_alerts WHERE execution_id=? AND run_id IS NULL", (execution_id,))
            if previous is None:
                db.execute("DELETE FROM flow_alert_state WHERE id=1 AND execution_id=?", (execution_id,))
            else:
                db.execute("UPDATE flow_alert_state SET execution_id=? WHERE id=1 AND execution_id=?",
                           (previous, execution_id))

    @staticmethod
    def _insert_alerts(db, execution_id, flow_id, items, run_id=None):
        created_at = _now()
        db.executemany("""INSERT OR IGNORE INTO flow_alerts
            (execution_id,observation_id,flow_id,run_id,created_at,payload) VALUES (?,?,?,?,?,?)""",
            ((execution_id, str(item["observation_id"]), flow_id, run_id, created_at, _json(item))
             for item in items if item.get("final_decision") in ALERT_DECISIONS))

    def record_alerts(self, execution_id: str, flow_id: str, items: list[dict[str, Any]], *, expired=()) -> None:
        if not expired and not any(item.get("final_decision") in ALERT_DECISIONS for item in items):
            return
        with self._database() as db:
            db.execute("INSERT OR IGNORE INTO flow_alert_state VALUES (1,?)", (execution_id,))
            self._insert_alerts(db, execution_id, flow_id, items)
            if expired:
                db.executemany("DELETE FROM flow_alerts WHERE execution_id=? AND observation_id=?",
                               ((execution_id, identity) for identity in expired))

    def alerts(self, before_id: int | None = None, limit: int = 50, severity: str | None = None) -> dict[str, Any]:
        conditions, args = ["execution_id=(SELECT execution_id FROM flow_alert_state WHERE id=1)"], []
        if before_id is not None:
            conditions.append("alert_id<?")
            args.append(before_id)
        if severity is not None:
            if severity not in ALERT_SEVERITIES:
                raise ValueError("Tipo de alerta no válido.")
            decisions = ALERT_SEVERITIES[severity]
            conditions.append("json_extract(payload,'$.final_decision') IN (" + ",".join("?" for _ in decisions) + ")")
            args.extend(decisions)
        clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self._database() as db:
            db.execute("BEGIN")
            current = db.execute("SELECT execution_id FROM flow_alert_state WHERE id=1").fetchone()
            rows = db.execute("SELECT alert_id,flow_id,run_id,created_at,payload FROM flow_alerts" + clause
                              + " ORDER BY alert_id DESC LIMIT ?", [*args, limit + 1]).fetchall()
        items = [{"alert_id": row[0], "flow_id": row[1], "run_id": row[2], "created_at": row[3],
                  "result": json.loads(row[4])} for row in rows[:limit]]
        return {"items": items, "next_before_id": items[-1]["alert_id"] if len(rows) > limit else None,
                "execution_id": current[0] if current else None}

    def latest_run_id(self) -> str | None:
        with self._database() as db:
            # Runs are appended on save. Read the last row without scanning its large JSON columns.
            row = db.execute("SELECT run_id FROM flow_runs ORDER BY rowid DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def list_runs(self, flow_id: str | None = None, *, compact: bool = False) -> list[dict[str, Any]]:
        clause, args = (" WHERE flow_id=?", [flow_id]) if flow_id else ("", [])
        table = "flow_run_display" if compact else "flow_runs"
        with self._database() as db:
            rows = db.execute(f"SELECT run_id,flow_id,snapshot,summary,created_at FROM {table}" + clause + " ORDER BY created_at DESC", args).fetchall()
            archives = {key: (filename, json.loads(counts), json.loads(timings)) for key, filename, counts, timings in
                        db.execute("SELECT run_id,filename,counts,timings FROM flow_run_archives")}
        items = []
        for row in rows:
            summary = json.loads(row[3])
            snapshot = json.loads(row[2])
            if row[0] in archives:
                filename, counts, timings = archives[row[0]]
                self._archive_performance(summary, counts, timings)
                if not compact:
                    missing = object()
                    diagnostic = ResultArchive(archive_path(self.path, filename)).diagnostic_snapshots(missing)
                    if diagnostic is not missing:
                        snapshot["diagnostic_snapshots"] = diagnostic
            items.append({"run_id": row[0], "flow_id": row[1], "snapshot": snapshot, "summary": summary, "created_at": row[4]})
        return items

    def run_metrics(self, run_id: str) -> dict[str, Any]:
        with self._database() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT summary,snapshot FROM flow_run_display WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            summary, snapshot = map(json.loads, row)
            metrics = summary.get("metrics")
            measurements = summary.get("measurements")
            if not isinstance(metrics, dict) or metrics.get("version") != METRICS_VERSION:
                metrics = None
            if not isinstance(measurements, dict) or measurements.get("version") != MEASUREMENTS_VERSION:
                measurements = None
            if metrics is None or measurements is None:
                # Recovered labels update the saved summary and invalidate previous cache entries.
                key = (run_id, METRICS_VERSION, MEASUREMENTS_VERSION, sha256(row[0].encode()).digest())
                historical, inferred = self._reconstruct_metrics(db, run_id, summary, key)
                metrics = metrics if metrics is not None else historical
                measurements = measurements if measurements is not None else inferred
        return {"run_id": run_id, **metrics, "timings": run_timings(summary, snapshot.get("flow") or {}),
                "measurements": {**measurements, "resources": (summary.get("performance") or {}).get("resources")}}

    def _reconstruct_metrics(self, db, run_id, summary, key):
        collector = RunMeasurements()
        try:
            archive = archive_for_run(db, self.path, run_id)
            if archive is not None and not archive.is_file():
                raise FileNotFoundError(archive)
            with self._metrics_lock:
                if key in self._metrics_cache:
                    self._metrics_cache.move_to_end(key)
                    return self._metrics_cache[key]
            metrics = reconstruct_metrics(summary, iter_run_results(db, self.path, run_id, "output"),
                                          measurements=collector)
        except FileNotFoundError:
            reason = "Falta el archivo de resultados de esta ejecución."
            return unavailable_metrics(saved_support(summary), reason), RunMeasurements().finalize(reason)
        except (OSError, sqlite3.Error, ValueError, KeyError, IndexError, TypeError, AttributeError, zlib.error):
            logging.getLogger(__name__).warning("No se pudieron leer las métricas de %s", run_id, exc_info=True)
            reason = "No se pudieron leer todos los resultados guardados de Salida."
            return unavailable_metrics(saved_support(summary), reason), RunMeasurements().finalize(reason)
        result = metrics, collector.finalize(metrics["reason"])
        with self._metrics_lock:
            self._metrics_cache[key] = result
            self._metrics_cache.move_to_end(key)
            if len(self._metrics_cache) > METRICS_CACHE_SIZE:
                self._metrics_cache.popitem(last=False)
        return result

    def results(self, run_id: str, node_id: str = "output", offset: int = 0, limit: int = 100) -> dict[str, Any]:
        with self._database() as db:
            if not db.execute("SELECT 1 FROM flow_runs WHERE run_id=?", (run_id,)).fetchone():
                raise KeyError(run_id)
            archive = archive_for_run(db, self.path, run_id)
            if archive is not None:
                total, items = ResultArchive(archive).page(node_id, offset, limit)
                return {"run_id": run_id, "node_id": node_id, "total": total, "offset": offset, "items": items,
                        "next_offset": offset + len(items) if offset + len(items) < total else None}
            total = db.execute("SELECT COUNT(*) FROM flow_results WHERE run_id=? AND node_id=?", (run_id, node_id)).fetchone()[0]
            rows = db.execute("SELECT payload FROM flow_results WHERE run_id=? AND node_id=? ORDER BY sequence LIMIT ? OFFSET ?", (run_id, node_id, limit, offset)).fetchall()
        items = [json.loads(row[0]) for row in rows]
        return {"run_id": run_id, "node_id": node_id, "total": total, "offset": offset, "items": items, "next_offset": offset + len(items) if offset + len(items) < total else None}

    def confidence_series(self, run_id, entity_id=None, offset=0, limit=1000):
        """Read the scalar index without decompressing detector evidence."""
        with self._database() as db:
            if not db.execute("SELECT 1 FROM flow_runs WHERE run_id=?", (run_id,)).fetchone():
                raise KeyError(run_id)
            archive = archive_for_run(db, self.path, run_id)
            catalog = points = None
            selected = entity_id
            if archive is not None:
                connection = ResultArchive(archive).connect()
                try:
                    indexed = connection.execute("SELECT 1 FROM sqlite_master WHERE name='confidence_series'").fetchone()
                    if indexed:
                        catalog = connection.execute("SELECT entity_id,COUNT(*) FROM confidence_series GROUP BY entity_id ORDER BY entity_id").fetchall()
                        selected = entity_id or (catalog[0][0] if catalog else None)
                        points = connection.execute("SELECT sequence,timestamp,value FROM confidence_series WHERE entity_id=? ORDER BY sequence LIMIT ? OFFSET ?",
                                                    (selected, limit, offset)).fetchall()
                finally:
                    connection.close()
            if catalog is None:
                # Existing archives/SQLite runs stay readable without migration.
                counts, points = {}, []
                for record in iter_run_results(db, self.path, run_id, "output"):
                    item = record["payload"]
                    if "confidence" not in item:
                        continue
                    source = item["entity_id"]
                    counts[source] = counts.get(source, 0) + 1
                catalog = sorted(counts.items())
                selected = entity_id or (catalog[0][0] if catalog else None)
                matching = 0
                for record in iter_run_results(db, self.path, run_id, "output"):
                    item = record["payload"]
                    if "confidence" not in item or item["entity_id"] != selected:
                        continue
                    if offset <= matching < offset + limit:
                        confidence = item["confidence"]
                        value = (round(confidence * 100, 6) if type(confidence) in (int, float)
                                 and math.isfinite(confidence) and 0 <= confidence <= 1 else None)
                        points.append((record["sequence"], item["timestamp"], value))
                    matching += 1
            total = dict(catalog).get(selected, 0)
        return {"run_id": run_id, "source_entity_id": selected,
                "entity_id": f"confianza.{selected.replace('.', '_')}" if selected else None,
                "entities": [{"source_entity_id": source, "entity_id": f"confianza.{source.replace('.', '_')}",
                              "count": count} for source, count in catalog],
                "points": [{"sequence": sequence, "timestamp": timestamp, "value": value}
                           for sequence, timestamp, value in points],
                "offset": offset, "total": total,
                "next_offset": offset + len(points) if offset + len(points) < total else None}
