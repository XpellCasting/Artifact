from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Any
from uuid import uuid4
from app.flow.definitions import _json, _now, validate_flow

ALERT_DECISIONS = {"fault_compatible", "environmental", "suspicion"}
ALERT_SEVERITIES = {"fault": ("fault_compatible",), "anomaly": ("environmental", "suspicion")}

class FlowStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as db:
            import_alerts = not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='flow_alerts'").fetchone()
            db.executescript("""
                CREATE TABLE IF NOT EXISTS flows (
                    flow_id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS flow_runs (
                    run_id TEXT PRIMARY KEY, flow_id TEXT NOT NULL, snapshot TEXT NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS flow_results (
                    run_id TEXT NOT NULL, node_id TEXT NOT NULL, observation_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (run_id, node_id, observation_id)
                );
                CREATE INDEX IF NOT EXISTS flow_results_page ON flow_results(run_id, node_id, sequence);
                CREATE TABLE IF NOT EXISTS flow_alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL, observation_id TEXT NOT NULL,
                    flow_id TEXT NOT NULL, run_id TEXT, created_at TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE (execution_id, observation_id)
                );
                CREATE INDEX IF NOT EXISTS flow_alerts_execution ON flow_alerts(execution_id,alert_id);
                CREATE TABLE IF NOT EXISTS flow_alert_state (
                    id INTEGER PRIMARY KEY CHECK (id=1), execution_id TEXT NOT NULL
                );
            """)
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
        item = {"flow_id": flow_id, "name": str(payload.get("name") or "Flujo sin nombre"), "nodes": payload["nodes"], "edges": payload["edges"]}
        with self._database() as db:
            db.execute("INSERT INTO flows VALUES (?,?,?,?)", (flow_id, item["name"], _json(item), _now()))
        return item

    def list(self) -> list[dict[str, Any]]:
        with self._database() as db:
            rows = db.execute("SELECT payload,updated_at FROM flows ORDER BY updated_at DESC").fetchall()
        return [{**json.loads(payload), "updated_at": updated} for payload, updated in rows]

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
        item = {"flow_id": flow_id, "name": str(payload.get("name") or "Flujo sin nombre"), "nodes": payload["nodes"], "edges": payload["edges"]}
        updated_at = _now()
        with self._database() as db:
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
    def _activate_alert_execution(db, execution_id):
        db.execute("INSERT INTO flow_alert_state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET execution_id=excluded.execution_id",
                   (execution_id,))

    def begin_alert_execution(self, execution_id: str) -> None:
        with self._database() as db:
            self._activate_alert_execution(db, execution_id)

    @staticmethod
    def _insert_alerts(db, execution_id, flow_id, items, run_id=None):
        created_at = _now()
        db.executemany("""INSERT OR IGNORE INTO flow_alerts
            (execution_id,observation_id,flow_id,run_id,created_at,payload) VALUES (?,?,?,?,?,?)""",
            ((execution_id, str(item["observation_id"]), flow_id, run_id, created_at, _json(item))
             for item in items if item.get("final_decision") in ALERT_DECISIONS))

    def record_alerts(self, execution_id: str, flow_id: str, items: list[dict[str, Any]]) -> None:
        if not any(item.get("final_decision") in ALERT_DECISIONS for item in items):
            return
        with self._database() as db:
            db.execute("INSERT OR IGNORE INTO flow_alert_state VALUES (1,?)", (execution_id,))
            self._insert_alerts(db, execution_id, flow_id, items)

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

    def list_runs(self, flow_id: str | None = None) -> list[dict[str, Any]]:
        clause, args = (" WHERE flow_id=?", [flow_id]) if flow_id else ("", [])
        with self._database() as db:
            rows = db.execute("SELECT run_id,flow_id,snapshot,summary,created_at FROM flow_runs" + clause + " ORDER BY created_at DESC", args).fetchall()
        return [{"run_id": row[0], "flow_id": row[1], "snapshot": json.loads(row[2]), "summary": json.loads(row[3]), "created_at": row[4]} for row in rows]

    def results(self, run_id: str, node_id: str = "output", offset: int = 0, limit: int = 100) -> dict[str, Any]:
        with self._database() as db:
            if not db.execute("SELECT 1 FROM flow_runs WHERE run_id=?", (run_id,)).fetchone():
                raise KeyError(run_id)
            total = db.execute("SELECT COUNT(*) FROM flow_results WHERE run_id=? AND node_id=?", (run_id, node_id)).fetchone()[0]
            rows = db.execute("SELECT payload FROM flow_results WHERE run_id=? AND node_id=? ORDER BY sequence LIMIT ? OFFSET ?", (run_id, node_id, limit, offset)).fetchall()
        items = [json.loads(row[0]) for row in rows]
        return {"run_id": run_id, "node_id": node_id, "total": total, "offset": offset, "items": items, "next_offset": offset + len(items) if offset + len(items) < total else None}
