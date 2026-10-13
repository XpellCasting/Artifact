from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Any
from uuid import uuid4
from app.flow.definitions import _json, _now, validate_flow

class FlowStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as db:
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
            """)

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
            db.execute("INSERT INTO flow_runs VALUES (?,?,?,?,?)", (run_id, flow_id, _json(snapshot), _json(summary), _now()))
            records = []
            for node_id, items in results.items():
                for sequence, item in enumerate(items):
                    records.append((run_id, node_id, str(item["observation_id"]), sequence, _json(item)))
            db.executemany("INSERT INTO flow_results VALUES (?,?,?,?,?)", records)
        return run_id

    def save_run_stream(self, flow_id: str, snapshot: dict[str, Any], summary: dict[str, Any], records) -> tuple[str, float]:
        """Persist a completed spool atomically without materializing its rows."""
        run_id = uuid4().hex
        started = perf_counter()
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            def values():
                for record in records:
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
            db.commit()
            return run_id, round((perf_counter() - started) * 1000, 3)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

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
