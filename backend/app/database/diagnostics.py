from datetime import datetime, timezone
from contextlib import contextmanager
import json
import sqlite3

class DiagnosticRepository:
    def initialize_database(self):
        with self._connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions(decision_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    source TEXT NOT NULL, entity_id TEXT NOT NULL, timestamp TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS decisions_source_time ON decisions(source,timestamp);
            """)

    def save_decisions(self, rows):
        with self._connect() as connection:
            connection.executemany("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)", rows)
    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=60)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()
    def _run_snapshot_for_model(self, source, run_id, model_id=None):
        key = f"{source}:{run_id}" if model_id is None else f"{source}:{run_id}:{model_id}"
        with self.lock, self._connect() as connection:
            existing = connection.execute("SELECT payload FROM runs WHERE run_id=?", (key,)).fetchone()
            if existing:
                snapshot = json.loads(existing[0])
                if snapshot.get("model"):
                    self._require_supported(snapshot["model"])
                return snapshot
            selected = model_id or self._active()
            trained = self._read("models", selected) if selected else None
            references = []
            snapshot = {"run_id": key, "model": trained, "references": references, "created_at": datetime.now(timezone.utc).isoformat()}
            connection.execute("INSERT INTO runs VALUES (?,?)", (key, json.dumps(snapshot, ensure_ascii=False)))
            return snapshot
    def results(self, source=None, offset=0, limit=50):
        where, args = (" WHERE source=?", [source]) if source else ("", [])
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM decisions" + where, args).fetchone()[0]
            rows = connection.execute("SELECT payload FROM decisions" + where + " ORDER BY timestamp DESC,decision_id DESC LIMIT ? OFFSET ?", [*args, limit, offset]).fetchall()
            grouped = connection.execute(
                "SELECT json_extract(payload,'$.status'), json_extract(payload,'$.reasons[0]'), COUNT(*) "
                "FROM decisions" + where + " GROUP BY 1,2", args).fetchall()
        items = [json.loads(row[0]) for row in rows]
        indeterminate_by_reason = {}
        indeterminate = 0
        for status, reason, count in grouped:
            if status == "indeterminate":
                key = reason or "unspecified"
                indeterminate_by_reason[key] = count
                indeterminate += count
        return {"items": items, "total": total, "offset": offset,
                "next_offset": offset + len(items) if offset + len(items) < total else None,
                "summary": {"total": total, "determinate": total - indeterminate,
                    "indeterminate": indeterminate,
                    "coverage": (total - indeterminate) / total if total else None,
                    "indeterminate_by_reason": indeterminate_by_reason}}
