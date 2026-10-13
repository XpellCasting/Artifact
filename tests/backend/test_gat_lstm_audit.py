from app.config import database_path
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from app.services.auditoria import audit_sources
from app.modulos.clasificador.config import Config


class AuditTests(unittest.TestCase):
    def test_declared_partition_conflicts_with_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = {}
            for name in ("desarrollo", "validacion", "prueba", "reserva_agosto"):
                path = root / f"{name}.csv"
                path.write_text("entity_id,state,last_changed\na,1,2026-02-01T00:00:00Z\n", encoding="utf-8")
                parts[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "rows": 1, "entities": {"a": 1}}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"boundaries": ["2026-02-01T00:00:00Z", "2026-02-02T00:00:00Z",
                                                           "2026-02-03T00:00:00Z", "2026-02-04T00:00:00Z"],
                                            "partitions": parts}), encoding="utf-8")
            review = root / "review"
            review.mkdir()
            db = sqlite3.connect(database_path("review.sqlite3", review))
            db.executescript("""CREATE TABLE datasets(metadata TEXT,source_name TEXT);
                CREATE TABLE annotations(payload TEXT);
                CREATE TABLE events(dataset_id TEXT,entity_id TEXT,ts INTEGER,state TEXT,is_valid INTEGER,source_row INTEGER);""")
            source = {"dataset_id": "bad", "filename": "bad.csv", "sha256": "abc", "partition": "development",
                      "start": "2026-02-02T12:00:00Z", "end": "2026-02-02T12:00:00Z", "rows": 1,
                      "indexed_rows": 1, "rejected_rows": 0, "entities": [{"entity_id": "a", "kind": "continuous"}]}
            db.execute("INSERT INTO datasets VALUES (?,?)", (json.dumps(source), "bad.csv"))
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)", ("bad", "a", 1780401600000000, "1", 1, 2))
            db.commit(); db.close()
            report = audit_sources(Config(), manifest_path=manifest, review_dir=review).as_dict()
            self.assertTrue(any(issue["kind"] == "partition_mismatch" for issue in report["issues"]))
            self.assertFalse(report["imported_sources"][0]["declared_partition"] ==
                             report["imported_sources"][0]["actual_start_partition"])


if __name__ == "__main__":
    unittest.main()
