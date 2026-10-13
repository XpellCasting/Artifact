import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.database.migration import migrate


class LayoutMigrationTests(unittest.TestCase):
    def fixture(self, root):
        review = root / "data/review"
        review.mkdir(parents=True)
        db = sqlite3.connect(review / "review.sqlite3")
        db.executescript("CREATE TABLE annotations(id TEXT PRIMARY KEY, payload TEXT); INSERT INTO annotations VALUES ('keep', 'manual evidence');")
        db.commit(); db.close()
        old = review / "diagnostics"
        (old / "models").mkdir(parents=True)
        identifier = "a" * 32
        (old / "models" / f"{identifier}.json").write_text(json.dumps({"model_id": identifier, "method": "robust_nearest_prototype"}))
        (old / "active.json").write_text(json.dumps({"model_id": identifier}))
        cache = root / "data/cache"
        cache.mkdir()
        db = sqlite3.connect(cache / "flow_lab.sqlite3")
        db.executescript("CREATE TABLE flows(flow_id TEXT, payload TEXT); CREATE TABLE flow_runs(run_id TEXT, snapshot TEXT); CREATE TABLE flow_results(run_id TEXT,node_id TEXT,payload TEXT);")
        db.execute("INSERT INTO flows VALUES (?,?)", ("keep-flow", json.dumps({"nodes": [{"id": "classifier", "type": "classifier", "config": {"model_id": identifier}}]})))
        db.commit(); db.close()
        return identifier

    def test_inventory_migration_repeat_and_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identifier = self.fixture(root)
            plan = migrate(root)
            self.assertEqual(plan["removed"]["models"], [identifier])
            self.assertTrue((root / "data/review/review.sqlite3").exists())
            result = migrate(root, apply=True, offline=True, progress=lambda *a, **k: None)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(migrate(root, apply=True, offline=True), result)
            self.assertFalse((root / "models/gat_lstm/models" / f"{identifier}.json").exists())
            db = sqlite3.connect(root / "storage/database/review.sqlite3")
            self.assertEqual(db.execute("SELECT * FROM annotations").fetchall(), [("keep", "manual evidence")]); db.close()
            db = sqlite3.connect(root / "storage/database/flow_lab.sqlite3")
            flow = json.loads(db.execute("SELECT payload FROM flows").fetchone()[0]); db.close()
            self.assertEqual(flow["requires_model_selection"], [identifier])

    def test_conflict_is_detected_before_any_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            target = root / "storage/database/review.sqlite3"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"different content")
            with self.assertRaisesRegex(ValueError, "Destino incompatible"):
                migrate(root, apply=True, offline=True)
            self.assertEqual(target.read_bytes(), b"different content")
            self.assertTrue((root / "data/review/review.sqlite3").exists())

    def test_explicit_offline_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "offline"):
                migrate(root, apply=True)

    def test_repeat_refuses_running_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "storage/migrations/layout-v2.json"
            journal.parent.mkdir(parents=True)
            journal.write_text(json.dumps({"status": "complete"}))
            with patch("app.database.migration.socket.socket") as socket:
                socket.return_value.__enter__.return_value.connect_ex.return_value = 0
                with self.assertRaisesRegex(ValueError, "puerto 8000"):
                    migrate(root, apply=True, offline=True)
