import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.database.flows import FlowStore


class FlowLoadingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "flows.sqlite"
        self.snapshot = {"flow": {"name": "Saved flow", "nodes": [], "edges": []},
                         "source": {"filename": "readings.csv", "rows": 3},
                         "baseline_provenance": {"continuous_models": {"node": {"sensor.a": {"eligible": True}}}},
                         "diagnostic_snapshots": {"model": "large audit data" * 40000}}
        self.summary = {"candidate_count": 3, "output_count": 0, "evaluation": {"labelled": True, "groups": []}}

    def test_compact_list_retains_graph_metrics_and_calibration_without_changing_full_snapshots(self):
        store = FlowStore(self.path)
        with patch("app.database.flows._now", side_effect=["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"]):
            first = store.save_run("a", self.snapshot, self.summary, {})
            second = store.save_run("b", self.snapshot, self.summary, {})
            third = store.save_run("a", self.snapshot, self.summary, {})
        compact = store.list_runs(compact=True)
        self.assertEqual([run["run_id"] for run in compact], [third, second, first])
        self.assertEqual([run["run_id"] for run in store.list_runs("a", compact=True)], [third, first])
        self.assertLess(len(json.dumps(compact)), len(json.dumps(store.list_runs())) // 100)
        for run in compact:
            self.assertEqual(run["summary"], self.summary)
            self.assertEqual(run["snapshot"], {key: value for key, value in self.snapshot.items() if key != "diagnostic_snapshots"})
        self.assertEqual(store.list_runs()[0]["snapshot"], self.snapshot)

    def test_existing_database_backfills_display_data_once_and_preserves_the_original(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE flow_runs (run_id TEXT PRIMARY KEY,flow_id TEXT,snapshot TEXT,summary TEXT,created_at TEXT)")
            db.execute("INSERT INTO flow_runs VALUES (?,?,?,?,?)", ("old", "a", json.dumps(self.snapshot), json.dumps(self.summary), "2026-01-01"))
        first = FlowStore(self.path).list_runs(compact=True)
        restarted = FlowStore(self.path)
        self.assertEqual(first, restarted.list_runs(compact=True))
        self.assertEqual(len(first), 1)
        self.assertNotIn("diagnostic_snapshots", first[0]["snapshot"])
        self.assertEqual(restarted.list_runs()[0]["snapshot"], self.snapshot)

    def test_updates_and_deletions_from_existing_tools_are_reflected_in_the_compact_list(self):
        store = FlowStore(self.path)
        identifier = store.save_run("a", self.snapshot, self.summary, {})
        changed = {**self.summary, "output_count": 2}
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE flow_runs SET summary=? WHERE run_id=?", (json.dumps(changed), identifier))
        self.assertEqual(store.list_runs(compact=True)[0]["summary"], changed)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM flow_runs WHERE run_id=?", (identifier,))
        self.assertEqual(store.list_runs(compact=True), [])

    def test_streamed_save_publishes_compact_data_atomically_and_rolls_back_failures(self):
        store = FlowStore(self.path)
        def failing_records():
            yield {"node_id": "output", "observation_id": "partial", "sequence": 0,
                   "payload": {"observation_id": "partial", "final_decision": "normal"}}
            raise ValueError("interrupted spool")
        with self.assertRaisesRegex(ValueError, "interrupted spool"):
            store.save_run_stream("a", self.snapshot, dict(self.summary), failing_records())
        self.assertEqual(store.list_runs(compact=True), [])
        identifier, _ = store.save_run_stream("a", self.snapshot, dict(self.summary), iter(()))
        self.assertEqual(store.list_runs(compact=True)[0]["run_id"], identifier)
        self.assertNotIn("diagnostic_snapshots", store.list_runs(compact=True)[0]["snapshot"])
        self.assertEqual(store.list_runs()[0]["snapshot"], self.snapshot)


if __name__ == "__main__":
    unittest.main()
