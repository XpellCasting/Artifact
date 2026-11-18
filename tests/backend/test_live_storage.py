import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.database.diagnostics import DiagnosticRepository
from app.database.flows import FlowStore
from app.database.confidence_entities import ConfidenceEntityStore
from app.flow.evaluacion import FlowEvaluationJobs


class LiveStorageTests(unittest.TestCase):
    def test_transient_wal_open_retries_only_before_any_reading_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ConfidenceEntityStore(Path(folder) / "flow.sqlite3")
            initial = store.start("run", "flow", ["sensor.x"])[0]
            item = {"observation_id": "1", "entity_id": "sensor.x", "confidence": .7}
            failure = sqlite3.OperationalError("disk I/O error")
            failure.sqlite_errorname = "SQLITE_IOERR_SHMOPEN"
            failure.sqlite_errorcode = sqlite3.SQLITE_IOERR_SHMOPEN
            blocked = Mock()
            blocked.execute.side_effect = failure
            connection = sqlite3.connect(store.path)
            with patch("app.database.confidence_entities.sqlite3.connect", side_effect=[blocked, connection]):
                with self.assertLogs("app.database.confidence_entities", level="WARNING"):
                    published = store.record("run", [item])
            self.assertEqual([p["revision"] for p in published], [initial["revision"] + 1])
            self.assertEqual(store.snapshot()["entities"][0]["value"], 70)
            blocked.close.assert_called_once()
            with patch("app.database.confidence_entities.sqlite3.connect", return_value=blocked) as opened:
                with self.assertLogs("app.database.confidence_entities", level="WARNING"), self.assertRaises(sqlite3.OperationalError):
                    store.record("run", [item])
                self.assertEqual(opened.call_count, 2)

    @patch.dict(os.environ, LIVE_HISTORY_READINGS="3", LIVE_HISTORY_SESSIONS="2")
    def test_live_eviction_during_reception_and_after_restart_preserves_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            store = FlowStore(Path(folder) / "flow.sqlite3")
            confidence = ConfidenceEntityStore(store.path)
            source = Mock(selected=["sensor.x"])
            source.status.return_value = {"status": "stopped"}
            jobs = FlowEvaluationJobs(store, confidence_store=confidence)

            def run(live):
                def evaluate(progress, sink):
                    for n in range(8):
                        item = {"observation_id": str(n), "entity_id": "sensor.x", "timestamp": "2026-11-17",
                                "confidence": n / 10, "final_decision": "suspicion"}
                        sink({"input": [item], "output": [item]})
                        self.assertEqual(len(store.alerts()["items"]), min(n + 1, 3) if live else n + 1)
                    return {"candidate_count": 8}, None, {}

                with patch("app.flow.evaluacion.Thread") as thread:
                    job = jobs.start("flow", evaluate, live_source=source if live else None)
                    thread.call_args.kwargs["target"]()
                final = jobs.get(job["job_id"])
                self.assertEqual(final["status"], "completed", final.get("error"))
                return final["run"]["run_id"]

            csv = run(False)
            first, second, last = run(True), run(True), run(True)
            reopened = FlowStore(store.path)
            self.assertEqual({r["run_id"] for r in reopened.list_runs(compact=True)}, {csv, second, last})
            self.assertFalse((store.path.with_suffix(".results") / (first + ".sqlite3")).exists())
            self.assertEqual(reopened.results(csv, "output")["total"], 8)
            for node in ("input", "output"):
                page = reopened.results(last, node)
                self.assertEqual(page["total"], 3)
                self.assertEqual([item["observation_id"] for item in page["items"]], ["5", "6", "7"])
            self.assertEqual([p["value"] for p in reopened.confidence_series(last, "sensor.x")["points"]],
                             [50, 60, 70])
            self.assertEqual(len(confidence.snapshot()["entities"]), 1)
            self.assertEqual(len(list(store.path.with_suffix(".results").glob("*.sqlite3"))), 3)

    def test_diagnostic_retention_is_global_for_new_live_decisions_only(self):
        with tempfile.TemporaryDirectory() as folder:
            store = DiagnosticRepository()
            store.database = Path(folder) / "diagnostics.sqlite3"
            store.initialize_database()
            def row(i):
                return (str(i), "run", "flow_evaluation", "sensor.x", "2026-11-17", "{}")
            store.save_decisions([row(0)])
            store.save_decisions([row(i) for i in range(1, 8)], live_limit=2)
            store.save_decisions([row(8)], live_limit=2)
            with store._connect() as db:
                self.assertEqual([r[0] for r in db.execute("SELECT decision_id FROM decisions ORDER BY decision_id")],
                                 ["0", "7", "8"])
