import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from threading import Event
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.database.flows import FlowStore
from app.flow.evaluacion import FlowEvaluationJobs


def reading(identifier, decision):
    return {"observation_id": identifier, "entity_id": "sensor.temperature", "timestamp": "2026-01-01T01:00:00Z",
            "final_decision": decision, "evidence": []}


class FlowAlertsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "flows.sqlite"
        self.store = FlowStore(self.path)

    def test_only_final_detections_are_saved_and_pagination_survives_new_arrivals_and_restart(self):
        output = [reading(str(index), decision) for index, decision in enumerate([
            "normal", "no_alert", "indeterminate", "not_routed", "fault_compatible", "environmental", "suspicion"])]
        run_id = self.store.save_run("flow", {}, {}, {"output": output, "classifier": [reading("intermediate", "fault_compatible")]})
        page = self.store.alerts(limit=2)
        self.assertEqual([row["result"]["final_decision"] for row in page["items"]], ["suspicion", "environmental"])
        self.assertTrue(all(row["run_id"] == run_id for row in page["items"]))
        self.store.record_alerts("live", "flow", [reading("new", "fault_compatible")])
        older = self.store.alerts(page["next_before_id"], limit=2)
        self.assertEqual([row["result"]["final_decision"] for row in older["items"]], ["fault_compatible"])
        self.assertIsNone(older["next_before_id"])
        self.assertEqual(len(FlowStore(self.path).alerts()["items"]), 4)

    def test_running_evaluations_emit_persistent_alerts_before_completion_and_final_saving_does_not_duplicate_them(self):
        jobs = FlowEvaluationJobs(self.store)
        emitted, finish = Event(), Event()
        item = reading("one", "fault_compatible")
        def evaluate(progress, sink):
            sink({"output": [item], "classifier": [reading("ignored", "fault_compatible")]})
            emitted.set()
            finish.wait(5)
            return {"candidate_count": 1}, None, {"flow": {}}
        try:
            job = jobs.start("flow", evaluate)
            self.assertTrue(emitted.wait(5))
            self.assertEqual(jobs.get(job["job_id"])["status"], "running")
            self.assertEqual(len(FlowStore(self.path).alerts()["items"]), 1)
            self.assertIsNone(self.store.alerts()["items"][0]["run_id"])
        finally:
            finish.set()
        for _ in range(200):
            status = jobs.get(job["job_id"])
            if status["status"] != "running":
                break
            Event().wait(.01)
        self.assertEqual(status["status"], "completed", status)
        alerts = self.store.alerts()["items"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["run_id"], status["run"]["run_id"])

    def test_stream_failure_rolls_back_new_alerts_with_the_run(self):
        record = {"node_id": "output", "sequence": 0, "observation_id": "same", "payload": reading("same", "fault_compatible")}
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_run_stream("flow", {}, {}, [record, record])
        self.assertEqual(self.store.alerts()["items"], [])

    def test_alerts_already_emitted_survive_an_evaluation_failure(self):
        jobs = FlowEvaluationJobs(self.store)
        def evaluate(progress, sink):
            sink({"output": [reading("one", "environmental")]})
            raise ValueError("interrupted fixture")
        with patch("app.flow.evaluacion.Thread") as thread, self.assertLogs("app.flow.evaluacion", level="ERROR"):
            thread.return_value.start.side_effect = lambda: thread.call_args.kwargs["target"]()
            job = jobs.start("flow", evaluate)
        self.assertEqual(jobs.get(job["job_id"])["status"], "failed")
        self.assertEqual(FlowStore(self.path).alerts()["items"][0]["result"]["final_decision"], "environmental")

    def test_existing_saved_runs_are_imported_once(self):
        item = reading("legacy", "environmental")
        with self.store._database() as db:
            db.execute("DROP TABLE flow_alerts")
            db.execute("INSERT INTO flow_runs VALUES (?,?,?,?,?)", ("legacy-run", "flow", "{}", "{}", "2026-01-01"))
            db.execute("INSERT INTO flow_results VALUES (?,?,?,?,?)", ("legacy-run", "output", "legacy", 0, json.dumps(item)))
        restored = FlowStore(self.path)
        self.assertEqual(restored.alerts()["items"][0]["result"], item)
        self.assertEqual(len(FlowStore(self.path).alerts()["items"]), 1)

    def test_alerts_api_returns_final_alerts_and_validates_the_cursor(self):
        runtime = ModuleType("app.services.runtime")
        runtime.state, runtime.flow_store = Mock(), self.store
        runtime.flow_evaluation_jobs, runtime.home_assistant_bridge = Mock(), Mock()
        spec = importlib.util.spec_from_file_location("isolated_flow_alerts_api", Path(__file__).parents[2] / "backend/app/api/flows.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"app.services.runtime": runtime}):
            spec.loader.exec_module(module)
        api = FastAPI()
        api.include_router(module.router)
        self.store.record_alerts("live", "flow", [reading("one", "suspicion")])
        with TestClient(api) as client:
            page = client.get("/flow-alerts?limit=1")
            self.assertEqual(page.status_code, 200)
            self.assertEqual(page.json()["items"][0]["result"]["final_decision"], "suspicion")
            self.assertEqual(client.get("/flow-alerts?before_id=0").status_code, 422)
            self.assertEqual(client.get("/flow-alerts?limit=5000").status_code, 422)
