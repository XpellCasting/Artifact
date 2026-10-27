import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database.flows import FlowStore
from app.flow.evaluacion import FlowEvaluationJobs


class FlowSyncTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "flows.sqlite"
        self.store = FlowStore(self.path)
        self.jobs = FlowEvaluationJobs(self.store)

    def api(self, jobs):
        runtime = ModuleType("app.services.runtime")
        runtime.state, runtime.flow_store = Mock(), self.store
        runtime.flow_evaluation_jobs, runtime.home_assistant_bridge = jobs, Mock()
        spec = importlib.util.spec_from_file_location("isolated_flow_sync_api", Path(__file__).parents[2] / "backend/app/api/flows.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"app.services.runtime": runtime}):
            spec.loader.exec_module(module)
        api = FastAPI()
        api.include_router(module.router)
        return api

    def test_independent_clients_discover_the_same_execution_and_results_without_a_job_id(self):
        flow = {"flow_id": "flow", "name": "Flujo compartido", "nodes": [], "edges": []}
        summary = {"candidate_count": 4}
        evaluate = Mock(return_value=(summary, None, {"flow": flow, "source": {}}))
        api = self.api(self.jobs)
        with TestClient(api) as first, TestClient(api) as second:
            self.assertEqual(first.get("/flow-evaluations").json(), {"job": None, "latest_run_id": None})
            with patch("app.flow.evaluacion.Thread") as worker:
                started = self.jobs.start("flow", evaluate, flow=flow)
                work = worker.call_args.kwargs["target"]
            flow["name"] = "Edición posterior"
            self.jobs.jobs[started["job_id"]].update(processed=2, total=4)
            observed = second.get("/flow-evaluations").json()
            self.assertEqual(observed, first.get("/flow-evaluations").json())
            self.assertEqual(observed["job"]["job_id"], started["job_id"])
            self.assertEqual(observed["job"]["processed"], 2)
            self.assertEqual(observed["job"]["flow"]["name"], "Flujo compartido")
            observed["job"]["flow"]["nodes"].append({"id": "browser-edit"})
            self.assertEqual(self.jobs.latest()["flow"]["nodes"], [])
            with self.assertRaisesRegex(ValueError, "en curso"):
                self.jobs.start("other", evaluate)
            work()
            completed = second.get("/flow-evaluations").json()
            self.assertEqual(completed["job"]["status"], "completed")
            self.assertNotIn("run", completed["job"])
            details = second.get(f"/flow-evaluations/{started['job_id']}").json()
            self.assertEqual(details["run"]["summary"], summary)
            run_id = completed["job"]["run_id"]
            self.assertEqual(completed["latest_run_id"], run_id)
            self.assertEqual(first.get("/flow-runs").json()["items"][0]["run_id"], run_id)

        # Discovery still announces persisted results after the API is restarted.
        with TestClient(self.api(FlowEvaluationJobs(FlowStore(self.path)))) as restarted:
            self.assertEqual(restarted.get("/flow-evaluations").json(), {"job": None, "latest_run_id": run_id})

    def test_live_progress_and_stop_are_shared_and_the_next_execution_replaces_the_previous_one(self):
        source = Mock()
        source.status.return_value = {"status": "connected", "received": 3}
        evaluate = Mock(return_value=({"candidate_count": 0}, None, {"source": {}}))
        with patch("app.flow.evaluacion.Thread") as worker:
            started = self.jobs.start("live-flow", evaluate, live_source=source)
            finish_live = worker.call_args.kwargs["target"]
        with TestClient(self.api(self.jobs)) as other_device:
            shared = other_device.get("/flow-evaluations").json()["job"]
            self.assertEqual(shared["mode"], "home_assistant")
            self.assertEqual(shared["connection"]["received"], 3)
            stopped = other_device.post(f"/flow-evaluations/{started['job_id']}/stop")
            self.assertEqual(stopped.status_code, 200)
            self.assertEqual(self.jobs.latest()["phase"], "stopping")
            source.stop.assert_called_once()
            finish_live()
            with patch("app.flow.evaluacion.Thread") as worker:
                next_job = self.jobs.start("csv-flow", evaluate)
                finish_csv = worker.call_args.kwargs["target"]
            shared = other_device.get("/flow-evaluations").json()["job"]
            self.assertEqual(shared["job_id"], next_job["job_id"])
            self.assertEqual(shared["mode"], "csv")
            finish_csv()


if __name__ == "__main__":
    unittest.main()
