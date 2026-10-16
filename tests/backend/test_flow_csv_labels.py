import hashlib
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from threading import Lock
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pandas as pd

from app.database.flows import FlowStore
from app.flow.recover_labels import recover_run_labels
from app.services.historial.cargar_historial import CargadorHistorial


class CsvFlowRequirementTests(unittest.TestCase):
    def setUp(self):
        runtime = ModuleType("app.services.runtime")
        frame = pd.DataFrame({"timestamp_dt": [pd.Timestamp("2026-01-01T00:00:00Z")]})
        runtime.state = SimpleNamespace(historial=SimpleNamespace(eventos=frame, entidades=frame),
            future_baseline_template={}, eventos_validacion_ordenados=frame,
            validation_replay_lock=Lock(), validation_replay_state={"labelled": False},
            future_context_map_template={}, filtro_estadistico=None, filtro_contextual=None,
            diagnostic_workbench=None, _ruta_almacen_resultados_validacion=lambda: Path("unused"))
        runtime.flow_store = Mock()
        runtime.flow_store.get.return_value = {"nodes": [{"id": "input", "type": "input"}, {"id": "output", "type": "output"}]}
        runtime.flow_evaluation_jobs = Mock()
        runtime.flow_evaluation_jobs.start.return_value = {"job_id": "job", "status": "running"}
        runtime.home_assistant_bridge = Mock()
        runtime.home_assistant_bridge.status.return_value = {"bridge_connected": True}
        spec = importlib.util.spec_from_file_location("isolated_csv_flow_api", Path(__file__).parents[2] / "backend/app/api/flows.py")
        self.api_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"app.services.runtime": runtime}):
            spec.loader.exec_module(self.api_module)
        app = FastAPI()
        app.include_router(self.api_module.router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.runtime = runtime

    def test_csv_without_labels_is_rejected_before_starting_a_job(self):
        response = self.client.post("/flows/flow/runs", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("etiquetado", response.json()["detail"])
        self.runtime.flow_evaluation_jobs.start.assert_not_called()

    def test_csv_with_no_stored_truth_cannot_produce_an_empty_comparison(self):
        self.runtime.state.validation_replay_state["labelled"] = True
        with patch.object(self.api_module.labelled_evaluation, "read_truth", return_value=[]):
            response = self.client.post("/flows/flow/runs", json={})
        self.assertEqual(response.status_code, 400)
        self.runtime.flow_evaluation_jobs.start.assert_not_called()

    def test_labelled_csv_starts_and_live_reception_requires_no_csv(self):
        self.runtime.state.validation_replay_state["labelled"] = True
        with patch.object(self.api_module.labelled_evaluation, "read_truth", return_value=[{"raw_index": 0}]):
            response = self.client.post("/flows/flow/runs", json={})
        self.assertEqual(response.status_code, 200)
        self.runtime.state.validation_replay_state = {}
        self.runtime.state.eventos_validacion_ordenados = None
        with patch.object(self.api_module, "HomeAssistantEventSource") as source:
            response = self.client.post("/flows/flow/live/start", json={"connection_mode": "integration"})
        self.assertEqual(response.status_code, 200)
        self.assertIs(self.runtime.flow_evaluation_jobs.start.call_args.kwargs["live_source"], source.return_value)


class RecoverFlowLabelsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.csv = root / "truth.csv"
        self.csv.write_text("entity_id,state,last_changed,es_sintetica,tipo,tipo_falla\n"
            "sensor.a,1,2026-01-01T00:00:00Z,false,normal,NULL\n"
            "sensor.a,2,2026-01-01T00:01:00Z,true,Falla,Deriva\n"
            "sensor.a,3,2026-01-01T00:02:00Z,false,anomalia,NULL\n", encoding="utf-8")
        events = CargadorHistorial().cargar_real(self.csv).eventos.sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable")
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(
            events[["entity_id", "timestamp_dt", "state", "_raw_index"]], index=False).values.tobytes()).hexdigest()
        self.store = FlowStore(root / "flows.sqlite3")
        snapshot = {"execution_mode": "causal_future", "source": {"rows": 3, "fingerprint": fingerprint},
            "flow": {"nodes": [{"id": "output", "type": "output"}, {"id": "statistical", "type": "statistical_focus"}]}}
        self.items = [{"observation_id": str(i), "entity_id": "sensor.a", "source_raw_index": i,
            "final_decision": decision, "evidence": [{"node_id": "statistical", "status": "compatible" if i == 1 else "clear",
                "data": {"fault_type": "drift" if i == 1 else None, "statistical_observation": {
                    "evaluable": True, "detected_before_suppression": i < 2, "alert_emitted": i == 0}}}]}
            for i, decision in enumerate(["suspicion", "fault_compatible", "no_alert"])]
        summary = {"candidate_count": 3, "output_count": 3, "evaluation": {"labelled": False, "groups": []},
            "statistical_evaluation_version": "pre_suppression_v1", "performance": {"unchanged": True}}
        self.run_id = self.store.save_run("flow", snapshot, summary, {"output": self.items, "statistical": self.items})

    def test_recovery_preserves_detections_and_rebuilds_saved_reports_with_a_backup(self):
        before = self.store.list_runs()[0]
        self.assertFalse(recover_run_labels(self.store.path, self.csv, self.run_id)["applied"])
        self.assertEqual(self.store.list_runs()[0], before)
        result = recover_run_labels(self.store.path, self.csv, self.run_id, apply=True)
        self.assertEqual(result["matched_rows"], 3)
        self.assertTrue(Path(result["backup"]).is_file())
        with closing(sqlite3.connect(result["backup"])) as db:
            self.assertFalse(json.loads(db.execute("SELECT summary FROM flow_runs").fetchone()[0])["evaluation"]["labelled"])
        run = self.store.list_runs()[0]
        groups = {g["expected"]: g for g in run["summary"]["evaluation"]["groups"]}
        self.assertEqual(groups["normal"]["false_positives"], 1)
        self.assertEqual(groups["Falla"]["type_identified"], 1)
        self.assertEqual(groups["anomalia"]["missed_evaluable"], 1)
        self.assertEqual(run["summary"]["performance"], before["summary"]["performance"])
        self.assertEqual(run["snapshot"]["source"]["fingerprint"], before["snapshot"]["source"]["fingerprint"])
        for old, new in zip(self.items, self.store.results(self.run_id)["items"]):
            self.assertIn("evaluation", new)
            self.assertEqual({k: v for k, v in new.items() if k != "evaluation"}, old)
        stats = {g["expected"]: g for g in run["summary"]["statistical_evaluation"]["statistical"]["groups"]}
        self.assertEqual(stats["normal"]["false_positive_alerts"], 1)
        self.assertEqual(stats["Falla"]["suppressed_rows"], 1)
        self.assertEqual(len(self.store.alerts()["items"]), 2)
        with self.assertRaisesRegex(ValueError, "ya tiene etiquetas"):
            recover_run_labels(self.store.path, self.csv, self.run_id, apply=True)

    def test_another_csv_cannot_overwrite_the_saved_results(self):
        self.csv.write_text(self.csv.read_text().replace("sensor.a,1,", "sensor.a,99,"))
        before = self.store.list_runs()[0]
        with self.assertRaisesRegex(ValueError, "no coincide"):
            recover_run_labels(self.store.path, self.csv, self.run_id, apply=True)
        self.assertEqual(self.store.list_runs()[0], before)
        self.assertFalse((self.store.path.parent / "backups").exists())

    def test_incomplete_saved_outputs_roll_back_the_report_and_row_annotations(self):
        with self.store._database() as db:
            summary = json.loads(db.execute("SELECT summary FROM flow_runs WHERE run_id=?", (self.run_id,)).fetchone()[0])
            summary["output_count"] = 4
            db.execute("UPDATE flow_runs SET summary=? WHERE run_id=?", (json.dumps(summary), self.run_id))
        before = self.store.list_runs()[0]
        with self.assertRaisesRegex(ValueError, "todos los resultados"):
            recover_run_labels(self.store.path, self.csv, self.run_id, apply=True)
        self.assertEqual(self.store.list_runs()[0], before)
        self.assertEqual(self.store.results(self.run_id)["items"], self.items)


if __name__ == "__main__":
    unittest.main()
