import tempfile
import time
import unittest
from pathlib import Path
import json
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.diagnostico import DiagnosticWorkbench
from app.api.diagnostico import crear_router_diagnostico


class ApiCompatibilityTests(unittest.TestCase):
    def test_flow_model_list_matches_diagnostic_versions_and_active_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            workbench = DiagnosticWorkbench(Path(tmp))
            identifier = "7732263585df42f49892a2a41b287840"
            model = {"model_id": identifier, "method": "gat_lstm",
                     "config": {"architecture_version": "gat_lstm_v3_segments_multiscale"},
                     "created_at": "2026-10-02T14:11:49-03:00"}
            model_dir = workbench.directory / "models"
            model_dir.mkdir(exist_ok=True)
            (model_dir / f"{identifier}.json").write_text(json.dumps(model), encoding="utf-8")
            with patch.object(workbench, "_active", return_value=identifier):
                diagnostic = workbench.status()
                flows = workbench.classifier_models()
            self.assertEqual(flows["active_model_id"], diagnostic["active_model_id"])
            self.assertEqual([item["model_id"] for item in flows["models"]],
                             [item["model_id"] for item in diagnostic["models"]])
            self.assertTrue(flows["models"][0]["active"])
            self.assertEqual(flows["models"][0]["architecture_version"], "gat_lstm_v3_segments_multiscale")

    def test_job_progress_preserves_training_epoch_and_timing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            workbench = DiagnosticWorkbench(Path(tmp))
            def action(progress):
                progress(5, "Iniciando", phase="training", epoch=0, max_epochs=10)
                progress(13, "Época 1", phase="training", epoch=1, max_epochs=10)
                return {"ok": True}
            started = workbench.start_job(action)
            deadline = time.monotonic() + 2
            while workbench.job(started["job_id"])["status"] in {"queued", "running"}:
                if time.monotonic() > deadline:
                    self.fail("El trabajo de prueba no terminó.")
                time.sleep(.01)
            result = workbench.job(started["job_id"])
            self.assertEqual(result["epoch"], 1)
            self.assertEqual(result["max_epochs"], 10)
            self.assertEqual(result["phase"], "training")
            self.assertTrue(result.get("phase_started_at"))
            self.assertTrue(result.get("updated_at"))


    def test_service_reports_missing_optional_torch_without_importing_neural_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            workbench = DiagnosticWorkbench(Path(tmp))
            with patch("app.services.diagnostico.importlib.util.find_spec", return_value=None):
                self.assertFalse(workbench.status()["capabilities"]["gat_lstm"])
                capabilities = workbench.status()["capabilities"]
                self.assertEqual(set(capabilities), {"gat_lstm", "pytorch_available", "gat_lstm_segments", "pyg_available"})
                self.assertFalse(capabilities["gat_lstm_segments"])
                self.assertFalse(capabilities["pyg_available"])



if __name__ == "__main__": unittest.main()
