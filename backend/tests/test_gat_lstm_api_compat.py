import tempfile
import time
import unittest
from pathlib import Path
import json
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.Diagnostico_Trabajo import DiagnosticWorkbench, crear_router_diagnostico


class ApiCompatibilityTests(unittest.TestCase):
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

    def test_legacy_build_and_train_payloads_keep_the_conventional_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            workbench = DiagnosticWorkbench(Path(tmp))
            calls = []
            workbench.start_job = lambda action, *args: calls.append((action.__name__, args)) or {"status": "queued"}
            workbench._read = lambda folder, identifier: {"method": "robust_nearest_prototype"}
            app = FastAPI(); app.include_router(crear_router_diagnostico(workbench))
            client = TestClient(app)
            build = client.post("/diagnostics/build", json={"annotation_ids": ["a"],
                "injection_ids": [], "window_seconds": 3600, "stride_seconds": 1800,
                "include_drafts": False})
            train = client.post("/diagnostics/train", json={"dataset_id": "a" * 32})
            self.assertEqual(build.status_code, 200)
            self.assertEqual(train.status_code, 200)
            self.assertEqual([row[0] for row in calls], ["build_dataset", "train"])

    def test_service_reports_missing_optional_torch_without_importing_neural_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            workbench = DiagnosticWorkbench(Path(tmp))
            with patch("app.Diagnostico_Trabajo.importlib.util.find_spec", return_value=None):
                self.assertFalse(workbench.status()["capabilities"]["gat_lstm"])
                self.assertTrue(workbench.status()["capabilities"]["robust_nearest_prototype"])

    def test_encoder_only_artifact_has_a_specific_activation_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            workbench = DiagnosticWorkbench(Path(tmp))
            identifier = "b" * 32
            (workbench.directory / "encoders" / f"{identifier}.json").write_text(
                json.dumps({"encoder_id": identifier, "can_activate": False}), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "requiere una cabeza diagnóstica"):
                workbench.activate(identifier)


if __name__ == "__main__": unittest.main()
