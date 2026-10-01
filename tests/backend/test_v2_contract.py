from pathlib import Path
import json
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.diagnostico import crear_router_diagnostico
from app.modulos.clasificador.config import Config
from app.services.diagnostico import DiagnosticWorkbench
from app.database.review import ReviewError


class V2ContractTests(unittest.TestCase):
    def test_retired_artifacts_cannot_be_selected_or_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = DiagnosticWorkbench(Path(tmp))
            identifier = "a" * 32
            for architecture in (None, "gat_lstm_v1", "gat_lstm_v1_entity"):
                config = {} if architecture is None else {"architecture_version": architecture}
                for folder in ("models", "datasets", "configurations"):
                    service._path(folder, identifier).write_text(json.dumps({"method": "gat_lstm", "config": config}))
                    with self.assertRaises(ReviewError):
                        service._read(folder, identifier)
                    self.assertEqual(service._list(folder), [])
                (Path(tmp) / "active.json").write_text(json.dumps({"model_id": identifier}))
                self.assertIsNone(service._active())

    def test_reject_legacy_and_predictive_architectures(self):
        self.assertEqual(Config().architecture_version, "gat_lstm_v2_shared_type")
        with self.assertRaises(ValueError):
            Config(architecture_version="gat_lstm_v1_entity")
        with self.assertRaises(ValueError):
            Config(pretrain_enabled=True)

    def test_no_model_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = DiagnosticWorkbench(Path(tmp))
            context = [{"entity_id": "sensor.a", "timestamp": "2026-01-01T00:00:00Z", "routing": {"terminal": False}},
                       {"entity_id": "sensor.b", "routing": {"terminal": True}}]
            result = service.enrich_contextual([], context)
            self.assertEqual(result[0]["diagnosis_mode"], "unavailable")
            self.assertEqual(result[0]["diagnosis"]["reasons"], ["classifier_unavailable"])
            self.assertNotIn("diagnosis", result[1])
            self.assertNotIn("diagnosis", context[0])
            self.assertEqual(service.status()["mode"], "unavailable")

    def test_http_rejects_retired_method_and_pretraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = DiagnosticWorkbench(Path(tmp))
            api = FastAPI(); api.include_router(crear_router_diagnostico(service))
            with TestClient(api) as client:
                self.assertEqual(client.post("/diagnostics/build", json={"method": "robust_nearest_prototype"}).status_code, 422)
                self.assertEqual(client.post("/diagnostics/train", json={"dataset_id": "a" * 32, "method": "robust_nearest_prototype"}).status_code, 400)
                self.assertEqual(client.post("/diagnostics/pretrain", json={}).status_code, 404)
                self.assertEqual(client.get("/diagnostics/sources?method=robust_nearest_prototype").status_code, 422)
