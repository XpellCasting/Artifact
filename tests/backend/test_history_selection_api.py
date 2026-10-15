import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.services.history import ServicesHistoryMixin
from app.services.history_selection import HistoryCandidates


class HistorySelectionApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.directory = Path(self.folder.name)
        self.candidates = HistoryCandidates(self.directory / "candidates")
        self.service = ServicesHistoryMixin()
        self.service.real_history_path = self.directory / "active.csv"
        self.original = "entity_id,state,last_changed\nsensor.previous,10,2026-01-01T00:00:00Z\n"
        self.service.real_history_path.write_text(self.original, encoding="utf-8")
        self.service._crear_job_carga = Mock(return_value="preparation-job")
        self.service._iniciar_hilo_carga = Mock()
        self.service.estado_carga_real = Mock(return_value={"job_id": "preparation-job", "status": "queued"})
        self.service.iniciar_importacion_home_assistant = Mock(return_value={"job_id": "home-assistant-job"})
        self.bridge = Mock()
        self.bridge.status.return_value = {"bridge_connected": True}
        self.bridge.history_transport.return_value = [
            {"entity_id": "sensor.temperature", "state": "20", "attributes": {"unit_of_measurement": "°C"}},
            {"entity_id": "binary_sensor.door", "state": "off"},
        ]
        runtime = ModuleType("app.services.runtime")
        runtime.state, runtime.home_assistant_bridge, runtime.history_candidates = self.service, self.bridge, self.candidates
        spec = importlib.util.spec_from_file_location("isolated_history_selection_api", Path(__file__).parents[2] / "backend/app/api/history.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"app.services.runtime": runtime}):
            spec.loader.exec_module(module)
        api = FastAPI()
        api.include_router(module.router)
        self.client = TestClient(api)
        self.addCleanup(self.client.close)

    def test_csv_upload_and_saved_preview_do_not_replace_history_or_start_preparation(self):
        body = b"entity_id,state,last_changed\nsensor.a,20,2026-01-01T00:00:00Z\nsensor.b,21,2026-01-01T00:01:00Z\n"
        response = self.client.post("/history/selection/csv", content=iter([body[:40], body[40:]]),
                                    headers={"Content-Type": "text/csv", "X-Filename": "house.csv"})
        self.assertEqual(response.status_code, 200, response.text)
        candidate = response.json()["candidate_id"]
        self.assertEqual(response.json()["entity_count"], 2)
        self.service._crear_job_carga.assert_not_called()
        self.assertEqual(self.service.real_history_path.read_text(), self.original)
        saved = self.client.post("/history/selection/saved")
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["entities"][0]["entity_id"], "sensor.previous")
        self.service._crear_job_carga.assert_not_called()
        rejected = self.client.post(f"/history/selection/{candidate}/prepare", json={"entity_ids": []})
        self.assertEqual(rejected.status_code, 422)
        self.service._crear_job_carga.assert_not_called()
        started = self.client.post(f"/history/selection/{candidate}/prepare", json={"entity_ids": ["sensor.b"]})
        self.assertEqual(started.status_code, 200, started.text)
        self.service._crear_job_carga.assert_called_once()
        self.service._iniciar_hilo_carga.call_args.kwargs["prepare"](Mock())
        self.assertNotIn("sensor.a", self.service.real_history_path.read_text())
        self.assertIn("sensor.b", self.service.real_history_path.read_text())

    def test_home_assistant_catalog_is_read_only_and_only_explicit_selection_starts_import(self):
        payload = {"connection_mode": "integration", "start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"}
        preview = self.client.post("/history/home-assistant/entities", json=payload)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()["entity_count"], 2)
        self.bridge.history_transport.assert_called_once_with("/states")
        self.service.iniciar_importacion_home_assistant.assert_not_called()
        rejected = self.client.post("/history/home-assistant/prepare", json=payload)
        self.assertEqual(rejected.status_code, 400)
        self.service.iniciar_importacion_home_assistant.assert_not_called()
        response = self.client.post("/history/home-assistant/prepare", json={**payload, "entity_ids": ["sensor.temperature"]})
        self.assertEqual(response.status_code, 200, response.text)
        source = self.service.iniciar_importacion_home_assistant.call_args.args[0]
        self.assertEqual(source.entity_ids, ["sensor.temperature"])
        self.assertEqual(self.service.real_history_path.read_text(), self.original)

    def test_invalid_csv_is_rejected_and_does_not_leave_staged_files(self):
        response = self.client.post("/history/selection/csv", content=b"bad,columns\n1,2\n")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(list(self.candidates.directory.iterdir()), [])
        self.assertEqual(self.service.real_history_path.read_text(), self.original)
        self.service._crear_job_carga.assert_not_called()


if __name__ == "__main__":
    unittest.main()
