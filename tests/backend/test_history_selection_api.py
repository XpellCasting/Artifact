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
        self.bridge.status.return_value = {"bridge_configured": True, "bridge_connected": True}
        self.bridge.entity_catalog.return_value = {"areas_available": False, "entities": [
            {"entity_id": "sensor.temperature", "state": "20", "attributes": {"unit_of_measurement": "°C"}},
            {"entity_id": "binary_sensor.door", "state": "off"},
        ]}
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
        payload = {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"}
        preview = self.client.post("/history/home-assistant/entities", json=payload)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()["entity_count"], 2)
        self.bridge.entity_catalog.assert_called_once_with()
        self.bridge.history_transport.assert_not_called()
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

    def test_import_and_prepare_reject_empty_selection_without_creating_jobs(self):
        payload = {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z", "entity_ids": []}
        for path in ("prepare", "import/start"):
            response = self.client.post(f"/history/home-assistant/{path}", json=payload)
            self.assertEqual(response.status_code, 400, response.text)
        self.service.iniciar_importacion_home_assistant.assert_not_called()
        self.bridge.entity_catalog.assert_not_called()

    def test_old_credentials_are_rejected_and_config_contains_only_link_status(self):
        self.assertEqual(self.client.get("/history/home-assistant/config").json(),
                         {"bridge_configured": True, "bridge_connected": True})
        payload = {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z", "entity_ids": ["sensor.a"]}
        for field, value in (("url", "http://ha.local"), ("token", "secret"), ("connection_mode", "integration")):
            for path in ("entities", "prepare", "import/start"):
                with self.subTest(field=field, path=path):
                    response = self.client.post(f"/history/home-assistant/{path}", json={**payload, field: value})
                    self.assertEqual(response.status_code, 422, response.text)
        self.service.iniciar_importacion_home_assistant.assert_not_called()
        self.bridge.entity_catalog.assert_not_called()

    def test_disconnect_blocks_catalog_and_import_until_reconnected(self):
        payload = {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z", "entity_ids": ["sensor.temperature"]}
        for message in ("Integración no configurada", "La integración está desconectada"):
            self.bridge.require_connected.side_effect = ValueError(message)
            for path in ("entities", "prepare", "import/start"):
                response = self.client.post(f"/history/home-assistant/{path}", json=payload)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(message, response.json()["detail"])
        self.service.iniciar_importacion_home_assistant.assert_not_called()
        self.bridge.entity_catalog.assert_not_called()
        self.bridge.require_connected.side_effect = None
        self.assertEqual(self.client.post("/history/home-assistant/entities", json=payload).status_code, 200)
        self.assertEqual(self.client.post("/history/home-assistant/prepare", json=payload).status_code, 200)

    def test_catalog_returns_assigned_and_unassigned_metadata(self):
        self.bridge.entity_catalog.return_value = {"areas_available": True, "entities": [
            {"entity_id": "sensor.temperature", "state": "20", "area_id": "kitchen", "area_name": "Cocina"},
            {"entity_id": "binary_sensor.door", "state": "off", "area_id": None, "area_name": None}]}
        response = self.client.post("/history/home-assistant/entities", json={
            "start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["areas_available"])
        self.assertEqual(response.json()["entities"][1]["area_name"], "Cocina")
        self.assertIsNone(response.json()["entities"][0]["area_id"])
        self.service.iniciar_importacion_home_assistant.assert_not_called()


if __name__ == "__main__":
    unittest.main()
