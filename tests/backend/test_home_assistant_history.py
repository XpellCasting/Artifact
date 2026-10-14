import csv
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from app.api.schemas import HomeAssistantHistoryRequest
from app.services.home_assistant_history import HomeAssistantHistorySource, NoRedirect
from app.services.history import ServicesHistoryMixin
from app.services.historial.cargar_historial import CargadorHistorial


class HomeAssistantHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.csv"
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.end = datetime(2026, 1, 3, tzinfo=timezone.utc)

    def source(self, **overrides):
        return HomeAssistantHistorySource(**{
            "url": "http://homeassistant.local:8123", "token": "private-token",
            "start": self.start, "end": self.end, "entity_ids": ["sensor.temperature"], **overrides,
        })

    def test_daily_import_preserves_metadata_and_deduplicates_boundary_states(self):
        first = {"entity_id": "sensor.temperature", "state": "21", "last_changed": "2026-01-01T08:00:00Z",
                 "attributes": {"unit_of_measurement": "°C", "device_class": "temperature"}}
        boundary = {"entity_id": "sensor.temperature", "state": "22", "last_changed": "2026-01-02T00:00:00Z",
                    "attributes": first["attributes"]}
        last = {"state": "23", "last_changed": "2026-01-02T08:00:00Z"}
        source = self.source()
        reports = []
        with patch.object(source, "_get", side_effect=[[[first, boundary]], [[boundary, last]]]) as get:
            size = source.export_csv(self.path, lambda *args: reports.append(args))
        with self.path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(size, 0)
        self.assertEqual([row["state"] for row in rows], ["21", "22", "23"])
        self.assertTrue(all(row["unit_of_measurement"] == "°C" for row in rows))
        self.assertTrue(all(row["history_origin"] == "home_assistant" for row in rows))
        self.assertIn("minimal_response=true", get.call_args.args[0])
        self.assertIn("skip_initial_state=true", get.call_args.args[0])
        self.assertIn("filter_entity_id=sensor.temperature", get.call_args.args[0])
        self.assertEqual([r[1] for r in reports], sorted(r[1] for r in reports))
        loaded = CargadorHistorial().cargar_real(self.path)
        self.assertEqual(loaded.metadatos["source_type"], "real")
        self.assertEqual(loaded.metadatos["history_origin"], "home_assistant")

    def test_auto_discovery_filters_compatible_domains_and_empty_period_is_rejected(self):
        source = self.source(entity_ids=[])
        with patch.object(source, "_get", side_effect=[[
            {"entity_id": "sensor.temperature"}, {"entity_id": "binary_sensor.door"}, {"entity_id": "light.kitchen"},
        ], [], []]) as get:
            with self.assertRaisesRegex(ValueError, "No hay lecturas"):
                source.export_csv(self.path, lambda *args: None)
        self.assertEqual(get.call_args_list[0].args[0], "/states")
        self.assertIn("binary_sensor.door%2Csensor.temperature", get.call_args_list[1].args[0])
        self.assertNotIn("light.kitchen", str(get.call_args_list))

    def test_explicit_entities_can_import_removed_entities_without_states(self):
        source = self.source()
        event = {"entity_id": "sensor.temperature", "state": "20", "last_changed": "2026-01-01T12:00:00Z"}
        with patch.object(source, "_get", return_value=[[event]]) as get:
            source.export_csv(self.path, lambda *args: None)
        self.assertTrue(all("/history/period/" in call.args[0] for call in get.call_args_list))

    def test_invalid_urls_dates_and_entities_are_rejected(self):
        for changes in ({"url": "file:///private"}, {"url": "http://user:password@host"},
                        {"url": "http://host/?token=private"}, {"token": ""}, {"token": "bad\ntoken"},
                        {"end": self.start}, {"start": self.start.replace(tzinfo=None)},
                        {"entity_ids": ["light.kitchen"]}, {"entity_ids": ["sensor.x&token=bad"]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.source(**changes)

    def test_credentials_are_redacted_and_schema_requires_timezone(self):
        self.assertNotIn("private-token", repr(self.source()))
        request = HomeAssistantHistoryRequest(token="private-token", start=self.start, end=self.end)
        self.assertNotIn("private-token", repr(request))
        with self.assertRaises(ValueError):
            HomeAssistantHistoryRequest(start="2026-01-01T00:00", end=self.end)

    def test_http_errors_are_actionable_without_returning_credentials(self):
        source = self.source()
        for code, message in ((401, "rechazó el token"), (404, "API de historial"), (500, "HTTP 500")):
            opener = Mock()
            opener.open.side_effect = HTTPError("http://private", code, "private-token", {}, None)
            with patch("app.services.home_assistant_history.build_opener", return_value=opener):
                with self.assertRaisesRegex(ValueError, message) as raised:
                    source._get("/states")
                self.assertNotIn("private-token", str(raised.exception))
        opener.open.side_effect = URLError("private-token")
        with patch("app.services.home_assistant_history.build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "No se pudo conectar"):
                source._get("/states")
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "http://another-host"))

    def test_failed_download_preserves_existing_csv_and_removes_temporary_file(self):
        self.path.write_text("existing history", encoding="utf-8")
        state = ServicesHistoryMixin()
        state.real_history_path = self.path
        state._crear_job_carga = Mock(return_value="job")
        state._iniciar_hilo_carga = Mock()
        state.estado_carga_real = Mock(return_value={"job_id": "job"})
        source = Mock()
        def fail(destination, report):
            destination.write_text("partial history", encoding="utf-8")
            raise ValueError("Home Assistant no disponible")
        source.export_csv.side_effect = fail
        state.iniciar_importacion_home_assistant(source)
        prepare = state._iniciar_hilo_carga.call_args.kwargs["prepare"]
        with self.assertRaises(ValueError):
            prepare(Mock())
        self.assertEqual(self.path.read_text(), "existing history")
        self.assertEqual(list(self.path.parent.glob("*.importing")), [])

    def test_csv_without_origin_remains_a_csv_reference(self):
        self.path.write_text("entity_id,state,last_changed\nsensor.temperature,20,2026-01-01T00:00:00Z\n")
        self.assertEqual(CargadorHistorial().cargar_real(self.path).metadatos["history_origin"], "csv")


if __name__ == "__main__":
    unittest.main()
