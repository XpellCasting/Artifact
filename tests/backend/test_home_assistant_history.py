import csv
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.api.schemas import HomeAssistantHistoryRequest
from app.services.home_assistant_history import HomeAssistantHistorySource
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
            "transport": Mock(),
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

    def test_empty_selection_never_discovers_or_downloads_history(self):
        source = self.source(entity_ids=[])
        with self.assertRaisesRegex(ValueError, "Activa al menos"):
            source.export_csv(self.path, lambda *args: None)
        source.transport.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_explicit_entities_can_import_removed_entities_without_states(self):
        source = self.source()
        event = {"entity_id": "sensor.temperature", "state": "20", "last_changed": "2026-01-01T12:00:00Z"}
        with patch.object(source, "_get", return_value=[[event]]) as get:
            source.export_csv(self.path, lambda *args: None)
        self.assertTrue(all("/history/period/" in call.args[0] for call in get.call_args_list))

    def test_invalid_dates_entities_and_missing_integration_are_rejected(self):
        for changes in ({"transport": None}, {"end": self.start},
                        {"start": self.start.replace(tzinfo=None)},
                        {"entity_ids": ["light.kitchen"]}, {"entity_ids": ["sensor.x&token=bad"]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.source(**changes)

    def test_schema_rejects_direct_connection_fields_and_requires_timezone(self):
        for field, value in (("url", "http://ha.local"), ("token", "private-token"), ("connection_mode", "integration")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                HomeAssistantHistoryRequest(start=self.start, end=self.end, **{field: value})
        with self.assertRaises(ValueError):
            HomeAssistantHistoryRequest(start="2026-01-01T00:00", end=self.end)

    def test_catalog_preserves_area_metadata_and_legacy_catalog_is_explicitly_unknown(self):
        rows = [{"entity_id": "sensor.temperature", "state": "20", "area_id": "kitchen", "area_name": "Cocina"},
                {"entity_id": "binary_sensor.door", "state": "off", "area_id": None, "area_name": None},
                {"entity_id": "light.room", "state": "on", "area_id": None, "area_name": None}]
        catalog = Mock(return_value={"entities": rows, "areas_available": True})
        source = self.source(catalog_transport=catalog)
        preview = source.discover_entities()
        self.assertTrue(preview["areas_available"])
        self.assertEqual(preview["entity_count"], 2)
        self.assertEqual(preview["entities"][1]["area_name"], "Cocina")
        self.assertIsNone(preview["entities"][0]["area_id"])
        source.transport.assert_not_called()
        catalog.return_value = {"entities": rows, "areas_available": False}
        preview = source.discover_entities()
        self.assertFalse(preview["areas_available"])
        self.assertTrue(all("area_id" not in row for row in preview["entities"]))
        catalog.return_value = {"entities": [{"entity_id": "sensor.a", "state": "1"}], "areas_available": True}
        self.assertFalse(source.discover_entities()["areas_available"])

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
