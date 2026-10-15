import csv
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.api.schemas import HistorySelectionRequest, HomeAssistantHistoryRequest
from app.services.history_selection import HistoryCandidates, confirm_selection, selection_confirmed
from threading import Lock
from app.services.history import ServicesHistoryMixin
from app.services.historial.cargar_historial import CargadorHistorial
from app.services.home_assistant_history import HomeAssistantHistorySource


class HistorySelectionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.directory = Path(self.folder.name)
        self.candidates = HistoryCandidates(self.directory / "candidates")

    def candidate(self, rows):
        identifier, path = self.candidates.create()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["entity_id", "state", "last_changed", "unit_of_measurement"])
            writer.writeheader()
            writer.writerows(rows)
        return identifier, path

    def test_5000_sensor_catalog_does_not_load_or_prepare_filters(self):
        identifier, path = self.candidate([{"entity_id": f"sensor.temperature_{i}", "state": "20",
            "last_changed": "2026-01-01T00:00:00Z", "unit_of_measurement": "°C"} for i in range(5000)])
        original = path.read_bytes()
        with patch.object(CargadorHistorial, "cargar_real", side_effect=AssertionError("Preview must not prepare history")):
            preview = self.candidates.inspect(identifier, "house.csv")
        self.assertEqual(preview["entity_count"], 5000)
        self.assertEqual(preview["rows"], 5000)
        self.assertTrue(all(entity["sensor_kind"] == "continuous" for entity in preview["entities"]))
        self.assertEqual(path.read_bytes(), original)
        self.assertNotIn("_states", preview["entities"][0])

    def test_only_confirmed_sensors_replace_active_csv_and_reach_the_loader(self):
        identifier, _ = self.candidate([{"entity_id": entity, "state": "20",
            "last_changed": "2026-01-01T00:00:00Z"} for entity in ["sensor.a", "sensor.b", "sensor.c"]])
        self.candidates.inspect(identifier, "house.csv")
        active = self.directory / "active.csv"
        active.write_text("previous active history", encoding="utf-8")
        service = ServicesHistoryMixin()
        service.real_history_path = active
        service._crear_job_carga = Mock(return_value="selected-job")
        service._iniciar_hilo_carga = Mock()
        service.estado_carga_real = Mock(return_value={"job_id": "selected-job"})
        service.iniciar_seleccion_csv(self.candidates, identifier, ["sensor.b"])
        self.assertEqual(active.read_text(), "previous active history")
        prepare = service._iniciar_hilo_carga.call_args.kwargs["prepare"]
        prepare(Mock())
        loaded = CargadorHistorial().cargar_real(active)
        self.assertEqual(list(loaded.entidades.entity_id), ["sensor.b"])
        self.assertTrue(selection_confirmed(active))
        # The unfiltered candidate remains available for revising the selection.
        self.assertEqual(self.candidates.selected(identifier, ["sensor.c"]), {"sensor.c"})

    def test_empty_unknown_and_expired_selections_are_rejected(self):
        identifier, _ = self.candidate([{"entity_id": "sensor.a", "state": "1", "last_changed": "2026-01-01T00:00:00Z"}])
        preview = self.candidates.inspect(identifier, "house.csv")
        for ids in [[], ["sensor.absent"]]:
            with self.assertRaises(ValueError):
                self.candidates.selected(identifier, ids)
        with patch("app.services.history_selection.time", return_value=preview["created_at"] + 86401):
            with self.assertRaisesRegex(ValueError, "caducó"):
                self.candidates.selected(identifier, ["sensor.a"])
        with self.assertRaises(ValueError):
            self.candidates.path("../active")
        with self.assertRaises(ValueError):
            HistorySelectionRequest(entity_ids=[])

    def test_saved_csv_preview_preserves_the_active_file_and_classifies_full_stream(self):
        active = self.directory / "active.csv"
        active.write_text("entity_id,state,last_changed\nsensor.mixed,20,2026-01-01T00:00:00Z\n"
                          "sensor.mixed,not-numeric,2026-01-01T00:01:00Z\nbinary_sensor.door,on,invalid\n", encoding="utf-8")
        original = active.read_bytes()
        preview = self.candidates.from_saved(active)
        self.assertEqual(preview["invalid_rows"], 1)
        self.assertEqual(preview["entities"][0]["sensor_kind"], "multistate")
        self.assertEqual(active.read_bytes(), original)

    def test_home_assistant_catalog_uses_states_only_and_accepts_5000_selected_ids(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 2, tzinfo=timezone.utc)
        ids = [f"sensor.temperature_{i}" for i in range(5000)]
        request = HomeAssistantHistoryRequest(start=start, end=end, entity_ids=ids)
        self.assertEqual(len(request.entity_ids), 5000)
        source = HomeAssistantHistorySource("http://ha.local", "private-token", start, end, [])
        states = [{"entity_id": key, "state": "20", "attributes": {"friendly_name": key, "unit_of_measurement": "°C"}} for key in ids]
        states.append({"entity_id": "automation.example", "state": "on"})
        with patch.object(source, "_get", return_value=states) as get:
            preview = source.discover_entities()
        get.assert_called_once_with("/states")
        self.assertEqual(preview["entity_count"], 5000)
        self.assertNotIn("private-token", str(preview))

    def test_a_failed_selection_never_replaces_the_active_file(self):
        identifier, _ = self.candidate([{"entity_id": "sensor.a", "state": "1", "last_changed": "2026-01-01T00:00:00Z"}])
        self.candidates.inspect(identifier, "test.csv")
        active = self.directory / "active.csv"
        active.write_text("original", encoding="utf-8")
        service = ServicesHistoryMixin()
        service.real_history_path = active
        service._crear_job_carga = Mock(return_value="job")
        service._iniciar_hilo_carga = Mock()
        service.estado_carga_real = Mock(return_value={"job_id": "job"})
        service.iniciar_seleccion_csv(self.candidates, identifier, ["sensor.a"])
        with patch.object(self.candidates, "export_selected", side_effect=ValueError("failed filtering")):
            with self.assertRaises(ValueError):
                service._iniciar_hilo_carga.call_args.kwargs["prepare"](Mock())
        self.assertEqual(active.read_text(), "original")
        self.assertEqual(list(self.directory.glob("*.selecting")), [])

    def test_saved_unconfirmed_history_never_autoloads_and_confirmation_survives_restart(self):
        active = self.directory / "active.csv"
        active.write_text("entity_id,state,last_changed\nsensor.a,20,2026-01-01T00:00:00Z\n", encoding="utf-8")
        service = ServicesHistoryMixin()
        service.real_history_path = active
        service.static_history_dir = self.directory / "static"
        service.history_load_jobs_lock = Lock()
        service.history_load_jobs = {}
        service.historial = None
        service.cargador = Mock()
        status = service.status(autoload=True)
        self.assertTrue(status["selection_required"])
        service.cargador.cargar_real.assert_not_called()
        confirm_selection(active, ["sensor.a"])
        self.assertTrue(selection_confirmed(active))
        # Another CSV must be reviewed even when it replaces the same active path.
        active.write_text("entity_id,state,last_changed\nsensor.b,21,2026-01-01T00:00:00Z\n", encoding="utf-8")
        self.assertFalse(selection_confirmed(active))


if __name__ == "__main__":
    unittest.main()
