"""The confirmed HA selection must be the only input to both reference models."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from app.modulos.estadistico.filtro import FiltroEstadistico
from app.services.aplicacion import EstadoAplicacion
from app.services.historial.cargar_historial import CargadorHistorial
from app.services.home_assistant_history import HomeAssistantHistorySource
from app.services.history_selection import selection_confirmed


class HomeAssistantReferenceTests(unittest.TestCase):
    def test_only_selected_entities_form_one_statistical_and_contextual_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = EstadoAplicacion(CargadorHistorial(), FiltroEstadistico(),
                                     root / "static", root / "history.csv", root / "cache" / "analysis.json")
            selected = {"binary_sensor.kitchen", "binary_sensor.living"}
            groups = [[{"entity_id": entity_id, "state": "on" if index % 4 < 2 else "off",
                        "last_changed": at.isoformat()}
                       for index, at in enumerate(pd.date_range("2026-01-01", periods=96, freq="30min", tz="UTC"))]
                      for entity_id in [*sorted(selected), "binary_sensor.excluded"]]

            def history(path):
                query = parse_qs(urlsplit(path).query)
                self.assertEqual(set(query["filter_entity_id"][0].split(",")), selected)
                # Even a recorder returning an extra entity cannot widen the selection.
                return groups

            source = HomeAssistantHistorySource(datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 3, tzinfo=timezone.utc), sorted(selected), transport=Mock(side_effect=history))
            with patch.object(state, "_iniciar_hilo_carga") as start:
                job = state.iniciar_importacion_home_assistant(source)
            source.transport.assert_not_called()
            self.assertIsNone(state.historial)
            state._ejecutar_job_carga(job["job_id"], prepare=start.call_args.kwargs["prepare"])
            result = state.estado_carga_real(job["job_id"])
            self.assertEqual(result["status"], "completed", result.get("error"))
            self.assertEqual(set(state.historial.entidades.entity_id), selected)
            self.assertEqual(set(state.historial.eventos.entity_id), selected)
            self.assertEqual(set(state.future_baseline_template["binary"]), selected)
            context = state.future_context_map_template
            self.assertEqual({node["entity_id"] for node in context["nodes"]}, selected)
            self.assertTrue(any({edge["source"], edge["target"]} == selected for edge in context["edges"]))
            self.assertEqual(set(state.future_context_events_template.entity_id), selected)
            self.assertTrue(selection_confirmed(state.real_history_path))
            self.assertNotIn("binary_sensor.excluded", state.real_history_path.read_text())


if __name__ == "__main__":
    unittest.main()
