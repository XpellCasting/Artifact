import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
import time
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.Cargar_Historial.Cargar_Historial import CargadorHistorial  # noqa: E402
from app.Filtro_Estadistico.Filtro import FiltroEstadistico  # noqa: E402
from app.main import (  # noqa: E402
    AnalyzeRequest,
    DEFAULT_SUSPICION_THRESHOLD,
    EstadoAplicacion,
    ValidationReplayAdvanceRequest,
)


class ValidationReplayControlTests(unittest.TestCase):
    def crear_estado(self, base: Path) -> EstadoAplicacion:
        return EstadoAplicacion(
            cargador=CargadorHistorial(),
            filtro_estadistico=FiltroEstadistico(),
            static_history_dir=base / "static",
            real_history_path=base / "history.csv",
            analysis_cache_path=base / "cache" / "analysis.json",
        )

    def test_umbral_de_sospecha_tiene_default_y_limites_en_la_api(self):
        self.assertEqual(
            AnalyzeRequest().suspicion_threshold,
            DEFAULT_SUSPICION_THRESHOLD,
        )
        self.assertEqual(AnalyzeRequest(suspicion_threshold=.731).suspicion_threshold, .731)
        self.assertEqual(
            ValidationReplayAdvanceRequest(suspicion_threshold=.01).suspicion_threshold,
            .01,
        )
        for value in [0, 1, -0.1, 1.1]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                AnalyzeRequest(suspicion_threshold=value)

    def test_status_es_acotado_y_resultados_se_consultan_paginados(self):
        with tempfile.TemporaryDirectory() as tmp:
            estado = self.crear_estado(Path(tmp))
            resultados = [
                {
                    "event_id": f"sensor.test@{index}",
                    "entity_id": "sensor.test",
                    "timestamp": f"2026-06-01T00:{index % 60:02d}:00+00:00",
                    "payload": "x" * 2000,
                }
                for index in range(1000)
            ]
            estado.validation_results = resultados
            estado.validation_contextual_results = [
                {
                    **item,
                    "context_outcome": "insufficient_coverage",
                    "routing": {"terminal": False, "send_to_classifier": True},
                }
                for item in resultados
            ]
            estado.validation_last_results = resultados[-2:]
            estado.validation_last_contextual_results = (
                estado.validation_contextual_results[-2:]
            )
            estado.validation_replay_state = {
                "status": "paused",
                "rows": 10000,
                "cursor": 1000,
                "detection_count": 1000,
                "contextual_count": 1000,
                "environmental_count": 0,
                "pending_classification_count": 1000,
                "contextual_outcomes": {"insufficient_coverage": 1000},
                "detection_delay_sum_seconds": 10.0,
            }

            status = estado.estado_reproduccion_validacion()
            pagina = estado.resultados_reproduccion_validacion(
                "statistical", offset=10, limit=25
            )

            self.assertLess(len(json.dumps(status)), 30000)
            self.assertEqual(len(status["results"]), 2)
            self.assertEqual(pagina["total"], 1000)
            self.assertEqual(len(pagina["items"]), 25)
            self.assertEqual(pagina["next_offset"], 35)

    def test_worker_se_pausa_entre_lotes_y_puede_reanudarse(self):
        with tempfile.TemporaryDirectory() as tmp:
            estado = self.crear_estado(Path(tmp))
            timestamps = pd.date_range(
                "2026-06-01T00:00:00Z", periods=40, freq="1min"
            )
            eventos = pd.DataFrame(
                {
                    "entity_id": ["sensor.test"] * len(timestamps),
                    "timestamp_dt": timestamps,
                    "_replay_index": range(len(timestamps)),
                }
            )
            estado.historial = Mock(entidades=pd.DataFrame())
            estado.historial_validacion = Mock(eventos=eventos)
            estado.eventos_validacion_ordenados = eventos
            estado.validation_baseline = {}
            estado.validation_context_map = {}
            estado.validation_context_events = eventos.iloc[:0].copy()
            estado.validation_replay_state = {
                "status": "ready",
                "rows": len(eventos),
                "cursor": 0,
                "start": timestamps[0].isoformat(),
                "current_timestamp": (timestamps[0] - pd.Timedelta(minutes=1)).isoformat(),
                "processing_elapsed_seconds": 0.0,
                "contextual_elapsed_ms": 0.0,
                "detection_count": 0,
                "contextual_count": 0,
                "environmental_count": 0,
                "pending_classification_count": 0,
                "contextual_outcomes": {},
                "possible_common_cause": 0,
                "detection_delay_sum_seconds": 0.0,
                "max_detection_delay_seconds": None,
            }
            estado.filtro_contextual.analizar_causal = Mock(
                return_value={"results": [], "summary": {}}
            )

            def lote_lento(**_kwargs):
                time.sleep(0.015)
                return {"results": [], "summary": {}}

            with patch(
                "app.main.MotorReproduccionValidacion.analizar_lote",
                side_effect=lote_lento,
            ):
                estado.iniciar_reproduccion_validacion(1, 0.73)
                limite = time.monotonic() + 2
                while (
                    estado.validation_replay_state.get("cursor", 0) < 1
                    and time.monotonic() < limite
                ):
                    time.sleep(0.005)

                pausado = estado.pausar_reproduccion_validacion()
                cursor_pausado = pausado["cursor"]
                self.assertEqual(pausado["status"], "paused")
                self.assertGreaterEqual(cursor_pausado, 1)
                self.assertLess(cursor_pausado, len(eventos))

                time.sleep(0.05)
                self.assertEqual(
                    estado.validation_replay_state["cursor"], cursor_pausado
                )

                estado.iniciar_reproduccion_validacion(5, 0.25)
                limite = time.monotonic() + 3
                while (
                    estado.validation_replay_state.get("status") != "completed"
                    and time.monotonic() < limite
                ):
                    time.sleep(0.01)

            final = estado.estado_reproduccion_validacion()
            self.assertEqual(final["status"], "completed")
            self.assertEqual(final["cursor"], len(eventos))
            self.assertEqual(final["suspicion_threshold"], 0.73)

    def test_resultados_antiguos_salen_de_ram_y_siguen_paginables(self):
        with tempfile.TemporaryDirectory() as tmp:
            estado = self.crear_estado(Path(tmp))
            estado.validation_replay_state = {
                "status": "paused",
                "rows": 2000,
                "cursor": 1200,
            }
            estado._reiniciar_almacen_resultados_validacion()
            resultados = [
                {
                    "event_id": f"sensor.test@{index}",
                    "entity_id": "sensor.test",
                    "timestamp": f"2026-06-01T00:00:{index % 60:02d}+00:00",
                }
                for index in range(1200)
            ]

            estado._guardar_resultados_validacion(
                resultados,
                [],
                {"environmental_results": [], "pending_classification_results": []},
            )
            estado.validation_results.extend(resultados)
            pagina_antigua = estado.resultados_reproduccion_validacion(
                "statistical", offset=900, limit=25
            )

            self.assertEqual(len(estado.validation_results), 500)
            self.assertEqual(pagina_antigua["total"], 1200)
            self.assertEqual(len(pagina_antigua["items"]), 25)
            self.assertEqual(
                pagina_antigua["items"][0]["event_id"], "sensor.test@299"
            )

    def test_status_usa_snapshot_y_no_espera_al_lote_activo(self):
        with tempfile.TemporaryDirectory() as tmp:
            estado = self.crear_estado(Path(tmp))
            estado.validation_replay_state = {
                "status": "running",
                "rows": 100,
                "cursor": 10,
            }
            estado._publicar_snapshot_reproduccion()
            respuestas = []
            estado.validation_replay_lock.acquire()
            try:
                lector = Thread(
                    target=lambda: respuestas.append(
                        estado.estado_reproduccion_validacion()
                    )
                )
                lector.start()
                lector.join(timeout=0.2)
                self.assertFalse(lector.is_alive())
            finally:
                estado.validation_replay_lock.release()
            self.assertEqual(respuestas[0]["cursor"], 10)


if __name__ == "__main__":
    unittest.main()
