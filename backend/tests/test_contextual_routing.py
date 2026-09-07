from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from app.Cargar_Historial.Cargar_Historial import CargadorHistorial, ResultadoHistorial
from app.Filtro_Contextual import FiltroContextualEspacioTemporal
from app.Filtro_Estadistico.Filtro import FiltroEstadistico
from app.main import EstadoAplicacion, app
from test_contextual_filter import construir_historial


def evaluar_apoyo(support, *, stale=False, concurrentes=None):
    filtro = FiltroContextualEspacioTemporal()
    timestamp = pd.Timestamp("2026-01-06T00:00:00Z")
    return filtro._evaluar_sospecha(
        sospecha={
            "entity_id": "sensor.a", "timestamp": timestamp.isoformat(),
            "suspicion_score": 0.8, "is_suspicious": True,
        },
        signals={
            "sensor.a": pd.Series([1.0], index=[timestamp]),
            "sensor.b": pd.Series([2.0 * support], index=[timestamp]),
        },
        stats={name: {"center": 0.0, "scale": 1.0} for name in ["sensor.a", "sensor.b"]},
        edges=[{
            "target": "sensor.b", "association_signed": 1.0,
            "stability_score": 1.0, "lag_seconds": 0,
        }],
        map_quality={"stale": stale},
        concurrent_entities=concurrentes or ["sensor.a"],
    )


class CompuertaContextualTest(unittest.TestCase):
    def test_umbral_unico_cierra_tambien_apoyo_entre_035_y_050(self):
        for support in [0.35, 0.4, 0.5, 1.0]:
            with self.subTest(support=support):
                result = evaluar_apoyo(support)
                self.assertEqual(result["context_outcome"], "supported")
                self.assertEqual(result["contextual_class"], "environmental_anomaly")
                self.assertFalse(result["routing"]["send_to_classifier"])
                self.assertEqual(result["alert"]["level"], "informational")

    def test_apoyo_bajo_o_contradiccion_no_cierran_evento(self):
        for support in [0.349, 0.0, -0.35, -1.0]:
            with self.subTest(support=support):
                result = evaluar_apoyo(support)
                self.assertFalse(result["routing"]["terminal"])
                self.assertTrue(result["routing"]["send_to_classifier"])
                self.assertIsNone(result["alert"])
                self.assertIsNone(result["contextual_class"])

    def test_mapa_obsoleto_o_desconocido_no_cierra_ambiental(self):
        for stale in [True, None]:
            result = evaluar_apoyo(1.0, stale=stale)
            self.assertEqual(result["context_outcome"], "supported")
            self.assertTrue(result["routing"]["send_to_classifier"])
            self.assertEqual(result["routing"]["reason"], "stale_or_unknown_context_map")
            self.assertIsNone(result["alert"])

    def test_coincidencia_sin_evidencia_causal_no_veta_apoyo(self):
        result = evaluar_apoyo(1.0, concurrentes=["sensor.a", "sensor.b", "sensor.c"])
        self.assertEqual(result["context_outcome"], "supported")
        self.assertEqual(result["routing"]["reason"], "context_supported")
        self.assertTrue(result["routing"]["terminal"])
        self.assertTrue(result["common_cause"]["contextual_coincidence"])
        self.assertFalse(result["common_cause"]["candidate"])
        self.assertLess(result["adjusted_suspicion_score"], result["statistical_score"])


class PipelineSinClasificadorTest(unittest.TestCase):
    # Nombre conservado: las pruebas de agrupación reutilizan esta fixture.
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.estado = EstadoAplicacion(
            cargador=CargadorHistorial(), filtro_estadistico=FiltroEstadistico(),
            static_history_dir=base / "static", real_history_path=base / "history.csv",
            analysis_cache_path=base / "cache.json",
        )
        self.history, self.entities = construir_historial()
        self.mapa = self.estado.filtro_contextual.construir_mapa(self.history, self.entities)
        self.future = []
        self.sospechas = []
        for hour, humidity in [(0, 16.0), (1, -16.0)]:
            timestamp = pd.Timestamp("2026-01-06T00:05:00Z") + pd.Timedelta(hours=hour)
            for entity, kind, value, state in [
                ("sensor.temperature", "continuous", 8.0, "8.0"),
                ("sensor.humidity", "continuous", humidity, str(humidity)),
                ("binary_sensor.climate", "binary", 1.0, "on"),
            ]:
                self.future.append({
                    "entity_id": entity, "sensor_kind": kind, "value": value,
                    "state": state, "timestamp": timestamp.isoformat(),
                    "timestamp_dt": timestamp, "is_valid_value": True,
                })
            self.sospechas.append({
                "entity_id": "sensor.temperature", "timestamp": timestamp.isoformat(),
                "sensor_kind": "continuous", "value": 8.0,
                "suspicion_score": 0.8, "is_suspicious": True,
            })
        self.future = pd.DataFrame(self.future)
        self.eventos = pd.concat([self.history, self.future], ignore_index=True)
        self.estado.historial = ResultadoHistorial(self.eventos, self.entities)
        self.estado._asegurar_mapa_contextual = Mock(return_value=self.mapa)

    def assert_rutas(self, payload):
        self.assertEqual(payload["environmental_count"], 1)
        self.assertEqual(payload["pending_classification_count"], 1)
        self.assertEqual(payload["pipeline_summary"]["classifier"], 1)
        self.assertEqual(payload["pipeline_summary"]["classifier_status"], "catalogue")
        self.assertNotIn("classification", payload)
        self.assertNotIn("classification_results", payload)
        ambient = payload["environmental_results"][0]
        pending = payload["pending_classification_results"][0]
        self.assertFalse(ambient["routing"]["send_to_classifier"])
        self.assertEqual(pending["classification_status"], "indeterminate")
        self.assertEqual(len(payload["classifier_results"]), 1)
        self.assertNotIn("diagnosis", ambient)
        self.assertNotEqual(ambient["event_id"], pending["event_id"])

    def test_historico_y_reanalisis_contextual_respetan_terminal_ambiental(self):
        self.assertFalse(hasattr(self.estado, "filtro_clasificacion"))
        # results es una vista recortada; all_results contiene ambas sospechas.
        self.estado.filtro_estadistico.analizar_historial = Mock(return_value={
            "results": self.sospechas[:1], "all_results": self.sospechas, "summary": {},
        })
        self.assert_rutas(self.estado.analizar_historial(0.6))
        self.assert_rutas(self.estado.contexto_actual())
        self.assert_rutas(self.estado.analizar_contexto_actual())

    def test_replay_causal_cierra_ambiental_y_preserva_pendiente(self):
        self.estado.historial_validacion = ResultadoHistorial(self.future, self.entities)
        self.estado.eventos_validacion_ordenados = self.future
        self.estado.validation_baseline = {}
        self.estado.validation_context_map = self.mapa
        self.estado.validation_context_events = self.history
        self.estado.validation_replay_state = {
            "rows": len(self.future), "cursor": 0, "status": "ready",
            "current_timestamp": self.history["timestamp_dt"].max().isoformat(),
        }
        with patch("app.main.MotorReproduccionValidacion.analizar_lote", return_value={
            "results": self.sospechas, "summary": {},
        }):
            payload = self.estado.avanzar_reproduccion_validacion(100, 0.6)
        self.assert_rutas(payload)
        self.assertTrue(payload["environmental_results"][0]["causal_runtime"]["causal"])
        self.assertEqual(len(payload["new_environmental_results"]), 1)
        self.assertEqual(len(payload["new_pending_classification_results"]), 1)
        self.assertEqual(payload["last_batch"]["classified"], 1)
        self.assert_rutas(self.estado.estado_reproduccion_validacion())
        self.estado.validation_replay_state["contextual_coincidences"] = 3
        self.estado.validation_context_concurrence[("sensor.a", "abrupt_change")] = pd.Timestamp("2026-01-06T00:00:00Z")
        restarted = self.estado.reiniciar_reproduccion_validacion()
        self.assertEqual(restarted["environmental_count"], 0)
        self.assertEqual(restarted["pending_classification_count"], 0)
        self.assertEqual(restarted["contextual_summary"]["contextual_coincidences"], 0)
        self.assertEqual(self.estado.validation_context_concurrence, {})

    def test_sin_contexto_no_se_pierde_evento_y_normal_no_avanza(self):
        payload = self.estado._salidas_pipeline([
            *self.sospechas, {"entity_id": "sensor.normal", "is_suspicious": False},
        ], [])
        self.assertEqual(payload["pending_classification_count"], 2)
        self.assertEqual(payload["environmental_count"], 0)
        self.assertTrue(all(item["routing"]["reason"] == "missing_contextual_result"
                            for item in payload["pending_classification_results"]))

    def test_join_normaliza_timestamp_y_lote_ambiental_no_crea_pendientes(self):
        contextual = evaluar_apoyo(1.0)
        statistical = {"entity_id": "sensor.a", "timestamp": "2026-01-05T21:00:00-03:00",
                       "is_suspicious": True}
        candidates, contexts = [], []
        for index in range(1200):
            entity = f"sensor.test_{index}"
            candidates.append({**statistical, "entity_id": entity})
            contexts.append({**deepcopy(contextual), "entity_id": entity,
                             "event_id": f"{entity}@{contextual['timestamp']}"})
        payload = self.estado._salidas_pipeline(candidates, contexts, limit=500)
        self.assertEqual(payload["environmental_count"], 1200)
        self.assertEqual(len(payload["environmental_results"]), 500)
        self.assertEqual(payload["pending_classification_count"], 0)
        self.assertEqual(payload["pipeline_summary"]["classifier"], 0)

    def test_endpoints_antiguos_del_clasificador_no_estan_registrados(self):
        self.assertFalse(any(route.path.startswith("/classifier/") for route in app.routes))


if __name__ == "__main__":
    unittest.main()
