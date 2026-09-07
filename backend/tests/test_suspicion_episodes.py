from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from app.Filtro_Estadistico.Episodios import agrupar_sospechas, enviar_a_contextual, resumen_episodios
from app.main import AnalyzeRequest, ValidationReplayAdvanceRequest
from pydantic import ValidationError
import test_contextual_routing as routing_fixtures


def suspicion(minute, entity="sensor.a", pattern="abrupt_change", score=.8):
    return {"entity_id": entity, "timestamp": (pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(minutes=minute)).isoformat(),
            "sensor_kind": "continuous", "suspected_pattern": pattern,
            "is_suspicious": True, "passed_to_contextual_filter": True, "suspicion_score": score}


class EpisodeTests(unittest.TestCase):
    def test_first_is_immediate_repeats_extend_silence_and_exact_boundary_reopens(self):
        inputs = [suspicion(0), suspicion(5), suspicion(34, score=.95), suspicion(64)]
        original = deepcopy(inputs)
        items, state = agrupar_sospechas(inputs, 1800)
        self.assertEqual(inputs, original)
        self.assertEqual([enviar_a_contextual(item) for item in items], [True, False, False, True])
        self.assertEqual(items[0]["episode"]["occurrences"], 1)
        self.assertEqual(items[2]["episode"]["occurrences"], 3)
        self.assertEqual(items[2]["episode"]["max_score"], .95)
        self.assertEqual(items[2]["episode"]["duration_seconds"], 34 * 60)
        self.assertNotEqual(items[2]["episode"]["episode_id"], items[3]["episode"]["episode_id"])
        self.assertTrue(all(item["is_suspicious"] for item in items))
        self.assertEqual(len(state), 1)

    def test_other_pattern_and_other_entity_are_not_silenced(self):
        items, _ = agrupar_sospechas([suspicion(0), suspicion(1, pattern="missing_gap"), suspicion(2, entity="sensor.b")])
        self.assertTrue(all(enviar_a_contextual(item) for item in items))

    def test_partitioning_and_pause_do_not_change_episode_decisions(self):
        inputs = [suspicion(i) for i in range(100)]
        whole, _ = agrupar_sospechas(inputs)
        state, split = {}, []
        for chunk in [inputs[:17], inputs[17:40], inputs[40:]]:
            before = deepcopy(state)
            values, next_state = agrupar_sospechas(chunk, anteriores=state)
            self.assertEqual(state, before)
            state = next_state
            split.extend(values)
        self.assertEqual(whole, split)
        self.assertEqual(resumen_episodios(split), {"detected": 100, "forwarded": 1, "suppressed": 99})
        self.assertEqual(len(state), 1)

    def test_disabled_and_missing_time_or_pattern_always_forward(self):
        items, state = agrupar_sospechas([suspicion(0), suspicion(1)], 0)
        self.assertTrue(all(enviar_a_contextual(item) for item in items))
        self.assertFalse(state)
        items, _ = agrupar_sospechas([{**suspicion(0), "timestamp": "invalid"}, suspicion(1, pattern="")])
        self.assertTrue(all(enviar_a_contextual(item) for item in items))

    def test_out_of_order_timestamp_cannot_overwrite_more_recent_state(self):
        _, state = agrupar_sospechas([suspicion(10)])
        items, after = agrupar_sospechas([suspicion(5)], anteriores=state)
        self.assertTrue(enviar_a_contextual(items[0]))
        self.assertEqual(after, state)

    def test_api_allows_zero_and_rejects_negative_or_unbounded_period(self):
        for model in [AnalyzeRequest, ValidationReplayAdvanceRequest]:
            self.assertEqual(model().suspicion_quiet_seconds, 1800)
            self.assertEqual(model(suspicion_quiet_seconds=0).suspicion_quiet_seconds, 0)
            for value in [-1, 86401, .1]:
                with self.assertRaises(ValidationError):
                    model(suspicion_quiet_seconds=value)


class EpisodePipelineTests(unittest.TestCase):
    def setUp(self):
        # Reutiliza los datos sinteticos y la configuracion del pipeline.
        self.fixture = routing_fixtures.PipelineSinClasificadorTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.state = self.fixture.estado

    def test_repeat_does_not_create_missing_context_pending_item(self):
        items, _ = agrupar_sospechas([suspicion(0), suspicion(1), suspicion(2)])
        routes = self.state._salidas_pipeline(items, [])
        self.assertEqual(routes["pending_classification_count"], 1)
        self.assertEqual(routes["pipeline_summary"]["grouping"]["suppressed"], 2)
        self.assertEqual(routes["pending_classification_results"][0]["episode"]["occurrences"], 1)

    def test_historical_reanalysis_does_not_reintroduce_suppressed_events(self):
        raw = [suspicion(i, entity="sensor.temperature") for i in range(4)]
        self.state.filtro_estadistico.analizar_historial = Mock(return_value={"all_results": raw, "summary": {}})
        with patch.object(self.state.filtro_contextual, "analizar", return_value={"results": [], "summary": {}}) as context:
            result = self.state.analizar_historial(.7, 1800)
            self.assertEqual(len(context.call_args.kwargs["sospechas"]), 1)
            self.assertEqual(len(result["all_results"]), 4)
            self.assertEqual(result["summary"]["grouping"]["suppressed"], 3)
            self.state.analizar_contexto_actual()
            self.assertEqual(len(context.call_args.kwargs["sospechas"]), 1)

    def test_replay_freezes_period_keeps_raw_events_and_reset_reopens(self):
        from app.Cargar_Historial.Cargar_Historial import ResultadoHistorial
        events = self.fixture.future
        self.state.historial_validacion = ResultadoHistorial(events, self.fixture.entities)
        self.state.eventos_validacion_ordenados = events
        self.state.validation_baseline = {}
        self.state.validation_context_map = self.fixture.mapa
        self.state.validation_context_events = self.fixture.history
        self.state.validation_replay_state = {"rows": len(events), "cursor": 0, "status": "ready"}
        self.state.filtro_contextual.analizar_causal = Mock(return_value={"results": [], "summary": {}})
        with patch("app.main.MotorReproduccionValidacion.analizar_lote", side_effect=[
            {"results": [suspicion(0), suspicion(1)], "summary": {}},
            {"results": [suspicion(2)], "summary": {}},
        ]):
            self.state.avanzar_reproduccion_validacion(3, .7, suspicion_quiet_seconds=1800)
            result = self.state.avanzar_reproduccion_validacion(3, .7, suspicion_quiet_seconds=0)
        self.assertEqual(result["suspicion_quiet_seconds"], 1800)
        self.assertEqual(result["detection_count"], 3)
        self.assertEqual(result["suppressed_count"], 2)
        self.assertEqual(result["forwarded_count"], 1)
        self.assertEqual(result["pending_classification_count"], 1)
        self.assertEqual(self.state.resultados_reproduccion_validacion("statistical", 0, 500)["total"], 3)
        self.state.reiniciar_reproduccion_validacion()
        self.assertFalse(self.state.validation_suspicion_episodes)
        self.assertEqual(self.state.validation_replay_state["suppressed_count"], 0)
