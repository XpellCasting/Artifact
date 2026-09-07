from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.Filtro_Contextual import ConfiguracionContextual, FiltroContextualEspacioTemporal
from test_contextual_filter import construir_historial
from test_contextual_observations import events_of


class ContextualV2Tests(unittest.TestCase):
    def setUp(self):
        self.filter = FiltroContextualEspacioTemporal()

    def test_quality_is_checked_before_ranking_lags(self):
        self.filter.config = replace(self.filter.config, max_lag_windows=1)
        good = pd.DataFrame({"source": [-1., 1.] * 60, "target": [-2., 2.] * 60})
        bad = good.iloc[:8]
        encode = lambda pair: {"pair": pair, "source_state": None, "target_state": None}
        with patch.object(self.filter, "_codificar_par", side_effect=[encode(bad), encode(good)]), \
                patch.object(self.filter, "_correlacion", side_effect=lambda pair: 1. if len(pair) == 8 else .8):
            edge = self.filter._mejor_relacion("a", "b", pd.Series(dtype=float), pd.Series(dtype=float),
                                                "continuous", "continuous", 120)
        self.assertTrue(edge["threshold_eligible"])
        self.assertEqual(edge["lag_seconds"], 3600)
        self.assertIn("insufficient_observations", edge["lag_candidates"][0]["rejection_reasons"])

    def test_cadence_ttl_is_development_only_capped_and_overridable(self):
        times = pd.date_range("2026-01-01", periods=80, freq="30s", tz="UTC")
        events = events_of(*[("sensor.a", t.isoformat(), str(i % 2)) for i, t in enumerate(times)])
        entities = pd.DataFrame({"entity_id": ["sensor.a"], "sensor_kind": ["continuous"]})
        result = self.filter.construir_mapa(events, entities)
        profile = result["nodes"][0]["temporal_profile"]
        self.assertEqual(profile["window_seconds"], 300)
        self.assertEqual(profile["max_age_seconds"], 60)
        self.assertEqual(profile["partition"], "development")
        self.assertEqual(result["nodes"][0]["context_status"], "no_learned_relations")
        self.assertEqual(json.loads(json.dumps(result))["config"], self.filter.config.como_dict())
        self.filter.config = replace(self.filter.config, continuous_max_age_seconds_by_entity={"sensor.a": 1200})
        profile = self.filter.construir_mapa(events, entities)["nodes"][0]["temporal_profile"]
        self.assertEqual(profile["max_age_seconds"], 1200)
        self.assertEqual(profile["ttl_source"], "explicit_entity_limit")

    def test_fifteen_minute_dependency_is_learned_and_runtime_uses_its_cutoff(self):
        rng = np.random.default_rng(184)
        times = pd.date_range("2026-01-01", periods=240, freq="5min", tz="UTC")
        context = rng.normal(size=len(times))
        rows = [("sensor.b", t.isoformat(), str(context[i])) for i, t in enumerate(times)]
        rows += [("sensor.a", t.isoformat(), str(2 * context[i - 3])) for i, t in enumerate(times) if i >= 3]
        events = events_of(*rows)
        entities = pd.DataFrame({"entity_id": ["sensor.a", "sensor.b"], "sensor_kind": "continuous"})
        mapa = self.filter.construir_mapa(events, entities)
        edge = next(e for e in mapa["edges"] if e["source"] == "sensor.a")
        self.assertTrue(edge["accepted"])
        self.assertEqual(edge["window_seconds"], 300)
        self.assertEqual(edge["lag_seconds"], 900)
        t = times[-1] + pd.Timedelta(hours=1)
        suspect = {"entity_id": "sensor.a", "timestamp": t.isoformat(), "is_suspicious": True,
                   "sensor_kind": "continuous", "value": 4, "suspicion_score": .8}
        future = events_of(("sensor.b", (t - pd.Timedelta(minutes=15)).isoformat(), "2"),
                           ("sensor.b", (t - pd.Timedelta(minutes=14)).isoformat(), "-200"),
                           ("sensor.a", t.isoformat(), "4"))
        frozen = json.dumps(mapa, sort_keys=True)
        result = self.filter.analizar_causal([suspect], future, entities, mapa)["results"][0]
        self.assertEqual(result["evidence"][0]["observed_value"], 2.)
        self.assertEqual(result["evidence"][0]["window_seconds"], 300)
        self.assertEqual(result["evidence"][0]["magnitude_check"]["status"], "compatible")
        self.assertEqual(frozen, json.dumps(mapa, sort_keys=True))

    def test_runtime_uses_ttl_frozen_in_the_edge(self):
        history, entities = construir_historial()
        mapa = self.filter.construir_mapa(history, entities)
        mapa["edges"] = [e for e in mapa["edges"] if e["source"] == "sensor.temperature" and e["target"] == "sensor.humidity"]
        self.filter.config = replace(self.filter.config, continuous_max_age_seconds=0)
        t = "2026-01-06T01:05:00Z"
        future = events_of(("sensor.temperature", t, "8"), ("sensor.humidity", "2026-01-06T01:04:30Z", "16"))
        suspect = {"entity_id": "sensor.temperature", "timestamp": t, "sensor_kind": "continuous",
                   "is_suspicious": True, "value": 8}
        result = self.filter.analizar_causal([suspect], future, entities, mapa)["results"][0]
        self.assertEqual(result["evidence"][0]["observed_value"], 16.)
        self.assertEqual(result["evidence"][0]["max_age_seconds"], 3600.)

    def test_missing_target_or_missing_gap_cannot_be_explained_by_neighbor_direction(self):
        history, entities = construir_historial()
        mapa = self.filter.construir_mapa(history, entities)
        mapa["edges"] = [e for e in mapa["edges"] if e["source"] == "sensor.temperature" and e["target"] == "sensor.humidity"]
        t = "2026-01-06T01:05:00Z"
        for pattern, value in [("missing_gap", 8), ("abrupt_change", None)]:
            suspect = {"entity_id": "sensor.temperature", "timestamp": t, "sensor_kind": "continuous",
                       "is_suspicious": True, "pattern": pattern, "value": value}
            future = events_of(("sensor.humidity", t, "16"))
            result = self.filter.analizar_causal([suspect], future, entities, mapa)["results"][0]
            self.assertFalse(result["directional_evidence_applicable"])
            self.assertTrue(result["routing"]["send_to_classifier"])
            self.assertEqual(result["context_outcome"], "unexplained")

    def test_same_direction_does_not_explain_disproportionate_magnitude(self):
        history, entities = construir_historial()
        mapa = self.filter.construir_mapa(history, entities)
        # Solo la relacion continua; no modificar los datos de ajuste.
        mapa["edges"] = [e for e in mapa["edges"] if e["source"] == "sensor.temperature" and e["target"] == "sensor.humidity"]
        t = "2026-01-06T01:05:00Z"
        suspect = {"entity_id": "sensor.temperature", "timestamp": t, "sensor_kind": "continuous",
                   "value": 80., "suspicion_score": .8, "is_suspicious": True}
        future = events_of(("sensor.temperature", t, "80"), ("sensor.humidity", t, "2"))
        result = self.filter.analizar_causal([suspect], future, entities, mapa)["results"][0]
        self.assertEqual(result["evidence"][0]["magnitude_check"]["status"], "incompatible")
        self.assertTrue(result["routing"]["send_to_classifier"])
        self.assertNotEqual(result["context_outcome"], "supported")

    def test_isolated_is_not_the_same_as_missing_current_observations(self):
        history, entities = construir_historial()
        mapa = self.filter.construir_mapa(history, entities)
        mapa["edges"] = [e for e in mapa["edges"] if e["target"] == "sensor.humidity"]
        suspect = {"entity_id": "sensor.temperature", "timestamp": "2026-01-08T00:00:00Z",
                   "is_suspicious": True, "value": 10, "sensor_kind": "continuous"}
        absent = self.filter.analizar_causal([suspect], history, entities, mapa)["results"][0]
        mapa["edges"] = []
        isolated = self.filter.analizar_causal([suspect], history, entities, mapa)["results"][0]
        self.assertEqual(absent["context_outcome"], "insufficient_coverage")
        self.assertEqual(isolated["context_outcome"], "no_learned_relations")
        self.assertEqual(isolated["coverage"]["status"], "not_applicable")
        self.assertTrue(isolated["routing"]["send_to_classifier"])
        self.assertIsNone(isolated["context_score"])

    def test_concurrence_requires_related_entities_same_pattern_and_past_window(self):
        state = {}
        edges = [{"target": "a"}, {"target": "b"}, {"target": "c"}]
        def observe(entity, time, pattern="abrupt_change"):
            return self.filter._concurrencia_relacionada(
                {"entity_id": entity, "timestamp": f"2026-01-01T{time}Z", "pattern": pattern}, edges, state)
        observe("unrelated", "10:59:00")
        observe("a", "10:59:10")
        observe("b", "11:00:00", "missing_gap")
        self.assertEqual(observe("c", "11:00:10"), ["a", "c"])
        self.assertEqual(observe("b", "11:01:00"), ["a", "b", "c"])
        self.assertEqual(observe("c", "11:06:01"), ["c"])
        self.assertEqual(len(state), 1)

    def test_concurrence_is_identical_across_batch_sizes_and_never_looks_ahead(self):
        history, entities = construir_historial()
        mapa = self.filter.construir_mapa(history, entities)
        suspects = [
            {"entity_id": e, "timestamp": (pd.Timestamp("2026-01-06T00:00:00Z") + pd.Timedelta(seconds=i)).isoformat(),
             "is_suspicious": True, "pattern": "missing_gap"}
            for i, e in enumerate(["sensor.temperature", "sensor.humidity", "binary_sensor.climate"])
        ]
        whole = self.filter.analizar_causal(suspects, history, entities, mapa)["results"]
        state, split = {}, []
        for suspect in suspects:
            split.extend(self.filter.analizar_causal([suspect], history, entities, mapa,
                                                    concurrence_state=state)["results"])
        self.assertEqual(whole, split)
        self.assertFalse(whole[0]["common_cause"]["contextual_coincidence"])
        self.assertTrue(whole[-1]["common_cause"]["contextual_coincidence"])
        for _ in range(100):
            self.filter.analizar_causal([suspects[-1]], history, entities, mapa, concurrence_state=state)
        self.assertEqual(len(state), 3)

    def test_binary_context_does_not_use_a_state_after_the_lag_reference(self):
        history, entities = construir_historial()
        mapa = self.filter.construir_mapa(history, entities)
        mapa["edges"] = [e for e in mapa["edges"] if e["source"] == "sensor.temperature" and e["target"] == "binary_sensor.climate"]
        mapa["edges"][0]["lag_seconds"] = 900
        t = pd.Timestamp("2026-01-06T00:20:00Z")
        rows = pd.DataFrame([
            {"entity_id": "binary_sensor.climate", "timestamp": (t - pd.Timedelta(minutes=16)).isoformat(),
             "sensor_kind": "binary", "state": "on", "value": 1., "is_valid_value": True},
            {"entity_id": "binary_sensor.climate", "timestamp": (t - pd.Timedelta(minutes=14)).isoformat(),
             "sensor_kind": "binary", "state": "unavailable", "value": None, "is_valid_value": False},
        ])
        suspect = {"entity_id": "sensor.temperature", "timestamp": t.isoformat(), "value": 8.,
                   "sensor_kind": "continuous", "is_suspicious": True}
        result = self.filter.analizar_causal([suspect], rows, entities, mapa)["results"][0]
        self.assertEqual(result["evidence"][0]["observed_value"], 1.)
        self.assertIsNone(result["evidence"][0]["missing_reason"])


if __name__ == "__main__":
    unittest.main()
