from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from app.Cargar_Historial.Cargar_Historial import CargadorHistorial
from app.Filtro_Contextual import ConfiguracionContextual, FiltroContextualEspacioTemporal
from app.Filtro_Estadistico.Filtro import FiltroEstadistico
from app.Reproduccion_Validacion import MotorReproduccionValidacion
from app.main import EstadoAplicacion


def events_of(*rows):
    rows = [(entity, timestamp, state) for entity, timestamp, state in rows]
    frame = pd.DataFrame(rows, columns=["entity_id", "timestamp", "state"])
    frame["timestamp_dt"] = pd.to_datetime(frame.timestamp, format="ISO8601", utc=True)
    frame["value"] = pd.to_numeric(frame.state, errors="coerce")
    frame["is_valid_value"] = frame.value.notna()
    frame["sensor_kind"] = "continuous"
    return frame


class ContextualObservationTests(unittest.TestCase):
    target_time = "2026-06-26T11:03:49.718Z"

    def setUp(self):
        self.filtro = FiltroContextualEspacioTemporal()
        self.entities = pd.DataFrame({"entity_id": ["sensor.a", "sensor.b"], "sensor_kind": "continuous"})
        self.map = {
            "map_version": self.filtro.MAP_VERSION, "refresh": {"stale": False},
            "nodes": [{"entity_id": entity, "center": 0, "scale": 1} for entity in ["sensor.a", "sensor.b"]],
            "edges": [{"source": "sensor.a", "target": "sensor.b", "accepted": True,
                       "association_signed": 1, "stability_score": 1, "lag_seconds": 0}],
        }
        self.suspicion = {"entity_id": "sensor.a", "timestamp": self.target_time,
                          "value": 2, "is_suspicious": True, "suspicion_score": 0.9}

    def evaluate(self, events, *, causal=True):
        method = self.filtro.analizar_causal if causal else self.filtro.analizar
        return method([self.suspicion], events, self.entities, self.map)["results"][0]

    def test_carries_previous_hour_with_real_age_and_date_in_both_paths(self):
        events = events_of(("sensor.a", self.target_time, "2"),
                           ("sensor.b", "2026-06-26T10:41:49.726Z", "2"))
        for causal in [False, True]:
            with self.subTest(causal=causal):
                result = self.evaluate(events, causal=causal)
                evidence = result["evidence"][0]
                self.assertEqual(result["context_outcome"], "supported")
                self.assertEqual(result["coverage"]["carried_neighbors"], 1)
                self.assertEqual(evidence["observation_method"], "last_known")
                self.assertEqual(evidence["observed_timestamp"], "2026-06-26T10:41:49.726000+00:00")
                self.assertAlmostEqual(evidence["observation_age_seconds"], 1319.992)
                self.assertFalse(result["routing"]["send_to_classifier"])

    def test_future_one_millisecond_later_never_changes_evidence(self):
        past = events_of(("sensor.a", self.target_time, "2"),
                         ("sensor.b", "2026-06-26T10:41:49.726Z", "2"))
        future = events_of(("sensor.b", "2026-06-26T11:03:49.719Z", "-100"))
        for causal in [False, True]:
            before = self.evaluate(past, causal=causal)
            after = self.evaluate(pd.concat([past, future]), causal=causal)
            self.assertEqual(before["evidence"], after["evidence"])
            if causal:
                self.assertEqual(before["routing"], after["routing"])

    def test_future_only_is_not_observed(self):
        result = self.evaluate(events_of(("sensor.a", self.target_time, "2"),
                                        ("sensor.b", "2026-06-26T11:03:49.719Z", "2")))
        self.assertEqual(result["coverage"]["observed_neighbors"], 0)
        self.assertEqual(result["evidence"][0]["missing_reason"], "no_previous_observation")
        self.assertIsNone(result["coherence_score"])
        self.assertIsNone(result["context_score"])
        self.assertFalse(result["context_evaluable"])
        self.assertTrue(result["routing"]["send_to_classifier"])

    def test_exact_age_boundary_and_one_millisecond_expired(self):
        for excess, observed in [(0, 1), (1000000, 0), (1, 0)]:
            timestamp = pd.Timestamp(self.target_time) - pd.Timedelta(hours=1, nanoseconds=excess)
            result = self.evaluate(events_of(("sensor.a", self.target_time, "2"),
                                            ("sensor.b", timestamp.isoformat(), "2")))
            self.assertEqual(result["coverage"]["observed_neighbors"], observed)
            if not observed:
                self.assertEqual(result["evidence"][0]["missing_reason"], "expired_observation")

    def test_ttl_applies_even_within_the_same_hour(self):
        self.filtro.config = replace(self.filtro.config, continuous_max_age_seconds=30)
        result = self.evaluate(events_of(("sensor.a", self.target_time, "2"),
                                        ("sensor.b", "2026-06-26T11:01:00Z", "2")))
        self.assertEqual(result["evidence"][0]["missing_reason"], "expired_observation")
        self.assertIsNone(result["coherence_score"])

    def test_per_entity_limit_and_zero_limit(self):
        events = events_of(("sensor.a", self.target_time, "2"),
                           ("sensor.b", "2026-06-26T10:41:49.726Z", "2"))
        self.filtro.config = replace(self.filtro.config, continuous_max_age_seconds=60,
                                     continuous_max_age_seconds_by_entity={"sensor.b": 1800})
        self.assertEqual(self.evaluate(events)["coverage"]["observed_neighbors"], 1)
        self.filtro.config = replace(self.filtro.config, continuous_max_age_seconds_by_entity={"sensor.b": 0})
        self.assertEqual(self.evaluate(events)["coverage"]["observed_neighbors"], 0)

    def test_invalid_report_blocks_older_valid_reading(self):
        for state in ["unavailable", "unknown", "NaN", "inf"]:
            events = events_of(("sensor.a", self.target_time, "2"),
                               ("sensor.b", "2026-06-26T10:41:49.726Z", "2"),
                               ("sensor.b", "2026-06-26T11:01:00Z", state))
            result = self.evaluate(events)
            self.assertEqual(result["evidence"][0]["missing_reason"], "invalid_latest_observation")
            self.assertEqual(result["coverage"]["observed_neighbors"], 0)

    def test_lag_uses_its_own_cutoff_and_age(self):
        self.map["edges"][0]["lag_seconds"] = 7200
        events = events_of(("sensor.a", self.target_time, "2"),
                           ("sensor.b", "2026-06-26T08:55:00Z", "2"),
                           ("sensor.b", "2026-06-26T09:10:00Z", "-100"),
                           ("sensor.b", "2026-06-26T10:59:00Z", "-100"))
        evidence = self.evaluate(events)["evidence"][0]
        self.assertEqual(evidence["observed_value"], 2)
        self.assertEqual(evidence["reference_timestamp"], "2026-06-26T09:03:49.718000+00:00")
        self.assertAlmostEqual(evidence["observation_age_seconds"], 529.718)

    def test_same_bucket_median_uses_only_available_valid_rows(self):
        events = events_of(("sensor.a", self.target_time, "2"),
                           ("sensor.b", "2026-06-26T11:00:00Z", "100"),
                           ("sensor.b", "2026-06-26T11:01:00Z", "unknown"),
                           ("sensor.b", "2026-06-26T11:02:00Z", "2"),
                           ("sensor.b", "2026-06-26T11:03:00Z", "4"),
                           ("sensor.b", "2026-06-26T11:04:00Z", "100"))
        evidence = self.evaluate(events)["evidence"][0]
        self.assertEqual(evidence["observed_value"], 3)
        self.assertEqual(evidence["observation_method"], "bucket_median")
        self.assertFalse(evidence["carried_forward"])

    def test_absent_catalogued_neighbor_is_not_imputed(self):
        result = self.evaluate(events_of(("sensor.a", self.target_time, "2")))
        self.assertEqual(result["evidence"][0]["missing_reason"], "no_previous_observation")
        self.assertFalse(result["routing"]["terminal"])

    def test_replay_buffer_retains_latest_report_without_rejuvenating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = EstadoAplicacion(CargadorHistorial(), FiltroEstadistico(), root, root / "history.csv", root / "cache.json")
            state.filtro_contextual = self.filtro
            self.filtro.config = replace(self.filtro.config, continuous_max_age_seconds_by_entity={"sensor.b": 40000})
            events = events_of(("sensor.b", "2026-06-26T02:19:49.238Z", "2"),
                               ("sensor.a", self.target_time, "2"))
            trimmed = state._recortar_eventos_contextuales(events, pd.Timestamp(self.target_time))
            trimmed_twice = state._recortar_eventos_contextuales(trimmed, pd.Timestamp(self.target_time) + pd.Timedelta(hours=1))
            self.assertEqual(self.evaluate(events)["evidence"], self.evaluate(trimmed_twice)["evidence"])
            self.assertEqual(trimmed_twice.loc[trimmed_twice.entity_id.eq("sensor.b"), "timestamp"].iloc[0], "2026-06-26T02:19:49.238Z")
            self.filtro.config = replace(self.filtro.config, continuous_max_age_seconds_by_entity={})
            self.assertEqual(self.evaluate(trimmed_twice)["evidence"][0]["missing_reason"], "expired_observation")

    def test_age_configuration_rejects_invalid_limits(self):
        for value in [-1, float("inf"), float("nan"), True, "3600"]:
            with self.assertRaises(ValueError):
                ConfiguracionContextual(continuous_max_age_seconds_by_entity={"sensor.b": value})
        for value in [-1, float("nan")]:
            with self.assertRaises(ValueError):
                ConfiguracionContextual(continuous_max_age_seconds=value)
        with self.assertRaises(ValueError):
            ConfiguracionContextual(continuous_max_age_seconds_by_entity=[])


class TimestampPrecisionTests(unittest.TestCase):
    def test_context_parses_mixed_iso_precision_without_timestamp_dt(self):
        frame = events_of(("sensor.a", "2026-06-26T11:03:49Z", "1"),
                          ("sensor.a", "2026-06-26T11:03:49.718Z", "2"))
        result = FiltroContextualEspacioTemporal()._preparar_eventos(frame.drop(columns="timestamp_dt"))
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[1].timestamp_dt, pd.Timestamp("2026-06-26T11:03:49.718Z"))

    def test_loader_retains_distinct_subsecond_keys_and_mixed_iso_precision(self):
        frame = pd.DataFrame({"entity_id": ["sensor.a"] * 4, "state": ["1", "2", "3", "4"],
                              "timestamp": ["2026-06-26T11:03:49Z", "2026-06-26T11:03:49.718Z",
                                            "2026-06-26T07:03:49.719-04:00", "2026-06-26T11:03:49.719123456Z"]})
        result = CargadorHistorial()._normalizar(frame, "test", Path("unused.csv"))
        self.assertEqual(len(result.eventos), 4)
        self.assertEqual(result.eventos.timestamp.nunique(), 4)
        self.assertTrue(all(pd.Timestamp(row.timestamp) == row.timestamp_dt for row in result.eventos.itertuples()))

    def test_replay_emits_precise_timestamp_for_new_suspicion(self):
        from test_contextual_filter import construir_historial
        history, entities = construir_historial()
        filtro = FiltroEstadistico()
        with tempfile.TemporaryDirectory() as directory:
            filtro.continuos.conf_path = Path(directory) / "conf.txt"
            motor = MotorReproduccionValidacion(filtro)
            baseline = motor.construir_baseline(history, entities)
            timestamp = "2026-01-06T00:05:00.719123Z"
            events = events_of(("sensor.temperature", timestamp, "80"))
            results = motor.analizar_lote(events, baseline, 0.6)["results"]
            self.assertTrue(results)
            self.assertEqual(pd.Timestamp(results[0]["timestamp"]), pd.Timestamp(timestamp))


if __name__ == "__main__":
    unittest.main()
