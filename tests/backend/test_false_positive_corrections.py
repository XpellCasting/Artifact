from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from app.flow.evaluacion import statistical_labelled_report
from app.modulos.contextual.contextual import ConfiguracionContextual, FiltroContextualEspacioTemporal
from app.modulos.estadistico.continuos_temporales import (
    FiltroContinuoTemporal, PREDICTION_MODE, causal_features, increment_features)
from app.modulos.estadistico.residuos_secuenciales import prepare_sensor, RESIDUAL_FOCUS_METHOD
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.services.reproduccion import MotorReproduccionValidacion


class IncrementPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times = pd.date_range("2026-01-01", periods=2400, freq="h", tz="UTC")
        values = 20 + np.sin(np.arange(len(times)) / 20) + np.arange(len(times)) * .001
        cls.frame = pd.DataFrame({"timestamp_dt": times, "value": values, "is_valid_value": True})
        with threadpool_limits(limits=1):
            cls.baseline = prepare_sensor(FiltroContinuoTemporal(), cls.frame, "sensor.a", RESIDUAL_FOCUS_METHOD)

    def test_level_shift_outside_training_range_shifts_forecast_without_changing_residual(self):
        self.assertTrue(self.baseline["eligible"])
        predictor = self.baseline["predictor"]
        self.assertEqual(self.baseline["prediction_mode"], PREDICTION_MODE)
        history = list(zip(self.frame.timestamp_dt.iloc[-12:], self.frame.value.iloc[-12:]))
        at = self.frame.timestamp_dt.iloc[-1] + pd.Timedelta(hours=1)
        vector = causal_features(history, at)
        shifted = causal_features([(t, value - 40) for t, value in history], at)
        original_inputs, _ = increment_features(vector)
        shifted_inputs, _ = increment_features(shifted)
        np.testing.assert_allclose(original_inputs, shifted_inputs, atol=1e-12)
        with threadpool_limits(limits=1):
            np.testing.assert_allclose(np.asarray(predictor.predict_raw(vector)) - 40,
                                       predictor.predict_raw(shifted), atol=1e-12)
            np.testing.assert_allclose(predictor.predict_medians(np.stack([vector, shifted])),
                                       [predictor.predict_median(vector), predictor.predict_median(shifted)])
            expected = predictor.predict_median(shifted)
        self.assertTrue(np.isfinite(expected))

    def test_short_bias_is_detected_in_shifted_range_and_future_cannot_change_first_detection(self):
        at = self.frame.timestamp_dt.iloc[-1] + pd.Timedelta(hours=1)
        model = deepcopy(self.baseline)
        model["runtime"].history = type(model["runtime"].history)(
            [(t, value - 40) for t, value in model["runtime"].history], maxlen=12)
        level = model["runtime"].history[-1][1]
        vector = causal_features(model["runtime"].history, at)
        with threadpool_limits(limits=1):
            expected = model["predictor"].predict_median(vector)
        rows = pd.DataFrame({"entity_id": "sensor.a", "sensor_kind": "continuous", "is_valid_value": True,
            "timestamp_dt": pd.date_range(at, periods=6, freq="h"), "value": [level + 10] * 5 + [level],
            "_raw_index": range(6)})
        motor = MotorReproduccionValidacion(FiltroEstadistico())
        baseline = {"continuous": {"sensor.a": model}}
        with threadpool_limits(limits=1):
            actual = motor._analizar_continuos_residuos(rows, deepcopy(baseline), RESIDUAL_FOCUS_METHOD)
            changed = rows.copy()
            changed.loc[changed.index > 0, "value"] = -99999.
            other = motor._analizar_continuos_residuos(changed, deepcopy(baseline), RESIDUAL_FOCUS_METHOD)
        by_index = {row["source_raw_index"]: row for row in actual}
        self.assertTrue(set(range(5)).issubset(by_index))
        self.assertAlmostEqual(by_index[0]["expected_value"], expected)
        self.assertEqual(actual[0]["expected_value"], other[0]["expected_value"])
        self.assertEqual(actual[0]["channels"], other[0]["channels"])

    def test_normalization_uses_same_absolute_forecast_as_runtime(self):
        x, y, dates = FiltroContinuoTemporal()._samples(self.frame)
        partitions = self.baseline["partitions"]
        mask = np.array([pd.Timestamp(partitions["normalization_start"]) <= t <
                         pd.Timestamp(partitions["threshold_start"]) for t in dates])
        with threadpool_limits(limits=1):
            residuals = y[mask] - self.baseline["predictor"].predict_medians(x[mask])
        self.assertAlmostEqual(self.baseline["centre"], float(np.median(residuals)))


class NeighborCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.filtro = FiltroContextualEspacioTemporal(ConfiguracionContextual(
            temporal_decision_min_observations=5))

    def test_counts_have_independent_thresholds_and_minimum_samples(self):
        samples = ([{"score": s, "neighbor_counts": {"level": 1}} for s in [1., 2., 3., 4., 5.]] +
                   [{"score": s, "neighbor_counts": {"level": 2}} for s in [.1, .2, .3, .4, .5]] +
                   [{"score": 99., "neighbor_counts": {"level": 3}}])
        strata = self.filtro._calibrar_grupos_decision({"level": samples})["level"]["by_neighbor_counts"]
        self.assertEqual(strata["level=1"]["threshold"], 5.)
        self.assertEqual(strata["level=2"]["threshold"], .5)
        self.assertEqual(strata["level=3"]["status"], "insufficient_calibration")
        self.assertNotIn("threshold", strata["level=3"])

    def test_same_maximum_count_does_not_pool_different_family_coverage(self):
        samples = ([{"score": 1., "neighbor_counts": {"level": 2, "trend": 1}}] * 5 +
                   [{"score": 4., "neighbor_counts": {"level": 1, "trend": 2}}] * 5)
        strata = self.filtro._calibrar_grupos_decision({"level+trend": samples})["level+trend"]["by_neighbor_counts"]
        self.assertEqual(len(strata), 2)
        self.assertEqual(strata["level=2+trend=1"]["threshold"], 1.)
        self.assertEqual(strata["level=1+trend=2"]["threshold"], 4.)

    def test_runtime_never_uses_other_counts_or_legacy_pooled_threshold(self):
        scores = {"available_families": ["level"], "decision_score": 2., "evaluable_neighbors": 1,
                  "neighbors": {}, "family_scores": {"level": {"score": 2., "coverage": .6, "evaluable_neighbors": 1}}}
        combinations = {"level": {"status": "available", "threshold": .1, "neighbor_counts": [2, 3],
            "by_neighbor_counts": {"level=2": {"status": "available", "threshold": .5}}}}
        def evaluate(calibrations):
            with patch.object(self.filtro, "_evaluar_sospecha_v4", return_value={"evidence": [], "coverage": {}}), \
                 patch.object(self.filtro, "_puntajes_familiares", return_value=deepcopy(scores)):
                return self.filtro._evaluar_sospecha_v5(sospecha={"entity_id": "sensor.a", "value": 16.,
                    "timestamp": "2026-06-01T00:00:00Z"}, signals={}, stats={}, edges=[],
                    map_quality={"stale": False}, concurrent_entities=[], continuous_observations={},
                    decision_calibrations={"sensor.a": {"combinations": calibrations}})
        result = evaluate(combinations)
        self.assertEqual(result["context_outcome"], "unexplained")
        self.assertFalse(result["context_evaluable"])
        self.assertIsNone(result["decision_threshold"])
        self.assertEqual(result["decision_calibration"]["reason"], "uncalibrated_neighbor_counts")
        combinations["level"]["by_neighbor_counts"]["level=1"] = {
            "status": "available", "threshold": 3., "neighbor_counts": {"level": 1}}
        self.assertEqual(evaluate(combinations)["context_outcome"], "supported")
        combinations["level"]["by_neighbor_counts"]["level=1"]["threshold"] = 1.
        self.assertEqual(evaluate(combinations)["context_outcome"], "contradicted")


class FalsePositiveAlertMetricsTests(unittest.TestCase):
    def test_suppressed_readings_remain_false_positives_but_are_not_emitted_alerts(self):
        truth = [{"raw_index": i, "entity": "sensor.a", "tipo": "normal", "synthetic": "false"} for i in range(3)]
        items = [{"source_raw_index": i, "evidence": [{"node_id": "statistical", "data": {
            "statistical_observation": {"evaluable": True, "detected_before_suppression": i < 2,
                                        "alert_emitted": i == 0}}}]} for i in range(3)]
        group = statistical_labelled_report(items, truth, "statistical")["groups"][0]
        self.assertEqual(group["false_positives"], 2)
        self.assertEqual(group["false_positive_alerts"], 1)
        self.assertEqual(group["suppressed_false_positives"], 1)


if __name__ == "__main__":
    unittest.main()
