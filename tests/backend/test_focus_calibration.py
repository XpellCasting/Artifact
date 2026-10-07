from collections import deque
from copy import deepcopy
import pickle
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from app.modulos.estadistico.continuos import FiltroContinuoEstadistico
from app.modulos.estadistico.continuos_temporales import FiltroContinuoTemporal, LEGACY_FEATURE_SCHEMA_VERSION
from app.modulos.estadistico.residuos_secuenciales import (
    RESIDUAL_FOCUS_METHOD, _choose_threshold, _channel_sequence, new_runtime, update_channels)
from app.services.reproduccion import MotorReproduccionValidacion
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.flow.evaluacion import CausalFlowRunner
from scripts.compare_focus_features import evaluate_statistical_batches


class ZeroPredictor:
    def predict_raw(self, features):
        return 0., 0., 0.


class FocusCalibrationTests(unittest.TestCase):
    def test_alerts_reset_focus_and_noise_but_preserve_predictor_history(self):
        at = pd.Timestamp("2026-01-01", tz="UTC")
        runtime = new_runtime(deque([(at - pd.Timedelta(minutes=12-i), 0.) for i in range(12)]), RESIDUAL_FOCUS_METHOD)
        baseline = {"continuous": {"sensor.a": {"runtime": runtime, "predictor": ZeroPredictor(),
            "centre": 0., "scale": 1., "channel_q99": {"P": 1., "D": 1., "N": 1.},
            "channel_thresholds": {"P": 1., "D": 1., "N": 1.}, "threshold": 1., "half_life_seconds": 1800.}}}
        rows = pd.DataFrame({"entity_id": "sensor.a", "sensor_kind": "continuous", "is_valid_value": True,
            "timestamp_dt": pd.date_range(at, periods=5, freq="min"), "value": [0., 3., 3., 3., 3.], "_raw_index": range(5)})
        motor = MotorReproduccionValidacion(SimpleNamespace(continuos=FiltroContinuoEstadistico()))
        detections = motor._analizar_continuos_residuos(rows, baseline, RESIDUAL_FOCUS_METHOD)
        self.assertEqual(len(detections), 4)
        self.assertEqual(runtime.focus_index, 0)
        self.assertEqual(runtime.focus_values, [])
        self.assertEqual(runtime.energy, 0.)
        self.assertEqual(len(runtime.history), 12)
        reference = new_runtime(deque(), RESIDUAL_FOCUS_METHOD)
        for index, value in enumerate([0., 3., 3., 3., 3.]):
            expected = update_channels(reference, value, at + pd.Timedelta(minutes=index), RESIDUAL_FOCUS_METHOD, 1800.)
            if value >= 1.:
                reference.reset_sequential(RESIDUAL_FOCUS_METHOD)
        self.assertEqual(detections[-1]["channels"]["D"], expected["D"])

    def test_active_focus_pickle_and_deepcopy_preserve_next_statistic(self):
        at = pd.Timestamp("2026-01-01", tz="UTC")
        runtime = new_runtime(deque(), RESIDUAL_FOCUS_METHOD)
        for index, value in enumerate([0., -.2, .1, 4., 4.]):
            update_channels(runtime, value, at + pd.Timedelta(minutes=index), RESIDUAL_FOCUS_METHOD, 1800.)
        restored = pickle.loads(pickle.dumps(runtime))
        copied = deepcopy(runtime)
        expected = update_channels(runtime, 4., at + pd.Timedelta(minutes=5), RESIDUAL_FOCUS_METHOD, 1800.)
        for other in (restored, copied):
            self.assertEqual(update_channels(other, 4., at + pd.Timedelta(minutes=5), RESIDUAL_FOCUS_METHOD, 1800.), expected)
            self.assertEqual(other.focus_values, runtime.focus_values)

    def test_one_long_episode_cannot_hide_ten_percent_flagged_rows(self):
        at = pd.date_range("2026-01-01", periods=1000, freq="min", tz="UTC")
        scores = [0.] * 450 + [2.] * 100 + [0.] * 450
        threshold, report = _choose_threshold(scores, at, 10.)
        self.assertGreater(threshold, 2.)
        self.assertLessEqual(report[-1]["flagged_row_rate"], .01)
        self.assertLessEqual(report[-1]["episodes_per_day"], 1.)
        self.assertTrue(any(row["episodes"] == 1 and row["flagged_row_rate"] == .1 for row in report))

    def test_calibration_and_runtime_share_simultaneous_group_updates(self):
        at = pd.Timestamp("2026-01-01", tz="UTC")
        timestamps = [at, at, at + pd.Timedelta(minutes=1)]
        actual = _channel_sequence([2., -2., 1.], timestamps, RESIDUAL_FOCUS_METHOD, 1800.)
        runtime = new_runtime(deque(), RESIDUAL_FOCUS_METHOD)
        first = update_channels(runtime, 0., at, RESIDUAL_FOCUS_METHOD, 1800.)
        last = update_channels(runtime, 1., timestamps[-1], RESIDUAL_FOCUS_METHOD, 1800.)
        self.assertEqual(actual[0]["D"], first["D"])
        self.assertEqual(actual[1]["N"], first["N"])
        self.assertEqual(actual[2], last)
        self.assertEqual(actual[0]["P"], 2.)

    def test_feature_comparison_uses_identical_targets_and_dates(self):
        at = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
        frame = pd.DataFrame({"timestamp_dt": at, "value": np.sin(np.arange(100)), "is_valid_value": True})
        old_x, old_y, old_dates = FiltroContinuoTemporal(LEGACY_FEATURE_SCHEMA_VERSION)._samples(frame)
        new_x, new_y, new_dates = FiltroContinuoTemporal()._samples(frame)
        np.testing.assert_array_equal(old_x, new_x[:, :30])
        np.testing.assert_array_equal(old_y, new_y)
        self.assertEqual(old_dates, new_dates)

    def test_batched_comparison_matches_full_causal_flow_with_simultaneous_rows(self):
        at = pd.Timestamp("2026-01-01", tz="UTC")
        history = [(at - pd.Timedelta(minutes=12-i), 0.) for i in range(12)]
        development = pd.DataFrame({"entity_id": "sensor.a", "sensor_kind": "continuous", "state": "0",
            "value": 0., "is_valid_value": True, "timestamp_dt": [t for t, _ in history]})
        baseline = {"continuous_method": RESIDUAL_FOCUS_METHOD, "continuous": {"sensor.a": {
            "runtime": new_runtime(deque(history), RESIDUAL_FOCUS_METHOD), "predictor": ZeroPredictor(),
            "centre": 0., "scale": 1., "channel_q99": {"P": 1., "D": 1., "N": 1.},
            "channel_thresholds": {"P": 1., "D": 1., "N": 1.}, "threshold": 1., "half_life_seconds": 1800.}},
            "binary": {}, "multistate": {}, "context": {}}
        future = pd.DataFrame({"entity_id": "sensor.a", "sensor_kind": "continuous", "is_valid_value": True,
            "timestamp_dt": [at + pd.Timedelta(minutes=i) for i in [0, 1, 1, 2, 3]],
            "value": [0., 3., 4., 0., 0.], "state": ["0", "3", "4", "0", "0"], "_raw_index": range(5)})
        entities = pd.DataFrame({"entity_id": ["sensor.a"], "sensor_kind": ["continuous"]})
        truth = [{"raw_index": i, "entity": "sensor.a", "tipo": "Falla" if i == 1 else "normal",
            "synthetic": "true", "fault_type": "spike" if i == 1 else None} for i in range(5)]
        flow = {"name": "comparison", "nodes": [{"id": name, "type": kind, "position": {},
            "config": {"evidence_policy": "combine_v1"}} for name, kind in
            [("input", "input"), ("statistical_focus", "statistical_focus"), ("output", "output")]],
            "edges": [{"id": "a", "source": "input", "target": "statistical_focus"},
                      {"id": "b", "source": "statistical_focus", "target": "output"}]}
        expected, _, _ = CausalFlowRunner(FiltroEstadistico(), None, None).evaluate(
            flow, development, future, entities, {}, baseline, truth)
        actual = evaluate_statistical_batches(FiltroEstadistico(), baseline, future, entities, truth, chunk_size=2)
        self.assertEqual(expected["evaluation"], actual["evaluation"])
        self.assertEqual(expected["by_final_decision"], actual["by_final_decision"])
