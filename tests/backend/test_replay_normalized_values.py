"""Regression coverage for reusing normalized inputs during future evaluation."""
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.modulos.estadistico.continuos import FiltroContinuoEstadistico
from app.modulos.estadistico.continuos_temporales import HISTORY_SIZE, causal_features, utc_timestamp
from app.modulos.estadistico.estados_multivalor import FiltroEstadoMultivalorEstadistico
from app.modulos.estadistico.residuos_secuenciales import RESIDUAL_FOCUS_METHOD, ResidualRuntime
from app.services.reproduccion import MotorReproduccionValidacion


class TimestampReuseTests(unittest.TestCase):
    def test_normalized_utc_is_reused_without_parsing(self):
        timestamp = pd.Timestamp("2026-06-01T00:00:00.123456789Z")
        with patch("app.modulos.estadistico.continuos_temporales.pd.to_datetime") as parse:
            self.assertIs(utc_timestamp(timestamp), timestamp)
        parse.assert_not_called()

    def test_other_formats_keep_original_utc_conversion(self):
        for value in ("2026-05-31T20:00:00-04:00", "2026-06-01T00:00:00Z",
                      pd.Timestamp("2026-06-01"), pd.Timestamp("2026-05-31T20:00:00-04:00"),
                      datetime(2026, 5, 31, 20, tzinfo=timezone(timedelta(hours=-4))),
                      "invalid", None, pd.NaT):
            with self.subTest(value=value):
                expected = pd.to_datetime(value, errors="coerce", utc=True)
                actual = utc_timestamp(value)
                if pd.isna(expected):
                    self.assertTrue(pd.isna(actual))
                else:
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual.isoformat(), expected.isoformat())

    def test_twelve_history_dates_need_no_parsing_and_keep_features(self):
        at = pd.Timestamp("2026-06-01T00:00:00Z")
        history = deque((at - pd.Timedelta(minutes=12-i), float(i)) for i in range(12))
        expected = np.r_[np.arange(12), np.log1p(np.arange(12, 0, -1) * 60),
                         [5.5, 3., 0., 11., 0., 1.]]
        with patch("app.modulos.estadistico.continuos_temporales.pd.to_datetime") as parse:
            actual = causal_features(history, at)
        parse.assert_not_called()
        np.testing.assert_array_equal(actual, expected)
        text_history = [(stamp.isoformat(), value) for stamp, value in history]
        np.testing.assert_array_equal(causal_features(text_history, at.isoformat()), actual)
        offset_history = [(stamp.tz_convert("America/Santiago"), value) for stamp, value in history]
        np.testing.assert_array_equal(causal_features(offset_history, at.tz_convert("America/Santiago")), actual)

    def test_invalid_or_non_past_history_still_has_no_features(self):
        at = pd.Timestamp("2026-06-01T00:00:00Z")
        valid = [(at - pd.Timedelta(minutes=12-i), float(i)) for i in range(12)]
        cases = [valid[:-1], valid[:-1] + [(at, 12.)],
                 valid[:-1] + [(at + pd.Timedelta(seconds=1), 12.)],
                 valid[:-1] + [(pd.NaT, 12.)], valid[:-1] + [(valid[-1][0], np.nan)]]
        for history in cases:
            with self.subTest(history=history):
                self.assertIsNone(causal_features(history, at))
        self.assertIsNone(causal_features(valid, "invalid"))


class _FocusFixture:
    """Deterministic detector fixture; native FOCuS itself is unchanged."""
    def __init__(self):
        self.total = 0.

    def update(self, value):
        self.total += value

    def get_statistics(self, **kwargs):
        return {"stat": abs(self.total), "changepoint": 0}


class _PredictorFixture:
    def predict_raw(self, vector):
        expected = float(vector[HISTORY_SIZE - 1])
        return expected - 1., expected, expected + 1.


class NormalizedReplayTests(unittest.TestCase):
    @staticmethod
    def continuous_frame():
        start = pd.Timestamp("2026-06-01T00:00:00Z")
        values = [12., 13., np.nan, np.inf, -np.inf, 17., 18., 19., 20.]
        times = [start + pd.Timedelta(seconds=60*i) for i in range(len(values))]
        times[6] = times[5]  # simultaneous reports must share a causal update
        return pd.DataFrame({"entity_id": "sensor.a", "sensor_kind": "continuous",
                             "value": values, "state": [str(value) for value in values],
                             "timestamp_dt": times, "_raw_index": range(len(values)),
                             "is_valid_value": [True] * 8 + [False]})

    @staticmethod
    def continuous_baseline():
        at = pd.Timestamp("2026-06-01T00:00:00Z")
        history = deque(((at - pd.Timedelta(minutes=12-i), float(i)) for i in range(12)), maxlen=HISTORY_SIZE)
        cached = {"runtime": ResidualRuntime(history), "predictor": _PredictorFixture(),
                  "centre": 0., "scale": 1., "channel_q99": {"P": 1., "D": 1., "N": 1.},
                  "threshold": 1., "half_life_seconds": 1800.}
        return {"continuous": {"sensor.a": cached}}

    def run_continuous(self, frame, forbid_conversion=False, baseline=None):
        baseline = baseline if baseline is not None else self.continuous_baseline()
        motor = MotorReproduccionValidacion(SimpleNamespace(continuos=FiltroContinuoEstadistico()))
        with patch("app.modulos.estadistico.residuos_secuenciales._focus", side_effect=_FocusFixture):
            if forbid_conversion:
                with patch("app.services.reproduccion.pd.to_numeric", side_effect=AssertionError("repeated numeric conversion")), \
                     patch("app.modulos.estadistico.continuos_temporales.pd.to_datetime", side_effect=AssertionError("repeated timestamp conversion")):
                    results = motor._analizar_continuos_residuos(frame, baseline, RESIDUAL_FOCUS_METHOD)
            else:
                results = motor._analizar_continuos_residuos(frame, baseline, RESIDUAL_FOCUS_METHOD)
        runtime = baseline["continuous"]["sensor.a"]["runtime"]
        return results, baseline["last_observations"], list(runtime.history), runtime.energy, runtime.last_timestamp

    def test_numeric_fast_path_keeps_scores_invalid_rows_and_simultaneous_reports(self):
        frame = self.continuous_frame()
        original_frame = frame.copy(deep=True)
        normalized = self.run_continuous(frame, forbid_conversion=True)
        fallback = frame.copy()
        fallback["value"] = fallback.value.astype(object)  # original scalar conversion path
        self.assertEqual(normalized, self.run_continuous(fallback))
        self.assertTrue(normalized[0])
        for raw_index in (2, 3, 4, 8):
            self.assertEqual(normalized[1][str(raw_index)]["reason"], "invalid_state")
        self.assertTrue(normalized[1]["5"]["evaluable"])
        self.assertTrue(normalized[1]["6"]["evaluable"])
        pd.testing.assert_frame_equal(frame, original_frame)

    def test_text_values_and_dates_still_match_normalized_readings(self):
        frame = self.continuous_frame()
        text = frame.copy()
        text["value"] = text.state
        text["timestamp_dt"] = text.timestamp_dt.map(lambda timestamp: timestamp.tz_convert("America/Santiago").isoformat())
        self.assertEqual(self.run_continuous(frame), self.run_continuous(text))

    def test_missing_numeric_value_remains_invalid(self):
        frame = self.continuous_frame().drop(columns=["value"])
        _, observations, history, _, _ = self.run_continuous(frame)
        self.assertTrue(all(item["reason"] == "invalid_state" for item in observations.values()))
        self.assertEqual(len(history), HISTORY_SIZE)

    def test_nullable_and_large_integer_values_match_scalar_conversion(self):
        for values in (pd.Series([1., pd.NA, 3.], dtype="Float64"),
                       pd.Series([2**60, 2**60+1, 2**60+2], dtype="int64")):
            with self.subTest(dtype=str(values.dtype)):
                frame = self.continuous_frame().iloc[:3].copy()
                frame["value"] = values
                fallback = frame.copy()
                fallback["value"] = fallback.value.astype(object)
                self.assertEqual(self.run_continuous(frame), self.run_continuous(fallback))

    def test_normalized_multistate_dates_keep_detections_and_runtime(self):
        filtro = FiltroEstadoMultivalorEstadistico()
        at = pd.Timestamp("2026-06-01T00:00:00Z")
        development = pd.DataFrame({"entity_id": "sensor.status", "sensor_kind": "multistate",
                                    "timestamp_dt": pd.date_range(at - pd.Timedelta(minutes=30), periods=30, freq="min"),
                                    "state": ["NONE", "ALERT", "DETECTION"] * 10, "is_valid_value": True})
        features = filtro._extraer_features(filtro._serie_valida(development))
        transitions = features[features.state_change_event.eq(1)]
        cached = {"base": filtro._construir_base(features), "state_counts": Counter(features.state_key),
                  "transition_counts": Counter(zip(transitions.previous_state_key, transitions.state_key)),
                  "source_counts": Counter(transitions.previous_state_key),
                  "runtime": {"last_timestamp": development.iloc[-1].timestamp_dt,
                              "last_state": "DETECTION", "last_state_key": "detection",
                              "state_started_at": development.iloc[-1].timestamp_dt, "change_times": deque()}}
        future = pd.DataFrame({"entity_id": "sensor.status", "sensor_kind": "multistate",
                               "timestamp_dt": [at, at+pd.Timedelta(seconds=1), at+pd.Timedelta(seconds=1), at+pd.Timedelta(hours=8)],
                               "state": ["UNSEEN", "NONE", "ALERT", "ALERT"], "_raw_index": range(4), "is_valid_value": True})
        motor = MotorReproduccionValidacion(SimpleNamespace(multivalor=filtro))
        normalized_baseline = {"multistate": {"sensor.status": deepcopy(cached)}}
        fallback_baseline = deepcopy(normalized_baseline)
        with patch("app.services.reproduccion.pd.to_datetime", side_effect=AssertionError("repeated timestamp conversion")):
            normalized = motor._analizar_multivalor(future, normalized_baseline, .6)
        text = future.copy()
        text["timestamp_dt"] = text.timestamp_dt.map(lambda stamp: stamp.isoformat())
        fallback = motor._analizar_multivalor(text, fallback_baseline, .6)
        self.assertTrue(normalized)
        self.assertEqual(normalized, fallback)
        self.assertEqual(normalized_baseline, fallback_baseline)


if __name__ == "__main__":
    unittest.main()
