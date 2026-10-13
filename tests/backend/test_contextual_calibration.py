from copy import deepcopy
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.modulos.contextual.calibracion import PreparedFeatures
from app.modulos.contextual.calibracion import calibrate_relations
from app.modulos.contextual.contextual import ConfiguracionContextual
from app.modulos.contextual.contextual import FiltroContextualEspacioTemporal
from contextual_reference import ReferenceContext


def fixture(count=240):
    dates = pd.date_range('2026-01-01', periods=count, freq='min', tz='UTC')
    observations = {}
    for i, entity in enumerate(('a', 'b', 'c', 'd')):
        values = np.sin(np.arange(count) / 13 + i * .1) + np.arange(count) * .01
        observations[entity] = pd.DataFrame({'state': values.astype(str), 'value': values,
                                             'is_valid_value': True}, index=dates)
    edges = [dict(source=source, target=target, window_seconds=300, lag_seconds=60,
                  max_age_seconds=180, accepted=True, relation_family='continuous_continuous',
                  association_signed=.9, stability_score=1.)
             for source, target in (('a', 'b'), ('a', 'c'), ('b', 'a'), ('c', 'b'), ('d', 'a'))]
    partitions = {name: {'start': dates[lo].isoformat(), 'end': dates[hi].isoformat()}
                  for name, lo, hi in [('fit', 0, count // 2), ('tolerance', count // 2, count * 3 // 4),
                                        ('decision', count * 3 // 4, count - 1)]}
    return observations, edges, partitions


class ContextualCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = ConfiguracionContextual(temporal_model_min_observations=5, temporal_decision_min_observations=5)
        self.fast = FiltroContextualEspacioTemporal(self.config)
        self.reference = ReferenceContext(self.config)

    def test_features_match_scalar_at_boundaries_duplicates_invalid_and_expiry(self):
        obs, _, partitions = fixture(80)
        frame = obs['a'].copy()
        frame.loc[frame.index[15], 'is_valid_value'] = False
        frame.loc[frame.index[30], 'value'] = np.inf
        frame.loc[frame.index[31], 'value'] = np.nan
        duplicate = frame.iloc[[10]].copy()
        duplicate['value'] = 42.
        frame = pd.concat([frame, duplicate]).sort_index(kind='stable')
        obs['a'] = frame
        queries = frame.index.unique().append(pd.DatetimeIndex([frame.index[0] - pd.Timedelta('1min'), frame.index[-1] + pd.Timedelta('20min')]))
        queries = queries.append(queries + pd.Timedelta('30s')).sort_values()
        prepared = PreparedFeatures(obs, self.config)
        for start in (None, frame.index[20]):
            for ttl in (0., 180.):
                for window in (60, 300):
                    actual = prepared.features('a', queries, window, ttl, start)
                    expected = []
                    for at in queries:
                        result = self.reference._caracteristicas_continuas(frame, at, 'a', window_seconds=window,
                                                                          max_age_seconds=ttl, partition_start=start)
                        expected.append([np.nan if result[k] is None else result[k] for k in ('level', 'innovation', 'trend')])
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
        values = pd.to_numeric(frame['value']).to_numpy()
        actual = prepared.features('a', frame.index, 300, 0, frame.index[0], values)
        expected = [[r[k] if r[k] is not None else np.nan for k in ('level', 'innovation', 'trend')]
                    for r in (self.reference._caracteristicas_continuas(frame, at, 'a', window_seconds=300,
                            max_age_seconds=0, partition_start=frame.index[0], current_value=value)
                              for at, value in zip(frame.index, values))]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_samples_models_and_decisions_equal_scalar(self):
        obs, edges, partitions = fixture()
        obs['b'].loc[obs['b'].index[195], 'is_valid_value'] = False
        for edge in edges:
            for bounds in partitions.values():
                fast = self.fast._muestras_modelo_temporal(edge, obs, bounds)
                original = self.reference._muestras_modelo_temporal(edge, obs, bounds)
                for key in fast:
                    pd.testing.assert_frame_equal(fast[key], original[key], check_dtype=False, check_exact=True)
            fast = self.fast._calibrar_modelos_temporales(edge, obs, partitions)
            original = self.reference._calibrar_modelos_temporales(edge, obs, partitions)
            self.assertEqual(fast, original)
            edge['temporal_models'] = fast
        reports = []
        actual = self.fast._calibrar_decisiones_temporales(edges, obs, partitions, lambda *args: reports.append(args))
        self.assertEqual(actual, self.reference._calibrar_decisiones_temporales(edges, obs, partitions))
        self.assertEqual([r[1] for r in reports], sorted(r[1] for r in reports))

    def test_parallel_matches_serial_and_reports_internal_work(self):
        obs, edges, partitions = fixture()
        sequential, parallel = deepcopy(edges), deepcopy(edges)
        reports = []
        with patch.dict(os.environ, CONTEXT_CALIBRATION_WORKERS='1'):
            calibrate_relations(self.config, sequential, obs, partitions, lambda *_: None)
        with patch.dict(os.environ, CONTEXT_CALIBRATION_WORKERS='4'):
            calibrate_relations(self.config, parallel, obs, partitions, lambda *args: reports.append(args))
        self.assertEqual(parallel, sequential)
        self.assertEqual(reports[-1][1], 75.)
        self.assertEqual([r[1] for r in reports], sorted(r[1] for r in reports))
        self.assertTrue(any('eventos' in r[2] for r in reports))

    def test_cache_reuse_budget_and_key_separation(self):
        obs, _, _ = fixture()
        queries = obs['a'].index
        prepared = PreparedFeatures(obs, self.config, cache_bytes=len(queries) * 24)
        first = prepared.features('a', queries, 60, 180, None)
        self.assertIs(first, prepared.features('a', queries, 60, 180, None))
        second = prepared.features('a', queries, 300, 180, None)
        self.assertIsNot(first, second)
        self.assertLessEqual(prepared.used_bytes, prepared.cache_bytes)
        self.assertEqual(len(prepared.cache), 1)

    def test_empty_missing_and_invalid_worker_count(self):
        obs, edges, partitions = fixture()
        self.assertTrue(all(frame.empty for frame in self.fast._muestras_modelo_temporal(edges[0], {}, partitions['fit']).values()))
        with patch.dict(os.environ, CONTEXT_CALIBRATION_WORKERS='0'):
            with self.assertRaisesRegex(ValueError, 'entero positivo'):
                calibrate_relations(self.config, edges, obs, partitions, lambda *_: None)

    def test_future_readings_do_not_change_earlier_calibration(self):
        obs, edges, partitions = fixture()
        changed = deepcopy(obs)
        boundary = pd.Timestamp(partitions['tolerance']['end'])
        for frame in changed.values():
            frame.loc[frame.index > boundary, 'value'] = 999999.
        for edge in edges:
            self.assertEqual(self.fast._calibrar_modelos_temporales(edge, obs, partitions),
                             self.fast._calibrar_modelos_temporales(edge, changed, partitions))

    def test_parallel_error_names_relation(self):
        obs, edges, partitions = fixture()
        del edges[0]['window_seconds']
        with patch.dict(os.environ, CONTEXT_CALIBRATION_WORKERS='4'):
            with self.assertRaisesRegex(RuntimeError, 'Error calibrando a → b'):
                calibrate_relations(self.config, edges, obs, partitions, lambda *_: None)


if __name__ == '__main__':
    unittest.main()
