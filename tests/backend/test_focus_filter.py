"""Contratos del filtro único FOCuS."""
from collections import deque
from copy import deepcopy
import os
import pickle
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from app.flow.definitions import default_flow, validate_flow
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.modulos.estadistico.residuos_secuenciales import RESIDUAL_FOCUS_METHOD, new_runtime, update_channels
from app.services.reproduccion import MotorReproduccionValidacion

class FocusFilterTests(unittest.TestCase):
    def test_channels_match_focus_cpt_known_zero_baseline(self):
        from focus_cpt import Detector
        from app.services.procedencia_normalidad import runtime_versions
        direct = Detector(type="univariate")
        runtime = new_runtime(deque(), RESIDUAL_FOCUS_METHOD)
        start = pd.Timestamp('2026-01-01', tz='UTC')
        values = [0., -.2, .1, .3, 2., 2., 2.]
        for index, value in enumerate(values):
            direct.update(value)
            expected = direct.get_statistics(family="gaussian", theta0=0.0)
            actual = update_channels(runtime, value, start + pd.Timedelta(minutes=index), RESIDUAL_FOCUS_METHOD, 1800)
            self.assertAlmostEqual(actual['D'], float(expected['stat']))
            self.assertEqual(actual['focus_changepoint_index'], int(expected['changepoint']))
        self.assertEqual(runtime_versions()['focus_cpt'], '0.1.10')
        self.assertNotIn('changepoint_online', runtime_versions())

    def test_library_focus_detects_mean_shift_and_baseline_state_can_be_cached(self):
        runtime = pickle.loads(pickle.dumps(new_runtime(deque(), RESIDUAL_FOCUS_METHOD)))
        start = pd.Timestamp('2026-01-01', tz='UTC')
        for index, value in enumerate(np.random.default_rng(42).normal(0, 1, 100)):
            before = update_channels(runtime, value, start + pd.Timedelta(minutes=index), RESIDUAL_FOCUS_METHOD, 1800)
        for index in range(100, 130):
            after = update_channels(runtime, 4., start + pd.Timedelta(minutes=index), RESIDUAL_FOCUS_METHOD, 1800)
        self.assertGreater(after['D'], before['D'] + 20)
        self.assertIsNotNone(after['focus_changepoint_index'])

    def test_retired_methods_and_nodes_are_rejected(self):
        motor = MotorReproduccionValidacion(FiltroEstadistico())
        for method in ('temporal_quantile_v1', 'residual_ewma_v1'):
            with self.assertRaisesRegex(ValueError, 'desconocido'):
                motor.construir_baseline(pd.DataFrame(), pd.DataFrame(), continuous_method=method)
        flow = default_flow()
        self.assertTrue(validate_flow(flow)['valid'])
        self.assertEqual(flow['nodes'][1]['type'], 'statistical_focus')
        flow['nodes'][1]['type'] = 'statistical'
        self.assertFalse(validate_flow(flow)['valid'])

    def test_discrete_rules_and_calibrated_threshold_are_retained(self):
        filtro = FiltroEstadistico()
        self.assertTrue(callable(filtro.binarios.analizar))
        self.assertTrue(callable(filtro.multivalor.analizar))
        motor = MotorReproduccionValidacion(filtro)
        result = {'entity_id': 'sensor.a', 'timestamp': '2026-01-01T00:00:00Z', 'value': 1.,
                  'sensor_kind': 'continuous', 'continuous_method': RESIDUAL_FOCUS_METHOD,
                  'suspicion_score': .55, 'is_suspicious': True}
        self.assertEqual(len(motor._aplicar_contexto([result], {}, .99)), 1)
        result.update(sensor_kind='binary', continuous_method=None)
        self.assertEqual(motor._aplicar_contexto([result], {}, .99), [])

    def test_replay_is_invariant_to_batch_boundaries(self):
        at = pd.date_range('2026-01-01', periods=2400, freq='h', tz='UTC')
        values = np.sin(np.arange(len(at))/20) + np.arange(len(at))*.001
        events = pd.DataFrame({'entity_id': 'sensor.a', 'timestamp_dt': at, 'value': values,
                               'state': values.astype(str), 'sensor_kind': 'continuous',
                               'is_valid_value': True, '_raw_index': np.arange(len(at))})
        entities = pd.DataFrame({'entity_id': ['sensor.a'], 'sensor_kind': ['continuous']})
        motor = MotorReproduccionValidacion(FiltroEstadistico())
        with patch.dict(os.environ, HISTORY_TRAINING_WORKERS='1'), threadpool_limits(limits=1):
            baseline = motor.construir_baseline(events, entities)
            self.assertIn('sensor.a', baseline['continuous'])
            future_at = pd.date_range(at[-1] + pd.Timedelta(hours=1), periods=30, freq='h')
            future_values = np.full(30, 20.)
            future = pd.DataFrame({'entity_id': 'sensor.a', 'timestamp_dt': future_at,
                                   'value': future_values, 'state': future_values.astype(str),
                                   'sensor_kind': 'continuous', 'is_valid_value': True,
                                   '_raw_index': np.arange(2400, 2430)})
            complete = motor.analizar_lote(future, deepcopy(baseline), .6)
            batched_baseline = deepcopy(baseline)
            batched = []
            for index in range(len(future)):
                batched.extend(motor.analizar_lote(future.iloc[index:index+1], batched_baseline, .6)['all_results'])
        self.assertTrue(complete['all_results'])
        key = lambda item: item['source_raw_index']
        self.assertEqual(sorted(complete['all_results'], key=key), sorted(batched, key=key))
