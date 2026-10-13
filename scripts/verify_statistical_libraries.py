"""Independent F1 reference checks; does not alter the existing test suite.

Run with the backend on PYTHONPATH and --output pointing to a JSON file.
Fixtures are reproducible synthetic examples, not labelled anomaly datasets.
"""
import argparse
from collections import deque
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from focus_cpt import Detector, focus_offline
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_info, threadpool_limits

from app.modulos.estadistico.continuos_temporales import (
    FiltroContinuoTemporal, MODEL_PARAMETERS, TemporalSensorModel,
)
from app.modulos.estadistico.estados_binarios import FiltroEstadoBinarioEstadistico, PerfilBinario
from app.modulos.estadistico.estados_multivalor import FiltroEstadoMultivalorEstadistico
from app.modulos.estadistico.residuos_secuenciales import (
    RESIDUAL_FOCUS_METHOD, _scale, new_runtime, update_channels,
)


def frame(times, states):
    at = pd.Timestamp('2026-01-01', tz='UTC') + pd.to_timedelta(times, unit='s')
    return pd.DataFrame({'entity_id': 'fixture', 'timestamp_dt': at,
                         'timestamp': at.astype(str), 'state': states,
                         'state_key': states, 'value': states})


class ReferenceChecks(unittest.TestCase):
    def test_numpy_pandas_documented_primitives(self):
        # NumPy's documented insertion example, including repeated values.
        self.assertEqual(np.searchsorted([1, 2, 3, 4, 5], 3, side='right'), 3)
        self.assertEqual(np.searchsorted([1, 2, 2, 3], 2, side='right'), 3)
        # Linear quantile: position (n-1)*q, interpolated by hand.
        values = pd.Series([0., 10., 20., 30.])
        self.assertAlmostEqual(values.quantile(.95), 28.5)
        self.assertAlmostEqual(float(np.quantile(values, .95)), 28.5)
        shifted = pd.Series(['a', 'b', 'c']).shift(1)
        self.assertTrue(pd.isna(shifted.iloc[0]))
        self.assertEqual(shifted.iloc[1:].tolist(), ['a', 'b'])
        self.assertEqual(pd.to_datetime('2026-01-01T03:00:00+03:00', utc=True),
                         pd.Timestamp('2026-01-01', tz='UTC'))

    def test_focus_official_gaussian_fixture_and_project_wrapper(self):
        # README's known theta0 example, preserving RandomState(seed=45).
        rng = np.random.RandomState(45)
        values = np.r_[rng.normal(0, 1, 1000), rng.normal(-1, 1, 500)]
        offline = focus_offline(values, threshold=np.inf, type='univariate',
                                family='gaussian', theta0=0.)
        runtime = new_runtime(deque(), RESIDUAL_FOCUS_METHOD)
        rows = [update_channels(runtime, float(z), pd.Timestamp('2026-01-01', tz='UTC')
                                + pd.Timedelta(seconds=i), RESIDUAL_FOCUS_METHOD, 1800)
                for i, z in enumerate(values)]
        np.testing.assert_allclose([r['D'] for r in rows], np.asarray(offline['stat']).ravel(),
                                   rtol=1e-10, atol=1e-10)
        expected_cp = [int(v) if math.isfinite(float(v)) else None
                       for v in np.asarray(offline['changepoint']).ravel()]
        self.assertEqual([r['focus_changepoint_index'] for r in rows], expected_cp)

    def test_focus_bernoulli_online_offline_capability(self):
        # Capability of the library; F1's binary branch does NOT use FOCuS.
        values = np.random.RandomState(123).binomial(1, .2, 250).astype(float)
        offline = focus_offline(values, threshold=np.inf, type='univariate',
                                family='bernoulli', theta0=.2)
        detector = Detector(type='univariate')
        actual = []
        for value in values:
            detector.update(float(value))
            actual.append(float(detector.get_statistics(family='bernoulli', theta0=.2)['stat']))
        np.testing.assert_allclose(actual, np.asarray(offline['stat']).ravel(), atol=1e-10)

    def test_binary_windows_against_interval_enumeration(self):
        filtro = FiltroEstadoBinarioEstadistico()
        perfil = PerfilBinario(60, 600, 600, .01)
        rng = np.random.RandomState(12)
        fixtures = [([0, 20, 60, 60, 85, 120], [1, 0, 1, 0, 1, 1])]
        fixtures += [(np.cumsum(rng.randint(0, 25, 50)).tolist(),
                      rng.randint(0, 2, 50).tolist()) for _ in range(20)]
        for times, states in fixtures:
            actual = filtro._extraer_features(frame(times, states), perfil)
            for i, t in enumerate(times):
                start = t - 60
                activations = sum(start < times[j] <= t and states[j] == 1
                                  and (j == 0 or states[j-1] == 0) for j in range(i+1))
                changes = sum(start < times[j] <= t and states[j] != states[j-1]
                              for j in range(1, i+1))
                active = sum(max(0, min(t, times[j+1]) - max(start, times[j]))
                             * states[j] for j in range(i))
                self.assertEqual(actual.iloc[i]['activations'], activations)
                self.assertEqual(actual.iloc[i]['state_changes'], changes)
                self.assertAlmostEqual(actual.iloc[i]['active_seconds'], active)
                self.assertAlmostEqual(actual.iloc[i]['active_ratio'], active/60)
                self.assertAlmostEqual(actual.iloc[i]['inactive_seconds'], 60-active)

    def test_multistate_counts_durations_and_leave_one_out_probabilities(self):
        times = [0, 20, 60, 60, 85, 120, 145, 190]
        states = ['a', 'b', 'a', 'c', 'c', 'a', 'b', 'a']
        filtro = FiltroEstadoMultivalorEstadistico()
        filtro.window_seconds = 60
        actual = filtro._extraer_features(frame(times, states))
        transitions = [(states[j-1], states[j]) for j in range(1, len(states))
                       if states[j-1] != states[j]]
        state_started = times[0]
        for i, (t, state) in enumerate(zip(times, states)):
            changed = i > 0 and states[i-1] != state
            if changed:
                state_started = t
            self.assertEqual(actual.iloc[i]['state_duration_seconds'], t-state_started)
            self.assertEqual(actual.iloc[i]['state_changes'], sum(
                t-60 < times[j] <= t and states[j] != states[j-1] for j in range(1, i+1)))
            other_states = states[:i] + states[i+1:]
            self.assertAlmostEqual(actual.iloc[i]['state_probability'],
                                   other_states.count(state)/len(other_states))
            expected = 1.
            if changed:
                others = transitions.copy()
                others.remove((states[i-1], state))
                source_rows = [target for source, target in others if source == states[i-1]]
                expected = source_rows.count(state)/max(len(source_rows), 1)
            self.assertAlmostEqual(actual.iloc[i]['transition_probability'], expected)

    def test_robust_summaries_and_population_std_fallback(self):
        for filtro in (FiltroEstadoBinarioEstadistico(), FiltroEstadoMultivalorEstadistico()):
            summary = filtro._resumen_robusto(pd.Series([0., 10., 20., 30.]), piso_mad=1.)
            self.assertEqual(summary['median'], 15.)
            self.assertEqual(summary['mad'], 10.)
            self.assertAlmostEqual(summary['scaled_mad'], 14.826)
            self.assertAlmostEqual(summary['p95'], 28.5)
        centre, scale, method = _scale([0., 0., 0., 4.], 1e-9)
        self.assertEqual(centre, 0.)
        self.assertEqual(method, 'standard_deviation')
        self.assertAlmostEqual(scale, math.sqrt(3))
        self.assertEqual(_scale([3., 3., 3.], .01), (3., .01, 'predictor_floor'))

    def test_sklearn_quantile_loss_against_empirical_median(self):
        # With constant X, q50's optimum is the sample median (15).
        x = np.zeros((100, 2))
        y = np.tile([0., 10., 20., 30.], 25)
        with threadpool_limits(limits=1):
            model = HistGradientBoostingRegressor(**MODEL_PARAMETERS, quantile=.5).fit(x, y)
            self.assertAlmostEqual(float(model.predict([[0., 0.]])[0]), 15.)

    def test_threadpool_limits_restores_native_threads(self):
        before = {p['filepath']: p['num_threads'] for p in threadpool_info()}
        self.assertTrue(before, 'No native pool loaded: check would be vacuous')
        with threadpool_limits(limits=1):
            self.assertTrue(all(p['num_threads'] == 1 for p in threadpool_info()))
        self.assertEqual(before, {p['filepath']: p['num_threads'] for p in threadpool_info()})

    def test_simultaneous_targets_do_not_enter_causal_features(self):
        data = frame(list(range(12))+[12, 12], list(range(12))+[1000., -1000.])
        x, y, _ = FiltroContinuoTemporal()._samples(data)
        self.assertEqual(y.tolist(), [1000., -1000.])
        np.testing.assert_array_equal(x[0], x[1])
        np.testing.assert_array_equal(x[0, :12], list(range(12)))

    def test_residual_central_prediction_remains_q50_when_quantiles_cross(self):
        class ConstantModel:
            def __init__(self, value): self.value = value
            def predict(self, x): return np.repeat(self.value, len(x))
        # q05=10, q50=5, q95=20: sorting must not substitute q05 for q50.
        predictor = TemporalSensorModel(tuple(ConstantModel(v) for v in (10., 5., 20.)),
                                        0., 1e-9, {})
        self.assertEqual(predictor.predict_raw(np.zeros(30))[1], 5.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceChecks)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    versions = {name: importlib.metadata.version(name) for name in
                ('focus-cpt', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'threadpoolctl')}
    source_root = Path(__file__).resolve().parents[1]/'backend/app/modulos/estadistico'
    report = {'versions': versions, 'tests_run': result.testsRun,
              'failures': [{'test': str(t), 'traceback': tb} for t, tb in result.failures],
              'errors': [{'test': str(t), 'traceback': tb} for t, tb in result.errors],
              'success': result.wasSuccessful(), 'log': log.getvalue(),
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source_root.glob('*.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(log.getvalue())
    print(json.dumps(versions))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
