from copy import deepcopy
import pickle
import unittest

import numpy as np
import pandas as pd
from focus_cpt import focus_offline

from app.modulos.estadistico.bernoulli import BernoulliRuntime, FiltroBernoulli, METHOD
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.services.reproduccion import MotorReproduccionValidacion
from app.services.procedencia_normalidad import effective_parameters


def events(values, offset=0):
    at = pd.date_range('2026-01-01', periods=len(values)+offset, freq='min', tz='UTC')[offset:]
    return pd.DataFrame({'entity_id': 'binary.a', 'sensor_kind': 'binary',
        'value': values, 'state': [str(v) for v in values], 'timestamp_dt': at,
        'timestamp': at.astype(str), 'is_valid_value': True,
        '_raw_index': np.arange(offset, offset+len(values))})


class BinaryBernoulliTests(unittest.TestCase):
    def test_official_bernoulli_fixture_matches_online(self):
        rng = np.random.RandomState(123)
        values = np.r_[rng.binomial(1, .2, 1000), rng.binomial(1, .5, 1000)]
        reference = focus_offline(values.astype(float), threshold=np.inf,
                                 type='univariate', family='bernoulli', theta0=.2)
        runtime = BernoulliRuntime()
        at = pd.Timestamp('2026-01-01', tz='UTC')
        actual = [runtime.update(float(v), at+pd.Timedelta(minutes=i), .2) for i, v in enumerate(values)]
        np.testing.assert_allclose([r['stat'] for r in actual], np.asarray(reference['stat']).ravel(), atol=1e-10)

    def test_temporal_calibration_and_baseline_probability(self):
        values = [0, 0, 0, 1]*50
        cached = FiltroBernoulli().prepare(events(values))
        self.assertEqual(cached['fit_rows'], 120)
        self.assertEqual(cached['calibration_rows'], 80)
        self.assertAlmostEqual(cached['p0'], 30.5/121)
        self.assertLess(pd.Timestamp(cached['fit_end']), pd.Timestamp(cached['calibration_start']))
        self.assertEqual(cached['runtime'].values, [])

    def test_change_detected_stable_sequence_quiet(self):
        filtro = FiltroBernoulli()
        cached = filtro.prepare(events([0, 0, 0, 1]*100))
        self.assertEqual(filtro.evaluate(events([0, 0, 0, 1]*25, 400), deepcopy(cached), 'binary.a'), [])
        results = filtro.evaluate(events([1]*100, 400), cached, 'binary.a')
        self.assertTrue(results)
        self.assertTrue(all(r['binary_method'] == METHOD for r in results))
        self.assertTrue(all(r['raw_score'] > r['reading']['focus_threshold'] for r in results))

    def test_pickle_deepcopy_and_batches_preserve_detector(self):
        filtro = FiltroBernoulli()
        cached = filtro.prepare(events([0, 0, 0, 1]*100))
        future = events([0, 1]*10+[1]*60, 400)
        expected = filtro.evaluate(future, deepcopy(cached), 'binary.a')
        actual = filtro.evaluate(future.iloc[:30], cached, 'binary.a')
        cached = pickle.loads(pickle.dumps(deepcopy(cached)))
        actual += filtro.evaluate(future.iloc[30:], cached, 'binary.a')
        self.assertEqual(expected, actual)

    def test_invalid_reports_not_coerced_to_zero(self):
        data = events([0, 1, .5, np.nan, 2, 1])
        data.loc[5, 'is_valid_value'] = False
        self.assertEqual(FiltroBernoulli().clean(data).value.tolist(), [0., 1.])

    def test_decrease_in_activation_probability_detected(self):
        filtro = FiltroBernoulli()
        cached = filtro.prepare(events([1, 1, 1, 0]*100))
        self.assertTrue(filtro.evaluate(events([0]*100, 400), cached, 'binary.a'))

    def test_simultaneous_reports_preserve_batch_order(self):
        filtro = FiltroBernoulli()
        cached = filtro.prepare(events([0, 0, 0, 1]*100))
        future = events([1]*80, 400)
        future.loc[39:41, 'timestamp_dt'] = future.iloc[39].timestamp_dt
        expected = filtro.evaluate(future, deepcopy(cached), 'binary.a')
        actual = filtro.evaluate(future.iloc[:40], cached, 'binary.a')
        actual += filtro.evaluate(future.iloc[40:], cached, 'binary.a')
        self.assertEqual(expected, actual)

    def test_constant_baselines_remain_finite(self):
        filtro = FiltroBernoulli()
        for state in (0, 1):
            cached = filtro.prepare(events([state]*100))
            self.assertTrue(0 < cached['p0'] < 1)
            self.assertEqual(filtro.evaluate(events([state]*20, 100), cached, 'binary.a'), [])

    def test_insufficient_history_and_simultaneous_partition(self):
        filtro = FiltroBernoulli()
        self.assertIsNone(filtro.prepare(events([0, 1]*20)))
        data = events([0, 1]*50)
        data['timestamp_dt'] = data.timestamp_dt.iloc[0]
        self.assertIsNone(filtro.prepare(data))

    def test_rejects_old_baseline_and_out_of_order(self):
        filtro = FiltroBernoulli()
        with self.assertRaisesRegex(ValueError, 'rebuild'):
            filtro.evaluate(events([1]), {'base': {}}, 'binary.a')
        cached = filtro.prepare(events([0, 1]*50))
        with self.assertRaisesRegex(ValueError, 'follow_development'):
            filtro.evaluate(events([1]), cached, 'binary.a')
        filtro.evaluate(events([1], 110), cached, 'binary.a')
        with self.assertRaisesRegex(ValueError, 'chronological'):
            filtro.evaluate(events([1], 105), cached, 'binary.a')

    def test_replay_uses_bernoulli_and_calibrated_threshold(self):
        motor = MotorReproduccionValidacion(FiltroEstadistico())
        data = events([0, 0, 0, 1]*100)
        entities = pd.DataFrame({'entity_id': ['binary.a'], 'sensor_kind': ['binary']})
        cached = motor.construir_baseline(data, entities)
        self.assertEqual(cached['binary']['binary.a']['method'], METHOD)
        effective_parameters(cached)  # Manifest serialization excludes runtime.
        results = motor.analizar_lote(events([1]*100, 400), cached, .99)['all_results']
        self.assertTrue(results, 'Display threshold must not veto calibrated Bernoulli decisions')
        self.assertTrue(all(r['binary_method'] == METHOD for r in results))


if __name__ == '__main__':
    unittest.main()
