import unittest

import numpy as np
import pandas as pd

from app.Filtro_Contextual import FiltroContextualEspacioTemporal


class ContextualWindowTests(unittest.TestCase):
    def build(self, frequency, constant=False):
        times = pd.date_range('2026-01-01', periods=240, freq=frequency, tz='UTC')
        values = np.random.default_rng(71).normal(size=len(times))
        rows = []
        for entity, factor in [('sensor.a', 1), ('sensor.b', 2)]:
            for t, value in zip(times, values):
                rows.append({'entity_id': entity, 'timestamp': t.isoformat(),
                             'state': str(1 if constant else factor * value)})
        events = pd.DataFrame(rows)
        events['sensor_kind'] = 'continuous'
        entities = pd.DataFrame({'entity_id': ['sensor.a', 'sensor.b'], 'sensor_kind': 'continuous'})
        self.f = FiltroContextualEspacioTemporal()
        mapa = self.f.construir_mapa(events, entities)
        edge = next(e for e in mapa['edges'] if e['source'] == 'sensor.a')
        return events, entities, mapa, edge, values

    def test_smallest_eligible_window_is_retained(self):
        _, _, _, edge, _ = self.build('5min')
        self.assertTrue(edge['accepted'])
        self.assertEqual(edge['window_seconds'], 300)
        self.assertEqual(len(edge['window_candidates']), 3)

    def test_sparse_pair_uses_fifteen_minutes(self):
        events, entities, mapa, edge, values = self.build('15min')
        self.assertTrue(edge['accepted'])
        self.assertEqual(edge['window_seconds'], 900)
        self.assertEqual(edge['lag_seconds'], 0)
        self.assertIn('insufficient_pair_coverage', edge['window_candidates'][0]['rejection_reasons'])
        # Resolucion y TTL congelados deben llegar a la evidencia de inferencia.
        t = pd.to_datetime(events.timestamp, utc=True).max() + pd.Timedelta(minutes=1)
        suspect = {'entity_id': 'sensor.a', 'timestamp': t.isoformat(), 'value': float(values[-1]),
                   'sensor_kind': 'continuous', 'is_suspicious': True}
        result = self.f.analizar_causal([suspect], events, entities, mapa)['results'][0]
        self.assertEqual(result['evidence'][0]['window_seconds'], 900)
        self.assertEqual(result['evidence'][0]['max_age_seconds'], edge['max_age_seconds'])
        suspect['timestamp'] = (t + pd.Timedelta(hours=2)).isoformat()
        result = self.f.analizar_causal([suspect], events, entities, mapa)['results'][0]
        self.assertEqual(result['context_outcome'], 'insufficient_coverage')

    def test_no_eligible_window_remains_rejected_with_audit(self):
        _, _, _, edge, _ = self.build('15min', constant=True)
        self.assertFalse(edge['accepted'])
        self.assertEqual(edge['window_seconds'], 300)
        self.assertTrue(all(not c['threshold_eligible'] for c in edge['window_candidates']))
        self.assertTrue(all('weak_association' in c['rejection_reasons'] for c in edge['window_candidates']))


if __name__ == '__main__':
    unittest.main()
