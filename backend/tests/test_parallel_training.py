import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from app.Entrenamiento_Paralelo import train_sensors
from app.Filtro_Estadistico.Continuos_Temporales import CONTINUOUS_METHOD, TemporalSensorModel
from app.Filtro_Estadistico.Filtro import FiltroEstadistico
from app.Filtro_Estadistico.Residuos_Secuenciales import RESIDUAL_EWMA_METHOD
from app.Procedencia_Normalidad import effective_parameters, sha256_canonical
from app.Reproduccion_Validacion import MotorReproduccionValidacion


class WorkerProbe:
    def __init__(self, directory):
        self.directory = directory

    def train_sensor(self, frame, entity_id):
        Path(self.directory, str(os.getpid())).touch()
        deadline = time.monotonic() + 20
        while len(list(Path(self.directory).iterdir())) < 4:
            if time.monotonic() >= deadline:
                raise RuntimeError('No se iniciaron cuatro procesos')
            time.sleep(.02)
        return {'pid': os.getpid(), 'threads': [p['num_threads'] for p in threadpool_info()]}

    def runtime(self, frame):
        return {}


class FailingSensor:
    def train_sensor(self, frame, entity_id):
        raise ValueError('fixture failure')

    def runtime(self, frame):
        return {}


class ParallelTrainingTests(unittest.TestCase):
    def test_four_processes_limit_native_threads(self):
        available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        if available < 4:
            self.skipTest('Requires four available CPUs')
        frame = pd.DataFrame({'entity_id': ['a', 'b', 'c', 'd']})
        reports = []
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, HISTORY_TRAINING_WORKERS='4'):
            results = list(train_sensors(WorkerProbe(root), frame.groupby('entity_id'), CONTINUOUS_METHOD,
                                         lambda *args: reports.append(args)))
        self.assertEqual(len({result['pid'] for _, result, _ in results}), 4)
        self.assertNotIn(os.getpid(), {result['pid'] for _, result, _ in results})
        for _, result, _ in results:
            self.assertTrue(result['threads'])
            self.assertTrue(all(count == 1 for count in result['threads']))
        self.assertEqual(reports[0][0], 0)
        self.assertEqual(reports[-1][0], 100)
        self.assertIn('4/4 sensores completados', reports[-1][1])

    def test_parallel_and_sequential_baselines_match(self):
        at = pd.date_range('2026-01-01', periods=1050, freq='10min', tz='UTC')
        values = np.sin(np.arange(len(at)) / 20) + np.arange(len(at)) * .001
        frames = []
        for entity, offset in [('sensor.a', 0), ('sensor.b', 3), ('sensor.short', 7)]:
            count = 20 if entity == 'sensor.short' else len(at)
            frames.append(pd.DataFrame({'entity_id': entity, 'timestamp_dt': at[:count],
                'value': values[:count] + offset, 'state': (values[:count] + offset).astype(str),
                'is_valid_value': True, 'sensor_kind': 'continuous'}))
        events = pd.concat(frames, ignore_index=True)
        entities = pd.DataFrame({'entity_id': ['sensor.a', 'sensor.b', 'sensor.short'],
                                 'sensor_kind': 'continuous'})
        motor = MotorReproduccionValidacion(FiltroEstadistico())
        reports = []
        with patch.dict(os.environ, HISTORY_TRAINING_WORKERS='1'), threadpool_limits(limits=1):
            sequential = motor.construir_baseline(events, entities)
        with patch.dict(os.environ, HISTORY_TRAINING_WORKERS='4'):
            parallel = motor.construir_baseline(events, entities,
                                                progress_callback=lambda *args: reports.append(args))
        self.assertEqual(sha256_canonical(effective_parameters(sequential)),
                         sha256_canonical(effective_parameters(parallel)))
        self.assertEqual(list(parallel['continuous']), ['sensor.a', 'sensor.b'])
        for entity in parallel['continuous']:
            actual, expected = parallel['continuous'][entity], sequential['continuous'][entity]
            self.assertIsInstance(actual['model'], TemporalSensorModel)
            self.assertEqual(actual['runtime'], expected['runtime'])
            with threadpool_limits(limits=1):
                np.testing.assert_array_equal(actual['model'].predict_raw(np.zeros(30)),
                                               expected['model'].predict_raw(np.zeros(30)))
        self.assertEqual([p for _, p, _ in reports], sorted(p for _, p, _ in reports))

    def test_residual_ineligible_results_match(self):
        events = pd.DataFrame({'entity_id': ['a', 'b'], 'timestamp_dt': pd.to_datetime(['2026-01-01'] * 2, utc=True),
                               'value': [1., 2.]})
        temporal = FiltroEstadistico().continuos_temporales
        with patch.dict(os.environ, HISTORY_TRAINING_WORKERS='1'):
            sequential = list(train_sensors(temporal, events.groupby('entity_id'), RESIDUAL_EWMA_METHOD, lambda *_: None))
        with patch.dict(os.environ, HISTORY_TRAINING_WORKERS='4'):
            parallel = list(train_sensors(temporal, events.groupby('entity_id'), RESIDUAL_EWMA_METHOD, lambda *_: None))
        self.assertEqual(sorted(sequential), sorted(parallel))

    def test_worker_errors_identify_sensor(self):
        frame = pd.DataFrame({'entity_id': ['a', 'b']})
        with patch.dict(os.environ, HISTORY_TRAINING_WORKERS='4'):
            with self.assertRaisesRegex(RuntimeError, 'Error al entrenar el sensor [ab]: fixture failure'):
                list(train_sensors(FailingSensor(), frame.groupby('entity_id'), CONTINUOUS_METHOD, lambda *_: None))

    def test_invalid_worker_configuration(self):
        frame = pd.DataFrame({'entity_id': ['a']})
        for value in ('0', '-1', 'invalid'):
            with self.subTest(value=value), patch.dict(os.environ, HISTORY_TRAINING_WORKERS=value):
                with self.assertRaisesRegex(ValueError, 'entero positivo'):
                    list(train_sensors(None, frame.groupby('entity_id'), CONTINUOUS_METHOD, lambda *_: None))


if __name__ == '__main__':
    unittest.main()
