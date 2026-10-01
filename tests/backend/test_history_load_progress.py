"""Progress must expose expensive work without changing detector results."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from app.services.aplicacion import EstadoAplicacion
from app.services.historial.cargar_historial import CargadorHistorial
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.services.reproduccion import MotorReproduccionValidacion
from app.services.procedencia_normalidad import effective_parameters
from app.services.procedencia_normalidad import sha256_canonical


class HistoryLoadProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.state = EstadoAplicacion(CargadorHistorial(), FiltroEstadistico(),
                                     root / 'static', root / 'history.csv', root / 'cache' / 'analysis.json')
        rows = []
        for index, at in enumerate(pd.date_range('2026-01-01', periods=48, freq='h', tz='UTC')):
            for entity in ('binary_sensor.a', 'binary_sensor.b'):
                rows.append({'entity_id': entity, 'state': 'on' if index % 4 < 2 else 'off',
                             'last_changed': at.isoformat()})
        pd.DataFrame(rows).to_csv(self.state.real_history_path, index=False)

    def test_real_load_reports_ordered_internal_phases_and_cache_reuse(self):
        reports = []
        status = self.state.cargar_real(lambda *args: reports.append(args))
        phases = [phase for phase, _, _ in reports]
        self.assertTrue(status['future_evaluation_ready'])
        for phase in ('causal_identity', 'causal_cache', 'baseline_binary', 'baseline_hourly',
                      'context_pairs', 'context_decisions', 'context_save', 'causal_save', 'finalizing'):
            self.assertIn(phase, phases)
        values = [value for _, value, _ in reports]
        self.assertEqual(values, sorted(values))
        self.assertTrue(any('pares completados' in message for _, _, message in reports))
        self.assertTrue(any('sensores completados' in message for _, _, message in reports))
        reports.clear()
        self.state.cargar_real(lambda *args: reports.append(args))
        self.assertTrue(self.state.future_preparation_cache['reused'])
        self.assertNotIn('context_pairs', [phase for phase, _, _ in reports])

    def test_callback_preserves_baseline(self):
        history = self.state.cargador.cargar_real(self.state.real_history_path)
        motor = MotorReproduccionValidacion(self.state.filtro_estadistico)
        plain = motor.construir_baseline(history.eventos, history.entidades)
        reports = []
        observed = motor.construir_baseline(history.eventos, history.entidades,
                                            progress_callback=lambda *args: reports.append(args))
        self.assertEqual(sha256_canonical(effective_parameters(plain)),
                         sha256_canonical(effective_parameters(observed)))
        self.assertEqual(reports[-1][0], 'baseline_ready')

    def test_phase_clock_and_no_invented_eta(self):
        job_id = self.state._crear_job_carga(100)
        with patch('app.services.history.time', return_value=100):
            self.state._actualizar_job_carga(job_id, status='running', phase='context_pairs',
                                             progress=97.5, message='pair a', _started_epoch=100)
        with patch('app.services.history.time', return_value=110):
            self.state._actualizar_job_carga(job_id, message='pair b')
        with patch('app.services.history.time', return_value=120):
            job = self.state.estado_carga_real(job_id)
        self.assertEqual(job['phase_elapsed_seconds'], 20)
        self.assertEqual(job['seconds_since_progress'], 10)
        self.assertIsNone(job['eta_seconds'])
        self.assertFalse(any(key.startswith('_') for key in job))
        with patch('app.services.history.time', return_value=125):
            self.state._actualizar_job_carga(job_id, phase='causal_save', message='saving')
        with patch('app.services.history.time', return_value=130):
            self.state._actualizar_job_carga(job_id, status='completed', phase='ready',
                                             progress=100, _completed_epoch=130)
        with patch('app.services.history.time', return_value=999):
            job = self.state.estado_carga_real(job_id)
        self.assertEqual(job['elapsed_seconds'], 30)
        self.assertEqual(job['phase_elapsed_seconds'], 0)
        self.assertEqual(job['eta_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
