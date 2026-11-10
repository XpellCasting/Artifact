"""Viewing a prepared map must never compete with live detection by training."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.aplicacion import EstadoAplicacion


class ContextMapReadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.state = EstadoAplicacion(Mock(), Mock(), root, root / 'history.csv', root / 'analysis.json')
        self.state.historial = SimpleNamespace(eventos=Mock(), entidades=Mock())
        self.map = {'built_at': '2020-01-01T00:00:00Z', 'map_version': 'fixture',
                    'config': {'max_map_age_days': 7}, 'refresh': {'stale': False},
                    'nodes': [{'entity_id': 'sensor.a', 'temporal_profile': {'bins': [1] * 1000}}],
                    'edges': [{'edge_id': 'relation-a', 'source': 'sensor.a', 'target': 'sensor.b',
                               'accepted': True, 'association_signed': .8,
                               'window_candidates': [{'observations': list(range(10000))}]}],
                    'decision_calibrations': {'sensor.a': {'scores': list(range(10000))}}}
        self.state.future_context_map_template = self.map

    def test_concurrent_reads_return_frozen_reference_without_rebuilding_or_copying(self):
        before = deepcopy(self.map)
        with patch.object(self.state, '_asegurar_mapa_contextual', side_effect=AssertionError('unexpected training')), \
             patch.object(self.state, 'status', side_effect=AssertionError('unexpected history loading')):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: self.state.mapa_contextual(), range(8)))
        self.assertTrue(all(result is self.map for result in results))
        self.assertEqual(before, self.map)

    def test_unprepared_map_does_not_start_work_from_a_read(self):
        self.state.future_context_map_template = None
        self.state.mapa_contextual_cache = self.map
        with patch.object(self.state, '_asegurar_mapa_contextual', side_effect=AssertionError('unexpected training')):
            with self.assertRaisesRegex(ValueError, 'no está preparado'):
                self.state.mapa_contextual()

    def test_explicit_rebuild_publishes_an_independent_reference(self):
        rebuilt = {**self.map, 'built_at': '2026-11-09T00:00:00Z'}
        with patch.object(self.state, '_asegurar_mapa_contextual', return_value=rebuilt) as build:
            self.assertIs(self.state.mapa_contextual(force=True), rebuilt)
        build.assert_called_once_with(force=True)
        self.assertEqual(self.state.future_context_map_template, rebuilt)
        self.assertIsNot(self.state.future_context_map_template, rebuilt)

    def test_duplicate_build_is_rejected_and_lock_releases_after_failure(self):
        entered, release = Event(), Event()
        def build(*args):
            entered.set()
            self.assertTrue(release.wait(5))
            raise ValueError('failed calibration')
        with patch.object(self.state, '_actualizar_mapa_contextual', side_effect=build):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.state._asegurar_mapa_contextual, True)
                try:
                    self.assertTrue(entered.wait(5))
                    with self.assertRaisesRegex(ValueError, 'Ya se está preparando'):
                        self.state._asegurar_mapa_contextual(force=True)
                finally:
                    release.set()
                with self.assertRaisesRegex(ValueError, 'failed calibration'):
                    future.result(timeout=5)
        with patch.object(self.state, '_actualizar_mapa_contextual', return_value=self.map):
            self.assertIs(self.state._asegurar_mapa_contextual(), self.map)

    def test_compact_api_keeps_full_evidence_available_without_mutating_calibration(self):
        spec = importlib.util.spec_from_file_location('isolated_context_api',
            Path(__file__).parents[2] / 'backend/app/api/contextual.py')
        module = importlib.util.module_from_spec(spec)
        jobs = Mock()
        jobs.latest.return_value = None
        with patch.dict(sys.modules, {'app.services.runtime': SimpleNamespace(state=self.state, flow_evaluation_jobs=jobs)}):
            spec.loader.exec_module(module)
        app = FastAPI()
        app.include_router(module.router)
        before = deepcopy(self.map)
        with TestClient(app) as client:
            compact = client.get('/context/map?compact=true')
            self.assertEqual(compact.status_code, 200)
            self.assertLess(len(compact.content), len(json.dumps(self.map)) / 20)
            view = compact.json()
            self.assertEqual(view['edges'][0]['association_signed'], .8)
            self.assertIn('max_age_exceeded', view['refresh']['stale_reasons'])
            self.assertEqual(client.get('/context/map/edges/relation-a').json(), self.map['edges'][0])
            self.assertEqual(client.get('/context/map/edges/missing').status_code, 404)
            full = client.get('/context/map').json()
            self.assertEqual(full['decision_calibrations'], self.map['decision_calibrations'])
            self.assertEqual(full['edges'], self.map['edges'])
            jobs.latest.return_value = {'status': 'running', 'mode': 'home_assistant'}
            with patch.object(self.state, '_asegurar_mapa_contextual', side_effect=AssertionError('unexpected training')):
                self.assertEqual(client.post('/context/map/rebuild').status_code, 409)
        self.assertEqual(self.map, before)


if __name__ == '__main__':
    unittest.main()
