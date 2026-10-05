import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import networkx as nx
from app.modulos.contextual.contextual import FiltroContextualEspacioTemporal, ConfiguracionContextual
from app.modulos.contextual.metrics import signed_association, nominal_association, mixed_information, numeric_dtw


class ContextualLibrariesTests(unittest.TestCase):
    def test_monotonic_numeric_and_constant_missing(self):
        pair = pd.DataFrame({'source': np.arange(20.), 'target': np.arange(20.)**3})
        self.assertAlmostEqual(signed_association(pair, 'spearman'), 1.)
        self.assertLess(signed_association(pair, 'pearson'), .95)
        pair['target'] = 1.
        self.assertIsNone(signed_association(pair))

    def test_nominal_metrics_are_invariant_to_state_names(self):
        source = pd.Series(['a', 'b', 'c']*20)
        target = source.map({'a': 'idle', 'b': 'playing', 'c': 'off'})
        before = nominal_association(source, target)
        after = nominal_association(source.map({'a':'99','b':'-5','c':'1000'}), target)
        self.assertEqual(before, after)
        self.assertAlmostEqual(before['cramers_v'], 1.)
        self.assertGreater(before['mutual_information_nats'], 0.)

    def test_mixed_information_handles_representations(self):
        category = pd.Series(['a']*20+['b']*20)
        numeric = pd.Series(np.r_[np.arange(20.)*.01, 10+np.arange(20.)*.01])
        self.assertGreater(mixed_information(numeric, category, 'continuous')['mutual_information_nats'], 0.)

    def test_dtw_is_bounded_directional_and_respects_missing_windows(self):
        index = pd.date_range('2026-01-01', periods=600, freq='min', tz='UTC')
        signal = np.sin(np.arange(600)/10)
        pair = pd.DataFrame({'source': signal, 'target': -signal}, index=index)
        score = numeric_dtw(pair, -1., max_points=64)
        self.assertEqual(score['points'], 64)
        self.assertAlmostEqual(score['similarity'], 1.)
        broken = pair.drop(pair.index[20:400])
        self.assertEqual(numeric_dtw(broken, -1.)['original_points'], 200)

    def test_networkx_reuses_graph_and_tracks_accepted_edges(self):
        filtro = FiltroContextualEspacioTemporal()
        edges = [{'edge_id': 'a-b', 'source':'a', 'target':'b', 'accepted':True},
                 {'edge_id': 'b-a', 'source':'b', 'target':'a', 'accepted':False}]
        mapa = {'edges': edges}
        graph = filtro.grafo_contextual(mapa)
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertIs(graph, filtro.grafo_contextual(mapa))
        self.assertEqual(list(graph.successors('a')), ['b'])
        self.assertEqual(list(graph.successors('b')), [])
        edges[1]['accepted'] = True
        self.assertIsNot(graph, filtro.grafo_contextual(mapa))

    def test_dtw_only_runs_for_shortlisted_numeric_pairs_and_payload_serializes(self):
        filtro = FiltroContextualEspacioTemporal(ConfiguracionContextual(top_k=1, dtw_candidate_multiplier=2))
        pair = pd.DataFrame({'source': np.arange(20.), 'target': np.arange(20.)})
        edges = [{'edge_id': f'a-{i}', 'source':'a', 'target':str(i), 'threshold_eligible':True,
                  'strength':1.-i*.1, 'association_signed':1., 'stability_score':1.,
                  'n_observations':20, 'lag_windows':0, '_dtw_pair':pair,
                  'acceptance_checks':{}, 'rejection_reasons':[]} for i in range(5)]
        with patch('app.modulos.contextual.contextual.numeric_dtw', wraps=numeric_dtw) as compute:
            filtro._aplicar_top_k(edges)
        self.assertEqual(compute.call_count, 2)
        self.assertEqual(sum(edge['accepted'] for edge in edges), 1)
        json.dumps(edges)

    def test_relations_select_metrics_and_nominal_evidence(self):
        config = ConfiguracionContextual(min_observations=4, min_pair_coverage=.1,
                                        min_stable_slices=1, max_lag_windows=0)
        filtro = FiltroContextualEspacioTemporal(config)
        index = pd.date_range('2026-01-01', periods=40, freq='h', tz='UTC')
        source = pd.Series(np.arange(40.), index=index)
        edge = filtro._mejor_relacion('a','b',source,source**3,'continuous','continuous',40)
        self.assertEqual(edge['metric'], 'spearman')
        self.assertAlmostEqual(edge['association_signed'], 1.)
        binary = pd.Series([0.,1.]*20, index=index)
        edge = filtro._mejor_relacion('a','b',binary,binary,'binary','binary',40)
        self.assertAlmostEqual(edge['nominal_association']['cramers_v'], 1.)
        self.assertIsNone(edge['_dtw_pair'])
        states = binary.map({0.: 'idle', 1.: 'playing'})
        renamed = states.map({'idle':'alpha', 'playing':'omega'})
        edge = filtro._mejor_relacion('a','b',states,renamed,'multistate','multistate',40)
        self.assertAlmostEqual(edge['nominal_association']['cramers_v'], 1.)
        self.assertIsNone(edge['_dtw_pair'])
