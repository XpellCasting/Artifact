import json
import unittest
import numpy as np
import pandas as pd
from app.modulos.contextual.contextual import FiltroContextualEspacioTemporal, ConfiguracionContextual
from app.modulos.contextual.views import reconstruct, change_view, observed_changes


def sparse_fixture(days=12, per_day=6):
    times = pd.date_range('2026-01-01', periods=days*per_day, freq=f'{24//per_day}h', tz='UTC')
    values = np.sin(np.arange(len(times))/3)+np.arange(len(times))*.02
    frames = []
    for entity, multiplier, offset in [('a',1.,0),('b',2.,30)]:
        frames.append(pd.DataFrame({'entity_id':entity, 'timestamp_dt':times+pd.Timedelta(seconds=offset),
                                   'state':(values*multiplier).astype(str), 'value':values*multiplier,
                                   'sensor_kind':'continuous', 'is_valid_value':True}))
    return pd.concat(frames).sort_values('timestamp_dt'), pd.DataFrame({'entity_id':['a','b'], 'sensor_kind':['continuous','continuous']})


class ContextualViewsTests(unittest.TestCase):
    def test_sparse_numeric_relation_uses_known_time_not_coincident_reports(self):
        events, entities = sparse_fixture()
        filtro = FiltroContextualEspacioTemporal(ConfiguracionContextual(window_candidates_seconds=(300,), window_seconds=300))
        catalog = filtro._catalogo(entities, events)
        grid = filtro._construir_senales(events,catalog,300)
        edge = filtro._seleccionar_ventana('a','b',catalog,{300:grid})
        self.assertTrue(edge['threshold_eligible'], edge['rejection_reasons'])
        self.assertGreater(edge['coverage_ratio'], .9)
        self.assertLess(edge['pair_support']['coincident_record_coverage'], .03)
        self.assertLess(edge['pair_support']['fresh_time_coverage'], .5)
        self.assertGreaterEqual(edge['pair_support']['support_days_utc'], 3)
        self.assertEqual(len(edge['view_candidates']), 2)
        self.assertLessEqual(edge['n_observations'], len(events))

    def test_one_day_is_not_enough_even_with_many_points(self):
        events, entities = sparse_fixture(days=1,per_day=24)
        filtro = FiltroContextualEspacioTemporal(ConfiguracionContextual(min_observations=5,min_stable_slices=1))
        catalog = filtro._catalogo(entities,events)
        grid = filtro._construir_senales(events,catalog)
        edge = filtro._seleccionar_ventana('a','b',catalog,{3600:grid})
        self.assertFalse(edge['threshold_eligible'])
        self.assertIn('insufficient_support_days', edge['rejection_reasons'])

    def test_unknown_invalid_stable_and_negative_changes_stay_distinct(self):
        index = pd.date_range('2026-01-01',periods=8,freq='h',tz='UTC')
        group = pd.DataFrame({'timestamp_dt':index[[1,3,5,6]],'state':['5','unavailable','7','0'],
                              'value':[5.,np.nan,7.,0.], 'is_valid_value':[True,False,True,True]})
        levels, profile = reconstruct(group,index,3600,'continuous',3600)
        self.assertTrue(pd.isna(levels.iloc[0]))
        self.assertEqual(levels.iloc[2],5.)
        self.assertFalse(profile.records.iloc[2])
        self.assertTrue(pd.isna(levels.iloc[4]))
        self.assertTrue(pd.isna(profile.changes.iloc[5]))
        self.assertEqual(profile.changes.iloc[6],-7.)
        self.assertEqual(profile.negative_changes,1)

    def test_nominal_transition_counts_do_not_depend_on_state_codes(self):
        index = pd.date_range('2026-01-01',periods=6,freq='h',tz='UTC')
        group = pd.DataFrame({'timestamp_dt':index,'state':['idle','idle','playing','off','off','idle'],'is_valid_value':True})
        levels,_ = reconstruct(group,index,3600,'multistate',3600)
        renamed=group.copy(); renamed.state=renamed.state.map({'idle':'999','playing':'-3','off':'17'})
        other,_ = reconstruct(renamed,index,3600,'multistate',3600)
        np.testing.assert_allclose(change_view(levels).to_numpy(),change_view(other).to_numpy(),equal_nan=True)

    def test_change_window_is_causal_and_requires_valid_seed(self):
        index = pd.date_range('2026-01-01',periods=5,freq='h',tz='UTC')
        events=pd.DataFrame({'state':['a','b','c','a','b'],'is_valid_value':True,'sensor_kind':'multistate'},index=index)
        value,_ = observed_changes(events,index[2],7200,'multistate')
        self.assertEqual(value,2.)
        future=events.copy(); future.loc[index[3]:,'state']='future'
        self.assertEqual(observed_changes(future,index[2],7200,'multistate')[0],value)
        events.loc[index[1],'is_valid_value']=False
        self.assertIsNone(observed_changes(events,index[2],7200,'multistate')[0])

    def test_transition_relation_evaluates_counts_instead_of_state_names(self):
        index = pd.date_range('2026-01-01',periods=5,freq='h',tz='UTC')
        events=pd.DataFrame({'state':['a','b','c','a','b'],'is_valid_value':True,'sensor_kind':'multistate'},index=index)
        edge={'source':'a','target':'b','source_kind':'multistate','target_kind':'multistate',
              'relation_view':'changes','relation_family':'transition_transition','window_seconds':7200,
              'lag_seconds':0,'association_signed':1.,'stability_score':1.,
              'expected_response':{'slope':1.,'intercept':0.,'tolerance':.5,'context_min':0.,'context_max':5.,'method':'linear_conditional_residual'}}
        filtro=FiltroContextualEspacioTemporal()
        result=filtro._evaluar_sospecha_v4(sospecha={'entity_id':'a','timestamp':index[2].isoformat(),
                   'sensor_kind':'multistate','state':'c','suspicion_score':.8},signals={},stats={},edges=[edge],
                   map_quality={'version':filtro.MAP_VERSION},concurrent_entities=[],continuous_observations={'a':events,'b':events})
        self.assertEqual(result['context_outcome'],'supported')

    def test_map_serialization_and_versioned_support_metadata(self):
        events,entities=sparse_fixture(days=20)
        filtro=FiltroContextualEspacioTemporal(ConfiguracionContextual(window_candidates_seconds=(3600,),max_lag_windows=0,
                                               temporal_model_min_observations=5,temporal_decision_min_observations=5))
        result=filtro.construir_mapa(events,entities)
        self.assertEqual(result['map_version'],'contextual-spatiotemporal-v7')
        self.assertTrue(result['edges'])
        self.assertIn('pair_support',result['edges'][0])
        json.dumps(result,allow_nan=False)
