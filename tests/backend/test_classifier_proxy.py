from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

import torch

from app.modulos.clasificador.config import Config
from app.modulos.clasificador.features import fit_preprocessor
from app.modulos.clasificador.network_segments import SegmentGATLSTM
from app.modulos.clasificador.runtime import ClassifierProxy, predict_at
from app.modulos.clasificador.segments import FEATURE_NAMES, fit_spec
from app.modulos.clasificador.temporal import build_causal_sample


class ClassifierProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.target = datetime(2026, 2, 5, tzinfo=timezone.utc)
        cls.config = Config(dropout=0, use_entropy=True, use_dtw=True)
        kinds = {"sensor.a": "continuous", "binary.b": "binary", "state.c": "multistate"}
        cls.rows = [
            {"entity_id": entity, "timestamp": (cls.target-timedelta(minutes=minute)).isoformat(),
             "source_row": minute*3+ordinal, "provenance": {"source": "fixture", "run_id": "same"},
             "state": (str(minute % 17) if kind == "continuous" else ("on" if minute % 2 else "off"))
                      if minute % 13 else "unavailable"}
            for minute in range(3*24*60, -1, -45)
            for ordinal, (entity, kind) in enumerate(kinds.items())
        ]
        # Equal timestamps, distinct rows, and a later event must keep their semantics.
        cls.rows += [{**cls.rows[-3], "state": "23", "source_row": 99999},
                     {**cls.rows[-3], "timestamp": (cls.target+timedelta(hours=1)).isoformat(), "state": "999"}]
        cls.prep = fit_preprocessor(cls.rows, [], cls.config, kinds)
        cls.graph = fit_spec(cls.rows, cls.prep, cls.config,
                             (cls.target-timedelta(days=4), cls.target))
        cls.model = SegmentGATLSTM(len(FEATURE_NAMES),
                                  [kinds[key] for key in cls.prep["entity_ids"]], cls.config).eval()
        cls.snapshot = (cls.model, cls.prep, cls.graph,
                        {"model_id": "fixture", "config": cls.config.as_dict(),
                         "rejection_policy": {"threshold": .4, "margin": 0}})

    def provider(self, start, end, entities):
        # Deliberately unsorted, with future rows, like the public sample contract allows.
        return list(reversed(self.rows))

    def requests(self, entities, at=None):
        return [{"entity_id": entity, "timestamp": (at or self.target).isoformat()}
                for entity in entities]

    def test_requested_entity_features_and_provenance_are_exact(self):
        full = build_causal_sample(self.provider, self.target, self.prep, self.graph, self.config)
        for index, entity in enumerate(self.prep["entity_ids"]):
            partial = build_causal_sample(self.provider, self.target, self.prep, self.graph,
                                          self.config, sample_entities={entity})
            for field in ("X", "state_ids", "node_present"):
                self.assertEqual([row[index] for row in partial[field]],
                                 [row[index] for row in full[field]])
            self.assertEqual(partial["metadata"]["scale_coverage"][entity], full["metadata"]["scale_coverage"][entity])
            self.assertEqual(partial["metadata"]["input_provenance"], full["metadata"]["input_provenance"])

    def test_sparse_and_simultaneous_predictions_match_full_sample(self):
        original_prep, original_graph = deepcopy(self.prep), deepcopy(self.graph)
        for entities in (["sensor.a"], ["binary.b"], ["state.c"], ["sensor.a", "binary.b", "sensor.a"]):
            metrics = {}
            proxy = ClassifierProxy(self.snapshot, self.provider, self.requests(entities), metrics=metrics)
            for entity in entities:
                actual = proxy.predict_at(entity, self.target.isoformat())
                expected = predict_at(self.snapshot, self.provider, entity, self.target.isoformat())
                # Exact comparison also covers thresholds, fault types and provenance.
                self.assertEqual(actual, expected)
            self.assertEqual(metrics["gat_samples"], 1)
            self.assertEqual(metrics["gat_inferences"], 1)
            self.assertEqual(metrics["gat_feature_entities"], len(set(entities)))
        self.assertEqual(self.prep, original_prep)
        self.assertEqual(self.graph, original_graph)

    def test_cache_is_bounded_and_never_shared_between_calls(self):
        times = [self.target+timedelta(seconds=i) for i in range(12)]
        requests = [item for at in times for item in self.requests(["sensor.a"], at)]
        proxy = ClassifierProxy(self.snapshot, self.provider, requests)
        for at in times:
            proxy.predict_at("sensor.a", at.isoformat())
        self.assertEqual(len(proxy.cache), 8)
        self.assertEqual(len(ClassifierProxy(self.snapshot, self.provider, requests).cache), 0)

    def test_unknown_model_capability_falls_back_to_complete_sample(self):
        with patch.object(self.model, "entity_independent", False):
            metrics = {}
            proxy = ClassifierProxy(self.snapshot, self.provider, self.requests(["sensor.a"]), metrics=metrics)
            self.assertEqual(proxy.predict_at("sensor.a", self.target.isoformat()),
                             predict_at(self.snapshot, self.provider, "sensor.a", self.target.isoformat()))
            self.assertEqual(metrics["gat_feature_entities"], len(self.prep["entity_ids"]))

    def test_insufficient_history_has_the_same_rejection(self):
        empty = lambda *args: []
        proxy = ClassifierProxy(self.snapshot, empty, self.requests(["sensor.a"]))
        self.assertEqual(proxy.predict_at("sensor.a", self.target.isoformat()),
                         predict_at(self.snapshot, empty, "sensor.a", self.target.isoformat()))


if __name__ == "__main__":
    unittest.main()
