from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


spec = importlib.util.spec_from_file_location(
    "causal_flow_benchmark", Path(__file__).resolve().parents[2] / "scripts/benchmark_causal_flow.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class CausalBenchmarkTests(unittest.TestCase):
    def test_prefix_completes_simultaneous_group_in_global_stable_order(self):
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        future = SimpleNamespace(eventos=pd.DataFrame([
            {"entity_id": entity, "timestamp_dt": base+pd.Timedelta(seconds=second), "_raw_index": raw}
            for second, entity, raw in [(2, "a", 0), (1, "b", 1), (0, "a", 2), (1, "a", 4), (1, "a", 3)]
        ]))
        prefix = benchmark.chronological_prefix(future, 2)
        self.assertEqual(list(prefix._raw_index), [2, 3, 4, 1])
        self.assertEqual(list(benchmark.chronological_prefix(future, None)._raw_index), [2, 3, 4, 1, 0])

    def evidence_result(self, execution):
        return {"observation_id": "source:observation:7", "signal": "positive", "source_raw_index": 7,
                "evidence": [{"node_id": "a", "observation_id": "source:observation:7",
                              "execution_id": execution, "evidence_id": execution+":evidence",
                              "data": {"run_id": execution+":classifier", "diagnosis_id": execution+":diagnosis",
                                       "input_provenance": ["{'run_id': '"+execution+":classifier'}"],
                                       "class_scores": {"normal": .2, "sensor_fault": .8}},
                              "used_evidence_ids": []}],
                "combined_evaluation": {"used_evidence_ids": [execution+":evidence"], "unused_evidence_ids": []}}

    def test_digest_normalization_keeps_row_scores_and_evidence_links(self):
        left, right = self.evidence_result("first"), self.evidence_result("second")
        original = deepcopy(left)
        self.assertEqual(benchmark._clean(left), benchmark._clean(right))
        right["source_raw_index"] = 8
        self.assertNotEqual(benchmark._clean(left), benchmark._clean(right))
        self.assertEqual(left, original)

    def test_broken_evidence_reference_is_rejected(self):
        item = self.evidence_result("first")
        item["combined_evaluation"]["used_evidence_ids"].append("missing")
        with self.assertRaisesRegex(AssertionError, "ausente"):
            benchmark._clean(item)

    def test_prepared_cache_rejects_another_history_or_altered_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "storage/cache/causal_preparation.pkl"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b'{"version":1,"key":"internal"}\n')
            baseline = {"provenance": {"identity": {"source": "expected"},
                                      "effective_parameters_sha256": "certified"}}
            with patch("app.services.cache_causal.load", return_value={"baseline": baseline}), \
                 patch("app.services.procedencia_normalidad.build_identity", return_value={"source": "other"}):
                with self.assertRaisesRegex(ValueError, "Desarrollo"):
                    benchmark._prepared_baseline(root, SimpleNamespace(eventos=None), None)
            with patch("app.services.cache_causal.load", return_value={"baseline": baseline}), \
                 patch("app.services.procedencia_normalidad.build_identity", return_value={"source": "expected"}), \
                 patch("app.services.procedencia_normalidad.sha256_canonical", return_value="altered"):
                with self.assertRaisesRegex(ValueError, "certificados"):
                    benchmark._prepared_baseline(root, SimpleNamespace(eventos=None), None)


if __name__ == "__main__":
    unittest.main()
