import unittest

import pandas as pd

from app.flow.definitions import thesis_flows
from app.flow.runner import FlowRunner


# All readings share a sensor and timestamp. Routing must use the observation ID.
CASES = [
    ("negative", "supported", "no_fault_evidence"),
    ("unknown", "supported", "indeterminate"),
    ("positive", "supported", "compatible"),
    ("positive", "supported", "environmental_anomaly"),
    ("positive", "contradicted", "compatible"),
    ("positive", "contradicted", "environmental_anomaly"),
    ("positive", "unexplained", "no_fault_evidence"),
    ("positive", "unexplained", "indeterminate"),
]


class StatisticalFixture(FlowRunner):
    def _statistical(self, node, incoming, *args):
        results = []
        for item in incoming:
            signal = CASES[item["source_raw_index"]][0]
            status = {"positive": "detected_suppressed", "negative": "no_suspicion", "unknown": "not_evaluable"}[signal]
            results.append(self._with_evidence(item, node, signal, status, {"suspicion_score": 0.8}))
        return results


class ContextFixture:
    def analizar_causal(self, requests, *args, **kwargs):
        return {"results": [{"observation_id": r["observation_id"], "context_outcome": CASES[r["source_raw_index"]][1],
                             "decision_score": 0.3, "map_quality": {"stale": False}} for r in requests]}


class ClassifierFixture:
    def __init__(self):
        self.requests = []

    def enrich_contextual(self, statistical, contexts, **kwargs):
        self.requests.extend(zip(statistical, contexts))
        return [{**r, "diagnosis": {"status": CASES[r["source_raw_index"]][2], "confidence": 0.9,
                                    "fault_type": "drift" if CASES[r["source_raw_index"]][2] == "compatible" else None}}
                for r in contexts]


class ThesisFlowTests(unittest.TestCase):
    def run_flow(self, index):
        diagnostic = ClassifierFixture()
        runner = StatisticalFixture(None, ContextFixture(), diagnostic)
        events = pd.DataFrame([{"entity_id": "sensor.a", "state": "on", "sensor_kind": "binary",
                                "timestamp_dt": pd.Timestamp("2026-01-01", tz="UTC"), "_raw_index": i}
                               for i in range(len(CASES))])
        summary, outputs, _ = runner.run(thesis_flows()[index], events, pd.DataFrame(), {})
        self.assertEqual(summary["output_count"], len(CASES))
        final = sorted(outputs["output"], key=lambda item: item["source_raw_index"])
        self.assertEqual([x["source_raw_index"] for x in final], list(range(len(CASES))))
        return [x["final_decision"] for x in final], summary["node_counts"], diagnostic.requests, outputs

    def test_sequential_gates_and_contextual_exit(self):
        decisions, counts, _, _ = self.run_flow(0)
        self.assertEqual((counts["contextual"], counts["classifier"]), (6, 4))
        self.assertEqual(decisions, ["no_alert", "indeterminate", "environmental", "environmental",
                                     "fault_compatible", "environmental", "normal", "indeterminate"])

    def test_parallel_passes_both_scores_to_classifier_for_every_reading(self):
        decisions, counts, requests, outputs = self.run_flow(1)
        self.assertTrue(all(count == len(CASES) for count in counts.values()))
        for temporal, context in requests:
            self.assertEqual(temporal["observation_id"], context["observation_id"])
            self.assertEqual((temporal["suspicion_score"], context["decision_score"]), (0.8, 0.3))
        self.assertEqual(decisions, ["normal", "indeterminate", "indeterminate", "environmental",
                                     "fault_compatible", "indeterminate", "normal", "indeterminate"])
        conflict = outputs["classifier"][2]["evidence"][-1]["data"]
        self.assertEqual(conflict["model_verdict"]["fault_type"], "drift")
        self.assertIsNone(conflict["fault_type"])

    def test_hybrid_fuses_only_after_both_branches(self):
        decisions, counts, requests, outputs = self.run_flow(2)
        self.assertEqual((counts["contextual"], counts["classifier"]), (6, 6))
        self.assertTrue(all("context_outcome" not in context for _, context in requests))
        self.assertEqual(decisions, ["no_alert", "indeterminate", "indeterminate", "environmental",
                                     "fault_compatible", "indeterminate", "normal", "indeterminate"])
        for item in outputs["output"]:
            if item["source_raw_index"] >= 2:
                self.assertCountEqual([e["module_type"] for e in item["evidence"]],
                                      ["statistical_focus", "contextual", "classifier", "output"])
