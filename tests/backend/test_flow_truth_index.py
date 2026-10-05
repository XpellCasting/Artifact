import copy
import unittest
from unittest.mock import patch

import pandas as pd

from app.flow.evaluacion import CausalFlowRunner, labelled_report, statistical_labelled_report


class CountingTruth(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


class TruthIndexReportTests(unittest.TestCase):
    def setUp(self):
        self.truth = CountingTruth([
            {"raw_index": i, "entity": "sensor.a", "tipo": kind,
             "synthetic": synthetic, "fault_type": fault}
            for i, (kind, synthetic, fault) in enumerate([
                ("normal", "false", None), ("normal", "false", None),
                ("normal", "false", None), ("Falla", "false", "noise"),
                ("Falla", "true", "spike"), ("anomalia", "false", None),
                ("NULL", "false", None), ("Falla", "false", "bias"),
            ])
        ])
        self.index = {int(row["raw_index"]): row for row in self.truth}

    def test_final_report_and_attached_labels_match_original_call(self):
        items = [
            {"source_raw_index": i, "final_decision": decision, "evidence": evidence}
            for i, (decision, evidence) in enumerate([
                ("normal", []), ("suspicion", []), ("indeterminate", []),
                ("fault_compatible", [{"status": "compatible", "data": {"fault_type": "noise"}}]),
                ("no_alert", []), ("environmental", []), ("not_routed", []),
                ("fault_compatible", [{"status": "compatible", "data": {"fault_type": "spike"}}]),
            ])
        ]
        items.append({"source_raw_index": 999, "final_decision": "normal"})
        original_items, indexed_items = copy.deepcopy(items), copy.deepcopy(items)
        original = labelled_report(original_items, self.truth)
        iterations = self.truth.iterations
        indexed = labelled_report(indexed_items, self.truth, truth_index=self.index)
        self.assertEqual(indexed, original)
        self.assertEqual(indexed_items, original_items)
        self.assertEqual(self.truth.iterations, iterations)
        self.assertEqual(indexed["matched_rows"], 8)
        self.assertTrue(indexed_items[3]["evaluation"]["type_identified"])
        self.assertFalse(indexed_items[7]["evaluation"]["type_identified"])
        self.assertNotIn("evaluation", indexed_items[-1])

    def test_statistical_metrics_and_duplicate_handling_match_original_call(self):
        items = []
        for i in range(8):
            items.append({"source_raw_index": i, "evidence": [
                {"node_id": node, "data": {"statistical_observation": {
                    "evaluable": i != 2, "detected_before_suppression": i in {1, 3, 4, 5},
                    "alert_emitted": i in {1, 3, 5} and node == "first",
                }}} for node in ("first", "second")
            ]})
        items += [copy.deepcopy(items[3]), {"source_raw_index": None},
                  {"source_raw_index": 999, "evidence": []}]
        unchanged = copy.deepcopy(items)
        for node in ("first", "second", "missing"):
            with self.subTest(node=node):
                original = statistical_labelled_report(items, self.truth, node)
                iterations = self.truth.iterations
                indexed = statistical_labelled_report(items, self.truth, node, truth_index=self.index)
                self.assertEqual(indexed, original)
                self.assertEqual(self.truth.iterations, iterations)
                self.assertEqual(indexed["matched_rows"], 0 if node == "missing" else 8)
        self.assertEqual(items, unchanged)


class TruthIndexEvaluationTests(unittest.TestCase):
    @staticmethod
    def flow():
        return {"name": "truth index fixture", "nodes": [
            {"id": node, "type": kind, "position": {}, "config": {}}
            for node, kind in (("input", "input"), ("first", "statistical_focus"),
                               ("second", "statistical_focus"), ("output", "output"))
        ], "edges": [
            {"id": f"{source}-{target}", "source": source, "target": target}
            for source, target in (("input", "first"), ("first", "second"), ("second", "output"))
        ]}

    def run_fixture(self, truth, sink=None, runner=None):
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        development = pd.DataFrame([{"entity_id": "sensor.a", "timestamp_dt": base - pd.Timedelta(minutes=1),
                                     "state": "off", "sensor_kind": "binary"}])
        future = pd.DataFrame([
            {"entity_id": "sensor.a", "timestamp_dt": base + pd.Timedelta(seconds=i),
             "state": "on", "sensor_kind": "binary", "_raw_index": i}
            for i in range(3)
        ])
        baseline = {"continuous_method": "residual_focus_v1", "continuous": {}, "binary": {}, "multistate": {}}
        # No model prediction is changed here; isolate report orchestration.
        with patch("app.flow.evaluacion.MotorReproduccionValidacion") as motor:
            motor.return_value.analizar_lote.return_value = {"results": [], "detections": []}
            return (runner or CausalFlowRunner(None, None, None)).evaluate(
                self.flow(), development, future, pd.DataFrame(), {}, baseline,
                truth=truth, result_sink=sink)

    @staticmethod
    def truth_rows():
        return [{"raw_index": i, "entity": "sensor.a", "tipo": "normal",
                 "synthetic": "false", "fault_type": None} for i in range(3)]

    def test_one_index_shared_by_all_groups_and_nodes_in_both_output_modes(self):
        for labelled in (True, False):
            summaries = []
            for streaming in (False, True):
                with self.subTest(labelled=labelled, streaming=streaming):
                    truth = CountingTruth(self.truth_rows() if labelled else [])
                    blocks = []
                    with patch("app.flow.evaluacion.labelled_report", wraps=labelled_report) as final_report, \
                         patch("app.flow.evaluacion.statistical_labelled_report", wraps=statistical_labelled_report) as stat_report:
                        summary, _, _ = self.run_fixture(truth, blocks.append if streaming else None)
                    self.assertEqual(truth.iterations, 1)
                    self.assertEqual(final_report.call_count, 3 if streaming else 1)
                    self.assertEqual(stat_report.call_count, 6 if streaming else 2)
                    indexes = [call.kwargs["truth_index"] for call in
                               final_report.call_args_list + stat_report.call_args_list]
                    self.assertTrue(all(index is indexes[0] for index in indexes))
                    self.assertEqual(len(indexes[0]), 3 if labelled else 0)
                    self.assertEqual(summary["evaluation"]["labelled"], labelled)
                    summaries.append(summary)
            self.assertEqual(summaries[0]["evaluation"], summaries[1]["evaluation"])
            self.assertEqual(summaries[0]["statistical_evaluation"], summaries[1]["statistical_evaluation"])

    def test_next_evaluation_rebuilds_index_from_current_truth(self):
        runner = CausalFlowRunner(None, None, None)
        truth = CountingTruth(self.truth_rows())
        _, original, _ = self.run_fixture(truth, runner=runner)
        truth[0] = {**truth[0], "tipo": "Falla", "synthetic": "true", "fault_type": "spike"}
        _, updated, _ = self.run_fixture(truth, runner=runner)
        self.assertEqual(truth.iterations, 2)
        self.assertEqual(original["output"][0]["evaluation"]["expected"], "normal")
        self.assertEqual(updated["output"][0]["evaluation"]["expected"], "Falla")
        self.assertTrue(updated["output"][0]["evaluation"]["synthetic"])


if __name__ == "__main__":
    unittest.main()
