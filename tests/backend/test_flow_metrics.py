from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.database.flows import FlowStore
from app.flow.evaluacion import labelled_report
from app.flow.metrics import FlowMetrics, run_timings
from app.flow.result_spool import FlowResultSpool


def output_report(pairs):
    truth = [{"raw_index": i, "tipo": label, "entity": "sensor.a", "synthetic": "false"}
             for i, (label, _) in enumerate(pairs) if label is not None]
    output = [{"observation_id": str(i), "entity_id": "sensor.a", "source_raw_index": i,
               "final_decision": decision, "evidence": []} for i, (_, decision) in enumerate(pairs)]
    accumulator = FlowMetrics()
    report = labelled_report(output, truth, metrics=accumulator)
    return output, report, accumulator.finalize()


class FlowMetricsTests(unittest.TestCase):
    def test_three_correct_diagnoses(self):
        _, _, metrics = output_report([("normal", "normal"), ("anomalia", "environmental"), ("Falla", "fault_compatible")])
        diagnosis = metrics["diagnosis"]
        self.assertEqual([list(row.values()) for row in diagnosis["matrix"].values()],
                         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.assertEqual(diagnosis["macro_f1"]["value"], 1)
        self.assertEqual(diagnosis["coverage"]["value"], 1)
        self.assertEqual(metrics["detection"]["normal_as_alteration"]["value"], 0)
        self.assertEqual(metrics["detection"]["valid_as_fault"]["value"], 0)

    def test_detection_and_diagnosis_use_different_decisions(self):
        _, _, metrics = output_report([
            ("normal", "suspicion"), ("normal", "no_alert"), ("anomalia", "fault_compatible"),
            ("Falla", "environmental"), ("Falla", "indeterminate"), ("Falla", "not_routed"),
            ("NULL", "normal"), (None, "fault_compatible"),
        ])
        diagnosis = metrics["diagnosis"]
        self.assertEqual(metrics["support"], {
            "total": 8, "labelled": 6, "unlabelled": 2, "by_class": {"normal": 2, "anomalia": 1, "Falla": 3},
        })
        self.assertEqual(diagnosis["matrix"]["Falla"], {"normal": 0, "anomalia": 1, "Falla": 0, "no_diagnosis": 2})
        self.assertEqual(diagnosis["by_class"]["Falla"]["fn"], 3)
        self.assertEqual(diagnosis["by_class"]["Falla"]["fp"], 1)
        self.assertEqual(diagnosis["by_class"]["normal"]["fn"], 2)
        self.assertEqual(diagnosis["no_diagnosis"]["by_decision"], {
            "suspicion": 1, "no_alert": 1, "indeterminate": 1, "not_routed": 1,
        })
        self.assertAlmostEqual(diagnosis["coverage"]["value"], 2 / 6)
        detection = metrics["detection"]
        self.assertEqual(detection["by_class"]["Falla"]["recall"]["value"], 1 / 3)
        self.assertEqual(detection["by_class"]["Falla"]["not_evaluated"], 2)
        self.assertEqual(detection["normal_as_alteration"]["value"], 1 / 2)
        self.assertEqual(detection["valid_as_fault"]["value"], 1 / 3)

    def test_missing_classes_and_zero_denominators(self):
        _, _, metrics = output_report([("Falla", "no_alert")])
        diagnosis = metrics["diagnosis"]
        self.assertEqual(diagnosis["macro_f1"]["missing_classes"], ["normal", "anomalia"])
        self.assertIsNone(diagnosis["macro_f1"]["value"])
        self.assertIsNone(diagnosis["by_class"]["Falla"]["precision"]["value"])
        self.assertTrue(diagnosis["by_class"]["Falla"]["precision"]["reason"])
        self.assertEqual(diagnosis["by_class"]["Falla"]["f1"]["value"], 0)
        self.assertIsNone(metrics["detection"]["normal_as_alteration"]["value"])
        self.assertIsNone(metrics["detection"]["valid_as_fault"]["value"])
        self.assertEqual(metrics["detection"]["by_class"]["Falla"]["no_signal"], 1)
        _, _, unlabelled = output_report([("NULL", "no_alert")])
        self.assertEqual(unlabelled["status"], "no_labels")
        self.assertEqual(unlabelled["support"]["unlabelled"], 1)

    def test_recorded_timings_and_legacy_breakdown(self):
        flow = {"nodes": [{"id": "a", "type": "statistical_focus"}, {"id": "b", "type": "statistical_focus"},
                          {"id": "c", "type": "classifier"}, {"id": "d", "type": "contextual"}]}
        summary = {"elapsed_ms": 200, "node_timings_ms": {"a": 15, "b": 25, "c": 35, "d": 5}}
        legacy = run_timings(summary, flow)
        self.assertEqual(legacy["total"], {"ms": 200, "source": "elapsed_ms", "reason": None})
        self.assertEqual(legacy["filters"]["statistical_focus"]["ms"], 40)
        self.assertEqual(legacy["filters"]["classifier"]["source"], "node_timings_ms")
        summary["performance"] = {"phases_ms": {"total": 300, "classifier_execution": 0}}
        current = run_timings(summary, flow)
        self.assertEqual(current["total"]["ms"], 300)
        self.assertEqual(current["filters"]["classifier"]["ms"], 0)
        self.assertIsNone(run_timings({"node_timings_ms": {"a": 15}}, flow)["filters"]["statistical_focus"]["ms"])
        self.assertIsNone(run_timings({"node_timings_ms": {"a": 15}}, flow)["total"]["ms"])

    def test_reconstructs_both_formats_and_reuses_persisted_or_cached_metrics(self):
        output, report, expected = output_report([
            ("normal", "normal"), ("anomalia", "no_alert"), ("Falla", "environmental"),
            ("NULL", "normal"), (None, "suspicion"),
        ])
        summary = {"evaluation": report, "candidate_count": len(output), "output_count": len(output), "elapsed_ms": 8}
        with TemporaryDirectory() as directory:
            store = FlowStore(Path(directory) / "flows.sqlite3")
            legacy = store.save_run("flow", {}, summary, {"output": output})
            current = store.save_run("flow", {}, {**summary, "metrics": expected}, {"output": output})
            spool = FlowResultSpool("fixture", Path(directory))
            try:
                for item in output:
                    spool.write_block({"output": [item]})
                archived, _ = store.save_run_stream("flow", {}, deepcopy(summary), spool.records())
                for run_id in (current, legacy, archived):
                    result = store.run_metrics(run_id)
                    self.assertEqual(result.pop("run_id"), run_id)
                    self.assertEqual(result.pop("timings")["total"]["ms"], 8)
                    self.assertEqual(result, expected)
                with patch("app.database.flows.iter_run_results", side_effect=AssertionError("unexpected rescan")):
                    for run_id in (current, legacy, archived):
                        self.assertEqual(store.run_metrics(run_id)["diagnosis"], expected["diagnosis"])
                summaries = {run["run_id"]: run["summary"] for run in store.list_runs(compact=True)}
                self.assertNotIn("metrics", summaries[legacy])
                self.assertNotIn("metrics", summaries[archived])
                self.assertEqual(summaries[current]["metrics"], expected)
                spool.path.unlink()
                self.assertIn("Falta el archivo", store.run_metrics(archived)["reason"])
                with self.assertRaises(KeyError):
                    store.run_metrics("missing")
            finally:
                spool.cleanup()

    def test_incomplete_labels_preserve_known_counts_timings_and_detail(self):
        output, report, _ = output_report([("normal", "normal"), ("anomalia", "suspicion")])
        output[1]["evaluation"]["expected"] = "normal"
        summary = {"candidate_count": 2, "output_count": 2, "evaluation": report, "elapsed_ms": 12}
        with TemporaryDirectory() as directory:
            store = FlowStore(Path(directory) / "flows.sqlite3")
            run_id = store.save_run("flow", {}, summary, {"output": output})
            result = store.run_metrics(run_id)
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("no coinciden", result["reason"])
            self.assertEqual(result["support"]["labelled"], 2)
            self.assertEqual(result["timings"]["total"]["ms"], 12)
            self.assertIsNone(result["diagnosis"])
            self.assertEqual(store.results(run_id)["items"], output)
