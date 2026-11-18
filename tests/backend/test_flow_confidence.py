from copy import deepcopy
from itertools import combinations, permutations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from app.database.flows import FlowStore
from app.database.confidence_entities import ConfidenceEntityStore
from app.flow.confidence import annotate_confidence
from app.flow.evaluacion import CausalFlowRunner
from app.flow.evidence import append_evidence
from app.flow.result_spool import FlowResultSpool
from app.flow.runner import FlowRunner


KINDS = ("statistical_focus", "contextual", "classifier")
SUSPICIONS = {"statistical_focus": .2, "contextual": .6, "classifier": .1}
DATA = {
    "statistical_focus": {"joint_score": .25, "threshold": 1.,
                          "statistical_observation": {"evaluable": True}},
    "contextual": {"context_outcome": "contradicted", "context_evaluable": True,
                   "decision_score": 1.5, "decision_threshold": 1.},
    "classifier": {"class_scores": {"normal": .7, "environmental_anomaly": .2, "sensor_fault": .1},
                   "confidence": .7},
}
STATUSES = {"statistical_focus": ("negative", "no_suspicion"), "contextual": ("positive", "contradicted"),
            "classifier": ("negative", "no_fault_evidence")}


def reading(kinds=KINDS, sensor_kind="continuous"):
    item = {"observation_id": "obs-001", "entity_id": "sensor.a", "timestamp": "2026-11-17T20:00:00Z",
            "sensor_kind": sensor_kind, "state": "24.5", "value": 24.5, "is_valid_value": True,
            "execution_id": "exec-001", "signal": "unknown", "evidence": []}
    filters, entries = {}, []
    for kind in kinds:
        signal, status = STATUSES[kind]
        item = append_evidence(item, kind, kind, signal, status, DATA[kind], "independent", "exec-001")
        entry = item["evidence"][-1]
        entries.append(entry)
        filters[kind] = {"module_type": kind, "enabled": True, "execution_status": "completed",
                         "evidence_ids": [entry["evidence_id"]]}
    item["final_decision"] = "normal"
    item["execution_trace"] = {"schema_version": "filter-execution-v1", "filters": filters,
                               "evidence": entries, "availability_mask": dict.fromkeys(kinds, True)}
    return item


class ConfidenceFixtureRunner(FlowRunner):
    def _filter(self, node_id, incoming):
        kind = self._evidence_nodes[node_id]["type"]
        signal, status = STATUSES[kind]
        output = []
        for item in incoming:
            data = deepcopy(DATA[kind])
            if item["value"] > 29:
                if kind == "statistical_focus":
                    data["joint_score"] = 4.
                if kind == "classifier":
                    data["class_scores"] = {"normal": .2, "environmental_anomaly": .1, "sensor_fault": .7}
            output.append(self._with_evidence(item, node_id, signal, status, data))
        return output

    def _statistical(self, node_id, incoming, *args):
        return self._filter(node_id, incoming)

    def _contextual(self, node_id, incoming, *args):
        return self._filter(node_id, incoming)

    def _classifier(self, node_id, incoming, *args):
        return self._filter(node_id, incoming)


class CausalConfidenceFixture(ConfidenceFixtureRunner, CausalFlowRunner):
    pass


class ConfidenceAdapterTests(unittest.TestCase):
    def test_seven_filter_combinations_use_uniform_mean(self):
        for count in (1, 2, 3):
            for kinds in combinations(KINDS, count):
                with self.subTest(kinds=kinds):
                    item = reading(kinds)
                    original = deepcopy(item)
                    result = annotate_confidence(item)
                    self.assertAlmostEqual(result["confidence"], 1 - sum(SUSPICIONS[kind] for kind in kinds) / count)
                    self.assertEqual(result["evidence_coverage"], 1.)
                    self.assertEqual(result["confidence_fusion"]["available_filters"], count)
                    self.assertFalse(result["confidence_calibrated"])
                    self.assertEqual(result["confidence_kind"], "operational_index")
                    for key, value in original.items():
                        self.assertEqual(result[key], value)

    def test_absence_is_not_a_normal_vote_and_disabled_is_not_missing_coverage(self):
        for status in ("disabled", "skipped", "insufficient_data", "pending", "error"):
            item = reading()
            record = item["execution_trace"]["filters"]["contextual"]
            record.update(execution_status=status, enabled=status != "disabled")
            annotate_confidence(item)
            if status in {"pending", "error"}:
                self.assertIsNone(item["confidence"])
            else:
                self.assertAlmostEqual(item["confidence"], .85)
            self.assertEqual(item["evidence_coverage"], 1. if status == "disabled" else 2 / 3)
            self.assertEqual(item["confidence_fusion"]["available_filters"], 2)
            self.assertEqual(item["confidence_fusion"]["excluded_filters"]["contextual"], status)

    def test_no_evidence_or_no_active_filters_returns_null(self):
        item = reading(())
        annotate_confidence(item)
        self.assertIsNone(item["confidence"])
        self.assertEqual(item["confidence_fusion"]["reason"], "no_active_filters")
        self.assertEqual(item["evidence_coverage"], 0.)
        item = reading()
        for record in item["execution_trace"]["filters"].values():
            record["execution_status"] = "insufficient_data"
        annotate_confidence(item)
        self.assertIsNone(item["confidence"])
        self.assertEqual(item["confidence_fusion"]["reason"], "no_usable_filter_results")
        item = reading()
        del item["execution_trace"]
        annotate_confidence(item)
        self.assertEqual(item["confidence_fusion"]["reason"], "missing_execution_trace")

    def test_statistical_ratio_and_already_bounded_scores(self):
        for sensor_kind, data, expected in [
            ("continuous", {"joint_score": 1., "threshold": 2.}, 2 / 3),
            ("continuous", {"joint_score": 4., "threshold": 1.}, .2),
            ("continuous", {"joint_score": 1e308, "threshold": 1e-308}, 0.),
            ("continuous", {"continuous_method": "residual_focus_v1", "suspicion_score": 2., "threshold": 1.}, 1 / 3),
            ("binary", {"suspicion_score": .35}, .65),
            ("multistate", {"suspicion_score": .35}, .65),
            ("binary", {}, 1.),
        ]:
            with self.subTest(sensor_kind=sensor_kind, data=data):
                item = reading(("statistical_focus",), sensor_kind=sensor_kind)
                item["execution_trace"]["evidence"][0]["data"] = data
                annotate_confidence(item)
                self.assertAlmostEqual(item["confidence"], expected)

    def test_explicit_persistence_has_maximum_suspicion(self):
        for extra in ({"constant_score": 1.}, {"persistence": {"detected": True}}, {"suspected_pattern": "unavailable"}):
            item = reading(("statistical_focus",))
            item["execution_trace"]["evidence"][0]["data"].update(extra)
            annotate_confidence(item)
            self.assertEqual(item["confidence"], 0.)

    def test_contextual_support_does_not_reuse_adjusted_statistical_score(self):
        item = reading(("contextual",))
        data = item["execution_trace"]["evidence"][0]["data"]
        data.update(context_outcome="supported", adjusted_suspicion_score=.99)
        annotate_confidence(item)
        self.assertEqual(item["confidence"], 1.)
        item = reading(("contextual",))
        item["execution_trace"]["evidence"][0]["data"] = {"context_outcome": "contradicted", "contradiction_score": .75}
        annotate_confidence(item)
        self.assertEqual(item["confidence"], .25)

    def test_classifier_uses_fault_component_including_environmental_and_abstained_results(self):
        for status in ("environmental_anomaly", "indeterminate"):
            item = reading(("classifier",))
            entry = item["execution_trace"]["evidence"][0]
            entry["status"] = status
            entry["data"] = {"confidence": .94, "class_scores": {
                "normal": .03, "environmental_anomaly": .94, "sensor_fault": .03}, "needs_review": True}
            annotate_confidence(item)
            self.assertAlmostEqual(item["confidence"], .97)
            self.assertEqual(entry["status"], status)
            self.assertTrue(entry["data"]["needs_review"])

    def test_invalid_scores_are_excluded_without_clipping_or_assuming_normal(self):
        invalid_data = [
            ("statistical_focus", {"joint_score": value, "threshold": 1.})
            for value in (-1, float("nan"), float("inf"), 10**400, None, "0.2", True)
        ] + [("statistical_focus", {"joint_score": .25, "threshold": value})
             for value in (0, -1, None, float("inf"))] + [
            ("statistical_focus", {"suspicion_score": 3.}),
            ("contextual", {"context_outcome": "contradicted", "decision_score": 1., "decision_threshold": 0}),
            ("contextual", {"context_outcome": "supported", "context_evaluable": False}),
            ("classifier", {"confidence": .99}),
            ("classifier", {"class_scores": {"normal": .5, "environmental_anomaly": .5, "sensor_fault": .5}}),
            ("classifier", {"class_scores": {"normal": .5, "environmental_anomaly": float("nan"), "sensor_fault": .5}}),
        ]
        for kind, data in invalid_data:
            with self.subTest(kind=kind, data=data):
                item = reading((kind,))
                item["execution_trace"]["evidence"][0]["data"] = data
                annotate_confidence(item)
                self.assertIsNone(item["confidence"])
                self.assertEqual(item["evidence_coverage"], 0.)
                self.assertEqual(item["confidence_fusion"]["available_filters"], 0)
                json.dumps({key: item[key] for key in ("confidence", "evidence_coverage", "confidence_fusion")}, allow_nan=False)

    def test_shared_evidence_and_control_nodes_do_not_get_extra_votes(self):
        item = reading()
        trace = item["execution_trace"]
        entry = trace["evidence"][0]
        trace["evidence"].append(deepcopy(entry))
        trace["filters"]["statistical_focus"]["evidence_ids"].append(entry["evidence_id"])
        trace["filters"]["gate"] = {"module_type": "merge", "enabled": True, "execution_status": "completed"}
        trace["evidence"].append({"node_id": "gate", "module_type": "merge", "data": {"suspicion_score": 1.}})
        annotate_confidence(item)
        self.assertAlmostEqual(item["confidence"], .7)
        self.assertEqual(item["confidence_fusion"]["configured_filters"], 3)

    def test_foreign_missing_and_ambiguous_evidence_is_not_consumed(self):
        for field, value in (("observation_id", "another-reading"), ("execution_id", "another-run"),
                             ("node_id", "another-node"), ("module_type", "merge")):
            item = reading(("classifier",))
            item["execution_trace"]["evidence"][0][field] = value
            annotate_confidence(item)
            self.assertIsNone(item["confidence"])
        item = reading(("classifier",))
        item["execution_trace"]["evidence"] = []
        annotate_confidence(item)
        self.assertEqual(item["confidence_fusion"]["excluded_filters"]["classifier"], "missing_evidence")
        item = reading(("classifier",))
        conflicting = deepcopy(item["execution_trace"]["evidence"][0])
        conflicting["data"]["class_scores"] = {"normal": .1, "environmental_anomaly": .1, "sensor_fault": .8}
        item["execution_trace"]["evidence"].append(conflicting)
        annotate_confidence(item)
        self.assertEqual(item["confidence_fusion"]["excluded_filters"]["classifier"], "ambiguous_evidence")

    def test_invalid_and_unrouted_readings_do_not_publish_valid_confidence(self):
        for field, value, reason in (("is_valid_value", False, "invalid_reading"),
                                     ("final_decision", "not_routed", "reading_not_routed")):
            item = reading()
            item[field] = value
            annotate_confidence(item)
            self.assertIsNone(item["confidence"])
            self.assertEqual(item["confidence_fusion"]["reason"], reason)


class ConfidenceIntegrationTests(unittest.TestCase):
    def test_published_explanations_for_every_filter_order_in_live_and_csv(self):
        for count in (1, 2, 3):
            for order in permutations(KINDS, count):
                for test in (False, True):
                    with self.subTest(order=order, test=test), tempfile.TemporaryDirectory() as directory:
                        store = ConfidenceEntityStore(Path(directory) / "test.sqlite")
                        store.start("session", "flow", ["sensor.a"], test=test)
                        first = reading(order)
                        first["execution_trace"]["execution_order"] = list(order)
                        second = deepcopy(first)
                        second["observation_id"] = "obs-002"
                        second["timestamp"] = "2026-11-17T20:01:00Z"
                        for entry in second["execution_trace"]["evidence"]:
                            entry["observation_id"] = "obs-002"
                            data = entry["data"]
                            if entry["module_type"] == "classifier":
                                data["class_scores"] = {"normal": .1, "environmental_anomaly": .1, "sensor_fault": .8}
                            elif entry["module_type"] == "contextual":
                                data.update(decision_score=4., evidence=[{"neighbor_id": "sensor.hall",
                                    "observation_status": "contradicts", "observed_value": 1}])
                            else:
                                data.update(joint_score=4.)
                        states = store.record("session", [annotate_confidence(row) for row in (first, second)])
                        result = states[1]
                        detail = result["confidence_change_detail"]
                        self.assertEqual(states[0]["confidence_change_detail"]["kind"], "initial")
                        self.assertEqual(detail["kind"], "decrease")
                        self.assertEqual([row["module_type"] for row in detail["filters"]], list(order))
                        self.assertAlmostEqual(sum(row["impact_pp"] for row in detail["filters"]), detail["delta_pp"], places=4)
                        if "contextual" in order:
                            self.assertIn("sensor.hall", result["confidence_change_reason"])
                        self.assertEqual(detail["previous_observation_id"], "obs-001")
                        self.assertEqual(store.snapshot()["entities"], [result])

    def test_plateaus_advance_evidence_and_changes_in_coverage_are_explained(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConfidenceEntityStore(Path(directory) / "test.sqlite")
            store.start("job", "flow", ["sensor.a"])
            def publish(identity, classifier, contextual, excluded=False):
                item = reading(("classifier", "contextual"))
                item["observation_id"] = identity
                trace = item["execution_trace"]
                for entry in trace["evidence"]:
                    entry["observation_id"] = identity
                trace["evidence"][0]["data"]["class_scores"] = {
                    "normal": 1-classifier, "environmental_anomaly": 0, "sensor_fault": classifier}
                trace["evidence"][1]["data"] = {"context_outcome": "contradicted", "contradiction_score": contextual}
                if excluded:
                    trace["filters"]["contextual"]["execution_status"] = "insufficient_data"
                return store.record("job", [annotate_confidence(item)])[0]
            a = publish("one", .25, .75)
            b = publish("two", .125, .875)
            self.assertEqual(a["confidence_change_detail"], b["confidence_change_detail"])
            c = publish("three", .125, 0., excluded=True)
            detail = c["confidence_change_detail"]
            self.assertEqual(detail["kind"], "increase")
            self.assertEqual(detail["previous_observation_id"], "two")
            self.assertEqual(detail["previous_coverage"], 1)
            self.assertEqual(detail["coverage"], .5)
            filters = {f["node_id"]: f for f in detail["filters"]}
            self.assertEqual(filters["classifier"]["before"], .125)
            self.assertEqual(filters["contextual"]["before"], .875)
            self.assertIsNone(filters["contextual"]["after"])
            self.assertIn("datos insuficientes", c["confidence_change_reason"])
            self.assertIn("empezó a aportar", publish("four", .125, .5)["confidence_change_reason"])

    @staticmethod
    def flow(active=KINDS, configured=KINDS, order=None, policy="combine_v1"):
        ordered = tuple(order or configured)
        nodes = [{"id": "input", "type": "input", "config": {"evidence_policy": policy}}]
        nodes += [{"id": kind, "type": kind, "config": {"enabled": kind in active, "evidence_policy": policy}}
                  for kind in ordered]
        nodes += [{"id": "output", "type": "output", "config": {"evidence_policy": policy}}]
        identifiers = [node["id"] for node in nodes]
        return {"flow_id": "confidence-flow", "name": "Confidence", "nodes": nodes,
                "edges": [{"source": source, "target": target} for source, target in zip(identifiers, identifiers[1:])]}

    @staticmethod
    def events():
        return pd.DataFrame([{"entity_id": "sensor.a", "timestamp_dt": pd.Timestamp("2026-11-17T20:00:00Z"),
                              "sensor_kind": "continuous", "state": "24.5", "value": 24.5, "_raw_index": 0}])

    def run_flow(self, flow, events=None):
        return ConfidenceFixtureRunner(None, None, None).run(
            flow, events if events is not None else self.events(), pd.DataFrame(), {})

    def test_output_integrates_every_active_combination_with_all_existing_policies(self):
        for count in (1, 2, 3):
            for active in combinations(KINDS, count):
                for policy in ("legacy", "independent", "combine_v1"):
                    with self.subTest(active=active, policy=policy):
                        flow = self.flow(active, policy=policy)
                        _, outputs, snapshot = self.run_flow(flow)
                        item = outputs["output"][0]
                        self.assertAlmostEqual(item["confidence"], 1 - sum(SUSPICIONS[kind] for kind in active) / count)
                        self.assertEqual(item["evidence_coverage"], 1.)
                        self.assertEqual(item["confidence_fusion"]["active_filters"], count)
                        self.assertEqual(snapshot["confidence_policy"]["method"], "uniform_v1")
                        self.assertEqual(snapshot["flow"], flow)
                        self.assertFalse(any("confidence" in candidate for node, rows in outputs.items()
                                             if node != "output" for candidate in rows))

    def test_one_two_and_three_configured_filters_and_all_orders(self):
        for count in (1, 2, 3):
            for configured in combinations(KINDS, count):
                _, outputs, _ = self.run_flow(self.flow(configured, configured))
                self.assertEqual(outputs["output"][0]["confidence_fusion"]["configured_filters"], count)
        for order in permutations(KINDS):
            _, outputs, _ = self.run_flow(self.flow(order=order))
            self.assertAlmostEqual(outputs["output"][0]["confidence"], .7)

    def test_confidence_is_per_reading_and_is_independent_of_ground_truth(self):
        events = pd.concat([self.events(), self.events()], ignore_index=True)
        events.loc[1, ["value", "state", "_raw_index"]] = [30., "30", 1]
        events["tipo"] = ["normal", "Falla"]
        _, outputs, _ = self.run_flow(self.flow(), events)
        self.assertAlmostEqual(outputs["output"][0]["confidence"], .7)
        self.assertAlmostEqual(outputs["output"][1]["confidence"], .3)
        events["tipo"] = ["Falla", "normal"]
        _, swapped, _ = self.run_flow(self.flow(), events)
        self.assertEqual([item["confidence"] for item in outputs["output"]],
                         [item["confidence"] for item in swapped["output"]])

    def test_executed_discarded_branches_still_contribute_and_logic_is_preserved(self):
        flow = self.flow()
        flow["nodes"].insert(-1, {"id": "gate", "type": "merge", "config": {"mode": "all", "evidence_policy": "combine_v1"}})
        flow["edges"] = [{"source": "input", "target": kind} for kind in KINDS] + [
            {"source": kind, "target": "gate"} for kind in KINDS] + [{"source": "gate", "target": "output"}]
        _, outputs, _ = self.run_flow(flow)
        item = outputs["output"][0]
        self.assertEqual(item["final_decision"], "no_alert")
        self.assertEqual(item["signal"], "negative")
        self.assertAlmostEqual(item["confidence"], .7)
        flow["edges"] = [{"source": "input", "target": kind} for kind in KINDS] + [
            {"source": kind, "target": "gate", "when": ["negative"]} for kind in KINDS] + [
            {"source": "gate", "target": "output"}]
        _, outputs, _ = self.run_flow(flow)
        item = outputs["output"][0]
        self.assertNotIn("contextual", {entry["node_id"] for entry in item["evidence"]})
        self.assertAlmostEqual(item["confidence"], .7)
        self.assertEqual(item["confidence_fusion"]["available_filters"], 3)

    def test_confidence_and_policy_survive_shared_archive_and_restart(self):
        flow = self.flow()
        summary, outputs, snapshot = self.run_flow(flow)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = FlowStore(root / "flows.sqlite3")
            spool = FlowResultSpool("confidence", root / "pending")
            try:
                spool.write_block(outputs)
                run_id, _ = store.save_run_stream(flow["flow_id"], snapshot, summary, spool.records())
                restarted = FlowStore(store.path)
                self.assertEqual(restarted.results(run_id)["items"], outputs["output"])
                self.assertEqual(restarted.list_runs()[0]["snapshot"]["confidence_policy"], snapshot["confidence_policy"])
                old_run = restarted.save_run("old", {}, {}, {"output": [{"observation_id": "old", "final_decision": "normal"}]})
                self.assertNotIn("confidence", restarted.results(old_run)["items"][0])
            finally:
                spool.cleanup()

    def test_causal_csv_and_live_produce_same_confidence_for_one_two_and_three_filters(self):
        future = self.events()
        development = future.copy()
        development["timestamp_dt"] -= pd.Timedelta(minutes=1)
        contextual = Mock()
        contextual.config.window_seconds = 300
        contextual.config.max_lag_windows = 0
        prepared = Mock()
        prepared.observations = {}
        contextual.preparar_eventos_causales.return_value = prepared
        for count in (1, 2, 3):
            results = []
            for live in (False, True):
                blocks = []
                kwargs = {"result_sink": blocks.append}
                if live:
                    kwargs["batch_source"] = iter(future.groupby("timestamp_dt", sort=True))
                with patch("app.flow.evaluacion.MotorReproduccionValidacion.analizar_lote", return_value={}):
                    _, _, snapshot = CausalConfidenceFixture(None, contextual, None).evaluate(
                        self.flow(active=KINDS[:count]), development, future.iloc[:0].copy() if live else future,
                        pd.DataFrame(), {}, {"continuous_method": "residual_focus_v1"}, **kwargs)
                results.append(blocks[0]["output"][0]["confidence"])
                self.assertEqual(snapshot["confidence_policy"]["method"], "uniform_v1")
            self.assertEqual(results[0], results[1])
            self.assertAlmostEqual(results[0], 1 - sum(SUSPICIONS[kind] for kind in KINDS[:count]) / count)


if __name__ == "__main__":
    unittest.main()
