from copy import deepcopy
from itertools import combinations, permutations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from app.database.flows import FlowStore
from app.flow.evaluacion import CausalFlowRunner, FlowEvaluationJobs
from app.flow.execution import FILTER_TYPES
from app.flow.result_spool import FlowResultSpool, decode_observation, encode_observation
from app.flow.runner import FlowRunner


KINDS = ("statistical_focus", "contextual", "classifier")


class FixtureRunner(FlowRunner):
    def _filter(self, node_id, incoming):
        kind = self._evidence_nodes[node_id]["type"]
        if kind == "contextual" and getattr(self, "fail_context", False):
            raise RuntimeError("context fixture failed")
        signal, status = "negative", {"statistical_focus": "no_suspicion", "contextual": "supported",
                                       "classifier": "no_fault_evidence"}[kind]
        data = {"version": "fixture-v2", "observed_value": 24.5}
        if kind == "contextual" and getattr(self, "missing_context", False):
            signal, status, data = "unknown", "not_evaluable", {"reason": "missing_neighbors"}
        if kind == "classifier" and getattr(self, "abstain", False):
            signal, status = "unknown", "indeterminate"
            data["class_scores"] = {"normal": .4, "environmental_anomaly": .3, "sensor_fault": .3}
        return [self._with_evidence(item, node_id, signal, status, data) for item in incoming]

    def _statistical(self, node_id, incoming, *args):
        return self._filter(node_id, incoming)

    def _contextual(self, node_id, incoming, *args):
        return self._filter(node_id, incoming)

    def _classifier(self, node_id, incoming, *args):
        return self._filter(node_id, incoming)


class CausalFixtureRunner(FixtureRunner, CausalFlowRunner):
    pass


class FilterTraceTests(unittest.TestCase):
    @staticmethod
    def flow(active=KINDS, configured=KINDS, order=None, policy="combine_v1"):
        ordered = tuple(order or configured)
        nodes = [{"id": "input", "type": "input", "config": {"evidence_policy": policy}}]
        nodes += [{"id": kind, "type": kind, "config": {"enabled": kind in active,
                   "evidence_policy": policy}} for kind in ordered]
        nodes += [{"id": "output", "type": "output", "config": {"evidence_policy": policy}}]
        identifiers = [node["id"] for node in nodes]
        return {"flow_id": "flow-fixture", "name": "Trace fixture", "nodes": nodes,
                "edges": [{"source": source, "target": target} for source, target in zip(identifiers, identifiers[1:])]}

    @staticmethod
    def events(count=1):
        return pd.DataFrame([{"entity_id": "sensor.a", "timestamp_dt": pd.Timestamp("2026-11-17T20:00:00Z"),
                              "sensor_kind": "continuous", "state": "24.5", "value": 24.5,
                              "_raw_index": index} for index in range(count)])

    def run_flow(self, flow=None, runner=None, events=None):
        return (runner or FixtureRunner(None, None, None)).run(
            flow or self.flow(), events if events is not None else self.events(), pd.DataFrame(), {})

    def test_all_combinations_of_one_two_and_three_active_filters(self):
        for count in (1, 2, 3):
            for active in combinations(KINDS, count):
                for policy in ("legacy", "independent", "combine_v1"):
                    with self.subTest(active=active, policy=policy):
                        flow = self.flow(active=active, policy=policy)
                        original = deepcopy(flow)
                        _, outputs, snapshot = self.run_flow(flow)
                        item = outputs["output"][0]
                        trace = item["execution_trace"]
                        self.assertEqual(trace["schema_version"], "filter-execution-v1")
                        self.assertEqual(trace["execution_order"], list(active))
                        self.assertEqual(trace["availability_mask"], {kind: kind in active for kind in KINDS})
                        self.assertEqual(len(trace["evidence"]), count)
                        for kind in KINDS:
                            record = trace["filters"][kind]
                            self.assertEqual(record["execution_status"], "completed" if kind in active else "disabled")
                            self.assertEqual(bool(record["evidence_ids"]), kind in active)
                        self.assertEqual(flow, original)
                        self.assertEqual(snapshot["flow"], original)
                        self.assertEqual(snapshot["execution_context"]["config_hash"], item["config_hash"])
                        self.assertEqual(item["flow_id"], "flow-fixture")
                        self.assertEqual(len(item["config_hash"]), 64)
                        self.assertFalse(item["confidence_calibrated"])
                        self.assertEqual(item["confidence_fusion"]["method"], "uniform_v1")

    def test_unconfigured_filters_are_identified_without_fabricating_evidence(self):
        for count in (1, 2, 3):
            for configured in combinations(KINDS, count):
                _, outputs, _ = self.run_flow(self.flow(configured=configured, active=configured))
                trace = outputs["output"][0]["execution_trace"]
                self.assertEqual(set(trace["filters"]), set(configured))
                self.assertEqual(trace["absent_filter_types"], sorted(FILTER_TYPES - set(configured)))

    def test_all_execution_orders_retain_actual_input_evidence_and_versions(self):
        for order in permutations(KINDS):
            _, outputs, _ = self.run_flow(self.flow(order=order))
            item = outputs["output"][0]
            trace = item["execution_trace"]
            self.assertEqual(trace["execution_order"], list(order))
            self.assertEqual(item["final_decision"], "normal")
            evidence = {entry["evidence_id"]: entry for entry in trace["evidence"]}
            for index, kind in enumerate(order):
                record = trace["filters"][kind]
                self.assertEqual(record["sequence"], index)
                self.assertEqual(record["module_version"], "fixture-v2")
                self.assertEqual(evidence[record["evidence_ids"][0]]["module_type"], kind)
                if index:
                    previous = trace["filters"][order[index-1]]["evidence_ids"][0]
                    self.assertIn(previous, record["input_evidence_ids"])

    def test_executed_branch_excluded_from_output_remains_in_trace(self):
        flow = self.flow(active=KINDS[:2], configured=KINDS[:2])
        flow["edges"] = [{"source": "input", "target": kind} for kind in KINDS[:2]] + [
            {"source": "statistical_focus", "target": "output", "when": ["positive"]},
            {"source": "contextual", "target": "output"}]
        _, outputs, _ = self.run_flow(flow)
        item = outputs["output"][0]
        self.assertNotIn("statistical_focus", {entry["node_id"] for entry in item["evidence"]})
        self.assertEqual({entry["node_id"] for entry in item["execution_trace"]["evidence"]}, set(KINDS[:2]))
        self.assertEqual(set(item["execution_trace"]["execution_order"]), set(KINDS[:2]))

    def test_disabled_and_routed_away_are_distinct_even_when_never_reached(self):
        flow = self.flow(active=KINDS[:2])
        flow["edges"][1]["when"] = ["positive"]
        flow["edges"].append({"source": "statistical_focus", "target": "output", "when": ["negative"]})
        _, outputs, _ = self.run_flow(flow)
        trace = outputs["output"][0]["execution_trace"]
        self.assertEqual(trace["execution_order"], ["statistical_focus"])
        self.assertEqual(trace["filters"]["contextual"]["execution_status"], "skipped")
        self.assertEqual(trace["filters"]["contextual"]["reason"], "not_routed_to_filter")
        self.assertEqual(trace["filters"]["classifier"]["execution_status"], "disabled")

    def test_logical_gate_inputs_have_resolvable_evidence_references(self):
        flow = self.flow()
        flow["nodes"].insert(-2, {"id": "gate", "type": "merge", "config": {"evidence_policy": "combine_v1"}})
        flow["edges"] = [
            {"source": "input", "target": kind} for kind in KINDS[:2]] + [
            {"source": kind, "target": "gate"} for kind in KINDS[:2]] + [
            {"source": "gate", "target": "classifier"}, {"source": "classifier", "target": "output"}]
        _, outputs, _ = self.run_flow(flow)
        trace = outputs["output"][0]["execution_trace"]
        evidence_ids = {entry["evidence_id"] for entry in trace["evidence"]}
        gate = next(entry for entry in trace["evidence"] if entry["module_type"] == "merge")
        self.assertIn(gate["evidence_id"], trace["filters"]["classifier"]["input_evidence_ids"])
        for record in trace["filters"].values():
            self.assertTrue(set(record["evidence_ids"] + record["input_evidence_ids"]) <= evidence_ids)
        self.assertEqual(len(trace["execution_order"]), 3)

    def test_insufficient_input_and_classifier_abstention_do_not_replace_diagnoses(self):
        runner = FixtureRunner(None, None, None)
        runner.missing_context, runner.abstain = True, True
        _, outputs, _ = self.run_flow(runner=runner)
        item = outputs["output"][0]
        trace = item["execution_trace"]
        self.assertEqual(trace["filters"]["contextual"]["execution_status"], "insufficient_data")
        self.assertEqual(trace["filters"]["contextual"]["reason"], "missing_neighbors")
        self.assertFalse(trace["availability_mask"]["contextual"])
        self.assertEqual(trace["filters"]["classifier"]["execution_status"], "completed")
        diagnosis = next(entry for entry in trace["evidence"] if entry["node_id"] == "classifier")
        self.assertEqual(diagnosis["status"], "indeterminate")
        self.assertEqual(diagnosis["data"]["class_scores"]["sensor_fault"], .3)

    def test_shared_call_timing_is_grouped_and_observations_stay_separate(self):
        _, outputs, _ = self.run_flow(events=self.events(2))
        first, second = outputs["output"]
        self.assertNotEqual(first["observation_id"], second["observation_id"])
        self.assertEqual(first["execution_id"], second["execution_id"])
        self.assertEqual(first["execution_trace"]["groups"], second["execution_trace"]["groups"])
        for group in first["execution_trace"]["groups"]:
            self.assertEqual(group["timing_scope"], "filter_call")
            self.assertEqual(group["observation_count"], 2)
            self.assertGreaterEqual(group["duration_ms"], 0)
        first["execution_trace"]["filters"]["classifier"]["reason"] = "changed"
        self.assertIsNone(second["execution_trace"]["filters"]["classifier"]["reason"])

    def test_configuration_hash_is_stable_but_tracks_processing_changes(self):
        flow = self.flow()
        _, outputs, snapshot = self.run_flow(flow)
        original_hash = outputs["output"][0]["config_hash"]
        presentation = deepcopy(flow)
        presentation["name"] = "Renamed"
        for node in presentation["nodes"]:
            node["position"] = {"x": 100, "y": 200}
            node["config"] = dict(reversed(list(node["config"].items())))
            node["config"]["label"] = "Display name"
        _, renamed, _ = self.run_flow(presentation)
        self.assertEqual(renamed["output"][0]["config_hash"], original_hash)
        changed = deepcopy(flow)
        changed["nodes"][1]["config"]["suspicion_threshold"] = .7
        _, updated, _ = self.run_flow(changed)
        self.assertNotEqual(updated["output"][0]["config_hash"], original_hash)
        self.assertNotEqual(updated["output"][0]["execution_id"], outputs["output"][0]["execution_id"])
        flow["nodes"][1]["config"]["enabled"] = False
        self.assertTrue(snapshot["flow"]["nodes"][1]["config"]["enabled"])

    def test_archived_trace_round_trips_and_reuses_the_evidence_pool(self):
        flow = self.flow()
        summary, outputs, snapshot = self.run_flow(flow)
        records = [{"node_id": node, "sequence": 0, "payload": items[0]} for node, items in outputs.items()]
        blob, _, unique = encode_observation(records)
        body = decode_observation(blob)
        self.assertEqual(unique, 4)  # Three filters and Output, shared with trace.
        self.assertIn("trace_evidence", body["nodes"][-1])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = FlowStore(root / "flows.sqlite3")
            spool = FlowResultSpool("trace", root / "pending")
            try:
                spool.write_block(outputs)
                run_id, _ = store.save_run_stream(flow["flow_id"], snapshot, summary, spool.records())
                restarted = FlowStore(store.path)
                restored = restarted.results(run_id)["items"][0]
                self.assertEqual(restored, outputs["output"][0])
                restored["execution_trace"]["evidence"][0]["data"]["observed_value"] = 999
                self.assertEqual(restarted.results(run_id)["items"][0], outputs["output"][0])
                self.assertEqual(restarted.list_runs()[0]["snapshot"]["execution_context"], snapshot["execution_context"])
            finally:
                spool.cleanup()

    def test_filter_errors_keep_fail_fast_and_expose_partial_trace(self):
        runner = FixtureRunner(None, None, None)
        runner.fail_context = True
        with self.assertRaisesRegex(RuntimeError, "context fixture failed") as caught:
            self.run_flow(runner=runner)
        context = caught.exception.filter_execution_context
        self.assertEqual(context["failed_filter"], "contextual")
        trace = context["observations"][0]["execution_trace"]
        self.assertEqual(trace["execution_order"], ["statistical_focus", "contextual"])
        self.assertEqual(trace["filters"]["contextual"]["execution_status"], "error")
        self.assertEqual(trace["filters"]["classifier"]["reason"], "execution_aborted")
        self.assertEqual(len(trace["evidence"]), 1)

    def test_error_sample_is_bounded_and_contains_the_affected_readings(self):
        runner = FixtureRunner(None, None, None)
        runner.fail_context = True
        with self.assertRaises(RuntimeError) as caught:
            self.run_flow(runner=runner, events=self.events(25))
        context = caught.exception.filter_execution_context
        self.assertEqual(context["observation_count"], 25)
        self.assertTrue(context["truncated"])
        self.assertEqual(len(context["observations"]), 20)
        for item in context["observations"]:
            self.assertEqual(item["execution_trace"]["filters"]["contextual"]["execution_status"], "error")

    def test_failed_job_exposes_filter_trace_without_publishing_a_completed_run(self):
        with tempfile.TemporaryDirectory() as folder:
            store = FlowStore(Path(folder) / "flows.sqlite3")
            jobs = FlowEvaluationJobs(store)
            runner = FixtureRunner(None, None, None)
            runner.fail_context = True
            def evaluate(progress, sink):
                return self.run_flow(runner=runner)
            with patch("app.flow.evaluacion.Thread") as worker:
                started = jobs.start("flow-fixture", evaluate, flow=self.flow())
            with self.assertLogs("app.flow.evaluacion", level="ERROR"):
                worker.call_args.kwargs["target"]()
            job = jobs.get(started["job_id"])
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["filter_execution_context"]["failed_filter"], "contextual")
            self.assertEqual(store.list_runs(), [])

    def test_causal_csv_and_live_batches_keep_execution_identity_and_timing_scope(self):
        for count in (1, 2, 3):
            flow = self.flow(active=KINDS[:count])
            future = self.events(4)
            future.loc[2:, "timestamp_dt"] += pd.Timedelta(minutes=1)
            development = self.events()
            development["timestamp_dt"] -= pd.Timedelta(minutes=1)
            baseline = {"continuous_method": "residual_focus_v1"}
            contextual = Mock()
            contextual.config.window_seconds = 300
            contextual.config.max_lag_windows = 0
            prepared = Mock()
            prepared.observations = {}
            contextual.preparar_eventos_causales.return_value = prepared
            for live in (False, True):
                with self.subTest(active=count, live=live):
                    blocks = []
                    runner = CausalFixtureRunner(None, contextual, None)
                    kwargs = {"result_sink": blocks.append}
                    evaluation_rows = future
                    if live:
                        evaluation_rows = future.iloc[:0].copy()
                        kwargs["batch_source"] = iter(future.groupby("timestamp_dt", sort=True))
                    with patch("app.flow.evaluacion.MotorReproduccionValidacion.analizar_lote", return_value={}):
                        summary, _, snapshot = runner.evaluate(
                            flow, development, evaluation_rows, pd.DataFrame(), {}, baseline, **kwargs)
                    rows = [item for block in blocks for item in block["output"]]
                    self.assertEqual(len(rows), 4)
                    self.assertEqual(summary["candidate_count"], 4)
                    self.assertEqual({row["execution_id"] for row in rows}, {snapshot["execution_context"]["execution_id"]})
                    self.assertEqual(snapshot["execution_mode"], "causal_live" if live else "causal_future")
                    self.assertEqual(rows[0]["latency_group_id"], rows[1]["latency_group_id"])
                    self.assertNotEqual(rows[0]["latency_group_id"], rows[2]["latency_group_id"])
                    first_groups = {group["group_id"] for group in rows[0]["execution_trace"]["groups"]}
                    last_groups = {group["group_id"] for group in rows[2]["execution_trace"]["groups"]}
                    self.assertFalse(first_groups & last_groups)
                    for row in rows:
                        self.assertEqual(row["latency_scope"], "timestamp_group")
                        self.assertEqual(row["latency_group_size"], 2)
                        self.assertEqual(row["execution_trace"]["execution_order"], list(KINDS[:count]))
                        statistical = row["execution_trace"]["groups"][0]
                        self.assertEqual(statistical["preparation"]["observation_count"], 2)
                        self.assertEqual(statistical["preparation"]["timing_scope"], "timestamp_group")
                        self.assertGreaterEqual(statistical["preparation"]["duration_ms"], 0)


if __name__ == "__main__":
    unittest.main()
