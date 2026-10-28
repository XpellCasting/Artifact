from copy import deepcopy
from itertools import combinations, product
import unittest

import pandas as pd

from app.flow.evaluacion import labelled_report
from app.flow.evidence import append_evidence
from app.flow.runner import FlowRunner


SIGNALS = {"T": "positive", "F": "negative", "U": "unknown"}
# Three-valued logic: an unevaluable input is unknown, not false.
OR = {"TT": "T", "TF": "T", "TU": "T", "FT": "T", "FF": "F", "FU": "U",
      "UT": "T", "UF": "U", "UU": "U"}
AND = {"TT": "T", "TF": "F", "TU": "U", "FT": "F", "FF": "F", "FU": "F",
       "UT": "U", "UF": "F", "UU": "U"}
POLICIES = ("legacy", "independent", "combine_v1")


class FixtureRunner(FlowRunner):
    def _filter(self, node, incoming, column):
        return [self._with_evidence(
            item, node, SIGNALS[item["state"][column]],
            "not_evaluable" if item["state"][column] == "U" else
            "compatible" if column == 2 and item["state"][column] == "T" else
            "no_fault_evidence" if column == 2 else
            "suspicion" if item["state"][column] == "T" else "no_suspicion", {},
        ) for item in incoming]

    def _statistical(self, node, incoming, *args):
        return self._filter(node, incoming, 0)

    def _contextual(self, node, incoming, *args):
        return self._filter(node, incoming, 1)

    def _classifier(self, node, incoming, *args):
        return self._filter(node, incoming, 2)


class LogicalFlowTests(unittest.TestCase):
    def setUp(self):
        self.item = {"observation_id": "one", "entity_id": "sensor.a",
                     "timestamp": "2026-01-01T00:00:00Z", "signal": "unknown", "evidence": []}

    def reading(self, code, index=0):
        status = {"T": "suspicion", "F": "no_fault_evidence", "U": "not_evaluable"}[code]
        return append_evidence(self.item, f"filter_{index}", "classifier", SIGNALS[code], status, {}, "independent")

    @staticmethod
    def runner(policy):
        runner = FixtureRunner(None, None, None)
        runner._evidence_nodes = {
            "gate": {"type": "merge", "config": {"evidence_policy": policy}},
            "decision": {"type": "decision", "config": {"evidence_policy": policy}},
            "output": {"type": "output", "config": {"evidence_policy": "combine_v1"}},
            "filter": {"type": "classifier", "config": {"evidence_policy": "combine_v1"}},
        }
        return runner

    def test_three_input_gates_follow_truth_tables_under_every_saved_policy(self):
        for codes, policy, mode in product(product("TFU", repeat=3), POLICIES, ("any", "all")):
            with self.subTest(codes=codes, policy=policy, mode=mode):
                table = OR if mode == "any" else AND
                expected = SIGNALS[table[table[codes[0] + codes[1]] + codes[2]]]
                incoming = [self.reading(code, index) for index, code in enumerate(codes)]
                original = deepcopy(incoming)
                runner = self.runner(policy)
                result = runner._merge("gate", incoming, {"mode": mode, "expected_inputs": 3})[0]
                self.assertEqual(result["signal"], expected)
                self.assertEqual(result["logical_evaluation"]["signal"], expected)
                self.assertEqual(result["evidence"][-1]["signal"], expected)
                self.assertEqual(len(result["evidence"]), 4)
                self.assertEqual(incoming, original)
                final = runner._merge("output", [result], {"mode": "output"})[0]
                self.assertEqual(final["signal"], expected)
                self.assertEqual(runner._final_decision(final), {
                    "positive": "suspicion", "negative": "no_alert", "unknown": "indeterminate",
                }[expected])

    def test_missing_branches_are_unknown_and_shared_evidence_is_not_duplicated(self):
        runner = self.runner("combine_v1")
        for code in "TFU":
            for mode, table in (("any", OR), ("all", AND)):
                result = runner._merge("gate", [self.reading(code)], {"mode": mode, "expected_inputs": 3})[0]
                self.assertEqual(result["signal"], SIGNALS[table[code + "U"]])
                self.assertEqual(result["evidence"][-1]["data"]["missing_branches"], 2)
        self.assertEqual(runner._merge("gate", [], {"mode": "all", "expected_inputs": 3}), [])
        shared = self.reading("T")
        result = runner._merge("gate", [shared, shared], {"mode": "all", "expected_inputs": 2})[0]
        self.assertEqual(len(result["evidence"]), 2)
        other = {**self.reading("F", 1), "observation_id": "another"}
        results = runner._merge("gate", [shared, other], {"mode": "any", "expected_inputs": 2})
        self.assertEqual(len(results), 2)
        self.assertEqual({item["observation_id"] for item in results}, {"one", "another"})

    def test_decision_conditions_and_negation_survive_combined_evidence_and_output(self):
        accept_sets = [set(values) for size in (1, 2, 3) for values in combinations(SIGNALS.values(), size)]
        for policy, code, accepted in product(POLICIES, "TFU", accept_sets):
            with self.subTest(policy=policy, code=code, accepted=accepted):
                runner = self.runner(policy)
                incoming = self.reading(code)
                expected = "positive" if SIGNALS[code] in accepted else {"T": "negative", "F": "negative", "U": "unknown"}[code]
                result = runner._decision("decision", [incoming], {"accept": sorted(accepted)})[0]
                self.assertEqual(result["signal"], expected)
                self.assertEqual(result["evidence"][0], incoming["evidence"][0])
                final = runner._merge("output", [result], {"mode": "output"})[0]
                self.assertEqual(final["signal"], expected)

    def test_next_filter_replaces_logic_but_disabled_filter_preserves_it(self):
        runner = self.runner("combine_v1")
        result = runner._decision("decision", [self.reading("F")], {"accept": ["negative"]})[0]
        self.assertEqual(result["signal"], "positive")
        omitted = runner._with_evidence(result, "filter", result["signal"], "omitted", {})
        self.assertEqual(omitted["signal"], "positive")
        self.assertEqual(omitted["logical_evaluation"], result["logical_evaluation"])
        evaluated = runner._with_evidence(result, "filter", "negative", "no_fault_evidence", {})
        self.assertEqual(evaluated["signal"], "negative")
        self.assertNotIn("logical_evaluation", evaluated)

    def test_output_preserves_logic_from_any_incoming_branch_and_reports_disagreement(self):
        runner = self.runner("combine_v1")
        logical = runner._decision("decision", [self.reading("F")], {"accept": ["negative"]})[0]
        for inputs in ([self.reading("T", 1), logical], [logical, self.reading("T", 1)]):
            result = runner._merge("output", inputs, {"mode": "output"})[0]
            self.assertEqual(result["signal"], "positive")
            self.assertEqual(runner._final_decision(result), "suspicion")
        result = runner._merge("output", [self.reading("F", 1), logical], {"mode": "output"})[0]
        self.assertEqual(result["signal"], "unknown")
        self.assertEqual(runner._final_decision(result), "indeterminate")

    @staticmethod
    def flow(formula):
        types = [("input", "input"), ("a", "statistical_focus"), ("b", "contextual"),
                 ("c", "classifier"), ("gate", "merge"), ("output", "output")]
        edges = [("input", node) for node in "abc"]
        config = {"gate": {"mode": "all" if formula == "all" else "any"}}
        if formula == "composed":
            types.append(("second", "merge"))
            config["second"] = {"mode": "all"}
            edges += [("a", "gate"), ("b", "gate"), ("gate", "second"), ("c", "second"), ("second", "output")]
        else:
            edges += [(node, "gate") for node in "abc"]
            if formula == "not":
                types.append(("not", "decision"))
                config["not"] = {"accept": ["negative"]}
                edges += [("gate", "not"), ("not", "output")]
            else:
                edges.append(("gate", "output"))
        return {"nodes": [{"id": node, "type": kind, "config": {
            "evidence_policy": "combine_v1", **config.get(node, {}),
        }} for node, kind in types], "edges": [{"source": source, "target": target} for source, target in edges]}

    def test_composed_three_filter_flows_change_final_metrics_and_preserve_diagnoses(self):
        codes = ["".join(values) for values in product("TFU", repeat=3)]
        events = pd.DataFrame([{"entity_id": "sensor.a", "timestamp_dt": pd.Timestamp(self.item["timestamp"]),
                                "state": code, "sensor_kind": "binary", "_raw_index": index}
                               for index, code in enumerate(codes)])
        truth = [{"raw_index": index, "entity": "sensor.a", "tipo": "normal", "synthetic": "false"}
                 for index in range(len(codes))]
        counts = {}
        for formula in ("any", "all", "composed", "not"):
            runner = FixtureRunner(None, None, None)
            summary, outputs, _ = runner.run(self.flow(formula), events, pd.DataFrame(), {})
            self.assertEqual(summary["output_count"], len(codes))
            expected = []
            for code, result in zip(codes, outputs["output"]):
                first = OR[code[:2]]
                value = AND[AND[code[:2]] + code[2]] if formula == "all" else (
                    AND[first + code[2]] if formula == "composed" else OR[first + code[2]])
                if formula == "not":
                    value = {"T": "F", "F": "T", "U": "U"}[value]
                expected.append(value)
                self.assertEqual(result["signal"], SIGNALS[value], (formula, code))
                self.assertEqual(len([entry for entry in result["evidence"] if entry.get("module_type") in {
                    "statistical_focus", "contextual", "classifier",
                }]), 3)
                if value == "F":
                    self.assertEqual(result["final_decision"], "no_alert")
                elif value == "U":
                    self.assertEqual(result["final_decision"], "indeterminate")
                else:
                    self.assertIn(result["final_decision"], {"suspicion", "fault_compatible"})
            report = labelled_report(outputs["output"], truth)
            counts[formula] = report["groups"][0]["false_positives"]
            self.assertEqual(counts[formula], expected.count("T"))
            self.assertEqual(report["groups"][0]["non_evaluable"], expected.count("U"))
            if formula == "all":
                self.assertEqual(outputs["output"][0]["final_decision"], "fault_compatible")
        self.assertGreater(counts["any"], counts["composed"])
        self.assertGreater(counts["composed"], counts["all"])

    def test_decision_routes_true_false_and_unknown_to_configured_connections(self):
        flow = self.flow("not")
        flow["nodes"].extend({"id": node, "type": "decision", "config": {"evidence_policy": "combine_v1"}}
                             for node in ("true", "false", "unknown"))
        flow["edges"] = [edge for edge in flow["edges"] if edge["source"] != "not"]
        for node, signal in (("true", "positive"), ("false", "negative"), ("unknown", "unknown")):
            flow["edges"].extend([{"source": "not", "target": node, "when": [signal]}, {"source": node, "target": "output"}])
        codes = ["FFF", "TFF", "UFF"]
        events = pd.DataFrame([{"entity_id": "sensor.a", "timestamp_dt": pd.Timestamp(self.item["timestamp"]),
                                "state": code, "sensor_kind": "binary", "_raw_index": index}
                               for index, code in enumerate(codes)])
        _, outputs, _ = FixtureRunner(None, None, None).run(flow, events, pd.DataFrame(), {})
        for node, code in zip(("true", "false", "unknown"), codes):
            self.assertEqual([item["state"] for item in outputs[node]], [code])


if __name__ == "__main__":
    unittest.main()
