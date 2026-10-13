from itertools import permutations
import unittest
import pandas as pd
from app.flow.evidence import append_evidence, merge_evidence
from app.flow.runner import FlowRunner


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.item = {"observation_id":"one","entity_id":"a","timestamp":"2026-02-01T00:00:00Z","signal":"unknown","evidence":[]}

    def test_own_result_survives_combination_and_conflicts(self):
        first = append_evidence(self.item,"f1","statistical_focus","positive","suspicion",{},"combine_v1")
        combined = append_evidence(first,"gat","classifier","negative","no_fault_evidence",{},"combine_v1")
        self.assertEqual(combined["own_evaluation"]["label"],"normal")
        self.assertEqual(combined["combined_evaluation"]["label"],"indeterminate")
        self.assertEqual(self.item["evidence"],[])
        self.assertEqual(len(merge_evidence([first,combined])),2)
        self.assertNotIn(combined["evidence"][-1]["evidence_id"],combined["evidence"][-1]["used_evidence_ids"])

    def test_contextual_support_does_not_cancel_fault(self):
        first = append_evidence(self.item,"gat","classifier","positive","compatible",{},"combine_v1")
        last = append_evidence(first,"f2","contextual","negative","supported",{},"combine_v1")
        self.assertEqual(last["combined_evaluation"]["label"],"sensor_fault")

    def test_other_observation_evidence_is_not_consumed(self):
        stale = append_evidence({**self.item,"observation_id":"other"},"gat","classifier","positive","compatible",{},"combine_v1")
        result = append_evidence({**self.item,"evidence":stale["evidence"]},"f1","statistical_focus","negative","no_suspicion",{},"combine_v1")
        self.assertEqual(result["combined_evaluation"]["label"],"no_alert")
        self.assertIn(stale["evidence"][0]["evidence_id"],result["combined_evaluation"]["unused_evidence_ids"])

    def test_all_six_orders_and_standalone_nodes(self):
        class FixtureRunner(FlowRunner):
            def _statistical(self,node,incoming,*args):
                return [self._with_evidence(item,node,"positive","suspicion",{}) for item in incoming]
            def _contextual(self,node,incoming,*args):
                return [self._with_evidence(item,node,"negative","supported",{}) for item in incoming]
            def _classifier(self,node,incoming,*args):
                return [self._with_evidence(item,node,"positive","compatible",{}) for item in incoming]
        events = pd.DataFrame([{"entity_id":"a","timestamp_dt":pd.Timestamp(self.item["timestamp"]),"state":"1","sensor_kind":"continuous","value":1}])
        for order in [*permutations(["statistical_focus","contextual","classifier"]), *[(t,) for t in ["statistical_focus","contextual","classifier"]]]:
            types = ["input",*order,"output"]
            flow = {"nodes":[{"id":t,"type":t,"config":{"evidence_policy":"combine_v1"}} for t in types],
                    "edges":[{"source":a,"target":b} for a,b in zip(types,types[1:])]}
            summary,outputs,_ = FixtureRunner(None,None,None).run(flow,events,pd.DataFrame(),{})
            result = outputs["output"][0]
            self.assertEqual(len(result["evidence"]),len(order)+1)
            if len(order) == 3:
                self.assertEqual(result["final_decision"],"fault_compatible")

    def test_modular_handlers_never_receive_future_history(self):
        class Observer(FlowRunner):
            def _classifier(self,node,incoming,events):
                for item in incoming:
                    assert events.timestamp_dt.max() <= pd.Timestamp(item["timestamp"])
                return [self._with_evidence(item,node,"negative","no_fault_evidence",{}) for item in incoming]
        events = pd.DataFrame([{"entity_id":"a","timestamp_dt":pd.Timestamp(f"2026-02-01T0{hour}:00:00Z"),"state":"1","value":1,"sensor_kind":"continuous"} for hour in range(3)])
        flow = {"nodes":[{"id":t,"type":t,"config":{"evidence_policy":"combine_v1"}} for t in ["input","classifier","output"]],
                "edges":[{"source":"input","target":"classifier"},{"source":"classifier","target":"output"}]}
        summary,_,_ = Observer(None,None,None).run(flow,events,pd.DataFrame(),{})
        self.assertEqual(summary["output_count"],3)


if __name__ == "__main__":
    unittest.main()
