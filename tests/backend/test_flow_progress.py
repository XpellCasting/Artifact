import unittest
from unittest.mock import Mock, patch

import pandas as pd

from app.flow.evaluacion import CausalFlowRunner, FLOW_PROGRESS_INTERVAL_SECONDS, FlowEvaluationJobs


class _Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class FlowProgressTests(unittest.TestCase):
    @staticmethod
    def flow():
        return {"name": "progress fixture", "nodes": [
            {"id": "input", "type": "input", "position": {}, "config": {}},
            {"id": "output", "type": "output", "position": {}, "config": {}},
        ], "edges": [{"id": "input-output", "source": "input", "target": "output"}]}

    def evaluate(self, durations, clock, progress=None, sink=None, fail_at=None):
        at = pd.Timestamp("2026-06-01T00:00:00Z")
        development = pd.DataFrame([{"entity_id": "binary.a", "timestamp_dt": at - pd.Timedelta(minutes=1),
                                     "state": "off", "sensor_kind": "binary"}])
        # Two simultaneous readings per group: count readings, not callbacks.
        future = pd.DataFrame([
            {"entity_id": "binary.a", "timestamp_dt": at + pd.Timedelta(minutes=i),
             "state": state, "sensor_kind": "binary", "_raw_index": i*2+j}
            for i in range(len(durations)) for j, state in enumerate(("off", "on"))
        ])
        truth = [{"raw_index": i, "entity": "binary.a", "tipo": "normal", "synthetic": "false", "fault_type": None}
                 for i in range(len(future))]
        runner = CausalFlowRunner(None, None, None)
        original_run = runner.run
        group = 0

        def timed_run(*args, **kwargs):
            nonlocal group
            if group == fail_at:
                raise RuntimeError("fixture group failed")
            result = original_run(*args, **kwargs)
            clock.now += durations[group]
            group += 1
            return result

        with patch.object(runner, "run", side_effect=timed_run):
            return runner.evaluate(self.flow(), development, future, pd.DataFrame(), {}, {},
                                   truth=truth, progress=progress, result_sink=sink)

    @staticmethod
    def capture(clock, updates):
        def progress(processed, total, phase, detail, **metadata):
            updates.append({"at": clock.now, "processed": processed, "total": total,
                            "phase": phase, "detail": detail, **metadata})
        return progress

    def test_updates_are_spaced_and_final_count_is_exact(self):
        clock, updates = _Clock(), []
        with patch("app.flow.evaluacion.perf_counter", side_effect=clock):
            self.evaluate([.125]*32, clock, self.capture(clock, updates))
        self.assertEqual(updates[0]["phase"], "preparation")
        evaluation = [item for item in updates if item["phase"] == "evaluation"]
        self.assertLessEqual(len(evaluation), 10)  # previously 65 evaluation notifications
        self.assertEqual(evaluation[0]["processed"], 0)
        self.assertEqual(evaluation[0]["current_batch_rows"], 2)
        self.assertIsNotNone(evaluation[0]["group_started_at"])
        for first, second in zip(evaluation, evaluation[1:-1]):
            self.assertGreaterEqual(second["at"] - first["at"], FLOW_PROGRESS_INTERVAL_SECONDS)
        self.assertEqual(evaluation[-1]["processed"], 64)
        self.assertEqual(evaluation[-1]["total"], 64)
        self.assertEqual(evaluation[-1]["groups_completed"], 32)
        self.assertIsNone(evaluation[-1]["group_started_at"])

    def test_fast_evaluation_publishes_final_count_without_waiting(self):
        clock, updates = _Clock(), []
        with patch("app.flow.evaluacion.perf_counter", side_effect=clock):
            self.evaluate([.01]*8, clock, self.capture(clock, updates))
        evaluation = [item for item in updates if item["phase"] == "evaluation"]
        self.assertEqual([item["processed"] for item in evaluation], [0, 16])
        self.assertLess(evaluation[-1]["at"], FLOW_PROGRESS_INTERVAL_SECONDS)
        self.assertIsNone(evaluation[-1]["group_started_at"])

    def test_slow_groups_and_last_short_group_keep_latest_metadata(self):
        clock, updates = _Clock(), []
        with patch("app.flow.evaluacion.perf_counter", side_effect=clock):
            self.evaluate([.75, .01], clock, self.capture(clock, updates))
        evaluation = [item for item in updates if item["phase"] == "evaluation"]
        self.assertEqual([item["processed"] for item in evaluation], [0, 2, 4])
        self.assertEqual(evaluation[1]["last_batch_ms"], 750.)
        self.assertEqual(evaluation[-1]["last_batch_ms"], 10.)
        self.assertEqual(evaluation[-1]["current_timestamp"], "2026-06-01T00:01:00+00:00")

    def test_throttling_preserves_outputs_metrics_and_streamed_results(self):
        regular_clock, throttled_clock, streaming_clock = _Clock(), _Clock(), _Clock()
        with patch("app.flow.evaluacion.perf_counter", side_effect=regular_clock):
            regular, regular_outputs, snapshot = self.evaluate([.125]*8, regular_clock)
        updates, blocks = [], []
        with patch("app.flow.evaluacion.perf_counter", side_effect=throttled_clock):
            throttled, throttled_outputs, throttled_snapshot = self.evaluate(
                [.125]*8, throttled_clock, self.capture(throttled_clock, updates))
        with patch("app.flow.evaluacion.perf_counter", side_effect=streaming_clock):
            streamed, _, streamed_snapshot = self.evaluate([.125]*8, streaming_clock, sink=blocks.append)
        self.assertEqual(regular_outputs, throttled_outputs)
        self.assertEqual(regular_outputs["output"], [item for block in blocks for item in block["output"]])
        for key in ("evaluation", "statistical_evaluation", "node_counts", "candidate_count", "output_count"):
            self.assertEqual(regular[key], throttled[key])
            self.assertEqual(regular[key], streamed[key])
        self.assertEqual(snapshot["source"], throttled_snapshot["source"])
        self.assertEqual(snapshot["source"], streamed_snapshot["source"])

    def test_saving_and_completed_states_are_immediate(self):
        clock, updates = _Clock(), []
        store = Mock()
        jobs = FlowEvaluationJobs(store)
        saving_states = []

        def save(flow_id, snapshot, summary, records):
            saving_states.append(jobs.get(next(iter(jobs.jobs))))
            self.assertTrue(list(records))
            return "fixture-run", 1.

        store.save_run_stream.side_effect = save

        def evaluate(progress, sink):
            def tracked(*args, **kwargs):
                progress(*args, **kwargs)
                updates.append(jobs.get(next(iter(jobs.jobs))))
            return self.evaluate([.01]*8, clock, tracked, sink)

        # Run the actual worker synchronously, without starting the application.
        with patch("app.flow.evaluacion.perf_counter", side_effect=clock), \
             patch("app.flow.evaluacion.Thread") as thread:
            thread.return_value.start.side_effect = lambda: thread.call_args.kwargs["target"]()
            started = jobs.start("fixture-flow", evaluate)
        final = jobs.get(started["job_id"])
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["processed"], 16)
        self.assertEqual(final["run"]["run_id"], "fixture-run")
        self.assertEqual(saving_states[0]["phase"], "saving")
        self.assertEqual(saving_states[0]["processed"], 16)
        self.assertEqual(saving_states[0]["elapsed_seconds"], final["elapsed_seconds"])
        self.assertEqual([state["phase"] for state in updates], ["preparation", "evaluation", "evaluation"])

    def test_failure_is_published_inside_update_interval(self):
        clock, store = _Clock(), Mock()
        jobs = FlowEvaluationJobs(store)

        def evaluate(progress, sink):
            return self.evaluate([.01]*8, clock, progress, sink, fail_at=2)

        with patch("app.flow.evaluacion.perf_counter", side_effect=clock), \
             patch("app.flow.evaluacion.Thread") as thread:
            thread.return_value.start.side_effect = lambda: thread.call_args.kwargs["target"]()
            started = jobs.start("fixture-flow", evaluate)
        state = jobs.get(started["job_id"])
        self.assertLess(clock.now, FLOW_PROGRESS_INTERVAL_SECONDS)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error"], "fixture group failed")
        store.save_run_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
