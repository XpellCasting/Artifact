import copy
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from app.services.diagnostico import DiagnosticRunSession
from app.flow.evaluacion import CausalFlowRunner
from app.modulos.contextual.contextual import FiltroContextualEspacioTemporal
from app.database.flows import FlowStore


class _SessionWorkbench:
    def _run_snapshot(self, source, run_id):
        return {"run_id": run_id, "model": {"method": "fixture", "window_seconds": 10},
                "references": []}

    def _list(self, name):
        return []


class CausalHistoryTests(unittest.TestCase):
    def test_context_index_incremental_matches_full_index_and_keeps_window_boundary(self):
        filtro = FiltroContextualEspacioTemporal()
        entities = pd.DataFrame([{"entity_id": "binary.a", "domain": "binary_sensor"}])
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        rows = [
            {"entity_id": "binary.a", "timestamp_dt": start + pd.Timedelta(seconds=second),
             "state": state, "sensor_kind": "binary", "_raw_index": index}
            for index, (second, state) in enumerate([
                (0, "off"), (5, "on"), (10, "unavailable"), (10, "on"),
                (20, "off"), (30, "on"), (40, "off"), (50, "on"),
            ])
        ]
        events = pd.DataFrame(rows)
        initial = filtro.preparar_eventos_causales(entities, events.iloc[:3])
        initial.advance(events.iloc[3:5], start + pd.Timedelta(seconds=10))
        initial.advance(events.iloc[5:7], start + pd.Timedelta(seconds=30))
        initial.advance(events.iloc[7:], start + pd.Timedelta(seconds=40))

        cutoff = start + pd.Timedelta(seconds=40)
        older = events[events.timestamp_dt < cutoff].sort_values("timestamp_dt", kind="stable").groupby(
            "entity_id", sort=False).tail(1)
        expected_rows = pd.concat([older, events[events.timestamp_dt >= cutoff]], ignore_index=True)
        prepared = filtro._preparar_eventos(expected_rows)
        catalog = filtro._catalogo(entities, prepared)
        expected = filtro._indexar_observaciones(prepared, catalog)
        actual = initial.observations
        self.assertEqual(set(actual), set(expected))
        for entity_id in expected:
            pd.testing.assert_frame_equal(actual[entity_id], expected[entity_id])

    def test_diagnostic_session_prunes_only_outside_inclusive_window(self):
        session = DiagnosticRunSession(_SessionWorkbench(), "test", "one")
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        for second in range(41):
            session.add_events(pd.DataFrame([{
                "entity_id": "sensor.a", "timestamp_dt": base + pd.Timedelta(seconds=second),
                "state": str(second), "_raw_index": second,
            }]), events_sorted=True)
        self.assertLessEqual(session.stats["buffered_event_rows"], 11)
        rows = session.rows_for("sensor.a", base + pd.Timedelta(seconds=40), 10)
        self.assertEqual([row["source_row"] for row in rows], list(range(30, 41)))
        other = DiagnosticRunSession(_SessionWorkbench(), "test", "two")
        self.assertEqual(other.stats["buffered_event_rows"], 0)
        self.assertEqual(other.rows_for("sensor.a", base + pd.Timedelta(seconds=40), 10), [])


class StreamingFlowTests(unittest.TestCase):
    @staticmethod
    def _flow():
        return {"name": "fixture", "nodes": [
            {"id": "input", "type": "input", "position": {}, "config": {}},
            {"id": "output", "type": "output", "position": {}, "config": {}},
        ], "edges": [{"id": "input-output", "source": "input", "target": "output"}]}

    def _run(self, sink=None):
        development = pd.DataFrame([{
            "entity_id": "sensor.a", "timestamp_dt": pd.Timestamp("2025-12-31T23:59:00Z"),
            "state": "0", "sensor_kind": "continuous",
        }])
        future = pd.DataFrame([
            {"entity_id": "sensor.a", "timestamp_dt": pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(seconds=i),
             "state": str(i), "sensor_kind": "continuous", "_raw_index": i, "tipo": "normal",
             "es_sintetica": "false", "tipo_falla": None}
            for i in range(3)
        ])
        truth = [{"raw_index": i, "entity": "sensor.a", "tipo": "normal", "synthetic": "false",
                  "fault_type": None} for i in range(3)]
        return CausalFlowRunner(None, None, None).evaluate(
            self._flow(), development, future, pd.DataFrame(), {}, {}, truth=truth, result_sink=sink)

    def test_streamed_outputs_and_reports_match_compatibility_return(self):
        regular_summary, regular_outputs, regular_snapshot = self._run()
        blocks = []
        streamed_summary, no_outputs, streamed_snapshot = self._run(blocks.append)
        self.assertIsNone(no_outputs)
        streamed_output = [item for block in blocks for item in block["output"]]
        self.assertEqual(streamed_output, regular_outputs["output"])
        self.assertEqual(streamed_summary["evaluation"], regular_summary["evaluation"])
        self.assertEqual(streamed_summary["node_counts"], regular_summary["node_counts"])
        self.assertEqual(streamed_snapshot["source"], regular_snapshot["source"])

    def test_stream_persistence_is_atomic_on_constraint_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FlowStore(Path(tmp) / "isolated.sqlite3")
            flow = store.create(self._flow())
            good = {"node_id": "output", "sequence": 0, "observation_id": "same",
                    "payload": {"observation_id": "same"}}
            duplicate = copy.deepcopy(good)
            with self.assertRaises(Exception):
                store.save_run_stream(flow["flow_id"], {}, {}, iter([good, duplicate]))
            self.assertEqual(store.list_runs(flow["flow_id"]), [])
            run_id, _ = store.save_run_stream(flow["flow_id"], {}, {}, iter([good]))
            self.assertEqual(store.results(run_id, "output")["total"], 1)


if __name__ == "__main__":
    unittest.main()
