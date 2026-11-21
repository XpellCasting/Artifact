import unittest
from unittest.mock import Mock

import pandas as pd

from app.database.evaluacion_etiquetada import separate_csv
from app.flow.context_replay import ContextReplay
from app.flow.evaluacion import CausalFlowRunner
from app.modulos.contextual.contextual import FiltroContextualEspacioTemporal
from app.services.diagnostico import DiagnosticRunSession
from app.services.historial.cargar_historial import CargadorHistorial


class ContextReplayTests(unittest.TestCase):
    def test_reception_advances_while_evaluation_has_yet_to_complete_a_group(self):
        start = pd.Timestamp("2026-06-01T10:00:00Z")
        later = start + pd.Timedelta(minutes=1)
        between = start + pd.Timedelta(seconds=10)
        candidates = pd.DataFrame({"entity_id": ["sensor.a"] * 2, "timestamp_dt": [start, later]})
        context = pd.DataFrame({"entity_id": ["sensor.b"] * 2, "timestamp_dt": [between, later]})
        now = 0.0
        replay = ContextReplay(context, candidates, start, "contexto.csv", clock=lambda: now)

        self.assertEqual(replay.status()["received"], 0)
        self.assertTrue(replay.take_through(start).empty)
        now = 0.025
        status = replay.status()
        self.assertEqual(status["received"], 1)
        self.assertEqual(status["buffered"], 1)
        self.assertEqual(status["evaluation_percent"], 0)
        self.assertAlmostEqual(status["reception_percent"], 50)

        now = 0.07
        self.assertEqual(replay.status()["buffered"], 2)
        replay.complete_group()
        self.assertEqual(replay.take_through(later)["timestamp_dt"].tolist(), [between, later])
        replay.complete_group()
        status = replay.status()
        self.assertEqual(status["buffered"], 0)
        self.assertEqual(status["evaluation_percent"], 100)

    def test_microsecond_context_accepts_submicrosecond_progress_and_candidate_cutoffs(self):
        start = pd.Timestamp("2026-06-01T10:00:00Z")
        end = start + pd.Timedelta(minutes=1)
        candidates = pd.DataFrame({"entity_id": ["sensor.a"] * 2, "timestamp_dt":
                                   [start, end + pd.Timedelta(nanoseconds=1)]})
        context = pd.DataFrame({"entity_id": ["sensor.b"] * 2, "timestamp_dt":
                                pd.DatetimeIndex([start + pd.Timedelta(seconds=10), end]).as_unit("us")})
        now = 0.0
        replay = ContextReplay(context, candidates, start, "contexto.csv", clock=lambda: now)
        self.assertTrue(replay.take_through(start).empty)

        now = 0.025000000002
        status = replay.status()
        self.assertGreater(pd.Timestamp(status["received_through"]).nanosecond, 0)
        self.assertEqual(status["buffered"], 1)
        replay.complete_group()
        self.assertEqual(len(replay.take_through(candidates["timestamp_dt"].iloc[-1])), 2)
        self.assertEqual(replay.status(advance=False)["buffered"], 0)

    def test_classifier_receives_causal_context_without_a_contextual_node(self):
        flow = {"name": "Clasificador con contexto", "nodes": [
            {"id": kind, "type": kind} for kind in ("input", "classifier", "output")
        ], "edges": [
            {"id": "in", "source": "input", "target": "classifier"},
            {"id": "out", "source": "classifier", "target": "output"},
        ]}
        loader = CargadorHistorial()
        def readings(rows):
            return loader.cargar_real_csv(("entity_id,state,last_changed\n" + "".join(
                f"sensor.a,{state},2026-06-01T10:{minute:02d}:00Z\n" for minute, state in rows
            )).encode(), "eventos.csv")

        development = readings([(0, 10)])
        future = readings([(2, 20), (4, 40)]).eventos
        context = readings([(3, 30), (1, 11), (2, 99)]).eventos
        for use_session in (True, False):
            with self.subTest(use_session=use_session):
                workbench = Mock(spec=["enrich_contextual", "_run_snapshot"] +
                                 (["create_run_session"] if use_session else []))
                workbench._run_snapshot.return_value = {"model": {"window_seconds": 3600}, "references": []}
                if use_session:
                    workbench.create_run_session.side_effect = lambda *args: DiagnosticRunSession(workbench, *args)
                histories = []
                def classify(statistical, contexts, *, events, session, **kwargs):
                    if session is None:
                        histories.append(events["state"].tolist())
                    else:
                        at = pd.Timestamp(contexts[0]["timestamp"])
                        histories.append([r["state"] for r in session.rows_for("sensor.a", at, 3600)])
                    return contexts

                workbench.enrich_contextual.side_effect = classify
                replay = ContextReplay(context, future, development.eventos["timestamp_dt"].max(), "contexto.csv")
                summary, _, _ = CausalFlowRunner(None, None, workbench).evaluate(
                    flow, development.eventos, future, development.entidades, {}, {}, context_source=replay)
                self.assertEqual(histories, [["10", "11", "20"], ["10", "11", "20", "30", "40"]])
                self.assertEqual(summary["candidate_count"], 2)
                self.assertEqual(summary["performance"]["counts"]["context_rows"], 2)

    def test_contextual_filter_uses_causal_context_and_keeps_injected_candidate(self):
        flow = {"name": "Replay con contexto", "nodes": [
            {"id": kind, "type": kind, "config": {"evidence_policy": "independent"}}
            for kind in ("input", "contextual", "output")
        ], "edges": [
            {"id": f"{source}-{target}", "source": source, "target": target}
            for source, target in (("input", "contextual"), ("contextual", "output"))
        ]}
        header = "entity_id,state,last_changed\n"
        loader = CargadorHistorial()
        development = loader.cargar_real_csv((header +
            "sensor.humidity,49,2026-05-31T10:00:00Z\n"
            "sensor.temperature,20,2026-05-31T10:01:00Z\n").encode(), "desarrollo.csv")
        csv, labels = separate_csv((header.strip() + ",es_sintetica,tipo,tipo_falla\n"
            "sensor.temperature,50,2026-06-01T10:05:00Z,false,normal,NULL\n"
            "sensor.temperature,90,2026-06-01T10:06:00Z,true,Falla,spike\n").encode())
        future = loader.cargar_real_csv(csv, "ground_truth.csv").eventos
        context = loader.cargar_real_csv((header +
            "sensor.humidity,50,2026-06-01T10:05:00Z\n"
            "sensor.humidity,60,2026-06-01T10:05:30Z\n"
            "sensor.temperature,60,2026-06-01T10:06:00Z\n"
            "sensor.humidity,99,2026-06-01T10:07:00Z\n").encode(), "contexto.csv").eventos
        truth = [dict(raw_index=i, entity=entity, tipo=kind, synthetic=synthetic, fault_type=fault)
                 for i, entity, kind, synthetic, fault in labels]
        context_map = {"map_version": "contextual-spatiotemporal-v4", "refresh": {"stale": False}, "edges": [{
            "source": "sensor.temperature", "target": "sensor.humidity", "accepted": True,
            "source_kind": "continuous", "target_kind": "continuous", "relation_family": "continuous_continuous",
            "window_seconds": 300, "lag_seconds": 0, "max_age_seconds": 300,
            "association_signed": 1.0, "stability_score": 1.0,
            "expected_response": {"slope": 1.0, "intercept": 0.0, "tolerance": 1.0,
                                  "context_min": 0, "context_max": 100, "method": "fixture"},
        }]}
        replay = ContextReplay(context, future, development.eventos["timestamp_dt"].max(), "contexto.csv")
        runner = CausalFlowRunner(None, FiltroContextualEspacioTemporal(), None)
        args = (flow, development.eventos, future, development.entidades, context_map, {})
        _, without, initial = runner.evaluate(*args, truth=truth)
        summary, with_context, snapshot = runner.evaluate(*args, truth=truth, context_source=replay)

        def evidence(outputs):
            return [item["evidence"][-1]["data"] for item in outputs["contextual"]]

        self.assertEqual(evidence(without)[0]["context_outcome"], "insufficient_coverage")
        improved = evidence(with_context)
        self.assertEqual([item["context_outcome"] for item in improved], ["supported", "contradicted"])
        # The continuous legacy check uses the median of the causal window.
        self.assertEqual([item["evidence"][0]["observed_value"] for item in improved], [50, 55])
        self.assertEqual([item["value"] for item in with_context["output"]], [50, 90])
        self.assertEqual([item["evaluation"]["expected"] for item in with_context["output"]], ["normal", "Falla"])
        self.assertEqual(summary["metrics"]["support"]["total"], 2)
        self.assertEqual(summary["performance"]["counts"]["context_rows"], 2)
        self.assertEqual(snapshot["source"]["fingerprint"], initial["source"]["fingerprint"])


if __name__ == "__main__":
    unittest.main()
