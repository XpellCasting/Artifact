"""Integration guarantees for the diagnostic stage; F1 and F2 remain unchanged."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.Cargar_Historial.Cargar_Historial import CargadorHistorial
from app.Filtro_Estadistico.Filtro import FiltroEstadistico
from app.main import EstadoAplicacion


def statistical(index=0):
    return {"entity_id": f"sensor.test_{index}",
            "timestamp": (pd.Timestamp("2026-06-01T00:00:00Z") + pd.Timedelta(minutes=index)).isoformat(),
            "value": 20.0, "is_suspicious": True, "suspected_pattern": "deviation"}


def contextual(item, terminal=False, diagnosis=None):
    result = {"entity_id": item["entity_id"], "timestamp": item["timestamp"],
              "context_outcome": "environmental" if terminal else "unexplained",
              "contextual_class": "environmental_anomaly" if terminal else None,
              "routing": {"terminal": terminal, "send_to_classifier": not terminal}}
    if diagnosis:
        result["diagnosis"] = {"status": diagnosis, "model_version": "catalogue-test"}
        result["diagnosis_mode"] = "catalogue"
    return result


class RecordingWorkbench:
    def __init__(self):
        self.calls = []
        self.version = "model-1"
        self.status_by_entity = {}
        self.window_seconds = 3600

    def window_seconds_for_run(self, source, run_id):
        return self.window_seconds

    def enrich_contextual(self, stats, contexts, events=None, source="history", run_id="default"):
        self.calls.append({"stats": deepcopy(stats), "contexts": deepcopy(contexts),
                           "events": events.copy() if events is not None else None,
                           "source": source, "run_id": run_id})
        output = deepcopy(contexts)
        for item in output:
            if not item["routing"]["terminal"]:
                status = self.status_by_entity.get(item["entity_id"], "compatible")
                item["diagnosis"] = {"status": status, "fault_type": "stuck" if status == "compatible" else None,
                                     "model_version": self.version}
                item["diagnosis_mode"] = "active_model"
        return output


class DiagnosticPipelineTests(unittest.TestCase):
    def make_state(self, base):
        return EstadoAplicacion(cargador=CargadorHistorial(), filtro_estadistico=FiltroEstadistico(),
                                static_history_dir=base / "static", real_history_path=base / "history.csv",
                                analysis_cache_path=base / "cache" / "analysis.json")

    def test_summary_is_pure_and_keeps_terminal_environmental_results_out_of_f3(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.make_state(Path(tmp))
            stats = [statistical(i) for i in range(6)]
            stats[-1]["suppressed_by_episode"] = True
            contexts = [contextual(stats[0], terminal=True),
                        contextual(stats[1], diagnosis="compatible"),
                        contextual(stats[2], diagnosis="indeterminate"),
                        contextual(stats[3], diagnosis="no_fault_evidence")]
            before = deepcopy((stats, contexts))
            state.diagnostic_workbench = Mock()
            summary = state._salidas_pipeline(stats, contexts)
            self.assertEqual(summary["environmental_count"], 1)
            self.assertEqual(summary["classification_count"], 3)
            self.assertEqual(summary["indeterminate_count"], 1)
            self.assertEqual(summary["no_fault_count"], 1)
            self.assertEqual(summary["pending_classification_count"], 2)
            self.assertEqual(summary["pipeline_summary"]["grouping"]["suppressed"], 1)
            self.assertEqual((stats, contexts), before)
            self.assertEqual(state.diagnostic_workbench.mock_calls, [])

    def test_catalogue_fallback_does_not_mutate_filter_results_or_classify_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.make_state(Path(tmp))
            stats = [statistical(i) for i in range(2)]
            contexts = [contextual(stats[0], terminal=True), contextual(stats[1])]
            before = deepcopy((stats, contexts))
            with patch("app.Diagnostico_Patrones.diagnose", return_value={"status": "indeterminate"}) as engine:
                result = state._diagnosticar_contextuales(stats, contexts)
            self.assertEqual(engine.call_count, 1)
            self.assertEqual(result[0], contexts[0])
            self.assertEqual(result[1]["diagnosis"]["status"], "indeterminate")
            self.assertEqual((stats, contexts), before)

    def test_explicit_cached_analysis_refreshes_f3_without_recomputing_f1_f2(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.make_state(Path(tmp))
            stats = [statistical()]
            contexts = [contextual(stats[0], diagnosis="indeterminate")]
            cached = {"results": stats, "contextual": {"results": contexts, "summary": {}}, "summary": {}}
            events = pd.DataFrame({"entity_id": [stats[0]["entity_id"]],
                                   "timestamp_dt": pd.to_datetime([stats[0]["timestamp"]], utc=True)})
            state.historial = Mock(eventos=events)
            workbench = RecordingWorkbench()
            state.diagnostic_workbench = workbench
            with patch.object(state, "_leer_cache_analisis_real", side_effect=lambda *args: deepcopy(cached)), \
                 patch.object(state, "_analizar_contextual", side_effect=AssertionError("F2 cache was discarded")), \
                 patch.object(state.filtro_estadistico, "analizar_historial", side_effect=AssertionError("F1 cache was discarded")):
                first = state.analizar_historial(.6)
                workbench.version = "model-2"
                snapshot = state.contexto_actual()
                self.assertEqual(snapshot["classifier_results"][0]["diagnosis"]["model_version"], "model-1")
                self.assertEqual(len(workbench.calls), 1)
                second = state.analizar_historial(.6)
            self.assertEqual(first["classifier_results"][0]["diagnosis"]["model_version"], "model-1")
            self.assertEqual(second["classifier_results"][0]["diagnosis"]["model_version"], "model-2")
            self.assertEqual(len(workbench.calls), 2)
            self.assertNotEqual(workbench.calls[0]["run_id"], workbench.calls[1]["run_id"])

    def test_replay_persists_f3_snapshots_and_cumulative_counts_with_causal_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.make_state(Path(tmp))
            stats = [statistical(i) for i in range(4)]
            events = pd.DataFrame({"entity_id": [item["entity_id"] for item in stats],
                                   "timestamp_dt": pd.to_datetime([item["timestamp"] for item in stats], utc=True),
                                   "_replay_index": range(4)})
            state.historial = Mock(entidades=pd.DataFrame())
            state.historial_validacion = Mock(eventos=events)
            state.eventos_validacion_ordenados = events
            state.validation_baseline = {}
            state.validation_context_map = {}
            state.validation_context_events = events.iloc[:0].copy()
            state.validation_replay_state = {"status": "ready", "rows": 4, "cursor": 0,
                                              "start": stats[0]["timestamp"], "diagnostic_run_id": "replay-1"}
            workbench = RecordingWorkbench()
            state.diagnostic_workbench = workbench

            def fake_f1(nuevos_eventos, **kwargs):
                return {"results": [deepcopy(stats[int(index)]) for index in nuevos_eventos["_replay_index"]], "summary": {}}

            def fake_f2(sospechas, **kwargs):
                return {"results": [contextual(item, terminal=item["entity_id"] == "sensor.test_0") for item in sospechas], "summary": {}}

            workbench.status_by_entity = {"sensor.test_2": "indeterminate", "sensor.test_3": "no_fault_evidence"}
            with patch("app.main.MotorReproduccionValidacion.analizar_lote", side_effect=fake_f1), \
                 patch.object(state.filtro_contextual, "analizar_causal", side_effect=fake_f2):
                first = state.avanzar_reproduccion_validacion(2, .6, suspicion_quiet_seconds=0)
                second = state.avanzar_reproduccion_validacion(2, .6, suspicion_quiet_seconds=0)
            self.assertEqual(first["classification_count"], 1)
            self.assertEqual(second["classification_count"], 3)
            self.assertEqual(second["environmental_count"], 1)
            self.assertEqual(second["indeterminate_count"], 1)
            self.assertEqual(second["no_fault_count"], 1)
            self.assertEqual(second["pending_classification_count"], 1)
            self.assertEqual(second["pipeline_summary"]["classifier"], 3)
            self.assertEqual(len(second["new_classifier_results"]), 2)
            self.assertEqual(workbench.calls[0]["events"]["timestamp_dt"].max(), events.iloc[1]["timestamp_dt"])
            self.assertEqual({call["run_id"] for call in workbench.calls}, {"replay-1"})
            workbench.version = "model-2"
            state.validation_contextual_results.clear()
            page = state.resultados_reproduccion_validacion("classifier", 1, 1)
            self.assertEqual(page["total"], 3)
            self.assertEqual(page["items"][0]["diagnosis"]["model_version"], "model-1")
            self.assertEqual(page["next_offset"], 2)
            state.estado_reproduccion_validacion()
            state.contexto_actual()
            self.assertEqual(len(workbench.calls), 2)

    def test_diagnostic_window_longer_than_f2_is_preserved_and_frozen_per_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.make_state(Path(tmp))
            timestamps = pd.date_range("2026-06-01T00:00:00Z", periods=10, freq="1h")
            events = pd.DataFrame({"entity_id": ["sensor.temperature"] * 10,
                                   "timestamp_dt": timestamps, "state": [str(20 + i / 10) for i in range(10)]})
            state.historial = Mock(eventos=events.iloc[:8].copy())
            state.validation_context_events = state._recortar_eventos_contextuales(state.historial.eventos, timestamps[7])
            self.assertNotIn(timestamps[2], list(state.validation_context_events["timestamp_dt"]))
            workbench = RecordingWorkbench()
            workbench.window_seconds = 6 * 3600
            state.diagnostic_workbench = workbench
            first = state._eventos_diagnosticos_del_lote(events.iloc[8:9], "long-window-run")
            self.assertEqual(first["timestamp_dt"].min(), timestamps[2])
            self.assertEqual(first["timestamp_dt"].max(), timestamps[8])
            self.assertEqual(len(first), 7)
            state.validation_diagnostic_events = state._recortar_eventos_diagnosticos(first, timestamps[8], 6 * 3600)
            workbench.window_seconds = 60
            second = state._eventos_diagnosticos_del_lote(events.iloc[9:], "long-window-run")
            frozen = state.validation_replay_state["diagnostic_window_seconds"]
            self.assertEqual(frozen, 6 * 3600)
            retained = state._recortar_eventos_diagnosticos(second, timestamps[9], frozen)
            self.assertEqual(retained["timestamp_dt"].min(), timestamps[3])
            self.assertEqual(retained["timestamp_dt"].max(), timestamps[9])
            self.assertEqual(len(retained), 7)


if __name__ == "__main__":
    unittest.main()
