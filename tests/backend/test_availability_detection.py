import unittest

import pandas as pd

from app.flow.definitions import observation_valid
from app.flow.evaluacion import CausalFlowRunner
from app.modulos.estadistico.disponibilidad import AVAILABILITY_SOURCE, availability_detection
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.nodos.handlers import NodeHandlers
from app.services.estados_validos import valid_state
from app.services.reproduccion import MotorReproduccionValidacion


class AvailabilityTests(unittest.TestCase):
    @staticmethod
    def frame(kind="continuous"):
        states = ["unavailable", "unknown", " UnAvailable ", "", None, "NaN", "inf", "-inf", "bad-number", "NONE"]
        return pd.DataFrame({
            "entity_id": "sensor.a", "sensor_kind": kind, "state": states,
            "value": float("nan"), "is_valid_value": False,
            "timestamp_dt": pd.date_range("2026-06-01", periods=len(states), freq="min", tz="UTC"),
            "_raw_index": range(len(states)),
        })

    def test_only_explicit_states_detected_and_original_preserved(self):
        frame = self.frame()
        for row in frame.to_dict("records"):
            with self.subTest(state=row["state"]):
                detection = availability_detection(row)
                if row["_raw_index"] < 3:
                    self.assertEqual(detection["observed_state"], row["state"])
                    self.assertEqual(detection["fault_type"], "unavailable")
                    self.assertEqual(detection["suspected_pattern"], "unavailable")
                    self.assertEqual(detection["detection_sources"], [AVAILABILITY_SOURCE])
                    self.assertIsNone(detection["value"])
                else:
                    self.assertIsNone(detection)

    def test_replay_detects_all_kinds_without_baseline_or_numeric_reading(self):
        for kind in ("continuous", "binary", "multistate"):
            with self.subTest(kind=kind):
                filtro = FiltroEstadistico()
                baseline = {}
                result = MotorReproduccionValidacion(filtro).analizar_lote(self.frame(kind), baseline, .6)
                self.assertEqual([r["source_raw_index"] for r in result["detections"]], [0, 1, 2])
                self.assertEqual(len(result["results"]), 1)
                self.assertTrue(all(o["evaluable"] for o in baseline["last_observations"].values()))

    def test_causal_flow_marks_availability_evaluable_even_without_baseline(self):
        filtro = FiltroEstadistico()
        baseline = {}
        frame = self.frame()
        result = MotorReproduccionValidacion(filtro).analizar_lote(frame, baseline, .6)
        runner = CausalFlowRunner(filtro, None, None)
        runner.statistical_results = {"stat": result}
        runner.baselines = {"stat": baseline}
        incoming = [{**row, "timestamp": row["timestamp_dt"].isoformat(),
                     "source_raw_index": row["_raw_index"], "observation_id": str(row["_raw_index"]), "evidence": []}
                    for row in frame.to_dict("records")]
        output = runner._statistical("stat", incoming, frame, pd.DataFrame(), {})
        for index, item in enumerate(output):
            evidence = item["evidence"][-1]
            observed = evidence["data"]["statistical_observation"]
            self.assertEqual(observed["evaluable"], index < 3)
            self.assertEqual(item["signal"], "positive" if index < 3 else "unknown")
            if index < 3:
                self.assertEqual(evidence["data"]["observed_state"], frame.iloc[index]["state"])
            else:
                self.assertEqual(observed["reason"], "invalid_state")

    def test_none_remains_valid_category_and_other_invalid_values_unevaluable(self):
        self.assertTrue(valid_state("NONE"))
        self.assertTrue(observation_valid({"state": "NONE", "sensor_kind": "multistate", "is_valid_value": True}))
        for value in (None, "", "NaN", float("nan"), float("inf"), float("-inf"), "bad-number"):
            self.assertFalse(observation_valid({"state": value, "value": value, "sensor_kind": "continuous"}))

    def test_historical_and_legacy_flow_detect_availability_before_validity_gate(self):
        frame = self.frame("binary")
        filtro = FiltroEstadistico()
        result = filtro.analizar_historial(frame, pd.DataFrame())
        self.assertEqual([r["source_raw_index"] for r in result["all_results"]], [2, 1, 0])
        handlers = NodeHandlers()
        handlers.statistical = filtro
        incoming = [{**row, "timestamp": row["timestamp_dt"].isoformat(),
                     "source_raw_index": row["_raw_index"], "evidence": []}
                    for row in frame.to_dict("records")]
        output = handlers._statistical("stat", incoming, frame, pd.DataFrame(), {})
        self.assertEqual([item["signal"] for item in output], ["positive"] * 3 + ["unknown"] * 7)

    def test_invalid_timestamp_does_not_create_detection(self):
        self.assertIsNone(availability_detection({"state": "unknown", "timestamp": "invalid"}))


if __name__ == "__main__":
    unittest.main()
