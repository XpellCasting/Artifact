from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.database.flows import FlowStore
from app.flow.measurements import RunMeasurements


class RunMeasurementsTests(unittest.TestCase):
    def test_multiple_sensors_fault_types_and_missed_episodes(self):
        measures = RunMeasurements()
        sequence = [
            ("a", "00:00:00", "Falla", "drift", "no_alert"),
            ("b", "00:00:01", "Falla", "stuck", "not_routed"),
            ("a", "00:00:03", "Falla", "drift", "suspicion"),
            ("a", "00:00:04", "Falla", "drift", "fault_compatible"),
            ("a", "00:00:05", "Falla", "stuck", "environmental"),
            ("a", "00:00:06", "NULL", None, "suspicion"),
            ("a", "00:00:07", "Falla", "stuck", "indeterminate"),
        ]
        for sensor, time, label, kind, decision in sequence:
            measures.observe({"entity_id": sensor, "timestamp": f"2026-01-01T{time}Z", "final_decision": decision},
                             label, kind)
        delay = measures.finalize()["fault_delay_seconds"]
        self.assertEqual((delay["episodes"], delay["detected"], delay["missed"]), (4, 2, 2))
        self.assertEqual((delay["median"], delay["p95"]), (1.5, 2.85))

    def test_persists_latency_and_reconstructs_utc_alerts_and_multiclass_brier(self):
        cases = [
            ("a", "2026-01-01T23:30:00-02:00", "normal", "suspicion", {"normal": 0.7, "environmental_anomaly": 0.2, "sensor_fault": 0.1}),
            ("a", "2026-01-02T02:00:00Z", "anomalia", "fault_compatible", {"normal": 0, "environmental_anomaly": 1, "sensor_fault": 0}),
            ("b", "2026-01-02T02:00:00Z", "anomalia", "environmental", {}),
            ("c", "2026-01-02T02:00:00Z", "NULL", "fault_compatible", {}),
        ]
        outputs = []
        for index, (sensor, timestamp, label, decision, scores) in enumerate(cases):
            entry = {"node_id": "model", "module_type": "classifier", "data": {"class_scores": scores}}
            outputs.append({"observation_id": str(index), "entity_id": sensor, "timestamp": timestamp,
                            "evaluation": {"expected": label}, "final_decision": decision, "evidence": [entry, entry]})
        with TemporaryDirectory() as path:
            store = FlowStore(Path(path) / "flow.sqlite3")
            run = store.save_run("flow", {}, {"candidate_count": 4}, {"output": outputs})
            hist = store.run_metrics(run)["measurements"]
            self.assertIsNone(hist["resources"])
            self.assertIsNone(hist["latency_ms"]["median"])
            self.assertEqual(hist["false_alerts_by_sensor_day"]["days"], [
                {"sensor": "a", "day": "2026-01-02", "valid_readings": 2, "false_alerts": 2, "false_fault_alerts": 1},
                {"sensor": "b", "day": "2026-01-02", "valid_readings": 1, "false_alerts": 0, "false_fault_alerts": 0},
            ])
            self.assertEqual(hist["false_alerts_by_sensor_day"]["mean"], 1)
            self.assertEqual(hist["brier"]["by_node"]["model"]["count"], 2)
            self.assertAlmostEqual(hist["brier"]["by_node"]["model"]["value"], .07)

            for row, latency in zip(outputs, (10, 40, 40, 70)):
                row["latency_ms"] = latency
            run = store.save_run("flow", {}, {"candidate_count": 4}, {"output": outputs})
            latency = store.run_metrics(run)["measurements"]["latency_ms"]
            self.assertEqual((latency["count"], latency["median"], latency["p95"]), (4, 40, 65.5))
            self.assertEqual(store.results(run)["items"][2]["latency_ms"], 40)
