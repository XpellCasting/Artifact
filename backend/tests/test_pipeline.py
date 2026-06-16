from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.canonical import generate_static_history  # noqa: E402
from app.estadistico import run_binary_filter  # noqa: E402
from app.pipeline import ArtifactState  # noqa: E402
from app.simulacion import DEFAULT_FAULTS, generate_simulation  # noqa: E402
from app.stat_utils import summarize_numeric  # noqa: E402


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.state = ArtifactState(
            dataset_dir=ROOT.parent / "dataset_simulado" / "output",
            fixture_dir=ROOT / "data" / "fixtures",
        )

    def test_historical_dataset_still_loads_for_compatibility(self):
        loaded = self.state.load_dataset()
        self.assertGreaterEqual(loaded["entities"], 8)
        if not (self.state.fixture_dir / "multistate_entities.csv").exists():
            self.skipTest("Multistate fixture files are not available in this checkout")
        fixture = self.state.load_multistate_fixture()
        self.assertEqual(fixture["entities_added"], 2)

        entities = self.state.list_entities()
        kinds = {item["entity_id"]: item["sensor_kind"] for item in entities}
        self.assertEqual(kinds["sensor.living_temperature"], "continuous")
        self.assertEqual(kinds["binary_sensor.living_motion"], "binary")
        self.assertEqual(kinds["climate.living"], "multistate")

        validation = self.state.validate()
        self.assertEqual(validation["missing_columns"], [])
        self.assertGreater(validation["checked_rows"], 0)

    def test_simulation_pipeline_is_primary_flow(self):
        payload = self.state.generate_simulation(
            duration_hours=12,
            sample_seconds=60,
            seed=99,
            faults=DEFAULT_FAULTS,
        )
        self.assertEqual(payload["config"]["entities"], 5)
        self.assertGreater(payload["ground_truth_labels"].get("sensor_fault", 0), 0)

        validation = self.state.validate()
        self.assertEqual(validation["missing_columns"], [])
        self.assertGreater(validation["checked_rows"], 0)
        normalized = self.state.normalize()
        self.assertEqual(normalized["entities"], 5)
        baseline = self.state.build_baseline()
        self.assertIn("binary_sensor.living_motion", baseline["baseline"])
        self.assertTrue(all(item["evaluation_mode"] == "new_events" for item in baseline["baseline"].values()))
        self.assertEqual(set(self.state.reference_normalized["entity_id"].unique()), set(self.state.event_normalized["entity_id"].unique()))
        self.assertIn("binary_sensor.living_motion", self.state.reference_features)
        self.assertIn("binary_sensor.living_motion", self.state.event_features)
        reference_truth = self.state.simulation_ground_truth[
            self.state.simulation_ground_truth["timestamp"].isin(set(self.state.reference_states["timestamp"]))
        ]
        self.assertEqual(set(reference_truth["label"].unique()), {"normal"})
        results = self.state.run_statistical_filter()
        self.assertGreater(results["results"], 0)
        self.assertIn("first_detection_delay_seconds", results)
        self.assertTrue(all("pattern" in item for item in self.state.results))
        self.assertTrue(all("pattern_detail" in item for item in self.state.results))
        self.assertTrue(all("reading" in item for item in self.state.results))
        self.assertTrue(all(0 <= item["suspicion_score"] <= 1 for item in self.state.results))
        self.assertIn("state_changes", baseline["baseline"]["binary_sensor.living_motion"]["summary"])
        self.assertIn("active_seconds", baseline["baseline"]["binary_sensor.living_motion"]["summary"])
        self.assertIn("max_active_seconds", baseline["baseline"]["binary_sensor.living_motion"]["summary"])

    def test_simulation_exports_csv(self):
        self.state.load_simulation(generate_simulation(duration_hours=4, sample_seconds=60, faults=DEFAULT_FAULTS))
        self.assertIn("entity_id", self.state.export_csv("entities"))
        self.assertIn("timestamp", self.state.export_csv("states"))
        self.assertIn("fault_type", self.state.export_csv("ground_truth"))

    def test_simulation_excludes_unselected_entities_from_filter_and_timeline(self):
        selected = ["sensor.living_temperature", "climate.living"]
        self.state.generate_simulation(
            duration_hours=6,
            sample_seconds=60,
            seed=123,
            faults=DEFAULT_FAULTS,
            selected_entities=selected,
        )
        self.assertEqual(set(self.state.entities), set(selected))
        self.assertEqual(set(self.state.states["entity_id"].unique()), set(selected))
        timeline_ids = {item["entity_id"] for item in self.state.simulation_timeline()["entities"]}
        self.assertEqual(timeline_ids, set(selected))

        self.state.validate()
        self.state.normalize()
        self.state.build_baseline()
        self.state.run_statistical_filter()
        result_ids = {item["entity_id"] for item in self.state.results}
        self.assertTrue(result_ids.issubset(set(selected)))

    def test_binary_timeline_uses_active_percentage_buckets(self):
        self.state.generate_simulation(
            duration_hours=4,
            sample_seconds=10,
            seed=20260614,
            faults=DEFAULT_FAULTS,
            selected_entities=["binary_sensor.living_motion"],
        )
        timeline = self.state.simulation_timeline()["entities"][0]

        self.assertEqual(timeline["chart_kind"], "active_percentage")
        self.assertEqual(timeline["scale_unit"], "%")
        self.assertGreaterEqual(timeline["bucket_seconds"], 60)
        self.assertTrue(timeline["points"])
        self.assertTrue(all(0 <= point["value"] <= 100 for point in timeline["points"]))
        self.assertTrue(all("activations" in point for point in timeline["points"]))

    def test_realtime_timeline_uses_requested_date_range(self):
        history = generate_static_history(duration_days=60, base_sample_seconds=3600, seed=20260614)
        self.state.load_static_history_frames(history.entities, history.events, history.ground_truth, history.metadata)
        self.state.selected_entities = {"sensor.living_temperature"}

        end = pd.to_datetime(history.events["timestamp"], utc=True).max()
        start = end - pd.Timedelta(days=14)
        timeline = self.state.realtime_timeline(
            start=start.isoformat(),
            end=end.isoformat(),
            max_points=2000,
        )
        points = timeline["entities"][0]["points"]
        point_times = pd.to_datetime([point["timestamp"] for point in points], utc=True)

        self.assertGreater(len(points), 24)
        self.assertGreaterEqual(point_times.min(), start)
        self.assertLessEqual(point_times.max(), end)
        self.assertGreaterEqual((point_times.max() - point_times.min()).total_seconds(), 13 * 24 * 3600)

    def test_real_history_low_continuous_value_is_historical_suspicion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=96, freq="15min")
            rows = [
                {
                    "entity_id": "sensor.bathroom_temp_bathroom_humidity",
                    "state": "65.0",
                    "last_changed": timestamp.isoformat(),
                }
                for timestamp in timestamps
            ]
            rows[40]["state"] = "7.1"
            pd.DataFrame(rows).to_csv(path, index=False)

            payload = self.state.load_real_history(str(path))
            low_results = [
                result for result in payload["historical_results"]
                if result["entity_id"] == "sensor.bathroom_temp_bathroom_humidity"
                and float(result["reading"]["value"]) == 7.1
            ]

            self.assertTrue(low_results)
            self.assertEqual(low_results[0]["result_source"], "historical")
            self.assertTrue(low_results[0]["is_suspicious"])

    def test_binary_filter_does_not_repeat_same_window_event(self):
        self.state.generate_simulation(
            duration_hours=24,
            sample_seconds=60,
            seed=20260614,
            faults=DEFAULT_FAULTS,
        )
        self.state.validate()
        self.state.normalize()
        self.state.build_baseline()
        self.state.run_statistical_filter()
        keys = [
            (item["entity_id"], item["timestamp"], tuple(item["triggered_indicators"]))
            for item in self.state.results
            if item["sensor_kind"] == "binary"
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_binary_results_are_not_reported_below_threshold(self):
        disabled_faults = [dict(fault, enabled=False) for fault in DEFAULT_FAULTS]
        self.state.generate_simulation(
            duration_hours=24,
            sample_seconds=10,
            seed=20260614,
            faults=disabled_faults,
            selected_entities=["binary_sensor.living_motion", "binary_sensor.main_door_contact"],
        )
        self.state.validate()
        self.state.normalize()
        self.state.build_baseline()
        self.state.run_statistical_filter()

        binary_scores = [
            item["suspicion_score"]
            for item in self.state.results
            if item["sensor_kind"] == "binary"
        ]
        self.assertTrue(all(score >= 0.5 for score in binary_scores))

    def test_binary_single_normal_activation_is_not_suspicious(self):
        timestamps = pd.date_range("2026-01-01T12:00:00-03:00", periods=7, freq="10s")
        states = ["off", "off", "on", "on", "on", "on", "off"]
        rows = pd.DataFrame(
            {
                "timestamp_dt": timestamps,
                "timestamp_iso": [timestamp.isoformat() for timestamp in timestamps],
                "entity_id": ["binary_sensor.living_motion"] * len(timestamps),
                "sensor_kind": ["binary"] * len(timestamps),
                "state": states,
                "value": [1.0 if state == "on" else 0.0 for state in states],
            }
        )
        zero_summary = summarize_numeric([0.0] * 30)
        base = {
            "summary": {
                "activations": zero_summary,
                "state_changes": zero_summary,
                "active_seconds": zero_summary,
                "active_ratio": zero_summary,
                "max_active_seconds": zero_summary,
                "gap": zero_summary,
                "mean_activation_interval": zero_summary,
                "percentiles": {
                    "activations_p95": 0.0,
                    "state_changes_p95": 0.0,
                    "active_seconds_p95": 0.0,
                    "active_ratio_p95": 0.0,
                    "max_active_seconds_p95": 0.0,
                    "gap_p95": 0.0,
                },
                "window_seconds": 60,
                "step_seconds": 10,
                "threshold": 9.0,
            }
        }
        truth = {
            (timestamp.isoformat(), "binary_sensor.living_motion"): {"label": "normal"}
            for timestamp in timestamps
        }

        results = run_binary_filter(rows, base, truth)

        self.assertEqual(results, [])

    def test_selection_rejects_unknown_entity(self):
        self.state.load_dataset()
        with self.assertRaises(ValueError):
            self.state.select_entities(["sensor.unknown"])

    def write_static_history(self, directory: Path) -> None:
        history = generate_static_history(duration_days=60, base_sample_seconds=3600, seed=20260614)
        directory.mkdir(parents=True, exist_ok=True)
        history.entities.to_csv(directory / "entities.csv", index=False)
        history.events.to_csv(directory / "events.csv", index=False)
        history.ground_truth.to_csv(directory / "ground_truth.csv", index=False)
        history.metadata.to_csv(directory / "metadata.csv", index=False)

    def test_canonical_static_history_is_loaded_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_dir = Path(tmp) / "static_history"
            self.write_static_history(static_dir)
            original_metadata = pd.read_csv(static_dir / "metadata.csv", dtype=str).to_dict("records")[0]
            state = ArtifactState(
                dataset_dir=ROOT.parent / "dataset_simulado" / "output",
                fixture_dir=ROOT / "data" / "fixtures",
                static_history_dir=static_dir,
            )

            status = state.ensure_static_history()
            metadata_after = pd.read_csv(static_dir / "metadata.csv", dtype=str).to_dict("records")[0]

            self.assertFalse(status["created"])
            self.assertTrue(status["exists"])
            self.assertGreaterEqual(int(status["metadata"]["duration_days"]), 60)
            self.assertEqual(status["entity_count"], 5)
            self.assertEqual(original_metadata["created_at"], metadata_after["created_at"])
            ground_truth = pd.read_csv(static_dir / "ground_truth.csv", dtype=str)
            self.assertEqual(set(ground_truth["label"].unique()), {"normal"})
            self.assertIn("tipo_interno", state.static_entities.columns)

    def test_realtime_tick_uses_selected_entities_and_standard_filter_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_dir = Path(tmp) / "static_history"
            self.write_static_history(static_dir)
            state = ArtifactState(
                dataset_dir=ROOT.parent / "dataset_simulado" / "output",
                fixture_dir=ROOT / "data" / "fixtures",
                static_history_dir=static_dir,
            )
            state.ensure_static_history()

            payload = state.realtime_tick(
                selected_entities=["sensor.living_temperature"],
                tick_seconds=60,
                faults=[
                    {
                        "entity_id": "sensor.living_temperature",
                        "fault_type": "random/spike",
                        "enabled": True,
                        "label": "sensor_fault",
                    }
                ],
                suspicion_threshold=0.3,
            )

            self.assertEqual(set(state.realtime_events["entity_id"].unique()), {"sensor.living_temperature"})
            self.assertEqual(set(state.realtime_ground_truth["label"].unique()), {"sensor_fault"})
            self.assertEqual(payload["tick_index"], 1)
            self.assertTrue(payload["faults"])
            self.assertTrue(payload["results"])
            self.assertEqual(state.baseline["sensor.living_temperature"]["evaluation_mode"], "new_events")
            for result in payload["results"]:
                self.assertIn("suspicion_score", result)
                self.assertIn("suspected_pattern", result)
                self.assertEqual(result["filter"], "statistical_temporal")
                self.assertIn("passed_to_contextual_filter", result)
                self.assertIn("tipo_interno", result)
                self.assertTrue(0 <= result["suspicion_score"] <= 1)

    def test_realtime_missing_fault_creates_ground_truth_without_event_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_dir = Path(tmp) / "static_history"
            self.write_static_history(static_dir)
            state = ArtifactState(
                dataset_dir=ROOT.parent / "dataset_simulado" / "output",
                fixture_dir=ROOT / "data" / "fixtures",
                static_history_dir=static_dir,
            )
            state.ensure_static_history()

            for _ in range(2):
                state.realtime_tick(
                    selected_entities=["binary_sensor.main_door_contact"],
                    tick_seconds=60,
                    faults=[
                        {
                            "entity_id": "binary_sensor.main_door_contact",
                            "fault_type": "missing",
                            "enabled": True,
                            "label": "sensor_fault",
                        }
                    ],
                    suspicion_threshold=0.5,
                )

            self.assertTrue(state.realtime_events.empty)
            self.assertEqual(len(state.realtime_ground_truth), 2)
            self.assertEqual(set(state.realtime_ground_truth["fault_type"].unique()), {"missing"})
            self.assertTrue(any(result["suspected_pattern"] == "missing" for result in state.results))


if __name__ == "__main__":
    unittest.main()
