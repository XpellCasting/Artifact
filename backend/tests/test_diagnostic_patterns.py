import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.Diagnostico_Patrones import (
    CATALOG, FEATURE_NAMES, build_reference, diagnose, extract_features, fit_model, predict_features,
)


class DiagnosticPatternTests(unittest.TestCase):
    entity = "sensor.temperature"
    normal_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    case_start = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def rows(self, values, start=None, seconds=60):
        start = start or self.case_start
        return [{"entity_id": self.entity, "timestamp": (start + timedelta(seconds=seconds * i)).isoformat(),
                 "state": str(value), "source_row": i + 2} for i, value in enumerate(values)]

    def reference(self, values=None, kind="continuous"):
        values = values if values is not None else [20 + math.sin(i / 2) * .2 for i in range(240)]
        return build_reference(self.rows(values, self.normal_start), self.entity, kind)

    def diagnose_values(self, values, **kwargs):
        rows = self.rows(values)
        statistical = {"entity_id": self.entity, "timestamp": rows[-1]["timestamp"], "sensor_kind": "continuous",
                       "is_suspicious": True, "suspected_pattern": "this_input_must_not_decide"}
        return diagnose(statistical, {"support_score": 0, "contradiction_score": .6, "coverage": {"ratio": .8}},
                        kwargs.pop("reference", self.reference()), rows, **kwargs)

    def test_reference_robustness_nonmutation_and_cadence(self):
        rows = self.rows([20, 20.1, 19.9, 20.2, 19.8, 10000], self.normal_start)
        original = copy.deepcopy(rows)
        reference = build_reference(rows, self.entity, "continuous")
        self.assertAlmostEqual(reference["stats"]["center"], 20.05)
        self.assertLess(reference["stats"]["scale"], 1)
        self.assertEqual(reference["stats"]["cadence_seconds"], 60)
        self.assertEqual(rows, original)
        self.assertTrue(reference["limitations"])
        json.dumps(reference, allow_nan=False)

    def test_spike_requires_observed_recovery_and_is_not_bias(self):
        values = [20 + math.sin(i) * .05 for i in range(20)]
        values[-2] = 24
        spike = self.diagnose_values(values)
        self.assertEqual(spike["fault_type"], "spike", spike)
        self.assertEqual(spike["confidence_kind"], "compatibility_not_probability")
        before_recovery = self.diagnose_values(values[:-1])
        self.assertNotEqual(before_recovery["fault_type"], "spike")
        bias = self.diagnose_values([23 + math.sin(i) * .05 for i in range(20)])
        self.assertEqual(bias["fault_type"], "bias", bias)

    def test_drift_uses_actual_time_and_noise_requires_fluctuation(self):
        drift = self.diagnose_values([20 + i * .1 for i in range(30)])
        self.assertEqual(drift["fault_type"], "drift", drift)
        noise = self.diagnose_values([20 + (1 if i % 2 else -1) * (1 + i / 100) for i in range(30)])
        self.assertEqual(noise["fault_type"], "noise", noise)

    def test_stuck_requires_reported_repetition_and_nonflat_baseline(self):
        frozen = self.diagnose_values([20.1] * 20)
        self.assertEqual(frozen["fault_type"], "stuck", frozen)
        naturally_constant = self.diagnose_values([20.1] * 20, reference=self.reference([20.1] * 240))
        self.assertIsNone(naturally_constant["fault_type"])
        sparse = self.rows([20.1, 20.1, 20.1], seconds=1200)
        reference = self.reference()
        result = diagnose({"entity_id": self.entity, "sensor_kind": "continuous", "timestamp": sparse[-1]["timestamp"]}, None, reference, sparse)
        self.assertIsNone(result["fault_type"])

    def test_future_data_excluded_even_microseconds_and_temporal_reference_rejected(self):
        reference = self.reference()
        rows = self.rows([20, 20.1, 20.2])
        cutoff = rows[-1]["timestamp"]
        future = {"timestamp": (datetime.fromisoformat(cutoff) + timedelta(microseconds=1)).isoformat(), "state": "9999"}
        self.assertEqual(extract_features(rows, reference, cutoff), extract_features(rows + [future], reference, cutoff))
        with self.assertRaisesRegex(ValueError, "antes"):
            extract_features(rows, {**reference, "end": cutoff}, cutoff)
        # A long actual time window differs from a row-count window.
        self.assertEqual(extract_features(rows, reference, cutoff, 30)["sample_count"], 1)

    def test_invalid_and_unknown_data_abstain_without_manufacturing_reports(self):
        unavailable = self.diagnose_values([20, 20.1, "unknown", "unavailable"])
        self.assertEqual(unavailable["fault_type"], "unavailable")
        rows = self.rows([20, 20.1, 20.2])
        later = (self.case_start + timedelta(hours=4)).isoformat()
        features = extract_features(rows, self.reference(), later)
        self.assertEqual(features["sample_count"], 0)
        self.assertIsNone(features["constant_duration_seconds"])
        result = diagnose({"entity_id": self.entity}, None)
        self.assertEqual(result["status"], "indeterminate")
        self.assertTrue(result["needs_review"])
        json.dumps(unavailable, allow_nan=False)

    def test_invalid_time_and_unusable_reference_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "zona"):
            build_reference([{"timestamp": "2026-01-01T00:00:00", "state": "20"}], self.entity, "continuous")
        with self.assertRaisesRegex(ValueError, "microsegundos"):
            build_reference([{"timestamp": "2026-01-01T00:00:00.1234567Z", "state": "20"}], self.entity, "continuous")
        with self.assertRaisesRegex(ValueError, "tres"):
            self.reference(["nan", "inf", "unknown"])
        with self.assertRaisesRegex(ValueError, "positiva"):
            extract_features(self.rows([20, 21]), self.reference(), window_seconds=0)

    def test_discrete_unknown_state_and_valid_rare_behavior(self):
        reference = self.reference(["off", "on"] * 120, "binary")
        rows = self.rows(["off", "on"] + ["invalid_mode"] * 5)
        result = diagnose({"entity_id": self.entity, "timestamp": rows[-1]["timestamp"], "sensor_kind": "binary"}, None, reference, rows)
        self.assertEqual(result["fault_type"], "state_replace", result)
        # Absent reports on event-driven devices do not imply frozen/unavailable.
        sparse = self.rows(["off", "on", "on", "on", "on"], seconds=900)
        result = diagnose({"entity_id": self.entity, "timestamp": sparse[-1]["timestamp"], "sensor_kind": "binary"}, None, reference, sparse)
        self.assertIsNone(result["fault_type"])

    def test_f2_terminal_and_lack_of_context_cannot_confirm_fault(self):
        rows = self.rows([20.1] * 20)
        statistical = {"entity_id": self.entity, "timestamp": rows[-1]["timestamp"], "sensor_kind": "continuous"}
        result = diagnose(statistical, {"routing": {"terminal": True}}, self.reference(), rows)
        self.assertEqual(result["status"], "no_fault_evidence")
        self.assertFalse(result["needs_review"])
        self.assertEqual(result["pattern_matches"], [])
        ordinary = self.rows([20 + math.sin(i) * .1 for i in range(30)])
        result = diagnose({**statistical, "timestamp": ordinary[-1]["timestamp"]}, {"coverage": {"ratio": 0}}, self.reference(), ordinary)
        self.assertIsNone(result["fault_type"])
        self.assertTrue(any("ausencia de respaldo" in reason for reason in result["reasons"]))

    def examples(self):
        return [{"example_id": f"{label}-{i}", "group_id": f"episode-{label}-{i}", "partition": "development",
                 "sensor_kind": "continuous", "label": label,
                 "features": {"level_z": base + i * .01, "noise_ratio": base + i * .01,
                              "constant_duration_ratio": base + i * .01, "seed": 12345, "kind": label}}
                for label, base in [("normal", .1), ("noise", 5)] for i in range(4)]

    def test_model_serializable_reproducible_allowlist_and_classifies_patterns(self):
        examples = self.examples()
        model = fit_model(examples)
        self.assertEqual(fit_model(examples)["model_version"], model["model_version"])
        json.dumps(model, allow_nan=False)
        self.assertNotIn("seed", model["feature_names"]["continuous"])
        self.assertNotIn("kind", model["feature_names"]["continuous"])
        for row in examples:
            prediction = predict_features(row["features"], row["sensor_kind"], model)
            self.assertEqual(prediction["predicted_label"], row["label"])
        changed_labels_only = copy.deepcopy(examples[4]["features"])
        changed_labels_only.update(seed=-9000, kind="normal", label="normal", injection_type="stuck", original_value=1e10)
        self.assertEqual(predict_features(changed_labels_only, "continuous", model)["fault_type"], "noise")

    def test_model_rejects_eval_leakage_missing_classes_unknown_and_missing_features(self):
        examples = self.examples()
        for partition in ("validation", "test", None):
            with self.assertRaisesRegex(ValueError, "desarrollo"):
                fit_model([{**examples[0], "partition": partition}, *examples[1:]])
        with self.assertRaises(ValueError):
            fit_model(examples[:4])
        model = fit_model(examples)
        unknown = {"level_z": 10000, "noise_ratio": 10000, "constant_duration_ratio": 10000}
        self.assertEqual(predict_features(unknown, "continuous", model)["status"], "indeterminate")
        self.assertEqual(predict_features({"level_z": 5}, "continuous", model)["status"], "indeterminate")
        self.assertEqual(predict_features(unknown, "binary", model)["status"], "indeterminate")
        midpoint = {"level_z": 2.565, "noise_ratio": 2.565, "constant_duration_ratio": 2.565}
        self.assertEqual(predict_features(midpoint, "continuous", model)["status"], "indeterminate")

    def test_environmental_model_class_is_no_fault_evidence(self):
        examples = self.examples()
        faults = copy.deepcopy(examples[4:])
        for row in faults:
            row["features"] = {"level_z": 20, "noise_ratio": 20, "constant_duration_ratio": 20}
        for row in examples[4:]:
            row["label"] = "environmental_anomaly"
        prediction = predict_features(examples[-1]["features"], "continuous", fit_model(examples + faults))
        self.assertEqual(prediction["predicted_label"], "environmental_anomaly")
        self.assertEqual(prediction["status"], "no_fault_evidence")
        self.assertIsNone(prediction["fault_type"])


if __name__ == "__main__":
    unittest.main()
