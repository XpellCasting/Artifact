import csv
import hashlib
import io
import json
import platform
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.Inyeccion_Fragmentos import GENERATOR_VERSION, InjectionStore
from app.Revision_Fragmentos import ReviewError, ReviewStore, parse_time


class FragmentInjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.store = ReviewStore(self.base / "review")
        self.injections = InjectionStore(self.store)

    def dataset(self, records=None, partition="validation", filename="source.csv"):
        records = records if records is not None else [
            ["sensor.room", "020.100", "2026-01-01T00:00:00.123456-03:00", "°C", " original,\nvalue "],
            ["binary_sensor.window", "off", "2026-01-01T00:00:00.123456-03:00", "", "context before"],
            ["sensor.room", "21", "2026-01-01T00:01:00.123456-03:00", "°C", " first valid "],
            ["binary_sensor.window", "on", "2026-01-01T00:01:00.123456-03:00", "", "context inside"],
            ["sensor.room", "unknown", "2026-01-01T00:02:00.123456-03:00", "°C", "existing gap"],
            ["sensor.room", "23", "2026-01-01T00:03:00.123456-03:00", "°C", "duplicate timestamp A"],
            ["sensor.room", "24", "2026-01-01T00:03:00.123456-03:00", "°C", "duplicate timestamp B"],
            ["sensor.room", "25", "2026-01-01T00:04:00.123456-03:00", "°C", "last valid"],
            ["binary_sensor.window", "off", "2026-01-01T00:04:00.123456-03:00", "", "context after"],
            ["sensor.unselected", "100", "2026-01-01T00:04:00.123456-03:00", "", "not selected"],
            ["sensor.room", "26", "2026-01-01T00:05:00.123456-03:00", "°C", "padding"],
        ]
        path = self.base / filename
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["entity_id", "state", "last_changed", "unit_of_measurement", "extra"])
            writer.writerows(records)
        return self.store.import_csv(path, filename, partition)

    def baseline_payload(self, dataset, **changes):
        return {
            "dataset_id": dataset["dataset_id"], "entity_id": "sensor.room",
            "context_entity_ids": ["binary_sensor.window"],
            "start": dataset["start"], "end": dataset["end"], "padding_minutes": 0,
            "label": "normal", "status": "draft", "reviewer": "Investigador",
            "evidence_type": "manual_review",
            "evidence": "Fragmento examinado junto con las entidades de contexto.",
            **changes,
        }

    def baseline(self, dataset, **changes):
        return self.store.save_annotation(self.baseline_payload(dataset, **changes))

    def request(self, annotation, **changes):
        return {
            "annotation_id": annotation["annotation_id"],
            "annotation_revision": annotation["revision"], "kind": "bias",
            "start": annotation["start"], "end": annotation["end"], "amount": 3.5,
            "seed": 1729, **changes,
        }

    def archive(self, result):
        with zipfile.ZipFile(self.injections.export_injection(result["injection_id"])) as archive:
            self.assertEqual(set(archive.namelist()), {
                "original.csv", "injected.csv", "interventions.csv", "ground_truth.csv", "manifest.json",
            })
            content = {
                name: list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))
                for name in ("original.csv", "injected.csv", "interventions.csv", "ground_truth.csv")
            }
            content["manifest.json"] = json.loads(archive.read("manifest.json"))
            digests = content["manifest.json"]["export_sha256"]
            self.assertEqual(set(digests), {"original.csv", "injected.csv", "interventions.csv", "ground_truth.csv"})
            for name, expected_digest in digests.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected_digest)
        return content

    def assert_rejected(self, payload, code=400):
        before = self.injections.list_injections()
        with self.assertRaises(ReviewError) as error:
            self.injections.create(payload)
        self.assertEqual(error.exception.status_code, code)
        self.assertEqual(self.injections.list_injections(), before)
        self.assertFalse(any(path.name.startswith(".pending-") for path in self.injections.directory.iterdir()))

    def test_bias_preserves_original_context_padding_raw_fields_and_microseconds(self):
        dataset = self.dataset()
        source_path = self.store.sources / (dataset["dataset_id"] + ".csv")
        original_source = source_path.read_bytes()
        annotation = self.baseline(dataset, start="2026-01-01T03:01:00.123456Z",
                                   end="2026-01-01T03:04:00.123456Z", padding_minutes=1)
        result = self.injections.create(self.request(annotation))
        files = self.archive(result)
        self.assertEqual(source_path.read_bytes(), original_source)
        self.assertEqual(hashlib.sha256(original_source).hexdigest(), result["source_sha256"])
        source_records = list(csv.DictReader(io.StringIO(original_source.decode("utf-8"))))
        expected = [row for row in source_records if row["entity_id"] != "sensor.unselected"]
        self.assertEqual(files["original.csv"], expected)
        changed = []
        for before, after in zip(files["original.csv"], files["injected.csv"], strict=True):
            self.assertEqual({k: v for k, v in before.items() if k != "state"},
                             {k: v for k, v in after.items() if k != "state"})
            eligible = (before["entity_id"] == "sensor.room" and before["state"] != "unknown"
                        and parse_time(annotation["start"])[0] <= parse_time(before["last_changed"])[0]
                        <= parse_time(annotation["end"])[0])
            if eligible:
                self.assertAlmostEqual(float(after["state"]), float(before["state"]) + 3.5)
                changed.append(before)
            else:
                self.assertEqual(after, before)
        self.assertEqual(result["changed_rows"], 4)
        self.assertEqual(len(files["interventions.csv"]), len(changed))
        self.assertEqual(result["exported_rows"], len(expected))
        self.assertTrue(any("1 estados inválidos" in warning for warning in result["warnings"]))
        self.assertEqual(len(files["ground_truth.csv"]), 1)
        self.assertEqual(files["ground_truth.csv"][0]["entity_id"], "sensor.room")
        self.assertEqual(files["ground_truth.csv"][0]["label"], "sensor_fault")
        self.assertEqual(files["manifest.json"]["selected_entities"], ["sensor.room", "binary_sensor.window"])

    def test_noise_repeats_values_with_same_seed_and_changes_with_another_seed(self):
        annotation = self.baseline(self.dataset())
        global_random_state = random.getstate()
        first = self.injections.create(self.request(annotation, kind="noise", amount=0.3, seed=101))
        repeat = self.injections.create(self.request(annotation, kind="noise", amount=0.3, seed=101))
        other = self.injections.create(self.request(annotation, kind="noise", amount=0.3, seed=102))
        first_files, repeat_files, other_files = map(self.archive, (first, repeat, other))
        self.assertEqual(first_files["injected.csv"], repeat_files["injected.csv"])
        self.assertEqual(first_files["interventions.csv"], repeat_files["interventions.csv"])
        self.assertEqual(first["preview"], repeat["preview"])
        self.assertNotEqual(first_files["injected.csv"], other_files["injected.csv"])
        self.assertEqual(first_files["original.csv"], other_files["original.csv"])
        self.assertEqual(first_files["manifest.json"]["seed"], 101)
        self.assertEqual(first_files["manifest.json"]["python_version"], platform.python_version())
        self.assertEqual(random.getstate(), global_random_state)

    def test_drift_uses_elapsed_event_time_for_irregular_and_duplicate_samples(self):
        annotation = self.baseline(self.dataset())
        result = self.injections.create(self.request(annotation, kind="drift", amount=10))
        files = self.archive(result)
        target = [row for row in files["injected.csv"] if row["entity_id"] == "sensor.room"]
        # Five minutes total, with no numeric observation at minute two and two at minute three.
        self.assertEqual([row["state"] for row in target], ["020.100", "23.0", "unknown", "29.0", "30.0", "33.0", "36.0"])
        self.assertEqual(result["changed_rows"], 5)
        self.assertEqual(files["ground_truth.csv"][0]["start"], annotation["start"])
        self.assertEqual(files["ground_truth.csv"][0]["end"], annotation["end"])

    def test_spike_changes_exactly_one_source_row_and_labels_its_timestamp(self):
        annotation = self.baseline(self.dataset())
        result = self.injections.create(self.request(annotation, kind="spike", amount=-7,
                                                      start="2026-01-01T03:02:00.123456Z",
                                                      end="2026-01-01T03:04:00.123456Z"))
        files = self.archive(result)
        changed = files["interventions.csv"]
        self.assertEqual(result["changed_rows"], 1)
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["original_state"], "23")
        self.assertEqual(changed[0]["injected_state"], "16.0")
        self.assertEqual(int(changed[0]["source_row"]), result["spike_source_row"])
        self.assertEqual(result["start"], "2026-01-01T03:03:00.123456+00:00")
        self.assertEqual(result["end"], result["start"])
        self.assertEqual(files["ground_truth.csv"][0]["start"], result["start"])
        self.assertEqual(files["ground_truth.csv"][0]["end"], result["start"])
        self.assertEqual(files["ground_truth.csv"][0]["source_row"], changed[0]["source_row"])
        duplicate = [row for row in files["injected.csv"] if row["extra"] == "duplicate timestamp B"]
        self.assertEqual(duplicate[0]["state"], "24")

    def test_stuck_holds_first_valid_numeric_state_and_preserves_invalid_gaps(self):
        annotation = self.baseline(self.dataset())
        result = self.injections.create(self.request(annotation, kind="stuck",
                                                      start="2026-01-01T03:02:00.123456Z"))
        files = self.archive(result)
        values = [row["state"] for row in files["injected.csv"] if row["entity_id"] == "sensor.room"]
        self.assertEqual(values, ["020.100", "21", "unknown", "23", "23", "23", "23"])
        self.assertEqual(result["changed_rows"], 3)

    def test_unavailable_replaces_valid_target_states_but_retains_existing_invalid_strings(self):
        annotation = self.baseline(self.dataset())
        result = self.injections.create(self.request(annotation, kind="unavailable"))
        files = self.archive(result)
        values = [row["state"] for row in files["injected.csv"] if row["entity_id"] == "sensor.room"]
        self.assertEqual(values, ["unavailable", "unavailable", "unknown", "unavailable", "unavailable", "unavailable", "unavailable"])
        self.assertEqual(result["changed_rows"], 6)
        self.assertTrue(all(row["original_state"] != "unknown" for row in files["interventions.csv"]))

    def test_visual_normal_draft_is_explicitly_exploratory(self):
        result = self.injections.create(self.request(self.baseline(self.dataset())))
        files = self.archive(result)
        self.assertFalse(result["evaluable"])
        self.assertEqual(result["provenance"], "semisynthetic")
        self.assertEqual(files["ground_truth.csv"][0]["evaluable"], "false")
        self.assertFalse(files["manifest.json"]["injection"]["evaluable"])
        self.assertEqual(result["generator_version"], GENERATOR_VERSION)
        self.assertTrue(any("exploratorio" in warning for warning in result["warnings"]))

    def test_verified_baseline_evidence_and_partition_remain_frozen_after_annotation_edit(self):
        dataset = self.dataset(partition="test")
        payload = self.baseline_payload(dataset, status="verified", evidence_type="reference_sensor",
                                        evidence="Sensor independiente R1 contrastado en el registro original.")
        baseline = self.store.save_annotation(payload)
        result = self.injections.create(self.request(baseline))
        archive_bytes = self.injections.export_injection(result["injection_id"]).read_bytes()
        self.store.save_annotation({**payload, "status": "draft", "label": "uncertain",
                                    "evidence": "Revisión posterior: evidencia pendiente de corroboración."}, baseline["annotation_id"])
        stored = self.injections.injection(result["injection_id"])
        files = self.archive(stored)
        self.assertTrue(stored["evaluable"])
        self.assertEqual(stored["baseline_snapshot"], baseline)
        self.assertEqual(files["manifest.json"]["baseline_snapshot"], baseline)
        self.assertEqual(files["ground_truth.csv"][0]["evaluable"], "true")
        self.assertEqual(files["ground_truth.csv"][0]["annotation_revision"], "1")
        self.assertEqual(stored["partition"], "test")
        self.assertEqual(files["manifest.json"]["partition"], "test")
        self.assertEqual(files["ground_truth.csv"][0]["partition"], "test")
        self.assertEqual(self.injections.export_injection(result["injection_id"]).read_bytes(), archive_bytes)

    def test_stale_annotation_revision_returns_conflict_and_creates_no_case(self):
        dataset = self.dataset()
        annotation = self.baseline(dataset)
        self.store.save_annotation(self.baseline_payload(dataset, notes="Segunda revisión documentada."), annotation["annotation_id"])
        self.assert_rejected(self.request(annotation), code=409)

    def test_non_normal_or_undocumented_baselines_cannot_be_used(self):
        dataset = self.dataset()
        for change in ({"label": "uncertain"}, {"label": "sensor_fault"},
                       {"label": "environmental_anomaly"}, {"reviewer": ""}, {"evidence": "Breve"}):
            with self.subTest(change=change):
                annotation = self.baseline(dataset, **change)
                self.assert_rejected(self.request(annotation))

    def test_intervention_must_stay_inside_reviewed_interval_even_when_padding_contains_data(self):
        annotation = self.baseline(self.dataset(), start="2026-01-01T03:01:00.123456Z",
                                   end="2026-01-01T03:04:00.123456Z", padding_minutes=1)
        for change in ({"start": "2026-01-01T03:01:00.123455Z"},
                       {"end": "2026-01-01T03:04:00.123457Z"},
                       {"start": annotation["end"], "end": annotation["start"]},
                       {"start": "2026-01-01T03:01:00.123456"}):
            with self.subTest(change=change):
                self.assert_rejected(self.request(annotation, **change))

    def test_invalid_parameters_and_no_effect_interventions_are_rejected_and_cleaned_up(self):
        annotation = self.baseline(self.dataset())
        cases = [
            {"amount": 0}, {"kind": "noise", "amount": 0}, {"kind": "noise", "amount": -1},
            {"amount": float("nan")}, {"amount": float("inf")}, {"amount": float("-inf")},
            {"seed": -1}, {"seed": 2_147_483_648}, {"annotation_revision": 0},
            {"kind": "not_a_fault"}, {"kind": "drift", "start": annotation["end"]},
            # One invalid reading, one constant value, and a gap without observations.
            {"start": "2026-01-01T03:02:00.123456Z", "end": "2026-01-01T03:02:00.123456Z"},
            {"kind": "stuck", "start": annotation["end"]},
            {"start": "2026-01-01T03:01:01Z", "end": "2026-01-01T03:01:02Z"},
            {"amount": 1e-30},
        ]
        for change in cases:
            with self.subTest(change=change):
                self.assert_rejected(self.request(annotation, **change))

    def test_unknown_fields_cannot_override_ground_truth_partition_or_evaluability(self):
        annotation = self.baseline(self.dataset())
        for name, value in {"partition": "test", "evaluable": True, "label": "normal",
                            "source_sha256": "forged", "target_entity_id": "sensor.other"}.items():
            with self.subTest(field=name):
                self.assert_rejected(self.request(annotation, **{name: value}))

    def test_discrete_replacement_requires_a_valid_observed_state_for_target_entity(self):
        dataset = self.dataset()
        baseline = self.baseline(dataset, entity_id="binary_sensor.window", context_entity_ids=["sensor.room"])
        result = self.injections.create(self.request(baseline, kind="state_replace", replacement_state="on"))
        files = self.archive(result)
        target = [row["state"] for row in files["injected.csv"] if row["entity_id"] == "binary_sensor.window"]
        self.assertEqual(target, ["on", "on", "on"])
        self.assertEqual(result["changed_rows"], 2)
        original_context = [row for row in files["original.csv"] if row["entity_id"] == "sensor.room"]
        injected_context = [row for row in files["injected.csv"] if row["entity_id"] == "sensor.room"]
        self.assertEqual(original_context, injected_context)
        for state in ("opened", "unknown", "unavailable", "NaN", "", "21"):
            with self.subTest(state=state):
                self.assert_rejected(self.request(baseline, kind="state_replace", replacement_state=state))
        for kind in ("bias", "drift", "noise", "spike"):
            with self.subTest(kind=kind):
                self.assert_rejected(self.request(baseline, kind=kind))
        continuous = self.baseline(dataset)
        self.assert_rejected(self.request(continuous, kind="state_replace", replacement_state="21"))

    def test_multistate_replacement_and_stuck_preserve_unknown_states(self):
        records = [["climate.room", state, f"2026-01-01T00:0{index}:00.123456Z", "", "raw"]
                   for index, state in enumerate(("auto", "heat", "unknown", "cool", "auto"))]
        dataset = self.dataset(records)
        annotation = self.baseline(dataset, entity_id="climate.room", context_entity_ids=[])
        self.assertEqual(dataset["entities"][0]["kind"], "multistate")
        for kind, replacement, expected in (
            ("state_replace", "heat", ["heat", "heat", "unknown", "heat", "heat"]),
            ("stuck", "", ["auto", "auto", "unknown", "auto", "auto"]),
        ):
            with self.subTest(kind=kind):
                result = self.injections.create(self.request(annotation, kind=kind, replacement_state=replacement))
                self.assertEqual([row["state"] for row in self.archive(result)["injected.csv"]], expected)

    def test_discrete_replacement_cannot_use_a_state_only_observed_outside_reviewed_baseline(self):
        dataset = self.dataset()
        annotation = self.baseline(dataset, entity_id="binary_sensor.window", context_entity_ids=[],
                                   start=dataset["start"], end=dataset["start"], padding_minutes=2)
        # "on" exists in the source and in exported padding, but was not reviewed as normal here.
        self.assert_rejected(self.request(annotation, kind="state_replace", replacement_state="on"))
        # The only admissible state is already present, so this would make no intervention.
        self.assert_rejected(self.request(annotation, kind="state_replace", replacement_state="off"))

    def test_existing_zip_and_metadata_survive_store_reload_with_dataset_filtering(self):
        first_dataset = self.dataset()
        first = self.injections.create(self.request(self.baseline(first_dataset)))
        other_dataset = self.dataset([["sensor.other", "1", "2026-01-01T00:00:00Z", "", ""]], filename="other.csv")
        archive_bytes = self.injections.export_injection(first["injection_id"]).read_bytes()
        reopened = InjectionStore(ReviewStore(self.store.directory))
        self.assertEqual(reopened.injection(first["injection_id"]), first)
        self.assertEqual(reopened.list_injections(first_dataset["dataset_id"]), [first])
        self.assertEqual(reopened.list_injections(other_dataset["dataset_id"]), [])
        self.assertEqual(reopened.export_injection(first["injection_id"]).read_bytes(), archive_bytes)
        self.assertEqual(self.archive(first)["manifest.json"]["source_sha256"], first_dataset["sha256"])
        for identifier in ("../source.csv", "not_an_id", "0" * 32):
            with self.subTest(identifier=identifier), self.assertRaises(ReviewError) as error:
                reopened.injection(identifier)
            self.assertEqual(error.exception.status_code, 404)

    def test_sampled_preview_retains_an_isolated_injected_spike_and_export_is_complete(self):
        origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
        records = [["sensor.room", str(index % 5), (origin + timedelta(seconds=index, microseconds=123456)).isoformat(), "°C", str(index)]
                   for index in range(1500)]
        dataset = self.dataset(records)
        annotation = self.baseline(dataset, context_entity_ids=[])
        event_time = (origin + timedelta(seconds=777, microseconds=123456)).isoformat()
        result = self.injections.create(self.request(annotation, kind="spike", start=event_time, end=event_time, amount=1000))
        self.assertTrue(result["preview_sampled"])
        self.assertLessEqual(len(result["preview"]), 400)
        changed_points = [point for point in result["preview"] if point["changed"]]
        self.assertEqual(len(changed_points), 1)
        self.assertEqual(changed_points[0]["timestamp"], event_time)
        self.assertEqual(changed_points[0]["injected_state"], "1002.0")
        files = self.archive(result)
        self.assertEqual(len(files["original.csv"]), 1500)
        self.assertEqual(len(files["injected.csv"]), 1500)
        self.assertEqual(len(files["interventions.csv"]), 1)


if __name__ == "__main__":
    unittest.main()
