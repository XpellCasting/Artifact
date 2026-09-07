import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.Revision_Fragmentos import ReviewError, ReviewStore, parse_time


class ReviewFragmentsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.store = ReviewStore(self.base / "review")

    def source(self, rows, filename="source.csv", headers=None):
        path = self.base / filename
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(headers or ["entity_id", "state", "last_changed", "unit_of_measurement", "extra"])
            writer.writerows(rows)
        return path

    def simple_dataset(self, filename="source.csv", partition="development"):
        source = self.source([
            ["sensor.room", "020.100", "2026-01-01T00:00:00.123456-03:00", "°C", " original "],
            ["binary_sensor.window", "off", "2026-01-01T00:01:00.123456-03:00", "", ""],
            ["sensor.room", "21", "2026-01-01T00:02:00.123456-03:00", "°C", ""],
            ["binary_sensor.window", "on", "2026-01-01T00:03:00.123456-03:00", "", ""],
            ["sensor.room", "22", "2026-01-01T00:04:00.123456-03:00", "°C", filename],
        ], filename)
        return self.store.import_csv(source, filename, partition)

    def payload(self, dataset, **changes):
        return {
            "dataset_id": dataset["dataset_id"], "entity_id": "sensor.room",
            "context_entity_ids": ["binary_sensor.window"],
            "start": dataset["start"], "end": dataset["end"], "padding_minutes": 0,
            **changes,
        }

    def verified(self, dataset, **changes):
        return self.payload(dataset, label="normal", status="verified",
                            reviewer="Investigador", evidence_type="reference_sensor",
                            evidence="Sensor de referencia R1, registro 2026-01-01, concordancia contrastada.", **changes)

    def test_streaming_import_preserves_source_microseconds_and_counts_rejections(self):
        source = self.source([
            ["sensor.room", "020.100", "2026-01-01T00:00:00.123456-03:00", "°C", " keep,\nme "],
            ["sensor.room", "unknown", "2026-01-01T00:00:01.000001-03:00", "°C", ""],
            ["sensor.room", "21", "2026-01-01T00:00:02", "°C", "no timezone"],
            ["sensor.room", "22", "not a date", "°C", "bad timestamp"],
            ["", "23", "2026-01-01T03:00:03Z", "°C", "missing entity"],
            ["sensor.room", "24"],
        ])
        original = source.read_bytes()
        dataset = self.store.import_csv(source, "original.csv", "development")
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual((self.store.sources / (dataset["dataset_id"] + ".csv")).read_bytes(), original)
        self.assertEqual(dataset["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual((dataset["rows"], dataset["indexed_rows"], dataset["rejected_rows"]), (6, 2, 4))
        self.assertEqual(dataset["start"], "2026-01-01T03:00:00.123456+00:00")
        self.assertEqual(dataset["entities"][0]["kind"], "continuous")
        self.assertTrue(dataset["warnings"])
        fragment = self.store.fragment(dataset["dataset_id"], ["sensor.room"], dataset["start"], dataset["end"], 0)
        self.assertEqual(fragment["series"][0]["points"][0]["state"], "020.100")
        self.assertEqual(fragment["series"][0]["points"][0]["timestamp"], "2026-01-01T00:00:00.123456-03:00")
        self.assertEqual(fragment["series"][0]["invalid_rows"], 1)
        self.assertIsNone(fragment["series"][0]["points"][1]["value"])

    def test_same_source_cannot_leak_across_partitions_and_import_is_idempotent(self):
        dataset = self.simple_dataset()
        duplicate = self.store.import_csv(self.base / "source.csv", "renamed.csv", "development")
        self.assertEqual(duplicate, dataset)
        with self.assertRaises(ReviewError) as error:
            self.store.import_csv(self.base / "source.csv", "source.csv", "test")
        self.assertEqual(error.exception.status_code, 409)
        second = self.simple_dataset("second.csv", "test")
        self.assertNotEqual(dataset["dataset_id"], second["dataset_id"])
        self.assertEqual(len(self.store.list_datasets()), 2)

    def test_raw_entity_ids_join_exactly_to_exported_labels(self):
        source = self.source([
            [" sensor.room ", "20", "2026-01-01T00:00:00Z", "°C", "invalid identifier"],
            ["sensor.room", "21", "2026-01-01T00:01:00Z", "°C", "valid identifier"],
        ])
        dataset = self.store.import_csv(source, source.name, "development")
        self.assertEqual(dataset["rejection_summary"]["entity_id_has_outer_whitespace"], 1)
        saved = self.store.save_annotation(self.payload(dataset, context_entity_ids=[]))
        with zipfile.ZipFile(self.store.export_annotation(saved["id"])) as archive:
            rows = list(csv.DictReader(io.StringIO(archive.read("events.csv").decode("utf-8"))))
            truth = list(csv.DictReader(io.StringIO(archive.read("ground_truth.csv").decode("utf-8"))))
        self.assertEqual(rows[0]["entity_id"], truth[0]["entity_id"])

    def test_invalid_upload_rolls_back_without_partial_dataset_or_source(self):
        source = self.source([["sensor.room", "12", "2026-01-01T00:00:00", "", ""]])
        with self.assertRaises(ReviewError):
            self.store.import_csv(source, source.name, "validation")
        self.assertEqual(self.store.list_datasets(), [])
        self.assertEqual(list(self.store.sources.iterdir()), [])

    def test_interval_validation_entity_membership_and_microsecond_bounds(self):
        dataset = self.simple_dataset()
        bad_payloads = [
            self.payload(dataset, start="2026-01-01T03:00:00.123455Z"),
            self.payload(dataset, end="2026-01-01T03:04:00.123457Z"),
            self.payload(dataset, start=dataset["end"], end=dataset["start"]),
            self.payload(dataset, start="2026-01-01T03:00:00.123456"),
            self.payload(dataset, entity_id="sensor.foreign"),
            self.payload(dataset, context_entity_ids=["sensor.foreign"]),
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(ReviewError):
                self.store.save_annotation(payload)
        # An interval with no target samples can document a suspected dropout.
        gap = self.store.save_annotation(self.payload(dataset, start="2026-01-01T03:01:10Z", end="2026-01-01T03:01:20Z"))
        self.assertEqual(gap["label"], "uncertain")

    def test_uncertain_and_visual_review_are_not_evaluable_and_evidence_is_required(self):
        dataset = self.simple_dataset()
        draft = self.store.save_annotation(self.payload(dataset))
        self.assertEqual((draft["label"], draft["status"], draft["evaluable"]), ("uncertain", "draft", False))
        for change in [
            {"label": "uncertain"}, {"evidence_type": "manual_review"},
            {"evidence_type": "insufficient"}, {"evidence_type": "other"},
            {"reviewer": ""}, {"evidence": ""}, {"evidence": "ok"},
        ]:
            payload = {**self.verified(dataset), **change}
            with self.subTest(change=change), self.assertRaises(ReviewError):
                self.store.save_annotation(payload)
        visual = self.store.save_annotation(self.payload(dataset, label="sensor_fault", evidence_type="manual_review", evidence="La curva parece muy irregular."))
        self.assertFalse(visual["evaluable"])
        verified = self.store.save_annotation(self.verified(dataset))
        self.assertTrue(verified["evaluable"])

    def test_persistent_edits_keep_audit_and_cannot_switch_datasets(self):
        first = self.simple_dataset()
        second = self.simple_dataset("second.csv", "validation")
        initial = self.store.save_annotation(self.payload(first))
        reopened = ReviewStore(self.store.directory)
        saved = reopened.save_annotation(self.verified(first), initial["annotation_id"])
        self.assertEqual(saved["revision"], 2)
        self.assertEqual(saved["created_at"], initial["created_at"])
        self.assertEqual(saved["source_sha256"], first["sha256"])
        self.assertEqual(reopened.list_annotations(first["dataset_id"]), [saved])
        self.assertEqual(reopened.list_annotations(second["dataset_id"]), [])
        with reopened.connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM annotation_revisions").fetchone()[0], 2)
        with self.assertRaises(ReviewError) as error:
            reopened.save_annotation(self.payload(second), saved["annotation_id"])
        self.assertEqual(error.exception.status_code, 409)

    def test_conflicting_verified_overlaps_are_rejected_but_drafts_allowed(self):
        dataset = self.simple_dataset()
        first = self.store.save_annotation(self.verified(dataset))
        opposite = {**self.verified(dataset), "label": "sensor_fault"}
        with self.assertRaises(ReviewError) as error:
            self.store.save_annotation(opposite)
        self.assertEqual(error.exception.status_code, 409)
        self.store.save_annotation({**opposite, "status": "draft"})
        # Editing the existing decision is permitted and retained in the audit.
        saved = self.store.save_annotation(opposite, first["annotation_id"])
        self.assertEqual(saved["revision"], 2)

    def test_preview_preserves_extreme_and_gap_export_keeps_all_raw_rows(self):
        origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
        records = []
        for index in range(1200):
            timestamp = (origin + timedelta(seconds=index, microseconds=123456)).isoformat()
            state = "9999.012300" if index == 515 else "unknown" if index == 520 else str(index % 7)
            records.append(["sensor.room", state, timestamp, "°C", f" raw {index}, data "])
            if index % 10 == 0:
                records.append(["binary_sensor.window", "on" if index % 20 else "off", timestamp, "", ""])
        source = self.source(records)
        dataset = self.store.import_csv(source, "long.csv", "test")
        start = (origin + timedelta(seconds=100, microseconds=123456)).isoformat()
        end = (origin + timedelta(seconds=1100, microseconds=123456)).isoformat()
        fragment = self.store.fragment(dataset["dataset_id"], ["sensor.room", "binary_sensor.window"], start, end, 30, 30)
        continuous = fragment["series"][0]
        self.assertTrue(continuous["sampled"])
        self.assertLessEqual(len(continuous["points"]), 30)
        self.assertTrue(any(point["state"] == "9999.012300" for point in continuous["points"]))
        self.assertTrue(any(not point["is_valid"] for point in continuous["points"]))
        self.assertEqual(len(fragment["rows"]), 150)
        self.assertTrue(fragment["preview_truncated"])
        annotation = self.store.save_annotation(self.payload(dataset, start=start, end=end, padding_minutes=30))
        archive_path = self.store.export_annotation(annotation["annotation_id"])
        with zipfile.ZipFile(archive_path) as archive:
            events = list(csv.reader(io.StringIO(archive.read("events.csv").decode("utf-8"))))
            truth = list(csv.DictReader(io.StringIO(archive.read("ground_truth.csv").decode("utf-8"))))
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(len(events) - 1, len(records))
        self.assertEqual(events[1:], records)
        self.assertEqual(len(truth), 1)
        self.assertEqual(truth[0]["entity_id"], "sensor.room")
        self.assertEqual(truth[0]["evaluable"], "false")
        self.assertEqual(truth[0]["start"], start)
        self.assertEqual(manifest["label_interval"]["end"], end)
        self.assertEqual(manifest["target_rows_in_label_interval"], 1001)
        self.assertEqual(manifest["dataset"]["sha256"], dataset["sha256"])
        self.assertNotIn(str(self.store.directory), json.dumps(manifest))
        archive_path.unlink()

    def test_binary_multistate_values_map_back_to_original_strings(self):
        path = self.source([
            ["binary_sensor.window", "off", "2026-01-01T00:00:00Z"],
            ["binary_sensor.window", "on", "2026-01-01T00:01:00Z"],
            ["climate.room", "heat", "2026-01-01T00:00:00Z"],
            ["climate.room", "auto", "2026-01-01T00:01:00Z"],
            ["climate.room", "unavailable", "2026-01-01T00:02:00Z"],
        ], headers=["entity_id", "state", "timestamp"])
        dataset = self.store.import_csv(path, path.name, "development")
        fragment = self.store.fragment(dataset["dataset_id"], ["binary_sensor.window", "climate.room"], dataset["start"], dataset["end"], 0)
        self.assertEqual([item["kind"] for item in fragment["series"]], ["binary", "multistate"])
        for series in fragment["series"]:
            for point in series["points"]:
                if point["is_valid"]:
                    self.assertEqual(series["states"][point["value"]], point["state"])
                else:
                    self.assertIsNone(point["value"])

    def test_submicrosecond_timestamps_are_not_silently_truncated(self):
        with self.assertRaises(ReviewError):
            parse_time("2026-01-01T00:00:00.123456789Z")
        self.assertEqual(parse_time("2026-01-01T00:00:00.123456000Z"),
                         parse_time("2026-01-01T00:00:00.123456Z"))


if __name__ == "__main__":
    unittest.main()
