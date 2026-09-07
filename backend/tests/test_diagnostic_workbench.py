import copy
import csv
import io
import json
import math
import os
import tempfile
import time
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.Diagnostico_Trabajo import DiagnosticWorkbench, _write
from app.Diagnostico_Patrones import extract_features, predict_features
from app.Revision_Fragmentos import ReviewError


class DiagnosticWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workbench = DiagnosticWorkbench(Path(self.temp.name) / "review")

    def source(self, day=1, partition="development", offset=0):
        path = Path(self.temp.name) / f"{day}-{partition}-{offset}.csv"
        start = datetime(2026, 1, day, tzinfo=timezone.utc)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["entity_id", "state", "timestamp", "unit_of_measurement"])
            for index in range(121):
                writer.writerow(["sensor.room", 20 + offset + .2 * math.sin(index / 3),
                    (start + timedelta(minutes=index)).isoformat(), "°C"])
        return self.workbench.review.import_csv(path, path.name, partition)

    def annotation(self, source, **changes):
        return self.workbench.review.save_annotation(dict(dataset_id=source["dataset_id"],
            entity_id="sensor.room", start=source["start"], end=source["end"], label="normal",
            status="verified", reviewer="Investigador", evidence_type="controlled_action",
            evidence="Registro de acción controlada en fixture sintético de prueba.", **changes))

    def dataset(self, day=1, partition="development", offset=0):
        annotation = self.annotation(self.source(day, partition, offset))
        injection = self.workbench.injections.create(dict(annotation_id=annotation["annotation_id"],
            annotation_revision=annotation["revision"], kind="bias", amount=5, seed=42,
            start=annotation["start"], end=annotation["end"]))
        dataset = self.workbench.build_dataset(dict(injection_ids=[injection["injection_id"]],
            window_seconds=600, stride_seconds=600))
        return dataset, annotation, injection

    def test_build_train_export_and_validation_use_fixed_training_reference(self):
        workbench = self.workbench
        training, _, _ = self.dataset()
        heldout, _, _ = self.dataset(2, "validation", offset=2)
        model = workbench.train(training["dataset_id"])
        self.assertEqual(workbench.status()["mode"], "catalogue")
        self.assertEqual(model["classes"], ["bias", "normal"])
        with self.assertRaises(ReviewError):
            workbench.activate(model["model_id"])
        # Poison the precomputed held-out features: evaluation MUST re-extract
        # using training references and immutable source/ZIP observations.
        stored = workbench._read("datasets", heldout["dataset_id"])
        for example in stored["examples"]:
            example["features"] = {"level_z": 999999}
        _write(workbench._path("datasets", heldout["dataset_id"]), stored)
        with patch("app.Diagnostico_Trabajo.extract_features", wraps=extract_features) as extracting:
            evaluated = workbench.evaluate(model["model_id"], heldout["dataset_id"])
        self.assertEqual(extracting.call_count, heldout["example_count"])
        for args in extracting.call_args_list:
            self.assertEqual(args.args[1]["partition"], "development")
        self.assertEqual(evaluated["validation"]["reference_policy"], "fixed_training_references")
        self.assertEqual(evaluated["validation"]["count"], heldout["example_count"])
        self.assertTrue(evaluated["validated"])
        workbench.activate(model["model_id"])
        self.assertEqual(workbench.status()["active_model_id"], model["model_id"])
        archive_path = workbench.export_dataset(training["dataset_id"])
        with zipfile.ZipFile(archive_path) as archive:
            rows = list(csv.DictReader(io.StringIO(archive.read("examples.csv").decode())))
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(len(rows), training["example_count"])
        self.assertTrue(any(key.startswith("feature.") for key in rows[0]))
        self.assertNotIn("feature.seed", rows[0])
        self.assertIn("semantics", manifest)
        archive_path.unlink()

    def test_only_development_trains_and_shared_periods_cannot_evaluate(self):
        workbench = self.workbench
        development, _, _ = self.dataset()
        heldout, _, _ = self.dataset(2, "validation")
        dev = workbench._read("datasets", development["dataset_id"])
        validation = workbench._read("datasets", heldout["dataset_id"])
        dev["examples"] += validation["examples"]
        _write(workbench._path("datasets", development["dataset_id"]), dev)
        trained = workbench.train(development["dataset_id"])
        self.assertEqual(trained["training_count"], development["example_count"])
        self.assertTrue(trained["training_end"].startswith("2026-01-01"))
        for example in validation["examples"]:
            example["group_id"] = trained["training_groups"][0]
        _write(workbench._path("datasets", heldout["dataset_id"]), validation)
        with self.assertRaisesRegex(ReviewError, "compartidos"):
            workbench.evaluate(trained["model_id"], heldout["dataset_id"])
        self.assertTrue(workbench._leaks(dict(group_id="a", entity_id="s", baseline_start="2026-01-01T00:00Z", baseline_end="2026-01-02T00:00Z", window_start="2026-01-01T00:00Z", window_end="2026-01-02T00:00Z"),
            dict(group_id="b", entity_id="s", baseline_start="2026-01-02T00:00Z", baseline_end="2026-01-03T00:00Z", window_start="2026-01-02T00:00Z", window_end="2026-01-03T00:00Z")))

    def test_deactivate_preserves_model_evaluation_and_existing_run(self):
        workbench = self.workbench
        dataset, _, _ = self.dataset()
        heldout, _, _ = self.dataset(2, "validation")
        model_id = workbench.train(dataset["dataset_id"])["model_id"]
        workbench.evaluate(model_id, heldout["dataset_id"])
        workbench.activate(model_id)
        snapshot = workbench._run_snapshot("validation_replay", "before")
        # Evaluation remains available even while the model is active.
        workbench.evaluate(model_id, heldout["dataset_id"])
        before = workbench._read("models", model_id)
        self.assertFalse(workbench.deactivate(model_id)["active"])
        self.assertEqual(workbench._read("models", model_id), before)
        self.assertEqual(workbench.status()["mode"], "catalogue")
        self.assertEqual(workbench._run_snapshot("validation_replay", "before"), snapshot)
        self.assertIsNone(workbench._run_snapshot("validation_replay", "after")["model"])
        self.assertFalse(workbench.deactivate(model_id)["active"])
        reevaluated = workbench.evaluate(model_id, heldout["dataset_id"])
        self.assertEqual(len(reevaluated["evaluations"]), 3)
        self.assertTrue(workbench.activate(model_id)["active"])
        other = workbench.train(dataset["dataset_id"])["model_id"]
        workbench.activate(other, allow_exploratory=True)
        with self.assertRaises(ReviewError) as error:
            workbench.deactivate(model_id)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(workbench.status()["active_model_id"], other)

    def test_training_reports_excluded_incomplete_windows(self):
        dataset, _, _ = self.dataset()
        saved = self.workbench._read("datasets", dataset["dataset_id"])
        short = copy.deepcopy(saved["examples"][0])
        short.update(example_id="short-window", label="spike")
        short["features"]["sample_count"] = 1
        saved["examples"].append(short)
        _write(self.workbench._path("datasets", dataset["dataset_id"]), saved)
        trained = self.workbench.train(dataset["dataset_id"])
        self.assertEqual(trained["training_count"], dataset["example_count"])
        self.assertEqual(trained["excluded_count"], 1)
        self.assertNotIn("spike", trained["classes"])

    def test_run_pins_model_terminal_skip_and_future_data_do_not_change_decision(self):
        workbench = self.workbench
        dataset, _, _ = self.dataset()
        workbench.window_seconds_for_run("validation_replay", "old")
        model = workbench.train(dataset["dataset_id"])
        workbench.activate(model["model_id"], allow_exploratory=True)
        self.assertIsNone(workbench._run_snapshot("validation_replay", "old")["model"])
        self.assertEqual(workbench.window_seconds_for_run("validation_replay", "new"), 600)
        timestamp = "2026-01-02T01:00:00Z"
        stat = [{"entity_id": "sensor.room", "timestamp": timestamp, "sensor_kind": "continuous", "is_suspicious": True}]
        ctx = [{"entity_id": "sensor.room", "timestamp": timestamp, "routing": {"terminal": False}}]
        events = pd.DataFrame([dict(entity_id="sensor.room", state=str(25 + .2 * math.sin(i)),
            timestamp=(datetime(2026, 1, 2, 0, 50, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat()) for i in range(11)])
        result = workbench.enrich_contextual(stat, ctx, events, "validation_replay", "new")
        future = pd.concat([events, pd.DataFrame([dict(entity_id="sensor.room", state="99999", timestamp="2026-01-03T00:00:00Z")])])
        second = workbench.enrich_contextual(stat, ctx, future, "validation_replay", "new")
        self.assertEqual(result, second)
        self.assertNotIn("diagnosis", ctx[0])
        self.assertEqual(workbench.results()["total"], 1)
        terminal = workbench.enrich_contextual(stat, [{**ctx[0], "routing": {"terminal": True}}], events, "validation_replay", "new")
        self.assertNotIn("diagnosis", terminal[0])
        before = [{**stat[0], "timestamp": "2026-01-01T01:00:00Z"}]
        earlier = workbench.enrich_contextual(before, [{**ctx[0], "timestamp": before[0]["timestamp"]}], events, "validation_replay", "earlier")
        self.assertEqual(earlier[0]["diagnosis"]["status"], "indeterminate")

    def test_fault_type_persisted_exported_and_rejected_for_normal(self):
        source = self.source()
        payload = dict(dataset_id=source["dataset_id"], entity_id="sensor.room", start=source["start"], end=source["end"],
            label="sensor_fault", fault_type="drift", reviewer="Investigador", evidence="Deriva documentada en fixture")
        saved = self.workbench.review.save_annotation(payload)
        self.assertEqual(self.workbench.review.annotation(saved["annotation_id"])["fault_type"], "drift")
        archive_path = self.workbench.review.export_annotation(saved["annotation_id"])
        with zipfile.ZipFile(archive_path) as archive:
            truth = list(csv.DictReader(io.StringIO(archive.read("ground_truth.csv").decode())))
        self.assertEqual(truth[0]["fault_type"], "drift")
        archive_path.unlink()
        with self.assertRaises(ReviewError):
            self.workbench.review.save_annotation({**payload, "label": "normal"})

    def test_jobs_recover_after_error_and_drafts_require_explicit_exploratory(self):
        workbench = self.workbench
        source = self.source()
        annotation = self.annotation(source)
        payload = {key: annotation[key] for key in ("dataset_id", "entity_id", "start", "end", "label", "reviewer", "evidence")}
        draft = workbench.review.save_annotation(payload)
        with self.assertRaisesRegex(ReviewError, "borradores"):
            workbench.build_dataset(dict(annotation_ids=[draft["annotation_id"]]))
        dataset = workbench.build_dataset(dict(annotation_ids=[draft["annotation_id"]], include_drafts=True))
        self.assertTrue(dataset["exploratory"])
        job = workbench.start_job(workbench.train, dataset["dataset_id"])
        deadline = time.monotonic() + 5
        while job["job_id"] in workbench.running_jobs and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(workbench.job(job["job_id"])["status"], "failed")
        self.assertTrue(workbench.job_lock.acquire(blocking=False))
        workbench.job_lock.release()

    def test_atomic_updates_tolerate_brief_windows_reader_lock(self):
        path = self.workbench.directory / "lock-test.json"
        _write(path, {"state": "old"})
        replace = os.replace
        calls = []
        def transient_lock(source, target):
            calls.append(target)
            if len(calls) == 1:
                raise PermissionError("File held by a reader")
            replace(source, target)
        with patch("app.Diagnostico_Trabajo.os.replace", side_effect=transient_lock):
            _write(path, {"state": "new"})
        self.assertEqual(json.loads(path.read_text()), {"state": "new"})
        self.assertEqual(len(calls), 2)
        self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_progress_polling_does_not_break_job_file_updates(self):
        def progress_job(progress):
            for index in range(100):
                progress(index, "Actualizando")
            return {"done": True}
        workbench = self.workbench
        job = workbench.start_job(progress_job)
        deadline = time.monotonic() + 5
        while job["job_id"] in workbench.running_jobs and time.monotonic() < deadline:
            self.assertIn(workbench.job(job["job_id"])["status"], {"queued", "running", "completed"})
        self.assertEqual(workbench.job(job["job_id"])["status"], "completed")


if __name__ == "__main__":
    unittest.main()
