import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from app.services.diagnostico import DiagnosticWorkbench
from app.services.diagnostico import _write
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.pipeline import build_sequence_dataset


class ReviewFixture:
    def __init__(self, annotations): self.annotations = {row["annotation_id"]: row for row in annotations}
    def annotation(self, identifier): return self.annotations[identifier]
    def list_annotations(self): return list(self.annotations.values())
    def list_datasets(self, partition=None): return []


class IntegrationTests(unittest.TestCase):
    def test_prepare_train_and_explicitly_activate_development_only_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bounds = [datetime(2026, 1, day, tzinfo=timezone.utc) for day in (1, 21, 25, 28)]
            development, validation = root / "development.csv", root / "validation.csv"
            parts = {}
            for name, path, lower, upper in (
                ("desarrollo", development, bounds[0], bounds[1]),
                ("validacion", validation, bounds[1], bounds[2]),
                ("prueba", root / "test.csv", bounds[2], bounds[3]),
                ("reserva_agosto", root / "reserve.csv", bounds[3], bounds[3] + timedelta(days=1)),
            ):
                rows = []
                cursor = lower
                while cursor < upper:
                    rows.append({"entity_id": "sensor.a", "state": str(20 + cursor.minute / 10),
                                 "last_changed": cursor.isoformat().replace("+00:00", "Z")})
                    rows.append({"entity_id": "binary.b", "state": "on" if cursor.minute % 2 else "off",
                                 "last_changed": cursor.isoformat().replace("+00:00", "Z")})
                    cursor += timedelta(minutes=1)
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["entity_id", "state", "last_changed"])
                    writer.writeheader(); writer.writerows(rows)
                parts[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "rows": len(rows), "entities": {"sensor.a": "continuous", "binary.b": "binary"}}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"boundaries": [value.isoformat() for value in bounds],
                                            "partitions": parts}), encoding="utf-8")

            annotations = []
            labels = ("normal", "environmental_anomaly", "sensor_fault")
            for partition, path, first_day in (("train", development, 5),
                                                ("dev", development, 20)):
                with path.open(encoding="utf-8", newline="") as stream:
                    source_rows = list(csv.DictReader(stream))
                offset = bounds[0] if path == development else bounds[1]
                for index, label in enumerate(labels):
                    target_time = datetime(2026, 1, first_day, 12 + index, tzinfo=timezone.utc)
                    row_number = next(i + 2 for i, row in enumerate(source_rows)
                                      if row["entity_id"] == "sensor.a"
                                      and row["last_changed"] == target_time.isoformat().replace("+00:00", "Z"))
                    annotations.append({"annotation_id": f"{partition}-{label}", "dataset_id": "fixture",
                        "source_sha256": parts["desarrollo"]["sha256"],
                        "entity_id": "sensor.a", "source_row": row_number,
                        "start": target_time.isoformat(), "end": target_time.isoformat(),
                        "label": label, "status": "reviewed", "fault_type": "spike" if label == "sensor_fault" else None,
                        "reference_origin": "historical_review", "parent_episode_id": f"{partition}-{label}"})

            workbench = DiagnosticWorkbench(root / "review")
            workbench.review = ReviewFixture(annotations)
            config = Config(sequence_steps=2, feature_window_seconds=60, labelled_stride_seconds=60,
                            gat_heads=2, gat_head_dim=16, batch_size=3, max_epochs=2, patience=2,
                            dropout=0.0, use_entropy=False, use_dtw=False)
            dataset = build_sequence_dataset(workbench.review,
                ["canonical:desarrollo"],
                [item["annotation_id"] for item in annotations], config,
                workbench.directory, manifest)
            _write(workbench._path("datasets", dataset["dataset_id"]), dataset)

            progress = []
            candidate = workbench.train_gat(dataset["dataset_id"], gat_config=config.as_dict(),
                                            progress=lambda value, message, **details: progress.append((value, message, details)))
            self.assertTrue(any("pérdida train" in message for _, message, _ in progress))
            self.assertTrue(any(details.get("phase") == "training"
                                and details.get("max_epochs") == config.max_epochs
                                for _, _, details in progress))
            self.assertFalse(candidate["active"])
            with self.assertRaisesRegex(Exception, "solo con Desarrollo"):
                workbench.evaluate_gat(candidate["model_id"], dataset["dataset_id"], "validation")
            active = workbench.activate(candidate["model_id"], allow_exploratory=True)
            self.assertTrue(active["active"])
            self.assertEqual(workbench._active(), candidate["model_id"])
            session = workbench.create_run_session("validation_replay", "fixture-run")
            replay_time = datetime(2026, 1, 6, 12, 0, tzinfo=timezone.utc)
            replay_events = pd.DataFrame([
                {"entity_id": entity, "timestamp_dt": replay_time - timedelta(minutes=minute),
                 "state": ("on" if minute % 2 else "off") if entity == "binary.b" else str(21 + minute),
                 "_raw_index": minute * 2 + ordinal}
                for minute in range(3) for ordinal, entity in enumerate(("sensor.a", "binary.b"))
            ])
            enriched = workbench.enrich_contextual([], [{"entity_id": "sensor.a",
                "timestamp": replay_time.isoformat(), "routing": {"terminal": False}}],
                replay_events, "validation_replay", "fixture-run", session)
            diagnosis = enriched[0]["diagnosis"]
            self.assertEqual(diagnosis["method"], "gat_lstm")
            self.assertIsNone(diagnosis["compatibility"])
            self.assertEqual(diagnosis["confidence_kind"], "uncalibrated_softmax")
            closed = workbench.enrich_contextual([], [{"entity_id": "sensor.a",
                "timestamp": replay_time.isoformat(), "routing": {"terminal": True}}],
                replay_events, "validation_replay", "fixture-run", session)
            self.assertNotIn("diagnosis", closed[0])
            session.flush()


if __name__ == "__main__":
    unittest.main()
