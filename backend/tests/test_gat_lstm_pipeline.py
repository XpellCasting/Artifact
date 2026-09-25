import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from itertools import chain
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.GAT_LSTM.config import Config, resolve_manifest_source
from app.GAT_LSTM.pipeline import (build_sequence_dataset, canonical_sources, development_sources, load_partition,
                                   load_pretraining_partition)


class ReviewFixture:
    def __init__(self, annotation=None):
        self.row, self.datasets, self.paths = annotation, [], {}
    def annotation(self, identifier):
        if self.row is None: raise KeyError(identifier)
        if identifier != self.row["annotation_id"]: raise KeyError(identifier)
        return self.row
    def list_annotations(self): return [self.row] if self.row else []
    def list_datasets(self, partition=None):
        return [item for item in self.datasets if partition is None or item.get("partition") == partition]
    def source_path(self, dataset_id): return self.paths[dataset_id]


class PipelineTests(unittest.TestCase):
    def test_windows_manifest_path_resolves_in_container_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "historico_desarrollo.csv"
            source.write_text("entity_id,state,last_changed\n", encoding="utf-8")
            manifest = root / "particiones_seleccionadas_resumen.json"
            manifest.write_text(json.dumps({"partitions": {"desarrollo": {
                "path": r"C:\Users\Author\Desktop\Tesis\HistoricoPuro\historico_desarrollo.csv",
                "sha256": "abc", "rows": 0}}}), encoding="utf-8")
            result = canonical_sources(manifest)
            self.assertEqual(result[0]["path"], str(source))
            self.assertEqual(result[0]["filename"], source.name)

    def test_persists_causal_dataset_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boundaries = [datetime(2026, 2, day, tzinfo=timezone.utc) for day in range(1, 5)]
            names = ["desarrollo", "validacion", "prueba", "reserva_agosto"]
            parts = {}
            dev_file = root / "development.csv"
            rows = []
            start = boundaries[0]
            for minute in range(0, 24 * 60, 5):
                at = start + timedelta(minutes=minute)
                rows.append({"entity_id": "sensor.a", "state": str(10 + minute / 100),
                             "last_changed": at.isoformat().replace("+00:00", "Z")})
            with dev_file.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["entity_id", "state", "last_changed"])
                writer.writeheader(); writer.writerows(rows)
            for name, day in zip(names, range(1, 5)):
                path = dev_file if name == "desarrollo" else root / f"{name}.csv"
                if name != "desarrollo": path.write_text("entity_id,state,last_changed\n", encoding="utf-8")
                parts[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "rows": len(rows) if name == "desarrollo" else 0, "entities": {}}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"boundaries": [b.isoformat() for b in boundaries],
                                            "partitions": parts}), encoding="utf-8")
            target_at = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)
            target_row = next(i for i, row in enumerate(rows, 2)
                              if datetime.fromisoformat(row["last_changed"].replace("Z", "+00:00")) == target_at)
            annotation = {"annotation_id": "a1", "dataset_id": "fixture", "source_sha256": parts["desarrollo"]["sha256"],
                          "entity_id": "sensor.a", "source_row": target_row,
                          "start": target_at.isoformat(), "end": target_at.isoformat(),
                          "label": "normal", "status": "reviewed", "fault_type": None,
                          "reference_origin": "historical_review", "parent_episode_id": "ep1"}
            graph = {"entity_ids": ["sensor.a"], "adjacency": [[True]], "edges": [],
                     "spatial_evidence": False, "graph_version": "fixture"}
            with patch("app.GAT_LSTM.pipeline.build_graph", return_value=graph):
                dataset = build_sequence_dataset(ReviewFixture(annotation), ["canonical:desarrollo"], ["a1"],
                    Config(pretrain_enabled=True), root / "diagnostics", manifest)
            train = load_partition(dataset, "train_fit")
            self.assertEqual(len(train), 1)
            self.assertTrue(train[0]["label_mask"][0])
            self.assertEqual(train[0]["metadata"]["source_sha256"], parts["desarrollo"]["sha256"])
            self.assertEqual(load_partition(dataset, "test"), [])
            future_train = load_pretraining_partition(dataset, "train_fit")
            future_dev = load_pretraining_partition(dataset, "dev_tune")
            self.assertTrue(future_train and future_dev)
            self.assertFalse(any(future_train[0]["label_mask"]))
            train.close()
            future_train.close()
            future_dev.close()

    def test_labeled_development_import_trains_directly_and_never_creates_holdout_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bounds = [datetime(2026, 2, day, tzinfo=timezone.utc) for day in range(1, 5)]
            csv_path = root / "development-labeled.csv"
            start = bounds[0]
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["entity_id", "state", "last_changed",
                    "es_sintetica", "tipo", "tipo_falla"])
                writer.writeheader()
                for index in range(24 * 12):
                    at = start + timedelta(minutes=5 * index)
                    label = ("normal", "anomalia", "Falla")[index % 3]
                    writer.writerow({"entity_id": "sensor.a", "state": str(10 + index % 20),
                        "last_changed": at.isoformat().replace("+00:00", "Z"),
                        "es_sintetica": "true" if index % 6 == 0 else "false",
                        "tipo": label, "tipo_falla": "ruido" if label == "Falla" else "NULL"})
            digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"boundaries": [b.isoformat() for b in bounds],
                "partitions": {"desarrollo": {"path": str(csv_path), "sha256": digest, "rows": 288},
                    "validacion": {"path": str(root / "validation.csv"), "sha256": "unused", "rows": 0},
                    "prueba": {"path": str(root / "test.csv"), "sha256": "unused", "rows": 0}}}),
                encoding="utf-8")
            review = ReviewFixture()
            review.paths["labeled-dev"] = csv_path
            review.datasets = [{"dataset_id": "labeled-dev", "filename": csv_path.name,
                "partition": "development", "labelled_import": True, "columns": ["entity_id", "state",
                    "last_changed", "es_sintetica", "tipo", "tipo_falla"], "imported_labels": 288,
                "imported_synthetic": 48, "rows": 288, "sha256": digest,
                "start": start.isoformat(), "end": (start + timedelta(minutes=1435)).isoformat()}]
            sources = development_sources(manifest, review)
            self.assertEqual([item["source_id"] for item in sources], ["canonical:desarrollo", "review:labeled-dev"])
            graph = {"entity_ids": ["sensor.a"], "adjacency": [[True]], "edges": [],
                     "spatial_evidence": False, "graph_version": "fixture"}
            with patch("app.GAT_LSTM.pipeline.build_graph", return_value=graph):
                dataset = build_sequence_dataset(review, ["review:labeled-dev"], [],
                    Config(sequence_steps=2, feature_window_seconds=60, labelled_stride_seconds=300,
                           max_nodes=32, graph_top_k=0, use_entropy=False, use_dtw=False),
                    root / "diagnostics", manifest)
            train = load_partition(dataset, "train_fit")
            tune = load_partition(dataset, "dev_tune")
            self.assertTrue(train and tune)
            self.assertEqual(dataset["training_scope"], "development_only")
            self.assertEqual(dataset["partition_counts"]["validation"], 0)
            self.assertEqual(dataset["partition_counts"]["test"], 0)
            self.assertTrue(any(row["metadata"]["es_sintetica"] for row in chain(train, tune)))
            self.assertTrue(any(row["metadata"]["fault_subtype"] == "noise" for row in chain(train, tune)))
            self.assertFalse((Path(dataset["sequence_dir"]) / "validation_X.npy").exists())
            self.assertFalse((Path(dataset["sequence_dir"]) / "test_X.npy").exists())
            train.close()
            tune.close()

    def test_validation_and_test_sources_are_rejected_before_their_files_are_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bounds = [datetime(2026, 2, day, tzinfo=timezone.utc) for day in range(1, 5)]
            dev = root / "development.csv"
            dev.write_text("entity_id,state,last_changed\nsensor.a,1,2026-02-01T00:00:00Z\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"boundaries": [b.isoformat() for b in bounds], "partitions": {
                "desarrollo": {"path": str(dev), "sha256": hashlib.sha256(dev.read_bytes()).hexdigest(), "rows": 1},
                "validacion": {"path": str(root / "never-open-validation.csv"), "sha256": "unused", "rows": 0},
                "prueba": {"path": str(root / "never-open-test.csv"), "sha256": "unused", "rows": 0}}}), encoding="utf-8")
            def development_only_resolver(manifest_path, source_path):
                if "never-open-" in str(source_path):
                    raise AssertionError("GAT attempted to resolve a held-out file")
                return resolve_manifest_source(manifest_path, source_path)
            with patch("app.GAT_LSTM.pipeline.resolve_manifest_source", side_effect=development_only_resolver):
                with self.assertRaisesRegex(ValueError, "solo admite fuentes de Desarrollo"):
                    build_sequence_dataset(ReviewFixture(), ["canonical:validacion"], [], Config(),
                                           root / "diagnostics", manifest)


if __name__ == "__main__":
    unittest.main()
