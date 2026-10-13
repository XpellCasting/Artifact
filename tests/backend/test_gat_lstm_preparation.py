import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from app.modulos.clasificador.config import Config
from app.modulos.clasificador.preparation import prepare_csv_dataset
from app.modulos.clasificador.pipeline import load_partition
from app.modulos.clasificador.pipeline import _load_events, _provider
from app.modulos.clasificador.temporal import build_causal_sample
from app.modulos.clasificador.transfer import export_bundle, import_bundle
from app.modulos.clasificador.training import load_candidate
from app.services.diagnostico import DiagnosticWorkbench


class MultiCsvTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = [self.root / "real.csv", self.root / "synthetic.csv"]
        self.config = Config(use_entropy=False, use_dtw=False, max_epochs=1,
                             patience=1, batch_size=6, dropout=0)
        self.arguments = {"config": self.config, "review_dir": self.root / "review",
            "development_start": "2026-02-01T00:00:00Z", "development_end": "2026-02-21T00:00:00Z"}
        for index, path in enumerate(self.paths):
            self.write_source(path, f"sensor.{index}", bool(index))

    def write_source(self, path, entity, synthetic):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["entity_id", "state", "last_changed", "es_sintetica", "tipo", "tipo_falla"])
            writer.writerow([entity, "10", "2026-02-01T00:00:00Z", str(synthetic).lower(), "normal", "NULL"])
            for day in (5, 20):
                for label, kind in enumerate(("normal", "anomalia", "Falla")):
                    at = datetime(2026, 2, day, 12 + label, tzinfo=timezone.utc)
                    writer.writerow([entity, str(10 + label), at.isoformat(), str(synthetic).lower(), kind,
                                     "Pico" if label == 2 else "NULL"])

    def test_two_sources_prepare_train_and_return_one_cpu_model(self):
        report = prepare_csv_dataset(self.paths, inspect_only=True, **self.arguments)
        self.assertEqual(report["rows"], 14)
        self.assertEqual(report["entities"], 2)
        self.assertFalse((self.root / "review").exists())
        dataset = prepare_csv_dataset(self.paths, **self.arguments)
        self.assertTrue(dataset["ready_for_training"])
        self.assertEqual(len(dataset["source_ids"]), 2)
        self.assertEqual(dataset["partition_counts"]["train_fit"], 6)
        self.assertEqual(dataset["partition_counts"]["dev_tune"], 6)
        self.assertEqual(dataset["evidence_origins"], ["controlled_injection", "historical_review"])
        with load_partition(dataset, "dev_tune") as rows:
            for row in rows:
                meta = row["metadata"]
                self.assertEqual(meta["input_provenance"], [meta["source_id"]])
                self.assertEqual(meta["context_entities"], [meta["target_entity_id"]])
                self.assertEqual(sum(row["label_mask"]), 1)
                other = 1 - meta["entity_ids"].index(meta["target_entity_id"])
                self.assertFalse(row["node_present"][:, other].any())
        workbench = DiagnosticWorkbench(self.root / "review")
        candidate = workbench.train_gat(dataset["dataset_id"])
        self.assertEqual(len(candidate["source_inputs"]), 2)
        separated = candidate["metrics"]["dev_metrics_by_source"]
        self.assertEqual(set(separated), set(dataset["source_ids"]))
        self.assertEqual([value["support"] for value in separated.values()], [3, 3])
        package = self.root / "candidate.tar.gz"
        export_bundle(dataset["dataset_id"], package, model_id=candidate["model_id"], review_dir=self.root / "review")
        destination = self.root / "destination"
        imported = import_bundle(package, review_dir=destination)
        self.assertFalse(imported["activated"])
        model, _, _, manifest = load_candidate(candidate["model_id"], destination / "diagnostics")
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        self.assertEqual(manifest["metrics"]["dev_metrics_by_source"], separated)

    def test_target_only_preparation_preserves_full_context_target_features(self):
        dataset = prepare_csv_dataset(self.paths, **self.arguments)
        directory = Path(dataset["sequence_dir"])
        preprocessor = json.loads((directory / "preprocessor.json").read_text())
        graph = json.loads((directory / "graph.json").read_text())
        events = {}
        for index, path in enumerate(self.paths):
            source, _ = _load_events(path, source_id=str(index))
            events.update(source)
        with load_partition(dataset, "dev_tune") as rows:
            row = rows[0]
            target = row["metadata"]["target_entity_id"]
            full = build_causal_sample(_provider(events), row["metadata"]["target_time"],
                                      preprocessor, graph, self.config)
            node = preprocessor["entity_ids"].index(target)
            torch.testing.assert_close(torch.tensor(row["X"][:, node]),
                                       torch.tensor(full["X"])[:, node], atol=1e-6, rtol=1e-6)

    def test_duplicate_sources_shared_entities_and_held_out_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismo CSV"):
            prepare_csv_dataset([self.paths[0], self.paths[0]], inspect_only=True, **self.arguments)
        self.write_source(self.paths[1], "sensor.0", True)
        with self.assertRaisesRegex(ValueError, "aparece en dos CSV"):
            prepare_csv_dataset(self.paths, inspect_only=True, **self.arguments)
        self.write_source(self.paths[1], "sensor.1", True)
        with self.paths[1].open("a", encoding="utf-8") as stream:
            stream.write("sensor.1,1,2026-02-21T00:00:00Z,true,normal,NULL\n")
        with self.assertRaisesRegex(ValueError, "fuera de Desarrollo"):
            prepare_csv_dataset(self.paths, inspect_only=True, **self.arguments)
        self.assertFalse((self.root / "review").exists())

    def test_cli_inspection_returns_separate_sources(self):
        spec = importlib.util.spec_from_file_location("prepare_cli",
            Path(__file__).resolve().parents[2] / "scripts" / "gat_lstm_prepare.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        output = self.root / "inspect.json"
        arguments = ["--csv", str(self.paths[0]), "--csv", str(self.paths[1]),
            "--development-start", self.arguments["development_start"],
            "--development-end", self.arguments["development_end"],
            "--inspect-only", "--result", str(output)]
        self.assertEqual(cli.main(arguments), 0)
        self.assertEqual(len(json.loads(output.read_text())["sources"]), 2)
