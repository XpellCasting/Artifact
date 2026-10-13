"""Flujo independiente: transferir datos, entrenar/reanudar y devolver un modelo."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from app.modulos.clasificador import LABELS
from app.modulos.clasificador.storage import PartitionWriter
from app.modulos.clasificador.training import load_candidate
from app.modulos.clasificador.transfer import export_bundle, import_bundle
import test_gat_lstm_transfer as transfer_fixture
from test_gat_lstm_training import sample


spec = importlib.util.spec_from_file_location("gat_training_cli",
    Path(__file__).resolve().parents[2] / "scripts" / "gat_lstm_train.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)


class StandaloneWorkflowTests(unittest.TestCase):
    def test_transfer_train_resume_and_return_without_server_or_activation(self):
        fixture = transfer_fixture.TransferTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for partition in ("train_fit", "dev_tune"):
            with PartitionWriter(fixture.sequence, partition) as writer:
                for label in range(3):
                    row = sample(label)
                    row["metadata"].update({"entity_ids": ["sensor.a"],
                        "context_start": "2026-01-01T00:00:00Z",
                        "context_end": "2026-01-04T00:00:00Z"})
                    writer.append(row)
                writer.finish()
        fixture.dataset.update({
            "partition_counts": {"train_fit": 3, "dev_tune": 3},
            "label_counts": {part: dict.fromkeys(LABELS, 1) for part in ("train_fit", "dev_tune")},
            "episode_counts": {part: dict.fromkeys(LABELS, 1) for part in ("train_fit", "dev_tune")}})
        fixture.write_json(fixture.dataset_path, fixture.dataset)
        export_bundle(fixture.dataset_id, fixture.archive, review_dir=fixture.source)
        import_bundle(fixture.archive, review_dir=fixture.target)
        overrides = fixture.root / "config.json"
        overrides.write_text(json.dumps({"batch_size": 3, "dropout": 0}), encoding="utf-8")
        checkpoints, result = fixture.root / "checkpoints", fixture.root / "result.json"
        arguments = [fixture.dataset_id, "--review-dir", str(fixture.target),
            "--device", "cpu", "--config", str(overrides),
            "--checkpoint-dir", str(checkpoints), "--result", str(result)]
        with patch("builtins.print"):
            self.assertEqual(CLI.main(arguments), 0)
        candidate = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(candidate["config"]["batch_size"], 3)
        self.assertEqual(candidate["config"]["grid_seconds"], fixture.config.grid_seconds)
        self.assertFalse(candidate["active"])
        self.assertTrue(candidate["metrics"]["provenance"]["code_sha256"])
        self.assertEqual(len(list(checkpoints.glob("trial-*.pt"))), 3)
        with patch("builtins.print"), patch.object(torch.optim.Adam, "step",
                side_effect=AssertionError("Se repitió un entrenamiento ya completo")):
            self.assertEqual(CLI.main([*arguments, "--resume"]), 0)
        candidate = json.loads(result.read_text(encoding="utf-8"))
        package = fixture.root / "candidate.tar.gz"
        export_bundle(fixture.dataset_id, package, model_id=candidate["model_id"],
                      review_dir=fixture.target)
        active = fixture.directory / "active.json"
        active.write_text('{"model_id":"previous"}', encoding="utf-8")
        imported = import_bundle(package, review_dir=fixture.source)
        self.assertFalse(imported["activated"])
        self.assertEqual(json.loads(active.read_text())["model_id"], "previous")
        loaded, _, _, manifest = load_candidate(candidate["model_id"], fixture.directory)
        self.assertEqual(next(loaded.parameters()).device.type, "cpu")
        self.assertEqual(manifest["metrics"]["rejection_policy"], candidate["metrics"]["rejection_policy"])
