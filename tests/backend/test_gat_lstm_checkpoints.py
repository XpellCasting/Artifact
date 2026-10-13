"""Checkpoint interruption/resume and explicit CPU/CUDA training contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from app.modulos.clasificador.checkpoints import TrainingRun
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.segments import FEATURE_NAMES, MAX_SEGMENTS
from app.modulos.clasificador.training import (
    _tensor_batch, load_candidate, resolve_device, save_candidate, train_classifier,
)
from test_gat_lstm_training import sample


class CheckpointTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.config = Config(sequence_steps=4, max_epochs=2, patience=5,
                             learning_rate=.01, dropout=.2, batch_size=2,
                             lstm_hidden=4, device="cpu")
        self.rows = [sample(label) for label in range(3)]
        # Include actual segment edges, and label 2 exercises the fault-type loss.
        for row in self.rows:
            for step in row["X"]:
                step[0][-MAX_SEGMENTS:] = [1.] * MAX_SEGMENTS

    def train(self, config=None, **kwargs):
        return train_classifier(self.rows, self.rows, config or self.config,
                                [3], ["continuous"], **kwargs)

    @staticmethod
    def interrupt(row):
        if row["threshold_trial"] == 1 and row["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    def assert_same_model(self, expected, actual):
        self.assertEqual(expected.threshold, actual.threshold)
        for key, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), actual.state_dict()[key].cpu()), key)
        with torch.no_grad():
            x, states, present, _, _, adjacency = _tensor_batch(self.rows, "cpu")
            expected_output = expected.cpu()(x, states, adjacency, present)
            actual_output = actual.cpu()(x, states, adjacency, present)
        for first, second in zip(expected_output, actual_output):
            self.assertTrue(torch.equal(first, second))

    def test_interrupted_multitrial_resume_matches_uninterrupted_and_skips_completed(self):
        uninterrupted, baseline = self.train()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.train(checkpoint_dir=path, progress=self.interrupt)

            # The epoch is durable before a progress consumer can interrupt it.
            state = torch.load(path / "trial-01.pt", map_location="cpu", weights_only=True)
            self.assertEqual(state["next_epoch"], 1)
            self.assertEqual(len(state["history"]), 1)
            self.assertTrue(state["optimizer"]["state"])
            self.assertTrue(state["best_dev_records"])
            self.assertTrue(state["best_type_records"])
            self.assertEqual(set(state["model"]), set(state["best_model"]))
            self.assertEqual(set(state["random_state"]), {"torch", "python", "numpy", "cuda"})
            self.assertTrue(all(value.device.type == "cpu" for value in state["model"].values()))
            self.assertFalse(list(path.glob("*.tmp")))
            manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["fingerprint"], state["fingerprint"])
            self.assertIn("samples_sha256", manifest["dataset"])
            self.assertEqual(manifest["dataset"]["entity_kinds"], ["continuous"])

            progress = []
            resumed, scores = self.train(checkpoint_dir=path, resume=True, progress=progress.append)
            self.assert_same_model(uninterrupted, resumed)
            self.assertEqual(scores["segment_threshold_search"], baseline["segment_threshold_search"])
            self.assertEqual(scores["rejection_policy"], baseline["rejection_policy"])
            self.assertEqual([(row["threshold_trial"], row["epoch"]) for row in progress],
                             [(1, 2), (2, 1), (2, 2), (3, 1), (3, 2)])
            self.assertEqual(len(list(path.glob("trial-*.pt"))), 3)

            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("completed trial was trained again")):
                completed, _ = self.train(checkpoint_dir=path, resume=True)
            self.assert_same_model(resumed, completed)
            with self.assertRaisesRegex(ValueError, "ya contiene"):
                self.train(checkpoint_dir=path)

    def test_resume_rejects_changed_samples_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.train(checkpoint_dir=path, progress=self.interrupt)
            changed = deepcopy(self.rows)
            changed[0]["X"][0][0][0] += .5
            with self.assertRaisesRegex(ValueError, "incompatible"):
                train_classifier(changed, self.rows, self.config, [3], ["continuous"],
                                 checkpoint_dir=path, resume=True)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                self.train(replace(self.config, learning_rate=.005), checkpoint_dir=path, resume=True)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                self.train(checkpoint_dir=path, resume=True,
                           checkpoint_identity={"dataset_id": "another-dataset"})
            with self.assertRaisesRegex(ValueError, "incompatible"):
                train_classifier(self.rows, self.rows, self.config, [3], ["multistate"],
                                 checkpoint_dir=path, resume=True)

    def test_resume_after_interruption_before_first_epoch_restarts_matching_run(self):
        baseline, expected = self.train()

        def interrupt_first_batch(row):
            self.assertEqual((row["threshold_trial"], row["epoch"], row["stage"], row["batch"]),
                             (1, 1, "train", 1))
            raise RuntimeError("interrupted before first checkpoint")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "before first checkpoint"):
                self.train(checkpoint_dir=path, batch_progress=interrupt_first_batch)
            self.assertTrue((path / "run.json").is_file())
            self.assertFalse(list(path.glob("trial-*.pt")))
            progress = []
            resumed, actual = self.train(checkpoint_dir=path, resume=True, progress=progress.append)
            self.assert_same_model(baseline, resumed)
            self.assertEqual(actual["segment_threshold_search"], expected["segment_threshold_search"])
            self.assertEqual(actual["rejection_policy"], expected["rejection_policy"])
            self.assertEqual([(row["threshold_trial"], row["epoch"]) for row in progress],
                             [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)])

    def test_can_extend_epochs_and_change_device_selection_but_cannot_reduce_completed_epochs(self):
        expected, _ = self.train()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.train(replace(self.config, max_epochs=1), checkpoint_dir=path)
            progress = []
            with patch("torch.cuda.is_available", return_value=False):
                resumed, _ = self.train(replace(self.config, device="auto"), checkpoint_dir=path,
                                        resume=True, progress=progress.append)
            self.assert_same_model(expected, resumed)
            self.assertEqual([(row["threshold_trial"], row["epoch"]) for row in progress],
                             [(1, 2), (2, 2), (3, 2)])
            with self.assertRaisesRegex(ValueError, "max_epochs"):
                self.train(replace(self.config, max_epochs=1), checkpoint_dir=path, resume=True)

    def test_resume_requires_saved_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "directorio"):
            self.train(resume=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No hay checkpoints"):
                self.train(checkpoint_dir=Path(directory), resume=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA no disponible en este equipo")
    def test_cuda_training_exports_candidate_and_resumes_on_cpu(self):
        config = replace(self.config, max_epochs=1, device="cuda")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model, metrics = self.train(config, checkpoint_dir=path / "checkpoints")
            self.assertEqual(next(model.parameters()).device.type, "cuda")
            self.assertTrue(metrics["runtime_parameters"]["device"].startswith("cuda"))
            self.assertIsNotNone(metrics["history"][0]["train_type_loss"])
            prep = {"entity_ids": ["a"], "feature_names": list(FEATURE_NAMES),
                    "entities": {"a": {"kind": "continuous", "normal_references": []}}}
            manifest = save_candidate(model, prep, {"adjacency": [[True]]}, config,
                                      path / "candidate", metrics)
            loaded, _, _, _ = load_candidate(manifest["model_id"], path / "candidate")
            self.assertEqual(next(loaded.parameters()).device.type, "cpu")
            with torch.no_grad():
                x, states, present, _, _, adjacency = _tensor_batch(self.rows, "cuda")
                gpu_outputs = model(x, states, adjacency, present)
                cpu_outputs = loaded(x.cpu(), states.cpu(), adjacency.cpu(), present.cpu())
            for gpu, cpu in zip(gpu_outputs, cpu_outputs):
                self.assertTrue(torch.allclose(gpu.cpu(), cpu, atol=1e-5, rtol=1e-4))
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("completed GPU trial was trained again")):
                resumed, _ = self.train(replace(config, device="cpu"),
                                        checkpoint_dir=path / "checkpoints", resume=True)
            self.assert_same_model(loaded, resumed)


class DeviceResolutionTests(unittest.TestCase):
    def test_cpu_and_auto_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(str(resolve_device("cpu")), "cpu")
            self.assertEqual(str(resolve_device("auto")), "cpu")
            for requested in ("cuda", "cuda:0"):
                with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "CUDA no está disponible"):
                    resolve_device(requested)

    def test_cuda_selects_requested_device_and_rejects_invalid_indices(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=2), \
                patch("torch.cuda.current_device", return_value=1):
            self.assertEqual(str(resolve_device("auto")), "cuda:1")
            self.assertEqual(str(resolve_device("cuda")), "cuda:1")
            self.assertEqual(str(resolve_device("cuda:0")), "cuda:0")
            with self.assertRaisesRegex(ValueError, "inexistente"):
                resolve_device("cuda:2")
        for requested in ("cuda:-1", "cuda:abc", "cuda:0:1", "mps", "", None):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "Dispositivo inválido"):
                resolve_device(requested)


class CheckpointStorageTests(unittest.TestCase):
    def test_failed_save_preserves_previous_epoch_and_releases_directory_lock(self):
        environment = {"library_versions": {"torch": "2.6.0"}, "code_sha256": {"training.py": "example"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            run = TrainingRun(path, Config(), {"dataset_id": "example"}, False, environment)
            try:
                run.save(1, {"next_epoch": 1, "model": {"weight": torch.tensor([1.])}})
                original = (path / "trial-01.pt").read_bytes()

                def interrupted_save(state, temporary):
                    Path(temporary).write_bytes(b"partial checkpoint")
                    raise OSError("simulated full disk")

                with patch("app.modulos.clasificador.checkpoints.torch.save", side_effect=interrupted_save):
                    with self.assertRaisesRegex(OSError, "simulated full disk"):
                        run.save(1, {"next_epoch": 2})
                self.assertEqual((path / "trial-01.pt").read_bytes(), original)
                self.assertFalse(list(path.glob("*.tmp")))
                with self.assertRaisesRegex(ValueError, "Otro proceso"):
                    TrainingRun(path, Config(), {"dataset_id": "example"}, True, environment)
            finally:
                run.close()
            # A new process/run can acquire the released lock and read epoch 1.
            resumed = TrainingRun(path, Config(), {"dataset_id": "example"}, True, environment)
            try:
                self.assertEqual(resumed.load(1)["next_epoch"], 1)
            finally:
                resumed.close()


if __name__ == "__main__":
    unittest.main()
