import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
import torch

from app.api.diagnostico import crear_router_diagnostico
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.network_segments import SegmentGATLSTM
from app.modulos.clasificador.segments import FEATURE_NAMES, SCALES
from app.modulos.clasificador.training import load_candidate, save_candidate
from app.modulos.clasificador.transfer import import_model_zip
from app.services.diagnostico import DiagnosticWorkbench


class ModelImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.target = self.root / "source", self.root / "target"
        config = Config(max_epochs=1, patience=1, device="cuda:0")
        prep = {"schema_version": config.schema_version, "architecture_version": config.architecture_version,
                "feature_names": list(FEATURE_NAMES), "entity_ids": ["sensor.a"],
                "entities": {"sensor.a": {"kind": "continuous"}}}
        graph = {"method": "within_entity_segment_dtw", "entity_ids": ["sensor.a"],
                 "schema_version": config.schema_version, "segment_steps": [10, 8, 12],
                 "scales": [list(scale[:2]) for scale in SCALES], "adjacency": [[True]],
                 "local_timezone": config.local_timezone, "threshold": config.segment_threshold}
        self.model = SegmentGATLSTM(len(FEATURE_NAMES), ["continuous"], config)
        self.manifest = save_candidate(self.model, prep, graph, config, self.source, {})
        self.identifier = self.manifest["model_id"]
        self.manifest.update(training_count=10, training_contexts=[{"context_end": "2026-01-01"}],
                             training_sample_ids=["sample"], training_groups=["episode"])
        self.model_path = self.source / "models" / f"{self.identifier}.json"
        self.model_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.files = {f"{folder}/{self.identifier}{suffix}": (self.source / folder / f"{self.identifier}{suffix}").read_bytes()
                      for folder, suffix in (("models", ".json"), ("weights", ".pt"),
                                             ("graphs", ".json"), ("preprocessors", ".json"))}
        self.archive = self.root / "model.zip"

    def write_zip(self, files=None, extra=None):
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("models", "weights", "graphs", "preprocessors"):
                archive.writestr(f"{folder}/", b"")
            for name, data in (self.files if files is None else files).items():
                archive.writestr(name, data)
            if extra:
                archive.writestr(*extra)

    def client(self):
        workbench = DiagnosticWorkbench(self.target)
        app = FastAPI()
        app.include_router(crear_router_diagnostico(workbench))
        return TestClient(app), workbench

    def test_complete_zip_roundtrip_preserves_every_file_and_loads_gpu_training_on_cpu(self):
        self.write_zip()
        result = import_model_zip(self.archive, self.target)
        self.assertEqual(result, {"model_id": self.identifier, "imported_files": 4,
                                  "existing_files": 0, "activated": False})
        for name, data in self.files.items():
            self.assertEqual((self.target / name).read_bytes(), data)
        loaded, _, _, manifest = load_candidate(self.identifier, self.target)
        self.assertEqual(str(next(loaded.parameters()).device), "cpu")
        self.assertEqual(manifest["config"]["device"], "cuda:0")
        self.assertFalse((self.target / "active.json").exists())
        for key, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, loaded.state_dict()[key]), key)

    def test_api_import_selectable_and_activation_preserves_training_metadata(self):
        self.write_zip()
        client, workbench = self.client()
        response = client.post("/diagnostics/models/import", content=self.archive.read_bytes(),
                               headers={"Content-Type": "application/zip"})
        self.assertEqual(response.status_code, 200, response.text)
        model = client.get("/diagnostics/status").json()["models"][0]
        self.assertEqual(model["model_id"], self.identifier)
        self.assertEqual(model["training_count"], 10)
        self.assertNotIn("training_contexts", model)
        self.assertNotIn("training_sample_ids", model)
        self.assertNotIn("training_groups", model)
        self.assertEqual(workbench._read("models", self.identifier)["training_contexts"],
                         self.manifest["training_contexts"])
        self.assertEqual(client.get("/diagnostics/classifier-models").json()["models"][0]["model_id"], self.identifier)
        self.assertIsNone(workbench._active())
        activated = client.post(f"/diagnostics/models/{self.identifier}/activate", json={"allow_exploratory": True})
        self.assertEqual(activated.status_code, 200, activated.text)
        self.assertEqual(workbench._active(), self.identifier)
        self.assertEqual(list(workbench.review.temporary.glob("model-upload-*")), [])

    def test_reimport_is_idempotent_and_preserves_active_version(self):
        self.write_zip()
        import_model_zip(self.archive, self.target)
        active = self.target / "active.json"
        active.write_text('{"model_id":"existing"}', encoding="utf-8")
        result = import_model_zip(self.archive, self.target)
        self.assertEqual(result["imported_files"], 0)
        self.assertEqual(result["existing_files"], 4)
        self.assertEqual(active.read_text(), '{"model_id":"existing"}')

    def test_conflict_returns_409_without_publishing_other_files(self):
        self.write_zip()
        client, workbench = self.client()
        conflict = workbench.directory / "models" / f"{self.identifier}.json"
        conflict.write_bytes(b"existing version")
        response = client.post("/diagnostics/models/import", content=self.archive.read_bytes())
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(conflict.read_bytes(), b"existing version")
        self.assertEqual(list((workbench.directory / "weights").iterdir()), [])

    def test_incomplete_invalid_or_unexpected_zip_is_rejected_and_upload_cleaned(self):
        client, workbench = self.client()
        missing = {name: data for name, data in self.files.items() if not name.startswith("graphs/")}
        self.write_zip(missing)
        for content in (self.archive.read_bytes(), b"not a ZIP", b""):
            response = client.post("/diagnostics/models/import", content=content)
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(workbench._list("models"), [])
        self.assertEqual(list(workbench.review.temporary.glob("model-upload-*")), [])
        self.write_zip(extra=("active.json", b"{}"))
        with self.assertRaises(ValueError):
            import_model_zip(self.archive, self.root / "unexpected")
        self.assertFalse((self.root / "unexpected").exists())

    def test_traversal_duplicate_and_symlink_are_rejected_before_publication(self):
        for name in ("../outside.json", "/absolute.json", "weights\\bad.pt", "weights/C:bad.pt"):
            with self.subTest(name=name):
                # No directory entries, so each attack reaches path validation.
                with zipfile.ZipFile(self.archive, "w") as archive:
                    for path, data in self.files.items():
                        archive.writestr(path, data)
                    archive.writestr(name, b"bad")
                with self.assertRaises(ValueError):
                    import_model_zip(self.archive, self.target)
        duplicate = f"weights/{self.identifier}.pt"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(duplicate, b"first")
            with self.assertWarns(UserWarning):
                archive.writestr(duplicate, b"second")
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            import_model_zip(self.archive, self.target)
        link = zipfile.ZipInfo(duplicate)
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(link, b"../../outside")
        with self.assertRaisesRegex(ValueError, "enlaces"):
            import_model_zip(self.archive, self.target)
        self.assertFalse(self.target.exists())

    def test_corrupt_hash_incompatible_weights_and_retired_architecture_are_rejected(self):
        weight_name = f"weights/{self.identifier}.pt"
        files = {**self.files, weight_name: b"corrupt"}
        self.write_zip(files)
        with self.assertRaisesRegex(ValueError, "Hash"):
            import_model_zip(self.archive, self.target)
        buffer = io.BytesIO()
        torch.save({"wrong_shape": torch.zeros(2)}, buffer)
        files[weight_name] = buffer.getvalue()
        manifest = {**self.manifest, "weights_sha256": hashlib.sha256(files[weight_name]).hexdigest()}
        files[f"models/{self.identifier}.json"] = json.dumps(manifest).encode()
        self.write_zip(files)
        with self.assertRaisesRegex(ValueError, "incompatibles"):
            import_model_zip(self.archive, self.target)
        manifest = {**self.manifest, "config": {**self.manifest["config"], "architecture_version": "retired"}}
        self.write_zip({**self.files, f"models/{self.identifier}.json": json.dumps(manifest).encode()})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            import_model_zip(self.archive, self.target)
        self.assertFalse(self.target.exists())

    def test_large_model_metadata_accepted_beyond_transfer_dataset_limit(self):
        manifest = {**self.manifest, "training_metadata": "x" * (17 * 1024 ** 2)}
        self.write_zip({**self.files, f"models/{self.identifier}.json": json.dumps(manifest).encode()})
        self.assertEqual(import_model_zip(self.archive, self.target)["imported_files"], 4)

    def test_publish_failure_rolls_back_new_files_and_keeps_existing(self):
        self.write_zip()
        self.target.mkdir()
        active = self.target / "active.json"
        active.write_bytes(b"existing active version")
        from app.modulos.clasificador import transfer
        publish = transfer._publish
        calls = []
        def fail_second(source, target):
            calls.append(target)
            if len(calls) == 2:
                raise OSError("disk full")
            publish(source, target)
        with patch.object(transfer, "_publish", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "disk full"):
                import_model_zip(self.archive, self.target)
        self.assertEqual([path for path in self.target.rglob("*") if path.is_file()], [active])

    def test_api_size_limit_and_missing_dependencies(self):
        self.write_zip()
        client, workbench = self.client()
        with patch("app.modulos.clasificador.transfer.MAX_MODEL_ZIP_BYTES", 10):
            response = client.post("/diagnostics/models/import", content=self.archive.read_bytes())
        self.assertEqual(response.status_code, 413, response.text)
        with patch("app.services.diagnostico.importlib.util.find_spec", return_value=None):
            response = client.post("/diagnostics/models/import", content=self.archive.read_bytes())
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(workbench._list("models"), [])
        self.assertEqual(list(workbench.review.temporary.glob("model-upload-*")), [])


if __name__ == "__main__":
    unittest.main()
