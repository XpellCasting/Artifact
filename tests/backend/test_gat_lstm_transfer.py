import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.modulos.clasificador.config import Config
from app.modulos.clasificador.segments import FEATURE_NAMES, MAX_SEGMENTS, SCALES
from app.modulos.clasificador.storage import FIELDS, PartitionWriter
from app.modulos.clasificador.transfer import MANIFEST, export_bundle, import_bundle


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = self.root / "source", self.root / "target"
        self.dataset_id = "a" * 32
        self.directory = self.source / "diagnostics"
        self.sequence = self.directory / "sequences" / self.dataset_id
        self.sequence.mkdir(parents=True)
        self.config = Config(max_epochs=1, patience=1)
        self.prep = {"schema_version": self.config.schema_version,
                     "architecture_version": self.config.architecture_version,
                     "feature_names": list(FEATURE_NAMES), "entity_ids": ["sensor.a"],
                     "entities": {"sensor.a": {"kind": "continuous"}}}
        self.graph = {"method": "within_entity_segment_dtw", "entity_ids": ["sensor.a"],
                      "schema_version": self.config.schema_version, "segment_steps": [10, 8, 12],
                      "scales": [list(scale[:2]) for scale in SCALES], "adjacency": [[True]],
                      "local_timezone": self.config.local_timezone, "threshold": self.config.segment_threshold}
        self.graph["graph_version"] = hashlib.sha256(json.dumps(self.graph, sort_keys=True).encode()).hexdigest()
        self.write_json(self.sequence / "config.json", self.config.as_dict())
        self.write_json(self.sequence / "preprocessor.json", self.prep)
        self.write_json(self.sequence / "graph.json", self.graph)
        for partition in ("train_fit", "dev_tune"):
            with PartitionWriter(self.sequence, partition) as writer:
                writer.append({"X": np.zeros((3 * MAX_SEGMENTS, 1, len(FEATURE_NAMES))),
                               "state_ids": np.zeros((3 * MAX_SEGMENTS, 1)),
                               "node_present": np.ones((3 * MAX_SEGMENTS, 1)),
                               "y": [0], "label_mask": [True],
                               "metadata": {"entity_ids": ["sensor.a"], "sample_id": partition}})
                writer.finish()
        self.dataset = {"dataset_id": self.dataset_id, "method": "gat_lstm", "config": self.config.as_dict(),
                        "schema_version": self.config.schema_version, "sequence_dir": str(self.sequence),
                        "entity_ids": ["sensor.a"], "graph_version": self.graph["graph_version"],
                        "partition_counts": {"train_fit": 1, "dev_tune": 1, "test": 0},
                        "source_hashes": {"development": "b" * 64}}
        self.dataset_path = self.directory / "datasets" / f"{self.dataset_id}.json"
        self.write_json(self.dataset_path, self.dataset)
        self.archive = self.root / "training.tar.gz"

    @staticmethod
    def write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def export(self, **kwargs):
        return export_bundle(self.dataset_id, self.archive, review_dir=self.source, **kwargs)

    def rewrite_archive(self, *, mutate_manifest=None, mutate_entry=None, append=None):
        with tarfile.open(self.archive) as source:
            rows = [(member, source.extractfile(member).read()) for member in source]
        manifest = json.loads(rows[0][1])
        if mutate_manifest:
            mutate_manifest(manifest)
        rows[0] = (rows[0][0], json.dumps(manifest).encode())
        temporary = self.root / "modified.tar.gz"
        with tarfile.open(temporary, "w:gz") as target:
            for member, data in rows:
                if mutate_entry:
                    member, data = mutate_entry(member, data)
                member.size = len(data)
                target.addfile(member, io.BytesIO(data))
            if append:
                member, data = append
                member.size = len(data)
                target.addfile(member, io.BytesIO(data))
        temporary.replace(self.archive)

    def test_roundtrip_relocates_and_preserves_activation_then_idempotent(self):
        self.export()
        with tarfile.open(self.archive) as archive:
            portable = json.load(archive.extractfile(f"datasets/{self.dataset_id}.json"))
        self.assertEqual(portable["sequence_dir"], f"sequences/{self.dataset_id}")
        active = self.target / "diagnostics" / "active.json"
        self.write_json(active, {"model_id": "local-active"})
        result = import_bundle(self.archive, review_dir=self.target)
        self.assertFalse(result["activated"])
        self.assertGreater(result["imported_files"], 0)
        imported = json.loads((active.parent / "datasets" / f"{self.dataset_id}.json").read_text())
        self.assertEqual(Path(imported["sequence_dir"]), (active.parent / "sequences" / self.dataset_id).resolve())
        self.assertEqual(json.loads(active.read_text()), {"model_id": "local-active"})
        for field in FIELDS:
            np.testing.assert_array_equal(np.load(self.sequence / f"train_fit_{field}.npy"),
                np.load(Path(imported["sequence_dir"]) / f"train_fit_{field}.npy"))
        repeat = import_bundle(self.archive, review_dir=self.target)
        self.assertEqual(repeat["imported_files"], 0)
        self.assertEqual(repeat["existing_files"], result["imported_files"])

    def test_default_layout_uses_redirected_artifact_paths(self):
        default_review, default_models = self.root / "review", self.root / "models"
        default_data = self.root / "data"
        with patch("app.modulos.clasificador.transfer.REVIEW_DATA_DIR", default_review), \
             patch("app.modulos.clasificador.transfer.MODELS_DIR", default_models), \
             patch("app.config.MODELS_DIR", default_models), patch("app.config.DATA_DIR", default_data):
            self.export()
            import_bundle(self.archive)
            self.assertTrue((default_data / "gat_lstm" / "datasets" / f"{self.dataset_id}.json").is_file())
            self.assertTrue((default_data / "gat_lstm" / "sequences" / self.dataset_id / "config.json").is_file())
            self.assertFalse((default_models / "active.json").exists())

    def test_conflict_rejected_before_any_new_file_is_published(self):
        self.export()
        conflict = self.target / "diagnostics" / "sequences" / self.dataset_id / "graph.json"
        conflict.parent.mkdir(parents=True)
        conflict.write_text("existing different graph")
        with self.assertRaisesRegex(FileExistsError, "conflicto"):
            import_bundle(self.archive, review_dir=self.target)
        self.assertEqual(list(conflict.parent.iterdir()), [conflict])
        self.assertFalse((self.target / "diagnostics" / "datasets").exists())

    def test_corruption_rejected_without_destination_writes(self):
        self.export()
        self.rewrite_archive(mutate_manifest=lambda manifest: manifest["files"][0].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "corrupto"):
            import_bundle(self.archive, review_dir=self.target)
        self.assertFalse(self.target.exists())

    def test_traversal_absolute_and_windows_names_rejected(self):
        self.export()
        original = self.archive.read_bytes()
        for bad_name in ("../outside.txt", "/outside.txt", "C:/outside.txt", "sequences/" + self.dataset_id + "/../bad", "sequences\\bad", f"sequences/{self.dataset_id}/CON"):
            with self.subTest(path=bad_name):
                self.archive.write_bytes(original)
                self.rewrite_archive(mutate_manifest=lambda manifest: manifest["files"][0].update(path=bad_name))
                with self.assertRaisesRegex(ValueError, "Ruta inválida"):
                    import_bundle(self.archive, review_dir=self.target)
                self.assertFalse(self.target.exists())

    def test_symlink_tar_member_and_unlisted_active_file_rejected(self):
        self.export()
        original = self.archive.read_bytes()
        def symlink(member, data):
            if member.name != MANIFEST:
                member.type, member.linkname = tarfile.SYMTYPE, "../../outside"
            return member, data
        self.rewrite_archive(mutate_entry=symlink)
        with self.assertRaisesRegex(ValueError, "Entrada TAR inválida"):
            import_bundle(self.archive, review_dir=self.target)
        self.assertFalse(self.target.exists())
        self.archive.write_bytes(original)
        self.rewrite_archive(append=(tarfile.TarInfo("active.json"), b"{}"))
        with self.assertRaisesRegex(ValueError, "no permitido"):
            import_bundle(self.archive, review_dir=self.target)
        self.assertFalse(self.target.exists())

    def test_size_and_missing_inventory_rejected(self):
        self.export()
        original = self.archive.read_bytes()
        self.rewrite_archive(mutate_manifest=lambda manifest: manifest["files"][0].update(size=1))
        with self.assertRaisesRegex(ValueError, "Entrada TAR inválida"):
            import_bundle(self.archive, review_dir=self.target)
        self.archive.write_bytes(original)
        self.rewrite_archive(mutate_manifest=lambda manifest: manifest["files"].append({
            "path": f"sequences/{self.dataset_id}/missing.bin", "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest()}))
        with self.assertRaisesRegex(ValueError, "Faltan archivos"):
            import_bundle(self.archive, review_dir=self.target)
        self.assertFalse(self.target.exists())

    def test_oversized_inventory_rejected_before_extraction(self):
        self.export()
        self.rewrite_archive(mutate_manifest=lambda manifest: manifest["files"][0].update(size=129 * 1024 ** 3))
        with self.assertRaisesRegex(ValueError, "Inventario duplicado o inválido"):
            import_bundle(self.archive, review_dir=self.target)
        self.assertFalse(self.target.exists())

    def test_valid_checksum_cannot_hide_invalid_graph_signature(self):
        self.export()
        graph_name = f"sequences/{self.dataset_id}/graph.json"
        changed = {**self.graph, "graph_version": "0" * 64}
        encoded = json.dumps(changed).encode()
        def mutate_manifest(manifest):
            next(item for item in manifest["files"] if item["path"] == graph_name).update(
                size=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())
        def mutate_entry(member, data):
            return member, encoded if member.name == graph_name else data
        self.rewrite_archive(mutate_manifest=mutate_manifest, mutate_entry=mutate_entry)
        with self.assertRaisesRegex(ValueError, "Firma graph_version"):
            import_bundle(self.archive, review_dir=self.target)
        self.assertFalse(self.target.exists())

    def test_failed_publication_removes_new_artifacts_and_preserves_existing(self):
        self.export()
        active = self.target / "diagnostics" / "active.json"
        self.write_json(active, {"model_id": "local-active"})
        with patch("app.modulos.clasificador.transfer.shutil.copyfileobj", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                import_bundle(self.archive, review_dir=self.target)
        self.assertEqual([path for path in self.target.rglob("*") if path.is_file()], [active])

    def test_invalid_graph_or_npy_contract_and_retired_config_rejected(self):
        self.graph["threshold"] = .7
        self.write_json(self.sequence / "graph.json", self.graph)
        with self.assertRaisesRegex(ValueError, "Umbral"):
            self.export()
        self.graph["threshold"] = .5
        self.write_json(self.sequence / "graph.json", self.graph)
        np.save(self.sequence / "train_fit_X.npy", np.zeros((1, 2, 1, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Forma o dtype"):
            self.export()
        self.dataset["config"]["architecture_version"] = "gat_lstm_v2_shared_type"
        self.write_json(self.dataset_path, self.dataset)
        with self.assertRaisesRegex(ValueError, "prepara nuevamente el dataset"):
            self.export()
        self.assertFalse(self.archive.exists())

    def test_output_existing_file_is_preserved(self):
        self.archive.write_bytes(b"existing package")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.archive.read_bytes(), b"existing package")

    def test_optional_candidate_roundtrip_and_cpu_load(self):
        import torch
        from app.modulos.clasificador.network_segments import SegmentGATLSTM
        from app.modulos.clasificador.training import load_candidate, save_candidate
        model = SegmentGATLSTM(len(FEATURE_NAMES), ["continuous"], self.config)
        candidate = save_candidate(model, self.prep, self.graph, self.config, self.directory,
                                   {"rejection_policy": {"threshold": .6, "margin": .1}})
        candidate.update(dataset_id=self.dataset_id, source_hashes=self.dataset["source_hashes"])
        self.write_json(self.directory / "models" / f"{candidate['model_id']}.json", candidate)
        self.write_json(self.directory / "active.json", {"model_id": candidate["model_id"]})
        self.export(model_id=candidate["model_id"])
        self.assertEqual(import_bundle(self.archive, review_dir=self.source)["imported_files"], 0)
        result = import_bundle(self.archive, review_dir=self.target)
        loaded, prep, _, metadata = load_candidate(candidate["model_id"], self.target / "diagnostics")
        self.assertEqual(result["model_id"], candidate["model_id"])
        self.assertEqual(prep, self.prep)
        self.assertEqual(metadata["rejection_policy"], candidate["rejection_policy"])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        self.assertFalse((self.target / "diagnostics" / "active.json").exists())


if __name__ == "__main__":
    unittest.main()
