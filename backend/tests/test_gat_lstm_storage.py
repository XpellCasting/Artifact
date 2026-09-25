import json
from pathlib import Path
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

from app.GAT_LSTM.storage import DiskPartition, FIELDS, PartitionWriter, sample_summaries


def sample(index):
    return {"X": np.full((8, 4, 19), index / 10).tolist(),
            "state_ids": [[index % 3] * 4] * 8, "node_present": [[True] * 4] * 8,
            "y": [index % 3, -1, -1, -1], "label_mask": [True, False, False, False],
            "metadata": {"sample_id": str(index), "parent_episode_id": str(index // 3)},
            "next_target": [1.0] * 4, "next_observed": [True, False, False, False]}


class StorageTests(unittest.TestCase):
    def test_streamed_arrays_match_eager_format_and_lazy_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [sample(i) for i in range(7)]
            (root / "graph.json").write_text(json.dumps({"adjacency": np.eye(4).tolist()}))
            with PartitionWriter(root, "train_fit") as writer:
                for row in rows:
                    writer.append(row)
                writer.finish()
                self.assertEqual(writer.count, 7)
                self.assertEqual(sum(writer.labels.values()), 7)
                self.assertEqual(len(writer.episodes["normal"]), 3)
            for name, dtype in FIELDS.items():
                np.testing.assert_array_equal(np.load(root / f"train_fit_{name}.npy"),
                                              np.asarray([row[name] for row in rows], dtype=dtype))
            self.assertFalse(list(root.glob("*.raw")))
            with DiskPartition(root, "train_fit") as dataset:
                self.assertEqual(len(dataset), 7)
                self.assertTrue(all(not isinstance(array, np.memmap) for array in dataset.arrays.values()))
                window = dataset[0]["X"]
                self.assertFalse(isinstance(window, np.memmap))
                np.testing.assert_array_equal(window, np.asarray(rows[0]["X"], dtype=np.float32))
                # Weighting and provenance must never read the full feature windows.
                read = np.fromfile
                def labels_only(path, **kwargs):
                    self.assertTrue(str(path).endswith(("_y.npy", "_label_mask.npy")))
                    return read(path, **kwargs)
                with patch("app.GAT_LSTM.storage.np.fromfile", side_effect=labels_only):
                    summaries = list(sample_summaries(dataset))
                self.assertEqual([s["y"] for s in summaries], [s["y"] for s in rows])
                self.assertEqual(dataset[-1]["metadata"], rows[-1]["metadata"])
                self.assertEqual([s["metadata"] for s in dataset[1:6:2]],
                                 [s["metadata"] for s in rows[1:6:2]])
                self.assertIsInstance(dataset[0]["label_mask"][0], bool)
                with self.assertRaises(IndexError):
                    dataset[7]

    def test_predictive_roundtrip_and_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "graph.json").write_text(json.dumps({"adjacency": np.eye(4).tolist()}))
            with PartitionWriter(root, "train_pretrain", predictive=True) as writer:
                writer.append(sample(1))
                writer.finish()
            with DiskPartition(root, "train_pretrain", predictive=True) as dataset:
                np.testing.assert_array_equal(dataset[0]["next_target"], [1.0] * 4)
                self.assertEqual(dataset[0]["y"], [-1] * 4)
                self.assertFalse(any(dataset[0]["label_mask"]))
            (root / "train_pretrain_metadata.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "no coinciden"):
                DiskPartition(root, "train_pretrain", predictive=True)

    def test_writer_does_not_retain_feature_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracemalloc.start()
            try:
                with PartitionWriter(tmp, "train_fit") as writer:
                    for index in range(500):
                        writer.append(sample(index))
                    _, peak = tracemalloc.get_traced_memory()
                    writer.finish()
                # Retaining all 500 nested float windows would exceed 10 MiB.
                self.assertLess(peak, 3 * 1024 * 1024)
            finally:
                tracemalloc.stop()


if __name__ == "__main__":
    unittest.main()
