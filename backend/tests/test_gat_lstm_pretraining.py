import unittest
import torch

from app.GAT_LSTM.config import Config
from app.GAT_LSTM.pretraining import predictive_loss, pretrain_encoder


def example(observed=True):
    return {"X": [[[0.0] * 19] for _ in range(3)], "state_ids": [[0]] * 3,
            "node_present": [[True]] * 3, "adjacency": [[True]],
            "y": [-1], "label_mask": [False],
            "next_target": [1.0], "next_observed": [observed],
            "metadata": {"target_time": "2026-02-01T01:00:00Z",
                         "next_time": "2026-02-01T01:01:00Z",
                         "partition_start": "2026-02-01T00:00:00Z",
                         "partition_end": "2026-02-02T00:00:00Z"}}


class PretrainingTests(unittest.TestCase):
    def test_missing_next_report_has_no_loss(self):
        value = torch.tensor([[9.0]], requires_grad=True)
        self.assertIsNone(predictive_loss(value, [torch.zeros(1, 3)],
                                           torch.tensor([[1.0]]), torch.tensor([[False]]), ["continuous"]))

    def test_future_boundary(self):
        row = example()
        row["metadata"]["next_time"] = row["metadata"]["partition_end"]
        with self.assertRaisesRegex(ValueError, "fuera"):
            pretrain_encoder([row], [example()], Config(sequence_steps=3), [3], ["continuous"])

    def test_encoder_only(self):
        config = Config(sequence_steps=3, max_epochs=2, patience=2)
        weights, metrics = pretrain_encoder([example(), example(False)], [example()],
                                            config, [3], ["continuous"])
        self.assertIsInstance(weights, dict)
        self.assertFalse(metrics["can_activate"])


if __name__ == "__main__":
    unittest.main()
