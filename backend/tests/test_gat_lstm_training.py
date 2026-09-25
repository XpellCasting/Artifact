import tempfile
import unittest
from pathlib import Path

import torch

from app.GAT_LSTM.config import Config
from app.GAT_LSTM.training import (class_weights_from_episodes, load_candidate, _episode_weights,
                                   masked_classification_loss, save_candidate, train_classifier)


def sample(label):
    value = float(label)
    features = [value] + [0.0] * 18
    return {"X": [[features] for _ in range(4)], "state_ids": [[0]] * 4,
            "node_present": [[True]] * 4, "adjacency": [[True]],
            "y": [label], "label_mask": [True],
            "metadata": {"sample_id": f"s{label}", "parent_episode_id": f"e{label}"}}


class TrainingTests(unittest.TestCase):
    def test_masked_loss_and_missing_class(self):
        logits = torch.tensor([[[5.0, 0, 0], [0, 0, 5.0]]])
        y = torch.tensor([[0, -1]])
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(masked_classification_loss(logits, y, mask).item(),
                               torch.nn.functional.cross_entropy(logits[:, 0], y[:, 0]).item())
        self.assertIsNone(masked_classification_loss(logits, y, torch.zeros_like(mask)))
        with self.assertRaisesRegex(ValueError, "insufficient_labels"):
            class_weights_from_episodes([sample(0)])

    def test_train_and_reload(self):
        config = Config(sequence_steps=4, max_epochs=12, patience=12, learning_rate=.01, dropout=0)
        rows = [sample(0), sample(1), sample(2)]
        model, metrics = train_classifier(rows, rows, config, [3], ["continuous"])
        self.assertTrue(metrics["history"])
        prep = {"entity_ids": ["a"], "feature_names": [str(i) for i in range(19)],
                "entities": {"a": {"kind": "continuous", "normal_references": []}}}
        with tempfile.TemporaryDirectory() as temp:
            manifest = save_candidate(model, prep, {"adjacency": [[True]]}, config, Path(temp), metrics)
            loaded, _, _, _ = load_candidate(manifest["model_id"], Path(temp))
            x = torch.tensor([rows[0]["X"]]); states = torch.zeros(1, 4, 1, dtype=torch.long)
            presence = torch.ones(1, 4, 1, dtype=torch.bool)
            adjacency = torch.ones(1, 1, dtype=torch.bool)
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(x, states, adjacency, presence),
                                               loaded(x, states, adjacency, presence), atol=1e-6))
            self.assertFalse(manifest["active"])
            self.assertEqual(manifest["rejection_policy"], metrics["rejection_policy"])
            self.assertEqual(loaded.kind_ids.tolist(), [0])

    def test_episode_weights_use_whole_dataset_counts(self):
        repeated = sample(0); repeated["metadata"]["parent_episode_id"] = "same"
        counts = {"same": 2}
        self.assertEqual(_episode_weights([repeated], "cpu", counts).item(), .5)


if __name__ == "__main__":
    unittest.main()
