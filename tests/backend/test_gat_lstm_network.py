import unittest
import torch
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.model_constants import FAULT_TYPES
from app.modulos.clasificador.network_segments import SegmentGATLSTM
from app.modulos.clasificador.segments import FEATURE_NAMES, MAX_SEGMENTS


class NetworkTests(unittest.TestCase):
    def test_shapes_gradients_and_absent_scales(self):
        torch.manual_seed(42)
        model = SegmentGATLSTM(len(FEATURE_NAMES), ["continuous", "multistate"], Config(dropout=0))
        x = torch.zeros(2, 3 * MAX_SEGMENTS, 2, len(FEATURE_NAMES))
        # Every segment has a self edge in its scale.
        for scale in range(3):
            for index in range(MAX_SEGMENTS):
                x[:, scale * MAX_SEGMENTS + index, :, -MAX_SEGMENTS + index] = 1
        states = torch.zeros(2, 3 * MAX_SEGMENTS, 2, dtype=torch.long)
        present = torch.ones_like(states, dtype=torch.bool)
        adjacency = torch.eye(2, dtype=torch.bool)
        logits, types = model(x, states, adjacency, present)
        self.assertEqual(tuple(logits.shape), (2, 2, 3))
        self.assertEqual(tuple(types.shape), (2, 2, len(FAULT_TYPES)))
        (logits.sum() + types.sum()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.eval()
        with torch.no_grad():
            encoded = model(x, states, adjacency, torch.zeros_like(present), return_encoder=True)
        self.assertEqual(encoded.abs().sum().item(), 0)
        self.assertEqual(model.kind_ids.tolist(), [0, 2])

    def test_rejects_legacy_temporal_layout(self):
        model = SegmentGATLSTM(len(FEATURE_NAMES), ["continuous"], Config())
        with self.assertRaisesRegex(ValueError, "tres escalas"):
            model(torch.zeros(1, 4, 1, len(FEATURE_NAMES)), torch.zeros(1, 4, 1),
                  torch.ones(1, 1), torch.ones(1, 4, 1))
