import unittest

import torch

from app.modulos.clasificador.config import Config
from app.modulos.clasificador.attention import DenseGAT
from app.modulos.clasificador.network_v2 import SharedSensorGATLSTM


class NetworkTests(unittest.TestCase):
    def test_shapes_gradients_and_empty(self):
        torch.manual_seed(42)
        config = Config(sequence_steps=4, dropout=0)
        model = SharedSensorGATLSTM(19, ["continuous", "multistate"], config)
        x = torch.randn(2, 4, 2, 19)
        states = torch.zeros(2, 4, 2, dtype=torch.long)
        present = torch.ones(2, 4, 2, dtype=torch.bool)
        adjacency = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        logits, types = model(x, states, adjacency, present)
        self.assertEqual(tuple(logits.shape), (2, 2, 3))
        logits.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        empty = model.gat(torch.randn(1, 2, 2, 32), adjacency,
                          torch.zeros(1, 2, 2, dtype=torch.bool))
        self.assertTrue(torch.isfinite(empty).all())
        self.assertEqual(empty.abs().sum().item(), 0)
        mixed = SharedSensorGATLSTM(19, ["continuous", "multistate"], config)
        self.assertEqual(mixed.kind_ids.tolist(), [0, 2])

    def test_connected_neighbor_only(self):
        torch.manual_seed(3)
        gat = DenseGAT(1, 1, 1)
        with torch.no_grad():
            gat.projection.weight.fill_(1)
            gat.attn_target.fill_(0)
            gat.attn_source.fill_(0)
        x = torch.tensor([[[[1.0], [2.0], [3.0]]]])
        present = torch.ones(1, 1, 3, dtype=torch.bool)
        a = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)
        output = gat(x, a, present)
        self.assertAlmostEqual(output[0, 0, 0, 0].item(), 1.5)
        x[0, 0, 1, 0] = 10
        changed = gat(x, a, present)
        self.assertAlmostEqual(changed[0, 0, 0, 0].item(), 5.5)
        x[0, 0, 2, 0] = 99
        disconnected = gat(x, a, present)
        self.assertEqual(changed[0, 0, 0, 0].item(), disconnected[0, 0, 0, 0].item())

    def test_manual_attention(self):
        gat = DenseGAT(1, 1, 1)
        with torch.no_grad():
            gat.projection.weight.fill_(1)
            gat.attn_target.fill_(0)
            gat.attn_source.fill_(1)
        x = torch.tensor([[[[1.0], [2.0]]]])
        a = torch.ones(2, 2, dtype=torch.bool)
        present = torch.ones(1, 1, 2, dtype=torch.bool)
        expected = (torch.exp(torch.tensor(1.0)) * 1 + torch.exp(torch.tensor(2.0)) * 2) / (
            torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(2.0)))
        self.assertAlmostEqual(gat(x, a, present)[0, 0, 0, 0].item(), expected.item(), places=6)


if __name__ == "__main__":
    unittest.main()
