"""Numerical references from official GATConv and LSTM equations.

CPU, float64, frozen dropout. Synthetic tensors are mathematical fixtures,
not a diagnostic benchmark or a reproduction of the bearing paper.
"""
import unittest
from datetime import timedelta

import torch
from torch.nn import functional as F

from app.modulos.clasificador.config import Config
from app.modulos.clasificador.network_segments import SegmentGATLSTM
from app.modulos.clasificador.segments import FEATURE_NAMES, MAX_SEGMENTS, SCALES, build_sample
from test_segment_multiscale import fixture


def gat_equations(layer, x, edges):
    z = F.linear(x, layer.lin.weight).reshape(len(x), layer.heads, layer.out_channels)
    rows = []
    for target in range(len(x)):
        source = edges[0, edges[1] == target]
        if len(source):
            score = F.leaky_relu((z[source] * layer.att_src[0]).sum(-1)
                + (z[target] * layer.att_dst[0]).sum(-1), layer.negative_slope)
            alpha = torch.softmax(score, dim=0)
            value = (alpha.unsqueeze(-1) * z[source]).sum(0)
        else:
            value = z.new_zeros(layer.heads, layer.out_channels)
        rows.append(value.flatten())
    return torch.stack(rows) + layer.bias


def lstm_equations(layer, x):
    h = x.new_zeros(len(x), layer.hidden_size)
    c = h.clone()
    outputs = []
    for t in range(x.shape[1]):
        gates = F.linear(x[:, t], layer.weight_ih_l0, layer.bias_ih_l0)
        gates = gates + F.linear(h, layer.weight_hh_l0, layer.bias_hh_l0)
        i, f, g, o = gates.chunk(4, -1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        outputs.append(h)
    return torch.stack(outputs, 1)


class OfficialReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = SegmentGATLSTM(len(FEATURE_NAMES), ["continuous", "multistate"],
                                   Config(dropout=0)).double().eval()

    def test_gat_output_and_gradients_match_documented_equations(self):
        layer = self.model.gat
        x = torch.randn(5, 32, dtype=torch.float64, requires_grad=True)
        # Directed edges, unequal incoming degrees and an isolated node.
        edges = torch.tensor([[0, 1, 2, 2, 3, 0], [0, 0, 1, 2, 2, 3]])
        actual = layer(x, edges)
        expected = gat_equations(layer, x, edges)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        parameters = [x, layer.lin.weight, layer.att_src, layer.att_dst, layer.bias]
        weight = torch.randn_like(actual)
        grads = torch.autograd.grad((actual * weight).sum(), parameters, retain_graph=True)
        reference_grads = torch.autograd.grad((expected * weight).sum(), parameters)
        for observed, reference in zip(grads, reference_grads):
            torch.testing.assert_close(observed, reference, atol=1e-10, rtol=1e-9)

    def test_attention_normalizes_over_incoming_neighbors(self):
        x = torch.randn(4, 32, dtype=torch.float64)
        edges = torch.tensor([[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 2, 2]])
        _, (returned_edges, alpha) = self.model.gat(x, edges, return_attention_weights=True)
        self.assertTrue(torch.equal(edges, returned_edges))
        for target in range(4):
            torch.testing.assert_close(alpha[returned_edges[1] == target].sum(0),
                torch.ones(self.model.gat.heads, dtype=torch.float64))

    def test_lstm_output_and_gradients_match_documented_gates(self):
        layer = self.model.lstm
        x = torch.randn(3, 5, 32, dtype=torch.float64, requires_grad=True)
        actual, (last, _) = layer(x)
        expected = lstm_equations(layer, x)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(last[0], expected[:, -1], atol=1e-10, rtol=1e-9)
        parameters = [x, *layer.parameters()]
        grads = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
        refs = torch.autograd.grad(expected.square().sum(), parameters)
        for observed, reference in zip(grads, refs):
            torch.testing.assert_close(observed, reference, atol=1e-10, rtol=1e-9)

    def test_batched_graphs_match_processing_each_entity_separately(self):
        x = torch.zeros(2, 36, 2, len(FEATURE_NAMES), dtype=torch.float64)
        x[:, :, :, :19] = torch.randn(2, 36, 2, 19, dtype=torch.float64)
        similarity = torch.eye(MAX_SEGMENTS, dtype=torch.float64)
        similarity[1, 0] = .9
        for scale in range(3):
            x[:, scale * 12:(scale + 1) * 12, :, -12:] = similarity[:, None, :]
        present = torch.zeros(2, 36, 2, dtype=torch.bool)
        for scale in range(3):
            for batch in range(2):
                for node in range(2):
                    count = 4 + batch + node + scale
                    present[batch, scale * 12:scale * 12 + count, node] = True
        with torch.no_grad():
            actual = self.model(x, None, None, present)
            for batch in range(2):
                for node in range(2):
                    expected = self.model(x[batch:batch+1, :, node:node+1], None, None,
                        present[batch:batch+1, :, node:node+1], self.model.kind_ids[node:node+1])
                    for head, ref in zip(actual, expected):
                        torch.testing.assert_close(head[batch, node], ref[0, 0], atol=1e-10, rtol=1e-9)

    def test_lstm_receives_chronological_prefix_mean_pooling(self):
        x = torch.zeros(1, 36, 2, len(FEATURE_NAMES), dtype=torch.float64)
        x[:, :, :, :19] = torch.randn(1, 36, 2, 19, dtype=torch.float64)
        for scale in range(3):
            x[:, scale*12:(scale+1)*12, :, -12:] = torch.eye(12)[:, None, :]
        present = torch.ones(1, 36, 2, dtype=torch.bool)
        gat_calls, lstm_inputs = [], []
        hooks = [self.model.gat.register_forward_hook(lambda m, args, out: gat_calls.append((args, out))),
                 self.model.lstm.register_forward_pre_hook(lambda m, args: lstm_inputs.append(args[0]))]
        try:
            with torch.no_grad():
                self.model(x, None, None, present)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(gat_calls), 36)
        self.assertEqual(len(lstm_inputs), 3)
        for scale in range(3):
            expected = []
            for step in range(12):
                args, output = gat_calls[scale * 12 + step]
                size = step + 1
                # Flattened graphs must have no cross-entity edges.
                self.assertTrue(torch.equal(args[1][0] // size, args[1][1] // size))
                expected.append(F.elu(output).reshape(2, size, 32).mean(1))
            torch.testing.assert_close(lstm_inputs[scale], torch.stack(expected, 1), atol=1e-10, rtol=1e-9)

    def test_every_declared_segmentation_candidate_fits_tensor_capacity(self):
        config, prep, graph, provider, _, end = fixture()
        for scale, (_, resolution, candidates) in enumerate(SCALES):
            for size in candidates:
                with self.subTest(scale=scale, segment_steps=size):
                    spec = dict(graph, segment_steps=list(graph["segment_steps"]))
                    spec["segment_steps"][scale] = size
                    sample = build_sample(provider, end, prep, spec, config)
                    self.assertEqual(len(sample["X"]), 36)
                    self.assertLessEqual(sample["metadata"]["scale_coverage"]["a"][scale]["segments"], 12)


if __name__ == "__main__":
    unittest.main()
