"""Shared sensor encoder and independent class/fault-type heads for GAT-LSTM v2."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .network import DenseGAT


FAULT_TYPES = ("spike", "bias", "drift", "noise", "stuck", "unavailable", "state_replace")
KIND_TO_ID = {"continuous": 0, "binary": 1, "multistate": 2}


class SharedSensorGATLSTM(nn.Module):
    """Uses one feature encoder for every node; entity IDs are output keys only."""

    architecture_version = "gat_lstm_v2_shared_type"

    def __init__(self, feature_dim: int, entity_kinds: list[str], config):
        super().__init__()
        if not entity_kinds or any(kind not in KIND_TO_ID for kind in entity_kinds):
            raise ValueError("Tipos de sensor insuficientes para GAT-LSTM v2")
        self.input = nn.Linear(feature_dim + len(KIND_TO_ID), 32)
        self.gat = DenseGAT(32, config.gat_heads, config.gat_head_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(32, config.lstm_hidden, batch_first=True)
        self.classifier = nn.Linear(config.lstm_hidden, 3)
        self.fault_type_classifier = nn.Linear(config.lstm_hidden, len(FAULT_TYPES))
        self.register_buffer("kind_ids", torch.tensor([KIND_TO_ID[kind] for kind in entity_kinds],
                                                       dtype=torch.long))

    def forward(self, x, state_ids, adjacency, node_present, kind_ids=None,
                return_encoder=False):
        batch, steps, nodes, _ = x.shape
        if adjacency.shape != (nodes, nodes):
            raise ValueError("Dimensiones del grafo incompatibles con la entrada")
        if kind_ids is None:
            if nodes != len(self.kind_ids):
                raise ValueError("Se requieren tipos declarados para los sensores nuevos")
            kind_ids = self.kind_ids.to(x.device)
        if kind_ids.shape != (nodes,):
            raise ValueError("Tipos de sensor incompatibles con los nodos")
        kind = F.one_hot(kind_ids.long(), num_classes=len(KIND_TO_ID)).to(x.dtype)
        kind = kind.view(1, 1, nodes, len(KIND_TO_ID)).expand(batch, steps, nodes, -1)
        encoded = F.elu(self.input(torch.cat([x, kind], dim=-1)))
        spatial = self.dropout(F.elu(self.gat(encoded, adjacency, node_present)))
        temporal = spatial.permute(0, 2, 1, 3).contiguous().view(batch * nodes, steps, 32)
        output, _ = self.lstm(temporal)
        representation = output[:, -1, :].view(batch, nodes, -1)
        if return_encoder:
            return representation
        return self.classifier(representation), self.fault_type_classifier(representation)
