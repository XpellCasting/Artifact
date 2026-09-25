"""GAT densa pequeña seguida de LSTM causal compartida por entidad.

Importar este módulo requiere PyTorch; el backend convencional no lo importa.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DenseGAT(nn.Module):
    def __init__(self, input_dim: int, heads: int = 4, head_dim: int = 8):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.projection = nn.Linear(input_dim, heads * head_dim, bias=False)
        self.attn_target = nn.Parameter(torch.empty(heads, head_dim))
        self.attn_source = nn.Parameter(torch.empty(heads, head_dim))
        nn.init.xavier_uniform_(self.attn_target)
        nn.init.xavier_uniform_(self.attn_source)

    def forward(self, x, adjacency, node_present):
        """x[B,T,N,D], A[dest,source], presence[B,T,N]."""
        b, t, n, _ = x.shape
        projected = self.projection(x).view(b, t, n, self.heads, self.head_dim)
        target_score = (projected * self.attn_target).sum(-1).permute(0, 1, 3, 2)
        source_score = (projected * self.attn_source).sum(-1).permute(0, 1, 3, 2)
        scores = F.leaky_relu(target_score.unsqueeze(-1) + source_score.unsqueeze(-2), .2)
        allowed = adjacency.bool().view(1, 1, 1, n, n) & node_present.bool().view(b, t, 1, 1, n)
        # A destination with no informative source has an all-zero attention
        # row. Softmax receives safe zeros, then the row is explicitly masked.
        any_source = allowed.any(-1, keepdim=True)
        safe = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        safe = torch.where(any_source, safe, torch.zeros_like(safe))
        weights = torch.softmax(safe, dim=-1) * allowed.to(scores.dtype)
        values = projected.permute(0, 1, 3, 2, 4)
        result = torch.einsum("bthin,bthnd->bthid", weights, values)
        return result.permute(0, 1, 3, 2, 4).reshape(b, t, n, self.heads * self.head_dim)


class EntityGATLSTM(nn.Module):
    def __init__(self, feature_dim: int, category_sizes: list[int], config, entity_kinds=None):
        super().__init__()
        self.node_count = len(category_sizes)
        self.embeddings = nn.ModuleList(nn.Embedding(max(3, count), 8) for count in category_sizes)
        self.input = nn.Linear(feature_dim + 8 + 3, 32)
        self.gat = DenseGAT(32, config.gat_heads, config.gat_head_dim)
        if config.gat_heads * config.gat_head_dim != 32:
            raise ValueError("La salida GAT debe tener dimensión 32")
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(32, config.lstm_hidden, batch_first=True)
        self.classifier = nn.Linear(config.lstm_hidden, 3)
        kind_to_id = {"continuous": 0, "binary": 1, "multistate": 2}
        entity_kinds = entity_kinds or ["continuous"] * len(category_sizes)
        if len(entity_kinds) != len(category_sizes) or any(kind not in kind_to_id for kind in entity_kinds):
            raise ValueError("Tipos de entidad incompletos o desconocidos")
        self.register_buffer("kind_ids", torch.tensor([kind_to_id[kind] for kind in entity_kinds], dtype=torch.long))

    def forward(self, x, state_ids, adjacency, node_present, kind_ids=None, return_encoder=False):
        b, t, n, _ = x.shape
        if n != self.node_count or state_ids.shape != (b, t, n):
            raise ValueError("Dimensiones de entidades incompatibles")
        embeddings = torch.stack([layer(state_ids[:, :, i]) for i, layer in enumerate(self.embeddings)], dim=2)
        if kind_ids is None:
            kind_ids = self.kind_ids.to(x.device)
        kind = F.one_hot(kind_ids.long(), num_classes=3).to(x.dtype).view(1, 1, n, 3).expand(b, t, n, 3)
        encoded = F.elu(self.input(torch.cat([x, embeddings, kind], dim=-1)))
        spatial = self.dropout(F.elu(self.gat(encoded, adjacency, node_present)))
        # [B,N,T,D] -> [B*N,T,D], never mixing time with node order.
        temporal = spatial.permute(0, 2, 1, 3).contiguous().view(b * n, t, 32)
        output, _ = self.lstm(temporal)
        final = output[:, -1, :].view(b, n, -1)
        if return_encoder:
            return final
        return self.classifier(final)
