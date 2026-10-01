"""Atención del grafo utilizada por GAT-LSTM v2."""
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
