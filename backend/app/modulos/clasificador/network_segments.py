"""Shared PyG segment encoder and chronological multi-scale LSTM."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv
from app.config import setting
from app.modulos.clasificador.model_constants import FAULT_TYPES, KIND_TO_ID
from app.modulos.clasificador.segments import ARCHITECTURE, MAX_SEGMENTS, MAX_POINTS, BASE_NAMES


class SegmentGATLSTM(nn.Module):
    architecture_version = ARCHITECTURE

    def __init__(self, feature_dim, entity_kinds, config):
        super().__init__()
        self.threshold = config.segment_threshold
        self.encoder_method = config.segment_encoder
        self.attention_backend = setting("GAT_ATTENTION_BACKEND")
        if self.attention_backend not in {"dense", "pyg"}:
            raise ValueError("GAT_ATTENTION_BACKEND debe ser dense o pyg")
        self.embedding = nn.Embedding(67, 4, padding_idx=0)
        self.scale_embedding = nn.Embedding(3, 4)
        self.base_dim = len(BASE_NAMES)+2*MAX_POINTS
        self.input = nn.Linear(self.base_dim+4*MAX_POINTS+4+3, 32)
        self.gat = GATConv(32, config.gat_head_dim, heads=config.gat_heads, add_self_loops=False)
        self.lstm = nn.LSTM(32, config.lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(3*config.lstm_hidden+3, 3)
        self.fault_type_classifier = nn.Linear(3*config.lstm_hidden+3, len(FAULT_TYPES))
        self.register_buffer("kind_ids", torch.tensor([KIND_TO_ID[k] for k in entity_kinds], dtype=torch.long))

    def _dense_prefixes(self, encoded, similarities, available):
        """Same GATConv weights/edges, batched over all twelve causal prefixes.

        Dense attention is bounded by the fixed twelve-segment graph. Avoids
        dynamic nonzero/scatter graphs and projects each segment only once.
        Rows with no incoming edges still receive GATConv's learned bias.
        """
        groups, steps, _ = encoded.shape
        heads, channels = self.gat.heads, self.gat.out_channels
        projected = self.gat.lin(encoded).view(groups, steps, heads, channels)
        source = (projected * self.gat.att_src).sum(-1)
        target = (projected * self.gat.att_dst).sum(-1)
        logits = F.leaky_relu(target[:, :, None] + source[:, None, :],
                              self.gat.negative_slope)
        position = torch.arange(steps, device=encoded.device)
        prefix = position[None, :] <= position[:, None]
        edges = ((similarities > self.threshold) & available[:, :, None]
                 & available[:, None, :])
        allowed = (edges[:, None] & prefix[None, :, :, None]
                   & prefix[None, :, None, :])
        # Finite sentinel avoids NaN gradients for entirely empty graphs.
        attention = logits[:, None].masked_fill(~allowed[..., None], torch.finfo(logits.dtype).min)
        attention = attention.softmax(dim=3) * allowed[..., None]
        spatial = torch.einsum("gpijh,gjhc->gpihc", attention, projected)
        spatial = spatial.reshape(groups, steps, steps, heads * channels)
        spatial = F.elu(spatial + self.gat.bias)
        mask = available[:, None, :] & prefix[None]
        return ((spatial * mask[..., None]).sum(2)
                / mask.sum(2).clamp_min(1)[..., None])

    def _pyg_prefixes(self, encoded, similarities, available):
        """Original sparse implementation, retained for numerical comparisons."""
        groups, steps, _ = encoded.shape
        pooled = []
        for end in range(1, steps + 1):
            mask = available[:, :end]
            allowed = ((similarities[:, :end, :end] > self.threshold)
                       & mask[:, :, None] & mask[:, None, :])
            graph_id, target, source = allowed.nonzero(as_tuple=True)
            edge_index = torch.stack([graph_id * end + source, graph_id * end + target])
            spatial = F.elu(self.gat(encoded[:, :end].reshape(-1, 32), edge_index)).reshape(groups, end, 32)
            pooled.append((spatial * mask[:, :, None]).sum(1) / mask.sum(1).clamp_min(1)[:, None])
        return torch.stack(pooled, 1)

    def forward(self, x, state_ids, adjacency, node_present, kind_ids=None, return_encoder=False):
        b, steps, n, _ = x.shape
        if steps != 3*MAX_SEGMENTS:
            raise ValueError("Se requieren tres escalas de doce posiciones de segmento")
        kinds = self.kind_ids if kind_ids is None else kind_ids
        kind = F.one_hot(kinds, 3).to(x.dtype).view(1,n,1,3).expand(b,n,MAX_SEGMENTS,3).reshape(b*n,MAX_SEGMENTS,3)
        final, masks = [], []
        if self.training:
            all_scales = node_present.reshape(b,3,MAX_SEGMENTS,n).any(2).permute(0,2,1).reshape(b*n,3)
            first = all_scales.long().argmax(-1)
        for scale in range(3):
            features = x[:,scale*MAX_SEGMENTS:(scale+1)*MAX_SEGMENTS].permute(0,2,1,3).reshape(b*n,MAX_SEGMENTS,-1)
            available = node_present[:,scale*MAX_SEGMENTS:(scale+1)*MAX_SEGMENTS].permute(0,2,1).reshape(b*n,MAX_SEGMENTS).bool()
            scale_present = available.any(-1)
            if self.training:
                # Keep at least one originally available scale per entity.
                keep = (torch.rand(b*n, device=x.device) >= .1) | (first == scale)
                available = available & keep[:,None]
                scale_present = available.any(-1)
            categories = features[:,:,self.base_dim:self.base_dim+MAX_POINTS].long().clamp(0,66)
            embedded = self.embedding(categories).flatten(-2)
            scale_code = self.scale_embedding.weight[scale].view(1,1,4).expand(b*n,MAX_SEGMENTS,4)
            encoded = F.elu(self.input(torch.cat([features[:,:,:self.base_dim],embedded,scale_code,kind],-1)))
            similarities = features[:,:,-MAX_SEGMENTS:]
            if self.encoder_method == "lstm_only":
                pooled = encoded * available[..., None]
            elif self.attention_backend == "dense":
                pooled = self._dense_prefixes(encoded, similarities, available)
            else:
                pooled = self._pyg_prefixes(encoded, similarities, available)
            # Preserve the contiguous dropout input layout of the sparse loop.
            chronological = [self.dropout(pooled[:, end].contiguous()) for end in range(MAX_SEGMENTS)]
            output, _ = self.lstm(torch.stack(chronological,1))
            last = available.sum(1).clamp_min(1)-1
            representation = output[torch.arange(b*n,device=x.device),last]*scale_present[:,None]
            final.append(representation)
            masks.append(scale_present.to(x.dtype))
        representation = torch.cat([*final,torch.stack(masks,-1)],-1).reshape(b,n,-1)
        if return_encoder:
            return representation
        return self.classifier(representation), self.fault_type_classifier(representation)
