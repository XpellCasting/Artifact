"""Shared PyG segment encoder and chronological multi-scale LSTM."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv
from app.modulos.clasificador.model_constants import FAULT_TYPES, KIND_TO_ID
from app.modulos.clasificador.segments import ARCHITECTURE, MAX_SEGMENTS, MAX_POINTS, BASE_NAMES


class SegmentGATLSTM(nn.Module):
    architecture_version = ARCHITECTURE

    def __init__(self, feature_dim, entity_kinds, config):
        super().__init__()
        self.threshold = config.segment_threshold
        self.encoder_method = config.segment_encoder
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

    def forward(self, x, state_ids, adjacency, node_present, kind_ids=None, return_encoder=False):
        b, steps, n, _ = x.shape
        if steps != 3*MAX_SEGMENTS:
            raise ValueError("Se requieren tres escalas de doce posiciones de segmento")
        kinds = self.kind_ids if kind_ids is None else kind_ids
        kind = F.one_hot(kinds, 3).to(x.dtype).view(1,n,1,3).expand(b,n,MAX_SEGMENTS,3).reshape(b*n,MAX_SEGMENTS,3)
        final, masks = [], []
        for scale in range(3):
            features = x[:,scale*MAX_SEGMENTS:(scale+1)*MAX_SEGMENTS].permute(0,2,1,3).reshape(b*n,MAX_SEGMENTS,-1)
            available = node_present[:,scale*MAX_SEGMENTS:(scale+1)*MAX_SEGMENTS].permute(0,2,1).reshape(b*n,MAX_SEGMENTS).bool()
            scale_present = available.any(-1)
            if self.training:
                # Keep at least one originally available scale per entity.
                all_scales = node_present.reshape(b,3,MAX_SEGMENTS,n).any(2).permute(0,2,1).reshape(b*n,3)
                first = all_scales.long().argmax(-1)
                keep = (torch.rand(b*n, device=x.device) >= .1) | (first == scale)
                available = available & keep[:,None]
                scale_present = available.any(-1)
            categories = features[:,:,self.base_dim:self.base_dim+MAX_POINTS].long().clamp(0,66)
            embedded = self.embedding(categories).flatten(-2)
            scale_code = self.scale_embedding.weight[scale].view(1,1,4).expand(b*n,MAX_SEGMENTS,4)
            encoded = F.elu(self.input(torch.cat([features[:,:,:self.base_dim],embedded,scale_code,kind],-1)))
            similarities = features[:,:,-MAX_SEGMENTS:]
            chronological = []
            for end in range(1,MAX_SEGMENTS+1):
                if self.encoder_method == "lstm_only":
                    chronological.append(self.dropout(encoded[:,end-1]*available[:,end-1,None]))
                    continue
                mask = available[:,:end]
                allowed = (similarities[:,:end,:end] > self.threshold) & mask[:,:,None] & mask[:,None,:]
                graph_id, target, source = allowed.nonzero(as_tuple=True)
                edge_index = torch.stack([graph_id*end+source,graph_id*end+target])
                spatial = F.elu(self.gat(encoded[:,:end].reshape(-1,32),edge_index)).reshape(b*n,end,32)
                pooled = (spatial*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
                chronological.append(self.dropout(pooled))
            output, _ = self.lstm(torch.stack(chronological,1))
            last = available.sum(1).clamp_min(1)-1
            representation = output[torch.arange(b*n,device=x.device),last]*scale_present[:,None]
            final.append(representation)
            masks.append(scale_present.to(x.dtype))
        representation = torch.cat([*final,torch.stack(masks,-1)],-1).reshape(b,n,-1)
        if return_encoder:
            return representation
        return self.classifier(representation), self.fault_type_classifier(representation)


class OptionalEvidenceGATLSTM(nn.Module):
    """Frozen autonomous backbone with an exactly gated residual correction."""
    architecture_version = "gat_lstm_v4_optional_evidence"

    def __init__(self, backbone):
        super().__init__()
        from app.modulos.clasificador.evidence_features import FEATURE_NAMES
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        hidden = backbone.classifier.in_features
        self.residual = nn.Sequential(nn.Linear(hidden+len(FEATURE_NAMES), 32), nn.ReLU(),
                                      nn.Linear(32, 3+len(FAULT_TYPES)))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @property
    def kind_ids(self):
        return self.backbone.kind_ids

    @property
    def threshold(self):
        return self.backbone.threshold

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x, state_ids, adjacency, node_present, kind_ids=None,
                evidence=None, evidence_valid=None):
        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone(x, state_ids, adjacency, node_present, kind_ids, return_encoder=True)
            logits = self.backbone.classifier(h)
            types = self.backbone.fault_type_classifier(h)
        if evidence is None or evidence_valid is None:
            return logits, types
        correction = self.residual(torch.cat([h, evidence], -1))
        correction = torch.where(evidence_valid[..., None], correction, torch.zeros_like(correction))
        return logits+correction[..., :3], types+correction[..., 3:]
