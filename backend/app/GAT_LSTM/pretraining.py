"""Preentrenamiento predictivo opcional sobre observaciones futuras originales."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
from itertools import chain
import os
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .network import EntityGATLSTM
from .config import parse_utc, required_context_seconds
from .storage import sample_summaries


class PredictiveEncoder(nn.Module):
    def __init__(self, feature_dim, category_sizes, entity_kinds, config):
        super().__init__()
        self.encoder = EntityGATLSTM(feature_dim, category_sizes, config, entity_kinds)
        self.continuous = nn.Linear(config.lstm_hidden, 1)
        self.categorical = nn.ModuleList(nn.Linear(config.lstm_hidden, max(3, count))
                                         for count in category_sizes)
        self.entity_kinds = entity_kinds

    def forward(self, x, state_ids, adjacency, node_present):
        representation = self.encoder(x, state_ids, adjacency, node_present, return_encoder=True)
        value = self.continuous(representation).squeeze(-1)
        category = [head(representation[:, i]) for i, head in enumerate(self.categorical)]
        return value, category


def predictive_loss(value, categories, targets, masks, kinds):
    """Solo máscaras de reportes originales del paso objetivo; sin copias retenidas."""
    losses = []
    for index, kind in enumerate(kinds):
        active = masks[:, index].bool()
        if not active.any():
            continue
        if kind == "continuous":
            loss = F.huber_loss(value[active, index], targets[active, index].float())
        else:
            loss = F.cross_entropy(categories[index][active], targets[active, index].long())
        losses.append(loss)
    return torch.stack(losses).mean() if losses else None


def pretrain_encoder(train_samples, dev_samples, config, category_sizes, entity_kinds, progress=None):
    """Cada muestra incluye next_target[N], next_observed[N] y next_time dentro del corte."""
    if not train_samples or not dev_samples:
        raise ValueError("Sin objetivos predictivos en train_fit o dev_tune")
    for sample in chain(sample_summaries(train_samples), sample_summaries(dev_samples)):
        metadata = sample["metadata"]
        target_time = parse_utc(metadata["target_time"])
        next_time = parse_utc(metadata["next_time"])
        partition_start = parse_utc(metadata["partition_start"])
        partition_end = parse_utc(metadata["partition_end"])
        if (next_time != target_time + timedelta(seconds=config.grid_seconds)
                or next_time >= partition_end
                or target_time.timestamp() - required_context_seconds(config) < partition_start.timestamp()):
            raise ValueError("Objetivo predictivo fuera del futuro permitido")
    from .training import _tensor_batch
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    feature_dim = len(train_samples[0]["X"][0][0])
    model = PredictiveEncoder(feature_dim, category_sizes, entity_kinds, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history, best, best_loss, stale = [], None, float("inf"), 0
    for epoch in range(config.max_epochs):
        model.train()
        total, count = 0.0, 0
        order = torch.randperm(len(train_samples)).tolist()
        for start in range(0, len(order), config.batch_size):
            batch = [train_samples[i] for i in order[start:start + config.batch_size]]
            x, ids, presence, _, _, adjacency = _tensor_batch(batch, device)
            targets = torch.as_tensor(np.stack([s["next_target"] for s in batch]), device=device)
            masks = torch.as_tensor(np.stack([s["next_observed"] for s in batch]), dtype=torch.bool, device=device)
            if not masks.any():
                continue
            optimizer.zero_grad()
            value, category = model(x, ids, adjacency, presence)
            loss = predictive_loss(value, category, targets, masks, entity_kinds)
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida predictiva no finita")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += loss.item(); count += 1
        if count == 0:
            raise ValueError("Sin observaciones futuras válidas para preentrenar")
        model.eval()
        with torch.no_grad():
            dev_total, dev_count = 0.0, 0
            for start in range(0, len(dev_samples), config.batch_size):
                batch = dev_samples[start:start + config.batch_size]
                x, ids, presence, _, _, adjacency = _tensor_batch(batch, device)
                targets = torch.as_tensor(np.stack([s["next_target"] for s in batch]), device=device)
                masks = torch.as_tensor(np.stack([s["next_observed"] for s in batch]), dtype=torch.bool, device=device)
                if masks.any():
                    value, category = model(x, ids, adjacency, presence)
                    dev_total += predictive_loss(value, category, targets, masks, entity_kinds).item()
                    dev_count += 1
        if not dev_count:
            raise ValueError("Sin observaciones predictivas en dev_tune")
        row = {"epoch": epoch + 1, "train_loss": total / count, "dev_loss": dev_total / dev_count}
        history.append(row)
        if progress:
            progress(row)
        if row["dev_loss"] < best_loss:
            best_loss, best, stale = row["dev_loss"], deepcopy(model.encoder.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    return best, {"history": history, "best_dev_loss": best_loss,
                  "kind": "encoder_only", "can_activate": False}


def save_encoder(weights, preprocessor, graph, config, directory: Path, metrics):
    directory = Path(directory)
    identifier = uuid4().hex
    target = directory / "encoders"
    target.mkdir(parents=True, exist_ok=True)
    payload = target / f"{identifier}.pt"
    temp = payload.with_suffix(".pt.tmp")
    torch.save({key: value.detach().cpu() for key, value in weights.items()}, temp)
    os.replace(temp, payload)
    manifest = {"encoder_id": identifier, "kind": "encoder_only", "can_activate": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": config.schema_version,
                "weights_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "config": config.as_dict(), "preprocessor": preprocessor, "graph": graph,
                "metrics": metrics}
    path = target / f"{identifier}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return manifest
