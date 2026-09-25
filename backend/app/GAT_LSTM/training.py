"""Ajuste supervisado de ventanas; no activa candidatos automáticamente."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import monotonic
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F

from .network import EntityGATLSTM
from .evaluation import choose_rejection_policy
from .storage import sample_summaries


def _tensor_batch(samples, device):
    def tensor(field, dtype):
        # Stack only the current batch.
        return torch.as_tensor(np.stack([s[field] for s in samples]), dtype=dtype, device=device)
    x = tensor("X", torch.float32)
    states = tensor("state_ids", torch.long)
    present = tensor("node_present", torch.bool)
    y = tensor("y", torch.long)
    mask = tensor("label_mask", torch.bool)
    adjacency = torch.tensor(samples[0]["adjacency"], dtype=torch.bool, device=device)
    return x, states, present, y, mask, adjacency


def _episode_weights(samples, device, full_episode_counts=None):
    key = lambda s, i: s["metadata"].get("parent_episode_id", s["metadata"].get("sample_id", str(i)))
    episodes = full_episode_counts or Counter(key(s, i) for i, s in enumerate(samples))
    return torch.tensor([1 / episodes[key(s, i)] for i, s in enumerate(samples)],
                        dtype=torch.float32, device=device)


def masked_classification_loss(logits, y, mask, class_weights=None, episode_weights=None):
    if not mask.any():
        return None
    safe_targets = y.clamp(min=0)
    loss = F.cross_entropy(logits.reshape(-1, 3), safe_targets.reshape(-1),
                           weight=class_weights, reduction="none").view_as(y)
    if episode_weights is not None:
        loss = loss * episode_weights.view(-1, 1)
    return loss[mask].sum() / ((episode_weights.view(-1, 1).expand_as(y)[mask].sum())
                               if episode_weights is not None else mask.sum())


def class_weights_from_episodes(samples, device="cpu"):
    by_class = [set() for _ in range(3)]
    for index, sample in enumerate(sample_summaries(samples)):
        group = sample["metadata"].get("parent_episode_id", sample["metadata"].get("sample_id", str(index)))
        for label, active in zip(sample["y"], sample["label_mask"]):
            if active:
                by_class[label].add(group)
    counts = [len(items) for items in by_class]
    if min(counts) == 0:
        raise ValueError(f"insufficient_labels: episodios por clase {counts}")
    weights = [min(5.0, max(counts) / value) for value in counts]
    average = sum(weights) / 3
    return torch.tensor([value / average for value in weights], dtype=torch.float32, device=device), counts


def train_classifier(train_samples: list[dict], dev_samples: list[dict], config,
                     category_sizes: list[int], entity_kinds: list[str], progress=None, encoder=None,
                     batch_progress=None):
    """Devuelve mejor red y métricas; exige las tres clases en ambas subdivisiones."""
    if not train_samples or not dev_samples:
        raise ValueError("insufficient_labels: faltan ejemplos de ajuste o dev_tune")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA no disponible")
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    weights, support = class_weights_from_episodes(train_samples, device)
    _, dev_support = class_weights_from_episodes(dev_samples, device)
    feature_dim = len(train_samples[0]["X"][0][0])
    if len(entity_kinds) != len(category_sizes):
        raise ValueError("Faltan tipos de entidad congelados en el preprocesador")
    model = EntityGATLSTM(feature_dim, category_sizes, config, entity_kinds).to(device)
    episode_key = lambda s, i: s["metadata"].get("parent_episode_id", s["metadata"].get("sample_id", str(i)))
    whole_episode_counts = Counter(episode_key(s, i) for i, s in enumerate(sample_summaries(train_samples)))
    supervised_train = sum(sum(s["label_mask"]) for s in sample_summaries(train_samples))
    supervised_dev = sum(sum(s["label_mask"]) for s in sample_summaries(dev_samples))
    if encoder is not None:
        own = model.state_dict()
        transfer = {key: value for key, value in encoder.items() if not key.startswith("classifier.")}
        own.update(transfer)
        model.load_state_dict(own)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_batch_count = math.ceil(len(train_samples) / config.batch_size)
    dev_batch_count = math.ceil(len(dev_samples) / config.batch_size)
    batches_per_epoch = train_batch_count + dev_batch_count
    last_report = 0.0

    def report_batch(epoch, stage, batch):
        nonlocal last_report
        if batch_progress is None:
            return
        total = train_batch_count if stage == "train" else dev_batch_count
        now = monotonic()
        if batch not in (0, total) and now - last_report < 2:
            return
        last_report = now
        done = epoch * batches_per_epoch + batch + (train_batch_count if stage == "dev" else 0)
        batch_progress({"epoch": epoch + 1, "max_epochs": config.max_epochs,
                        "stage": stage, "batch": batch, "batches": total,
                        "completed_batches": done,
                        "total_batches": config.max_epochs * batches_per_epoch})

    best, best_loss, stale, history = None, float("inf"), 0, []
    for epoch in range(config.max_epochs):
        report_batch(epoch, "train", 0)
        model.train()
        order = torch.randperm(len(train_samples)).tolist()
        train_total, train_batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = [train_samples[i] for i in order[start:start + config.batch_size]]
            x, states, present, y, mask, adjacency = _tensor_batch(batch, device)
            if not mask.any():
                report_batch(epoch, "train", start // config.batch_size + 1)
                continue
            episode = _episode_weights(batch, device, whole_episode_counts)
            optimizer.zero_grad()
            logits = model(x, states, adjacency, present)
            loss = masked_classification_loss(logits, y, mask, weights, episode)
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida supervisada no finita")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += loss.item(); train_batches += 1
            report_batch(epoch, "train", start // config.batch_size + 1)
        if not train_batches:
            raise ValueError("insufficient_labels: época sin objetivos")
        model.eval()
        dev_records = []
        report_batch(epoch, "dev", 0)
        with torch.no_grad():
            dev_total, dev_batches = 0.0, 0
            for start in range(0, len(dev_samples), config.batch_size):
                batch = dev_samples[start:start + config.batch_size]
                x, states, present, y, mask, adjacency = _tensor_batch(batch, device)
                if not mask.any():
                    report_batch(epoch, "dev", start // config.batch_size + 1)
                    continue
                logits = model(x, states, adjacency, present)
                loss = masked_classification_loss(logits, y, mask, weights,
                                                  _episode_weights(batch, device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Pérdida de dev_tune no finita")
                dev_total += loss.item(); dev_batches += 1
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
                labels = y.cpu().tolist(); masks = mask.cpu().tolist()
                for bi, sample in enumerate(batch):
                    for ni, active in enumerate(masks[bi]):
                        if active:
                            dev_records.append({"true_label": labels[bi][ni], "scores": probabilities[bi][ni],
                                "episode_id": sample["metadata"].get("parent_episode_id"),
                                "evidence_origin": sample["metadata"].get("evidence_origin", "unspecified")})
                report_batch(epoch, "dev", start // config.batch_size + 1)
        if not dev_batches:
            raise ValueError("insufficient_labels: dev_tune vacío")
        dev_loss = dev_total / dev_batches
        row = {"epoch": epoch + 1, "train_loss": train_total / train_batches, "dev_loss": dev_loss,
               "supervised_train": supervised_train,
               "supervised_dev": supervised_dev}
        history.append(row)
        if progress:
            progress(row)
        if dev_loss < best_loss:
            best_loss, best, stale = dev_loss, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best)
    model.eval()
    policy = choose_rejection_policy(dev_records)
    return model, {"history": history, "best_dev_loss": best_loss,
                   "train_episode_support": support, "dev_episode_support": dev_support,
                   "rejection_policy": policy}


def save_candidate(model, preprocessor: dict, graph: dict, config, directory: Path, metrics: dict):
    """Publica primero pesos y metadatos; manifiesto de modelo al final."""
    directory = Path(directory)
    identifier = uuid4().hex
    for folder in ("weights", "preprocessors", "graphs", "models"):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    weights = directory / "weights" / f"{identifier}.pt"
    temporary = weights.with_suffix(".pt.tmp")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    os.replace(temporary, weights)
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    for folder, value in (("preprocessors", preprocessor), ("graphs", graph)):
        path = directory / folder / f"{identifier}.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)
    manifest = {"model_id": identifier, "method": "gat_lstm", "schema_version": config.schema_version,
                "created_at": datetime.now(timezone.utc).isoformat(), "weights_sha256": sha,
                "weights_file": weights.name, "config": config.as_dict(), "metrics": metrics,
                "rejection_policy": metrics.get("rejection_policy", {"threshold": .4, "margin": 0.0,
                                                                         "validated": False}),
                "status": "candidate", "validated": False, "active": False}
    path = directory / "models" / f"{identifier}.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)
    return manifest


def load_candidate(identifier: str, directory: Path):
    if len(identifier) != 32 or any(char not in "0123456789abcdef" for char in identifier):
        raise ValueError("ID de modelo inválido")
    directory = Path(directory)
    manifest = json.loads((directory / "models" / f"{identifier}.json").read_text(encoding="utf-8"))
    if manifest.get("method") != "gat_lstm" or manifest["model_id"] != identifier:
        raise ValueError("Manifiesto incompatible")
    weights = directory / "weights" / f"{identifier}.pt"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Hash de pesos incompatible")
    from .config import Config
    config = Config(**manifest["config"])
    preprocessor = json.loads((directory / "preprocessors" / f"{identifier}.json").read_text(encoding="utf-8"))
    graph = json.loads((directory / "graphs" / f"{identifier}.json").read_text(encoding="utf-8"))
    category_sizes = [max(preprocessor["entities"][e].get("vocab", {}).values(), default=2) + 1
                      for e in preprocessor["entity_ids"]]
    kinds = [preprocessor["entities"][e]["kind"] for e in preprocessor["entity_ids"]]
    model = EntityGATLSTM(len(preprocessor["feature_names"]), category_sizes, config, kinds)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model, preprocessor, graph, manifest
