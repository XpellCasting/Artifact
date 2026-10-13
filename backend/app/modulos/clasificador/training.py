"""Ajuste supervisado de ventanas; no activa candidatos automáticamente."""
from __future__ import annotations
from app.config import setting

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from time import monotonic
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F

from app.modulos.clasificador import LABELS
from app.modulos.clasificador.evaluation import choose_rejection_policy
from app.modulos.clasificador.storage import iter_partition_batches
from app.modulos.clasificador.storage import sample_summaries
from app.modulos.clasificador.checkpoints import (TrainingRun, cpu_state, provenance,
    random_state, restore_random_state, sample_identity)


def resolve_device(requested):
    """CUDA explícito falla con un diagnóstico útil; auto permite CPU."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return torch.device("cpu")
    if not isinstance(requested, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", requested):
        raise ValueError("Dispositivo inválido; usa auto, cpu, cuda o cuda:N")
    if not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible en PyTorch. Revisa el controlador NVIDIA y la instalación de PyTorch con CUDA; ejecuta scripts/gat_lstm_check.py --device cuda")
    index = int(requested.split(":")[1]) if ":" in requested else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise ValueError(f"GPU cuda:{index} inexistente; hay {torch.cuda.device_count()} dispositivos CUDA")
    return torch.device(f"cuda:{index}")


def _tensor_batch(samples, device):
    def tensor(field, dtype):
        values = samples[field] if isinstance(samples, dict) else np.stack([s[field] for s in samples])
        return torch.as_tensor(values, dtype=dtype, device=device)
    x = tensor("X", torch.float32)
    states = tensor("state_ids", torch.long)
    present = tensor("node_present", torch.bool)
    y = tensor("y", torch.long)
    mask = tensor("label_mask", torch.bool)
    adjacency_values = samples["adjacency"] if isinstance(samples, dict) else samples[0]["adjacency"]
    adjacency = torch.tensor(adjacency_values, dtype=torch.bool, device=device)
    return x, states, present, y, mask, adjacency


def _episode_weights(samples, device, full_episode_counts=None):
    metadata = samples["metadata"] if isinstance(samples, dict) else [s["metadata"] for s in samples]
    key = lambda m, i: m.get("parent_episode_id", m.get("sample_id", str(i)))
    episodes = full_episode_counts or Counter(key(meta, i) for i, meta in enumerate(metadata))
    return torch.tensor([1 / episodes[key(meta, i)] for i, meta in enumerate(metadata)],
                        dtype=torch.float32, device=device)


def _batch_groups(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def _batches(samples, groups, batch_size):
    if hasattr(samples, "read_batch"):
        yield from iter_partition_batches(samples, groups,
            batch_bytes=samples.batch_nbytes() * batch_size)
    else:
        for indices in groups:
            started = monotonic()
            yield [samples[index] for index in indices], monotonic() - started


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


def train_classifier(train_samples, dev_samples, config, category_sizes, entity_kinds,
                     progress=None, encoder=None, batch_progress=None,
                     checkpoint_dir=None, resume=False, checkpoint_identity=None):
    if config.architecture_version not in {"gat_lstm_v3_segments_multiscale"} or encoder is not None:
        raise ValueError("Solo se admite GAT-LSTM por segmentos sin encoder predictivo externo")
    from dataclasses import replace
    device = resolve_device(config.device)
    environment = provenance()
    if resume and checkpoint_dir is None:
        raise ValueError("Reanudar requiere un directorio de checkpoints")
    identity = {**(checkpoint_identity if checkpoint_identity is not None
                   else sample_identity(train_samples, dev_samples)),
                "entity_kinds": list(entity_kinds)} if checkpoint_dir is not None else None
    run = TrainingRun(checkpoint_dir, config, identity,
                      resume, environment) if checkpoint_dir is not None else None
    try:
        model, scores = _train_trials(train_samples, dev_samples, replace(config, device=str(device)),
                                     entity_kinds, progress, batch_progress, run, environment)
        scores["runtime_parameters"]["requested_device"] = config.device
        return model, scores
    finally:
        if run is not None:
            run.close()


def _train_trials(train_samples, dev_samples, config, entity_kinds, progress,
                  batch_progress, run, environment):
    from dataclasses import replace
    trials = []
    best = None
    for trial, threshold in enumerate((.3, .5, .7) if config.segment_encoder == "gat" else (.5,)):
        notify = (lambda row: progress({**row, "threshold_trial": trial+1, "graph_threshold": threshold})) if progress else None
        notify_batch = (lambda row: batch_progress({**row, "threshold_trial": trial+1, "graph_threshold": threshold})) if batch_progress else None
        candidate, scores = _train_segment_classifier(train_samples, dev_samples,
            replace(config, segment_threshold=threshold), entity_kinds, notify, notify_batch,
            run=run, trial=trial+1)
        rank = (scores.get("dev_macro_f1") or 0., threshold)
        trials.append({"threshold": threshold, "macro_f1": rank[0]})
        if best is None or rank > best[0]:
            best = (rank, candidate, scores)
    _, model, scores = best
    scores["segment_threshold_search"] = trials
    scores["selected_segment_threshold"] = model.threshold
    scores["provenance"] = environment
    scores["checkpoint"] = {"enabled": run is not None, "resumed": bool(run and run.resume),
                            "interval": "epoch", "threshold_trials": len(trials)}
    return model, scores



def _train_segment_classifier(train_samples, dev_samples, config, entity_kinds,
                                  progress=None, batch_progress=None, run=None, trial=1):
    from app.modulos.clasificador.model_constants import FAULT_TYPES

    if not train_samples or not dev_samples:
        raise ValueError("insufficient_labels: faltan ejemplos de ajuste o dev_tune")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    weights, support = class_weights_from_episodes(train_samples, device)
    _, dev_support = class_weights_from_episodes(dev_samples, device)
    feature_dim = len(train_samples[0]["X"][0][0])
    if len(entity_kinds) != len(train_samples[0]["X"][0]):
        raise ValueError("Tipos de sensor incompletos para GAT-LSTM por segmentos")
    from app.modulos.clasificador.network_segments import SegmentGATLSTM
    model = SegmentGATLSTM(feature_dim, entity_kinds, config).to(device)
    episode_key = lambda meta, i: meta.get("parent_episode_id", meta.get("sample_id", str(i)))
    episode_counts = Counter(episode_key(row["metadata"], i)
                             for i, row in enumerate(sample_summaries(train_samples)))
    supervised_train = sum(sum(row["label_mask"]) for row in sample_summaries(train_samples))
    supervised_dev = sum(sum(row["label_mask"]) for row in sample_summaries(dev_samples))
    type_episodes = {fault: set() for fault in FAULT_TYPES}
    for index, summary in enumerate(sample_summaries(train_samples)):
        metadata = summary["metadata"]
        fault = metadata.get("fault_subtype")
        if metadata.get("target_label") in (LABELS.index("sensor_fault"), "sensor_fault") and fault in type_episodes:
            type_episodes[fault].add(episode_key(metadata, index))
    type_support = {fault: len(groups) for fault, groups in type_episodes.items()}
    supported_types = [fault for fault in FAULT_TYPES if type_support[fault] > 0]
    if not supported_types:
        raise ValueError("insufficient_fault_types: Desarrollo no contiene fallas con tipo conocido")
    type_weights = torch.tensor([
        min(5.0, max(type_support.values()) / max(1, type_support[fault]))
        if type_support[fault] else 0.0 for fault in FAULT_TYPES], dtype=torch.float32, device=device)
    present_weights = type_weights[type_weights > 0]
    type_weights = type_weights / present_weights.mean()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_batch_count = math.ceil(len(train_samples) / config.batch_size)
    dev_batch_count = math.ceil(len(dev_samples) / config.batch_size)
    batches_per_epoch = train_batch_count + dev_batch_count
    runtime = {"epoch_durations_seconds": [], "batches_per_second": [], "data_wait_seconds": 0.0}
    last_report = 0.0

    def report_batch(epoch, stage, batch, started, waiting):
        nonlocal last_report
        if batch_progress is None:
            return
        total = train_batch_count if stage == "train" else dev_batch_count
        now = monotonic()
        if batch not in (0, total) and now - last_report < 2:
            return
        last_report = now
        batch_progress({"epoch": epoch + 1, "max_epochs": config.max_epochs, "stage": stage,
            "batch": batch, "batches": total,
            "completed_batches": epoch * batches_per_epoch + batch + (train_batch_count if stage == "dev" else 0),
            "total_batches": config.max_epochs * batches_per_epoch,
            "source": getattr(train_samples, "source", "original"),
            "batches_per_second": (batch + (train_batch_count if stage == "dev" else 0)) /
                                  max(monotonic() - started, 1e-9),
            "data_wait_seconds": runtime["data_wait_seconds"],
            "epoch_data_wait_seconds": waiting})

    best, best_loss, stale, history = None, float("inf"), 0, []
    best_dev_records, best_type_records = [], []
    next_epoch = 0
    checkpoint = run.load(trial) if run is not None else None
    if checkpoint is not None:
        next_epoch = checkpoint["next_epoch"]
        if config.max_epochs < next_epoch:
            raise ValueError("max_epochs no puede ser menor que las épocas ya completadas")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best, best_loss, stale = checkpoint["best_model"], checkpoint["best_loss"], checkpoint["stale"]
        history, runtime = checkpoint["history"], checkpoint["runtime"]
        best_dev_records, best_type_records = checkpoint["best_dev_records"], checkpoint["best_type_records"]
        restore_random_state(checkpoint["random_state"], device)
    for epoch in range(next_epoch, config.max_epochs) if stale < config.patience else ():
        epoch_started, epoch_waiting = monotonic(), 0.0
        model.train()
        train_total = train_count = type_total = type_count = 0
        order = torch.randperm(len(train_samples)).tolist()
        groups = _batch_groups(order, config.batch_size)
        for batch_index, (batch, waited) in enumerate(_batches(train_samples, groups, config.batch_size)):
            epoch_waiting += waited; runtime["data_wait_seconds"] += waited
            x, states, present, y, mask, adjacency = _tensor_batch(batch, device)
            metadata = batch["metadata"] if isinstance(batch, dict) else [row["metadata"] for row in batch]
            fault_y = torch.full_like(y, -1)
            fault_mask = torch.zeros_like(mask)
            for row_index, meta in enumerate(metadata):
                fault = meta.get("fault_subtype")
                if meta.get("target_label") in (LABELS.index("sensor_fault"), "sensor_fault") and fault in FAULT_TYPES:
                    node = meta.get("entity_ids", []).index(meta.get("next_entity", "")) if meta.get("next_entity") in meta.get("entity_ids", []) else None
                    if node is None:
                        active_nodes = torch.nonzero(mask[row_index], as_tuple=False).flatten()
                        node = int(active_nodes[0]) if len(active_nodes) else None
                    if node is not None:
                        fault_y[row_index, node] = FAULT_TYPES.index(fault)
                        fault_mask[row_index, node] = True
            if not mask.any():
                report_batch(epoch, "train", batch_index + 1, epoch_started, epoch_waiting)
                continue
            kind_ids = model.kind_ids.to(device)
            logits, type_logits = model(x, states, adjacency, present, kind_ids)
            episode = _episode_weights(batch, device, episode_counts)
            class_loss = masked_classification_loss(logits, y, mask, weights, episode)
            if fault_mask.any():
                type_loss_all = F.cross_entropy(type_logits.reshape(-1, len(FAULT_TYPES)),
                    fault_y.clamp(min=0).reshape(-1), weight=type_weights, reduction="none").view_as(fault_y)
                weighted_mask = fault_mask * episode.view(-1, 1)
                type_loss = (type_loss_all * weighted_mask).sum() / weighted_mask.sum().clamp(min=1)
                loss = class_loss + type_loss
                type_total += type_loss.item(); type_count += 1
            else:
                loss = class_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida compartida GAT-LSTM no finita")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_total += class_loss.item(); train_count += 1
            report_batch(epoch, "train", batch_index + 1, epoch_started, epoch_waiting)
        if not train_count:
            raise ValueError("insufficient_labels: época sin objetivos")

        model.eval(); dev_records = []; type_records = []
        dev_total = dev_count = 0
        dev_groups = _batch_groups(range(len(dev_samples)), config.batch_size)
        with torch.no_grad():
            for batch_index, (batch, waited) in enumerate(_batches(dev_samples, dev_groups, config.batch_size)):
                epoch_waiting += waited; runtime["data_wait_seconds"] += waited
                x, states, present, y, mask, adjacency = _tensor_batch(batch, device)
                metadata = batch["metadata"] if isinstance(batch, dict) else [row["metadata"] for row in batch]
                fault_y = torch.full_like(y, -1); fault_mask = torch.zeros_like(mask)
                for row_index, meta in enumerate(metadata):
                    fault = meta.get("fault_subtype")
                    if meta.get("target_label") in (LABELS.index("sensor_fault"), "sensor_fault") and fault in FAULT_TYPES:
                        active_nodes = torch.nonzero(mask[row_index], as_tuple=False).flatten()
                        if len(active_nodes):
                            fault_y[row_index, int(active_nodes[0])] = FAULT_TYPES.index(fault)
                            fault_mask[row_index, int(active_nodes[0])] = True
                if not mask.any():
                    report_batch(epoch, "dev", batch_index + 1, epoch_started, epoch_waiting)
                    continue
                logits, type_logits = model(x, states, adjacency, present, model.kind_ids.to(device))
                class_loss = masked_classification_loss(logits, y, mask, weights)
                dev_total += class_loss.item(); dev_count += 1
                class_probs = torch.softmax(logits, dim=-1).cpu().tolist()
                type_probs = torch.softmax(type_logits, dim=-1).cpu().tolist()
                labels = y.cpu().tolist(); masks = mask.cpu().tolist()
                for row_index, meta in enumerate(metadata):
                    for node, active in enumerate(masks[row_index]):
                        if active:
                            dev_records.append({"true_label": labels[row_index][node],
                                "scores": class_probs[row_index][node],
                                "episode_id": meta.get("parent_episode_id"),
                                "evidence_origin": meta.get("evidence_origin", "unspecified")})
                    if fault_mask[row_index].any():
                        node = int(torch.nonzero(fault_mask[row_index], as_tuple=False)[0].item())
                        type_records.append({"true_type": int(fault_y[row_index, node]),
                            "scores": type_probs[row_index][node],
                            "episode_id": meta.get("parent_episode_id")})
                report_batch(epoch, "dev", batch_index + 1, epoch_started, epoch_waiting)
        if not dev_count:
            raise ValueError("insufficient_labels: dev_tune vacío")
        dev_loss = dev_total / dev_count
        duration = monotonic() - epoch_started
        row = {"epoch": epoch + 1, "train_loss": train_total / train_count,
               "dev_loss": dev_loss, "train_type_loss": type_total / type_count if type_count else None,
               "supervised_train": supervised_train, "supervised_dev": supervised_dev,
               "duration_seconds": duration, "data_wait_seconds": epoch_waiting}
        history.append(row); runtime["epoch_durations_seconds"].append(duration)
        runtime["batches_per_second"].append(batches_per_epoch / max(duration, 1e-9))
        if dev_loss < best_loss:
            best_loss, best, stale = dev_loss, cpu_state(model.state_dict()), 0
            best_dev_records, best_type_records = deepcopy(dev_records), deepcopy(type_records)
        else:
            stale += 1
        if run is not None:
            run.save(trial, {"next_epoch": epoch + 1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_model": best, "best_loss": best_loss,
                "stale": stale, "history": history, "runtime": runtime,
                "best_dev_records": best_dev_records, "best_type_records": best_type_records,
                "random_state": random_state(device), "max_epochs": config.max_epochs})
        if progress: progress(row)
        if stale >= config.patience: break
    model.load_state_dict(best); model.eval()
    type_correct = sum(max(range(len(row["scores"])), key=row["scores"].__getitem__) == row["true_type"]
                       for row in best_type_records)
    type_accuracy = type_correct / len(best_type_records) if best_type_records else None
    type_validation_support = {fault: sum(row["true_type"] == index for row in best_type_records)
                               for index, fault in enumerate(FAULT_TYPES)}
    type_validation_episodes = {fault: len({row["episode_id"] for row in best_type_records
        if row["true_type"] == index}) for index, fault in enumerate(FAULT_TYPES)}
    type_validation_recall = {fault: (sum(max(range(len(row["scores"])), key=row["scores"].__getitem__) == index
        for row in best_type_records if row["true_type"] == index) / type_validation_support[fault]
        if type_validation_support[fault] else None) for index, fault in enumerate(FAULT_TYPES)}
    from app.modulos.clasificador.evaluation import metrics as classification_metrics
    rejection = choose_rejection_policy(best_dev_records)
    return model, {"history": history, "best_dev_loss": best_loss,
        "dev_macro_f1": classification_metrics(best_dev_records, rejection["threshold"], rejection["margin"])["macro_f1"],
        "train_episode_support": support, "dev_episode_support": dev_support,
        "rejection_policy": choose_rejection_policy(best_dev_records),
        "fault_types": list(FAULT_TYPES), "fault_type_train_episode_support": type_support,
        "fault_type_validation": {"support": len(best_type_records), "accuracy": type_accuracy,
            "independent_episodes": len({row["episode_id"] for row in best_type_records}),
            "support_by_type": type_validation_support,
            "independent_episodes_by_type": type_validation_episodes,
            "recall_by_type": type_validation_recall},
        "runtime": runtime,
        "runtime_parameters": {"device": str(device), "requested_device": config.device,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda, "batch_size": config.batch_size,
            "omp_num_threads": setting('OMP_NUM_THREADS'),
            "mkl_num_threads": setting('MKL_NUM_THREADS'),
            "torch_num_threads": torch.get_num_threads()}}


def save_candidate(model, preprocessor: dict, graph: dict, config, directory: Path, metrics: dict):
    """Publica primero pesos y metadatos; manifiesto de modelo al final."""
    if config.architecture_version == "gat_lstm_v3_segments_multiscale":
        from dataclasses import replace
        from importlib.metadata import version
        config = replace(config, segment_threshold=model.threshold)
        graph = {**graph, "threshold": model.threshold}
        graph["graph_version"] = hashlib.sha256(json.dumps({k: v for k, v in graph.items() if k != "graph_version"}, sort_keys=True).encode()).hexdigest()
        metrics = {**metrics, "library_versions": {name: version(name) for name in ("torch", "torch-geometric", "tzdata")}}
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
                "graph_version": graph.get("graph_version"), "window_seconds": config.context_seconds,
                "architecture_version": config.architecture_version,
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
    from app.modulos.clasificador.config import Config
    if manifest.get("config", {}).get("architecture_version") not in {"gat_lstm_v3_segments_multiscale"}:
        raise ValueError("Arquitectura retirada o no declarada")
    config = Config(**manifest["config"])
    architecture = manifest.get("architecture_version", config.architecture_version)
    if architecture != config.architecture_version:
        raise ValueError("La arquitectura declarada en el manifiesto no coincide con su configuración")
    preprocessor = json.loads((directory / "preprocessors" / f"{identifier}.json").read_text(encoding="utf-8"))
    graph = json.loads((directory / "graphs" / f"{identifier}.json").read_text(encoding="utf-8"))
    kinds = [preprocessor["entities"][e]["kind"] for e in preprocessor["entity_ids"]]
    if architecture not in {"gat_lstm_v3_segments_multiscale"}:
        raise ValueError("El modelo requiere GAT-LSTM por segmentos explícito")
    from app.modulos.clasificador.network_segments import SegmentGATLSTM
    model = SegmentGATLSTM(len(preprocessor["feature_names"]), kinds, config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model, preprocessor, graph, manifest
