"""Inferencia causal GAT-LSTM con artefactos verificados y política congelada."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import LABELS
from .temporal import build_causal_sample
from .evaluation import decide


@lru_cache(maxsize=8)
def load_snapshot(directory: str, model_id: str):
    from .training import load_candidate
    return load_candidate(model_id, Path(directory))


def predict_at(snapshot, event_provider, entity_id, timestamp):
    """Un snapshot cargado una vez por ejecución; no aprende ni recalcula el grafo."""
    model, preprocessor, graph, manifest = snapshot
    from .config import Config
    config = Config(**manifest["config"])
    base = {"entity_id": entity_id, "timestamp": timestamp, "method": "gat_lstm",
            "compatibility": None, "fault_type": None, "model_version": manifest["model_id"],
            "preprocessor_version": preprocessor.get("schema_version"),
            "graph_version": graph.get("graph_version"),
            "confidence_kind": "uncalibrated_softmax", "evaluation_scope": "causal_runtime"}

    def reject(reason, coverage=None):
        return {**base, "status": "indeterminate", "predicted_label": None,
                "confidence": 0.0, "class_scores": {}, "needs_review": True,
                "reasons": [reason], "input_coverage": coverage or {}}

    if entity_id not in preprocessor["entity_ids"]:
        return reject("unsupported_entity")
    sample = build_causal_sample(event_provider, timestamp, preprocessor, graph, config)
    index = preprocessor["entity_ids"].index(entity_id)
    present = sum(row[index] for row in sample["node_present"])
    coverage = {"present_steps": present, "required_steps": config.sequence_steps,
                "neighbor_ids": [preprocessor["entity_ids"][j] for j, allowed in
                                 enumerate(graph["adjacency"][index]) if allowed and j != index]}
    if present == 0:
        return reject("context_not_recoverable", coverage)
    import torch
    from .training import _tensor_batch
    with torch.no_grad():
        x, state_ids, node_present, _, _, adjacency = _tensor_batch([sample], "cpu")
        scores = torch.softmax(model(x, state_ids, adjacency, node_present)[0, index], dim=-1).tolist()
    policy = manifest.get("rejection_policy", {"threshold": .4, "margin": 0.0})
    choice = decide(scores, policy["threshold"], policy["margin"])
    label = LABELS[choice] if choice is not None else None
    return {**base, "status": ("indeterminate" if choice is None else
                                "compatible" if choice == 2 else "no_fault_evidence"),
            "predicted_label": label, "confidence": max(scores),
            "class_scores": dict(zip(LABELS, scores)), "needs_review": choice is None,
            "reasons": (["score_below_rejection_policy"] if choice is None else []),
            "input_coverage": coverage}
