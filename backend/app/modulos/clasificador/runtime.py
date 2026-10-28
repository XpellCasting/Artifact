"""Inferencia causal GAT-LSTM con artefactos verificados y política congelada."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import perf_counter

from app.modulos.clasificador import LABELS
from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.temporal import CausalInputError
from app.modulos.clasificador.temporal import build_causal_sample
from app.modulos.clasificador.evaluation import decide


@lru_cache(maxsize=8)
def load_snapshot(directory: str, model_id: str):
    from app.modulos.clasificador.training import load_candidate
    return load_candidate(model_id, Path(directory))


def predict_at(snapshot, event_provider, entity_id, timestamp, sensor_profiles=None, metrics=None,
               prediction_cache=None, sensor_kind=None):
    """Pesos congelados; calibra entradas nuevas con el pasado disponible."""
    model, preprocessor, graph, manifest = snapshot
    from app.modulos.clasificador.config import Config
    config = Config(**manifest["config"])
    base = {"entity_id": entity_id, "timestamp": timestamp, "method": "gat_lstm",
            "compatibility": None, "fault_type": None, "model_version": manifest["model_id"],
            "preprocessor_version": preprocessor.get("schema_version"),
            "graph_version": graph.get("graph_version"),
            "confidence_kind": "uncalibrated_softmax", "evaluation_scope": "causal_runtime",
            "fault_type_status": "not_supported"}

    def reject(reason, coverage=None):
        return {**base, "status": "indeterminate", "predicted_label": None,
                "confidence": 0.0, "class_scores": {}, "needs_review": True,
                "reasons": [reason], "input_coverage": coverage or {}}

    newly_onboarded = entity_id not in preprocessor["entity_ids"]
    if newly_onboarded:
        profile = (sensor_profiles or {}).get(entity_id)
        if not profile or profile.get("status") != "available":
            from app.modulos.clasificador.profiles import profile_from_past
            try:
                profile = profile_from_past(event_provider, entity_id, parse_utc(timestamp), config,
                    {info["kind"] for info in preprocessor["entities"].values()}, sensor_kind)
            except CausalInputError as exc:
                return reject(exc.reason)
            if sensor_profiles is not None:
                sensor_profiles[entity_id] = profile
        if (parse_utc(timestamp) <= parse_utc(profile["reference_end"])
                or (profile.get("available_from")
                    and parse_utc(timestamp) < parse_utc(profile["available_from"]))):
            return reject("reference_not_prior_to_target")
        base["onboarding"] = {"reference_start": profile.get("reference_start"),
            "reference_end": profile["reference_end"], "reference_rows": profile.get("reference_rows"),
            "available_from": profile.get("available_from"),
            "policy": profile.get("onboarding_policy", "reviewed_normal_reference_v1"),
            "reference_origin": profile.get("reference_origin", "reviewed_normal_interval"),
            "normality_verified": profile.get("normality_verified", True),
            "sensor_kind": profile.get("sensor_kind", profile["entity"]["kind"]),
            "input_kind": profile["entity"]["kind"], "compatible_neighbors": 0}
        info = profile["entity"]
        preprocessor = {**preprocessor, "entity_ids": [entity_id], "entities": {entity_id: info},
                        "entity_kinds": {entity_id: info["kind"]}}
        graph = {**graph, "entity_ids": [entity_id], "adjacency": [[True]],
                 "onboarding_policy": "within_entity_segments_v1"}
        graph["graph_version"] = hashlib.sha256(json.dumps({"graph": graph, "profile": profile}, sort_keys=True).encode()).hexdigest()
        base["graph_version"] = graph["graph_version"]
    cache_key = (manifest["model_id"], str(timestamp), tuple(preprocessor["entity_ids"]),
                 graph.get("graph_version"), tuple(tuple(row) for row in graph["adjacency"]))
    cache_entry = prediction_cache.get(cache_key) if prediction_cache is not None else None
    if cache_entry is None:
        sample_started = perf_counter()
        try:
            sample = build_causal_sample(event_provider, timestamp, preprocessor, graph, config)
        except CausalInputError as exc:
            if metrics is not None:
                metrics["gat_sample_ms"] = metrics.get("gat_sample_ms", 0.0) + (perf_counter() - sample_started) * 1000
                metrics["gat_sample_rejections"] = metrics.get("gat_sample_rejections", 0) + 1
            return reject(exc.reason)
        except (ValueError, KeyError, TypeError, OverflowError):
            if metrics is not None:
                metrics["gat_sample_ms"] = metrics.get("gat_sample_ms", 0.0) + (perf_counter() - sample_started) * 1000
                metrics["gat_sample_rejections"] = metrics.get("gat_sample_rejections", 0) + 1
            return reject("invalid_event_format")
        if metrics is not None:
            metrics["gat_sample_ms"] = metrics.get("gat_sample_ms", 0.0) + (perf_counter() - sample_started) * 1000
            metrics["gat_samples"] = metrics.get("gat_samples", 0) + 1
        cache_entry = {"sample": sample, "scores_by_node": None, "type_scores_by_node": None}
        if prediction_cache is not None:
            prediction_cache[cache_key] = cache_entry
    else:
        sample = cache_entry["sample"]
        if metrics is not None:
            metrics["gat_sample_cache_hits"] = metrics.get("gat_sample_cache_hits", 0) + 1
    index = preprocessor["entity_ids"].index(entity_id)
    present = sum(row[index] for row in sample["node_present"])
    valid_feature = preprocessor["feature_names"].index("known_fraction")
    valid_steps = sum(bool(row[index][valid_feature]) for row in sample["X"])
    required_steps = len(sample["X"])
    coverage = {"present_steps": present, "required_steps": required_steps,
                "valid_value_steps": valid_steps,
                "valid_value_coverage": valid_steps / required_steps,
                "neighbor_ids": [preprocessor["entity_ids"][j] for j, allowed in
                                 enumerate(graph["adjacency"][index]) if allowed and j != index]}
    coverage["scales"] = sample["metadata"]["scale_coverage"][entity_id]
    coverage["graph_method"] = "within_entity_segment_dtw"
    if not any(s["usable"] for s in coverage["scales"]):
        return reject("insufficient_scale_coverage", coverage)
    if present == 0:
        return reject("insufficient_data", coverage)
    from app.modulos.clasificador.model_constants import FAULT_TYPES
    fault_types = FAULT_TYPES
    if cache_entry["scores_by_node"] is None:
        inference_started = perf_counter()
        import torch
        from app.modulos.clasificador.training import _tensor_batch
        with torch.no_grad():
            x, state_ids, node_present, _, _, adjacency = _tensor_batch([sample], "cpu")
            from app.modulos.clasificador.model_constants import KIND_TO_ID
            kinds = torch.tensor([KIND_TO_ID[preprocessor["entities"][key]["kind"]]
                                  for key in preprocessor["entity_ids"]], dtype=torch.long)
            class_logits, type_logits = model(x, state_ids, adjacency, node_present, kinds)
            scores_by_node = torch.softmax(class_logits[0], dim=-1).tolist()
            type_scores_by_node = torch.softmax(type_logits[0], dim=-1).tolist()
        cache_entry["scores_by_node"] = scores_by_node
        cache_entry["type_scores_by_node"] = type_scores_by_node
        if metrics is not None:
            metrics["gat_inference_ms"] = metrics.get("gat_inference_ms", 0.0) + (perf_counter() - inference_started) * 1000
            metrics["gat_inferences"] = metrics.get("gat_inferences", 0) + 1
    elif metrics is not None:
        metrics["gat_inference_cache_hits"] = metrics.get("gat_inference_cache_hits", 0) + 1
    scores = cache_entry["scores_by_node"][index]
    type_scores = (cache_entry["type_scores_by_node"][index]
                   if cache_entry["type_scores_by_node"] is not None else None)
    policy = manifest.get("rejection_policy", {"threshold": .4, "margin": 0.0})
    choice = decide(scores, policy["threshold"], policy["margin"])
    label = LABELS[choice] if choice is not None else None
    status = ("indeterminate" if choice is None else "compatible" if choice == 2
              else "environmental_anomaly" if choice == 1 else "no_fault_evidence")
    type_index = max(range(len(type_scores)), key=type_scores.__getitem__) if type_scores else None
    type_confidence = type_scores[type_index] if type_index is not None else None
    metrics = manifest.get("metrics", {})
    type_name = fault_types[type_index] if type_index is not None else None
    type_validation = metrics.get("fault_type_validation", {})
    type_identified = (type_name is not None
        and metrics.get("fault_type_train_episode_support", {}).get(type_name, 0) >= 3
        and type_validation.get("independent_episodes_by_type", {}).get(type_name, 0) >= 3
        and (type_validation.get("recall_by_type", {}).get(type_name) or 0) >= .5)
    reasons = ["score_below_rejection_policy"] if choice is None else []
    return {**base, "status": status,
            "fault_type_status": ("identified" if type_identified else "unknown") if choice == 2 else "not_applicable",
            "fault_type": type_name if type_identified else None,
            "fault_type_confidence": type_confidence,
            "predicted_label": label, "confidence": max(scores),
            "class_scores": dict(zip(LABELS, scores)), "needs_review": choice is None,
            "input_provenance": sample["metadata"].get("input_provenance", []),
            "reasons": reasons,
            "input_coverage": coverage}
