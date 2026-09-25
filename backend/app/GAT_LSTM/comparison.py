"""Comparación justa de predicciones sobre un conjunto fijo de objetivos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from . import LABELS
from .evaluation import decide, metrics
from .pipeline import load_partition


def compare_prediction_files(dataset: dict, partition: str, variants: dict[str, Path]) -> dict:
    """Faltantes estructurales se cuentan como abstenciones, nunca se filtran."""
    samples = load_partition(dataset, partition)
    targets = []
    for sample in samples:
        meta = sample["metadata"]
        for node, active in enumerate(sample["label_mask"]):
            if not active:
                continue
            targets.append({"sample_id": meta["sample_id"], "entity_id": meta["entity_ids"][node],
                "true_label": sample["y"][node], "episode_id": meta.get("parent_episode_id") or meta["sample_id"],
                "evidence_origin": meta.get("evidence_origin", "unspecified")})
    target_fingerprint = hashlib.sha256(json.dumps(sorted(
        (row["sample_id"], row["entity_id"], row["true_label"]) for row in targets),
        separators=(",", ":")).encode()).hexdigest()
    results = {}
    for name, path in variants.items():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        predictions = {(row.get("sample_id"), row.get("entity_id")): row
                       for row in payload.get("predictions", [])}
        threshold, margin = payload.get("threshold", .4), payload.get("margin", 0.0)
        records = []
        missing = 0
        for target in targets:
            prediction = predictions.get((target["sample_id"], target["entity_id"]))
            if prediction is None or not isinstance(prediction.get("scores"), list):
                scores = [0.0, 0.0, 0.0]
                missing += 1
            else:
                scores = prediction["scores"]
            records.append({**target, "scores": scores})
        result = metrics(records, threshold, margin)
        accepted_correct = sum(result["matrix"][i][i] for i in range(len(LABELS)))
        accepted_records = [row for row in records if decide(row["scores"], threshold, margin) is not None]
        accepted_metrics = metrics(accepted_records, 0.0, 0.0) if accepted_records else None
        origins = {}
        for origin in sorted({row["evidence_origin"] for row in records}):
            origin_records = [row for row in records if row["evidence_origin"] == origin]
            origins[origin] = metrics(origin_records, threshold, margin)
        episodes = {}
        for row in records:
            episodes.setdefault(row["episode_id"], []).append(row)
        bootstrap = {"replicates": 500, "unit": "parent_episode_id", "sensor_fault_recall_95": None,
                     "macro_f1_95": None, "reason": "fewer_than_two_episodes"}
        if len(episodes) >= 2:
            keys, rng, recalls, macro = list(episodes), random.Random(42), [], []
            for _ in range(bootstrap["replicates"]):
                draw = [key for key in (rng.choice(keys) for _ in keys)]
                replicate = [row for key in draw for row in episodes[key]]
                values = metrics(replicate, threshold, margin)
                recall = values["per_class"]["sensor_fault"]["recall"]
                if recall is not None: recalls.append(recall)
                if values["macro_f1"] is not None: macro.append(values["macro_f1"])
            percentile = lambda values, p: sorted(values)[round((len(values) - 1) * p)] if values else None
            bootstrap = {"replicates": 500, "unit": "parent_episode_id",
                "sensor_fault_recall_95": [percentile(recalls, .025), percentile(recalls, .975)],
                "macro_f1_95": [percentile(macro, .025), percentile(macro, .975)],
                "reason": None if recalls and macro else "some_resamples_lacked_class_support"}
        result.update({"accepted_accuracy": accepted_correct / result["accepted"] if result["accepted"] else None,
                       "accepted_metrics": accepted_metrics, "by_evidence_origin": origins,
                       "episode_bootstrap": bootstrap,
                       "structural_abstentions": missing, "target_count": len(targets),
                       "prediction_count": len(targets) - missing, "threshold": threshold,
                       "margin": margin, "scope": "classifier_windows_only",
                       "evidence_origin_support": {origin: sum(row["evidence_origin"] == origin for row in targets)
                                                   for origin in sorted({row["evidence_origin"] for row in targets})}})
        results[name] = result
    return {"schema_version": "gat-lstm-comparison-v1", "dataset_id": dataset["dataset_id"],
            "partition": partition, "target_count": len(targets),
            "target_fingerprint_sha256": target_fingerprint,
            "class_support": {label: sum(row["true_label"] == index for row in targets)
                               for index, label in enumerate(LABELS)},
            "episode_support": len({row["episode_id"] for row in targets}),
            "variants": results}
