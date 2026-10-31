"""Métricas de tres clases con abstenciones y denominadores explícitos."""
from __future__ import annotations

from collections import Counter

from app.modulos.clasificador import LABELS


THRESHOLDS = (.4, .5, .6, .7, .8, .9)
MARGINS = (0.0, .05, .1, .2)


def decide(scores, threshold: float, margin: float):
    if len(scores) != 3:
        raise ValueError("Se esperan tres puntuaciones")
    ranked = sorted(range(3), key=lambda i: (-scores[i], i))
    return ranked[0] if scores[ranked[0]] >= threshold and (
        scores[ranked[0]] - scores[ranked[1]]) >= margin else None


def metrics(records: list[dict], threshold=.4, margin=0.0) -> dict:
    """records: true_label 0..2, scores[3], episode_id, evidence_origin."""
    matrix = [[0] * 4 for _ in range(3)]
    origins = Counter()
    episodes = [set() for _ in range(3)]
    for row in records:
        truth = row["true_label"]
        if truth not in range(3):
            raise ValueError("Etiqueta real inválida")
        frozen = row.get("rejection_policy", {})
        prediction = decide(row["scores"], frozen.get("threshold", threshold), frozen.get("margin", margin))
        matrix[truth][prediction if prediction is not None else 3] += 1
        episodes[truth].add(row.get("episode_id", row.get("sample_id")))
        origins[row.get("evidence_origin", "unspecified")] += 1
    total = len(records)
    accepted = total - sum(row[3] for row in matrix)
    accuracy = sum(matrix[index][index] for index in range(3)) / total if total else None
    per_class = {}
    for index, name in enumerate(LABELS):
        support = sum(matrix[index])
        tp = matrix[index][index]
        fp = sum(matrix[k][index] for k in range(3) if k != index)
        fn = support - tp  # includes abstentions
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / support if support else None
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else None
        per_class[name] = {"support": support, "episodes": len(episodes[index]),
                           "tp": tp, "fp": fp, "fn": fn,
                           "precision": precision, "recall": recall, "f1": f1}
    f1_values = [entry["f1"] for entry in per_class.values()]
    return {"matrix": matrix, "columns": [*LABELS, "abstain"], "support": total,
            "accuracy": accuracy,
            "accepted": accepted, "coverage": accepted / total if total else None,
            "macro_f1": sum(f1_values) / 3 if all(v is not None for v in f1_values) else None,
            "per_class": per_class, "evidence_origins": dict(origins),
            "false_alarms_normal": sum(matrix[0][1:3]),
            "false_alarms_environmental": matrix[1][2],
            "complete": bool(total) and all(sum(row) > 0 for row in matrix)}


def choose_rejection_policy(dev_records: list[dict]):
    if not dev_records or any(not any(r["true_label"] == i for r in dev_records) for i in range(3)):
        return {"threshold": .4, "margin": 0.0, "validated": False}
    candidates = []
    for threshold in THRESHOLDS:
        for margin in MARGINS:
            result = metrics(dev_records, threshold, margin)
            candidates.append((result["macro_f1"] if result["macro_f1"] is not None else -1,
                               result["coverage"], -threshold, -margin, threshold, margin))
    _, _, _, _, threshold, margin = max(candidates)
    return {"threshold": threshold, "margin": margin, "validated": True}
