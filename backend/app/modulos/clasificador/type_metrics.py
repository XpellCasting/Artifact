"""Complete-diagnosis subtype scores: missed faults remain false negatives."""
from app.modulos.clasificador.evaluation import decide
from app.modulos.clasificador.model_constants import FAULT_TYPES


def diagnostic_metrics(records, threshold=.4, margin=0., supported=None):
    counts = [[0, 0, 0] for _ in FAULT_TYPES]
    faults = detected = false_alarms = normals = 0
    for row in records:
        true = row.get("true_type")
        fault = row["true_label"] == 2
        frozen=row.get("rejection_policy",{})
        choice = decide(row["scores"], frozen.get("threshold",threshold), frozen.get("margin",margin))
        detected += fault and choice == 2
        faults += fault
        normals += row["true_label"] == 0
        false_alarms += row["true_label"] == 0 and choice == 2
        scores = row.get("type_scores", [])
        candidate = max(range(len(scores)), key=scores.__getitem__) if scores and choice == 2 else None
        allowed=row.get("supported_types",supported) if supported is not None else None
        if candidate is not None and allowed is not None and FAULT_TYPES[candidate] not in allowed:
            candidate = None
        for i in range(len(FAULT_TYPES)):
            counts[i][0] += true == i and candidate == i
            counts[i][1] += candidate == i and (true != i if true is not None else not fault)
            counts[i][2] += true == i and candidate != i
    per_type = {}
    for name, (tp, fp, fn) in zip(FAULT_TYPES, counts):
        per_type[name] = {"precision": tp/(tp+fp) if tp+fp else 0.,
                          "recall": tp/(tp+fn) if tp+fn else 0.,
                          "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                          "tp": tp, "fp": fp, "fn": fn}
    return {"macro_f1": sum(r["f1"] for r in per_type.values())/len(FAULT_TYPES),
            "macro_recall": sum(r["recall"] for r in per_type.values())/len(FAULT_TYPES),
            "fault_recall": detected/faults if faults else 0.,
            "false_alarm_rate": false_alarms/normals if normals else 0., "per_type": per_type}


def select_type_policy(records, train_support):
    best = None
    for threshold in (.34, .4, .5, .6, .7):
        for margin in (0., .05, .1, .2):
            internal = diagnostic_metrics(records, threshold, margin)
            supported = []
            for i, kind in enumerate(FAULT_TYPES):
                groups = {r.get("episode_id") for r in records if r.get("true_type") == i}
                count=len(groups)
                if any(r.get("context_end") for r in records):
                    from app.modulos.clasificador.context_support import disjoint_context_count
                    count=disjoint_context_count([r for r in records if r.get("true_type")==i and r.get("evaluable",True)])
                if train_support.get(kind, 0) >= 3 and count >= 3 and internal["per_type"][kind]["recall"] >= .5:
                    supported.append(kind)
            metrics = diagnostic_metrics(records, threshold, margin, supported)
            rank = (metrics["macro_f1"], -metrics["false_alarm_rate"], metrics["fault_recall"])
            if best is None or rank > best[0]:
                best = rank, {"threshold": threshold, "margin": margin, "validated": bool(records)}, metrics, supported
    return best[1:]
