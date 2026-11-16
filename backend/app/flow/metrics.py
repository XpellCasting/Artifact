"""Run-level metrics for the final decision on each input reading."""
from collections import Counter
from math import isfinite

METRICS_VERSION = 1
CLASSES = ("normal", "anomalia", "Falla")
REFERENCE_LABELS = (*CLASSES, "NULL", None)
DIAGNOSES = (*CLASSES, "no_diagnosis")
EXPLICIT_DIAGNOSES = {"normal": "normal", "environmental": "anomalia", "fault_compatible": "Falla"}
DETECTION_STATES = {
    "environmental": "detected", "fault_compatible": "detected", "suspicion": "detected",
    "normal": "no_signal", "no_alert": "no_signal",
    "indeterminate": "not_evaluated", "not_routed": "not_evaluated",
}
NO_LABELS = "La ejecución no tiene lecturas etiquetadas."


def rate(numerator, denominator, reason):
    return {"value": numerator / denominator if denominator else None,
            "numerator": numerator, "denominator": denominator,
            "reason": None if denominator else reason}


class FlowMetrics:
    def __init__(self):
        self.total = 0
        self.support = Counter()
        self.decisions = {label: Counter() for label in CLASSES}
        self.invalid_decisions = 0

    def add(self, expected, decision):
        self.total += 1
        if expected not in CLASSES:
            return
        self.support[expected] += 1
        if decision in DETECTION_STATES:
            self.decisions[expected][decision] += 1
        else:
            self.invalid_decisions += 1

    def finalize(self):
        labelled = self.support.total()
        support = {"total": self.total, "labelled": labelled, "unlabelled": self.total - labelled,
                   "by_class": {label: self.support[label] for label in CLASSES}}
        if not labelled:
            return unavailable_metrics(support, NO_LABELS, status="no_labels")
        if self.invalid_decisions:
            return unavailable_metrics(support, "Hay lecturas etiquetadas sin una decisión final reconocida.")

        matrix = {label: dict.fromkeys(DIAGNOSES, 0) for label in CLASSES}
        detection = {label: dict.fromkeys(DETECTION_STATES.values(), 0) for label in CLASSES}
        breakdown = dict.fromkeys((decision for decision in DETECTION_STATES if decision not in EXPLICIT_DIAGNOSES), 0)
        for label, decisions in self.decisions.items():
            for decision, count in decisions.items():
                diagnosis = EXPLICIT_DIAGNOSES.get(decision, "no_diagnosis")
                matrix[label][diagnosis] += count
                detection[label][DETECTION_STATES[decision]] += count
                if diagnosis == "no_diagnosis":
                    breakdown[decision] += count

        by_class = {}
        for label in CLASSES:
            tp = matrix[label][label]
            predicted = sum(row[label] for row in matrix.values())
            actual = self.support[label]
            fp, fn = predicted - tp, actual - tp
            by_class[label] = {
                "support": actual, "tp": tp, "fp": fp, "fn": fn,
                "precision": rate(tp, predicted, "No hubo predicciones de esta clase."),
                "recall": rate(tp, actual, "No hay lecturas etiquetadas de esta clase."),
                "f1": rate(2 * tp, 2 * tp + fp + fn, "La clase no aparece en las etiquetas ni en los diagnósticos."),
            }
        missing = [label for label in CLASSES if not self.support[label]]
        macro = {
            "value": None if missing else sum(item["f1"]["value"] for item in by_class.values()) / len(CLASSES),
            "reason": "Faltan clases reales para calcular F1 macro." if missing else None,
            "missing_classes": missing,
        }
        undiagnosed = sum(breakdown.values())
        groups = {
            label: {**detection[label], "total": self.support[label],
                    "recall": rate(detection[label]["detected"], self.support[label],
                                   "No hay lecturas etiquetadas de esta clase.")}
            for label in ("anomalia", "Falla")
        }
        return {
            "version": METRICS_VERSION, "status": "available", "reason": None, "support": support,
            "diagnosis": {
                "matrix": matrix, "by_class": by_class, "macro_f1": macro,
                "coverage": rate(labelled - undiagnosed, labelled, NO_LABELS),
                "no_diagnosis": {**rate(undiagnosed, labelled, NO_LABELS), "by_decision": breakdown},
            },
            "detection": {
                "by_class": groups,
                "normal_as_alteration": rate(detection["normal"]["detected"], self.support["normal"],
                                             "No hay lecturas normales etiquetadas."),
                "valid_as_fault": rate(matrix["normal"]["Falla"] + matrix["anomalia"]["Falla"],
                                       self.support["normal"] + self.support["anomalia"],
                                       "No hay lecturas normales ni ambientales etiquetadas."),
            },
        }


def unavailable_metrics(support, reason, *, status="incomplete"):
    return {"version": METRICS_VERSION, "status": status, "reason": reason, "support": support,
            "diagnosis": None, "detection": None}


def _count(value):
    return type(value) is int and value >= 0


def saved_support(summary):
    report = summary.get("evaluation") or {}
    groups = report.get("groups")
    counts = None
    if isinstance(groups, list) and all(
        isinstance(group, dict) and "expected" in group and group["expected"] in REFERENCE_LABELS
        and _count(group.get("total"))
        for group in groups
    ):
        counts = Counter()
        for group in groups:
            counts[group["expected"]] += group["total"]
    total = next((value for value in (summary.get("candidate_count"), summary.get("output_count"),
                                     (summary.get("node_counts") or {}).get("output")) if _count(value)), None)
    labelled = sum(counts[label] for label in CLASSES) if counts is not None else None
    if report.get("labelled") is False and labelled is None:
        labelled = 0
    unlabelled = total - labelled if total is not None and labelled is not None and total >= labelled else None
    return {"total": total, "labelled": labelled, "unlabelled": unlabelled,
            "by_class": {label: counts[label] for label in CLASSES} if counts is not None else None}


def reconstruct_metrics(summary, records):
    support = saved_support(summary)
    report = summary.get("evaluation") or {}
    accumulator = FlowMetrics()
    matched = 0
    invalid_labels = 0
    for record in records:
        item = record["payload"]
        evaluation = item.get("evaluation") or {}
        expected = evaluation.get("expected")
        matched += "expected" in evaluation
        invalid_labels += expected not in REFERENCE_LABELS
        accumulator.add(expected, item.get("final_decision"))

    output_counts = (support["total"], summary.get("output_count"), (summary.get("node_counts") or {}).get("output"))
    if (not accumulator.total and support["total"] != 0) or any(
        _count(count) and count != accumulator.total for count in output_counts
    ):
        return unavailable_metrics(support, "Los resultados guardados de Salida no contienen todas las lecturas de la ejecución.")
    expected_matches = report.get("matched_rows")
    groups = report.get("groups")
    grouped_matches = sum(group["total"] for group in groups) if support["by_class"] is not None else None
    if any(_count(count) and count != matched for count in (expected_matches, grouped_matches)) or (
        support["by_class"] is not None
        and any(support["by_class"][label] != accumulator.support[label] for label in CLASSES)
    ) or (report.get("labelled") is False and accumulator.support.total()):
        return unavailable_metrics(support, "Las etiquetas recuperadas no coinciden con el resumen guardado.")
    if invalid_labels or (
        report.get("labelled") is True and not matched
    ) or (
        0 < matched < accumulator.total and not _count(expected_matches) and grouped_matches is None
    ):
        return unavailable_metrics(support, "Las etiquetas guardadas son insuficientes para reconstruir las métricas.")
    return accumulator.finalize()


FILTER_PHASES = {
    "statistical_focus": "statistical_execution",
    "contextual": "contextual_execution",
    "classifier": "classifier_execution",
}


def _duration(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def run_timings(summary, flow):
    phases = (summary.get("performance") or {}).get("phases_ms") or {}
    if _duration(phases.get("total")):
        total = {"ms": phases["total"], "source": "total", "reason": None}
    elif _duration(summary.get("elapsed_ms")):
        total = {"ms": summary["elapsed_ms"], "source": "elapsed_ms", "reason": None}
    else:
        total = {"ms": None, "source": None, "reason": "No se registró un tiempo total."}

    filters = {}
    node_timings = summary.get("node_timings_ms") or {}
    nodes = flow.get("nodes") or []
    for kind, phase in FILTER_PHASES.items():
        if _duration(phases.get(phase)):
            filters[kind] = {"ms": phases[phase], "source": "phases_ms", "reason": None}
            continue
        times = [node_timings.get(node["id"]) for node in nodes if node.get("type") == kind]
        if times and all(_duration(value) for value in times):
            filters[kind] = {"ms": round(sum(times), 3), "source": "node_timings_ms", "reason": None}
        else:
            filters[kind] = {"ms": None, "source": None, "reason": "No se registró un tiempo acumulado para este filtro."}
    return {"total": total, "filters": filters}
