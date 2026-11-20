"""Decisions at the classifier and output of the three thesis architectures."""

MODEL_DECISIONS = {"compatible": "fault_compatible", "environmental_anomaly": "environmental",
                   "no_fault_evidence": "normal"}
DECISION_SIGNALS = {"fault_compatible": "positive", "environmental": "negative",
                    "normal": "negative", "no_alert": "negative", "indeterminate": "unknown"}


def _filters(evidence):
    return {entry.get("module_type"): entry for entry in evidence
            if entry.get("status") != "omitted"}


def _conflict(filters, decision):
    # Use the decisions at the filters' native calibrated thresholds. Their raw
    # scores have different scales and remain in the original evidence entries.
    if filters.get("statistical_focus", {}).get("signal") != "positive":
        return False
    context = filters.get("contextual", {}).get("signal")
    return ((context == "negative" and decision in {"normal", "fault_compatible"})
            or (context == "positive" and decision == "environmental"))


def parallel_diagnosis(evidence, model):
    decision = MODEL_DECISIONS.get(model.get("status"))
    if not _conflict(_filters(evidence), decision):
        return model
    return {**model, "model_verdict": {key: model.get(key) for key in (
                "status", "predicted_label", "confidence", "fault_type", "fault_type_status", "fault_type_confidence")},
            "status": "indeterminate", "predicted_label": None, "fault_type": None,
            "fault_type_status": "not_applicable", "fault_type_confidence": None,
            "needs_review": True, "reasons": [*model.get("reasons", []), "filter_model_disagreement"]}


def output_verdict(policy, evidence):
    filters = _filters(evidence)
    if policy != "classifier_v1":
        temporal = filters.get("statistical_focus", {}).get("signal")
        if temporal == "negative":
            return "no_alert", "no_statistical_suspicion"
        if temporal != "positive":
            return "indeterminate", "statistical_evidence_unavailable"
        if policy == "sequential_v1" and filters.get("contextual", {}).get("signal") == "negative":
            return "environmental", "context_supports_suspicious_change"

    model = filters.get("classifier", {})
    decision = MODEL_DECISIONS.get(model.get("status"), "indeterminate")
    if policy == "hybrid_v1" and _conflict(filters, decision):
        return "indeterminate", "filter_model_disagreement"
    if decision == "indeterminate":
        return decision, "classifier_abstention"
    return decision, "accepted_classifier_verdict"
