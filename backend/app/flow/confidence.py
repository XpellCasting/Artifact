"""Versioned, deterministic confidence annotation for a single output reading.

This is an operational index, not a calibrated probability. Adapters retain
filter-specific score semantics; only original assessments receive a vote.
"""
from copy import deepcopy
import math
from numbers import Real

from app.flow.execution import FILTER_TYPES


CONFIDENCE_POLICY = {
    "schema_version": "reading-confidence-v1",
    "method": "uniform_v1",
    "confidence_kind": "operational_index",
    "confidence_calibrated": False,
    "aggregation": "one_minus_mean_normalized_suspicion",
    "coverage_denominator": "enabled_filter_nodes",
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bounded(value):
    number = _number(value)
    return number if number is not None and 0 <= number <= 1 else None


def _relative(score, threshold):
    score, threshold = _number(score), _number(threshold)
    if score is None or threshold is None or score < 0 or threshold <= 0:
        return None
    ratio = score / threshold
    return 1.0 if math.isinf(ratio) else ratio / (1 + ratio)


def _statistical(entry, sensor_kind):
    data = entry.get("data") or {}
    observation = data.get("statistical_observation") or {}
    if observation.get("evaluable") is False:
        return None, None, observation.get("reason") or "statistical_not_evaluable"
    if (data.get("suspected_pattern", data.get("pattern")) == "unavailable"
            or data.get("constant_score") == 1
            or (data.get("persistence") or {}).get("detected") is True):
        return 1.0, "explicit_availability_or_persistence_v1", None
    if "joint_score" in data or "joint_score" in observation:
        value = _relative(data.get("joint_score", observation.get("joint_score")),
                          data.get("threshold", observation.get("threshold")))
        return value, "relative_statistic_v1", None if value is not None else "invalid_statistical_ratio"
    if "suspicion_score" in data:
        # Continuous FOCuS strengths can exceed 1; never silently clip these.
        if sensor_kind == "continuous" and data.get("continuous_method") == "residual_focus_v1":
            value = _relative(data["suspicion_score"], data.get("threshold"))
            return value, "relative_statistic_v1", None if value is not None else "invalid_statistical_ratio"
        value = _bounded(data["suspicion_score"])
        return value, "bounded_statistical_strength_v1", None if value is not None else "invalid_statistical_score"
    if entry.get("status") == "no_suspicion":
        # Binary/multistate and older records may expose only an explicit
        # negative assessment, rather than a sub-threshold numeric statistic.
        return 0.0, "explicit_no_suspicion_v1", None
    return None, None, "missing_statistical_score"


def _contextual(entry):
    data = entry.get("data") or {}
    if data.get("context_evaluable") is False or (data.get("map_quality") or {}).get("stale") is True:
        return None, None, "context_not_evaluable"
    outcome = data.get("context_outcome", entry.get("status"))
    if outcome == "supported":
        return 0.0, "contextual_support_v1", None
    if outcome == "contradicted":
        if "decision_score" in data or "decision_threshold" in data:
            value = _relative(data.get("decision_score"), data.get("decision_threshold"))
            return value, "relative_contextual_contradiction_v1", None if value is not None else "invalid_contextual_ratio"
        value = _bounded(data.get("contradiction_score"))
        return value, "bounded_contextual_contradiction_v1", None if value is not None else "invalid_contextual_score"
    # adjusted_suspicion_score can already include F1. Reusing it as a second
    # independent score would count the same statistical evidence twice.
    return None, None, "context_not_evaluable"


def _classifier(entry):
    scores = (entry.get("data") or {}).get("class_scores")
    if not isinstance(scores, dict) or set(scores) != {"normal", "environmental_anomaly", "sensor_fault"}:
        return None, None, "missing_classifier_scores"
    values = [_bounded(value) for value in scores.values()]
    if any(value is None for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-5, rel_tol=0):
        return None, None, "invalid_classifier_scores"
    # data.confidence is max(class_scores), which can be confidence in Normal.
    return float(scores["sensor_fault"]), "classifier_fault_component_v1", None


def annotate_confidence(item):
    """Append confidence to Output without changing diagnostic or trace fields."""
    trace = item.get("execution_trace")
    trace = trace if isinstance(trace, dict) else {}
    filters = {key: record for key, record in trace.get("filters", {}).items()
               if record.get("module_type") in FILTER_TYPES}
    active = {key: record for key, record in filters.items() if record.get("enabled") is True}
    evidence, ambiguous = {}, set()
    for entry in trace.get("evidence", []):
        identity = entry.get("evidence_id")
        if not identity:
            continue
        if identity in evidence and evidence[identity] != entry:
            ambiguous.add(identity)
        evidence[identity] = entry

    contributions, excluded = {}, {}
    for node_id, record in sorted(filters.items()):
        if node_id not in active:
            excluded[node_id] = "disabled"
            continue
        if record.get("execution_status") != "completed":
            excluded[node_id] = record.get("execution_status") or "not_evaluable"
            continue
        if trace.get("availability_mask", {}).get(node_id) is not True:
            excluded[node_id] = "unavailable_filter_result"
            continue
        identities = list(dict.fromkeys(record.get("evidence_ids", [])))
        if any(identity in ambiguous for identity in identities):
            excluded[node_id] = "ambiguous_evidence"
            continue
        entries = [evidence[identity] for identity in identities if identity in evidence]
        if not entries or len(entries) != len(identities):
            excluded[node_id] = "missing_evidence"
            continue
        if any(entry.get("node_id") != node_id or entry.get("module_type") != record["module_type"]
               or entry.get("observation_id") != item.get("observation_id")
               or entry.get("execution_id") != item.get("execution_id") for entry in entries):
            excluded[node_id] = "mismatched_evidence"
            continue
        entry = entries[-1]
        if entry.get("status") in {"omitted", "error", "not_evaluable"}:
            excluded[node_id] = "evidence_not_evaluable"
            continue
        if record["module_type"] == "statistical_focus":
            score, adapter, reason = _statistical(entry, item.get("sensor_kind"))
        elif record["module_type"] == "contextual":
            score, adapter, reason = _contextual(entry)
        else:
            score, adapter, reason = _classifier(entry)
        if score is None:
            excluded[node_id] = reason
            continue
        contributions[node_id] = {"suspicion_score": score, "adapter": adapter,
                                  "evidence_ids": [entry["evidence_id"]]}

    reason = None
    if not trace:
        reason = "missing_execution_trace"
    elif item.get("is_valid_value") is False:
        reason = "invalid_reading"
    elif item.get("final_decision") == "not_routed":
        reason = "reading_not_routed"
    elif any(record.get("execution_status") == "error" for record in active.values()):
        reason = "filter_execution_error"
    elif any(record.get("execution_status") == "pending" for record in active.values()):
        reason = "execution_pending"
    elif not active:
        reason = "no_active_filters"
    elif not contributions:
        reason = "no_usable_filter_results"
    confidence = None if reason else 1 - math.fsum(
        record["suspicion_score"] for record in contributions.values()) / len(contributions)
    item.update(confidence=confidence, confidence_kind=CONFIDENCE_POLICY["confidence_kind"],
                confidence_calibrated=False,
                evidence_coverage=len(contributions) / len(active) if active else 0.0,
                confidence_fusion={
                    "schema_version": CONFIDENCE_POLICY["schema_version"],
                    "method": CONFIDENCE_POLICY["method"],
                    "status": "not_evaluable" if reason else "completed", "reason": reason,
                    "configured_filters": len(filters), "active_filters": len(active),
                    "available_filters": len(contributions), "contributions": contributions,
                    "excluded_filters": excluded,
                })
    return item


def confidence_policy():
    """Snapshot a policy once per run, rather than copy it into every reading."""
    return deepcopy(CONFIDENCE_POLICY)
