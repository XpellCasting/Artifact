"""Explain changes to the published percentage using the votes selected by fusion."""
import json

LABELS = {"statistical_focus": "Estadístico", "contextual": "Contextual", "classifier": "Clasificador"}
EXCLUSIONS = {"insufficient_data": "datos insuficientes", "skipped": "omitido por el flujo",
              "disabled": "deshabilitado", "context_not_evaluable": "contexto no evaluable",
              "error": "error de ejecución", "pending": "pendiente"}


def _pick(data, keys):
    return {key: data[key] for key in keys if data.get(key) is not None}


def _filter_evidence(kind, entry, neighbor_budget):
    data = entry.get("data") or {}
    if kind == "classifier":
        return _pick(data, ("class_scores",))
    if kind == "statistical_focus":
        observation = data.get("statistical_observation") or {}
        facts = _pick({**observation, **data}, ("suspected_pattern", "pattern", "joint_score", "threshold",
                      "suspicion_score", "constant_score", "continuous_method", "alert_emitted", "suppressed"))
        if data.get("persistence"):
            facts["persistence"] = _pick(data["persistence"], ("detected", "reason"))
        return facts
    if kind != "contextual":
        return {}
    detail = _pick(data, ("context_outcome", "decision_score", "decision_threshold", "contradiction_score"))
    neighbors = []
    for row in data.get("evidence") or []:
        if not row.get("neighbor_id"):
            continue
        neighbor = _pick(row, ("neighbor_id", "observed_value", "carried_forward", "observation_status",
                               "weighted_contribution"))
        if row.get("evidence_method") == "v5_temporal_feature_models":
            # In V5 the temporal checks supersede the directional verdicts from V4.
            neighbor = _pick(row, ("neighbor_id", "carried_forward"))
            neighbor["checks"] = {family: _pick(row.get(family + "_check") or {}, ("status", "ratio"))
                                  for family in ("level", "innovation", "trend")}
        neighbors.append(neighbor)
    # Keep Recorder attributes compact even for a large custom neighborhood.
    # The full list remains available in the referenced filter evidence.
    neighbors.sort(key=lambda n: abs(n.get("weighted_contribution") or 0), reverse=True)
    selected = []
    for neighbor in neighbors:
        neighbor_budget -= len(json.dumps(neighbor, ensure_ascii=False, default=str).encode("utf-8")) + 2
        if neighbor_budget < 0:
            break
        selected.append(neighbor)
    detail["neighbors"] = selected
    if len(selected) < len(neighbors):
        detail["total_neighbors"] = len(neighbors)
    return detail


def _number(value, digits=6):
    return f"{value:.{digits}f}".rstrip("0").rstrip(".").replace(".", ",")


def _evidence_reason(kind, evidence):
    if kind == "contextual":
        for status, wording in (("contradicted", "Contradicción"), ("supported", "Respaldo")):
            if evidence.get("context_outcome") == status:
                verdict = "contradicts" if status == "contradicted" else "supports"
                neighbors = [n["neighbor_id"] for n in evidence["neighbors"] if n.get("observation_status") == verdict]
                if neighbors:
                    return wording + " según " + ", ".join(neighbors[:3]) + "."
                if evidence.get("decision_score") is not None and evidence.get("decision_threshold") is not None:
                    return (f"{wording} contextual con puntuación {_number(evidence['decision_score'])} "
                            f"y umbral {_number(evidence['decision_threshold'])}.")
                return wording + " contextual."
    if kind == "statistical_focus":
        if (evidence.get("persistence") or {}).get("detected"):
            return "El filtro detectó persistencia."
        pattern = evidence.get("suspected_pattern") or evidence.get("pattern")
        if pattern:
            return f"El patrón registrado fue {pattern}."
        if evidence.get("joint_score") is not None and evidence.get("threshold") is not None:
            return f"El estadístico fue {_number(evidence['joint_score'])} con umbral {_number(evidence['threshold'])}."
    return ""


def explain_confidence_change(previous, current, reading):
    """Keep the last transition on plateaus, but always advance its next comparison baseline."""
    fusion = reading.get("confidence_fusion") or {}
    trace = reading.get("execution_trace") or {}
    votes = fusion.get("contributions") or {}
    scores = {node: vote["suspicion_score"] for node, vote in votes.items()}
    current["_scores"] = scores
    if previous.get("evaluated_at") is not None and previous["value"] == current["value"]:
        return

    old_value, value = previous.get("value"), current["value"]
    old_scores = previous.get("_scores", {})
    comparable = value is not None and old_value is not None
    delta = round(value - old_value, 6) if comparable else None
    if comparable:
        kind = "increase" if delta > 0 else "decrease"
        digits = 6 if round(old_value, 2) == round(value, 2) else 2
        sentences = [f"{'Subió' if delta > 0 else 'Bajó'} de {_number(old_value, digits)} % a {_number(value, digits)} %."]
    elif value is None:
        kind = "unavailable"
        sentences = ["Lectura sin confianza evaluable."]
    elif previous.get("evaluated_at") is None:
        kind = "initial"
        sentences = [f"Primera evaluación con {_number(value)} % de confianza."]
    else:
        kind = "resumed"
        sentences = [f"Vuelve a haber confianza evaluable ({_number(value)} %)."]

    evidence = {entry["evidence_id"]: entry for entry in trace.get("evidence", []) if entry.get("evidence_id")}
    order = trace.get("execution_order", [])
    excluded = fusion.get("excluded_filters") or {}
    modules = trace.get("filters", {})
    contextual_count = sum(modules.get(node, {}).get("module_type") == "contextual" for node in votes)
    # Recorder limits the entire attributes object to 16 KiB. Reserve space for
    # the sensor metadata, the three filter assessments and the explanation.
    neighbor_budget = 8_192 // max(1, contextual_count)
    rows = []
    for node in dict.fromkeys([*order, *votes, *old_scores]):
        if node not in votes and node not in old_scores:
            continue
        module = modules.get(node, {}).get("module_type")
        label = LABELS.get(module, node)
        before, after = old_scores.get(node), scores.get(node)
        row = {"node_id": node, "module_type": module, "before": before, "after": after}
        if node in votes:
            vote = votes[node]
            ids = vote.get("evidence_ids", [])
            entry = evidence.get(ids[-1], {}) if ids else {}
            facts = _filter_evidence(module, entry, neighbor_budget)
            row.update(evidence_ids=ids, adapter=vote.get("adapter"), status=entry.get("status"), evidence=facts)
            if comparable and before is not None and before != after:
                subject = "puntuación de falla de sensor" if module == "classifier" else "sospecha"
                sentences.append(f"{label} cambió su {subject} de {_number(before)} a {_number(after)}.")
                if explanation := _evidence_reason(module, facts):
                    sentences.append(explanation)
            elif comparable and before is None and old_scores:
                sentences.append(f"{label} empezó a aportar (sospecha {_number(after)}).")
        elif comparable:
            reason = excluded.get(node, "sin resultado")
            sentences.append(f"{label} dejó de aportar ({EXCLUSIONS.get(reason, reason)}).")
        if comparable and scores and old_scores and fusion.get("method") == "uniform_v1":
            # Change of each normalized vote including the change of denominator.
            row["impact_pp"] = round(100 * ((before or 0) / len(old_scores) - (after or 0) / len(scores)), 6)
        rows.append(row)
    if comparable and (not scores or not old_scores):
        sentences.append("Evidencia insuficiente para atribuir el cambio a un filtro.")
    if comparable and scores.keys() != old_scores.keys() and old_scores and scores:
        sentences.append(f"Filtros con aporte: {len(old_scores)} → {len(scores)}.")
    current["confidence_change_reason"] = " ".join(sentences)
    current["confidence_change_detail"] = {
        "kind": kind, "before_pct": old_value, "after_pct": value, "delta_pp": delta,
        "evaluated_at": current["evaluated_at"],
        "source_timestamp": current["source_timestamp"], "previous_source_timestamp": previous.get("source_timestamp"),
        "source_state": current["source_state"], "previous_source_state": previous.get("source_state"),
        "source_value": current["source_value"], "previous_source_value": previous.get("source_value"),
        "observation_id": current["observation_id"], "previous_observation_id": previous.get("observation_id"),
        "previous_coverage": previous.get("evidence_coverage"), "coverage": current["evidence_coverage"],
        "execution_order": order, "filters": rows, "excluded_filters": excluded, "reason": fusion.get("reason"),
    }
