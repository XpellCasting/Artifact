"""Explain changes to the published percentage using the votes selected by fusion."""
import json
import math
from itertools import permutations

LABELS = {"statistical_focus": "Estadístico", "contextual": "Contextual", "classifier": "Clasificador"}
EXCLUSIONS = {"insufficient_data": "datos insuficientes", "skipped": "omitido por el flujo",
              "disabled": "deshabilitado", "context_not_evaluable": "contexto no evaluable",
              "error": "error de ejecución", "pending": "pendiente"}
FAMILIES = {"level": "nivel", "innovation": "innovación", "trend": "tendencia"}
RELATIONS = {"supports": "respaldo", "contradicts": "contradicción", "observed_neutral": "neutral"}


def _pick(data, keys):
    return {key: data[key] for key in keys if data.get(key) is not None}


def _filter_evidence(kind, entry, neighbor_budget):
    data = entry.get("data") or {}
    if kind == "classifier":
        result = _pick(data, ("class_scores", "predicted_label", "fault_type", "model_version", "window_evidence"))
        result["coverage"] = _pick(data.get("input_coverage") or {}, ("valid_value_steps", "required_steps"))
        return result
    if kind == "statistical_focus":
        observation = data.get("statistical_observation") or {}
        facts = _pick({**observation, **data}, ("suspected_pattern", "pattern", "joint_score", "threshold",
                      "suspicion_score", "constant_score", "continuous_method", "alert_emitted", "suppressed",
                      "expected_value", "residual", "dominant_channel", "channel_scores", "pattern_detail"))
        facts.update(_pick(data.get("reading") or {}, ("state_changes", "activations", "gap_seconds",
                           "report_gap_seconds", "state_duration_seconds", "transition_probability", "window_seconds")))
        persistence = data.get("persistence") or observation.get("persistence")
        if persistence:
            facts["persistence"] = _pick(persistence, ("detected", "reason", "duration_seconds", "threshold_seconds"))
        return facts
    if kind != "contextual":
        return {}
    detail = _pick(data, ("context_outcome", "decision_score", "decision_threshold", "contradiction_score"))
    families = {key: row for key, row in (data.get("family_scores") or {}).items()
                if row.get("status") == "available" and row.get("score") is not None}
    dominant = max(families, key=lambda key: families[key]["score"]) if families else None
    if dominant:
        detail["decision_family"] = dominant
    neighbors = []
    related = data.get("evidence") or []
    observed_statuses = {"supports", "contradicts", "observed_neutral"}
    total_weight = sum(row.get("weight", 0) for row in related if row.get("observation_status") in observed_statuses)
    for row in related:
        if not row.get("neighbor_id"):
            continue
        if data.get("decision_method") == "calibrated_max_weighted_family_ratio" and row.get("evidence_method") != "v5_temporal_feature_models":
            continue  # The V4 envelope can also contain directional neighbors unused by V5.
        neighbor = _pick(row, ("neighbor_id", "observed_value", "carried_forward", "observation_status",
                               "weighted_contribution", "expected_direction_or_state", "lag_seconds"))
        if row.get("evidence_method") == "v5_temporal_feature_models":
            # In V5 the temporal checks supersede the directional verdicts from V4.
            neighbor = _pick(row, ("neighbor_id", "carried_forward"))
            neighbor["checks"] = {family: _pick(row[family + "_check"], ("status", "ratio", "expected_value",
                                  "predictor_value", "residual", "tolerance", "reason"))
                                  for family in data.get("available_families", ("level", "innovation", "trend"))
                                  if row.get(family + "_check")}
            check = neighbor["checks"].get(dominant, {})
            if (dominant and check.get("ratio") is not None and row.get("weight") is not None
                    and check.get("status") == "evaluated" and families[dominant].get("evaluated_weight")):
                neighbor["decision_term"] = (row["weight"] * check["ratio"] / families[dominant]["evaluated_weight"])
        elif total_weight and row.get("observation_status") in observed_statuses and row.get("weighted_contribution") is not None:
            neighbor["contradiction_term"] = (abs(row["weighted_contribution"]) / total_weight
                                               if row["observation_status"] == "contradicts" else 0.)
        neighbors.append(neighbor)
    # Keep Recorder attributes compact even for a large custom neighborhood.
    # The full list remains available in the referenced filter evidence.
    neighbors.sort(key=lambda n: abs(n.get("decision_term", n.get("weighted_contribution")) or 0), reverse=True)
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


def _window_impacts(previous, current):
    """Average the marginal change in fault softmax over all six window replacement orders."""
    old = previous.get("window_evidence") or {}
    new = current.get("window_evidence") or {}
    if not old or not new or previous.get("model_version") != current.get("model_version") or old["bias"] != new["bias"]:
        return {}
    before, after = old["windows"], new["windows"]
    horizons = [window["seconds"] for window in before]
    if len(horizons) != 3 or horizons != [window["seconds"] for window in after]:
        return {}

    def fault(mask):
        windows = [after[i] if mask & (1 << i) else before[i] for i in range(3)]
        logits = [bias + sum(w["logits"][label] for w in windows) for label, bias in enumerate(new["bias"])]
        pivot = max(logits)
        exp = [math.exp(value - pivot) for value in logits]
        return exp[2] / sum(exp)

    probabilities = [fault(mask) for mask in range(8)]
    impacts = dict.fromkeys(horizons, 0.)
    for order in permutations(range(3)):
        mask = 0
        for i in order:
            updated = mask | (1 << i)
            impacts[horizons[i]] += (probabilities[updated] - probabilities[mask]) * 100 / 6
            mask = updated
    return {str(key): round(value, 6) for key, value in impacts.items()}


def _evidence_reason(kind, evidence, previous, window_impacts):
    if kind == "classifier" and window_impacts:
        horizon = max(window_impacts, key=lambda h: abs(window_impacts[h]))
        impact = window_impacts[horizon]
        return (f"La ventana de {int(horizon) // 3600} h del propio sensor "
                f"{'elevó' if impact >= 0 else 'redujo'} la puntuación de falla en {_number(abs(impact), 4)} puntos.")
    if kind == "contextual":
        old = {n["neighbor_id"]: n for n in previous.get("neighbors", [])}
        temporal = [n for n in evidence["neighbors"] if "decision_term" in n]
        if temporal:
            def term_change(n):
                was = old.get(n["neighbor_id"], {}).get("decision_term", 0.)
                return abs(n["decision_term"] - was)
            n = max(temporal, key=term_change)
            check = n["checks"][evidence["decision_family"]]
            return (f"{n['neighbor_id']} aportó {_number(n['decision_term'])} al estadístico de "
                    f"{FAMILIES[evidence['decision_family']]} (desviación/tolerancia {_number(check['ratio'])}).")
        for status, wording in (("contradicted", "Contradicción"), ("supported", "Respaldo")):
            if evidence.get("context_outcome") == status:
                verdict = "contradicts" if status == "contradicted" else "supports"
                neighbors = [n for n in evidence["neighbors"] if n.get("observation_status") == verdict]
                if neighbors:
                    def changed(n):
                        prior = old.get(n["neighbor_id"], {})
                        return (prior.get("observation_status") != verdict,
                                abs((n.get("weighted_contribution") or 0) - (prior.get("weighted_contribution") or 0)))
                    neighbors.sort(key=changed, reverse=True)
                    examples = []
                    for n in neighbors[:3]:
                        prior = old.get(n["neighbor_id"], {})
                        notes = []
                        if "observed_value" in n:
                            value = str(n["observed_value"])
                            if "observed_value" in prior:
                                value = f"{prior['observed_value']} → {value}"
                            notes.append("valor " + value)
                        if prior.get("observation_status") in RELATIONS:
                            notes.append(RELATIONS[prior["observation_status"]] + " → " + RELATIONS[verdict])
                        examples.append(n["neighbor_id"] + (" (" + ", ".join(notes) + ")" if notes else ""))
                    return wording + " según " + ", ".join(examples) + "."
                if evidence.get("decision_score") is not None and evidence.get("decision_threshold") is not None:
                    return (f"{wording} contextual con puntuación {_number(evidence['decision_score'])} "
                            f"y umbral {_number(evidence['decision_threshold'])}.")
                return wording + " contextual."
    if kind == "statistical_focus":
        if (evidence.get("persistence") or {}).get("detected"):
            return "El filtro detectó persistencia."
        if evidence.get("expected_value") is not None and evidence.get("residual") is not None:
            return (f"El valor esperado era {_number(evidence['expected_value'])} y la lectura se apartó "
                    f"de él en {_number(evidence['residual'])}.")
        if evidence.get("pattern_detail"):
            return str(evidence["pattern_detail"]) + "."
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
    evidence = {entry["evidence_id"]: entry for entry in trace.get("evidence", []) if entry.get("evidence_id")}
    modules = trace.get("filters", {})
    contextual_count = sum(modules.get(node, {}).get("module_type") == "contextual" for node in votes)
    # Two snapshots share the available Recorder attribute space.
    budget = 3_072 // max(1, contextual_count)
    facts_by_node = {}
    for node, vote in votes.items():
        ids = vote.get("evidence_ids") or []
        entry = evidence.get(ids[-1], {}) if ids else {}
        facts_by_node[node] = _filter_evidence(modules.get(node, {}).get("module_type"), entry, budget)
    current["_facts"] = facts_by_node
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
        source_before = previous.get("source_state")
        source_after = current.get("source_state")
        if source_before != source_after and source_before is not None and source_after is not None:
            sentences.append(f"La lectura pasó de {source_before} a {source_after}.")
    elif value is None:
        kind = "unavailable"
        sentences = ["Lectura sin confianza evaluable."]
    elif previous.get("evaluated_at") is None:
        kind = "initial"
        sentences = [f"Primera evaluación con {_number(value)} % de confianza."]
    else:
        kind = "resumed"
        sentences = [f"Vuelve a haber confianza evaluable ({_number(value)} %)."]

    order = trace.get("execution_order", [])
    excluded = fusion.get("excluded_filters") or {}
    rows = []
    for node in dict.fromkeys([*order, *votes, *old_scores]):
        if node not in votes and node not in old_scores:
            continue
        module = modules.get(node, {}).get("module_type")
        label = LABELS.get(module, node)
        before, after = old_scores.get(node), scores.get(node)
        row = {"node_id": node, "module_type": module, "before": before, "after": after}
        previous_facts = previous.get("_facts", {}).get(node)
        if previous_facts is not None:
            row["previous_evidence"] = previous_facts
        if node in votes:
            vote = votes[node]
            ids = vote.get("evidence_ids", [])
            entry = evidence.get(ids[-1], {}) if ids else {}
            facts = facts_by_node[node]
            row.update(evidence_ids=ids, adapter=vote.get("adapter"), status=entry.get("status"), evidence=facts)
            if comparable and module == "classifier" and previous_facts:
                impacts = _window_impacts(previous_facts, facts)
                if impacts:
                    row["window_fault_delta_pp"] = impacts
            if comparable and before is not None and before != after:
                subject = "puntuación de falla de sensor" if module == "classifier" else "sospecha"
                sentences.append(f"{label} cambió su {subject} de {_number(before)} a {_number(after)}.")
                if explanation := _evidence_reason(module, facts, previous_facts or {}, row.get("window_fault_delta_pp")):
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
