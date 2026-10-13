export function isEnvironmentalResult(result = {}) {
  return result.contextual_class === "environmental_anomaly"
    && result.routing?.terminal === true
    && result.routing?.send_to_classifier === false;
}

export function contextRoutingLabel(result = {}) {
  if (isEnvironmentalResult(result)) return "Anomalia ambiental · fin en F2";
  if (result.routing?.send_to_classifier === true) return "Pendiente de clasificacion";
  return "Ruta no informada";
}

export function contextRoutingReasonLabel(result = {}) {
  const labels = {
    context_supported: "Cambio respaldado por el contexto",
    ambiguous_common_cause: "Posible causa comun sin evidencia independiente",
    stale_or_unknown_context_map: "Mapa obsoleto o sin vigencia verificable",
    contradicted: "El contexto contradice el cambio",
    unexplained: "El contexto no explica el cambio",
    insufficient_coverage: "Cobertura contextual insuficiente",
    no_learned_relations: "Sin relaciones aprendidas; contexto no evaluable",
    missing_contextual_result: "Falta el resultado contextual",
  };
  return labels[result.routing?.reason] || result.routing?.reason || "Sin decision de ruta";
}

export function contextRoutingCounts(results = [], summary = {}) {
  const environmental = results.filter(isEnvironmentalResult).length;
  const pending = results.filter((result) => result.routing?.send_to_classifier === true).length;
  return {
    environmental: summary?.environmental_anomalies ?? environmental,
    pending: summary?.requires_classification ?? pending,
  };
}

export function contextCoherenceLabel(result = {}) {
  // Tambien cubre resultados antiguos que devolvian 0.5 sin observaciones.
  if (result.context_evaluable === false || result.coverage?.observed_neighbors === 0
      || result.coverage?.ratio === 0 || result.coherence_score === null) {
    return "No evaluable";
  }
  const value = result.coherence_score ?? result.contextual_score ?? result.context_score ?? result.score;
  return value !== undefined && value !== "" && Number.isFinite(Number(value))
    ? Number(value).toFixed(4) : "No evaluable";
}

export function contextObservationLabel(item = {}) {
  const reasons = {
    no_previous_observation: "Sin lectura anterior",
    invalid_latest_observation: "Ultimo reporte invalido",
    expired_observation: "Lectura caducada",
  };
  return reasons[item.missing_reason] || {
    last_known: "Ultimo valor conocido",
    bucket_median: "Mediana de ventana disponible",
    state_as_of: "Estado disponible hasta la referencia",
  }[item.observation_method] || "";
}

export function preciseContextTimestamp(value) {
  return value ? String(value).replace("T", " ") : "-";
}

export function contextCommonCause(result = {}) {
  const value = result.common_cause || {};
  const yes = (v) => v === true || v === "true" || v === 1;
  const coincidence = yes(value.contextual_coincidence);
  const independentEvidence = yes(value.independent_evidence_available ?? value.independent_signal_present);
  // Disponibilidad de telemetria no equivale a atribucion. Respetar un false
  // explicito del backend y nunca deducir identificabilidad en el navegador.
  const identifiable = yes(value.identifiable ?? result.common_cause_identifiable);
  return {
    key: identifiable ? "identifiable" : "unknown",
    label: identifiable ? "Identificable" : "No evaluada · falta evidencia causal",
    possible: yes(value.candidate), coincidence, identifiable, independentEvidence,
    score: value.score,
    domain: value.domain || value.failure_domain || null,
    detail: value.interpretation || value.detail || value.explanation,
    entities: Array.isArray(value.concurrent_entities) ? value.concurrent_entities : [],
    windowSeconds: value.window_seconds,
  };
}
