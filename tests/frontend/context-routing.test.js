import test from "node:test";
import assert from "node:assert/strict";
import { isEnvironmentalResult, contextRoutingCounts, contextRoutingLabel, contextCoherenceLabel, contextObservationLabel, preciseContextTimestamp, contextCommonCause, contextRoutingReasonLabel } from "../../frontend/src/shared/context-routing.js";

const environmental = {
  contextual_class: "environmental_anomaly",
  routing: { terminal: true, send_to_classifier: false },
};
const pending = { context_outcome: "supported", routing: { terminal: false, send_to_classifier: true } };

test("solo la decision terminal del backend se muestra como ambiental", () => {
  assert.equal(isEnvironmentalResult(environmental), true);
  assert.equal(isEnvironmentalResult(pending), false);
  assert.equal(isEnvironmentalResult({ context_outcome: "supported" }), false);
  assert.match(contextRoutingLabel(environmental), /fin en F2/);
  assert.match(contextRoutingLabel(pending), /Pendiente/);
});

test("contadores respetan el total aunque la lista este recortada", () => {
  assert.deepEqual(contextRoutingCounts([environmental, pending]), { environmental: 1, pending: 1 });
  assert.deepEqual(contextRoutingCounts([environmental], {
    environmental_anomalies: 1200, requires_classification: 8,
  }), { environmental: 1200, pending: 8 });
});

test("sin evidencia no se muestra el 0.5 neutro ni se convierte null en cero", () => {
  assert.equal(contextCoherenceLabel({ coherence_score: 0.5, coverage: { observed_neighbors: 0 } }), "No evaluable");
  assert.equal(contextCoherenceLabel({ coherence_score: null, context_score: 0 }), "No evaluable");
  assert.equal(contextCoherenceLabel({ coherence_score: 0.5, context_evaluable: false }), "No evaluable");
  assert.equal(contextCoherenceLabel({ coherence_score: 0.5, coverage: { observed_neighbors: 2 } }), "0.5000");
  assert.equal(contextCoherenceLabel({ coherence_score: 0 }), "0.0000");
  assert.equal(contextCoherenceLabel({}), "No evaluable");
});

test("evidencia temporal conserva precision y distingue arrastre de caducidad", () => {
  assert.match(contextObservationLabel({ observation_method: "last_known" }), /Ultimo valor conocido/);
  assert.match(contextObservationLabel({ missing_reason: "expired_observation" }), /caducada/);
  assert.match(contextObservationLabel({ missing_reason: "invalid_latest_observation" }), /invalido/);
  assert.equal(preciseContextTimestamp("2026-06-26T11:03:49.718Z"), "2026-06-26 11:03:49.718Z");
});

test("telemetria disponible y coincidencia no permiten inferir causa identificable", () => {
  const result = contextCommonCause({ common_cause: {
    status: "not_evaluated", candidate: false, contextual_coincidence: true,
    identifiable: false, independent_evidence_available: true,
    independent_domains_required: ["network", "power", "integration"],
  } });
  assert.equal(result.identifiable, false);
  assert.equal(result.coincidence, true);
  assert.equal(result.possible, false);
  assert.equal(result.domain, null);
  assert.match(result.label, /No evaluada/);
  assert.equal(contextCommonCause({ common_cause: { candidate: true, independent_evidence_available: true } }).identifiable, false);
});

test("sin relaciones tiene una razon diferente de cobertura insuficiente", () => {
  assert.match(contextRoutingReasonLabel({ routing: { reason: "no_learned_relations" } }), /Sin relaciones/);
  assert.match(contextRoutingReasonLabel({ routing: { reason: "insufficient_coverage" } }), /Cobertura/);
});
