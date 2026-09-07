import test from "node:test";
import assert from "node:assert/strict";
import { annotationDiagnosticProblem, faultTypeForLabel } from "./diagnostic-data.js";

const normal = { label: "normal", status: "verified", reviewer: "Revisor", evidence: "Registro independiente revisado." };

test("diagnostic sources distinguish reviewed normal data from unresolved and untyped faults", () => {
  assert.equal(annotationDiagnosticProblem(normal), "");
  assert.equal(annotationDiagnosticProblem({ ...normal, label: "environmental_anomaly" }), "");
  assert.match(annotationDiagnosticProblem({ ...normal, label: "uncertain" }), /normalidad/);
  assert.match(annotationDiagnosticProblem({ ...normal, label: "sensor_fault" }), /tipo de falla/);
  assert.equal(annotationDiagnosticProblem({ ...normal, label: "sensor_fault", fault_type: "drift" }), "");
});

test("draft sources require explicit exploratory use and documented review", () => {
  assert.match(annotationDiagnosticProblem({ ...normal, status: "draft" }), /borrador/);
  assert.equal(annotationDiagnosticProblem({ ...normal, status: "draft" }, true), "");
  assert.match(annotationDiagnosticProblem({ ...normal, status: "draft", evidence: "" }, true), /evidencia/);
});

test("changing a review class clears a fault type that no longer applies", () => {
  assert.equal(faultTypeForLabel("sensor_fault", "stuck"), "stuck");
  assert.equal(faultTypeForLabel("normal", "stuck"), null);
  assert.equal(faultTypeForLabel("environmental_anomaly", "stuck"), null);
  assert.equal(faultTypeForLabel("sensor_fault", "unsupported"), null);
});
