import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "../../frontend/test-runtime.js";
import { React } from "../../frontend/test-runtime.js";
import { renderToStaticMarkup } from "../../frontend/test-runtime.js";
import { annotationForPoint } from "../../frontend/src/features/review/point-labels.js";

test("imported labels colour points, while a new manual mark overrides the imported class", () => {
  const inherited = { label: "sensor_fault", fault_type: "spike", evaluable: true, es_sintetica: "true" };
  const point = { source_row: 2, timestamp: "2026-06-01T00:00:00Z", imported_label: inherited };
  assert.equal(annotationForPoint([], "sensor.a", point), inherited);
  const correction = { entity_id: "sensor.a", source_row: 2, label: "normal", evaluable: true };
  assert.equal(annotationForPoint([correction], "sensor.a", point), correction);
  assert.equal(point.imported_label.es_sintetica, "true");
});

test("evaluation table presents per-sensor numerators, denominators, false positives and partial progress", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: "custom" });
  try {
    const { LabelledEvaluationReport } = await server.ssrLoadModule("/src/features/review/LabelledEvaluation.jsx");
    const html = renderToStaticMarkup(React.createElement(LabelledEvaluationReport, { data: {
      filename: "ground_truth.csv", status: "paused", unmatched_detections: 2,
      entities: [{ entity_id: "sensor.x", normal: 100, real_detected: 0, real_faults: 10, synthetic_detected: 3, synthetic_faults: 20,
        anomalies_detected: 2, anomalies: 4, false_positives: 5, processed: 70, total: 140, unlabelled: 6 }],
    } }));
    for (const value of ["sensor.x", "0 / 10", "3 / 20", "2 / 4", "70 / 140", "Falsos positivos", "Resultados parciales", "2 alertas"]) assert.ok(html.includes(value), value);
    assert.match(html, /NULL no se consideran normales/);
  } finally { await server.close(); }
});

test("flow report separates unassessed readings and displays the partial block's last value", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: "custom" });
  try {
    const { FlowLabelReport, CalibrationReport } = await server.ssrLoadModule("/src/features/flows/FlowLab.jsx");
    const html = renderToStaticMarkup(React.createElement(FlowLabelReport, { report: { labelled: true,
      groups: [{ entity_id: "sensor.uv", expected: "Falla", synthetic: true, fault_type: "bias",
        total: 5, detected: 1, missed: 4, missed_evaluable: 1, non_evaluable: 3, indeterminate: 3,
        false_positives: 0, type_identified: 0 }] } }));
    assert.match(html, /Evaluadas sin sospecha/);
    assert.match(html, /Sin evaluación \/ sin conclusión/);
    assert.match(html, /1 \/ 5/);
    const coverage = renderToStaticMarkup(React.createElement(CalibrationReport, { models: {
      "sensor.uv": { eligible: true, reporting_policy: { mode: "on_change" },
        coverage: { calendar_days: 9.5, exposure_days: 9.5 },
        joint_calibration: { flagged_row_rate: .005, episodes_per_day: .2 },
        partitions: { final_partial_block_hours: 12 },
        last_observed_value: "0", last_observation_at: "2026-01-31T18:30:00+00:00" },
    } }));
    for (const value of ["Por cambios", "9.50", "12.00", "2026-01-31T18:30:00+00:00", "Último valor recibido", "0.50 %", "0.20", "Marcadas en calibración", "FOCuS reinicia su acumulación"]) assert.ok(coverage.includes(value), value);
  } finally { await server.close(); }
});
