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
