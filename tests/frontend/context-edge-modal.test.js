import test from "node:test";
import assert from "node:assert/strict";
import { createServer, React, renderToString, renderInteractive } from "../../frontend/test-runtime.js";

test("association detail renders missing stability periods without crashing", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  try {
    const { ContextEdgeModal } = await server.ssrLoadModule("/src/features/contextual/ContextualWorkspace.jsx");
    const edge = { source: "sensor.a", target: "sensor.b", association_signed: 1.,
      accepted: true, expected_response: { slope: 1., intercept: 0. },
      relation_view: "state", pair_support: { basis: "known_state_overlap", coincident_record_coverage: .01,
        known_time_coverage: .95, fresh_time_coverage: .15, independent_observations: 116, support_days_utc: 5 },
      view_candidates: [{ relation_view: "changes", threshold_eligible: false, rejection_reasons: ["insufficient_support_days"] }],
      slice_associations: [1., null, 1., null],
      window_candidates: [{ window_seconds: 300, threshold_eligible: false }] };
    const html = renderToString(React.createElement(ContextEdgeModal, { edge, onClose: () => {} }));
    assert.match(html, /Detalle de la arista contextual/);
    assert.match(html, /Periodo 2/);
    assert.match(html, /Sin datos suficientes/);
    assert.match(html, /Cobertura comparable/);
    assert.match(html, /Registros coincidentes/);
    assert.match(html, /Días de apoyo/);
    assert.match(html, /Cambios \/ transiciones/);
  } finally { await server.close(); }
});

test("clicking a 1.00 cell opens its detail and closes normally", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  let ui;
  try {
    const { ContextualWorkspace } = await server.ssrLoadModule("/src/features/contextual/ContextualWorkspace.jsx");
    const edge = { source: "sensor.a", target: "sensor.b", association_signed: 1.,
      accepted: true, slice_associations: [1., null], expected_sign: "positive" };
    const mapData = { map_version: "contextual-spatiotemporal-v6", nodes: [], edges: [edge],
      matrices: { entities: ["sensor.a", "sensor.b"], association: [[1.,1.],[1.,1.]], accepted: [[true,true],[true,true]] } };
    ui = await renderInteractive(ContextualWorkspace, { mapData, results: [], summary: {}, onReload: () => {}, onRebuild: () => {} });
    const cell = [...ui.container.querySelectorAll('.context-matrix-cell')].find((item) => !item.disabled);
    assert.equal(cell.textContent, "1.00");
    await ui.event(cell, "click");
    assert.ok(ui.container.querySelector('[role="dialog"]'));
    await ui.event(ui.container.querySelector('[aria-label="Cerrar detalle"]'), "click");
    assert.equal(ui.container.querySelector('[role="dialog"]'), null);
    assert.ok(ui.container.querySelector('.context-matrix'));
  } finally { if (ui) await ui.close(); await server.close(); }
});
