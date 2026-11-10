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

test("compact maps load complete evidence only when a relation is opened", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let ui;
  const calls = [];
  const edge = { edge_id: "relation-a", source: "sensor.a", target: "sensor.b", association_signed: 1, accepted: true };
  globalThis.fetch = async (url) => {
    calls.push(url);
    return { ok: true, json: async () => ({ ...edge, slice_associations: [1, null],
      pair_support: { basis: "known_state_overlap", support_days_utc: 5 } }) };
  };
  try {
    const { ContextualWorkspace } = await server.ssrLoadModule("/src/features/contextual/ContextualWorkspace.jsx");
    const mapData = { compact: true, nodes: [], edges: [edge], matrices: {
      entities: ["sensor.a", "sensor.b"], association: [[1, 1], [1, 1]], accepted: [[true, true], [true, true]] } };
    ui = await renderInteractive(ContextualWorkspace, { apiBaseUrl: "/api", mapData, results: [], summary: {} });
    assert.deepEqual(calls, []);
    await ui.event([...ui.container.querySelectorAll('.context-matrix-cell')].find((cell) => !cell.disabled), "click");
    assert.deepEqual(calls, ["/api/context/map/edges/relation-a"]);
    assert.match(ui.container.querySelector('[role="dialog"]').textContent, /Días de apoyo/);
    assert.match(ui.container.querySelector('[role="dialog"]').textContent, /Periodo 2/);
    await ui.event(ui.container.querySelector('[aria-label="Cerrar detalle"]'), "click");
    assert.equal(ui.container.querySelector('[role="dialog"]'), null);
  } finally { if (ui) await ui.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("a failed detail request explains that the compact evidence is incomplete", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let ui;
  globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({ detail: "Actualiza el mapa." }) });
  try {
    const { ContextualWorkspace } = await server.ssrLoadModule("/src/features/contextual/ContextualWorkspace.jsx");
    const edge = { edge_id: "old-relation", source: "sensor.a", target: "sensor.b", association_signed: 1, accepted: true };
    const mapData = { compact: true, nodes: [], edges: [edge], matrices: {
      entities: ["sensor.a", "sensor.b"], association: [[1, 1], [1, 1]], accepted: [[true, true], [true, true]] } };
    ui = await renderInteractive(ContextualWorkspace, { apiBaseUrl: "/api", mapData, results: [], summary: {} });
    await ui.event([...ui.container.querySelectorAll('.context-matrix-cell')].find((cell) => !cell.disabled), "click");
    assert.match(ui.container.querySelector('[role="dialog"] [role="alert"]').textContent, /evidencia completa.*Actualiza el mapa/);
  } finally { if (ui) await ui.close(); globalThis.fetch = previousFetch; await server.close(); }
});
