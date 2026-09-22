import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "vite";
import React from "react";
import { renderToString } from "react-dom/server";

test("scenario preview renders every selected entity with fault names and changed records", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previous = globalThis.localStorage;
  globalThis.localStorage = { getItem: () => JSON.stringify(["a", "b"]) };
  try {
    const { default: Scenario } = await server.ssrLoadModule("/src/ScenarioWorkspace.jsx");
    const { Comparison } = await server.ssrLoadModule("/src/InjectionWorkspace.jsx");
    const interval = { start: "2026-01-01T00:00:00Z", end: "2026-01-01T01:00:00Z" };
    const runs = [
      { ...interval, injection_id: "a", entity_id: "sensor.temperature", kind: "bias", changed_rows: 1, preview: [{ timestamp: interval.start, source_row: 2, original_state: "20", injected_state: "23", changed: true }] },
      { ...interval, injection_id: "b", entity_id: "sensor.status", kind: "state_replace", changed_rows: 1, preview: [{ timestamp: interval.start, source_row: 3, original_state: "NONE", injected_state: "ALERT", changed: true }] },
    ];
    const html = renderToString(React.createElement(Scenario, { dataset: { dataset_id: "fixture", rows: 100, entities: [{ entity_id: "sensor.temperature", kind: "continuous" }, { entity_id: "sensor.status", kind: "multistate" }] }, apiBaseUrl: "http://localhost:8000", runs, Comparison }));
    assert.match(html, /Desplazamiento/);
    assert.match(html, /Estado sustituido/);
    assert.match(html, /NONE/);
    assert.match(html, /ALERT/);
    assert.equal((html.match(/role="img"/g) || []).length, 2);
    assert.match(html, /Guardar escenario y descargar CSV etiquetado/);
    assert.doesNotMatch(html, /Hay intervalos superpuestos/);
  } finally {
    if (previous === undefined) delete globalThis.localStorage; else globalThis.localStorage = previous;
    await server.close();
  }
});
