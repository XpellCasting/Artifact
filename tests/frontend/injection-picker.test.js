import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "../../frontend/test-runtime.js";
import { React } from "../../frontend/test-runtime.js";
import { renderToString } from "../../frontend/test-runtime.js";

test("original preview exposes exact microsecond timestamps and hides synthetic comparison fields", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: "custom" });
  try {
    const { Comparison } = await server.ssrLoadModule("/src/features/injections/InjectionWorkspace.jsx");
    const time = "2026-06-06T12:34:56.123456Z";
    const html = renderToString(React.createElement(Comparison, { run: { entity_id: "sensor.a", start: time, end: time, preview: [{ source_row: 0, timestamp: time, original_state: "NONE", injected_state: "NONE" }] }, continuous: false, originalOnly: true, onSelectTime: () => {}, title: "Fragmento original" }));
    assert.match(html, /Seleccionar 2026-06-06T12:34:56.123456Z, valor NONE/);
    assert.match(html, /role="button"/);
    assert.doesNotMatch(html, /Fila de origen|Con inyección|Comparación del caso guardado/);
  } finally { await server.close(); }
});
