import test from "node:test";
import assert from "node:assert/strict";
import { createServer, renderInteractive } from "../../frontend/test-runtime.js";

test("preparation exposes dataset action and submits the selected development source", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let submitted;
  globalThis.fetch = async (url, options = {}) => {
    let value = {};
    if (String(url).includes("/diagnostics/status")) value = { datasets: [], models: [] };
    if (String(url).includes("/diagnostics/sources")) value = { annotations: [], injections: [], training_sources: [{ source_id: "development-source", partition: "development", filename: "history.csv", rows: 10, labelled_import: true }] };
    if (String(url).endsWith("/diagnostics/build")) { submitted = JSON.parse(options.body); value = { status: "completed", progress: 100, result: {} }; }
    return new Response(JSON.stringify(value), { headers: { "Content-Type": "application/json" } });
  };
  let ui;
  try {
    const { default: Workspace } = await server.ssrLoadModule("/src/features/diagnostics/DiagnosticWorkspace.jsx");
    ui = await renderInteractive(Workspace, { apiBaseUrl: "http://localhost:8000" });
    const button = (text) => [...ui.container.querySelectorAll("button")].find((item) => item.textContent.includes(text));
    await ui.event(button("Preparación del investigador"), "click");
    assert.equal(button("Preparar dataset").disabled, false);
    assert.match(ui.container.textContent, /Dataset y entrenamiento/);
    await ui.event(button("Preparar dataset"), "click");
    assert.deepEqual(submitted.source_ids, ["development-source"]);
    assert.equal(submitted.method, "gat_lstm");
    assert.equal(submitted.gat_config.architecture_version, "gat_lstm_v3_segments_multiscale");
    assert.match(ui.container.textContent, /Dataset listo/);
  } finally {
    if (ui) await ui.close();
    globalThis.fetch = previousFetch;
    await server.close();
  }
});
