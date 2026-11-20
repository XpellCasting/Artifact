import test from "node:test";
import assert from "node:assert/strict";
import { createServer, renderInteractive } from "../../frontend/test-runtime.js";

test("preparation exposes dataset action and submits the selected development source", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let submitted;
  const calls = [];
  const dataset = { dataset_id: "prepared", method: "gat_lstm", partition_counts: { train_fit: 8, dev_tune: 2 } };
  globalThis.fetch = async (url, options = {}) => {
    calls.push(String(url));
    let value = {};
    if (String(url).includes("/diagnostics/status")) value = { datasets: submitted ? [dataset] : [], models: [] };
    if (String(url).includes("/diagnostics/sources")) value = { annotations: [], injections: [], training_sources: [{ source_id: "development-source", partition: "development", filename: "history.csv", rows: 10, labelled_import: true }] };
    if (String(url).endsWith("/diagnostics/build")) { submitted = JSON.parse(options.body); value = { status: "completed", progress: 100, result: dataset }; }
    if (String(url).endsWith("/diagnostics/datasets/prepared")) value = dataset;
    return new Response(JSON.stringify(value), { headers: { "Content-Type": "application/json" } });
  };
  let ui;
  try {
    const { default: Workspace } = await server.ssrLoadModule("/src/features/diagnostics/DiagnosticWorkspace.jsx");
    ui = await renderInteractive(Workspace, { apiBaseUrl: "http://localhost:8000" });
    const button = (text) => [...ui.container.querySelectorAll("button")].find((item) => item.textContent.includes(text));
    assert.equal(button("Preparación del investigador").getAttribute("aria-pressed"), "true");
    assert.equal(button("Diagnósticos"), undefined);
    assert.equal(button("Preparar dataset").disabled, false);
    await ui.event(button("Preparar dataset"), "click");
    assert.deepEqual(submitted.source_ids, ["development-source"]);
    assert.equal(submitted.method, "gat_lstm");
    assert.equal(submitted.gat_config.architecture_version, "gat_lstm_v3_segments_multiscale");
    assert.match(ui.container.textContent, /Dataset listo/);
    assert.equal(button("Descargar dataset").disabled, false);
    assert.equal(button("Entrenar"), undefined);
    assert.equal(calls.some((url) => url.includes("/diagnostics/results")), false);
  } finally {
    if (ui) await ui.close();
    globalThis.fetch = previousFetch;
    await server.close();
  }
});
