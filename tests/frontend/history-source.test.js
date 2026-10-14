import test from "node:test";
import assert from "node:assert/strict";
import { createServer, React, renderToString, renderInteractive } from "../../frontend/test-runtime.js";

test("flow evaluation opens first and links to the separate history selector", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousStorage = globalThis.localStorage;
  globalThis.localStorage = { getItem: () => null };
  try {
    const { App } = await server.ssrLoadModule("/src/app/App.jsx");
    const html = renderToString(React.createElement(App));
    assert.match(html, /module-tab active[\s\S]*?<span>Evaluación de flujos<\/span>/);
    assert.match(html, /Selector de Historial/);
    assert.match(html, /Prepara tu historial de referencia/);
    assert.match(html, /Configura el recorrido de los filtros/);
    assert.match(html, /Evaluación con eventos futuros/);
    assert.doesNotMatch(html, /Origen del historial de referencia|name="history-source"|Paso [123]/);
  } finally {
    if (previousStorage === undefined) delete globalThis.localStorage; else globalThis.localStorage = previousStorage;
    await server.close();
  }
});

test("user can select CSV, upload a file, and switch back to Home Assistant", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ url: "http://ha.local:8123", token_configured: false }) });
  let view;
  try {
    const { default: HistorySource } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    let uploaded;
    view = await renderInteractive(HistorySource, { apiBaseUrl: "/api", onUpload: (event) => { uploaded = event.target.files[0]; } });
    assert.equal(view.container.querySelector('input[value="home_assistant"]').checked, true);
    assert.ok(view.container.querySelector('input[type="password"]').required);
    await view.event(view.container.querySelector('input[value="csv"]'), "click");
    assert.equal(view.container.querySelector("form"), null);
    const input = view.container.querySelector('input[type="file"]');
    const file = { name: "history.csv", size: 120 };
    Object.defineProperty(input, "files", { value: [file] });
    await view.event(input, "change");
    assert.equal(uploaded, file);
    assert.match(view.container.textContent, /entity_id.*state.*last_changed/);
    await view.event(view.container.querySelector('input[value="home_assistant"]'), "click");
    assert.ok(view.container.querySelector("form"));
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("import uses timezone-aware dates and existing server connection without persisting a token", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ url: "http://ha.local:8123", token_configured: true }) });
  let view;
  try {
    const { default: HistorySource } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    let imported;
    view = await renderInteractive(HistorySource, { apiBaseUrl: "/api", onImport: async (payload) => { imported = payload; return { status: "completed" }; } });
    assert.equal(view.container.querySelector('input[type="password"]').required, false);
    await view.event(view.container.querySelector("form"), "submit");
    assert.equal(imported.url, "http://ha.local:8123");
    assert.equal(imported.token, "");
    assert.deepEqual(imported.entity_ids, []);
    assert.ok(imported.start.endsWith("Z") && imported.end.endsWith("Z"));
    assert.ok(new Date(imported.start) < new Date(imported.end));
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("a ready reference shows its persisted origin and offers changing it", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  try {
    const { default: HistorySource } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    for (const origin of ["home_assistant", "csv"]) {
      const html = renderToString(React.createElement(HistorySource, { apiBaseUrl: "/api",
        status: { exists: true, history_origin: origin, future_evaluation_ready: true, rows: 100, entity_count: 3 } }));
      assert.match(html, origin === "home_assistant" ? /Historial de Home Assistant/ : /Historial CSV/);
      assert.match(html, /Listo para evaluar/);
      assert.match(html, /Cambiar historial/);
      assert.doesNotMatch(html, /type="password"/);
    }
  } finally { await server.close(); }
});

test("the paired integration supplies history without URL or token fields", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ url: "http://ha.local:8123", token_configured: false,
    bridge_configured: true, bridge_connected: true }) });
  let view;
  try {
    const { default: HistorySource } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    let imported;
    view = await renderInteractive(HistorySource, { apiBaseUrl: "/api", onImport: async (value) => { imported = value; return true; } });
    assert.match(view.container.textContent, /Conectado mediante la integración/);
    assert.equal(view.container.querySelector('input[type="url"]'), null);
    assert.equal(view.container.querySelector('input[type="password"]'), null);
    await view.event(view.container.querySelector("form"), "submit");
    assert.equal(imported.connection_mode, "integration");
    assert.equal(imported.url, "");
    assert.equal(imported.token, "");
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});
