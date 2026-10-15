import test from "node:test";
import assert from "node:assert/strict";
import { createServer, React, renderToString, renderInteractive } from "../../frontend/test-runtime.js";

const entities = [
  { entity_id: "sensor.temperature", friendly_name: "Temperatura", domain: "sensor", sensor_kind: "continuous" },
  { entity_id: "binary_sensor.door", domain: "binary_sensor", sensor_kind: "binary" },
];
function fakeFetch(config, calls = []) {
  return async (url, options = {}) => {
    calls.push({ url, body: options.body ? JSON.parse(options.body) : null });
    return { ok: true, json: async () => url.endsWith("/entities") ? { source: "home_assistant", entities, entity_count: 2 }
      : url.endsWith("/saved") ? { source: "csv", candidate_id: "saved", filename: "CSV guardado", entities, entity_count: 2 } : config };
  };
}
const button = (view, text) => [...view.container.querySelectorAll("button")].find((item) => item.textContent.includes(text));

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
    assert.doesNotMatch(html, /Origen del historial de referencia|name="history-source"|Paso [123]/);
  } finally {
    if (previousStorage === undefined) delete globalThis.localStorage; else globalThis.localStorage = previousStorage;
    await server.close();
  }
});

test("a CSV is inspected first and only confirmed active sensors are prepared", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch, previousXhr = globalThis.XMLHttpRequest;
  globalThis.fetch = fakeFetch({ url: "http://ha.local:8123", token_configured: false });
  const uploaded = [];
  globalThis.XMLHttpRequest = class {
    upload = {};
    open(method, url) { this.url = url; }
    setRequestHeader() {}
    send(file) {
      uploaded.push({ url: this.url, file });
      this.upload.onprogress?.({ lengthComputable: true, loaded: file.size, total: file.size });
      this.status = 200;
      this.responseText = JSON.stringify({ source: "csv", candidate_id: "candidate-one", filename: file.name, entities, entity_count: 2 });
      this.onload();
    }
  };
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    let prepared;
    view = await renderInteractive(Source, { apiBaseUrl: "/api", onPrepareCsv: async (...args) => { prepared = args; return true; } });
    await view.event(view.container.querySelector('input[value="csv"]'), "click");
    const input = view.container.querySelector('input[type="file"]');
    const file = { name: "house.csv", size: 120 };
    Object.defineProperty(input, "files", { value: [file] });
    await view.event(input, "change");
    assert.equal(uploaded[0].url, "/api/history/selection/csv");
    assert.equal(prepared, undefined);
    assert.match(view.container.textContent, /2 de 2 sensores activos/);
    await view.event(view.container.querySelector('input[aria-label="Analizar binary_sensor.door"]'), "click");
    await view.event(button(view, "Preparar filtros con"), "click");
    assert.deepEqual(prepared, ["candidate-one", ["sensor.temperature"]]);
  } finally {
    if (view) await view.close(); globalThis.fetch = previousFetch;
    if (previousXhr === undefined) delete globalThis.XMLHttpRequest; else globalThis.XMLHttpRequest = previousXhr;
    await server.close();
  }
});

test("Home Assistant discovery precedes import and confirmation preserves timezone and explicit selection", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = fakeFetch({ url: "http://ha.local:8123", token_configured: true }, calls);
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    let imported;
    view = await renderInteractive(Source, { apiBaseUrl: "/api", onImport: async (payload) => { imported = payload; return true; } });
    assert.equal(view.container.querySelector('input[type="password"]').required, false);
    await view.event(view.container.querySelector("form"), "submit");
    assert.equal(imported, undefined);
    assert.equal(calls.filter((call) => call.url.endsWith("/entities")).length, 1);
    assert.match(view.container.textContent, /2 de 2 sensores activos/);
    await view.event(button(view, "Preparar filtros con"), "click");
    assert.equal(imported.url, "http://ha.local:8123");
    assert.equal(imported.token, "");
    assert.deepEqual(imported.entity_ids, entities.map((row) => row.entity_id));
    assert.ok(imported.start.endsWith("Z") && imported.end.endsWith("Z"));
    assert.ok(new Date(imported.start) < new Date(imported.end));
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("a ready reference keeps its persisted origin while choosing another history", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    for (const origin of ["home_assistant", "csv"]) {
      const html = renderToString(React.createElement(Source, { apiBaseUrl: "/api",
        status: { exists: true, history_origin: origin, future_evaluation_ready: true, rows: 100, entity_count: 3 } }));
      assert.match(html, origin === "home_assistant" ? /Historial de Home Assistant/ : /Historial CSV/);
      assert.match(html, /Listo para evaluar/);
      assert.match(html, /Cambiar historial/);
      assert.doesNotMatch(html, /type="password"/);
    }
  } finally { await server.close(); }
});

test("the native connection discovers sensors without asking for a URL or token", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = fakeFetch({ bridge_configured: true, bridge_connected: true });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    let imported;
    view = await renderInteractive(Source, { apiBaseUrl: "/api", onImport: async (value) => { imported = value; return true; } });
    assert.equal(view.container.querySelector('input[type="url"]'), null);
    assert.equal(view.container.querySelector('input[type="password"]'), null);
    await view.event(view.container.querySelector("form"), "submit");
    assert.equal(imported, undefined);
    await view.event(button(view, "Preparar filtros con"), "click");
    assert.equal(imported.connection_mode, "integration");
    assert.equal(imported.url, "");
    assert.equal(imported.token, "");
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("a saved CSV also waits for sensor selection before preparing", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = fakeFetch({ bridge_configured: true, bridge_connected: true });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    let prepared;
    view = await renderInteractive(Source, { apiBaseUrl: "/api", status: { real_history_available: true, selection_required: true },
      onPrepareCsv: async (...args) => { prepared = args; return true; } });
    assert.equal(view.container.querySelector('input[value="csv"]').checked, true);
    assert.match(view.container.textContent, /CSV guardado pendiente de selección/);
    await view.event(view.container.querySelector('input[value="csv"]'), "click");
    await view.event(button(view, "Revisar CSV guardado"), "click");
    assert.equal(prepared, undefined);
    await view.event(button(view, "Desactivar todos"), "click");
    assert.equal(button(view, "Preparar filtros con").disabled, true);
    await view.event(button(view, "Activar todos"), "click");
    await view.event(button(view, "Preparar filtros con"), "click");
    assert.deepEqual(prepared, ["saved", entities.map((row) => row.entity_id)]);
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});
