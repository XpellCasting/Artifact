import test from "node:test";
import assert from "node:assert/strict";
import { createServer, React, renderToString, renderInteractive } from "../../frontend/test-runtime.js";

const button = (root, text) => [...root.querySelectorAll("button")].find((item) => item.textContent.includes(text));

test("connection checking, absent integration, disconnect and reconnect gate history operations without losing the selection", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch, previousTimeout = globalThis.setTimeout, previousClear = globalThis.clearTimeout;
  const timer = {};
  let refresh, view, imported;
  let connection = { bridge_configured: false, bridge_connected: false };
  let failed = false;
  const calls = [];
  globalThis.setTimeout = (fn, delay, ...args) => delay === 10000 ? (refresh = fn, timer) : previousTimeout(fn, delay, ...args);
  globalThis.clearTimeout = (id) => { if (id !== timer) previousClear(id); };
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    if (failed) throw new Error("API unavailable");
    return { ok: true, json: async () => url.endsWith("/config") ? connection : {
      source: "home_assistant", areas_available: true, entities: [{ entity_id: "sensor.one", sensor_kind: "continuous", domain: "sensor", area_id: null, area_name: null }], entity_count: 1 } };
  };
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/history/HistorySource.jsx");
    assert.match(renderToString(React.createElement(Source, { apiBaseUrl: "/api" })), /Comprobando conexión/);
    view = await renderInteractive(Source, { apiBaseUrl: "/api", onImport: async (value) => { imported = value; return true; } });
    assert.match(view.container.textContent, /Integración no configurada/);
    assert.equal(view.container.querySelector('input[type="url"],input[type="password"]'), null);
    await view.event(view.container.querySelector("form"), "submit");
    assert.equal(calls.filter((call) => call.url.endsWith("/entities")).length, 0);
    connection = { bridge_configured: true, bridge_connected: false };
    await React.act(async () => { await refresh(); });
    assert.match(view.container.textContent, /Desconectada/);
    assert.equal(button(view.container, "Consultar sensores").disabled, true);
    connection = { bridge_configured: true, bridge_connected: true };
    await React.act(async () => { await refresh(); });
    assert.match(view.container.textContent, /Conectada/);
    await view.event(view.container.querySelector("form"), "submit");
    await view.event(view.container.querySelector('input[type="checkbox"]'), "click");
    assert.equal(button(view.container, "Preparar filtros con").disabled, false);
    failed = true;
    await React.act(async () => { await refresh(); });
    assert.match(view.container.textContent, /Desconectada/);
    assert.equal(view.container.querySelector('input[type="checkbox"]').checked, true);
    assert.equal(button(view.container, "Preparar filtros con").disabled, true);
    await view.event(button(view.container, "Preparar filtros con"), "click");
    assert.equal(imported, undefined);
    failed = false;
    await React.act(async () => { await refresh(); });
    await view.event(button(view.container, "Preparar filtros con"), "click");
    assert.deepEqual(imported.entity_ids, ["sensor.one"]);
    assert.deepEqual(Object.keys(imported).sort(), ["end", "entity_ids", "start"]);
  } finally {
    if (view) await view.close();
    globalThis.fetch = previousFetch; globalThis.setTimeout = previousTimeout; globalThis.clearTimeout = previousClear;
    await server.close();
  }
});

test("live start is blocked for an unconfigured or unreachable integration, including direct form submission", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    for (const config of [{ bridge_configured: false, bridge_connected: false }, null]) {
      globalThis.fetch = async () => { if (config === null) throw new Error("unreachable"); return { ok: true, json: async () => config }; };
      let started = false;
      view = await renderInteractive(Source, { apiBaseUrl: "/api", canRun: true, onStartLive: async () => { started = true; } });
      await view.event(view.container.querySelectorAll('input[type="radio"]')[1], "click");
      assert.match(view.container.textContent, config ? /Integración no configurada/ : /Desconectada/);
      assert.equal(view.container.querySelector('input[type="url"],input[type="password"]'), null);
      assert.equal(view.container.querySelector('button[type="submit"]').disabled, true);
      await view.event(view.container.querySelector("form"), "submit");
      assert.equal(started, false);
      await view.close(); view = undefined;
    }
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});
