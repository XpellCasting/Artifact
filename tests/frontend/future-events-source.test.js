import test from "node:test";
import assert from "node:assert/strict";
import { createServer, renderInteractive } from "../../frontend/test-runtime.js";

test("future evaluation switches from CSV to live events and starts without an uploaded CSV", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ url: "http://ha.local:8123", token_configured: true }) });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    let connection;
    view = await renderInteractive(Source, { apiBaseUrl: "/api", canRun: true, dirty: true,
      onStartLive: async (value) => { connection = value; return true; } });
    assert.equal(view.container.querySelectorAll('input[type="file"]').length, 2);
    assert.equal(view.container.querySelector("button").disabled, true);
    await view.event(view.container.querySelectorAll('input[type="radio"]')[1], "click");
    assert.equal(view.container.querySelectorAll('input[type="file"]').length, 0);
    assert.equal(view.container.querySelector('input[type="password"]').required, false);
    assert.equal(view.container.querySelector('button[type="submit"]').disabled, false);
    await view.event(view.container.querySelector("form"), "submit");
    assert.deepEqual(connection, { url: "http://ha.local:8123", token: "", entity_ids: [] });
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("restored live reception displays results and keeps stop enabled while the flow is busy", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ url: "http://ha.local:8123", token_configured: true }) });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    let stopped = false;
    view = await renderInteractive(Source, { apiBaseUrl: "/api", busy: true, canRun: true,
      job: { mode: "home_assistant", status: "running", processed: 4, connection: { status: "connected", received: 5 },
        recent_results: [{ observation_id: "one", entity_id: "sensor.temperature", timestamp: "2026-01-01T01:00:00Z", final_decision: "suspicion" }] },
      onStopLive: () => { stopped = true; } });
    assert.match(view.container.textContent, /5 recibidos · 4 evaluados/);
    assert.match(view.container.textContent, /sensor.temperature/);
    assert.equal(view.container.querySelector('input[type="password"]'), null);
    const stop = [...view.container.querySelectorAll("button")].find((button) => button.textContent.includes("Detener recepción"));
    assert.equal(stop.disabled, false);
    await view.event(stop, "click");
    assert.equal(stopped, true);
    assert.equal(view.container.querySelector('input[type="radio"]').matches(":disabled"), true);
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("live reception requires a prepared reference and a valid flow", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ url: "http://ha.local:8123", token_configured: false }) });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    view = await renderInteractive(Source, { apiBaseUrl: "/api", canRun: false });
    await view.event(view.container.querySelectorAll('input[type="radio"]')[1], "click");
    assert.equal(view.container.querySelector('button[type="submit"]').disabled, true);
    assert.equal(view.container.querySelector('input[type="password"]').required, true);
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("live events use the integration automatically and wait for it when disconnected", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    for (const connected of [true, false]) {
      globalThis.fetch = async () => ({ ok: true, json: async () => ({ url: "http://ha.local:8123", token_configured: false,
        bridge_configured: true, bridge_connected: connected }) });
      let started;
      view = await renderInteractive(Source, { apiBaseUrl: "/api", canRun: true, onStartLive: async (value) => { started = value; return true; } });
      await view.event(view.container.querySelectorAll('input[type="radio"]')[1], "click");
      assert.equal(view.container.querySelector('input[type="url"]'), null);
      assert.equal(view.container.querySelector('input[type="password"]'), null);
      assert.equal(view.container.querySelector('button[type="submit"]').disabled, !connected);
      if (connected) {
        await view.event(view.container.querySelector("form"), "submit");
        assert.deepEqual(started, { connection_mode: "integration", url: "", token: "", entity_ids: [] });
      } else assert.match(view.container.textContent, /Esperando la conexión/);
      await view.close(); view = undefined;
    }
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});
