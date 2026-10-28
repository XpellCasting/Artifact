import test from "node:test";
import assert from "node:assert/strict";
import { createServer, renderInteractive } from "../../frontend/test-runtime.js";

test("CSV accepts only labelled uploads, while live events start without an uploaded CSV", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ bridge_configured: true, bridge_connected: true }) });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    let connection, uploaded;
    view = await renderInteractive(Source, { apiBaseUrl: "/api", canRun: true, dirty: true,
      onLabelledUpload: (event) => { uploaded = event.target.files[0]; },
      onStartLive: async (value) => { connection = value; return true; } });
    assert.equal(view.container.querySelectorAll('input[type="file"]').length, 1);
    assert.doesNotMatch(view.container.textContent, /Cargar eventos futuros/);
    const input = view.container.querySelector('input[type="file"]');
    const file = { name: "ground_truth.csv" };
    Object.defineProperty(input, "files", { value: [file] });
    await view.event(input, "change");
    assert.equal(uploaded, file);
    assert.equal(view.container.querySelector("button").disabled, true);
    await view.event(view.container.querySelectorAll('input[type="radio"]')[1], "click");
    assert.equal(view.container.querySelectorAll('input[type="file"]').length, 0);
    assert.equal(view.container.querySelector('input[type="password"]'), null);
    assert.equal(view.container.querySelector('button[type="submit"]').disabled, false);
    await view.event(view.container.querySelector("form"), "submit");
    assert.deepEqual(connection, { entity_ids: [] });
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("CSV evaluation requires reference labels, including previously uploaded files", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({}) });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    for (const labelled of [false, true]) {
      let evaluated = false;
      view = await renderInteractive(Source, { apiBaseUrl: "/api", canRun: true,
        replay: { loaded: true, labelled, filename: "events.csv", rows: 3 }, onEvaluate: () => { evaluated = true; } });
      const button = view.container.querySelector('button');
      assert.equal(button.disabled, !labelled);
      await view.event(button, "click");
      assert.equal(evaluated, labelled);
      await view.close(); view = undefined;
    }
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("restored live reception displays results and keeps stop enabled while the flow is busy", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ bridge_configured: true, bridge_connected: true }) });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    let stopped = false;
    view = await renderInteractive(Source, { apiBaseUrl: "/api", busy: true, canRun: true,
      job: { mode: "home_assistant", status: "running", processed: 4, connection: { status: "connected", received: 5 },
        recent_results: [
          { observation_id: "one", entity_id: "sensor.temperature", timestamp: "2026-01-01T01:00:00Z", final_decision: "suspicion" },
          { observation_id: "two", entity_id: "sensor.humidity", timestamp: "2026-01-01T01:01:00Z", final_decision: "fault_compatible",
            evidence: [{ status: "compatible", data: { fault_type: "drift", fault_type_status: "identified" } }] },
          { observation_id: "three", entity_id: "sensor.pressure", timestamp: "2026-01-01T01:02:00Z", final_decision: "fault_compatible" },
          { observation_id: "four", entity_id: "sensor.light", timestamp: "2026-01-01T01:03:00Z", final_decision: "normal" },
        ] },
      onStopLive: () => { stopped = true; } });
    assert.match(view.container.textContent, /5 recibidos · 4 evaluados/);
    assert.match(view.container.textContent, /sensor.temperature/);
    assert.equal(view.container.querySelectorAll(".fl-output-alert").length, 0);
    assert.match(view.container.textContent, /Compatible con falla/);
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
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ bridge_configured: false, bridge_connected: false }) });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    view = await renderInteractive(Source, { apiBaseUrl: "/api", canRun: false });
    await view.event(view.container.querySelectorAll('input[type="radio"]')[1], "click");
    assert.equal(view.container.querySelector('button[type="submit"]').disabled, true);
    assert.equal(view.container.querySelector('input[type="password"]'), null);
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
        assert.deepEqual(started, { entity_ids: [] });
      } else assert.match(view.container.textContent, /Desconectada/);
      await view.close(); view = undefined;
    }
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("live reception explains idle and reconnecting states and keeps the manual stop available", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ bridge_configured: true, bridge_connected: true }) });
  let view;
  try {
    const { default: Source } = await server.ssrLoadModule("/src/features/flows/FutureEventsSource.jsx");
    for (const [connection, polling_error, expected] of [
      [{ status: "connected", received: 0 }, null, /Esperando cambios de estado/],
      [{ status: "reconnecting", received: 2, last_error: "Se perdió la conexión", interruptions: 1 }, null, /Reconectando con Home Assistant/],
      [{ status: "connected", received: 2 }, "No se pudo consultar la API", /Recuperando comunicación con Artifact/],
    ]) {
      view = await renderInteractive(Source, { apiBaseUrl: "/api", busy: true, job: {
        mode: "home_assistant", status: "running", phase: "evaluation", connection, polling_error } });
      assert.match(view.container.textContent, expected);
      const stop = [...view.container.querySelectorAll("button")].find((button) => button.textContent.includes("Detener recepción"));
      assert.equal(stop.disabled, false);
      await view.close(); view = undefined;
    }
    view = await renderInteractive(Source, { apiBaseUrl: "/api", canRun: true,
      error: "No se puede iniciar la evaluación: el filtro clasificador no tiene un modelo disponible. Selecciona un modelo válido o desactiva ese filtro." });
    assert.match(view.container.querySelector('[role="alert"]').textContent, /clasificador.*Selecciona un modelo válido/);
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});
