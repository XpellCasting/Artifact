import test from "node:test";
import assert from "node:assert/strict";
import { React, createServer, renderInteractive } from "../../frontend/test-runtime.js";
import { templates } from "../../frontend/src/features/flows/flow-lab-model.js";

test("saving a repeated name loads the saved design and CSV or live evaluation waits until it is reviewed", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previous = new Map();
  const stored = new Map();
  for (const [key, value] of Object.entries({
    localStorage: { getItem: (key) => stored.get(key) || null, setItem: (key, value) => stored.set(key, value) },
    ResizeObserver: class { observe() {} unobserve() {} disconnect() {} },
    getComputedStyle: (element) => window.getComputedStyle(element),
    requestAnimationFrame: () => 1, cancelAnimationFrame: () => {},
  })) {
    previous.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  const previousFetch = globalThis.fetch;
  let view;
  try {
    const { default: FlowLab } = await server.ssrLoadModule("/src/features/flows/FlowLab.jsx");
    for (const action of ["save", "csv", "live"]) {
      const draft = { ...templates()[2], name: "Flujo existente" };
      draft.nodes[1].config.label = "Cambio sin guardar";
      const saved = structuredClone(draft);
      saved.flow_id = "saved-flow";
      saved.nodes[1].config.label = "Configuración guardada";
      stored.clear();
      stored.set("flow-lab-selected-draft-v1", JSON.stringify({ draft, savedKey: "" }));
      const requests = [];
      globalThis.fetch = async (url, options) => {
        requests.push({ url, method: options.method || "GET", body: options.body });
        let payload;
        if (url === "/api/flows" && options.method === "POST") payload = { ...saved, loaded_existing: true };
        else if (url === "/api/flows") payload = { items: [saved] };
        else if (url === "/api/flow-runs?compact=true") payload = { items: [] };
        else if (url === "/api/flow-evaluations") payload = { job: null, latest_run_id: null };
        else if (url === "/api/diagnostics/classifier-models") payload = { models: [] };
        else if (url === "/api/history/home-assistant/config") payload = { bridge_configured: true, bridge_connected: true };
        else if (url === "/api/flows/live/entities") payload = { entities: [
          { entity_id: "sensor.temperature", friendly_name: "Temperatura", unavailable_reason: null },
          { entity_id: "sensor.door", unavailable_reason: null },
          { entity_id: "sensor.other", unavailable_reason: "La entidad sensor.other no se encuentra en el historial de referencia utilizado." },
        ] };
        else if (url === "/api/flows/saved-flow/runs" || url === "/api/flows/saved-flow/live/start") payload = { job_id: "reviewed-run", status: "running" };
        else assert.fail(`Unexpected request: ${url}`);
        return { ok: true, json: async () => structuredClone(payload) };
      };
      function Host() {
        window.ResizeObserver = globalThis.ResizeObserver;
        return React.createElement(FlowLab, { apiBaseUrl: "/api", historyReady: true,
          replay: { loaded: true, labelled: true, rows: 3, filename: "test.csv" } });
      }
      context.mock.timers.enable({ apis: ["setTimeout", "setInterval"] });
      view = await renderInteractive(Host, {});
      if (action === "save") await view.event(view.container.querySelector(".fl-toolbar button.fl-primary"), "click");
      else if (action === "csv") await view.event(view.container.querySelector(".fl-future-csv button"), "click");
      else {
        await view.event(view.container.querySelectorAll('input[type="radio"]')[1], "click");
        const selector = view.container.querySelector(".fl-live-form select");
        const add = view.container.querySelector(".fl-live-form .fl-result-controls button");
        selector.value = "sensor.other";
        await view.event(selector, "change");
        await view.event(add, "click");
        assert.match(view.container.querySelector(".fl-live-form [role=alert]").textContent, /no se encuentra en el historial/);
        assert.equal(view.container.querySelector(".fl-live-form button[type=submit]").disabled, true);
        selector.value = "sensor.temperature";
        await view.event(selector, "change");
        await view.event(add, "click");
        assert.match(view.container.querySelector(".fl-entity-selection").textContent, /sensor.temperature/);
        await view.event(view.container.querySelector(".fl-live-form"), "submit");
      }
      assert.match(view.container.textContent, /Ya existe un flujo llamado «Flujo existente». Se cargó el guardado/);
      assert.equal(view.container.querySelector('select[aria-label="Abrir flujo guardado"]').value, "saved-flow");
      assert.deepEqual(JSON.parse(stored.get("flow-lab-selected-draft-v1")).draft.nodes, saved.nodes);
      assert.equal(view.container.querySelectorAll('select[aria-label="Abrir flujo guardado"] option[value="saved-flow"]').length, 1);
      assert.equal(requests.filter((item) => item.method === "POST" && /\/runs$|\/live\/start$/.test(item.url)).length, 0);
      if (action !== "save") {
        assert.match(view.container.textContent, /Revisa el diseño y vuelve a iniciar la evaluación/);
        if (action === "csv") await view.event(view.container.querySelector(".fl-future-csv button"), "click");
        else await view.event(view.container.querySelector(".fl-live-form"), "submit");
        assert.equal(requests.filter((item) => item.method === "POST" && /\/runs$|\/live\/start$/.test(item.url)).length, 1);
        if (action === "live") {
          const started = requests.find((item) => item.url.endsWith("/live/start"));
          assert.deepEqual(JSON.parse(started.body).entity_ids, ["sensor.temperature"]);
        }
      }
      await view.close(); view = undefined;
      context.mock.timers.reset();
    }
  } finally {
    if (view) await view.close();
    context.mock.timers.reset();
    globalThis.fetch = previousFetch;
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
    await server.close();
  }
});
