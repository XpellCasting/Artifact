import test from "node:test";
import assert from "node:assert/strict";
import { React, createServer, renderInteractive } from "../../frontend/test-runtime.js";
import { templates } from "../../frontend/src/features/flows/flow-lab-model.js";

test("the workspace follows a remote run and its graph, preserves the local draft and shows saved results after a restart", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previous = new Map();
  const draft = { ...templates()[2], name: "Mi borrador local" };
  const stored = new Map([["flow-lab-selected-draft-v1", JSON.stringify({ draft, savedKey: "" })]]);
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
    const flow = { ...templates()[2], flow_id: "remote-flow", name: "Prueba de otro ordenador" };
    const run = { run_id: "persisted", created_at: "2026-01-01T00:00:00Z", snapshot: { flow, source: { filename: "prueba.csv" } },
      summary: { candidate_count: 10, output_count: 3 } };
    let shared = { job: null, latest_run_id: "persisted" }, runs = [run];
    let releaseRuns, releaseModels, firstRuns = true, firstModels = true;
    const runsReady = new Promise((resolve) => { releaseRuns = resolve; });
    const modelsReady = new Promise((resolve) => { releaseModels = resolve; });
    const requests = [];
    globalThis.fetch = async (url) => {
      requests.push(url);
      let payload;
      if (url === "/api/flow-evaluations") payload = shared;
      else if (url === "/api/flows") payload = { items: [flow] };
      else if (url === "/api/flow-runs?compact=true") {
        if (firstRuns) { firstRuns = false; await runsReady; }
        payload = { items: runs };
      }
      else if (url === "/api/diagnostics/classifier-models") {
        if (firstModels) { firstModels = false; await modelsReady; }
        payload = { models: [] };
      }
      else if (url === "/api/history/home-assistant/config") payload = {};
      else if (url.startsWith("/api/flow-runs/")) payload = { items: [], total: 0, offset: 0, next_offset: null };
      else assert.fail(`Unexpected request: ${url}`);
      return { ok: true, json: async () => structuredClone(payload) };
    };
    function Host() {
      window.ResizeObserver = globalThis.ResizeObserver;
      return React.createElement(FlowLab, { apiBaseUrl: "/api", historyReady: true });
    }
    context.mock.timers.enable({ apis: ["setTimeout", "setInterval"] });
    view = await renderInteractive(Host, {});
    assert.ok(view.container.querySelector('select[aria-label="Abrir flujo guardado"] option[value="remote-flow"]'), "saved flows appear while runs and models are still loading");
    assert.match(view.container.textContent, /Cargando ejecuciones guardadas/);
    await React.act(async () => releaseRuns());
    assert.doesNotMatch(view.container.textContent, /Tu primera ejecución aparecerá aquí/);
    assert.match(view.container.querySelector('[aria-label="Resultados de flujos"]').textContent, /prueba.csv/);
    assert.ok(requests.some((url) => url.startsWith("/api/flow-runs/persisted/results")));
    await React.act(async () => releaseModels());
    shared.job = { job_id: "remote-job", status: "running", mode: "csv", phase: "evaluation", flow, processed: 4, total: 10 };
    await React.act(async () => context.mock.timers.tick(1500));
    assert.match(view.container.textContent, /Ejecución sincronizada entre dispositivos/);
    assert.match(view.container.querySelector(".fl-toolbar").textContent, /Prueba de otro ordenador/);
    assert.equal(view.container.querySelector('[role="progressbar"]').getAttribute("aria-valuenow"), "40");
    assert.equal(view.container.querySelector('button.fl-primary').disabled, true);
    const completed = { ...run, run_id: "finished" };
    runs = [completed, run];
    shared = { job: { ...shared.job, status: "completed", run: completed }, latest_run_id: "finished" };
    await React.act(async () => context.mock.timers.tick(1500));
    assert.match(view.container.textContent, /Inspección de una ejecución/);
    assert.equal(view.container.querySelector('select[aria-label="Ejecución guardada"]').value, "finished");
    const back = [...view.container.querySelectorAll("button")].find((item) => item.textContent.includes("Volver al diseño"));
    await view.event(back, "click");
    await React.act(async () => context.mock.timers.tick(1500));
    assert.equal(view.container.querySelector('input[aria-label="Nombre del flujo"]').value, draft.name);
    assert.equal(JSON.parse(stored.get("flow-lab-selected-draft-v1")).draft.name, draft.name);
    assert.equal(stored.has("flow-evaluation-job"), false);
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
