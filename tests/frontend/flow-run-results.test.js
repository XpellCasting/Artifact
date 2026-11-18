import test from "node:test";
import assert from "node:assert/strict";
import { React, createServer, renderInteractive } from "../../frontend/test-runtime.js";

const run = (id) => ({
  run_id: id, created_at: "2026-01-01T00:00:00Z",
  snapshot: { flow: { name: "Flujo de prueba" }, source: { filename: `${id}.csv` } },
});
const metrics = {
  support: { labelled: 2, unlabelled: 4 },
  diagnosis: {
    macro_f1: { value: null, missing_classes: ["Falla"] },
    coverage: { value: .5, numerator: 1 },
    no_diagnosis: { value: .5, numerator: 1, by_decision: { no_alert: 1 } },
    matrix: {
      normal: { normal: 1, anomalia: 0, Falla: 0, no_diagnosis: 0 },
      anomalia: { normal: 0, anomalia: 0, Falla: 0, no_diagnosis: 1 },
      Falla: { normal: 0, anomalia: 0, Falla: 0, no_diagnosis: 0 },
    },
    by_class: { normal: { support: 1 }, anomalia: { support: 1 }, Falla: { support: 0 } },
  },
};

test("run tabs load on demand, preserve controls, retry and ignore an obsolete response", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const fetchBefore = globalThis.fetch;
  let view;
  try {
    const { default: FlowRunResults } = await server.ssrLoadModule("/src/features/flows/FlowRunResults.jsx");
    const requests = [];
    let finishOld;
    const oldResponse = new Promise((resolve) => { finishOld = resolve; });
    globalThis.fetch = async (url, { signal }) => {
      requests.push({ url, signal });
      if (url.endsWith("/old/metrics")) return oldResponse;
      if (requests.filter((request) => request.url === url).length === 1) {
        return { ok: false, status: 503, json: async () => ({ detail: "Error de prueba" }) };
      }
      return { ok: true, json: async () => metrics };
    };
    function Host() {
      const [current, setCurrent] = React.useState("old");
      const [tab, setTab] = React.useState("detail");
      return React.createElement(React.Fragment, null,
        React.createElement("button", { onClick: () => { setCurrent("new"); setTab("summary"); } }, "Cambiar ejecución"),
        React.createElement(FlowRunResults, { key: current, run: run(current), apiBaseUrl: "/api", activeTab: tab, onTabChange: setTab },
          React.createElement("input", { "aria-label": "Detalle de prueba", defaultValue: "Salida" })));
    }
    view = await renderInteractive(Host, {});
    const activePanel = () => view.container.querySelector('[role="tabpanel"]:not([hidden])');
    const chooseTab = (label) => view.event(
      [...view.container.querySelectorAll('[role="tab"]')].find((tab) => tab.textContent === label), "click");
    assert.equal(requests.length, 0);
    assert.equal(view.container.querySelectorAll('[role="tab"]').length, 7);
    await chooseTab("Resumen");
    assert.match(activePanel().textContent, /Cargando métricas/);
    await view.event(view.container.querySelector("button"), "click");
    assert.equal(requests[0].signal.aborted, true);
    assert.match(activePanel().textContent, /Error de prueba/);
    await view.event(activePanel().querySelector("button"), "click");
    assert.match(activePanel().textContent, /Lecturas etiquetadas2/);
    await React.act(async () => finishOld({ ok: true, json: async () => ({ ...metrics, support: { labelled: 99 } }) }));
    assert.doesNotMatch(activePanel().textContent, /99/);

    await chooseTab("Matriz de confusión");
    const select = activePanel().querySelector("select");
    assert.equal(select.value, "count");
    assert.equal(activePanel().querySelector("tbody td").textContent, "1");
    select.value = "percent";
    await view.event(select, "change");
    assert.match(activePanel().querySelector("tbody td").textContent, /100\s*%/);
    await chooseTab("Detalle");
    activePanel().querySelector("input").value = "Otro nodo";
    await chooseTab("Matriz de confusión");
    assert.equal(activePanel().querySelector("select").value, "percent");
    await chooseTab("Detalle");
    assert.equal(activePanel().querySelector("input").value, "Otro nodo");
    assert.equal(requests.length, 3);
  } finally {
    if (view) await view.close();
    globalThis.fetch = fetchBefore;
    await server.close();
  }
});
