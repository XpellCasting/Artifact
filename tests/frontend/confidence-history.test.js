import test from "node:test";
import assert from "node:assert/strict";
import { React, createServer, renderInteractive } from "../../frontend/test-runtime.js";
import { confidenceChartData, confidencePercent } from "../../frontend/src/features/flows/confidence-history-model.js";

test("confidence percentages preserve zero and reject unavailable or out-of-range values", () => {
  assert.match(confidencePercent(0), /^0\s*%$/);
  assert.match(confidencePercent(1), /^100\s*%$/);
  assert.match(confidencePercent(.93), /^93\s*%$/);
  for (const value of [null, undefined, NaN, Infinity, -.1, 1.1, "0.9", true]) {
    assert.equal(confidencePercent(value), "No evaluable");
  }
});

test("chart data stays chronological and preserves unknown confidence as gaps", () => {
  const data = confidenceChartData([
    { timestamp: "2026-11-18T01:02:00Z", value: 100 },
    { timestamp: "2026-11-18T01:00:00Z", value: 0 },
    { timestamp: "2026-11-18T01:01:00Z", value: null },
    { timestamp: "not-a-date", value: 100 },
    { timestamp: "2026-11-18T01:03:00Z", value: 101 },
  ]);
  assert.deepEqual(data.values, [0, null, 100, null]);
  assert.deepEqual(data.timestamps.map((date) => date.toISOString()), [
    "2026-11-18T01:00:00.000Z", "2026-11-18T01:01:00.000Z", "2026-11-18T01:02:00.000Z", "2026-11-18T01:03:00.000Z",
  ]);
});

test("saved confidence history isolates runs, fixes graph bounds, retries and pages per sensor", async () => {
  const chartId = "\0confidence-chart-fixture";
  const server = await createServer({ ssr: { noExternal: ["@mui/x-charts"] }, plugins: [{
    name: "confidence-chart-fixture",
    enforce: "pre",
    resolveId(id) { if (id === "@mui/x-charts/LineChart") return chartId; },
    load(id) {
      if (id === chartId) return `import React from "react";
        export function LineChart(props) { return React.createElement("div", {
          "data-chart": "confidence", "data-min": props.yAxis[0].min, "data-max": props.yAxis[0].max,
          "data-values": JSON.stringify(props.series[0].data), "data-connect-nulls": String(props.series[0].connectNulls)
        }); }`;
    },
  }] });
  const beforeFetch = globalThis.fetch;
  let view, finishOld;
  const calls = [];
  const old = new Promise((resolve) => { finishOld = resolve; });
  const run = (id) => ({ run_id: id, snapshot: { execution_mode: "causal_future" } });
  const page = (source = "sensor.a", offset = 0) => ({
    run_id: "new", source_entity_id: source, entity_id: `confianza.${source.replace(".", "_")}`,
    entities: ["sensor.a", "sensor.b"].map((source_entity_id) => ({ source_entity_id })),
    points: offset ? [{ timestamp: "2026-11-18T02:00:00Z", value: 35 }]
      : [0, null, 100].map((value, index) => ({ timestamp: `2026-11-18T01:0${index}:00Z`, value })),
    offset, total: 1001, next_offset: offset ? null : 1000,
  });
  try {
    const { default: History } = await server.ssrLoadModule("/src/features/flows/FlowConfidenceHistory.jsx");
    globalThis.fetch = async (url, options) => {
      calls.push({ url, signal: options.signal });
      if (url.includes("/old/")) return old;
      const parsed = new URL(url, "http://localhost");
      const offset = Number(parsed.searchParams.get("offset"));
      if (offset === 1000 && calls.filter((call) => call.url === url).length === 1) {
        return { ok: false, status: 503, json: async () => ({ detail: "Error de prueba" }) };
      }
      return { ok: true, json: async () => page(parsed.searchParams.get("entity_id") || "sensor.a", offset) };
    };
    function Host() {
      const [current, setCurrent] = React.useState("old");
      return React.createElement(React.Fragment, null,
        React.createElement("button", { onClick: () => setCurrent("new") }, "Cambiar ejecución"),
        React.createElement(History, { apiBaseUrl: "/api", run: run(current) }));
    }
    view = await renderInteractive(Host, {});
    assert.match(view.container.textContent, /Cargando historial/);
    await view.event(view.container.querySelector("button"), "click");
    assert.equal(calls[0].signal.aborted, true);
    let chart = view.container.querySelector('[data-chart="confidence"]');
    assert.equal(chart.dataset.min, "0");
    assert.equal(chart.dataset.max, "100");
    assert.equal(chart.dataset.values, "[0,null,100]");
    assert.equal(chart.dataset.connectNulls, "false");
    assert.match(view.container.textContent, /fechas originales de las lecturas del CSV/);
    await React.act(async () => finishOld({ ok: true, json: async () => ({ ...page("sensor.stale"), run_id: "old" }) }));
    assert.doesNotMatch(view.container.textContent, /stale/);
    await view.event([...view.container.querySelectorAll("button")].find((button) => button.textContent === "Siguiente"), "click");
    assert.match(view.container.textContent, /Error de prueba/);
    await view.event([...view.container.querySelectorAll("button")].find((button) => button.textContent === "Reintentar"), "click");
    assert.equal(view.container.querySelector('[data-chart="confidence"]').dataset.values, "[35]");
    const select = view.container.querySelector("select");
    select.value = "sensor.b";
    await view.event(select, "change");
    assert.match(calls.at(-1).url, /entity_id=sensor.b/);
    assert.match(calls.at(-1).url, /offset=0/);
    assert.match(view.container.textContent, /confianza.sensor_b/);
  } finally {
    if (view) await view.close();
    globalThis.fetch = beforeFetch;
    await server.close();
  }
});
