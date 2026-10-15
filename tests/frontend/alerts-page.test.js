import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { React, createServer, renderInteractive } from "../../frontend/test-runtime.js";

const fault = { alert_id: 3, created_at: "2026-01-01T01:00:00Z", result: {
  entity_id: "sensor.temperature", timestamp: "2026-01-01T01:00:00Z", final_decision: "fault_compatible",
  evidence: [{ status: "compatible", data: { fault_type: "drift", fault_type_status: "identified" } }],
} };
const response = (data) => ({ ok: true, json: async () => data });
const button = (view, text) => [...view.container.querySelectorAll("button")].find((item) => item.textContent.includes(text));

test("the independent alerts page shows faults and anomalies and polls without an open evaluation", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view;
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(url);
    return response({ items: calls.length === 1 ? [fault] : [fault,
      { ...fault, alert_id: 2, result: { ...fault.result, entity_id: "sensor.humidity", final_decision: "environmental" } },
      { ...fault, alert_id: 1, result: { ...fault.result, entity_id: "sensor.pressure", evidence: [] } },
    ], next_before_id: null });
  };
  try {
    const { default: Page } = await server.ssrLoadModule("/src/features/alerts/AlertsPage.jsx");
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Page, { apiBaseUrl: "/api" });
    assert.equal(view.container.querySelector("h1").textContent, "Artifact-Alertas");
    assert.equal(document.title, "Artifact-Alertas");
    assert.match(view.container.textContent, /sensor.temperature/);
    assert.equal(view.container.querySelector(".fl-output-alert-fault").textContent, "Falla · Posible falla: Deriva");
    await React.act(async () => context.mock.timers.tick(3000));
    assert.equal(view.container.querySelector(".fl-output-alert-anomaly").textContent, "Se encontró una anomalía.");
    assert.equal(view.container.querySelectorAll(".fl-output-alert-fault")[1].textContent, "Falla");
    assert.deepEqual(calls, ["/api/flow-alerts?limit=50", "/api/flow-alerts?limit=50"]);
    await view.close(); view = undefined;
    await React.act(async () => context.mock.timers.tick(9000));
    assert.equal(calls.length, 2);
  } finally { if (view) await view.close(); context.mock.timers.reset(); globalThis.fetch = previousFetch; await server.close(); }
});

test("older alerts use a stable cursor and temporary API errors retain displayed alerts", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view, fail = false;
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(url);
    if (fail) throw new Error("API temporalmente desconectada");
    return response({ items: [url.includes("before_id") ? { ...fault, alert_id: 1 } : fault], next_before_id: url.includes("before_id") ? null : 3 });
  };
  try {
    const { default: Page } = await server.ssrLoadModule("/src/features/alerts/AlertsPage.jsx");
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Page, { apiBaseUrl: "/api" });
    fail = true;
    await React.act(async () => context.mock.timers.tick(3000));
    assert.match(view.container.textContent, /No se pudieron actualizar/);
    assert.match(view.container.textContent, /sensor.temperature/);
    fail = false;
    await React.act(async () => context.mock.timers.tick(3000));
    assert.doesNotMatch(view.container.textContent, /No se pudieron actualizar/);
    await view.event(button(view, "Ver alertas anteriores"), "click");
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50&before_id=3");
    const count = calls.length;
    await React.act(async () => context.mock.timers.tick(6000));
    assert.equal(calls.length, count);
    await view.event(button(view, "Volver a las recientes"), "click");
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50");
  } finally { if (view) await view.close(); context.mock.timers.reset(); globalThis.fetch = previousFetch; await server.close(); }
});

test("the alerts route opens directly without loading history or starting the flow editor", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url) => { calls.push(url); return response({ items: [], next_before_id: null }); };
  let view;
  try {
    const { App } = await server.ssrLoadModule("/src/app/App.jsx");
    function Host() { window.history.replaceState({}, "", "/alertas"); return React.createElement(App); }
    view = await renderInteractive(Host, {});
    assert.match(view.container.textContent, /Artifact-Alertas/);
    assert.match(view.container.textContent, /No hay alertas de salida/);
    assert.deepEqual(calls, ["/api/flow-alerts?limit=50"]);
  } finally { if (view) await view.close(); globalThis.fetch = previousFetch; await server.close(); }
});

test("the Home Assistant sidebar panel embeds the alerts view, supports the menu and offers a separate window", () => {
  const require = createRequire(new URL("../../frontend/package.json", import.meta.url));
  const { JSDOM } = require("jsdom");
  const dom = new JSDOM("<!doctype html><body></body>", { url: "http://ha.local:8123", runScripts: "outside-only" });
  try {
    dom.window.eval(readFileSync(new URL("../../custom_components/artifact/frontend/alerts-panel.js", import.meta.url), "utf8"));
    const panel = dom.window.document.createElement("artifact-alerts-panel");
    dom.window.document.body.append(panel);
    panel.panel = { config: { url: "http://artifact.local:5173/alertas" } };
    assert.equal(panel.shadowRoot.querySelector("iframe").src, "http://artifact.local:5173/alertas");
    const link = panel.shadowRoot.querySelector("a");
    assert.equal(link.href, "http://artifact.local:5173/alertas");
    assert.equal(link.target, "_blank");
    let toggled = false;
    dom.window.document.addEventListener("hass-toggle-menu", () => { toggled = true; });
    panel.shadowRoot.querySelector("button").click();
    assert.equal(toggled, true);
    panel.panel = { config: { url: "javascript:alert(1)" } };
    assert.equal(panel.shadowRoot.querySelector("iframe").src, "http://artifact.local:5173/alertas");
  } finally { dom.window.close(); }
});
