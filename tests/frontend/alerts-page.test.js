import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { React, createServer, renderInteractive } from "../../frontend/test-runtime.js";
import { connectHomeAssistantTheme } from "../../frontend/src/features/alerts/home-assistant-theme.js";

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
    assert.ok(calls.length > count);
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50&before_id=3");
    await view.event(button(view, "Volver a las recientes"), "click");
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50");
  } finally { if (view) await view.close(); context.mock.timers.reset(); globalThis.fetch = previousFetch; await server.close(); }
});

test("severity filtering applies to polling and older pages, and changing it returns to recent alerts", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  const anomaly = { ...fault, alert_id: 2, result: { ...fault.result, final_decision: "environmental" } };
  const suspicion = { ...fault, alert_id: 1, result: { ...fault.result, final_decision: "suspicion" } };
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(url);
    const query = new URL(url, "http://artifact.local").searchParams;
    const severity = query.get("severity");
    return response({ items: severity === "fault" ? [fault] : severity === "anomaly" ? [anomaly, suspicion] : [fault, anomaly],
      next_before_id: severity === "fault" && !query.has("before_id") ? 3 : null });
  };
  let view;
  try {
    const { default: Page } = await server.ssrLoadModule("/src/features/alerts/AlertsPage.jsx");
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Page, { apiBaseUrl: "/api" });
    const select = view.container.querySelector('select[aria-label="Tipo de alerta"]');
    assert.equal(select.value, "all");
    assert.equal(view.container.querySelectorAll(".artifact-alert-card").length, 2);
    select.value = "fault";
    await view.event(select, "change");
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50&severity=fault");
    assert.equal(view.container.querySelectorAll(".artifact-alert-card-anomaly").length, 0);
    await React.act(async () => context.mock.timers.tick(3000));
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50&severity=fault");
    await view.event(button(view, "Ver alertas anteriores"), "click");
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50&before_id=3&severity=fault");
    select.value = "anomaly";
    await view.event(select, "change");
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50&severity=anomaly");
    assert.equal(view.container.querySelectorAll(".artifact-alert-card-fault").length, 0);
    assert.equal(view.container.querySelectorAll(".artifact-alert-card-anomaly").length, 2);
    assert.doesNotMatch(view.container.textContent, /Consultando alertas anteriores/);
    await view.event(button(view, "Actualizar"), "click");
    assert.equal(select.value, "anomaly");
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50&severity=anomaly");
    select.value = "all";
    await view.event(select, "change");
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50");
    assert.equal(view.container.querySelectorAll(".artifact-alert-card").length, 2);
  } finally { if (view) await view.close(); context.mock.timers.reset(); globalThis.fetch = previousFetch; await server.close(); }
});

test("a response from a previous filter cannot replace the selected filter's empty state", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let resolveFault;
  globalThis.fetch = async (url) => url.includes("severity=fault") ? new Promise((resolve) => { resolveFault = resolve; })
    : response({ items: [], next_before_id: null });
  let view;
  try {
    const { default: Page } = await server.ssrLoadModule("/src/features/alerts/AlertsPage.jsx");
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Page, { apiBaseUrl: "/api" });
    const select = view.container.querySelector('select[aria-label="Tipo de alerta"]');
    select.value = "fault";
    await view.event(select, "change");
    select.value = "anomaly";
    await view.event(select, "change");
    await React.act(async () => resolveFault(response({ items: [fault], next_before_id: null })));
    assert.equal(select.value, "anomaly");
    assert.match(view.container.textContent, /No hay anomalías/);
    assert.equal(view.container.querySelectorAll(".artifact-alert-card").length, 0);
  } finally { if (view) await view.close(); context.mock.timers.reset(); globalThis.fetch = previousFetch; await server.close(); }
});

test("an older page follows the newest execution automatically without keeping the previous execution's alerts", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let current = "first";
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(url);
    const older = url.includes("before_id");
    return response({ execution_id: current,
      items: current === "first" ? [fault] : older ? [] : [{ ...fault, alert_id: 4, result: { ...fault.result, entity_id: "sensor.new" } }],
      next_before_id: current === "first" && !older ? 3 : null });
  };
  let view;
  try {
    const { default: Page } = await server.ssrLoadModule("/src/features/alerts/AlertsPage.jsx");
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Page, { apiBaseUrl: "/api" });
    await view.event(button(view, "Ver alertas anteriores"), "click");
    assert.match(view.container.textContent, /Consultando alertas anteriores/);
    current = "second";
    await React.act(async () => context.mock.timers.tick(3000));
    assert.equal(calls.at(-1), "/api/flow-alerts?limit=50");
    assert.match(view.container.textContent, /sensor.new/);
    assert.doesNotMatch(view.container.textContent, /sensor.temperature|Consultando alertas anteriores/);
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

test("the Home Assistant sidebar panel embeds the alerts view without a duplicate toolbar", () => {
  const require = createRequire(new URL("../../frontend/package.json", import.meta.url));
  const { JSDOM } = require("jsdom");
  const dom = new JSDOM("<!doctype html><body></body>", { url: "http://ha.local:8123", runScripts: "outside-only", pretendToBeVisual: true });
  try {
    dom.window.eval(readFileSync(new URL("../../custom_components/artifact/frontend/alerts-panel.js", import.meta.url), "utf8"));
    const panel = dom.window.document.createElement("artifact-alerts-panel");
    dom.window.document.body.append(panel);
    panel.panel = { config: { url: "http://artifact.local:5173/alertas" } };
    assert.equal(panel.shadowRoot.querySelector("iframe").src, "http://artifact.local:5173/alertas?ha_origin=http%3A%2F%2Fha.local%3A8123");
    assert.equal(panel.shadowRoot.querySelector("header, button, a"), null);
    panel.panel = { config: { url: "javascript:alert(1)" } };
    assert.equal(panel.shadowRoot.querySelector("iframe").src, "http://artifact.local:5173/alertas?ha_origin=http%3A%2F%2Fha.local%3A8123");
  } finally { dom.window.close(); }
});

test("the panel shares only theme properties with its configured alerts origin and reacts to theme changes", async () => {
  const require = createRequire(new URL("../../frontend/package.json", import.meta.url));
  const { JSDOM } = require("jsdom");
  const dom = new JSDOM("<!doctype html><body></body>", { url: "http://ha.local:8123", runScripts: "outside-only", pretendToBeVisual: true });
  try {
    dom.window.eval(readFileSync(new URL("../../custom_components/artifact/frontend/alerts-panel.js", import.meta.url), "utf8"));
    const panel = dom.window.document.createElement("artifact-alerts-panel");
    panel.style.setProperty("--primary-background-color", "#111111");
    panel.style.setProperty("--primary-color", "#03a9f4");
    panel.style.setProperty("--ha-font-family-body", "Roboto, sans-serif");
    panel.hass = { themes: { darkMode: true }, auth: { secret: "never-shared" } };
    dom.window.document.body.append(panel);
    panel.panel = { config: { url: "http://artifact.local:5173/alertas" } };
    const frame = panel.shadowRoot.querySelector("iframe");
    const messages = [];
    // JSDOM does not create browsing contexts for iframes inside shadow roots.
    Object.defineProperty(frame, "contentWindow", { value: { postMessage: (data, origin) => messages.push({ data, origin }) } });
    await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
    assert.equal(messages.at(-1).origin, "http://artifact.local:5173");
    assert.equal(messages.at(-1).data.dark, true);
    assert.equal(messages.at(-1).data.variables["--primary-background-color"], "#111111");
    assert.equal(messages.at(-1).data.variables["--ha-font-family-body"], "Roboto, sans-serif");
    assert.doesNotMatch(JSON.stringify(messages), /never-shared|auth/);
    const count = messages.length;
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", { origin: "http://other.local", source: frame.contentWindow, data: { type: "artifact-alerts-theme-ready" } }));
    assert.equal(messages.length, count);
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", { origin: "http://artifact.local:5173", source: frame.contentWindow, data: { type: "artifact-alerts-theme-ready" } }));
    assert.equal(messages.length, count + 1);
    panel.style.setProperty("--primary-background-color", "#fafafa");
    panel.hass = { themes: { darkMode: false } };
    await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
    assert.equal(messages.at(-1).data.dark, false);
    assert.equal(messages.at(-1).data.variables["--primary-background-color"], "#fafafa");
    panel.remove();
  } finally { dom.window.close(); }
});

test("the alerts document accepts theme changes only from its Home Assistant parent and restores its styles on exit", () => {
  const require = createRequire(new URL("../../frontend/package.json", import.meta.url));
  const { JSDOM } = require("jsdom");
  const dom = new JSDOM("<!doctype html><body></body>", { url: "http://artifact.local/alertas?ha_origin=http%3A%2F%2Fha.local%3A8123" });
  let disconnect;
  try {
    const messages = [], parent = { postMessage: (...args) => messages.push(args) };
    const target = { document: dom.window.document, location: dom.window.location, parent,
      addEventListener: dom.window.addEventListener.bind(dom.window), removeEventListener: dom.window.removeEventListener.bind(dom.window) };
    const root = dom.window.document.documentElement;
    root.style.setProperty("--primary-color", "#123456");
    disconnect = connectHomeAssistantTheme(target);
    assert.equal(root.classList.contains("artifact-alerts-document"), true);
    assert.deepEqual(messages, [[{ type: "artifact-alerts-theme-ready" }, "http://ha.local:8123"]]);
    const data = { type: "artifact-alerts-theme", dark: true, variables: {
      "--primary-background-color": "#111111", "--primary-color": "#03a9f4", "--ha-card-background": "#1c1c1c", "--unrelated": "ignored",
    } };
    for (const [origin, source] of [["http://other.local", parent], ["http://ha.local:8123", {}]]) {
      dom.window.dispatchEvent(new dom.window.MessageEvent("message", { origin, source, data }));
      assert.equal(root.style.getPropertyValue("--primary-color"), "#123456");
    }
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", { origin: "http://ha.local:8123", source: parent, data }));
    assert.equal(root.style.getPropertyValue("--primary-background-color"), "#111111");
    assert.equal(root.style.getPropertyValue("--primary-color"), "#03a9f4");
    assert.equal(root.style.getPropertyValue("--unrelated"), "");
    assert.equal(root.style.colorScheme, "dark");
    dom.window.dispatchEvent(new dom.window.MessageEvent("message", { origin: "http://ha.local:8123", source: parent,
      data: { type: "artifact-alerts-theme", dark: false, variables: { "--primary-background-color": "#fafafa" } } }));
    assert.equal(root.style.colorScheme, "light");
    assert.equal(root.style.getPropertyValue("--ha-card-background"), "");
    disconnect(); disconnect = null;
    assert.equal(root.classList.contains("artifact-alerts-document"), false);
    assert.equal(root.style.getPropertyValue("--primary-color"), "#123456");
  } finally { disconnect?.(); dom.window.close(); }
});
