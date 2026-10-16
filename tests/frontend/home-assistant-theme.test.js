import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { connectHomeAssistantTheme } from "../../frontend/src/shared/home-assistant-theme.js";
const require = createRequire(new URL("../../frontend/package.json", import.meta.url));
const { JSDOM } = require("jsdom");

test("a dashboard iframe without referrer requests its parent's theme and pins subsequent updates to that origin", () => {
  const dom = new JSDOM("<!doctype html><body></body>", { url: "http://artifact.local/" });
  const requests = [], parent = { postMessage: (...args) => requests.push(args) };
  const target = { document: dom.window.document, location: dom.window.location, parent,
    addEventListener: dom.window.addEventListener.bind(dom.window), removeEventListener: dom.window.removeEventListener.bind(dom.window) };
  const disconnect = connectHomeAssistantTheme(target);
  try {
    assert.equal(dom.window.document.documentElement.classList.contains("artifact-ha-theme"), true);
    assert.deepEqual(requests, [[{ type: "artifact-alerts-theme-ready" }, "*"]]);
    const send = (origin, background, source = parent) => dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      origin, source, data: { type: "artifact-alerts-theme", dark: true, variables: { "--primary-background-color": background } },
    }));
    send("http://ha.local", "#111111", {});
    assert.equal(dom.window.document.documentElement.style.getPropertyValue("--primary-background-color"), "");
    send("http://ha.local", "#111111");
    send("http://another.local", "#ffffff");
    assert.equal(dom.window.document.documentElement.style.getPropertyValue("--primary-background-color"), "#111111");
  } finally { disconnect(); dom.window.close(); }
});

test("the main dashboard can reuse the theme received by Alerts with an older installed integration", () => {
  const stored = new Map(), storage = { getItem: (key) => stored.get(key), setItem: (key, value) => stored.set(key, value) };
  const makeView = (path) => {
    const dom = new JSDOM("<!doctype html><body></body>", { url: `http://artifact.local${path}` });
    const parent = { postMessage() {} };
    const target = { document: dom.window.document, location: dom.window.location, parent, localStorage: storage,
      addEventListener: dom.window.addEventListener.bind(dom.window), removeEventListener: dom.window.removeEventListener.bind(dom.window) };
    return { dom, parent, target, disconnect: connectHomeAssistantTheme(target) };
  };
  const alerts = makeView("/alertas?ha_origin=http%3A%2F%2Fha.local");
  let dashboard;
  try {
    alerts.dom.window.dispatchEvent(new alerts.dom.window.MessageEvent("message", { source: alerts.parent, origin: "http://ha.local",
      data: { type: "artifact-alerts-theme", dark: true, variables: { "--primary-background-color": "#111111", "--primary-color": "#03a9f4" } } }));
    dashboard = makeView("/");
    const root = dashboard.dom.window.document.documentElement;
    assert.equal(root.style.getPropertyValue("--primary-background-color"), "#111111");
    assert.equal(root.style.getPropertyValue("--primary-color"), "#03a9f4");
    assert.equal(root.style.colorScheme, "dark");
  } finally { alerts.disconnect(); alerts.dom.window.close(); dashboard?.disconnect(); dashboard?.dom.window.close(); }
});

test("Home Assistant's dashboard bridge recognizes nested Artifact frames and sends theme changes without reloading them", async () => {
  const dom = new JSDOM("<!doctype html><body><home-assistant></home-assistant></body>", {
    url: "http://ha.local/", runScripts: "outside-only", pretendToBeVisual: true,
  });
  try {
    const host = dom.window.document.querySelector("home-assistant");
    host.hass = { themes: { darkMode: true }, auth: { secret: "not-for-sharing" } };
    const shadow = host.attachShadow({ mode: "open" });
    const frame = dom.window.document.createElement("iframe");
    frame.src = "http://artifact.local/app/";
    frame.style.setProperty("--primary-background-color", "#111111");
    shadow.append(frame);
    const messages = [], source = { postMessage: (data, origin) => messages.push({ data, origin }) };
    Object.defineProperty(frame, "contentWindow", { value: source });
    const moduleUrl = "http://ha.local/artifact-static/dashboard-theme.js?web_url=http%3A%2F%2Fartifact.local%2Fapp";
    const script = readFileSync(new URL("../../custom_components/artifact/frontend/dashboard-theme.js", import.meta.url), "utf8");
    dom.window.eval(script.replace("import.meta.url", JSON.stringify(moduleUrl)));
    const ready = (origin = "http://artifact.local") => dom.window.dispatchEvent(new dom.window.MessageEvent("message", {
      origin, source, data: { type: "artifact-alerts-theme-ready" },
    }));
    ready("http://unknown.local");
    assert.equal(messages.length, 0);
    ready();
    assert.equal(messages.at(-1).data.variables["--primary-background-color"], "#111111");
    assert.equal(messages.at(-1).origin, "http://artifact.local");
    assert.doesNotMatch(JSON.stringify(messages), /not-for-sharing|auth/);
    host.hass = { themes: { darkMode: false } };
    frame.style.setProperty("--primary-background-color", "#fafafa");
    // HA updates an ancestor; this also invalidates JSDOM's shadow style cache.
    host.style.setProperty("--primary-background-color", "#fafafa");
    await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
    await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
    assert.equal(messages.at(-1).data.variables["--primary-background-color"], "#fafafa");
    assert.equal(messages.at(-1).data.dark, false);
    assert.equal(frame.src, "http://artifact.local/app/");
    frame.src = "http://artifact.local/different-app/";
    frame.remove();
    ready();
  } finally { dom.window.close(); }
});
