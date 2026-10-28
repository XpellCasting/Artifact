import { createServer as createViteServer } from "vite";
import { fileURLToPath } from "node:url";
import React, { act } from "react";
import { JSDOM } from "jsdom";
export { default as React } from "react";
export { renderToString, renderToStaticMarkup } from "react-dom/server";
export function createServer(options = {}) {
  return createViteServer({ ...options, root: fileURLToPath(new URL(".", import.meta.url)),
    optimizeDeps: { noDiscovery: true, include: [] },
    resolve: { dedupe: ["react", "react-dom"] },
    server: { ...options.server, hmr: false } });
}

export async function renderInteractive(Component, props) {
  const dom = new JSDOM('<!doctype html><html><body><div id="test"></div></body></html>', { url: "http://localhost" });
  const previous = new Map();
  for (const [name, value] of Object.entries({ window: dom.window, document: dom.window.document,
    navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, SVGElement: dom.window.SVGElement,
    IS_REACT_ACT_ENVIRONMENT: true })) {
    previous.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  }
  const container = dom.window.document.getElementById("test");
  // React detects input event support when the client module first loads.
  const { createRoot } = await import("react-dom/client");
  const root = createRoot(container);
  await act(async () => root.render(React.createElement(Component, props)));
  return {
    container,
    async event(element, kind, options = {}) {
      const Constructor = kind.startsWith("key") ? dom.window.KeyboardEvent : kind === "click" ? dom.window.MouseEvent : dom.window.Event;
      await act(async () => element.dispatchEvent(new Constructor(kind, { bubbles: true, ...options })));
    },
    async close() {
      await act(async () => root.unmount());
      dom.window.close();
      for (const [name, descriptor] of previous) {
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else delete globalThis[name];
      }
    },
  };
}
