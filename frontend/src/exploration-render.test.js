import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "vite";
import React from "react";
import { renderToString } from "react-dom/server";

test("explorer renders its empty initial state without selecting or labelling an entity", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  try {
    const { default: Explorer } = await server.ssrLoadModule("/src/HistoryExplorer.jsx");
    const html = renderToString(React.createElement(Explorer, { apiBaseUrl: "http://localhost:8000", dataset: { dataset_id: "fixture" }, onReview: () => assert.fail("No automatic selection") }));
    assert.match(html, /Explorar histórico/);
    assert.match(html, /Preparando resumen/);
    assert.doesNotMatch(html, /Revisar este intervalo/);
  } finally { await server.close(); }
});
