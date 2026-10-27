import test from "node:test";
import assert from "node:assert/strict";
import { createServer, renderInteractive } from "../../frontend/test-runtime.js";

for (const succeeds of [true, false]) {
  test(`model ZIP import ${succeeds ? "selects the imported version without activation" : "shows the API error and allows retry"}`, async () => {
    const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
    const previousFetch = globalThis.fetch;
    const oldId = "a".repeat(32), importedId = "b".repeat(32);
    const model = (model_id) => ({ model_id, method: "gat_lstm", created_at: "2026-10-22T10:00:00Z", training_count: 10 });
    const calls = [];
    let imported = false;
    globalThis.fetch = async (url, options = {}) => {
      calls.push({ url: String(url), options });
      let value = {};
      if (String(url).includes("/diagnostics/status")) value = {
        datasets: [], models: imported ? [model(oldId), model(importedId)] : [model(oldId)], active_model_id: oldId,
      };
      if (String(url).includes("/diagnostics/sources")) value = { annotations: [], injections: [] };
      if (String(url).endsWith("/diagnostics/models/import")) {
        if (!succeeds) return new Response(JSON.stringify({ detail: "Hash de pesos incompatible" }), { status: 400 });
        imported = true;
        value = { model_id: importedId, imported_files: 4, existing_files: 0, activated: false };
      }
      return new Response(JSON.stringify(value), { headers: { "Content-Type": "application/json" } });
    };
    let ui;
    try {
      const { default: Workspace } = await server.ssrLoadModule("/src/features/diagnostics/DiagnosticWorkspace.jsx");
      ui = await renderInteractive(Workspace, { apiBaseUrl: "http://localhost:8000" });
      const button = (text) => [...ui.container.querySelectorAll("button")].find((item) => item.textContent.includes(text));
      await ui.event(button("Modelos y evaluación"), "click");
      assert.equal(button("Importar ZIP").disabled, true);
      const input = ui.container.querySelector('input[type="file"]');
      const file = new File(["zip contents"], "modelo_real.zip", { type: "application/zip" });
      Object.defineProperty(input, "files", { configurable: true, value: [file] });
      await ui.event(input, "change");
      assert.equal(button("Importar ZIP").disabled, false);
      await ui.event(button("Importar ZIP"), "click");
      const upload = calls.find(({ url }) => url.endsWith("/diagnostics/models/import"));
      assert.equal(upload.options.method, "POST");
      assert.equal(upload.options.headers["Content-Type"], "application/zip");
      assert.equal(upload.options.body, file);
      assert.equal(calls.some(({ url }) => url.endsWith("/activate")), false);
      if (succeeds) {
        assert.match(ui.container.textContent, /Modelo importado/);
        assert.equal([...ui.container.querySelectorAll("select")].some((select) => select.value === importedId), true);
        assert.equal(button("Importar ZIP").disabled, true);
        assert.ok(button("Activar versión experimental"));
      } else {
        assert.match(ui.container.querySelector('[role="alert"]').textContent, /Hash de pesos incompatible/);
        assert.equal(button("Importar ZIP").disabled, false);
        assert.match(ui.container.textContent, new RegExp(oldId));
      }
    } finally {
      if (ui) await ui.close();
      globalThis.fetch = previousFetch;
      await server.close();
    }
  });
}
