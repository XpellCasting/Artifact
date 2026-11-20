import assert from "node:assert/strict";
import test from "node:test";
import { createServer, renderInteractive } from "../../frontend/test-runtime.js";

test("removes only the confirmed inactive confidence sensor and reports pending synchronization", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  const requests = [];
  let entities = [
    { entity_id: "confianza.sensor_temperature", source_entity_id: "sensor.temperature", source_mode: "live", revision: 4, available: false },
    { entity_id: "confianza.test_sensor_temperature", source_entity_id: "sensor.temperature", source_mode: "test", revision: 8, available: true },
  ];
  let view;
  try {
    globalThis.fetch = async (url, options) => {
      if (options.method === "DELETE") {
        assert.equal(url, "/api/confidence/entities");
        requests.push(JSON.parse(options.body));
        entities = entities.slice(1);
        return { ok: true, json: async () => ({ removed: 1, synchronization: "pending" }) };
      }
      assert.equal(url, "/api/confidence/entities?compact=true");
      return { ok: true, json: async () => ({ entities }) };
    };
    const { default: Manager } = await server.ssrLoadModule("/src/features/flows/ConfidenceEntityManager.jsx");
    view = await renderInteractive(Manager, { apiBaseUrl: "/api" });
    const button = (label) => [...view.container.querySelectorAll("button")].find((item) => item.textContent.trim() === label);
    await view.event(button("Gestionar sensores de confianza"), "click");
    const [live, activeTest] = view.container.querySelectorAll('input[type="checkbox"]');
    assert.equal(activeTest.disabled, true);
    await view.event(live, "click");
    await view.event(button("Borrar selección (1)"), "click");
    assert.match(view.container.textContent, /ejecuciones guardadas seguirán disponibles/);
    assert.equal(requests.length, 0);
    await view.event(button("Cancelar"), "click");
    assert.equal(requests.length, 0);
    await view.event(button("Borrar selección (1)"), "click");
    await view.event(button("Confirmar borrado"), "click");
    assert.deepEqual(requests, [{ entities: [{ source_entity_id: "sensor.temperature", source_mode: "live", revision: 4 }] }]);
    assert.match(view.container.querySelector('[role="status"]').textContent, /pendiente de sincronización/);
    assert.equal(view.container.querySelectorAll('input[type="checkbox"]').length, 1);
  } finally {
    if (view) await view.close();
    globalThis.fetch = previousFetch;
    await server.close();
  }
});
