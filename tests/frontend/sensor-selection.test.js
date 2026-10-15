import test from "node:test";
import assert from "node:assert/strict";
import { createServer, renderInteractive } from "../../frontend/test-runtime.js";

test("5000 initially active sensors render only 50 rows and preserve selection across pages and bulk filters", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  let view;
  try {
    const { default: Selection } = await server.ssrLoadModule("/src/features/history/SensorSelection.jsx");
    const entities = Array.from({ length: 5000 }, (_, index) => ({ entity_id: `${index % 2 ? "binary_sensor" : "sensor"}.entity_${index}`,
      domain: index % 2 ? "binary_sensor" : "sensor", sensor_kind: index % 2 ? "binary" : "continuous" }));
    let prepared;
    view = await renderInteractive(Selection, { catalog: { source: "csv", filename: "large.csv", entities }, onPrepare: (ids) => { prepared = ids; } });
    const button = (text) => [...view.container.querySelectorAll("button")].find((item) => item.textContent.includes(text));
    assert.equal(view.container.querySelectorAll('[data-testid="history-sensor-row"]').length, 50);
    assert.match(view.container.textContent, /5\.000 de 5\.000 sensores activos/);
    assert.match(view.container.textContent, /12\.497\.500 pares/);
    assert.equal(prepared, undefined);
    await view.event(view.container.querySelector('input[type="checkbox"]'), "click");
    await view.event(button("Página siguiente"), "click");
    assert.equal(view.container.querySelector('input[type="checkbox"]').getAttribute("aria-label"), "Analizar sensor.entity_50");
    await view.event(button("Página anterior"), "click");
    assert.equal(view.container.querySelector('input[type="checkbox"]').checked, false);
    const kind = view.container.querySelectorAll("select")[1];
    kind.value = "binary";
    await view.event(kind, "change");
    await view.event(button("Desactivar resultados"), "click");
    assert.match(view.container.textContent, /2\.499 de 5\.000 sensores activos/);
    await view.event(button("Página siguiente"), "click");
    assert.ok([...view.container.querySelectorAll('input[type="checkbox"]')].every((input) => !input.checked));
    kind.value = "continuous";
    await view.event(kind, "change");
    assert.equal(view.container.querySelectorAll('[data-testid="history-sensor-row"]').length, 50);
    await view.event(button("Preparar filtros con"), "click");
    assert.equal(prepared.length, 2499);
    assert.ok(prepared.every((id) => id.startsWith("sensor.")));
    assert.ok(!prepared.includes("sensor.entity_0"));
  } finally { if (view) await view.close(); await server.close(); }
});
