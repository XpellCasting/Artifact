import test from "node:test";
import assert from "node:assert/strict";
import { createServer, renderInteractive } from "../../frontend/test-runtime.js";

const button = (root, text) => [...root.querySelectorAll("button")].find((item) => item.textContent.includes(text));
const rows = (root) => root.querySelectorAll('[data-testid="history-sensor-row"]');
async function change(view, label, value) {
  const input = [...view.container.querySelectorAll("label")].find((item) => item.textContent.startsWith(label)).querySelector("input,select");
  const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), "value").set;
  setter.call(input, value);
  await view.event(input, input.tagName === "SELECT" ? "change" : "input");
}
const entity = (index, area_id, area_name) => ({ entity_id: `sensor.entity_${index}`, friendly_name: `Temperatúra ${index}`,
  domain: "sensor", sensor_kind: "continuous", area_id, area_name });

test("5000 HA entities start unselected, with lazy area panels, separate unassigned rows and 50 rows per list", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  let view;
  try {
    const { default: Selection } = await server.ssrLoadModule("/src/features/history/SensorSelection.jsx");
    const entities = Array.from({ length: 5000 }, (_, index) => entity(index, index < 4900 ? "living" : null, index < 4900 ? "Salón" : null));
    let prepared;
    view = await renderInteractive(Selection, { catalog: { source: "home_assistant", areas_available: true, entities }, onPrepare: (ids) => { prepared = ids; } });
    const root = view.container;
    assert.match(root.textContent, /0 de 5\.000 sensores activos/);
    assert.equal(button(root, "Preparar filtros con").disabled, true);
    const panel = root.querySelector('[aria-label="Área Salón"]');
    assert.equal(panel.querySelector('[aria-expanded]').getAttribute("aria-expanded"), "false");
    assert.equal(rows(panel).length, 0);
    assert.equal(rows(root).length, 50);
    assert.equal(root.querySelector('[aria-label="Sin área asignada"]').closest(".hs-area"), null);
    await view.event(panel.querySelector('[aria-expanded]'), "click");
    assert.equal(rows(panel).length, 50);
    await view.event(panel.querySelector('input[type="checkbox"]'), "click");
    assert.match(panel.textContent, /1 de 4\.900 seleccionados/);
    await view.event(button(panel, "Página siguiente"), "click");
    assert.equal(panel.querySelector('input[type="checkbox"]').getAttribute("aria-label"), "Analizar sensor.entity_50");
    await view.event(button(panel, "Activar resultados de"), "click");
    assert.match(root.textContent, /4\.900 de 5\.000 sensores activos/);
    assert.match(root.textContent, /pares/);
    await view.event(panel.querySelector('[aria-expanded]'), "click");
    assert.equal(rows(panel).length, 0);
    await view.event(panel.querySelector('[aria-expanded]'), "click");
    assert.equal(panel.querySelector('input[type="checkbox"]').checked, true);
    assert.match(panel.textContent, /Página 2 de 98/);
    assert.equal(prepared, undefined);
    await view.event(button(root, "Preparar filtros con"), "click");
    assert.equal(prepared.length, 4900);
    assert.ok(prepared.every((id) => Number(id.split("_").at(-1)) < 4900));
  } finally { if (view) await view.close(); await server.close(); }
});

test("area and entity searches ignore accents, combine filters and apply actions across every page", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  let view;
  try {
    const { default: Selection } = await server.ssrLoadModule("/src/features/history/SensorSelection.jsx");
    const entities = [...Array.from({ length: 120 }, (_, index) => entity(index, "living", "Salón")), entity(120, "kitchen", "Cocina"), entity(121, null, null)];
    entities[119] = { ...entities[119], entity_id: "binary_sensor.door", domain: "binary_sensor", sensor_kind: "binary" };
    let prepared;
    view = await renderInteractive(Selection, { catalog: { source: "home_assistant", areas_available: true, entities }, onPrepare: (ids) => { prepared = ids; } });
    const root = view.container;
    await change(view, "Buscar sensores", "SALON");
    assert.equal(root.querySelectorAll(".hs-area").length, 1);
    assert.match(root.textContent, /120 sensores que coinciden/);
    await view.event(button(root, "Activar resultados"), "click");
    assert.match(root.textContent, /120 de 122 sensores activos/);
    await change(view, "Área", "kitchen");
    assert.match(root.textContent, /No hay sensores/);
    await change(view, "Buscar sensores", "temperatura");
    assert.equal(root.querySelectorAll(".hs-area").length, 1);
    assert.match(root.textContent, /Cocina/);
    await change(view, "Área", "living");
    await change(view, "Tipo", "binary");
    await change(view, "Dominio", "binary_sensor");
    assert.match(root.textContent, /1 sensores que coinciden/);
    await view.event(button(root, "Desactivar resultados"), "click");
    assert.match(root.textContent, /119 de 122 sensores activos/);
    await change(view, "Estado de selección", "active");
    assert.match(root.textContent, /No hay sensores/);
    await change(view, "Estado de selección", "inactive");
    await change(view, "Buscar sensores", "BINARY_SENSOR.DOOR");
    assert.match(root.textContent, /1 sensores que coinciden/);
    await view.event(button(root, "Activar todos"), "click");
    assert.match(root.textContent, /122 de 122 sensores activos/);
    await view.event(button(root, "Desactivar todos"), "click");
    assert.match(root.textContent, /0 de 122 sensores activos/);
    await change(view, "Buscar sensores", "");
    await change(view, "Dominio", "");
    await change(view, "Tipo", "");
    await change(view, "Estado de selección", "");
    await change(view, "Área", "__unassigned__");
    assert.equal(root.querySelectorAll(".hs-area").length, 0);
    assert.equal(rows(root).length, 1);
    await view.event(root.querySelector('input[type="checkbox"]'), "click");
    await view.event(button(root, "Preparar filtros con"), "click");
    assert.deepEqual(prepared, ["sensor.entity_121"]);
  } finally { if (view) await view.close(); await server.close(); }
});

test("no assigned areas uses one list; unavailable metadata stays flat and never claims unassigned", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  let view;
  try {
    const { default: Selection } = await server.ssrLoadModule("/src/features/history/SensorSelection.jsx");
    for (const available of [true, false, undefined]) {
      view = await renderInteractive(Selection, { catalog: { source: "home_assistant", areas_available: available,
        entities: [entity(1, null, null)] } });
      assert.equal(rows(view.container).length, 1);
      assert.equal(view.container.querySelectorAll(".hs-area").length, 0);
      assert.equal(view.container.querySelector('input[type="checkbox"]').checked, false);
      if (available) {
        assert.match(view.container.textContent, /Sin área asignada/);
        assert.doesNotMatch(view.container.textContent, /No se pudieron consultar/);
      } else {
        assert.match(view.container.textContent, /No se pudieron consultar las áreas/);
        assert.doesNotMatch(view.container.textContent, /Sin área asignada/);
        assert.equal(view.container.querySelector('select[aria-label="Área"]').disabled, true);
      }
      await view.close(); view = undefined;
    }
  } finally { if (view) await view.close(); await server.close(); }
});
