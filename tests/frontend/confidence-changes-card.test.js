import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(new URL("../../frontend/package.json", import.meta.url));
const { JSDOM } = require("jsdom");

test("recorded decreases and increases retain their own evidence across plateaus and entity switches", async () => {
  const dom = new JSDOM("", { runScripts: "outside-only" });
  try {
    const script = readFileSync(new URL("../../custom_components/artifact/frontend/confidence-changes.js", import.meta.url), "utf8");
    dom.window.eval(script);
    const card = dom.window.document.createElement("paranoid-confidence-changes");
    card.setConfig({ title: "Cambios", entities: ["sensor.test", "sensor.live"] });
    const recorded = (kind, id, pct, time, message) => ({ state: String(pct), last_changed: time,
      attributes: { execution_id: "flow", activity: "receiving", evaluated_at: time, observation_id: id,
        confidence_change_reason: message, confidence_change_detail: { kind, after_pct: pct,
          evaluated_at: time, observation_id: id, source_timestamp: "2026-06-01T01:00:00Z",
          filters: [{ node_id: "f2", module_type: "contextual", before: 0, after: 0.6, impact_pp: -30 }] } } });
    const down = recorded("decrease", "a", 70, "2026-11-18T01:00:00Z", "Bajó por el contexto");
    const neighbor = { neighbor_id: "binary_sensor.hall", observed_value: 1, observation_status: "contradicts" };
    down.attributes.confidence_change_detail.filters[0].evidence = { neighbors: [neighbor] };
    const plateau = { ...down, attributes: { ...down.attributes, evaluated_at: "2026-11-18T01:01:00Z" } };
    const up = recorded("increase", "b", 100, "2026-11-18T01:02:00Z", "Volvió a subir");
    const change = up.attributes.confidence_change_detail;
    change.previous_observation_id = "a";
    change.filters[0].evidence = { neighbors: [{ ...neighbor, observation_status: "supports" }] };
    change.filters.push({ node_id: "f1", module_type: "statistical_focus", before: .4, after: 0,
      previous_evidence: { residual: 5, expected_value: 20 }, evidence: { residual: 0, expected_value: 21 } },
    { node_id: "f3", module_type: "classifier", before: .3, after: .1, evidence: { window_evidence: {
      windows: [{ seconds: 3600, usable: true, records: 5, changes: 2 }] } }, window_fault_delta_pp: { 3600: -20 } });
    let pending;
    const requests = [];
    card.hass = { states: { "sensor.test": up }, callApi: async (method, path) => {
      requests.push(path);
      return [[down, plateau, up]];
    } };
    await card.load();
    assert.match(requests[0], /filter_entity_id=sensor.test/);
    assert.ok(!requests[0].includes("no_attributes"));
    const root = card.shadowRoot;
    assert.equal(root.querySelectorAll(".rows > details").length, 2);
    const summaries = [...root.querySelectorAll(".rows > details > summary")].map(row => row.textContent);
    assert.match(summaries[0], /Volvió a subir/);
    assert.match(summaries[1], /Bajó por el contexto/);
    assert.match(root.querySelector(".rows pre").textContent, /2026-06-01/);
    const visibleDetails = root.querySelector(".rows > details > ul").textContent;
    assert.match(visibleDetails, /binary_sensor.hall · valor observado 1 → 1 · relación contradice → respalda/);
    assert.match(visibleDetails, /Desviación de la lectura 5 → 0/);
    assert.match(visibleDetails, /Ventana de 1 h.*-20 puntos en la puntuación de falla/);
    assert.match(root.querySelector(".current-evidence").textContent, /binary_sensor.hall/);
    card.hass = { states: {}, callApi: () => new Promise(resolve => { pending = resolve; }) };
    const old = card.load();
    card.select.value = "sensor.live";
    card.hass = { states: {}, callApi: async () => [] };
    await card.load();
    pending([[down]]);
    await old;
    assert.equal(root.querySelectorAll(".rows > details").length, 0);
    assert.match(card.status.textContent, /No hay alzas/);
  } finally { dom.window.close(); }
});
