import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "vite";
import React from "react";
import { renderToString } from "react-dom/server";
import { annotationForPoint, pointFaultKinds } from "./point-labels.js";
import { splitReviewAnnotations } from "./review-annotations.js";

test("persisted point labels are hidden from interval notes but remain available to charts and export", () => {
  const point = { annotation_id: "point", source_row: 2, entity_id: "sensor.a", label: "normal", evaluable: true };
  const interval = { annotation_id: "interval", source_row: null };
  const legacy = { annotation_id: "legacy" };
  const annotations = [point, interval, legacy];
  const separated = splitReviewAnnotations(annotations);
  assert.deepEqual(separated.points, [point]);
  assert.deepEqual(separated.intervals, [interval, legacy]);
  assert.equal(annotationForPoint(annotations, "sensor.a", { source_row: 2 }), point);
  assert.equal(annotations.length, 3);
});

function nodes(tree) {
  if (Array.isArray(tree)) return tree.flatMap(nodes);
  if (!tree || typeof tree !== "object") return [];
  return [tree, ...nodes(tree.props?.children)];
}

test("point identity overrides an interval without labelling duplicate timestamps or unrelated records", () => {
  const timestamp = "2026-06-06T12:34:56.123456Z";
  const interval = { entity_id: "sensor.a", source_row: null, start: timestamp, end: timestamp, label: "normal", evaluable: true };
  const point = { ...interval, source_row: 7, label: "sensor_fault", fault_type: "bias" };
  assert.equal(annotationForPoint([interval, point], "sensor.a", { timestamp, source_row: 7 }), point);
  assert.equal(annotationForPoint([interval, point], "sensor.a", { timestamp, source_row: 8 }), interval);
  assert.equal(annotationForPoint([interval, point], "sensor.b", { timestamp, source_row: 9 }), undefined);
  assert.equal(annotationForPoint([{ ...interval, source_row: undefined }], "sensor.a", { timestamp: "2026-06-07T00:00:00Z" }), undefined);
  assert.equal(annotationForPoint([interval], "sensor.a", { timestamp: "2026-06-06T12:34:56.123457Z", source_row: 10 }), undefined);
});

test("click, keyboard and overlapping-point selector identify the exact original row, including invalid states", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: "custom" });
  try {
    const { SeriesPlot: SeriesChart } = await server.ssrLoadModule("/src/ReviewWorkspace.jsx");
    const timestamp = "2026-06-06T12:34:56.123456Z";
    const points = [
      { source_row: 7, timestamp, state: "NONE", is_valid: true },
      { source_row: 8, timestamp, state: "NONE", is_valid: true },
      { source_row: 9, timestamp, state: "unavailable", is_valid: false },
    ];
    const series = { entity_id: "sensor.status", kind: "multistate", points, states: ["NONE", "DETECTION", "ALERT"] };
    const selected = [];
    const props = { series, fragment: { start: timestamp, end: timestamp, view_start: timestamp, view_end: timestamp },
      annotations: [{ entity_id: series.entity_id, source_row: 7, start: timestamp, end: timestamp, evaluable: true, label: "sensor_fault", fault_type: "stuck" }],
      onPoint: (entity, point) => selected.push([entity.entity_id, point.source_row]), target: true };
    const html = renderToString(React.createElement(SeriesChart, props));
    assert.match(html, /review-label-sensor_fault/);
    assert.match(html, /review-label-unlabelled/);
    assert.match(html, /Valor congelado/);
    assert.deepEqual(selected, []);
    const elements = nodes(SeriesChart(props));
    const buttons = elements.filter(node => node.type === "circle" && node.props.role === "button");
    buttons[0].props.onClick();
    buttons[2].props.onKeyDown({ key: "Enter", preventDefault() {} });
    elements.find(node => node.type === "select").props.onChange({ target: { value: "8" } });
    assert.deepEqual(selected, [[series.entity_id, 7], [series.entity_id, 9], [series.entity_id, 8]]);
    assert.equal(nodes(SeriesChart({ ...props, onPoint: undefined })).filter(node => node.props.role === "button").length, 0);
  } finally { await server.close(); }
});

test("label menu saves only the chosen class, limits fault kinds by entity type and disables buttons while saving", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: "custom" });
  try {
    const { default: PointLabeler } = await server.ssrLoadModule("/src/PointLabeler.jsx");
    for (const kind of ["continuous", "binary", "multistate"]) {
      const saved = [];
      const props = { selection: { series: { entity_id: "sensor.a", kind }, point: { source_row: 7, timestamp: "2026-06-06T12:34:56.123456Z", state: "NONE" } },
        annotations: [], reviewer: "", onReviewer: () => {}, onSave: (...args) => saved.push(args), disabled: false };
      const tree = PointLabeler(props);
      const html = renderToString(tree);
      assert.match(html.replace(/<!-- -->/g, ""), /Fila original 7/);
      assert.match(html, /Falla → Tipo de falla/);
      assert.deepEqual(saved, []);
      const buttons = nodes(tree).filter(node => node.type === "button");
      for (const button of buttons) button.props.onClick();
      assert.deepEqual(saved, [["normal"], ["environmental_anomaly"], ...pointFaultKinds(kind).map(fault => ["sensor_fault", fault])]);
      if (kind !== "continuous") assert.doesNotMatch(html, /Pico|Desplazamiento|Deriva|Ruido/);
      assert.ok(nodes(PointLabeler({ ...props, disabled: true })).filter(node => node.type === "button").every(node => node.props.disabled));
    }
  } finally { await server.close(); }
});
