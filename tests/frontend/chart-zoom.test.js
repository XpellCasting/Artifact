import { renderInteractive } from "../../frontend/test-runtime.js";
import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "../../frontend/test-runtime.js";
import { zoomRange } from "../../frontend/src/shared/chart-zoom.js";

test("wheel zoom preserves the cursor's instant and can return to the complete day", () => {
  const total = 86400000000;
  const anchor = 0.73;
  const range = zoomRange([0, total], total, anchor, -160);
  assert.ok(range[1] - range[0] < total);
  assert.ok(Math.abs(range[0] + (range[1] - range[0]) * anchor - total * anchor) < 0.0001);
  const restored = zoomRange(range, total, anchor, 160);
  assert.ok(Math.abs(restored[0]) < 0.0001);
  assert.ok(Math.abs(restored[1] - total) < 0.0001);
  assert.deepEqual(zoomRange([0, 1], 1, 0.5, -100), [0, 1]);
  assert.ok(zoomRange([0, 50], 100, 0, 10000)[0] >= 0);
  assert.ok(zoomRange([50, 100], 100, 1, 10000)[1] <= 100);
});

test("zoom separates microsecond neighbours and does not pile offscreen records onto the edges", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: "custom" });
  try {
    const { SeriesPlot } = await server.ssrLoadModule("/src/features/review/ReviewWorkspace.jsx");
    const times = ["2026-06-06T12:00:00.000001Z", "2026-06-06T12:00:00.000002Z", "2026-06-06T12:00:00.000010Z"];
    const rendered = await renderInteractive(SeriesPlot, { series: { entity_id: "sensor.a", kind: "continuous", points: times.map((timestamp, index) => ({ timestamp, source_row: index + 2, state: "20", value: 20 })) },
      fragment: { start: times[0], end: times[2], view_start: times[0], view_end: times[2] }, viewport: [0, 2], onPoint() {} });
    try {
      const circles = [...rendered.container.querySelectorAll("circle")];
      assert.equal(circles.length, 2);
      assert.ok(Number(circles[1].getAttribute("cx")) - Number(circles[0].getAttribute("cx")) > 400);
      assert.ok(rendered.container.querySelector("clipPath"));
    } finally { await rendered.close(); }
  } finally { await server.close(); }
});
