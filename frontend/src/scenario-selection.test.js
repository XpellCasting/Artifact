import test from "node:test";
import assert from "node:assert/strict";
import { scenarioConflicts } from "./scenario-selection.js";

test("scenario conflicts use inclusive UTC microsecond intervals per entity", () => {
  const first = { entity_id: "sensor.a", start: "2026-01-01T00:00:00Z", end: "2026-01-01T00:01:00.000001Z" };
  assert.deepEqual(scenarioConflicts([first, { ...first, entity_id: "sensor.b" }]), []);
  assert.deepEqual(scenarioConflicts([first, { ...first, start: "2025-12-31T21:01:00.000001-03:00" }]), ["sensor.a"]);
  assert.deepEqual(scenarioConflicts([first, { ...first, start: "2026-01-01T00:01:00.000002Z", end: "2026-01-01T00:02:00Z" }]), []);
});
