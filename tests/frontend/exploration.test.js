import test from "node:test";
import assert from "node:assert/strict";
import { firstObservationDay, explorationRows, reviewDayRange } from "../../frontend/src/features/history/exploration.js";

test("daily review includes the entire last second and limits only file boundary days", () => {
  assert.deepEqual(reviewDayRange("2026-06-06", "2026-06-01T03:00:00Z", "2026-06-30T14:00:00Z"), { start: "2026-06-06T00:00:00.000000Z", end: "2026-06-06T23:59:59.999999Z", partial: false });
  assert.deepEqual(reviewDayRange("2026-06-01", "2026-06-01T03:00:00.000001Z", "2026-06-01T14:00:00Z"), { start: "2026-06-01T03:00:00.000001Z", end: "2026-06-01T14:00:00.000000Z", partial: true });
  assert.equal(reviewDayRange("2026-02-30", "2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"), null);
  assert.equal(reviewDayRange("2026-05-31", "2026-06-01T00:00:00Z", "2026-06-30T00:00:00Z"), null);
});

test("first observed UTC day preserves microseconds and clamps to entity end", () => {
  assert.deepEqual(firstObservationDay("2026-05-13T20:34:10.000001Z", "2026-08-01T10:00:00Z"),
    { start: "2026-05-13T20:34:10.000001Z", end: "2026-05-13T23:59:59.999999Z" });
  assert.deepEqual(firstObservationDay("2026-01-01T00:00:00.000001Z", "2026-01-01T00:00:00.000001Z"),
    { start: "2026-01-01T00:00:00.000001Z", end: "2026-01-01T00:00:00.000001Z" });
});
test("filter and numeric sort keep unknown values last without altering source", () => {
  const rows = [{ entity_id: "sensor.a", kind: "binary", rows: 10 }, { entity_id: "sensor.b", kind: "binary", rows: 2 }, { entity_id: "other", kind: "continuous", rows: 5 }];
  assert.deepEqual(explorationRows(rows, "SENSOR", "binary", "rows", 1).map(r => r.rows), [2, 10]);
  assert.equal(rows[0].rows, 10);
});
