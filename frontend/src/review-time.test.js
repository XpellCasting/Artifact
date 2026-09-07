import test from "node:test";
import assert from "node:assert/strict";
import { reviewUtcTimestamp, validReviewInterval, initialReviewRange, shiftReviewRange } from "./review-time.js";

test("UTC conversion retains microseconds and rejects impossible calendar dates", () => {
  assert.equal(reviewUtcTimestamp("2026-05-01T12:00:00.123456-03:00"), "2026-05-01T15:00:00.123456Z");
  assert.equal(reviewUtcTimestamp("2026-05-01T12:30"), "2026-05-01T12:30:00.000000Z");
  assert.equal(reviewUtcTimestamp("2026-02-30T12:00:00Z"), "");
  assert.equal(reviewUtcTimestamp("2026-05-01T12:00:00.1234567Z"), "");
  assert.equal(reviewUtcTimestamp("1969-12-31T23:59:59.999999Z"), "1969-12-31T23:59:59.999999Z");
});

test("selection permits distinct instants within one millisecond and a single observation", () => {
  assert.equal(validReviewInterval("2026-05-01T00:00:00.123456Z", "2026-05-01T00:00:00.123457Z"), true);
  assert.equal(validReviewInterval("2026-05-01T00:00:00.123457Z", "2026-05-01T00:00:00.123456Z"), false);
  assert.equal(validReviewInterval("2026-05-01T00:00:00Z", "2026-05-01T00:00:00Z"), true);
});

test("initial and shifted windows preserve submilliseconds and stay inside the dataset", () => {
  const low = "2026-05-01T00:00:00.123456Z";
  const high = "2026-05-01T01:30:00.123456Z";
  const initial = initialReviewRange(low, high);
  assert.deepEqual(initial, { start: low, end: "2026-05-01T01:00:00.123456Z" });
  const next = shiftReviewRange(initial.start, initial.end, low, high, 1);
  assert.deepEqual(next, { start: "2026-05-01T00:30:00.123456Z", end: high });
  assert.equal(shiftReviewRange(next.start, next.end, low, high, 1), null);
  assert.deepEqual(shiftReviewRange(next.start, next.end, low, high, -1), initial);
  assert.equal(shiftReviewRange(initial.start, initial.end, low, high, -1), null);
});
