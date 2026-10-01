import test from "node:test";
import assert from "node:assert/strict";
import { estimateTrainingRemainingSeconds, formatRemainingTime } from "../../frontend/src/features/diagnostics/diagnostic-progress.js";

test("estimates GAT training time from completed epochs and returns no estimate before the first epoch", () => {
  const startedAt = Date.parse("2026-09-24T12:00:00Z");
  const job = { phase: "training", epoch: 2, max_epochs: 10,
    phase_started_at: new Date(startedAt).toISOString() };
  assert.equal(estimateTrainingRemainingSeconds({ ...job, epoch: 0 }, startedAt + 10_000), null);
  assert.equal(estimateTrainingRemainingSeconds(job, startedAt + 10_000), 40);
  assert.equal(estimateTrainingRemainingSeconds({ ...job, phase: "saving" }, startedAt + 10_000), null);
});

test("formats remaining time in seconds, minutes and hours", () => {
  assert.equal(formatRemainingTime(32), "32 s");
  assert.equal(formatRemainingTime(360), "6 min");
  assert.equal(formatRemainingTime(3720), "1 h 2 min");
});
