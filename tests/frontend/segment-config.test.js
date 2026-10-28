import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_SEGMENT_CONFIG, SEGMENT_SCALES, gatContextSeconds, gatDataConfigCompatible, isGatConfigValid } from "../../frontend/src/features/diagnostics/diagnostic-gat.js";
import { templates } from "../../frontend/src/features/flows/flow-lab-model.js";

test("segment models retain 72-hour context without requiring minute reports", () => {
  assert.equal(isGatConfigValid(DEFAULT_SEGMENT_CONFIG), true);
  assert.equal(gatContextSeconds(DEFAULT_SEGMENT_CONFIG), 259200);
  assert.deepEqual(SEGMENT_SCALES.map(scale => scale.hours), [1,24,72]);
  assert.equal(gatDataConfigCompatible(DEFAULT_SEGMENT_CONFIG,{config:DEFAULT_SEGMENT_CONFIG}),true);
  assert.equal(gatDataConfigCompatible({...DEFAULT_SEGMENT_CONFIG,local_timezone:"UTC"},{config:DEFAULT_SEGMENT_CONFIG}),false);
});

test("new templates compose filter evidence and preserve logical gate results", () => {
  for (const flow of templates()) for (const node of flow.nodes) {
    assert.equal(node.config.evidence_policy, ["merge", "decision"].includes(node.type) ? "independent" : "combine_v1");
  }
});
