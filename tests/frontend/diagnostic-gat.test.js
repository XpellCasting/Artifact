import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_GAT_CONFIG, gatContextSeconds, gatDataConfigCompatible, gatSupportStatus, isGatConfigValid } from "../../frontend/src/features/diagnostics/diagnostic-gat.js";

test("GAT defaults expose the planned causal CPU configuration", () => {
  assert.equal(isGatConfigValid(DEFAULT_GAT_CONFIG), true);
  assert.equal(gatContextSeconds(DEFAULT_GAT_CONFIG), 259200);
  assert.equal(DEFAULT_GAT_CONFIG.device, "cpu");
  assert.equal(DEFAULT_GAT_CONFIG.pretrain_enabled, false);
});

test("the supervised action requires three classes and episode support in fit and tune", () => {
  assert.equal(gatSupportStatus(null).ready, false);
  const complete = Object.fromEntries(["train_fit", "dev_tune"].map((partition) => [partition,
    { normal: 1, environmental_anomaly: 2, sensor_fault: 1 }]));
  assert.equal(gatSupportStatus({ label_counts: complete, episode_counts: complete }).ready, true);
  const missingEpisodes = { ...complete, dev_tune: { ...complete.dev_tune, sensor_fault: 0 } };
  assert.equal(gatSupportStatus({ label_counts: complete, episode_counts: missingEpisodes }).ready, false);
});

test("the visible architecture rejects a head dimension other than 32", () => {
  assert.equal(isGatConfigValid({ ...DEFAULT_GAT_CONFIG, gat_heads: 2, gat_head_dim: 16 }), true);
  assert.equal(isGatConfigValid({ ...DEFAULT_GAT_CONFIG, gat_heads: 3, gat_head_dim: 8 }), false);
  assert.equal(isGatConfigValid({ ...DEFAULT_GAT_CONFIG, sequence_steps: 1 }), false);
});

test("temporal and graph changes require a newly prepared dataset", () => {
  const dataset = { config: { ...DEFAULT_GAT_CONFIG } };
  assert.equal(gatDataConfigCompatible(DEFAULT_GAT_CONFIG, dataset), true);
  assert.equal(gatDataConfigCompatible({ ...DEFAULT_GAT_CONFIG, graph_top_k: 3 }, dataset), false);
  assert.equal(gatDataConfigCompatible({ ...DEFAULT_GAT_CONFIG, pretrain_enabled: true }, dataset), false);
});

test("retired sensor architecture is rejected", () => {
  assert.equal(isGatConfigValid({ ...DEFAULT_GAT_CONFIG, architecture_version: "gat_lstm_v2_shared_type" }), false);
});
