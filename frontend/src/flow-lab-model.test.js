import test from "node:test";
import assert from "node:assert/strict";
import { templates, validateDraft, connect, removeNode, fingerprint, clone } from "./flow-lab-model.js";

test("all researcher templates are valid graphs", () => {
  for (const flow of templates()) assert.deepEqual(validateDraft(flow), [], flow.name);
});
test("continuous-method validation preserves legacy quantiles and accepts the new methods", () => {
  const flow = templates()[0];
  flow.nodes.find((node) => node.type === "statistical").config.continuous_method = "legacy";
  assert.ok(validateDraft(flow).some((message) => message.includes("método")));
  delete flow.nodes.find((node) => node.type === "statistical").config.continuous_method;
  assert.deepEqual(validateDraft(flow), []);
  const residual = templates().find((item) => item.name.includes("Residuos + EWMA"));
  const focus = templates().find((item) => item.name === "Solo estadístico FOCuS");
  assert.deepEqual(validateDraft(residual), []);
  assert.deepEqual(validateDraft(focus), []);
  focus.nodes.find((node) => node.type === "statistical_focus").config.continuous_method = "residual_ewma_v1";
  assert.ok(validateDraft(focus).some((message) => message.includes("FOCuS")));
});
test("connections reject cycles, duplicates and implicit branch convergence", () => {
  const flow = templates()[0];
  assert.equal(connect(flow, "input", "statistical"), null);
  assert.equal(connect(flow, "classifier", "statistical"), null);
  assert.equal(connect(flow, "input", "classifier"), null);
  assert.equal(connect(flow, "output", "input"), null);
});
test("deleting a filter removes incident edges and reports the broken path", () => {
  const flow = templates()[2];
  const result = removeNode(flow, "statistical");
  assert.equal(result.edges.length, 0);
  assert.ok(validateDraft(result).length);
  assert.equal(removeNode(flow, "input"), flow);
});
test("empty conditions and cleared thresholds cannot be saved", () => {
  const flow = templates()[2];
  flow.edges[0].when = [];
  flow.nodes[1].config.suspicion_threshold = null;
  assert.equal(validateDraft(flow).length, 2);
});
test("unsaved connection changes differ from the saved snapshot", () => {
  const saved = templates()[2];
  const draft = clone(saved);
  draft.edges[0].when = ["positive"];
  assert.notEqual(fingerprint(saved), fingerprint(draft));
  assert.equal(saved.edges[0].when, undefined);
});
