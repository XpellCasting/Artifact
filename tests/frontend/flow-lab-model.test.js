import test from "node:test";
import assert from "node:assert/strict";
import { templates, validateDraft, connect, removeNode, fingerprint, clone } from "../../frontend/src/features/flows/flow-lab-model.js";

test("all researcher templates are valid graphs", () => {
  for (const flow of templates()) assert.deepEqual(validateDraft(flow), [], flow.name);
});
test("only FOCuS is offered and retired methods cannot be saved", () => {
  const all = templates();
  assert.ok(all.every(flow => flow.nodes.every(node => node.type !== "statistical")));
  assert.ok(all.every(flow => !/EWMA|Comparación estadística/.test(flow.name)));
  const flow = all[0];
  const node = flow.nodes.find(node => node.type === "statistical_focus");
  for (const method of ["temporal_quantile_v1", "residual_ewma_v1"]) {
    node.config.continuous_method = method;
    assert.ok(validateDraft(flow).some(message => message.includes("FOCuS")));
  }
  delete node.config.continuous_method;
  assert.deepEqual(validateDraft(flow), []);
});
test("connections reject cycles, duplicates and implicit branch convergence", () => {
  const flow = templates()[0];
  assert.equal(connect(flow, "input", "statistical_focus"), null);
  assert.equal(connect(flow, "classifier", "statistical_focus"), null);
  assert.equal(connect(flow, "input", "classifier"), null);
  assert.equal(connect(flow, "output", "input"), null);
});
test("deleting a filter removes incident edges and reports the broken path", () => {
  const flow = templates()[2];
  const result = removeNode(flow, "statistical_focus");
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
