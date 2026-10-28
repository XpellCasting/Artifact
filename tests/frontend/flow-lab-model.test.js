import test from "node:test";
import assert from "node:assert/strict";
import { templates, validateDraft, connect, defaultNodeConfig, removeNode, fingerprint, clone, nextNodeId } from "../../frontend/src/features/flows/flow-lab-model.js";

test("adding multiple nodes preserves distinct IDs in a restored flow without browser cryptography", () => {
  const previousCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: {} });
  try {
    const flow = clone(templates()[0]);
    flow.nodes.push({ id: "statistical_focus_1", type: "statistical_focus" },
      { id: "statistical_focus_3", type: "statistical_focus" },
      { id: "decision_a1b2c3d4", type: "decision" });
    const originalIds = flow.nodes.map((node) => node.id);
    for (const type of ["statistical_focus", "statistical_focus", "decision", "contextual", "classifier", "merge"]) {
      const id = nextNodeId(flow, type);
      assert.ok(!flow.nodes.some((node) => node.id === id));
      flow.nodes.push({ id, type });
    }
    assert.deepEqual(flow.nodes.slice(0, originalIds.length).map((node) => node.id), originalIds);
    assert.equal(new Set(flow.nodes.map((node) => node.id)).size, flow.nodes.length);
  } finally {
    if (previousCrypto) Object.defineProperty(globalThis, "crypto", previousCrypto);
    else delete globalThis.crypto;
  }
});

test("all researcher templates are valid graphs", () => {
  for (const flow of templates()) assert.deepEqual(validateDraft(flow), [], flow.name);
});

test("logical nodes and comparison templates use their own logical result", () => {
  assert.equal(defaultNodeConfig("merge").mode, "any");
  assert.deepEqual(defaultNodeConfig("decision").accept, ["positive"]);
  for (const flow of templates()) {
    for (const node of flow.nodes.filter(node => ["merge", "decision"].includes(node.type))) {
      assert.equal(node.config.evidence_policy, "independent", flow.name);
    }
  }
  const all = templates().find(flow => flow.name === "Tres filtros: A Y B Y C");
  assert.equal(all.nodes.find(node => node.type === "merge").config.mode, "all");
  const composed = templates().find(flow => flow.name === "Tres filtros: (A O B) Y C");
  assert.equal(composed.nodes.find(node => node.id === "any").config.mode, "any");
  assert.equal(composed.nodes.find(node => node.id === "all").config.mode, "all");
  assert.ok(composed.edges.some(edge => edge.source === "any" && edge.target === "all"));
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
