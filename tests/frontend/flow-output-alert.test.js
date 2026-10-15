import test from "node:test";
import assert from "node:assert/strict";
import { React, createServer, renderToStaticMarkup } from "../../frontend/test-runtime.js";
import { flowOutputAlert } from "../../frontend/src/features/flows/flow-output-alert.js";

const diagnosis = (fault_type, fault_type_status = "identified") => ({
  status: "compatible", data: { fault_type, fault_type_status },
});

test("a fault displays its identified diagnosis, including older saved diagnoses", () => {
  for (const legacy of [false, true]) {
    const entry = diagnosis("drift");
    if (legacy) delete entry.data.fault_type_status;
    assert.deepEqual(flowOutputAlert({ final_decision: "fault_compatible",
      evidence: [entry] }), {
      severity: "fault", message: "Falla · Posible falla: Deriva",
    });
  }
});

test("an untyped fault shows only Falla and never borrows its expected CSV label or an earlier diagnosis", () => {
  for (const entry of [diagnosis(null), diagnosis("drift", "unknown"), diagnosis("drift", "not_supported"), diagnosis("unsupported")]) {
    assert.deepEqual(flowOutputAlert({ final_decision: "fault_compatible",
      evidence: [diagnosis("spike"), entry], evaluation: { fault_type: "bias", type_identified: true } }), {
      severity: "fault", message: "Falla",
    });
  }
  assert.equal(flowOutputAlert({ final_decision: "fault_compatible" }).message, "Falla");
});

test("combined faults show only identified types from evidence used by the final evaluation", () => {
  const entries = [
    { ...diagnosis("drift"), evidence_id: "used" },
    { ...diagnosis("spike"), evidence_id: "ignored" },
    { ...diagnosis("drift"), evidence_id: "duplicate" },
  ];
  assert.deepEqual(flowOutputAlert({ final_decision: "fault_compatible", evidence_policy: "combine_v1",
    combined_evaluation: { used_evidence_ids: ["used", "duplicate"] }, evidence: entries }), {
    severity: "fault", message: "Falla · Posible falla: Deriva",
  });
});

test("anomalies and unresolved suspicions show only the yellow anomaly message", () => {
  for (const final_decision of ["environmental", "suspicion"]) {
    assert.deepEqual(flowOutputAlert({ final_decision, evidence: [diagnosis("drift")],
      evaluation: { expected: "Falla", fault_type: "spike" } }), {
      severity: "anomaly", message: "Se encontró una anomalía.",
    });
  }
});

test("normal, indeterminate, unrouted and intermediate readings do not emit detection alerts", () => {
  for (const final_decision of ["normal", "no_alert", "indeterminate", "not_routed", undefined]) {
    assert.equal(flowOutputAlert({ final_decision, signal: "positive", evidence: [diagnosis("drift")] }), null);
  }
});

test("output results render accessible red and yellow notices and preserve intermediate results", async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  try {
    const { default: Result } = await server.ssrLoadModule("/src/features/flows/FlowOutputResult.jsx");
    const fault = renderToStaticMarkup(React.createElement(Result, { item: {
      final_decision: "fault_compatible", evidence: [diagnosis("stuck")],
    } }));
    assert.match(fault, /role="alert"/);
    assert.match(fault, /fl-output-alert-fault/);
    assert.match(fault, /Falla · Posible falla: Valor congelado/);
    const anomaly = renderToStaticMarkup(React.createElement(Result, { item: { final_decision: "environmental" } }));
    assert.match(anomaly, /fl-output-alert-anomaly/);
    assert.match(anomaly, /Se encontró una anomalía\./);
    for (const props of [{ item: { final_decision: "normal" } },
      { item: { final_decision: "fault_compatible" }, alertEnabled: false }]) {
      const html = renderToStaticMarkup(React.createElement(Result, props));
      assert.doesNotMatch(html, /role="alert"/);
      assert.match(html, /falla/);
    }
  } finally { await server.close(); }
});
