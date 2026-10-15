import test from "node:test";
import assert from "node:assert/strict";
import { React, createServer, renderInteractive } from "../../frontend/test-runtime.js";

test("a temporary API outage keeps polling the same reception until it completes", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view;
  try {
    const { useFlowEvaluationPolling } = await server.ssrLoadModule("/src/features/flows/use-flow-evaluation-polling.js");
    const requests = [], updates = [], errors = [], failures = [], completed = [];
    globalThis.fetch = async (url) => {
      requests.push(url);
      if (requests.length === 1) throw new TypeError("network interrupted");
      return { ok: true, json: async () => ({ job_id: "reception-one", status: requests.length === 2 ? "running" : "completed" }) };
    };
    function Host() {
      useFlowEvaluationPolling({ apiBaseUrl: "/api", jobId: "reception-one", onUpdate: (value) => updates.push(value),
        onConnectionError: (value) => errors.push(value), onFailure: (value) => failures.push(value), onComplete: (value) => completed.push(value) });
      return React.createElement("output", null, "Reception");
    }
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Host, {});
    assert.equal(errors.length, 1);
    assert.equal(failures.length, 0);
    await React.act(async () => context.mock.timers.tick(3000));
    assert.equal(updates[0].status, "running");
    await React.act(async () => context.mock.timers.tick(1500));
    assert.equal(completed[0].status, "completed");
    await React.act(async () => context.mock.timers.tick(10000));
    assert.deepEqual(requests, Array(3).fill("/api/flow-evaluations/reception-one"));
    assert.equal(failures.length, 0);
  } finally {
    if (view) await view.close();
    context.mock.timers.reset();
    globalThis.fetch = previousFetch;
    await server.close();
  }
});

test("a definitive server failure or a lost job shows a notice and ends polling", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view;
  try {
    const { useFlowEvaluationPolling } = await server.ssrLoadModule("/src/features/flows/use-flow-evaluation-polling.js");
    for (const missing of [false, true]) {
      let calls = 0;
      const failures = [];
      globalThis.fetch = async () => { calls++; return { ok: !missing, status: missing ? 404 : 200,
        json: async () => missing ? { detail: "Not found" } : { status: "failed", error: "El filtro clasificador no tiene un modelo disponible." } }; };
      function Host() {
        useFlowEvaluationPolling({ apiBaseUrl: "/api", jobId: "one", onFailure: (value) => failures.push(value),
          onConnectionError: () => assert.fail("A definitive failure must be visible"), onUpdate: () => {}, onComplete: () => {} });
        return null;
      }
      context.mock.timers.enable({ apis: ["setTimeout"] });
      view = await renderInteractive(Host, {});
      assert.equal(failures.length, 1);
      assert.match(failures[0], missing ? /reinició/ : /clasificador/);
      await React.act(async () => context.mock.timers.tick(10000));
      assert.equal(calls, 1);
      await view.close(); view = undefined;
      context.mock.timers.reset();
    }
  } finally {
    if (view) await view.close();
    context.mock.timers.reset();
    globalThis.fetch = previousFetch;
    await server.close();
  }
});
