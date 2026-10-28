import test from "node:test";
import assert from "node:assert/strict";
import { React, createServer, renderInteractive } from "../../frontend/test-runtime.js";

test("independent clients discover remote executions, progress and results and keep watching for the next run", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view;
  try {
    const { useFlowEvaluationPolling } = await server.ssrLoadModule("/src/features/flows/use-flow-evaluation-polling.js");
    let shared = { job: null, latest_run_id: "persisted-run" };
    const requests = [], completed = [[], []], changed = [[], []];
    globalThis.fetch = async (url, options) => {
      requests.push(url);
      assert.equal(options.cache, "no-store");
      return { ok: true, json: async () => structuredClone(shared) };
    };
    function Client({ index }) {
      const [state, setState] = React.useState(null);
      useFlowEvaluationPolling({ apiBaseUrl: "/api", discover: true, onUpdate: setState,
        onComplete: (value) => { completed[index].push(value); setState(null); },
        onRunsChanged: (id) => changed[index].push(id),
        onConnectionError: assert.fail, onFailure: assert.fail });
      return React.createElement("output", null, state ? `${state.flow.name}: ${state.processed}/${state.total}` : "Idle");
    }
    function Host() { return React.createElement(React.Fragment, null, React.createElement(Client, { index: 0 }), React.createElement(Client, { index: 1 })); }
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Host, {});
    assert.deepEqual(changed, [["persisted-run"], ["persisted-run"]]);
    shared.job = { job_id: "remote-csv", status: "running", flow: { name: "Remote flow" }, processed: 2, total: 10 };
    await React.act(async () => context.mock.timers.tick(1500));
    assert.deepEqual([...view.container.querySelectorAll("output")].map((item) => item.textContent), ["Remote flow: 2/10", "Remote flow: 2/10"]);
    shared.job.processed = 8;
    await React.act(async () => context.mock.timers.tick(1500));
    assert.deepEqual([...view.container.querySelectorAll("output")].map((item) => item.textContent), ["Remote flow: 8/10", "Remote flow: 8/10"]);
    shared = { job: { ...shared.job, status: "completed", run: { run_id: "new-run" } }, latest_run_id: "new-run" };
    await React.act(async () => context.mock.timers.tick(1500));
    await React.act(async () => context.mock.timers.tick(1500));
    assert.deepEqual(completed.map((items) => items.map((item) => item.run.run_id)), [["new-run"], ["new-run"]]);
    assert.deepEqual(changed, [["persisted-run", "new-run"], ["persisted-run", "new-run"]]);
    shared.job = { job_id: "remote-live", status: "running", mode: "home_assistant", flow: { name: "Live flow" }, processed: 0, total: 0 };
    await React.act(async () => context.mock.timers.tick(1500));
    assert.deepEqual([...view.container.querySelectorAll("output")].map((item) => item.textContent), ["Live flow: 0/0", "Live flow: 0/0"]);
    assert.ok(requests.every((url) => url === "/api/flow-evaluations"));
  } finally {
    if (view) await view.close();
    context.mock.timers.reset();
    globalThis.fetch = previousFetch;
    await server.close();
  }
});

test("discovery recovers from connection errors and reports an interrupted job once after a server restart", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view;
  try {
    const { useFlowEvaluationPolling } = await server.ssrLoadModule("/src/features/flows/use-flow-evaluation-polling.js");
    let shared = { job: { job_id: "live", status: "running" }, latest_run_id: null }, offline = false;
    const failures = [], updates = [], errors = [];
    globalThis.fetch = async () => {
      if (offline) throw new TypeError("offline");
      return { ok: true, json: async () => structuredClone(shared) };
    };
    function Host() {
      useFlowEvaluationPolling({ apiBaseUrl: "/api", discover: true, onUpdate: (state) => updates.push(state),
        onFailure: (value) => failures.push(value), onConnectionError: (value) => errors.push(value) });
      return null;
    }
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Host, {});
    offline = true;
    await React.act(async () => context.mock.timers.tick(1500));
    assert.equal(errors.length, 1);
    assert.equal(failures.length, 0);
    offline = false;
    await React.act(async () => context.mock.timers.tick(3000));
    assert.equal(updates.length, 2);
    shared.job = null;
    await React.act(async () => context.mock.timers.tick(1500));
    await React.act(async () => context.mock.timers.tick(1500));
    assert.equal(failures.length, 1);
    assert.match(failures[0], /reinició/);
    shared.job = { job_id: "next", status: "running" };
    await React.act(async () => context.mock.timers.tick(1500));
    assert.equal(updates.at(-1).job_id, "next");
  } finally {
    if (view) await view.close();
    context.mock.timers.reset();
    globalThis.fetch = previousFetch;
    await server.close();
  }
});

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

test("completed discovery fetches the saved snapshot only once while continuing to watch for executions", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view;
  try {
    const { useFlowEvaluationPolling } = await server.ssrLoadModule("/src/features/flows/use-flow-evaluation-polling.js");
    const requests = [], completed = [];
    globalThis.fetch = async (url) => {
      requests.push(url);
      return { ok: true, json: async () => url.endsWith("/done?compact=true")
        ? { job_id: "done", status: "completed", run: { run_id: "saved", snapshot: { flow: { name: "Shared" } } } }
        : { job: { job_id: "done", status: "completed", run_id: "saved" }, latest_run_id: "saved" } };
    };
    function Host() {
      useFlowEvaluationPolling({ apiBaseUrl: "/api", discover: true, onUpdate: assert.fail, onFailure: assert.fail,
        onConnectionError: assert.fail, onComplete: (state) => completed.push(state) });
      return null;
    }
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Host, {});
    await React.act(async () => context.mock.timers.tick(1500));
    await React.act(async () => context.mock.timers.tick(1500));
    assert.equal(completed.length, 1);
    assert.equal(completed[0].run.snapshot.flow.name, "Shared");
    assert.equal(requests.filter((url) => url === "/api/flow-evaluations/done?compact=true").length, 1);
    assert.equal(requests.filter((url) => url === "/api/flow-evaluations").length, 3);
  } finally {
    if (view) await view.close();
    context.mock.timers.reset();
    globalThis.fetch = previousFetch;
    await server.close();
  }
});

test("an older discovery response cannot replace a newly started job and a restart clears that job", async (context) => {
  const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  const previousFetch = globalThis.fetch;
  let view;
  try {
    const { useFlowEvaluationPolling } = await server.ssrLoadModule("/src/features/flows/use-flow-evaluation-polling.js");
    let resolveFirst, calls = 0;
    const failures = [], updates = [], completed = [];
    globalThis.fetch = async () => {
      calls++;
      if (calls === 1) return new Promise((resolve) => { resolveFirst = resolve; });
      return { ok: true, json: async () => ({ job: null, latest_run_id: null }) };
    };
    function Host() {
      const [jobId, setJobId] = React.useState(null);
      useFlowEvaluationPolling({ apiBaseUrl: "/api", discover: true, jobId,
        onUpdate: (value) => updates.push(value), onComplete: (value) => completed.push(value),
        onFailure: (value) => { failures.push(value); setJobId(null); }, onConnectionError: assert.fail });
      return React.createElement("button", { onClick: () => setJobId("just-started") }, jobId || "Idle");
    }
    context.mock.timers.enable({ apis: ["setTimeout"] });
    view = await renderInteractive(Host, {});
    await view.event(view.container.querySelector("button"), "click");
    await React.act(async () => resolveFirst({ ok: true, json: async () => ({ job: { job_id: "old", status: "completed" }, latest_run_id: "old-run" }) }));
    assert.equal(completed.length, 0);
    assert.equal(updates.length, 0);
    assert.equal(view.container.textContent, "just-started");
    await React.act(async () => context.mock.timers.tick(1500));
    await React.act(async () => context.mock.timers.tick(1500));
    assert.equal(failures.length, 1);
    assert.match(failures[0], /reinició/);
    assert.equal(view.container.textContent, "Idle");
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
