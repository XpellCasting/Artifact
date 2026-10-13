import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { once } from "node:events";
import { createServer } from "../../frontend/test-runtime.js";

test("the web interface proxies LAN health checks, uploads and API errors", async () => {
  const received = [];
  const backend = http.createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    received.push({ method: request.method, url: request.url, body: Buffer.concat(chunks).toString() });
    response.writeHead(request.url === "/failure" ? 503 : 200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ status: "ok" }));
  });
  backend.listen(0, "127.0.0.1");
  await once(backend, "listening");
  const previousTarget = process.env.ARTIFACT_API_TARGET;
  process.env.ARTIFACT_API_TARGET = `http://127.0.0.1:${backend.address().port}`;
  let web;
  try {
    web = await createServer({ server: { host: "127.0.0.1", port: 0 } });
    await web.listen();
    const origin = `http://127.0.0.1:${web.httpServer.address().port}`;
    const health = await fetch(`${origin}/api/health?check=1`, { headers: { Host: "192.0.2.10" } });
    assert.equal(health.status, 200);
    assert.deepEqual(await health.json(), { status: "ok" });
    const upload = await fetch(`${origin}/api/history/real/upload/start`, { method: "POST", body: "entity_id,state\nsensor.test,1\n" });
    assert.equal(upload.status, 200);
    assert.equal((await fetch(`${origin}/api/failure`)).status, 503);
    assert.deepEqual(received, [
      { method: "GET", url: "/health?check=1", body: "" },
      { method: "POST", url: "/history/real/upload/start", body: "entity_id,state\nsensor.test,1\n" },
      { method: "GET", url: "/failure", body: "" },
    ]);
    const constants = await web.ssrLoadModule("/src/shared/analysis-constants.js");
    assert.equal(constants.API_BASE_URL, "/api");
  } finally {
    if (previousTarget === undefined) delete process.env.ARTIFACT_API_TARGET;
    else process.env.ARTIFACT_API_TARGET = previousTarget;
    if (web) await web.close();
    backend.closeAllConnections();
    await new Promise((resolve) => backend.close(resolve));
  }
});
