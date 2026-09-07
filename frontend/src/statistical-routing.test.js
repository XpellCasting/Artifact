import test from "node:test";
import assert from "node:assert/strict";
import { isStatisticalSuspicion, statisticalCounts, episodeLabel } from "./statistical-routing.js";

test("la marca de F1 no depende del indice del array ni del umbral editado", () => {
  const values = Array.from({ length: 10 }, () => ({ is_suspicious: true, suspicion_score: .5 }));
  assert.equal(values.filter(isStatisticalSuspicion).length, 10);
});

test("los contadores usan los totales globales aunque solo se carguen 500 filas", () => {
  const repeated = { is_suspicious: true, suppressed_by_episode: true, episode: { occurrences: 8 } };
  assert.deepEqual(statisticalCounts([repeated]), { detected: 1, forwarded: 0, suppressed: 1 });
  assert.deepEqual(statisticalCounts([repeated], { grouping: { detected: 7039, forwarded: 2039, suppressed: 5000 } }),
    { detected: 7039, forwarded: 2039, suppressed: 5000 });
  assert.match(episodeLabel(repeated), /agrupada.*8/);
});
