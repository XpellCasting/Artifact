import test from "node:test";
import assert from "node:assert/strict";
import { mergeAnnotations, injectionBaselineProblems } from "./review-annotations.js";

const baseline = { annotation_id: "a", dataset_id: "history", label: "normal", reviewer: "Revisor", evidence: "Inspección visual documentada.", status: "draft", revision: 1, updated_at: "2026-01-01T00:00:00Z" };

test("una lista atrasada no borra una anotación recién guardada ni restaura una revisión anterior", () => {
  const edited = { ...baseline, revision: 2, evidence: "Referencia corregida y documentada." };
  const created = { ...baseline, annotation_id: "new" };
  const result = mergeAnnotations([edited, created], [baseline], "history");
  assert.equal(result.length, 2);
  assert.equal(result.find((a) => a.annotation_id === "a").revision, 2);
  assert.equal(result.find((a) => a.annotation_id === "a").evidence, edited.evidence);
  assert.ok(result.some((a) => a.annotation_id === "new"));
});

test("las anotaciones incompletas e inciertas siguen visibles, con razones para no generar", () => {
  const incomplete = { ...baseline, evidence: "", notes: "Notas extensas que no equivalen al campo de evidencia." };
  const uncertain = { ...baseline, annotation_id: "uncertain", label: "uncertain" };
  const result = mergeAnnotations([], [incomplete, uncertain], "history");
  assert.equal(result.length, 2);
  assert.match(injectionBaselineProblems(incomplete).join(" "), /Evidencia y referencia verificable/);
  assert.match(injectionBaselineProblems(uncertain).join(" "), /Normal/);
});

test("completar y guardar evidencia habilita el borrador sin exigir verificación", () => {
  const incomplete = { ...baseline, evidence: "" };
  assert.equal(injectionBaselineProblems(incomplete).length, 1);
  const updated = mergeAnnotations([incomplete], [{ ...baseline, revision: 2 }], "history");
  assert.equal(updated.length, 1);
  assert.deepEqual(injectionBaselineProblems(updated[0]), []);
  assert.equal(updated[0].status, "draft");
});

test("el cambio de histórico no conserva anotaciones de otra partición", () => {
  const other = { ...baseline, annotation_id: "other", dataset_id: "other-history" };
  assert.deepEqual(mergeAnnotations([baseline], [other], "other-history"), [other]);
});
