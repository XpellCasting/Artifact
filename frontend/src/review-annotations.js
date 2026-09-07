export const annotationKey = (annotation) => annotation?.annotation_id || annotation?.id || "";

export function injectionBaselineProblems(annotation) {
  if (!annotation) return ["Selecciona una anotación guardada."];
  const problems = [];
  if (annotation.label !== "normal") problems.push("La clase debe ser Normal.");
  if (String(annotation.reviewer || "").trim().length < 2) problems.push("Completa el responsable de la revisión (al menos 2 caracteres).");
  if (String(annotation.evidence || "").trim().length < 12) problems.push("Completa Evidencia y referencia verificable (al menos 12 caracteres); las notas no sustituyen ese campo.");
  return problems;
}

// A delayed list response must not erase a save or replace its newer revision.
export function mergeAnnotations(current, incoming, datasetId) {
  const byId = new Map();
  for (const annotation of [...current, ...incoming]) {
    if (annotation.dataset_id !== datasetId) continue;
    const id = annotationKey(annotation);
    const previous = byId.get(id);
    if (id && (!previous || Number(annotation.revision) >= Number(previous.revision))) byId.set(id, annotation);
  }
  return [...byId.values()].sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
}
