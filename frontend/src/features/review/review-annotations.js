export const annotationKey = (annotation) => annotation?.annotation_id || annotation?.id || "";
export const reviewStatus = annotation => ["reviewed", "verified"].includes(annotation?.status) ? "reviewed" : "pending";
export const reviewStatusLabel = annotation => reviewStatus(annotation) === "reviewed" ? "Revisado" : "Pendiente";

export function injectionBaselineProblems(annotation) {
  if (!annotation) return ["Selecciona una anotación guardada."];
  const problems = [];
  if (annotation.label !== "normal") problems.push("La clase debe ser Normal.");
  if (reviewStatus(annotation) !== "reviewed") problems.push("Marca el fragmento como Revisado y guarda los cambios.");
  if (String(annotation.reviewer || "").trim().length < 2) problems.push("Completa el responsable de la revisión (al menos 2 caracteres).");
  if (String(annotation.evidence || "").trim().length < 12) problems.push("Completa Observaciones que respaldan la interpretación (al menos 12 caracteres); las notas no sustituyen ese campo.");
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
// Point labels remain persisted and exported; only interval notes use the list.
export function splitReviewAnnotations(annotations) {
  return {
    points: annotations.filter(annotation => annotation.source_row != null),
    intervals: annotations.filter(annotation => annotation.source_row == null),
  };
}
