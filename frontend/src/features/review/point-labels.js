import { reviewMicros } from "./review-time.js";
export const POINT_FAULTS = {
  spike: "Pico", bias: "Desplazamiento", drift: "Deriva", noise: "Ruido",
  stuck: "Valor congelado", unavailable: "Indisponibilidad", state_replace: "Estado sustituido",
};
export const pointFaultKinds = kind => kind === "continuous" ? ["spike", "bias", "drift", "noise", "stuck", "unavailable"] : ["stuck", "unavailable", "state_replace"];
export function annotationForPoint(annotations, entity, point) {
  const matching = point.source_row == null ? [] : annotations.filter(a => a.entity_id === entity && a.source_row === point.source_row);
  if (matching.length) return matching[0];
  const at = reviewMicros(point.timestamp);
  return annotations.find(a => a.entity_id === entity && a.source_row == null && a.evaluable && at !== null && reviewMicros(a.start) <= at && reviewMicros(a.end) >= at) || point.imported_label;
}
