import React from "react";
import { POINT_FAULTS, pointFaultKinds } from "./point-labels.js";
import { reviewUtcTimestamp } from "./review-time.js";

export default function PointLabeler({ selection, annotations, reviewer, onReviewer, onSave, disabled }) {
  if (!selection) return <p className="review-help">Pulsa una lectura para marcarla como Normal, Anomalía o Falla. Cada marca afecta solo a ese registro.</p>;
  const { series, point } = selection;
  const existing = annotations.find(a => a.entity_id === series.entity_id && a.source_row === point.source_row) || point.imported_label;
  return <section className="review-point-editor" aria-label="Etiquetar registro seleccionado">
    <strong>{series.entity_id} · {reviewUtcTimestamp(point.timestamp)} · Valor: {String(point.state)}</strong>
    <p className="review-help">Fila original {point.source_row}. {existing ? `Etiqueta guardada: ${POINT_FAULTS[existing.fault_type] || ({ normal: "Normal", environmental_anomaly: "Anomalía", sensor_fault: "Falla", uncertain: "Incierto" })[existing.label]}.` : "Selecciona una etiqueta para esta lectura."}</p>
    <label>Responsable de la marcación<input value={reviewer || "Revisor local"} disabled={disabled} onChange={e => onReviewer(e.target.value)} /></label>
    <div className="review-actions"><button type="button" className="review-button" disabled={disabled} onClick={() => onSave("normal")}>Normal</button><button type="button" className="review-button" disabled={disabled} onClick={() => onSave("environmental_anomaly")}>Anomalía</button></div>
    <details className="review-point-faults"><summary aria-disabled={disabled} onClick={event => { if (disabled) event.preventDefault(); }}>Falla → Tipo de falla</summary><div className="review-actions">{pointFaultKinds(series.kind).map(kind => <button type="button" className="review-button" key={kind} disabled={disabled} onClick={() => onSave("sensor_fault", kind)}>{POINT_FAULTS[kind]}</button>)}</div></details>
  </section>;
}
