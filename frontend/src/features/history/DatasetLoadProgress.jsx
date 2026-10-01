import React from "react";
import { Check, Loader2 } from "lucide-react";

export default function DatasetLoadProgress({ job, onRetry }) {
  if (!job) return null;
  const ready = job.status === "completed";
  const failed = job.status === "failed";
  const paused = job.tracking_paused;
  const percent = ready ? 100 : Math.max(0, Math.min(99, Number(job.progress) || 0));
  const title = ready ? "Dataset listo" : failed ? "No se pudo preparar el dataset"
    : paused ? "Seguimiento interrumpido" : "Preparando dataset…";
  return <section className={`diagnostic-dataset-progress ${ready ? "ready" : ""} ${failed ? "failed" : ""}`} aria-live="polite" aria-busy={!ready && !failed && !paused}>
    <div className="diagnostic-dataset-progress-heading">
      <strong>{ready ? <Check size={18} /> : !failed && !paused ? <Loader2 size={18} className="diagnostic-spin" /> : null}{title}</strong>
      <span>{Math.round(percent)} %</span>
    </div>
    <progress max="100" value={percent} aria-label="Avance de preparación del dataset" />
    <p>{ready ? "Las secuencias se guardaron y el dataset ya está disponible para entrenar, si tiene las clases requeridas."
      : failed ? job.error || job.message
      : paused ? "No se pudo consultar el avance. Reintenta el seguimiento para saber cuándo termina."
      : job.message || "Iniciando la preparación del dataset…"}</p>
    {!ready && !failed && !paused && <small>Al terminar aparecerá «Dataset listo» y la barra llegará al 100 %.</small>}
    {paused && <button className="diagnostic-button" onClick={onRetry}>Reintentar seguimiento</button>}
  </section>;
}
