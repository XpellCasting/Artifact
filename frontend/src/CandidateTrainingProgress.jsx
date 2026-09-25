import React, { useEffect, useState } from "react";
import { Check, Loader2 } from "lucide-react";
import { estimateTrainingRemainingSeconds, formatRemainingTime } from "./diagnostic-progress.js";

export default function CandidateTrainingProgress({ job, onRetry, onShowCandidate }) {
  const [now, setNow] = useState(Date.now);
  const active = ["queued", "running"].includes(job?.status);
  useEffect(() => {
    if (!active) return undefined;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  if (!job) return null;
  const ready = job.status === "completed";
  const failed = job.status === "failed";
  const paused = job.tracking_paused;
  const saving = job.phase === "saving";
  const percent = ready ? 100 : Math.max(0, Math.min(99, Number(job.progress) || 0));
  const stagePercent = Math.max(0, Math.min(100, 100 * Number(job.batch || 0) / Number(job.batches || 1)));
  const remaining = estimateTrainingRemainingSeconds(job, now);
  const title = ready ? "Entrenamiento completado: candidato listo"
    : failed ? "El entrenamiento no se completó" : paused ? "Seguimiento interrumpido"
    : saving ? "Guardando candidato…" : "Entrenando candidato…";
  return <section className={`diagnostic-dataset-progress ${ready ? "ready" : ""} ${failed ? "failed" : ""}`} aria-live="polite" aria-busy={active && !paused}>
    <div className="diagnostic-dataset-progress-heading">
      <strong>{ready ? <Check size={18} /> : active && !paused ? <Loader2 size={18} className="diagnostic-spin" /> : null}{title}</strong>
      <span>{percent.toLocaleString("es-CL", { maximumFractionDigits: 1 })} %</span>
    </div>
    <progress max="100" value={percent} aria-label="Avance total del entrenamiento del candidato" />
    <p>{ready ? "El modelo quedó guardado y está disponible en Modelos y evaluación. Su activación es un paso posterior."
      : failed ? job.error || job.message
      : paused ? "No se pudo consultar el progreso. Reintenta el seguimiento sin iniciar otro entrenamiento."
      : job.message || "Iniciando el entrenamiento…"}</p>
    {active && !paused && job.phase === "training" && Number(job.batches) > 0 && <>
      <small>{job.stage === "dev" ? "Evaluación en Desarrollo" : "Ajuste de pesos"} · lote {Number(job.batch || 0).toLocaleString("es-CL")} de {Number(job.batches).toLocaleString("es-CL")}</small>
      <progress max="100" value={stagePercent} aria-label="Avance de lotes de la etapa actual" />
    </>}
    {active && !paused && job.phase === "training" && <small>
      {Number(job.epoch) > 0 ? `Época ${job.epoch} de un máximo de ${job.max_epochs}. ` : ""}
      {remaining == null ? "Estimando el tiempo restante…" : `Tiempo estimado hasta el máximo de épocas: ${formatRemainingTime(remaining)}.`}
      {" "}Puede terminar antes si deja de mejorar.
    </small>}
    {active && !paused && <small>Estará listo cuando termine de guardarse y aparezca «Candidato listo» al 100 %.</small>}
    {paused && <button className="diagnostic-button" onClick={onRetry}>Reintentar seguimiento</button>}
    {ready && job.result?.model_id && <button className="diagnostic-button" onClick={onShowCandidate}>Ver candidato</button>}
  </section>;
}
