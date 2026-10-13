import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Download,
  X,
  Database,
  Eye,
  FlaskConical,
  GitBranch,
  Loader2,
  RotateCcw,
  Zap,
} from "lucide-react";
import { downloadCsv } from "../../shared/csv.js";
import ReviewWorkspace from "../review/ReviewWorkspace.jsx";
import DiagnosticWorkspace from "../diagnostics/DiagnosticWorkspace.jsx";
import FlowLab from "../flows/FlowLab.jsx";
import { isStatisticalSuspicion, statisticalCounts, episodeLabel } from "../../shared/statistical-routing.js";
import { isEnvironmentalResult, contextRoutingLabel, contextRoutingReasonLabel, contextRoutingCounts, contextCoherenceLabel, contextObservationLabel, preciseContextTimestamp, contextCommonCause } from "../../shared/context-routing.js";

import { compareResultsByExecutionDate, formatDuration, formatBytes } from "../../shared/analysis.jsx";
import { API_BASE_URL } from "../../shared/analysis-constants.js";

export function replayResultKey(result) {
  return result?.event_id || `${result?.entity_id || ""}@${result?.timestamp || ""}`;
}


export function mergeReplayResults(current, incoming, limit = 500) {
  const merged = new Map();
  [...(incoming || []), ...(current || [])].forEach((item) => {
    merged.set(replayResultKey(item), item);
  });
  return [...merged.values()].sort(compareResultsByExecutionDate).slice(0, limit);
}


export function startHistoryUpload(file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE_URL}/history/real/upload/start`);
    request.setRequestHeader("Content-Type", "text/csv");
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded, event.total);
      }
    };
    request.onerror = () => reject(new Error("No se pudo conectar con la API para subir el historico."));
    request.onload = () => {
      const payload = (() => {
        try {
          return JSON.parse(request.responseText || "{}");
        } catch {
          return {};
        }
      })();
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(payload.detail || `HTTP ${request.status}`));
        return;
      }
      resolve(payload);
    };
    request.send(file);
  });
}


export function startValidationUpload(file, onProgress, onProcessing) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE_URL}/validation/replay/upload`);
    request.setRequestHeader("Content-Type", "text/csv");
    request.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded, event.total);
      }
    };
    request.upload.onload = () => onProcessing?.();
    request.onerror = () => reject(new Error("No se pudo conectar con la API para subir Evaluacion."));
    request.onload = () => {
      const payload = (() => {
        try {
          return JSON.parse(request.responseText || "{}");
        } catch {
          return {};
        }
      })();
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(payload.detail || `HTTP ${request.status}`));
        return;
      }
      resolve(payload);
    };
    request.send(file);
  });
}


export function HistoryLoadProgress({ job }) {
  const progress = Math.min(100, Math.max(0, Number(job?.progress || 0)));
  const ready = job?.status === "completed";
  const failed = job?.status === "failed";
  return (
    <section
      className={`history-load-progress ${ready ? "ready" : ""} ${failed ? "failed" : ""}`}
      aria-live="polite"
    >
      <div className="history-load-heading">
        <span>{ready ? "Historico listo" : failed ? "Carga fallida" : historyLoadPhaseLabel(job?.phase)}</span>
        <strong>{Number(progress.toFixed(1))}%</strong>
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-label="Progreso de carga del historico"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={Math.round(progress)}
      >
        <span style={{ width: `${progress}%` }} />
      </div>
      <p>{job?.error || job?.message || "Preparando carga"}</p>
      {!ready && !failed && <small>El porcentaje indica etapas completadas; cada etapa puede tardar un tiempo diferente.</small>}
      <div className="history-load-meta">
        <span>Transcurrido: {formatDuration(job?.elapsed_seconds)}</span>
        <span>En esta fase: {formatDuration(job?.phase_elapsed_seconds)}</span>
        <span>Último avance hace: {formatDuration(job?.seconds_since_progress)}</span>
        <span>Restante: {ready ? "0 s" : formatDuration(job?.eta_seconds, "sin estimación fiable")}</span>
        <span>Archivo: {formatBytes(job?.file_size_bytes)}</span>
      </div>
    </section>
  );
}


export function historyLoadPhaseLabel(phase) {
  const labels = {
    queued: "Carga en espera",
    uploading: "Subiendo archivo",
    saving_upload: "Guardando archivo",
    upload_saved: "Archivo recibido",
    preparing: "Preparando carga",
    reading_csv: "Leyendo CSV",
    validating_columns: "Validando columnas",
    normalizing: "Normalizando estructura",
    normalizing_timestamps: "Normalizando fechas",
    classifying_entities: "Clasificando entidades",
    normalizing_values: "Normalizando valores",
    sorting_history: "Ordenando historico",
    building_catalog: "Construyendo catalogo",
    history_ready: "Historico normalizado",
    resetting_state: "Reiniciando estado",
    persisting_normality: "Persistiendo normalidad",
    preparing_future_evaluation: "Preparando estado causal",
    causal_identity: "Identificando histórico",
    causal_cache: "Comprobando preparación guardada",
    baseline_prepare: "Ordenando base estadística",
    baseline_continuous: "Calculando normalidad continua",
    baseline_training: "Entrenando modelos por sensor",
    baseline_binary: "Preparando sensores binarios",
    baseline_multistate: "Preparando sensores multivalor",
    baseline_hourly: "Calculando referencias horarias",
    baseline_ready: "Base estadística preparada",
    baseline_manifest: "Guardando parámetros estadísticos",
    context_cache: "Comprobando mapa contextual",
    context_prepare: "Preparando contexto",
    context_signals: "Construyendo señales",
    context_timing: "Calibrando tiempos",
    context_windows: "Construyendo ventanas",
    context_pairs: "Comparando pares de sensores",
    context_select: "Seleccionando relaciones",
    context_models: "Calibrando modelos contextuales",
    context_decisions: "Calibrando decisiones contextuales",
    context_finalize: "Finalizando mapa contextual",
    context_save: "Guardando mapa contextual",
    causal_snapshot: "Preparando estado final",
    causal_save: "Guardando preparación causal",
    finalizing: "Finalizando carga",
    building_timeline: "Preparando visualizacion",
    ready: "Historico listo",
    failed: "Carga fallida",
  };
  return labels[phase] || "Procesando historico";
}

