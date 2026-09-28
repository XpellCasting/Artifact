import { reviewStatusLabel } from "./review-annotations.js";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, ArrowLeft, ArrowRight, BookOpen, Check, Database, Download, FlaskConical, Loader2, Play, RefreshCw, Search, ShieldCheck, Save } from "lucide-react";
import { annotationDiagnosticProblem, diagnosticLabel, DIAGNOSTIC_PARTITIONS as PARTITIONS } from "./diagnostic-data.js";
import { DEFAULT_GAT_CONFIG, gatContextSeconds, gatDataConfigCompatible, gatSupportStatus, isGatConfigValid } from "./diagnostic-gat.js";
import { estimateTrainingRemainingSeconds, formatRemainingTime } from "./diagnostic-progress.js";
import "./diagnostic-workspace.css";
import DatasetLoadProgress from "./DatasetLoadProgress.jsx";
import CandidateTrainingProgress from "./CandidateTrainingProgress.jsx";

const time = (value) => String(value || "—").replace("T", " ").replace(/\+00:00$/, "Z");
const number = (value) => Number(value || 0).toLocaleString("es-CL");
const metric = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)} %` : "—";
const idOf = (value) => value.annotation_id || value.id;
const ACTIVE_JOB = new Set(["queued", "running"]);
const STATE_LABELS = { queued: "En espera", running: "Procesando", completed: "Completado", failed: "No se completó" };
const RESULT_LABELS = { compatible: "Patrón compatible", environmental_anomaly: "Anomalía ambiental", indeterminate: "Resultado indeterminado", no_fault_evidence: "Sin evidencia suficiente de falla" };
const REASON_LABELS = {
  unsupported_entity: "Entidad no incluida en la versión del modelo",
  incompatible_entity: "Entidad incompatible con el modelo actual",
  reference_not_prior_to_target: "La predicción debe ser posterior a la referencia normal de incorporación",
  no_compatible_neighbors: "No hay vecinos con tipo y unidad compatibles; la predicción usa el sensor incorporado",
  insufficient_data: "Datos insuficientes para construir la ventana",
  invalid_event_format: "Formato de evento inválido: falta entidad, fecha o estado",
  context_not_recoverable: "No se pudo recuperar el contexto temporal",
  score_below_rejection_policy: "La puntuación no alcanzó el criterio de aceptación",
  target_not_after_training_period: "La fecha no es posterior al período de entrenamiento",
  pytorch_dependency_missing: "PyTorch no está disponible en el API",
};
const reasonLabel = (reason) => REASON_LABELS[reason] || reason;
const FEATURE_LABELS = {
  valid_count: "Lecturas válidas", sample_count: "Lecturas", duration_seconds: "Duración (s)",
  invalid_ratio: "Proporción de lecturas no válidas", constant_ratio: "Proporción constante",
  constant_seconds: "Persistencia constante (s)", gap_seconds: "Brecha de reportes (s)",
  slope: "Pendiente", trend: "Tendencia", variance: "Varianza", mean: "Media", median: "Mediana",
  mad: "Desviación absoluta mediana", score: "Puntaje", confidence: "Indicador de respaldo",
  entity_id: "Entidad", start: "Inicio", end: "Fin", reference_id: "Referencia", source: "Origen",
};

async function request(base, path, options = {}) {
  const response = await fetch(`${base.replace(/\/$/, "")}${path}`, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === "string" ? payload.detail : payload.detail ? JSON.stringify(payload.detail) : `Error HTTP ${response.status}`);
  }
  return response.json();
}

function Counts({ counts, partitions = false }) {
  return <div className="diagnostic-tags">{Object.entries(counts || {}).map(([name, count]) => <span key={name}>{partitions ? PARTITIONS[name] || name : diagnosticLabel(name)} <b>{number(count)}</b></span>)}</div>;
}

function ValuesTable({ values }) {
  const entries = Object.entries(values || {});
  if (!entries.length) return <p className="diagnostic-help">Sin información adicional.</p>;
  return <div className="diagnostic-table-scroll"><table><tbody>{entries.map(([key, value]) => <tr key={key}><th scope="row">{FEATURE_LABELS[key] || key.replaceAll("_", " ")}</th><td>{value == null ? "—" : typeof value === "number" ? Number(value.toPrecision(6)).toLocaleString("es-CL") : typeof value === "object" ? JSON.stringify(value) : String(value)}</td></tr>)}</tbody></table></div>;
}

function Evaluation({ value }) {
  if (!value) return <p className="diagnostic-help">Todavía no hay una evaluación guardada.</p>;
  return <div className="diagnostic-evaluation"><span>{PARTITIONS[value.partition] || "Evaluación"}</span><span>Exactitud <b>{metric(value.accuracy)}</b></span><span>F1 macro <b>{metric(value.macro_f1)}</b></span>{Number.isFinite(value.coverage) && <span>Cobertura <b>{metric(value.coverage)}</b></span>}{Number.isFinite(value.evaluable_accuracy) && <span>Exactitud en casos evaluables <b>{metric(value.evaluable_accuracy)}</b></span>}{Object.entries(value.by_reference_origin || {}).map(([origin, group]) => <span key={origin}>{({ historical_review: "Acuerdo con revisión del histórico", controlled_injection: "Exactitud en inyecciones controladas", legacy_unspecified: "Referencias antiguas sin origen registrado" })[origin] || origin}: <b>{metric(group.evaluable_accuracy)}</b> · {group.evaluable_count} ventanas evaluables</span>)}<details><summary>Detalle de evaluación</summary><ValuesTable values={value} /></details></div>;
}

export default function DiagnosticWorkspace({ apiBaseUrl = "http://localhost:8000" }) {
  const [status, setStatus] = useState(null);
  const [sources, setSources] = useState({ annotations: [], injections: [], datasets: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [view, setView] = useState("results");
  const [selectedAnnotations, setSelectedAnnotations] = useState([]);
  const [selectedInjections, setSelectedInjections] = useState([]);
  const [includeDrafts, setIncludeDrafts] = useState(false);
  const [search, setSearch] = useState("");
  const [windowSeconds, setWindowSeconds] = useState("3600");
  const [strideSeconds, setStrideSeconds] = useState("1800");
  const [engine, setEngine] = useState("robust_nearest_prototype");
  const [canonicalIds, setCanonicalIds] = useState([]);
  const [gatConfig, setGatConfig] = useState(DEFAULT_GAT_CONFIG);
  const [gatConfigName, setGatConfigName] = useState("GAT-LSTM inicial");
  const [sensorEntityId, setSensorEntityId] = useState("");
  const [sensorReferenceId, setSensorReferenceId] = useState("");
  const [encoderId, setEncoderId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [datasetDetail, setDatasetDetail] = useState(null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [modelId, setModelId] = useState("");
  const [evaluationDatasetId, setEvaluationDatasetId] = useState("");
  const [evaluationPartition, setEvaluationPartition] = useState("validation");
  const [busy, setBusy] = useState("");
  const [exporting, setExporting] = useState("");
  const [resultsSource, setResultsSource] = useState("history");
  const [resultsOffset, setResultsOffset] = useState(0);
  const [results, setResults] = useState({ items: [], total: 0, next_offset: null });
  const [resultsLoading, setResultsLoading] = useState(true);
  const [resultsRefresh, setResultsRefresh] = useState(0);
  const storageKey = `diagnostic-job:${apiBaseUrl}`;
  const [job, setJob] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem(storageKey) || "null"); } catch { return null; }
  });
  const mounted = useRef(true);
  const controllers = useRef(new Set());
  const datasetList = status?.datasets || [];
  const models = status?.models || [];
  const activeJob = ACTIVE_JOB.has(job?.status);
  const datasetBuildJob = job?.client_action === "gat-build" || job?.operation === "build_gat_dataset";
  const candidateTrainingJob = job?.client_action === "gat-train" || job?.operation === "train_gat";
  const candidateProgress = candidateTrainingJob && <CandidateTrainingProgress job={job}
    onRetry={() => { setError(""); setJob((value) => ({ ...value, tracking_paused: false })); }}
    onShowCandidate={() => { setModelId(job.result.model_id); setView("models"); }} />;
  const trainingRemainingSeconds = estimateTrainingRemainingSeconds(job);
  const operationProgress = Math.max(0, Math.min(100, Number(job?.progress) || 0));
  const visibleProgress = job?.phase === "training" && Number(job.max_epochs) > 0
    ? Math.max(0, Math.min(100, 100 * Number(job.epoch || 0) / Number(job.max_epochs)))
    : operationProgress;
  const jobClockStartedAt = Date.parse(job?.phase === "training" ? job.phase_started_at : job?.created_at);
  const jobElapsedSeconds = Number.isFinite(jobClockStartedAt)
    ? Math.max(0, Math.floor((Date.now() - jobClockStartedAt) / 1000)) : null;
  const locked = Boolean(busy) || activeJob;
  const selectedModel = models.find((item) => item.model_id === modelId);
  const selectedDataset = datasetList.find((item) => item.dataset_id === datasetId);
  const selectedGatDataset = selectedDataset?.method === "gat_lstm" ? selectedDataset : null;
  const evaluationDataset = datasetList.find((item) => item.dataset_id === evaluationDatasetId);
  const evaluationCount = evaluationDataset?.partition_counts?.[evaluationPartition] || 0;
  const gatConfigValid = isGatConfigValid(gatConfig);
  const gatSupport = gatSupportStatus(selectedGatDataset);
  const gatDataCompatible = gatDataConfigCompatible(gatConfig, selectedGatDataset);
  const hasFaultTypeTrainingSupport = Object.values(selectedGatDataset?.fault_type_inventory?.train_fit || {}).some((item) => Number(item.independent_episodes) > 0);
  const typeHeadReady = gatConfig.architecture_version !== "gat_lstm_v2_shared_type" || hasFaultTypeTrainingSupport;
  const gatTrainReady = gatSupport.ready && gatDataCompatible && typeHeadReady;
  const gatMissing = [...gatSupport.missing.map(({ partition, label }) => `${PARTITIONS[partition] || partition}: ${diagnosticLabel(label)}`), ...(!gatDataCompatible && selectedGatDataset ? ["La grilla, ventana, grafo o arquitectura difiere del dataset; vuelve a prepararlo."] : []), ...(!typeHeadReady ? ["V2 requiere al menos un episodio de falla con tipo conocido en train_fit."] : [])];

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; controllers.current.forEach((controller) => controller.abort()); controllers.current.clear(); };
  }, []);

  const call = useCallback(async (path, options = {}) => {
    const controller = new AbortController();
    controllers.current.add(controller);
    try { return await request(apiBaseUrl, path, { ...options, signal: controller.signal }); }
    finally { controllers.current.delete(controller); }
  }, [apiBaseUrl]);

  const refresh = useCallback(async () => {
    if (mounted.current) setLoading(true);
    try {
      const sourcePath = `/diagnostics/sources${engine === "gat_lstm" ? "?method=gat_lstm" : ""}`;
      const [nextStatus, nextSources] = await Promise.all([call("/diagnostics/status"), call(sourcePath)]);
      if (!mounted.current) return;
      setStatus(nextStatus);
      setSources(nextSources);
      setDatasetId((current) => nextStatus.datasets?.some((item) => item.dataset_id === current) ? current : nextStatus.datasets?.[0]?.dataset_id || "");
      setModelId((current) => nextStatus.models?.some((item) => item.model_id === current) ? current : nextStatus.active_model_id || nextStatus.models?.[0]?.model_id || "");
      setEvaluationDatasetId((current) => nextStatus.datasets?.some((item) => item.dataset_id === current) ? current : nextStatus.datasets?.[0]?.dataset_id || "");
      setSelectedAnnotations((current) => current.filter((id) => nextSources.annotations?.some((item) => idOf(item) === id)));
      setSelectedInjections((current) => current.filter((id) => nextSources.injections?.some((item) => item.injection_id === id)));
      const trainingSources = nextSources.training_sources || nextSources.canonical_sources || [];
      setCanonicalIds((current) => {
        const retained = current.filter((id) => trainingSources.some((item) => item.source_id === id));
        if (retained.length) return [retained[0]];
        const preferred = trainingSources.find((item) => item.labelled_import) || trainingSources[0];
        return preferred ? [preferred.source_id] : [];
      });
    } catch (failure) { if (mounted.current && failure.name !== "AbortError") setError(failure.message); }
    finally { if (mounted.current) setLoading(false); }
  }, [call, engine]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    try { if (job?.job_id && ACTIVE_JOB.has(job.status)) sessionStorage.setItem(storageKey, JSON.stringify(job)); else sessionStorage.removeItem(storageKey); } catch { /* A blocked storage area does not stop the current job. */ }
  }, [job, storageKey]);

  useEffect(() => {
    if (!job?.job_id || !ACTIVE_JOB.has(job.status)) return undefined;
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const next = await request(apiBaseUrl, `/diagnostics/jobs/${encodeURIComponent(job.job_id)}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setJob((current) => ({ ...next, client_action: current?.client_action }));
        if (ACTIVE_JOB.has(next.status)) timer = setTimeout(poll, 1500);
        else if (next.status === "completed") {
          setNotice(next.message || "Operación completada.");
          if (next.result?.dataset_id) setDatasetId(next.result.dataset_id);
          if (next.result?.model_id) setModelId(next.result.model_id);
          if (next.result?.encoder_id) setEncoderId(next.result.encoder_id);
          await refresh();
        } else setError(typeof next.error === "string" ? next.error : next.message || "La operación no se completó.");
      } catch (failure) {
        if (failure.name !== "AbortError" && !controller.signal.aborted) {
          setError(`Se interrumpió el seguimiento: ${failure.message}. Puedes reintentar la consulta sin repetir la operación.`);
          setJob((current) => ({ ...current, tracking_paused: true }));
        }
      }
    }
    if (!job.tracking_paused) poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [apiBaseUrl, job?.job_id, job?.status, job?.tracking_paused, refresh]);

  useEffect(() => {
    const controller = new AbortController();
    setDatasetDetail(null);
    if (!datasetId) { setDatasetLoading(false); return () => controller.abort(); }
    setDatasetLoading(true);
    request(apiBaseUrl, `/diagnostics/datasets/${encodeURIComponent(datasetId)}`, { signal: controller.signal }).then((value) => {
      if (!controller.signal.aborted) setDatasetDetail(value);
    }).catch((failure) => { if (failure.name !== "AbortError") setError(failure.message); }).finally(() => { if (!controller.signal.aborted) setDatasetLoading(false); });
    return () => controller.abort();
  }, [apiBaseUrl, datasetId]);

  useEffect(() => {
    const controller = new AbortController();
    setResultsLoading(true);
    request(apiBaseUrl, `/diagnostics/results?source=${resultsSource}&offset=${resultsOffset}&limit=30`, { signal: controller.signal }).then((value) => {
      if (!controller.signal.aborted) setResults(value);
    }).catch((failure) => { if (failure.name !== "AbortError") setError(failure.message); }).finally(() => { if (!controller.signal.aborted) setResultsLoading(false); });
    return () => controller.abort();
  }, [apiBaseUrl, resultsSource, resultsOffset, resultsRefresh]);

  const annotations = useMemo(() => (sources.annotations || []).filter((item) => ["normal", "sensor_fault", "environmental_anomaly"].includes(item.label)), [sources]);
  const eligibleAnnotations = annotations.filter((item) => !annotationDiagnosticProblem(item, includeDrafts));
  const eligibleInjections = (sources.injections || []).filter((item) => item.evaluable || includeDrafts);
  const annotationIds = selectedAnnotations.filter((id) => eligibleAnnotations.some((item) => idOf(item) === id));
  const injectionIds = selectedInjections.filter((id) => eligibleInjections.some((item) => item.injection_id === id));
  const matched = (item) => `${item.entity_id} ${item.partition} ${diagnosticLabel(item.fault_type || item.kind || item.label)} ${item.annotation_id || ""} ${item.injection_id || ""}`.toLowerCase().includes(search.toLowerCase());
  const normalCount = annotations.filter((item) => annotationIds.includes(idOf(item)) && item.label === "normal").length;
  const validWindow = [windowSeconds, strideSeconds].every((value) => Number.isInteger(Number(value)) && Number(value) >= 60 && Number(value) <= 604800);
  const hasBaseline = normalCount > 0 || injectionIds.length > 0;

  function toggle(id, setter) { setter((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id]); }

  async function startJob(path, payload, action) {
    if (locked) return;
    setBusy(action); setError(""); setNotice("");
    setJob({ status: "queued", progress: 0, client_action: action,
      created_at: new Date().toISOString(), message: "Enviando solicitud…" });
    try {
      const next = await call(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (!mounted.current) return;
      setJob({ ...next, client_action: action });
      if (next.status === "completed") {
        if (next.result?.dataset_id) setDatasetId(next.result.dataset_id);
        if (next.result?.model_id) setModelId(next.result.model_id);
        if (next.result?.encoder_id) setEncoderId(next.result.encoder_id);
        setNotice(next.message || "Operación completada."); await refresh();
      }
      else if (next.status === "failed") setError(next.error || next.message || "La operación no se completó.");
    } catch (failure) { if (mounted.current && failure.name !== "AbortError") {
      setError(failure.message);
      setJob({ status: "failed", client_action: action, progress: 0, error: failure.message });
    } }
    finally { if (mounted.current) setBusy(""); }
  }

  async function saveGatConfiguration() {
    if (!gatConfigValid) { setError("La salida de GAT debe conservar 32 dimensiones: cabezas × dimensión de cabeza = 32."); return; }
    setBusy("save-config"); setError(""); setNotice("");
    try {
      await call("/diagnostics/configurations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: gatConfigName, config: gatConfig }) });
      setNotice("Configuración GAT-LSTM guardada en el workbench."); await refresh();
    } catch (failure) { if (failure.name !== "AbortError") setError(failure.message); }
    finally { setBusy(""); }
  }

  async function onboardGatSensor() {
    if (!sensorEntityId.trim() || !sensorReferenceId) { setError("Indica la entidad y una referencia normal revisada."); return; }
    setBusy("sensor-onboard"); setError(""); setNotice("");
    try {
      await call("/diagnostics/sensors/onboard", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entity_id: sensorEntityId.trim(), reference_annotation_id: sensorReferenceId }) });
      setNotice("Sensor incorporado con la referencia normal revisada."); setSensorEntityId(""); setSensorReferenceId(""); await refresh();
    } catch (failure) { if (failure.name !== "AbortError") setError(failure.message); }
    finally { setBusy(""); }
  }

  function startGatBuild() {
    if (!gatConfigValid) { setError("La salida de GAT debe conservar 32 dimensiones: cabezas × dimensión de cabeza = 32."); return; }
    const source = (sources.training_sources || sources.canonical_sources || []).find((item) => item.source_id === canonicalIds[0]);
    if (canonicalIds.length !== 1 || source?.partition !== "development") { setError("Selecciona una única fuente de Desarrollo."); return; }
    startJob("/diagnostics/build", { method: "gat_lstm", source_ids: canonicalIds,
      annotation_ids: annotationIds, gat_config: gatConfig, injection_ids: [], include_drafts: false }, "gat-build");
  }

  async function activateModel() {
    if (!selectedModel || locked) return;
    setBusy("activate"); setError(""); setNotice("");
    try {
      await call(`/diagnostics/models/${encodeURIComponent(modelId)}/activate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ allow_exploratory: !selectedModel.validated || Boolean(selectedModel.exploratory) }) });
      if (mounted.current) { setNotice("Versión activada para las próximas ejecuciones. Inicia una nueva reproducción para utilizarla; las ejecuciones ya iniciadas conservan su versión."); await refresh(); }
    } catch (failure) { if (mounted.current && failure.name !== "AbortError") setError(failure.message); }
    finally { if (mounted.current) setBusy(""); }
  }

  async function deactivateModel() {
    if (!selectedModel || locked || status?.active_model_id !== modelId) return;
    setBusy("deactivate"); setError(""); setNotice("");
    try {
      await call(`/diagnostics/models/${encodeURIComponent(modelId)}/deactivate`, { method: "POST" });
      if (mounted.current) {
        setNotice("Versión desactivada. Puedes evaluarla y volver a activarla. Las nuevas ejecuciones usarán el catálogo; las reproducciones ya iniciadas conservan su versión hasta reiniciarlas.");
        await refresh();
      }
    } catch (failure) { if (mounted.current && failure.name !== "AbortError") setError(failure.message); }
    finally { if (mounted.current) setBusy(""); }
  }

  async function downloadDataset(id) {
    setExporting(id); setError("");
    const controller = new AbortController(); controllers.current.add(controller);
    try {
      const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/diagnostics/datasets/${encodeURIComponent(id)}/export`, { signal: controller.signal });
      if (!response.ok) { const value = await response.json().catch(() => ({})); throw new Error(typeof value.detail === "string" ? value.detail : `Error HTTP ${response.status}`); }
      const blob = await response.blob();
      if (!mounted.current) return;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a"); anchor.href = url; anchor.download = `diagnostic-dataset-${id}.zip`; anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (failure) { if (mounted.current && failure.name !== "AbortError") setError(failure.message); }
    finally { controllers.current.delete(controller); if (mounted.current) setExporting(""); }
  }

  return <section className="diagnostic-workspace" aria-label="Clasificador diagnóstico">
    <header className="diagnostic-header"><div><span className="diagnostic-eyebrow">Tercera etapa · Diagnóstico</span><h2>Del comportamiento habitual al patrón de falla</h2><p>F3 compara cada sospecha pendiente con referencias históricas y patrones conocidos. La preparación del clasificador corresponde al investigador; la operación utiliza la versión activa.</p></div><button className="diagnostic-button" onClick={() => { setError(""); refresh(); setResultsRefresh((value) => value + 1); }} disabled={loading}><RefreshCw size={16} className={loading ? "diagnostic-spin" : ""} />Actualizar</button></header>
    <div className="diagnostic-mode"><Activity size={18} /><div><strong>{status?.mode === "active_model" ? "Modelo entrenado activo" : loading && !status ? "Consultando estado…" : "Diagnóstico por catálogo de patrones"}</strong><span>{status?.active_model_id ? `Versión ${status.active_model_id}` : "Sin una versión entrenada activa. Las reglas del catálogo proponen compatibilidad y pueden abstenerse."}</span></div>{status?.models?.some((item) => item.model_id === status.active_model_id && (item.exploratory || !item.validated)) && <span className="diagnostic-badge diagnostic-exploratory">Experimental</span>}</div>
    {error && <div className="diagnostic-alert" role="alert">{error}</div>}
    {notice && <div className="diagnostic-notice" role="status">{notice}</div>}
    {candidateTrainingJob && !(view === "prepare" && engine === "gat_lstm" && selectedGatDataset) && candidateProgress}
    {job && !candidateTrainingJob && !(datasetBuildJob && view === "prepare" && engine === "gat_lstm") && <section className={`history-load-progress ${job.status === "completed" ? "ready" : ""} ${job.status === "failed" ? "failed" : ""}`} aria-live="polite"><div className="history-load-heading"><span>{job.phase === "training" ? "Entrenamiento GAT-LSTM" : STATE_LABELS[job.status] || job.status}</span><strong>{Math.round(visibleProgress)}%</strong></div><div className="progress-track" role="progressbar" aria-label={job.phase === "training" ? "Avance del entrenamiento GAT-LSTM" : "Avance de la operación"} aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(visibleProgress)}><span style={{ width: `${visibleProgress}%` }} /></div><p>{job.message || "Preparando operación…"}</p><div className="history-load-meta">{job.phase === "training" && <span>Época: {number(job.epoch)} de {number(job.max_epochs)}</span>}{jobElapsedSeconds != null && <span>Transcurrido: {formatRemainingTime(jobElapsedSeconds)}</span>}<span>Restante: {job.status === "completed" ? "0 s" : job.phase === "training" ? trainingRemainingSeconds == null ? "estimando" : `aprox. ${formatRemainingTime(trainingRemainingSeconds)}` : "—"}</span></div>{job.tracking_paused && <button className="diagnostic-button" onClick={() => { setError(""); setJob((value) => ({ ...value, tracking_paused: false })); }}>Reintentar seguimiento</button>}</section>}
    <nav className="diagnostic-tabs" aria-label="Vistas del clasificador">{[["results", "Diagnósticos", Activity], ["prepare", "Preparación del investigador", Database], ["models", "Modelos y evaluación", FlaskConical], ["catalog", "Catálogo y referencias", BookOpen]].map(([key, label, Icon]) => <button key={key} className={view === key ? "active" : ""} onClick={() => setView(key)} aria-pressed={view === key}><Icon size={16} />{label}</button>)}</nav>

    {view === "results" && <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Diagnósticos de las sospechas pendientes</h3><span className="diagnostic-badge">{number(results.total)} resultados</span></div><div className="diagnostic-result-controls"><label>Origen de los resultados<select value={resultsSource} onChange={(event) => { setResultsSource(event.target.value); setResultsOffset(0); }}><option value="history">Histórico y eventos</option><option value="validation_replay">Reproducción de evaluación</option></select></label><button className="diagnostic-button" disabled={resultsLoading} onClick={() => setResultsRefresh((value) => value + 1)}><RefreshCw size={15} />Actualizar diagnósticos</button></div><p className="diagnostic-help">La compatibilidad expresa semejanza con un patrón; no es una probabilidad calibrada ni confirma una avería física. Los casos resueltos como ambientales por F2 conservan su recorrido.</p>{!resultsLoading && results.summary && <div className="diagnostic-evaluation"><span>Lecturas con resultado: <b>{number(results.summary.total)}</b></span><span>Indeterminadas: <b>{number(results.summary.indeterminate)}</b></span><span>Cobertura: <b>{metric(results.summary.coverage)}</b></span>{results.summary.total === 0 && <span>No hay lecturas evaluables; las métricas no se calculan.</span>}{Object.entries(results.summary.indeterminate_by_reason || {}).map(([reason, count]) => <span key={reason}>{reasonLabel(reason)}: <b>{number(count)}</b></span>)}</div>}
      {resultsLoading ? <p className="diagnostic-empty"><Loader2 size={19} className="diagnostic-spin" />Cargando diagnósticos…</p> : !results.items?.length ? <div className="diagnostic-empty"><Activity size={28} /><p>Aún no hay diagnósticos en este origen.</p><span>Procesa eventos con F1 y F2 para que las sospechas pendientes lleguen a F3.</span></div> : <div className="diagnostic-results">{results.items.map((item, index) => <article key={item.diagnosis_id || item.diagnostic_id || item.result_id || `${item.entity_id}:${item.timestamp}:${index}`} className="diagnostic-result"><div className="diagnostic-result-heading"><div><span className="diagnostic-eyebrow">{item.predicted_label === "environmental_anomaly" ? "Anomalía ambiental" : item.predicted_label === "sensor_fault" ? "Falla de sensor detectada" : item.predicted_label === "normal" ? "Normal" : RESULT_LABELS[item.status] || item.status}</span><h4>{item.predicted_label ? diagnosticLabel(item.predicted_label) : item.fault_type ? diagnosticLabel(item.fault_type) : item.status === "no_fault_evidence" ? "Sin patrón de falla concluyente" : "Tipo indeterminado"}</h4><p>{item.entity_id}</p></div>{Number.isFinite(item.confidence) && <div className="diagnostic-compatibility"><span>Confianza de clase</span><strong>{item.confidence.toFixed(2)}</strong><small>{item.confidence_kind === "uncalibrated_softmax" ? "Softmax no calibrado" : "Índice, no probabilidad"}</small></div>}{!Number.isFinite(item.confidence) && Number.isFinite(item.compatibility) && <div className="diagnostic-compatibility"><span>Compatibilidad</span><strong>{item.compatibility.toFixed(2)}</strong><small>Índice, no probabilidad</small></div>}</div><small>{time(item.timestamp)} · modelo {item.model_version || `Catálogo ${item.catalog_version || "—"}`}{item.preprocessor_version && ` · preprocesador ${item.preprocessor_version}`}{item.graph_version && ` · grafo ${item.graph_version}`}</small>{item.input_coverage && <small>Cobertura de datos válidos: {number(item.input_coverage.valid_value_steps)} de {number(item.input_coverage.required_steps)} pasos ({metric(item.input_coverage.valid_value_coverage)}) · vecinos: {number(item.input_coverage.neighbor_ids?.length)}</small>}{item.input_provenance?.length > 0 && <details><summary>Procedencia de las lecturas de entrada</summary><ValuesTable values={{ input_provenance: item.input_provenance, source: item.source, run_id: item.run_id }} /></details>}{item.needs_review && <span className="diagnostic-badge diagnostic-exploratory">Evidencia pendiente de revisión</span>}{item.method === "gat_lstm" && <span className="diagnostic-badge">Tipo de falla: {item.fault_type_status === "not_supported" ? "No disponible en este modelo" : item.fault_type_status === "identified" ? diagnosticLabel(item.fault_type) : item.fault_type_status === "unknown" ? "Tipo indeterminado por respaldo insuficiente" : "No aplicable"}</span>}{item.class_scores && <div className="diagnostic-tags">{Object.entries(item.class_scores).map(([label, score]) => <span key={label}>{diagnosticLabel(label)} <b>{metric(score)}</b></span>)}</div>}<ul className="diagnostic-reasons">{(item.reasons || []).map((reason, reasonIndex) => <li key={reasonIndex}>{typeof reason === "string" ? reasonLabel(reason) : JSON.stringify(reason)}</li>)}</ul><details><summary>Características y referencia utilizadas</summary><h5>Características disponibles al decidir</h5><ValuesTable values={item.features} /><h5>Referencia histórica</h5><ValuesTable values={item.reference} /></details></article>)}</div>}
      <div className="diagnostic-pagination"><button className="diagnostic-button" disabled={resultsLoading || resultsOffset === 0} onClick={() => setResultsOffset((value) => Math.max(0, value - 30))}><ArrowLeft size={15} />Anterior</button><span>{results.total ? `${number(resultsOffset + 1)}–${number(Math.min(resultsOffset + (results.items?.length || 0), results.total))} de ${number(results.total)}` : "0 resultados"}</span><button className="diagnostic-button" disabled={resultsLoading || results.next_offset == null} onClick={() => setResultsOffset(results.next_offset)} >Siguiente<ArrowRight size={15} /></button></div>
    </section>}

    {view === "prepare" && <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Motor del clasificador</h3><span className="diagnostic-badge">{status?.capabilities?.gat_lstm ? "GAT-LSTM disponible" : "Modo convencional disponible"}</span></div><label>Motor<select value={engine} onChange={(event) => setEngine(event.target.value)}><option value="robust_nearest_prototype">Convencional · prototipos</option><option value="gat_lstm" disabled={!status?.capabilities?.gat_lstm}>GAT-LSTM por entidad{status?.capabilities?.gat_lstm ? "" : " · PyTorch no disponible"}</option></select></label>{!status?.capabilities?.gat_lstm && <p className="diagnostic-help">El servicio convencional sigue disponible. GAT-LSTM requiere instalar backend/requirements-gat.txt en el entorno del servidor.</p>}</section>}

    {view === "prepare" && engine === "gat_lstm" && <>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Configuración GAT-LSTM</h3><span className="diagnostic-badge">Tres clases · abstención aparte</span></div><p className="diagnostic-help">La red predice normal, anomalía ambiental o falla del sensor para cada entidad. Las etiquetas de subtipo se conservan como evidencia. La puntuación softmax es no calibrada y no representa una probabilidad física.</p><details><summary>Guía rápida del recorrido</summary><ol><li>Selecciona una sola fuente de Desarrollo, incluido el CSV etiquetado importado.</li><li>Prepara las secuencias: preprocesador y grafo usan solo train_fit; train_fit/dev_tune ajustan red y política de abstención.</li><li>Entrena cuando las tres clases tengan ejemplos y episodios independientes en ambas particiones.</li><li>Validación y Prueba no se leen ni se usan para ajustar GAT-LSTM; quedan reservadas para generar eventos futuros.</li></ol><p className="diagnostic-help">El histórico canónico no tiene soporte supervisado suficiente en train_fit/dev_tune; guardar configuración y preparar datasets sí está disponible.</p></details><label>Configuración guardada<select value="" onChange={(event) => { const saved = status?.gat_configurations?.find((item) => item.configuration_id === event.target.value); if (saved) { setGatConfig(saved.config); setGatConfigName(saved.name); } }}><option value="">Usar o editar configuración actual</option>{(status?.gat_configurations || []).map((item) => <option key={item.configuration_id} value={item.configuration_id}>{item.name} · {time(item.created_at)}</option>)}</select></label><label>Arquitectura<select value={gatConfig.architecture_version} onChange={(event) => setGatConfig((value) => ({ ...value, architecture_version: event.target.value, pretrain_enabled: event.target.value === "gat_lstm_v2_shared_type" ? false : value.pretrain_enabled }))}><option value="gat_lstm_v1_entity">V1 · especializada por entidad</option><option value="gat_lstm_v2_shared_type">V2 · encoder compartido y tipo de falla</option></select></label><p className="diagnostic-help">V2 comparte el encoder entre sensores y tiene una cabeza para clase y otra para tipo de falla. Los modelos v1 existentes y sus artefactos se conservan.</p><div className="diagnostic-build-controls"><label>Nombre<input value={gatConfigName} onChange={(event) => setGatConfigName(event.target.value)} maxLength={100} /></label><label>Separación entre pasos (s)<input type="number" min="1" value={gatConfig.grid_seconds} onChange={(event) => setGatConfig((value) => ({ ...value, grid_seconds: Number(event.target.value) }))} /></label><label>Número de pasos<input type="number" min="2" value={gatConfig.sequence_steps} onChange={(event) => setGatConfig((value) => ({ ...value, sequence_steps: Number(event.target.value) }))} /></label><label>Ventana de características (s)<input type="number" min="1" value={gatConfig.feature_window_seconds} onChange={(event) => setGatConfig((value) => ({ ...value, feature_window_seconds: Number(event.target.value) }))} /></label><label>Vecinos máximos<input type="number" min="0" max="31" value={gatConfig.graph_top_k} onChange={(event) => setGatConfig((value) => ({ ...value, graph_top_k: Number(event.target.value) }))} /></label></div><p className="diagnostic-help">Contexto causal total: <b>{number((gatConfig.sequence_steps - 1) * gatConfig.grid_seconds + gatConfig.feature_window_seconds)} segundos</b>. Lookback derivado: {number((gatConfig.sequence_steps - 1) * gatConfig.grid_seconds)} segundos.</p><div className="diagnostic-checks"><label className="diagnostic-check"><input type="checkbox" checked={gatConfig.use_entropy} onChange={(event) => setGatConfig((value) => ({ ...value, use_entropy: event.target.checked }))} /><span>Usar entropía</span></label><label className="diagnostic-check"><input type="checkbox" checked={gatConfig.use_dtw} onChange={(event) => setGatConfig((value) => ({ ...value, use_dtw: event.target.checked }))} /><span>Usar DTW</span></label><label className="diagnostic-check"><input type="checkbox" checked={gatConfig.pretrain_enabled} disabled={gatConfig.architecture_version === "gat_lstm_v2_shared_type"} onChange={(event) => setGatConfig((value) => ({ ...value, pretrain_enabled: event.target.checked }))} /><span>Preparar preentrenamiento predictivo opcional</span></label></div><details><summary>Opciones avanzadas</summary><div className="diagnostic-build-controls"><label>Cabezas GAT<input type="number" min="1" value={gatConfig.gat_heads} onChange={(event) => setGatConfig((value) => ({ ...value, gat_heads: Number(event.target.value) }))} /></label><label>Dimensión por cabeza<input type="number" min="1" value={gatConfig.gat_head_dim} onChange={(event) => setGatConfig((value) => ({ ...value, gat_head_dim: Number(event.target.value) }))} /></label><label>LSTM unidades<input type="number" min="1" value={gatConfig.lstm_hidden} onChange={(event) => setGatConfig((value) => ({ ...value, lstm_hidden: Number(event.target.value) }))} /></label><label>Dropout<input type="number" min="0" max="0.99" step="0.05" value={gatConfig.dropout} onChange={(event) => setGatConfig((value) => ({ ...value, dropout: Number(event.target.value) }))} /></label><label>Lote<input type="number" min="1" value={gatConfig.batch_size} onChange={(event) => setGatConfig((value) => ({ ...value, batch_size: Number(event.target.value) }))} /></label><label>Tasa de aprendizaje<input type="number" min="0.000001" step="0.0001" value={gatConfig.learning_rate} onChange={(event) => setGatConfig((value) => ({ ...value, learning_rate: Number(event.target.value) }))} /></label><label>Máximo de épocas<input type="number" min="1" max="500" value={gatConfig.max_epochs} onChange={(event) => setGatConfig((value) => ({ ...value, max_epochs: Number(event.target.value) }))} /></label><label>Paciencia<input type="number" min="1" value={gatConfig.patience} onChange={(event) => setGatConfig((value) => ({ ...value, patience: Number(event.target.value) }))} /></label><label>Semilla<input type="number" value={gatConfig.seed} onChange={(event) => setGatConfig((value) => ({ ...value, seed: Number(event.target.value) }))} /></label><label>Dispositivo<select value={gatConfig.device} onChange={(event) => setGatConfig((value) => ({ ...value, device: event.target.value }))}><option value="cpu">CPU</option><option value="cuda" disabled>GPU CUDA · etapa futura</option></select></label><label>Máximo de objetivos predictivos por partición<input type="number" min="1" value={gatConfig.pretrain_max_samples} onChange={(event) => setGatConfig((value) => ({ ...value, pretrain_max_samples: Number(event.target.value) }))} /></label></div><p className="diagnostic-help">Esta versión entrena en CPU y Docker fija el máximo de hilos en 4. La salida de atención debe mantener 32 dimensiones; cabezas × dimensión por cabeza debe ser 32.</p></details>{gatConfig.architecture_version === "gat_lstm_v2_shared_type" && <div className="diagnostic-dataset-summary"><strong>Incorporar sensor a V2</strong><p>Selecciona una referencia Normal revisada en Desarrollo. Esa ventana fija la escala y debe terminar antes de los diagnósticos del sensor.</p><label>Entity ID<input value={sensorEntityId} onChange={(event) => setSensorEntityId(event.target.value)} placeholder="sensor.nuevo" /></label><label>Referencia normal<select value={sensorReferenceId} onChange={(event) => setSensorReferenceId(event.target.value)}><option value="">Selecciona referencia</option>{annotations.filter((item) => item.label === "normal" && item.review_status === "reviewed" && item.partition === "development").map((item) => <option key={idOf(item)} value={idOf(item)}>{item.entity_id} · {time(item.start)} → {time(item.end)}</option>)}</select></label><button className="diagnostic-button" disabled={locked || !sensorEntityId.trim() || !sensorReferenceId} onClick={onboardGatSensor}>{busy === "sensor-onboard" ? "Incorporando…" : "Incorporar con referencia revisada"}</button>{(status?.sensor_profiles || []).map((profile) => <small key={profile.profile_id}>{profile.entity_id} · {profile.kind} · referencia hasta {time(profile.reference_end)} · {number(profile.reference_rows)} lecturas</small>)}</div>}<button className="diagnostic-button" disabled={locked || !gatConfigValid} onClick={saveGatConfiguration}><Save size={15} />Guardar configuración</button></section>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>1. Fuente de Desarrollo y etiquetas</h3></div><p className="diagnostic-help">GAT-LSTM solo recibe una fuente de Desarrollo. El CSV etiquetado importado aporta directamente sus etiquetas por fila. Validación y Prueba no se consultan para preparar ni ajustar el modelo; quedan reservadas para generar eventos futuros.</p><div className="diagnostic-source-list">{(sources.training_sources || sources.canonical_sources || []).map((item) => <label className="diagnostic-source" key={item.source_id}><input type="radio" name="gat-development-source" checked={canonicalIds.includes(item.source_id)} disabled={locked} onChange={() => setCanonicalIds([item.source_id])} /><span><strong>{item.labelled_import ? "CSV etiquetado · " : "Histórico · "}{PARTITIONS[item.partition] || item.partition}</strong><small>{item.filename} · {number(item.rows)} filas</small><small>SHA‑256 {item.sha256}</small>{item.labelled_import && <small>{number(item.imported_labels)} filas etiquetadas · {number(item.imported_synthetic)} sintéticas</small>}</span></label>)}</div><h4>Anotaciones adicionales disponibles</h4><div className="diagnostic-source-list">{annotations.map((item) => { const id = idOf(item); const selectable = ["reviewed", "verified"].includes(item.status) && item.source_sha256 && canonicalIds.some((sourceId) => (sources.training_sources || sources.canonical_sources || []).find((source) => source.source_id === sourceId)?.sha256 === item.source_sha256); return <label className={`diagnostic-source ${selectable ? "" : "diagnostic-source-disabled"}`} key={id}><input type="checkbox" checked={annotationIds.includes(id)} disabled={locked || !selectable} onChange={() => toggle(id, setSelectedAnnotations)} /><span><strong>{diagnosticLabel(item.fault_type || item.label)}</strong><span>{item.entity_id}</span><small>{time(item.start)} → {time(item.end)} · {item.status}</small>{!selectable && <em>Fuente no seleccionada o identidad no verificable.</em>}</span></label>; })}</div></section>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>2. Preparar secuencias y grafo</h3><span className="diagnostic-badge">{selectedGatDataset ? "Dataset guardado" : "Listo para configurar"}</span></div><label>Dataset GAT-LSTM<select value={selectedGatDataset?.dataset_id || ""} onChange={(event) => setDatasetId(event.target.value)}><option value="">Selecciona un dataset guardado</option>{datasetList.filter((item) => item.method === "gat_lstm").map((item) => <option key={item.dataset_id} value={item.dataset_id}>{time(item.created_at)} · {item.dataset_id.slice(0, 12)}</option>)}</select></label><button className="diagnostic-button diagnostic-primary" disabled={locked || !gatConfigValid || canonicalIds.length !== 1} onClick={startGatBuild}>{busy === "gat-build" || (datasetBuildJob && activeJob) ? <Loader2 size={16} className="diagnostic-spin" /> : <Database size={16} />}{busy === "gat-build" || (datasetBuildJob && activeJob) ? "Preparando dataset…" : "Preparar dataset GAT-LSTM"}</button>{datasetBuildJob && <DatasetLoadProgress job={job} onRetry={() => { setError(""); setJob((value) => ({ ...value, tracking_paused: false })); }} />}{selectedGatDataset && <><div className="diagnostic-dataset-summary"><span>{selectedGatDataset.dataset_id}</span><Counts counts={selectedGatDataset.partition_counts} partitions />{Object.entries(selectedGatDataset.label_counts || {}).map(([part, counts]) => <div key={part}><strong>{PARTITIONS[part] || part}</strong><Counts counts={counts} /><small>Episodios independientes</small><Counts counts={selectedGatDataset.episode_counts?.[part]} /></div>)}</div>{Object.entries(selectedGatDataset.fault_type_inventory || {}).map(([part, types]) => <div key={part} className="diagnostic-dataset-summary"><strong>{PARTITIONS[part] || part} · respaldo por tipo de falla</strong>{Object.entries(types).map(([faultType, support]) => <span key={faultType}>{diagnosticLabel(faultType)} · {number(support.windows)} ventanas · {number(support.independent_episodes)} episodios · {support.support_status === "supported" ? "respaldo mínimo disponible" : "respaldo insuficiente"}</span>)}</div>)}<p className="diagnostic-help">Evidencia espacial F2: {selectedGatDataset.spatial_evidence ? "hay relaciones entre entidades" : "solo hay autoaristas"}. Objetivos de preentrenamiento: {number(selectedGatDataset.pretrain_counts?.train_fit)} en train_fit y {number(selectedGatDataset.pretrain_counts?.dev_tune)} en dev_tune.</p>{(selectedGatDataset.exclusions || []).map((item, index) => <p className="diagnostic-help" key={index}>Exclusión: {item.annotation_id || item.source_id || item.episode_id} · {item.reason}</p>)}<div className="diagnostic-actions"><button className="diagnostic-button diagnostic-primary" disabled={locked || !gatTrainReady} title={!gatTrainReady ? `Falta soporte: ${gatMissing.join("; ")}` : ""} onClick={() => startJob("/diagnostics/train", { dataset_id: datasetId, method: "gat_lstm", gat_config: gatConfig, ...(encoderId ? { encoder_id: encoderId } : {}) }, "gat-train")}>{candidateTrainingJob && activeJob ? <Loader2 size={15} className="diagnostic-spin" /> : <FlaskConical size={15} />}{candidateTrainingJob && activeJob ? "Entrenando candidato…" : "Entrenar candidato"}</button><button className="diagnostic-button" disabled={locked || gatConfig.architecture_version === "gat_lstm_v2_shared_type" || !gatConfig.pretrain_enabled || !gatDataCompatible || !(selectedGatDataset.pretrain_counts?.train_fit && selectedGatDataset.pretrain_counts?.dev_tune)} onClick={() => startJob("/diagnostics/pretrain", { dataset_id: datasetId, gat_config: gatConfig }, "gat-pretrain")}><Play size={15} />Preentrenar encoder</button></div>{candidateProgress}{!gatTrainReady && <p className="diagnostic-help">Ajuste deshabilitado por soporte insuficiente: {gatMissing.join(" · ")}. Puedes conservar el dataset y la configuración; no se crean etiquetas ni se entrena una cabeza sin clases.</p>}<label>Encoder predictivo guardado<select value={encoderId} onChange={(event) => setEncoderId(event.target.value)}><option value="">Entrenar sin encoder</option>{(status?.gat_encoders || []).map((item) => <option key={item.encoder_id} value={item.encoder_id}>{item.encoder_id.slice(0, 12)} · encoder sin cabeza</option>)}</select></label>{encoderId && <p className="diagnostic-help">Encoder predictivo listo: {encoderId}. El ajuste diagnóstico seguirá requiriendo las tres clases en train_fit y dev_tune.</p>}</>}</section>
    </>}

    {view === "prepare" && engine !== "gat_lstm" && <>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>1. Seleccionar fragmentos y casos documentados</h3><span className="diagnostic-badge">{annotationIds.length + injectionIds.length} seleccionados</span></div><p className="diagnostic-help">Marca períodos de funcionamiento correcto para construir referencias y controles. Añade interpretaciones del histórico con un tipo documentado o inyecciones controladas guardadas. Las etiquetas históricas expresan el juicio del revisor; las inyecciones identifican una intervención conocida. Se conserva la partición y la revisión de cada fuente.</p><div className="diagnostic-source-tools"><label className="diagnostic-search">Buscar entidad, patrón o identificador<div><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filtrar fuentes" /></div></label><label className="diagnostic-check"><input type="checkbox" checked={includeDrafts} disabled={locked} onChange={(event) => setIncludeDrafts(event.target.checked)} /><span>Incluir pendientes y casos exploratorios<small>El conjunto y sus modelos conservarán esta condición.</small></span></label></div>
        <div className="diagnostic-source-columns"><section><div className="diagnostic-panel-title"><h4>Fragmentos revisados</h4><button className="diagnostic-text-button" disabled={locked} onClick={() => setSelectedAnnotations(eligibleAnnotations.filter(matched).map(idOf))}>Seleccionar aptos visibles</button></div><div className="diagnostic-source-list">{annotations.filter(matched).map((annotation) => { const id = idOf(annotation); const problem = annotationDiagnosticProblem(annotation, includeDrafts); return <label key={id} className={`diagnostic-source ${problem ? "diagnostic-source-disabled" : ""}`}><input type="checkbox" checked={annotationIds.includes(id)} disabled={locked || Boolean(problem)} onChange={() => toggle(id, setSelectedAnnotations)} /><span><strong>{diagnosticLabel(annotation.fault_type || annotation.label)}</strong><span>{annotation.entity_id}</span><small>{time(annotation.start)} → {time(annotation.end)}</small><small>{PARTITIONS[annotation.partition] || annotation.partition} · revisión {annotation.revision} · {reviewStatusLabel(annotation)}</small>{problem && <em>{problem}</em>}</span></label>; })}{!annotations.filter(matched).length && <p className="diagnostic-empty">No hay fragmentos coincidentes. Guarda anotaciones normales, fallas identificadas o anomalías ambientales en Revisión y etiquetado.</p>}</div></section><section><div className="diagnostic-panel-title"><h4>Inyecciones guardadas</h4><button className="diagnostic-text-button" disabled={locked} onClick={() => setSelectedInjections(eligibleInjections.filter(matched).map((item) => item.injection_id))}>Seleccionar aptas visibles</button></div><div className="diagnostic-source-list">{(sources.injections || []).filter(matched).map((injection) => <label key={injection.injection_id} className={`diagnostic-source ${!injection.evaluable && !includeDrafts ? "diagnostic-source-disabled" : ""}`}><input type="checkbox" checked={injectionIds.includes(injection.injection_id)} disabled={locked || (!injection.evaluable && !includeDrafts)} onChange={() => toggle(injection.injection_id, setSelectedInjections)} /><span><strong>{diagnosticLabel(injection.kind)}</strong><span>{injection.entity_id}</span><small>{time(injection.start)} → {time(injection.end)}</small><small>{PARTITIONS[injection.partition] || injection.partition} · revisión fijada {injection.annotation_revision} · {injection.evaluable ? "Inyección controlada evaluable" : "Exploratorio"}</small><small>Semilla {injection.seed} · {injection.injection_id}</small>{!injection.evaluable && !includeDrafts && <em>Habilita el uso exploratorio para seleccionar este caso.</em>}</span></label>)}{!(sources.injections || []).filter(matched).length && <p className="diagnostic-empty">No hay inyecciones coincidentes. Puedes generarlas sobre fragmentos normales revisados.</p>}</div></section></div>
        <div className="diagnostic-build-controls"><label>Ventana de análisis (segundos)<input type="number" min="60" max="604800" step="1" value={windowSeconds} disabled={locked} onChange={(event) => setWindowSeconds(event.target.value)} /></label><label>Paso del barrido (segundos)<input type="number" min="60" max="604800" step="1" value={strideSeconds} disabled={locked} onChange={(event) => setStrideSeconds(event.target.value)} /></label><button className="diagnostic-button diagnostic-primary" disabled={locked || !validWindow || !hasBaseline || !(annotationIds.length + injectionIds.length)} onClick={() => startJob("/diagnostics/build", { annotation_ids: annotationIds, injection_ids: injectionIds, window_seconds: Number(windowSeconds), stride_seconds: Number(strideSeconds), include_drafts: includeDrafts }, "build")}>{locked ? <Loader2 size={16} className="diagnostic-spin" /> : <Database size={16} />}Construir referencias y ejemplos</button></div><p className="diagnostic-help">{number(normalCount)} fragmentos normales seleccionados. Las inyecciones incorporan automáticamente su referencia normal original. Selecciona al menos una de estas fuentes. Ventana y paso admiten enteros entre 60 y 604800 segundos. Para entrenar un clasificador se necesitan controles normales y al menos un tipo de falla en Desarrollo. Validación y Prueba se reservan para evaluar; sus ejemplos no entrenan el modelo.</p>
      </section>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>2. Conjuntos de aprendizaje</h3><span className="diagnostic-badge">{datasetList.length} guardados</span></div>{!datasetList.length ? <p className="diagnostic-empty">Construye el primer conjunto para revisar sus ejemplos y descargarlo.</p> : <><label>Conjunto guardado<select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>{datasetList.map((item) => <option key={item.dataset_id} value={item.dataset_id}>{time(item.created_at)} · {number(item.example_count)} ejemplos · {item.dataset_id.slice(0, 10)}{item.exploratory ? " · Exploratorio" : ""}</option>)}</select></label><div className="diagnostic-dataset-summary"><span>{selectedDataset?.dataset_id}</span>{selectedDataset?.exploratory && <span className="diagnostic-badge diagnostic-exploratory">Exploratorio</span>}<Counts counts={selectedDataset?.label_counts} /><Counts counts={selectedDataset?.partition_counts} partitions /></div>{(selectedDataset?.warnings || []).map((warning, index) => <p className="diagnostic-help" key={index}>{warning}</p>)}<div className="diagnostic-actions"><button className="diagnostic-button" disabled={!datasetId || Boolean(exporting)} onClick={() => downloadDataset(datasetId)}>{exporting === datasetId ? <Loader2 size={15} className="diagnostic-spin" /> : <Download size={15} />}Descargar conjunto ZIP</button><button className="diagnostic-button diagnostic-primary" disabled={locked || !selectedDataset?.partition_counts?.development} onClick={() => startJob("/diagnostics/train", { dataset_id: datasetId }, "train")}><FlaskConical size={16} />Entrenar versión candidata</button></div><p className="diagnostic-help">El entrenamiento utiliza únicamente Desarrollo y no activa automáticamente la versión. El conjunto conserva la procedencia y las características de cada ejemplo.</p><details><summary>Vista previa de ejemplos</summary>{datasetLoading ? <p className="diagnostic-help">Cargando ejemplos…</p> : <div className="diagnostic-example-list">{(datasetDetail?.examples_preview || []).map((example, index) => <details key={example.example_id || index}><summary>{diagnosticLabel(example.fault_type || example.label)} · {example.entity_id} · {PARTITIONS[example.partition] || example.partition} · {time(example.timestamp || example.end)}</summary><ValuesTable values={example} /></details>)}{!datasetDetail?.examples_preview?.length && <p className="diagnostic-help">Sin ejemplos en la vista previa.</p>}</div>}</details></>}</section>
    </>}

    {view === "models" && <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Versiones, evaluación y activación</h3><span className="diagnostic-badge">{models.length} versiones</span></div><p className="diagnostic-help">Cada versión aprende de las características de Desarrollo. Evalúa su comportamiento sobre episodios separados antes de incorporarla; una etiqueta nueva se utiliza en el próximo entrenamiento.</p>{!models.length ? <div className="diagnostic-empty"><FlaskConical size={26} /><p>Aún no hay modelos entrenados.</p><span>Construye un conjunto con normalidad y fallas de Desarrollo en Preparación del investigador.</span></div> : <><label>Versión candidata o activa<select value={modelId} onChange={(event) => setModelId(event.target.value)}>{models.map((model) => <option key={model.model_id} value={model.model_id}>{time(model.created_at)} · {model.model_id.slice(0, 12)}{status?.active_model_id === model.model_id ? " · Activa" : ""}{model.exploratory ? " · Exploratoria" : ""}</option>)}</select></label>{selectedModel && <><div className="diagnostic-model-summary"><span className="diagnostic-id">{selectedModel.model_id}</span><span>{number(selectedModel.training_count)} ejemplos de entrenamiento</span>{selectedModel.excluded_count > 0 && <details><summary>{number(selectedModel.excluded_count)} ejemplos omitidos</summary><p className="diagnostic-help">Faltan reportes o caracteristicas compatibles para aprender de estas ventanas. Incluye contexto previo en los casos de pico; cada ventana requiere al menos tres reportes.</p></details>}{status?.active_model_id === modelId && <span className="diagnostic-badge"><Check size={13} />Activa</span>}{selectedModel.validated ? <span className="diagnostic-badge"><ShieldCheck size={13} />Evaluación independiente disponible</span> : <span className="diagnostic-badge diagnostic-exploratory">Sin evaluación independiente</span>}{selectedModel.exploratory && <span className="diagnostic-badge diagnostic-exploratory">Entrenamiento exploratorio</span>}</div><div className="diagnostic-tags">{(selectedModel.classes || []).map((label) => <span key={label}>{diagnosticLabel(label)}</span>)}</div>{selectedModel.method === "gat_lstm" ? <p className="diagnostic-help">GAT-LSTM se ajusta solo con Desarrollo. Validación y Prueba permanecen sin leer y se reservan para generar eventos futuros.</p> : <><div className="diagnostic-evaluate-controls"><label>Conjunto para evaluar<select value={evaluationDatasetId} onChange={(event) => setEvaluationDatasetId(event.target.value)}><option value="">Selecciona un conjunto</option>{datasetList.map((item) => <option key={item.dataset_id} value={item.dataset_id}>{time(item.created_at)} · {item.dataset_id.slice(0, 12)}</option>)}</select></label><label>Partición de evaluación<select value={evaluationPartition} onChange={(event) => setEvaluationPartition(event.target.value)}><option value="validation">Validación</option><option value="test">Prueba</option></select></label><button className="diagnostic-button" disabled={locked || !evaluationDatasetId || !evaluationCount} onClick={() => startJob(`/diagnostics/models/${encodeURIComponent(modelId)}/evaluate`, { dataset_id: evaluationDatasetId, partition: evaluationPartition }, "evaluate")}><Play size={15} />Evaluar versión</button></div><p className="diagnostic-help">{number(evaluationCount)} ejemplos disponibles en {PARTITIONS[evaluationPartition]}. Prueba mide el resultado final y no sustituye la evaluación independiente necesaria para revisar una versión.</p><Evaluation value={selectedModel.validation} />{(selectedModel.evaluations || []).filter((evaluation) => evaluation.partition === "test").map((evaluation, index) => <Evaluation key={evaluation.evaluation_id || index} value={evaluation} />)}</>}<div className="diagnostic-actions">{status?.active_model_id === modelId ? <button className="diagnostic-button" disabled={locked} onClick={deactivateModel}>Desactivar versión</button> : <button className="diagnostic-button diagnostic-primary" disabled={locked} onClick={activateModel}><Check size={16} />{selectedModel.validated && !selectedModel.exploratory ? "Activar versión evaluada" : "Activar versión experimental"}</button>}</div>{(!selectedModel.validated || selectedModel.exploratory) && <p className="diagnostic-help">La activación experimental permite probar esta versión en el artefacto. Sus diagnósticos deben interpretarse con las limitaciones de sus datos y evaluación.</p>}</>}</>}</section>}

    {view === "catalog" && <>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Patrones que puede proponer F3</h3><BookOpen size={19} /></div><p className="diagnostic-help">Los patrones describen cambios medibles. Un período estable puede ser válido: el diagnóstico considera la referencia histórica y la evidencia disponible, y se abstiene cuando no alcanza.</p><div className="diagnostic-catalog">{(status?.catalog || []).map((item) => <article key={item.fault_type}><h4>{item.label || diagnosticLabel(item.fault_type)}</h4><p>{item.description}</p><div className="diagnostic-tags">{(item.entity_kinds || []).map((kind) => <span key={kind}>{({ continuous: "Continua", binary: "Binaria", multistate: "Multivalórica", state: "De estado" })[kind] || kind}</span>)}</div></article>)}</div></section><section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Referencias de funcionamiento habitual</h3><span className="diagnostic-badge">{status?.references?.length || 0} referencias</span></div><p className="diagnostic-help">El diagnóstico utiliza referencias anteriores al episodio. Selecciona períodos normales representativos en Preparación del investigador para construirlas.</p>{!status?.references?.length ? <p className="diagnostic-empty">Todavía no hay referencias guardadas.</p> : <div className="diagnostic-reference-list">{status.references.map((reference, index) => <details key={reference.reference_id || reference.profile_id || index}><summary>{reference.entity_id} · {time(reference.end || reference.end_time)}{reference.partition ? ` · ${PARTITIONS[reference.partition] || reference.partition}` : ""}{reference.exploratory ? " · Exploratoria" : ""}</summary><ValuesTable values={reference} /></details>)}</div>}</section>
    </>}
  </section>;
}
