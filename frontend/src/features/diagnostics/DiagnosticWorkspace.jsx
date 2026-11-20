import { request } from "../../shared/api.js";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, BookOpen, Check, Database, Download, FlaskConical, Loader2, Play, RefreshCw, ShieldCheck, Save, Upload } from "lucide-react";
import { annotationDiagnosticProblem, diagnosticLabel, DIAGNOSTIC_PARTITIONS as PARTITIONS } from "./diagnostic-data.js";
import { DEFAULT_SEGMENT_CONFIG, SEGMENT_SCALES, isGatConfigValid } from "./diagnostic-gat.js";
import { formatRemainingTime } from "./diagnostic-progress.js";
import "./diagnostic-workspace.css";
import DatasetLoadProgress from "../history/DatasetLoadProgress.jsx";

const time = (value) => String(value || "—").replace("T", " ").replace(/\+00:00$/, "Z");
const number = (value) => Number(value || 0).toLocaleString("es-CL");
const metric = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)} %` : "—";
const idOf = (value) => value.annotation_id || value.id;
const ACTIVE_JOB = new Set(["queued", "running"]);
const STATE_LABELS = { queued: "En espera", running: "Procesando", completed: "Completado", failed: "No se completó" };
const FEATURE_LABELS = {
  valid_count: "Lecturas válidas", sample_count: "Lecturas", duration_seconds: "Duración (s)",
  invalid_ratio: "Proporción de lecturas no válidas", constant_ratio: "Proporción constante",
  constant_seconds: "Persistencia constante (s)", gap_seconds: "Brecha de reportes (s)",
  slope: "Pendiente", trend: "Tendencia", variance: "Varianza", mean: "Media", median: "Mediana",
  mad: "Desviación absoluta mediana", score: "Puntaje", confidence: "Indicador de respaldo",
  entity_id: "Entidad", start: "Inicio", end: "Fin", reference_id: "Referencia", source: "Origen",
};


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
  const [view, setView] = useState("prepare");
  const [selectedAnnotations, setSelectedAnnotations] = useState([]);
  const [canonicalIds, setCanonicalIds] = useState([]);
  const [gatConfig, setGatConfig] = useState(DEFAULT_SEGMENT_CONFIG);
  const [gatConfigName, setGatConfigName] = useState("GAT-LSTM inicial");
  const [datasetId, setDatasetId] = useState("");
  const [datasetDetail, setDatasetDetail] = useState(null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [modelId, setModelId] = useState("");
  const [evaluationDatasetId, setEvaluationDatasetId] = useState("");
  const [evaluationPartition, setEvaluationPartition] = useState("validation");
  const [busy, setBusy] = useState("");
  const [exporting, setExporting] = useState("");
  const [modelFile, setModelFile] = useState(null);
  const modelFileInput = useRef(null);
  const storageKey = `diagnostic-job:${apiBaseUrl}`;
  const [job, setJob] = useState(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey) || "null");
      return saved?.client_action === "gat-train" || saved?.operation === "train_gat" ? null : saved;
    } catch { return null; }
  });
  const mounted = useRef(true);
  const controllers = useRef(new Set());
  const datasetList = status?.datasets || [];
  const models = status?.models || [];
  const activeJob = ACTIVE_JOB.has(job?.status);
  const datasetBuildJob = job?.client_action === "gat-build" || job?.operation === "build_gat_dataset";
  const operationProgress = Math.max(0, Math.min(100, Number(job?.progress) || 0));
  const jobClockStartedAt = Date.parse(job?.created_at);
  const jobElapsedSeconds = Number.isFinite(jobClockStartedAt)
    ? Math.max(0, Math.floor((Date.now() - jobClockStartedAt) / 1000)) : null;
  const locked = Boolean(busy) || activeJob;
  const selectedModel = models.find((item) => item.model_id === modelId);
  const selectedDataset = datasetList.find((item) => item.dataset_id === datasetId);
  const selectedGatDataset = selectedDataset?.method === "gat_lstm" ? selectedDataset : null;
  const evaluationDataset = datasetList.find((item) => item.dataset_id === evaluationDatasetId);
  const evaluationCount = evaluationDataset?.partition_counts?.[evaluationPartition] || 0;
  const gatConfigValid = isGatConfigValid(gatConfig);
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
      const sourcePath = "/diagnostics/sources?method=gat_lstm";
      const [nextStatus, nextSources] = await Promise.all([call("/diagnostics/status"), call(sourcePath)]);
      if (!mounted.current) return;
      setStatus(nextStatus);
      setSources(nextSources);
      setDatasetId((current) => nextStatus.datasets?.some((item) => item.dataset_id === current) ? current : nextStatus.datasets?.[0]?.dataset_id || "");
      setModelId((current) => nextStatus.models?.some((item) => item.model_id === current) ? current : nextStatus.active_model_id || nextStatus.models?.[0]?.model_id || "");
      setEvaluationDatasetId((current) => nextStatus.datasets?.some((item) => item.dataset_id === current) ? current : nextStatus.datasets?.[0]?.dataset_id || "");
      setSelectedAnnotations((current) => current.filter((id) => nextSources.annotations?.some((item) => idOf(item) === id)));
      const trainingSources = nextSources.training_sources || nextSources.canonical_sources || [];
      setCanonicalIds((current) => {
        const retained = current.filter((id) => trainingSources.some((item) => item.source_id === id));
        if (retained.length) return [retained[0]];
        const preferred = trainingSources.find((item) => item.labelled_import) || trainingSources[0];
        return preferred ? [preferred.source_id] : [];
      });
    } catch (failure) { if (mounted.current && failure.name !== "AbortError") setError(failure.message); }
    finally { if (mounted.current) setLoading(false); }
  }, [call]);

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

  const annotations = useMemo(() => (sources.annotations || []).filter((item) => ["normal", "sensor_fault", "environmental_anomaly"].includes(item.label)), [sources]);
  const eligibleAnnotations = annotations.filter((item) => !annotationDiagnosticProblem(item, false));
  const annotationIds = selectedAnnotations.filter((id) => eligibleAnnotations.some((item) => idOf(item) === id));
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
    if (!gatConfigValid) { setError("Revisa la configuración del dataset y selecciona al menos una escala temporal."); return; }
    setBusy("save-config"); setError(""); setNotice("");
    try {
      await call("/diagnostics/configurations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: gatConfigName, config: gatConfig }) });
      setNotice("Configuración GAT-LSTM guardada en el workbench."); await refresh();
    } catch (failure) { if (failure.name !== "AbortError") setError(failure.message); }
    finally { setBusy(""); }
  }


  function startGatBuild() {
    if (!gatConfigValid) { setError("Revisa la configuración del dataset y selecciona al menos una escala temporal."); return; }
    const available = sources.training_sources || sources.canonical_sources || [];
    if (!canonicalIds.length || canonicalIds.some((id) => available.find((item) => item.source_id === id)?.partition !== "development")) { setError("Selecciona fuentes de Desarrollo."); return; }
    startJob("/diagnostics/build", { method: "gat_lstm", source_ids: canonicalIds,
      annotation_ids: annotationIds, gat_config: gatConfig, injection_ids: [], include_drafts: false }, "gat-build");
  }

  async function importModel() {
    if (!modelFile || locked) return;
    if (!modelFile.name.toLowerCase().endsWith(".zip")) {
      setError("Selecciona un archivo ZIP con los pesos y los metadatos del modelo."); return;
    }
    if (modelFile.size > 256 * 1024 ** 2) {
      setError("El ZIP supera el límite de 256 MiB."); return;
    }
    setBusy("import-model"); setError(""); setNotice("");
    try {
      const imported = await call("/diagnostics/models/import", {
        method: "POST", headers: { "Content-Type": "application/zip" }, body: modelFile,
      });
      if (!mounted.current) return;
      setModelId(imported.model_id);
      setModelFile(null);
      if (modelFileInput.current) modelFileInput.current.value = "";
      setNotice(imported.imported_files
        ? "Modelo importado: pesos, metadatos, grafo y preprocesador disponibles. Puedes activar la versión seleccionada."
        : "Este modelo ya está importado. Se seleccionó su versión existente.");
      await refresh();
    } catch (failure) { if (mounted.current && failure.name !== "AbortError") setError(failure.message); }
    finally { if (mounted.current) setBusy(""); }
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
        setNotice("Versión desactivada. Puedes evaluarla y volver a activarla. Las nuevas ejecuciones indicarán que falta un modelo activo; las reproducciones ya iniciadas conservan su versión hasta reiniciarlas.");
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
    <header className="diagnostic-header"><div><span className="diagnostic-eyebrow">Tercera etapa · Diagnóstico</span><h2>Del comportamiento habitual al patrón de falla</h2><p>F3 clasifica las sospechas pendientes con GAT-LSTM v2. La preparación del modelo corresponde al investigador; la operación utiliza la versión activa y puede abstenerse cuando falta evidencia.</p></div><button className="diagnostic-button" onClick={() => { setError(""); refresh(); }} disabled={loading}><RefreshCw size={16} className={loading ? "diagnostic-spin" : ""} />Actualizar</button></header>
    <div className="diagnostic-mode"><Activity size={18} /><div><strong>{status?.mode === "active_model" ? "Modelo entrenado activo" : loading && !status ? "Consultando estado…" : "Clasificador GAT-LSTM v2 no disponible"}</strong><span>{status?.active_model_id ? `Versión ${status.active_model_id}` : "Se necesita un modelo GAT-LSTM v2 utilizable para emitir diagnósticos."}</span></div>{status?.models?.some((item) => item.model_id === status.active_model_id && (item.exploratory || !item.validated)) && <span className="diagnostic-badge diagnostic-exploratory">Experimental</span>}</div>
    {error && <div className="diagnostic-alert" role="alert">{error}</div>}
    {notice && <div className="diagnostic-notice" role="status">{notice}</div>}
    {job && !(datasetBuildJob && view === "prepare") && <section className={`history-load-progress ${job.status === "completed" ? "ready" : ""} ${job.status === "failed" ? "failed" : ""}`} aria-live="polite">
      <div className="history-load-heading"><span>{STATE_LABELS[job.status] || job.status}</span><strong>{Math.round(operationProgress)}%</strong></div>
      <div className="progress-track" role="progressbar" aria-label="Avance de la operación" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(operationProgress)}><span style={{ width: `${operationProgress}%` }} /></div>
      <p>{job.message || "Preparando operación…"}</p>
      {jobElapsedSeconds != null && <div className="history-load-meta">Transcurrido: {formatRemainingTime(jobElapsedSeconds)}</div>}
      {job.tracking_paused && <button className="diagnostic-button" onClick={() => { setError(""); setJob((value) => ({ ...value, tracking_paused: false })); }}>Reintentar seguimiento</button>}
    </section>}
    <nav className="diagnostic-tabs" aria-label="Vistas del clasificador">{[["prepare", "Preparación del investigador", Database], ["models", "Modelos y evaluación", FlaskConical], ["catalog", "Catálogo y sensores", BookOpen]].map(([key, label, Icon]) => <button key={key} className={view === key ? "active" : ""} onClick={() => setView(key)} aria-pressed={view === key}><Icon size={16} />{label}</button>)}</nav>

    {view === "models" && <section className="diagnostic-panel" aria-label="Importar modelo entrenado">
      <div className="diagnostic-panel-title"><h3>Importar modelo entrenado</h3><Upload size={19} /></div>
      <p className="diagnostic-help">Selecciona el ZIP con las carpetas weights, models, graphs y preprocessors. Se cargarán los pesos, metadatos, grafo y preprocesador, listos para activar sin volver a entrenar.</p>
      <div className="diagnostic-import-controls">
        <label>Archivo del modelo (.zip)<input ref={modelFileInput} type="file" accept=".zip,application/zip,application/x-zip-compressed" disabled={locked} onChange={(event) => setModelFile(event.target.files?.[0] || null)} /></label>
        <button className="diagnostic-button diagnostic-primary" disabled={locked || !modelFile} onClick={importModel}>{busy === "import-model" ? <Loader2 size={16} className="diagnostic-spin" /> : <Upload size={16} />}{busy === "import-model" ? "Importando modelo…" : "Importar ZIP"}</button>
      </div>
      {busy === "import-model" && <p className="diagnostic-help" role="status">Cargando el ZIP y comprobando los pesos y la compatibilidad del modelo…</p>}
    </section>}

    {view === "prepare" && <>
      <section className="diagnostic-panel">
        <div className="diagnostic-panel-title"><h3>Configuración del dataset</h3></div>
        <p className="diagnostic-help">Prepara secuencias de Desarrollo con tres clases: normal, anomalía ambiental y falla del sensor. Las secuencias conservan el subtipo de falla cuando está etiquetado.</p>
        <label>Configuración guardada
          <select value="" onChange={(event) => {
            const saved = status?.gat_configurations?.find((item) => item.configuration_id === event.target.value);
            if (saved) { setGatConfig(saved.config); setGatConfigName(saved.name); }
          }}>
            <option value="">Usar o editar configuración actual</option>
            {(status?.gat_configurations || []).map((item) => <option key={item.configuration_id} value={item.configuration_id}>{item.name} · {time(item.created_at)}</option>)}
          </select>
        </label>
        <div className="diagnostic-build-controls">
          <label>Nombre<input value={gatConfigName} onChange={(event) => setGatConfigName(event.target.value)} maxLength={100} /></label>
          <label>Zona horaria de los patrones diarios<input value={gatConfig.local_timezone || "America/Santiago"} onChange={(event) => setGatConfig((value) => ({ ...value, local_timezone: event.target.value }))} /></label>
        </div>
        <details>
          <summary>Escalas temporales</summary>
          <div className="diagnostic-build-controls">{SEGMENT_SCALES.map(({ hours, resolutionMinutes }, index) =>
            <label key={hours}>
              <input type="checkbox" checked={(gatConfig.active_scales || DEFAULT_SEGMENT_CONFIG.active_scales)[index]} onChange={(event) => {
                const checked = event.target.checked;
                setGatConfig((value) => ({ ...value, active_scales: (value.active_scales || DEFAULT_SEGMENT_CONFIG.active_scales).map((active, i) => i === index ? checked : active) }));
              }} />
              Contexto de {hours} h · resolución de {resolutionMinutes} min
            </label>
          )}</div>
          <p className="diagnostic-help">Selecciona al menos una escala.</p>
        </details>
        <button className="diagnostic-button" disabled={locked || !gatConfigValid} onClick={saveGatConfiguration}><Save size={15} />Guardar configuración</button>
      </section>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>1. Fuente de Desarrollo y etiquetas</h3></div><p className="diagnostic-help">GAT-LSTM recibe una o varias fuentes de Desarrollo. Cada CSV aporta sus etiquetas por fila y conserva su contexto y sus episodios por separado. Validación y Prueba no se consultan para preparar ni ajustar el modelo; quedan reservadas para generar eventos futuros.</p><div className="diagnostic-source-list">{(sources.training_sources || sources.canonical_sources || []).map((item) => <label className="diagnostic-source" key={item.source_id}><input type="checkbox" checked={canonicalIds.includes(item.source_id)} disabled={locked} onChange={() => toggle(item.source_id, setCanonicalIds)} /><span><strong>{item.labelled_import ? "CSV etiquetado · " : "Histórico · "}{PARTITIONS[item.partition] || item.partition}</strong><small>{item.filename} · {number(item.rows)} filas</small><small>SHA‑256 {item.sha256}</small>{item.labelled_import && <small>{number(item.imported_labels)} filas etiquetadas · {number(item.imported_synthetic)} sintéticas</small>}</span></label>)}</div><h4>Anotaciones adicionales disponibles</h4><div className="diagnostic-source-list">{annotations.map((item) => { const id = idOf(item); const selectable = ["reviewed", "verified"].includes(item.status) && item.source_sha256 && canonicalIds.some((sourceId) => (sources.training_sources || sources.canonical_sources || []).find((source) => source.source_id === sourceId)?.sha256 === item.source_sha256); return <label className={`diagnostic-source ${selectable ? "" : "diagnostic-source-disabled"}`} key={id}><input type="checkbox" checked={annotationIds.includes(id)} disabled={locked || !selectable} onChange={() => toggle(id, setSelectedAnnotations)} /><span><strong>{diagnosticLabel(item.fault_type || item.label)}</strong><span>{item.entity_id}</span><small>{time(item.start)} → {time(item.end)} · {item.status}</small>{!selectable && <em>Fuente no seleccionada o identidad no verificable.</em>}</span></label>; })}</div></section>
      <section className="diagnostic-panel">
        <div className="diagnostic-panel-title"><h3>2. Preparar dataset</h3></div>
        <p className="diagnostic-help">Prepara las secuencias con la configuración actual y las fuentes de Desarrollo seleccionadas. Las etiquetas incluidas en cada CSV se incorporan automáticamente.</p>
        <button className="diagnostic-button diagnostic-primary" disabled={locked || !gatConfigValid || !canonicalIds.length} onClick={startGatBuild}><Database size={16} />Preparar dataset</button>
        {!canonicalIds.length && <p className="diagnostic-help">Selecciona una fuente de Desarrollo para habilitar la preparación.</p>}
        {!gatConfigValid && <p className="diagnostic-help">Revisa la configuración antes de preparar el dataset.</p>}
        {datasetBuildJob && <DatasetLoadProgress job={job} onRetry={() => { setError(""); setJob((value) => ({ ...value, tracking_paused: false })); }} />}
      </section>
      <section className="diagnostic-panel">
        <div className="diagnostic-panel-title"><h3>3. Descargar dataset</h3></div>
        <p className="diagnostic-help">El entrenamiento se realiza por consola en un equipo con GPU compatible (NVIDIA con CUDA o AMD con ROCm).</p>
        <label>Dataset preparado<select value={datasetId} disabled={locked} onChange={(event) => setDatasetId(event.target.value)}><option value="">Selecciona un dataset</option>{datasetList.filter((item) => item.method === "gat_lstm").map((item) => <option key={item.dataset_id} value={item.dataset_id}>{time(item.created_at)} · {item.dataset_id.slice(0, 12)}</option>)}</select></label>
        {!selectedGatDataset ? <p className="diagnostic-help">Prepara un dataset para ver sus particiones y descargarlo.</p> : <>
          {datasetLoading && <p className="diagnostic-help">Cargando detalles del dataset…</p>}
          <div className="diagnostic-dataset-summary"><strong>Ejemplos por partición</strong><ValuesTable values={datasetDetail?.partition_counts || selectedGatDataset.partition_counts} /></div>
          <button className="diagnostic-button" disabled={locked || Boolean(exporting)} onClick={() => downloadDataset(datasetId)}><Download size={15} />Descargar dataset</button>
        </>}
      </section>
    </>}



    {view === "models" && <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Versiones, evaluación y activación</h3><span className="diagnostic-badge">{models.length} versiones</span></div><p className="diagnostic-help">Cada versión aprende de las características de Desarrollo. Evalúa su comportamiento sobre episodios separados antes de incorporarla; una etiqueta nueva se utiliza en el próximo entrenamiento.</p>{!models.length ? <div className="diagnostic-empty"><FlaskConical size={26} /><p>Aún no hay modelos entrenados.</p><span>Importa el ZIP del modelo entrenado para poder activarlo.</span></div> : <><label>Versión candidata o activa<select value={modelId} onChange={(event) => setModelId(event.target.value)}>{models.map((model) => <option key={model.model_id} value={model.model_id}>{time(model.created_at)} · {model.model_id.slice(0, 12)}{status?.active_model_id === model.model_id ? " · Activa" : ""}{model.exploratory ? " · Exploratoria" : ""}</option>)}</select></label>{selectedModel && <><div className="diagnostic-model-summary"><span className="diagnostic-id">{selectedModel.model_id}</span><span>{number(selectedModel.training_count)} ejemplos de entrenamiento</span>{selectedModel.excluded_count > 0 && <details><summary>{number(selectedModel.excluded_count)} ejemplos omitidos</summary><p className="diagnostic-help">Faltan reportes o caracteristicas compatibles para aprender de estas ventanas. Incluye contexto previo en los casos de pico; cada ventana requiere al menos tres reportes.</p></details>}{status?.active_model_id === modelId && <span className="diagnostic-badge"><Check size={13} />Activa</span>}{selectedModel.validated ? <span className="diagnostic-badge"><ShieldCheck size={13} />Evaluación independiente disponible</span> : <span className="diagnostic-badge diagnostic-exploratory">Sin evaluación independiente</span>}{selectedModel.exploratory && <span className="diagnostic-badge diagnostic-exploratory">Entrenamiento exploratorio</span>}</div><div className="diagnostic-tags">{(selectedModel.classes || []).map((label) => <span key={label}>{diagnosticLabel(label)}</span>)}</div>{selectedModel.method === "gat_lstm" ? <p className="diagnostic-help">GAT-LSTM se ajusta solo con Desarrollo. Validación y Prueba permanecen sin leer y se reservan para generar eventos futuros.</p> : <><div className="diagnostic-evaluate-controls"><label>Conjunto para evaluar<select value={evaluationDatasetId} onChange={(event) => setEvaluationDatasetId(event.target.value)}><option value="">Selecciona un conjunto</option>{datasetList.map((item) => <option key={item.dataset_id} value={item.dataset_id}>{time(item.created_at)} · {item.dataset_id.slice(0, 12)}</option>)}</select></label><label>Partición de evaluación<select value={evaluationPartition} onChange={(event) => setEvaluationPartition(event.target.value)}><option value="validation">Validación</option><option value="test">Prueba</option></select></label><button className="diagnostic-button" disabled={locked || !evaluationDatasetId || !evaluationCount} onClick={() => startJob(`/diagnostics/models/${encodeURIComponent(modelId)}/evaluate`, { dataset_id: evaluationDatasetId, partition: evaluationPartition }, "evaluate")}><Play size={15} />Evaluar versión</button></div><p className="diagnostic-help">{number(evaluationCount)} ejemplos disponibles en {PARTITIONS[evaluationPartition]}. Prueba mide el resultado final y no sustituye la evaluación independiente necesaria para revisar una versión.</p><Evaluation value={selectedModel.validation} />{(selectedModel.evaluations || []).filter((evaluation) => evaluation.partition === "test").map((evaluation, index) => <Evaluation key={evaluation.evaluation_id || index} value={evaluation} />)}</>}<div className="diagnostic-actions">{status?.active_model_id === modelId ? <button className="diagnostic-button" disabled={locked} onClick={deactivateModel}>Desactivar versión</button> : <button className="diagnostic-button diagnostic-primary" disabled={locked} onClick={activateModel}><Check size={16} />{selectedModel.validated && !selectedModel.exploratory ? "Activar versión evaluada" : "Activar versión experimental"}</button>}</div>{(!selectedModel.validated || selectedModel.exploratory) && <p className="diagnostic-help">La activación experimental permite probar esta versión en el artefacto. Sus diagnósticos deben interpretarse con las limitaciones de sus datos y evaluación.</p>}</>}</>}</section>}

    {view === "catalog" && <>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Tipos de falla de GAT-LSTM v2</h3><BookOpen size={19} /></div><p className="diagnostic-help">Los patrones describen cambios medibles. Un período estable puede ser válido: el diagnóstico considera la referencia histórica y la evidencia disponible, y se abstiene cuando no alcanza.</p><div className="diagnostic-catalog">{(status?.catalog || []).map((item) => <article key={item.fault_type}><h4>{item.label || diagnosticLabel(item.fault_type)}</h4><p>{item.description}</p><div className="diagnostic-tags">{(item.entity_kinds || []).map((kind) => <span key={kind}>{({ continuous: "Continua", binary: "Binaria", multistate: "Multivalórica", state: "De estado" })[kind] || kind}</span>)}</div></article>)}</div></section><section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Referencias de funcionamiento habitual</h3><span className="diagnostic-badge">{status?.sensor_profiles?.length || 0} referencias</span></div><p className="diagnostic-help">El clasificador prepara automáticamente una referencia para cada sensor nuevo a partir del histórico anterior a la evaluación, sin reentrenar el modelo. Las referencias normales revisadas que guardes aquí tienen prioridad.</p>{!status?.sensor_profiles?.length ? <p className="diagnostic-empty">Todavía no hay referencias guardadas.</p> : <div className="diagnostic-reference-list">{status.sensor_profiles.map((reference, index) => <details key={reference.reference_id || reference.profile_id || index}><summary>{reference.entity_id} · {time(reference.reference_end || reference.end || reference.end_time)}{reference.partition ? ` · ${PARTITIONS[reference.partition] || reference.partition}` : ""}{reference.exploratory ? " · Exploratoria" : ""}</summary><ValuesTable values={reference} /></details>)}</div>}</section>
    </>}
  </section>;
}
