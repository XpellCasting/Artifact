import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, ArrowLeft, ArrowRight, BookOpen, Check, Database, Download, FlaskConical, Loader2, Play, RefreshCw, Search, ShieldCheck } from "lucide-react";
import { annotationDiagnosticProblem, diagnosticLabel, DIAGNOSTIC_PARTITIONS as PARTITIONS } from "./diagnostic-data.js";
import "./diagnostic-workspace.css";

const time = (value) => String(value || "—").replace("T", " ").replace(/\+00:00$/, "Z");
const number = (value) => Number(value || 0).toLocaleString("es-CL");
const metric = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)} %` : "—";
const idOf = (value) => value.annotation_id || value.id;
const ACTIVE_JOB = new Set(["queued", "running"]);
const STATE_LABELS = { queued: "En espera", running: "Procesando", completed: "Completado", failed: "No se completó" };
const RESULT_LABELS = { compatible: "Patrón compatible", indeterminate: "Tipo indeterminado", no_fault_evidence: "Sin evidencia suficiente de falla" };
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
  return <div className="diagnostic-evaluation"><span>{PARTITIONS[value.partition] || "Evaluación"}</span><span>Exactitud <b>{metric(value.accuracy)}</b></span><span>F1 macro <b>{metric(value.macro_f1)}</b></span>{Number.isFinite(value.evaluable_accuracy) && <span>Exactitud en casos evaluables <b>{metric(value.evaluable_accuracy)}</b></span>}<details><summary>Detalle de evaluación</summary><ValuesTable values={value} /></details></div>;
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
  const locked = Boolean(busy) || activeJob;
  const selectedModel = models.find((item) => item.model_id === modelId);
  const selectedDataset = datasetList.find((item) => item.dataset_id === datasetId);
  const evaluationDataset = datasetList.find((item) => item.dataset_id === evaluationDatasetId);
  const evaluationCount = evaluationDataset?.partition_counts?.[evaluationPartition] || 0;

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
      const [nextStatus, nextSources] = await Promise.all([call("/diagnostics/status"), call("/diagnostics/sources")]);
      if (!mounted.current) return;
      setStatus(nextStatus);
      setSources(nextSources);
      setDatasetId((current) => nextStatus.datasets?.some((item) => item.dataset_id === current) ? current : nextStatus.datasets?.[0]?.dataset_id || "");
      setModelId((current) => nextStatus.models?.some((item) => item.model_id === current) ? current : nextStatus.active_model_id || nextStatus.models?.[0]?.model_id || "");
      setEvaluationDatasetId((current) => nextStatus.datasets?.some((item) => item.dataset_id === current) ? current : nextStatus.datasets?.[0]?.dataset_id || "");
      setSelectedAnnotations((current) => current.filter((id) => nextSources.annotations?.some((item) => idOf(item) === id)));
      setSelectedInjections((current) => current.filter((id) => nextSources.injections?.some((item) => item.injection_id === id)));
    } catch (failure) { if (mounted.current && failure.name !== "AbortError") setError(failure.message); }
    finally { if (mounted.current) setLoading(false); }
  }, [call]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    try { if (ACTIVE_JOB.has(job?.status)) sessionStorage.setItem(storageKey, JSON.stringify(job)); else sessionStorage.removeItem(storageKey); } catch { /* A blocked storage area does not stop the current job. */ }
  }, [job, storageKey]);

  useEffect(() => {
    if (!job?.job_id || !ACTIVE_JOB.has(job.status)) return undefined;
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const next = await request(apiBaseUrl, `/diagnostics/jobs/${encodeURIComponent(job.job_id)}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setJob(next);
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
    try {
      const next = await call(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (!mounted.current) return;
      setJob(next);
      if (next.status === "completed") {
        if (next.result?.dataset_id) setDatasetId(next.result.dataset_id);
        if (next.result?.model_id) setModelId(next.result.model_id);
        setNotice(next.message || "Operación completada."); await refresh();
      }
      else if (next.status === "failed") setError(next.error || next.message || "La operación no se completó.");
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
    {job && <section className="diagnostic-job" aria-live="polite"><div><strong>{STATE_LABELS[job.status] || job.status}</strong><span>{job.message || "Preparando operación…"}</span></div>{activeJob && <progress max="100" value={Math.max(0, Math.min(100, Number(job.progress) || 0))} aria-label="Avance de la operación" />}{job.tracking_paused && <button className="diagnostic-button" onClick={() => { setError(""); setJob((value) => ({ ...value, tracking_paused: false })); }}>Reintentar seguimiento</button>}</section>}
    <nav className="diagnostic-tabs" aria-label="Vistas del clasificador">{[["results", "Diagnósticos", Activity], ["prepare", "Preparación del investigador", Database], ["models", "Modelos y evaluación", FlaskConical], ["catalog", "Catálogo y referencias", BookOpen]].map(([key, label, Icon]) => <button key={key} className={view === key ? "active" : ""} onClick={() => setView(key)} aria-pressed={view === key}><Icon size={16} />{label}</button>)}</nav>

    {view === "results" && <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Diagnósticos de las sospechas pendientes</h3><span className="diagnostic-badge">{number(results.total)} resultados</span></div><div className="diagnostic-result-controls"><label>Origen de los resultados<select value={resultsSource} onChange={(event) => { setResultsSource(event.target.value); setResultsOffset(0); }}><option value="history">Histórico y eventos</option><option value="validation_replay">Reproducción de evaluación</option></select></label><button className="diagnostic-button" disabled={resultsLoading} onClick={() => setResultsRefresh((value) => value + 1)}><RefreshCw size={15} />Actualizar diagnósticos</button></div><p className="diagnostic-help">La compatibilidad expresa semejanza con un patrón; no es una probabilidad calibrada ni confirma una avería física. Los casos resueltos como ambientales por F2 conservan su recorrido.</p>
      {resultsLoading ? <p className="diagnostic-empty"><Loader2 size={19} className="diagnostic-spin" />Cargando diagnósticos…</p> : !results.items?.length ? <div className="diagnostic-empty"><Activity size={28} /><p>Aún no hay diagnósticos en este origen.</p><span>Procesa eventos con F1 y F2 para que las sospechas pendientes lleguen a F3.</span></div> : <div className="diagnostic-results">{results.items.map((item, index) => <article key={item.diagnosis_id || item.diagnostic_id || item.result_id || `${item.entity_id}:${item.timestamp}:${index}`} className="diagnostic-result"><div className="diagnostic-result-heading"><div><span className="diagnostic-eyebrow">{RESULT_LABELS[item.status] || item.status}</span><h4>{item.fault_type ? diagnosticLabel(item.fault_type) : item.status === "no_fault_evidence" ? "Sin patrón de falla concluyente" : "Tipo indeterminado"}</h4><p>{item.entity_id}</p></div>{Number.isFinite(item.compatibility) && <div className="diagnostic-compatibility"><span>Compatibilidad</span><strong>{item.compatibility.toFixed(2)}</strong><small>Índice, no probabilidad</small></div>}</div><small>{time(item.timestamp)} · {item.model_version || `Catálogo ${item.catalog_version || "—"}`}</small>{item.needs_review && <span className="diagnostic-badge diagnostic-exploratory">Evidencia pendiente de revisión</span>}<ul className="diagnostic-reasons">{(item.reasons || []).map((reason, reasonIndex) => <li key={reasonIndex}>{typeof reason === "string" ? reason : JSON.stringify(reason)}</li>)}</ul><details><summary>Características y referencia utilizadas</summary><h5>Características disponibles al decidir</h5><ValuesTable values={item.features} /><h5>Referencia histórica</h5><ValuesTable values={item.reference} /></details></article>)}</div>}
      <div className="diagnostic-pagination"><button className="diagnostic-button" disabled={resultsLoading || resultsOffset === 0} onClick={() => setResultsOffset((value) => Math.max(0, value - 30))}><ArrowLeft size={15} />Anterior</button><span>{results.total ? `${number(resultsOffset + 1)}–${number(Math.min(resultsOffset + (results.items?.length || 0), results.total))} de ${number(results.total)}` : "0 resultados"}</span><button className="diagnostic-button" disabled={resultsLoading || results.next_offset == null} onClick={() => setResultsOffset(results.next_offset)} >Siguiente<ArrowRight size={15} /></button></div>
    </section>}

    {view === "prepare" && <>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>1. Seleccionar fragmentos y casos documentados</h3><span className="diagnostic-badge">{annotationIds.length + injectionIds.length} seleccionados</span></div><p className="diagnostic-help">Marca períodos de funcionamiento correcto para construir referencias y controles. Añade fallas reales con un tipo documentado, anomalías ambientales verificadas o inyecciones guardadas para construir ejemplos de esos patrones. Se conserva la partición y la revisión de cada fuente.</p><div className="diagnostic-source-tools"><label className="diagnostic-search">Buscar entidad, patrón o identificador<div><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filtrar fuentes" /></div></label><label className="diagnostic-check"><input type="checkbox" checked={includeDrafts} disabled={locked} onChange={(event) => setIncludeDrafts(event.target.checked)} /><span>Incluir borradores y casos exploratorios<small>El conjunto y sus modelos conservarán esta condición.</small></span></label></div>
        <div className="diagnostic-source-columns"><section><div className="diagnostic-panel-title"><h4>Fragmentos revisados</h4><button className="diagnostic-text-button" disabled={locked} onClick={() => setSelectedAnnotations(eligibleAnnotations.filter(matched).map(idOf))}>Seleccionar aptos visibles</button></div><div className="diagnostic-source-list">{annotations.filter(matched).map((annotation) => { const id = idOf(annotation); const problem = annotationDiagnosticProblem(annotation, includeDrafts); return <label key={id} className={`diagnostic-source ${problem ? "diagnostic-source-disabled" : ""}`}><input type="checkbox" checked={annotationIds.includes(id)} disabled={locked || Boolean(problem)} onChange={() => toggle(id, setSelectedAnnotations)} /><span><strong>{diagnosticLabel(annotation.fault_type || annotation.label)}</strong><span>{annotation.entity_id}</span><small>{time(annotation.start)} → {time(annotation.end)}</small><small>{PARTITIONS[annotation.partition] || annotation.partition} · revisión {annotation.revision} · {annotation.status === "verified" ? "Verificado" : "Borrador"}</small>{problem && <em>{problem}</em>}</span></label>; })}{!annotations.filter(matched).length && <p className="diagnostic-empty">No hay fragmentos coincidentes. Guarda anotaciones normales, fallas identificadas o anomalías ambientales en Revisión y etiquetado.</p>}</div></section><section><div className="diagnostic-panel-title"><h4>Inyecciones guardadas</h4><button className="diagnostic-text-button" disabled={locked} onClick={() => setSelectedInjections(eligibleInjections.filter(matched).map((item) => item.injection_id))}>Seleccionar aptas visibles</button></div><div className="diagnostic-source-list">{(sources.injections || []).filter(matched).map((injection) => <label key={injection.injection_id} className={`diagnostic-source ${!injection.evaluable && !includeDrafts ? "diagnostic-source-disabled" : ""}`}><input type="checkbox" checked={injectionIds.includes(injection.injection_id)} disabled={locked || (!injection.evaluable && !includeDrafts)} onChange={() => toggle(injection.injection_id, setSelectedInjections)} /><span><strong>{diagnosticLabel(injection.kind)}</strong><span>{injection.entity_id}</span><small>{time(injection.start)} → {time(injection.end)}</small><small>{PARTITIONS[injection.partition] || injection.partition} · revisión fijada {injection.annotation_revision} · {injection.evaluable ? "Evaluable semisintético" : "Exploratorio"}</small><small>Semilla {injection.seed} · {injection.injection_id}</small>{!injection.evaluable && !includeDrafts && <em>Habilita el uso exploratorio para seleccionar este caso.</em>}</span></label>)}{!(sources.injections || []).filter(matched).length && <p className="diagnostic-empty">No hay inyecciones coincidentes. Puedes generarlas sobre fragmentos normales revisados.</p>}</div></section></div>
        <div className="diagnostic-build-controls"><label>Ventana de análisis (segundos)<input type="number" min="60" max="604800" step="1" value={windowSeconds} disabled={locked} onChange={(event) => setWindowSeconds(event.target.value)} /></label><label>Paso del barrido (segundos)<input type="number" min="60" max="604800" step="1" value={strideSeconds} disabled={locked} onChange={(event) => setStrideSeconds(event.target.value)} /></label><button className="diagnostic-button diagnostic-primary" disabled={locked || !validWindow || !hasBaseline || !(annotationIds.length + injectionIds.length)} onClick={() => startJob("/diagnostics/build", { annotation_ids: annotationIds, injection_ids: injectionIds, window_seconds: Number(windowSeconds), stride_seconds: Number(strideSeconds), include_drafts: includeDrafts }, "build")}>{locked ? <Loader2 size={16} className="diagnostic-spin" /> : <Database size={16} />}Construir referencias y ejemplos</button></div><p className="diagnostic-help">{number(normalCount)} fragmentos normales seleccionados. Las inyecciones incorporan automáticamente su referencia normal original. Selecciona al menos una de estas fuentes. Ventana y paso admiten enteros entre 60 y 604800 segundos. Para entrenar un clasificador se necesitan controles normales y al menos un tipo de falla en Desarrollo. Validación y Prueba se reservan para evaluar; sus ejemplos no entrenan el modelo.</p>
      </section>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>2. Conjuntos de aprendizaje</h3><span className="diagnostic-badge">{datasetList.length} guardados</span></div>{!datasetList.length ? <p className="diagnostic-empty">Construye el primer conjunto para revisar sus ejemplos y descargarlo.</p> : <><label>Conjunto guardado<select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>{datasetList.map((item) => <option key={item.dataset_id} value={item.dataset_id}>{time(item.created_at)} · {number(item.example_count)} ejemplos · {item.dataset_id.slice(0, 10)}{item.exploratory ? " · Exploratorio" : ""}</option>)}</select></label><div className="diagnostic-dataset-summary"><span>{selectedDataset?.dataset_id}</span>{selectedDataset?.exploratory && <span className="diagnostic-badge diagnostic-exploratory">Exploratorio</span>}<Counts counts={selectedDataset?.label_counts} /><Counts counts={selectedDataset?.partition_counts} partitions /></div>{(selectedDataset?.warnings || []).map((warning, index) => <p className="diagnostic-help" key={index}>{warning}</p>)}<div className="diagnostic-actions"><button className="diagnostic-button" disabled={!datasetId || Boolean(exporting)} onClick={() => downloadDataset(datasetId)}>{exporting === datasetId ? <Loader2 size={15} className="diagnostic-spin" /> : <Download size={15} />}Descargar conjunto ZIP</button><button className="diagnostic-button diagnostic-primary" disabled={locked || !selectedDataset?.partition_counts?.development} onClick={() => startJob("/diagnostics/train", { dataset_id: datasetId }, "train")}><FlaskConical size={16} />Entrenar versión candidata</button></div><p className="diagnostic-help">El entrenamiento utiliza únicamente Desarrollo y no activa automáticamente la versión. El conjunto conserva la procedencia y las características de cada ejemplo.</p><details><summary>Vista previa de ejemplos</summary>{datasetLoading ? <p className="diagnostic-help">Cargando ejemplos…</p> : <div className="diagnostic-example-list">{(datasetDetail?.examples_preview || []).map((example, index) => <details key={example.example_id || index}><summary>{diagnosticLabel(example.fault_type || example.label)} · {example.entity_id} · {PARTITIONS[example.partition] || example.partition} · {time(example.timestamp || example.end)}</summary><ValuesTable values={example} /></details>)}{!datasetDetail?.examples_preview?.length && <p className="diagnostic-help">Sin ejemplos en la vista previa.</p>}</div>}</details></>}</section>
    </>}

    {view === "models" && <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Versiones, evaluación y activación</h3><span className="diagnostic-badge">{models.length} versiones</span></div><p className="diagnostic-help">Cada versión aprende de las características de Desarrollo. Evalúa su comportamiento sobre episodios separados antes de incorporarla; una etiqueta nueva se utiliza en el próximo entrenamiento.</p>{!models.length ? <div className="diagnostic-empty"><FlaskConical size={26} /><p>Aún no hay modelos entrenados.</p><span>Construye un conjunto con normalidad y fallas de Desarrollo en Preparación del investigador.</span></div> : <><label>Versión candidata o activa<select value={modelId} onChange={(event) => setModelId(event.target.value)}>{models.map((model) => <option key={model.model_id} value={model.model_id}>{time(model.created_at)} · {model.model_id.slice(0, 12)}{status?.active_model_id === model.model_id ? " · Activa" : ""}{model.exploratory ? " · Exploratoria" : ""}</option>)}</select></label>{selectedModel && <><div className="diagnostic-model-summary"><span className="diagnostic-id">{selectedModel.model_id}</span><span>{number(selectedModel.training_count)} ejemplos de entrenamiento</span>{selectedModel.excluded_count > 0 && <details><summary>{number(selectedModel.excluded_count)} ejemplos omitidos</summary><p className="diagnostic-help">Faltan reportes o caracteristicas compatibles para aprender de estas ventanas. Incluye contexto previo en los casos de pico; cada ventana requiere al menos tres reportes.</p></details>}{status?.active_model_id === modelId && <span className="diagnostic-badge"><Check size={13} />Activa</span>}{selectedModel.validated ? <span className="diagnostic-badge"><ShieldCheck size={13} />Evaluación independiente disponible</span> : <span className="diagnostic-badge diagnostic-exploratory">Sin evaluación independiente</span>}{selectedModel.exploratory && <span className="diagnostic-badge diagnostic-exploratory">Entrenamiento exploratorio</span>}</div><div className="diagnostic-tags">{(selectedModel.classes || []).map((label) => <span key={label}>{diagnosticLabel(label)}</span>)}</div><div className="diagnostic-evaluate-controls"><label>Conjunto para evaluar<select value={evaluationDatasetId} onChange={(event) => setEvaluationDatasetId(event.target.value)}><option value="">Selecciona un conjunto</option>{datasetList.map((item) => <option key={item.dataset_id} value={item.dataset_id}>{time(item.created_at)} · {item.dataset_id.slice(0, 12)}</option>)}</select></label><label>Partición de evaluación<select value={evaluationPartition} onChange={(event) => setEvaluationPartition(event.target.value)}><option value="validation">Validación</option><option value="test">Prueba</option></select></label><button className="diagnostic-button" disabled={locked || !evaluationDatasetId || !evaluationCount} onClick={() => startJob(`/diagnostics/models/${encodeURIComponent(modelId)}/evaluate`, { dataset_id: evaluationDatasetId, partition: evaluationPartition }, "evaluate")}><Play size={15} />Evaluar versión</button></div><p className="diagnostic-help">{number(evaluationCount)} ejemplos disponibles en {PARTITIONS[evaluationPartition]}. Prueba mide el resultado final y no sustituye la evaluación independiente necesaria para revisar una versión.</p><Evaluation value={selectedModel.validation} />{(selectedModel.evaluations || []).filter((evaluation) => evaluation.partition === "test").map((evaluation, index) => <Evaluation key={evaluation.evaluation_id || index} value={evaluation} />)}<div className="diagnostic-actions"><button className="diagnostic-button diagnostic-primary" disabled={locked || status?.active_model_id === modelId} onClick={activateModel}><Check size={16} />{selectedModel.validated && !selectedModel.exploratory ? "Activar versión evaluada" : "Activar versión experimental"}</button></div>{(!selectedModel.validated || selectedModel.exploratory) && <p className="diagnostic-help">La activación experimental permite probar esta versión en el artefacto. Sus diagnósticos deben interpretarse con las limitaciones de sus datos y evaluación.</p>}</>}</>}</section>}

    {view === "catalog" && <>
      <section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Patrones que puede proponer F3</h3><BookOpen size={19} /></div><p className="diagnostic-help">Los patrones describen cambios medibles. Un período estable puede ser válido: el diagnóstico considera la referencia histórica y la evidencia disponible, y se abstiene cuando no alcanza.</p><div className="diagnostic-catalog">{(status?.catalog || []).map((item) => <article key={item.fault_type}><h4>{item.label || diagnosticLabel(item.fault_type)}</h4><p>{item.description}</p><div className="diagnostic-tags">{(item.entity_kinds || []).map((kind) => <span key={kind}>{({ continuous: "Continua", binary: "Binaria", multistate: "Multivalórica", state: "De estado" })[kind] || kind}</span>)}</div></article>)}</div></section><section className="diagnostic-panel"><div className="diagnostic-panel-title"><h3>Referencias de funcionamiento habitual</h3><span className="diagnostic-badge">{status?.references?.length || 0} referencias</span></div><p className="diagnostic-help">El diagnóstico utiliza referencias anteriores al episodio. Selecciona períodos normales representativos en Preparación del investigador para construirlas.</p>{!status?.references?.length ? <p className="diagnostic-empty">Todavía no hay referencias guardadas.</p> : <div className="diagnostic-reference-list">{status.references.map((reference, index) => <details key={reference.reference_id || reference.profile_id || index}><summary>{reference.entity_id} · {time(reference.end || reference.end_time)}{reference.partition ? ` · ${PARTITIONS[reference.partition] || reference.partition}` : ""}{reference.exploratory ? " · Exploratoria" : ""}</summary><ValuesTable values={reference} /></details>)}</div>}</section>
    </>}
  </section>;
}
