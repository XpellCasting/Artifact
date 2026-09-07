import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, Check, Download, FileText, Loader2, Save, Search, Upload } from "lucide-react";
import { reviewUtcTimestamp as utcTimestamp, validReviewInterval, initialReviewRange, shiftReviewRange } from "./review-time.js";
import "./review-workspace.css";
import InjectionWorkspace from "./InjectionWorkspace.jsx";
import { mergeAnnotations } from "./review-annotations.js";
import { FAULT_TYPES, faultTypeForLabel } from "./diagnostic-data.js";

const PARTITIONS = { development: "Desarrollo", validation: "Validación", test: "Prueba" };
const LABELS = { normal: "Normal", environmental_anomaly: "Anomalía ambiental", sensor_fault: "Falla del sensor", uncertain: "Incierto" };
const EVIDENCE = { insufficient: "Sin evidencia suficiente", manual_review: "Inspección visual", maintenance_log: "Registro de mantenimiento", controlled_action: "Acción controlada documentada", reference_sensor: "Sensor de referencia", system_log: "Registro independiente del sistema", other: "Otra evidencia" };
const INDEPENDENT_EVIDENCE = new Set(["maintenance_log", "controlled_action", "reference_sensor", "system_log"]);
const NEW_LABEL = { label: "uncertain", fault_type: null, status: "draft", evidence_type: "insufficient", evidence: "", reviewer: "", notes: "" };

function dateInput(value) {
  const utc = utcTimestamp(value);
  return utc ? utc.replace(/(\.\d{3})\d+Z$/, "$1Z").replace(/Z$/, "") : "";
}

function timeLabel(value) {
  return value ? String(value).replace("T", " ").replace(/\+00:00$/, "Z") : "—";
}

function selectionId({ datasetId, entityId, contextIds, start, end, padding }) {
  return JSON.stringify([datasetId, entityId, [...contextIds].sort(), start, end, Number(padding)]);
}

async function request(base, path, options = {}) {
  const response = await fetch(`${base.replace(/\/$/, "")}${path}`, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === "string" ? data.detail : data.detail ? JSON.stringify(data.detail) : `Error HTTP ${response.status}`);
  }
  return response.json();
}

function PreciseDateInput({ label, value, onChange, disabled }) {
  const [preciseOpen, setPreciseOpen] = useState(false);
  return <div className="review-date-field">
    <label>{label} (UTC)
      <input type="datetime-local" step="0.001" value={dateInput(value)} disabled={disabled} onChange={(event) => onChange(utcTimestamp(event.target.value))} />
    </label>
    <small className="review-exact-time">{value || "Selecciona una fecha"}</small>
    <button className="review-text-button" type="button" onClick={() => setPreciseOpen(!preciseOpen)} aria-expanded={preciseOpen}>{preciseOpen ? "Ocultar precisión ISO" : "Editar precisión ISO"}</button>
    {preciseOpen && <label className="review-precise-label">{label} ISO 8601, con zona horaria
      <input type="text" value={value} placeholder="2026-01-01T12:00:00.123456Z" disabled={disabled} onChange={(event) => onChange(event.target.value)} spellCheck={false} />
    </label>}
  </div>;
}

function SeriesChart({ series, fragment, target }) {
  const width = 1000;
  const height = 184;
  const left = 100;
  const right = 24;
  const top = 25;
  const bottom = 36;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const from = Date.parse(fragment.view_start);
  const to = Date.parse(fragment.view_end);
  const span = Math.max(1, to - from);
  const x = (timestamp) => left + Math.max(0, Math.min(1, (Date.parse(timestamp) - from) / span)) * plotWidth;
  const continuous = series.kind === "continuous";
  const points = series.points || [];
  const states = series.states?.length ? series.states : [...new Set(points.filter((point) => point.is_valid !== false).map((point) => String(point.state)))];
  const numberOf = (point) => continuous ? (point.value == null ? NaN : Number(point.value)) : states.indexOf(String(point.state));
  const valid = (point) => point.is_valid !== false && Number.isFinite(numberOf(point)) && (continuous || numberOf(point) >= 0);
  const numbers = points.filter(valid).map(numberOf);
  let min = continuous ? Math.min(...numbers) : 0;
  let max = continuous ? Math.max(...numbers) : Math.max(1, states.length - 1);
  if (!numbers.length) { min = 0; max = 1; }
  if (min === max) { min -= 0.5; max += 0.5; }
  const margin = continuous ? (max - min) * 0.1 : 0.3;
  const low = min - margin;
  const high = max + margin;
  const y = (value) => top + (high - value) / (high - low) * plotHeight;
  let connected = false;
  let path = "";
  for (const point of points) {
    if (!valid(point)) { connected = false; continue; }
    const px = x(point.timestamp);
    const py = y(numberOf(point));
    path += connected ? (continuous ? ` L${px},${py}` : ` H${px} V${py}`) : ` M${px},${py}`;
    connected = true;
  }
  const axis = continuous ? [min, (min + max) / 2, max].map((value) => ({ value, text: Number(value.toPrecision(5)).toString() })) : states.map((text, value) => ({ value, text })).filter((_, index) => states.length <= 5 || index % Math.ceil((states.length - 1) / 4) === 0 || index === states.length - 1);
  return <article className={`review-chart ${target ? "review-chart-target" : ""}`}>
    <div className="review-chart-heading"><strong>{series.entity_id}</strong><div className="review-chart-meta"><span>{target ? "Entidad etiquetada" : "Solo contexto"}</span><span>{series.unit || (continuous ? "Continua" : "Estados")}</span><span>{(series.rows ?? points.length).toLocaleString("es-CL")} registros</span>{series.sampled && <span className="review-sampling">Vista muestreada · {points.length} puntos</span>}{series.invalid_rows > 0 && <span>{series.invalid_rows} valores no válidos</span>}</div></div>
    {!points.length ? <p className="review-empty">No hay lecturas de esta entidad en la ventana ampliada.</p> : <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Serie de ${series.entity_id}. Intervalo a etiquetar sombreado; fechas UTC. Pasa el puntero por cada punto para ver la lectura original.`}>
      <rect x={x(fragment.start)} y={top - 10} width={Math.max(1, x(fragment.end) - x(fragment.start))} height={plotHeight + 10} className="review-chart-selection" />
      {axis.map((tick) => <g key={tick.value}><line x1={left} y1={y(tick.value)} x2={width - right} y2={y(tick.value)} className="review-chart-grid" /><text x={left - 10} y={y(tick.value) + 4} textAnchor="end" className="review-chart-axis"><title>{tick.text}</title>{tick.text.length > 13 ? `${tick.text.slice(0, 12)}…` : tick.text}</text></g>)}
      <line x1={x(fragment.start)} x2={x(fragment.start)} y1={top - 10} y2={height - bottom} className="review-chart-boundary" />
      <line x1={x(fragment.end)} x2={x(fragment.end)} y1={top - 10} y2={height - bottom} className="review-chart-boundary" />
      <path d={path} className="review-chart-line" />
      {points.map((point, index) => valid(point) ? <circle key={index} cx={x(point.timestamp)} cy={y(numberOf(point))} r={3} className="review-chart-point"><title>{`${point.timestamp}\n${series.entity_id}\nValor original: ${String(point.state)}${series.unit ? ` ${series.unit}` : ""}`}</title></circle> : <g key={index}><circle cx={x(point.timestamp)} cy={height - bottom - 3} r={4} className="review-chart-invalid"><title>{`${point.timestamp}\nValor no válido: ${String(point.state)}`}</title></circle></g>)}
      {[0, 0.5, 1].map((position) => <text key={position} x={left + plotWidth * position} y={height - 9} textAnchor={position === 0 ? "start" : position === 1 ? "end" : "middle"} className="review-chart-axis">{new Date(from + span * position).toISOString().replace("T", " ").slice(0, 19)}</text>)}
    </svg>}
  </article>;
}

export default function ReviewWorkspace({ apiBaseUrl = "http://localhost:8000" }) {
  const [datasets, setDatasets] = useState([]);
  const [datasetsBusy, setDatasetsBusy] = useState(true);
  const [datasetId, setDatasetId] = useState("");
  const [partition, setPartition] = useState("development");
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [uploadProcessing, setUploadProcessing] = useState(false);
  const [entityId, setEntityId] = useState("");
  const [entitySearch, setEntitySearch] = useState("");
  const [contextSearch, setContextSearch] = useState("");
  const [contextIds, setContextIds] = useState([]);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [padding, setPadding] = useState(30);
  const [fragment, setFragment] = useState(null);
  const [fragmentBusy, setFragmentBusy] = useState(false);
  const [annotations, setAnnotations] = useState([]);
  const [lastSavedAnnotation, setLastSavedAnnotation] = useState(null);
  const [annotationsBusy, setAnnotationsBusy] = useState(false);
  const [annotationId, setAnnotationId] = useState(null);
  const [form, setForm] = useState({ ...NEW_LABEL });
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [openPending, setOpenPending] = useState(false);
  const fragmentRequest = useRef(null);
  const uploadRequest = useRef(null);
  const mounted = useRef(true);
  const selectedDataset = datasets.find((dataset) => dataset.dataset_id === datasetId);
  const entities = selectedDataset?.entities || [];
  const selection = { datasetId, entityId, contextIds, start, end, padding };
  const currentKey = selectionId(selection);
  const currentKeyRef = useRef(currentKey);
  currentKeyRef.current = currentKey;
  const datasetIdRef = useRef(datasetId);
  datasetIdRef.current = datasetId;
  const activeFragment = fragment?.selectionKey === currentKey ? fragment : null;
  const validInterval = validReviewInterval(start, end);
  const validPadding = padding !== "" && Number.isFinite(Number(padding)) && Number(padding) >= 0 && Number(padding) <= 10080;
  const canVerify = form.label !== "uncertain" && INDEPENDENT_EVIDENCE.has(form.evidence_type) && form.evidence.trim().length >= 12 && form.reviewer.trim().length >= 2;
  const uploadBusy = uploadProgress !== null;
  const targets = useMemo(() => entities.filter((entity) => entity.entity_id === entityId || entity.entity_id.toLowerCase().includes(entitySearch.toLowerCase())), [entities, entityId, entitySearch]);
  const contextOptions = useMemo(() => entities.filter((entity) => entity.entity_id !== entityId && entity.entity_id.toLowerCase().includes(contextSearch.toLowerCase())), [entities, entityId, contextSearch]);

  const loadDatasets = useCallback(async (signal) => {
    setDatasetsBusy(true);
    try {
      const result = await request(apiBaseUrl, "/review/datasets", { signal });
      if (signal?.aborted || !mounted.current) return;
      setDatasets(result.datasets || []);
      setDatasetId((previous) => previous || result.datasets?.[0]?.dataset_id || "");
    } catch (exception) {
      if (exception.name !== "AbortError" && mounted.current) setError(exception.message);
    } finally { if (!signal?.aborted && mounted.current) setDatasetsBusy(false); }
  }, [apiBaseUrl]);

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    loadDatasets(controller.signal);
    return () => { mounted.current = false; controller.abort(); fragmentRequest.current?.abort(); uploadRequest.current?.abort(); };
  }, [loadDatasets]);

  useEffect(() => {
    if (!selectedDataset) return;
    setEntityId(selectedDataset.entities?.[0]?.entity_id || "");
    setContextIds([]);
    setEntitySearch("");
    setContextSearch("");
    const initial = initialReviewRange(selectedDataset.start, selectedDataset.end);
    setStart(initial.start);
    setEnd(initial.end);
    setAnnotationId(null);
    setForm({ ...NEW_LABEL });
    setNotice("");
  }, [datasetId]); // Dataset selection deliberately resets the review workspace.

  useEffect(() => {
    const controller = new AbortController();
    setAnnotations([]);
    setLastSavedAnnotation(null);
    if (!datasetId) { setAnnotationsBusy(false); return () => controller.abort(); }
    setAnnotationsBusy(true);
    request(apiBaseUrl, `/review/annotations?dataset_id=${encodeURIComponent(datasetId)}`, { signal: controller.signal })
      .then((result) => { if (!controller.signal.aborted) setAnnotations((current) => mergeAnnotations(current, result.annotations || [], datasetId)); })
      .catch((exception) => { if (exception.name !== "AbortError") setError(exception.message); })
      .finally(() => { if (!controller.signal.aborted) setAnnotationsBusy(false); });
    return () => controller.abort();
  }, [apiBaseUrl, datasetId]);

  useEffect(() => {
    fragmentRequest.current?.abort();
    setFragmentBusy(false);
  }, [currentKey]);

  const loadFragment = useCallback(async () => {
    if (!datasetId || !entityId || !validInterval || !validPadding) return;
    fragmentRequest.current?.abort();
    const controller = new AbortController();
    fragmentRequest.current = controller;
    const requestedKey = currentKey;
    setFragmentBusy(true);
    setError("");
    const query = new URLSearchParams({ entity_ids: [entityId, ...contextIds].join(","), start: utcTimestamp(start), end: utcTimestamp(end), padding_minutes: String(padding), max_points: "600" });
    try {
      const result = await request(apiBaseUrl, `/review/datasets/${encodeURIComponent(datasetId)}/fragment?${query}`, { signal: controller.signal });
      if (!controller.signal.aborted && requestedKey === currentKeyRef.current && mounted.current) setFragment({ ...result, selectionKey: requestedKey });
    } catch (exception) {
      if (exception.name !== "AbortError" && requestedKey === currentKeyRef.current && mounted.current) setError(exception.message);
    } finally { if (!controller.signal.aborted && requestedKey === currentKeyRef.current && mounted.current) setFragmentBusy(false); }
  }, [apiBaseUrl, currentKey, datasetId, entityId, contextIds, start, end, padding, validInterval, validPadding]);

  useEffect(() => {
    if (openPending) { setOpenPending(false); loadFragment(); }
  }, [openPending, loadFragment]);

  function upload() {
    if (!uploadFile || uploadBusy) return;
    const xhr = new XMLHttpRequest();
    uploadRequest.current = xhr;
    setUploadProgress(0);
    setUploadProcessing(false);
    setError("");
    setNotice("");
    xhr.open("POST", `${apiBaseUrl.replace(/\/$/, "")}/review/datasets/upload?partition=${partition}`);
    xhr.setRequestHeader("Content-Type", "text/csv");
    xhr.setRequestHeader("X-Filename", encodeURIComponent(uploadFile.name));
    xhr.upload.onprogress = (event) => { if (mounted.current && event.lengthComputable) setUploadProgress(Math.round(event.loaded / event.total * 100)); };
    xhr.upload.onload = () => { if (mounted.current) { setUploadProgress(100); setUploadProcessing(true); } };
    xhr.onload = () => {
      if (!mounted.current) return;
      let data;
      try { data = JSON.parse(xhr.responseText); } catch { data = {}; }
      if (xhr.status >= 200 && xhr.status < 300) {
        const dataset = data.dataset || data;
        setDatasets((previous) => [dataset, ...previous.filter((item) => item.dataset_id !== dataset.dataset_id)]);
        setDatasetId(dataset.dataset_id);
        setNotice("Histórico importado para revisión. Selecciona una entidad y un intervalo.");
      } else setError(typeof data.detail === "string" ? data.detail : `No se pudo importar el CSV (HTTP ${xhr.status}).`);
      setUploadProgress(null);
      setUploadProcessing(false);
    };
    xhr.onerror = () => { if (mounted.current) { setError("No se pudo conectar con el servidor para importar el CSV."); setUploadProgress(null); setUploadProcessing(false); } };
    xhr.onabort = () => { if (mounted.current) { setUploadProgress(null); setUploadProcessing(false); } };
    xhr.send(uploadFile);
  }

  function moveWindow(direction) {
    if (!validInterval) return;
    const next = shiftReviewRange(start, end, selectedDataset?.start, selectedDataset?.end, direction);
    if (!next) { setNotice("Llegaste al límite temporal de este histórico o el intervalo no permite avanzar."); return; }
    setStart(next.start);
    setEnd(next.end);
    setAnnotationId(null);
    setForm((previous) => ({ ...NEW_LABEL, reviewer: previous.reviewer }));
    setNotice("");
    setOpenPending(true);
  }

  function setFormField(field, value) {
    setForm((previous) => {
      const next = { ...previous, [field]: value };
      next.fault_type = faultTypeForLabel(next.label, next.fault_type);
      if (next.label === "uncertain" || !INDEPENDENT_EVIDENCE.has(next.evidence_type) || next.evidence.trim().length < 12 || next.reviewer.trim().length < 2) next.status = "draft";
      return next;
    });
  }

  async function saveAnnotation() {
    if (!activeFragment || saving || (form.status === "verified" && !canVerify)) return;
    setSaving(true);
    setError("");
    setNotice("");
    const savedKey = currentKey;
    const payload = { dataset_id: datasetId, entity_id: entityId, context_entity_ids: contextIds, start: utcTimestamp(start), end: utcTimestamp(end), padding_minutes: Number(padding), ...form };
    try {
      const result = await request(apiBaseUrl, annotationId ? `/review/annotations/${encodeURIComponent(annotationId)}` : "/review/annotations", { method: annotationId ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (!mounted.current) return;
      const saved = result.annotation || result;
      const id = saved.annotation_id || saved.id;
      if (datasetIdRef.current === payload.dataset_id) {
        setAnnotations((previous) => mergeAnnotations(previous, [saved], payload.dataset_id));
        setLastSavedAnnotation(saved);
      }
      if (currentKeyRef.current === savedKey) { setAnnotationId(id); setNotice("Anotación guardada y seleccionada en el paso 5. Allí se indica si falta completar algún requisito para generar inyecciones."); }
    } catch (exception) { if (mounted.current) setError(exception.message); }
    finally { if (mounted.current) setSaving(false); }
  }

  function openAnnotation(annotation) {
    setEntityId(annotation.entity_id);
    setContextIds(annotation.context_entity_ids || []);
    setStart(utcTimestamp(annotation.start));
    setEnd(utcTimestamp(annotation.end));
    setPadding(annotation.padding_minutes ?? 30);
    setAnnotationId(annotation.annotation_id || annotation.id);
    setForm(Object.fromEntries(Object.keys(NEW_LABEL).map((key) => [key, annotation[key] ?? NEW_LABEL[key]])));
    setNotice("Anotación abierta para revisión y edición.");
    setOpenPending(true);
  }

  async function exportAnnotation(id) {
    if (!id || exporting) return;
    setExporting(id);
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/review/annotations/${encodeURIComponent(id)}/export`);
      if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(typeof data.detail === "string" ? data.detail : `No se pudo exportar (HTTP ${response.status}).`); }
      const blob = await response.blob();
      if (!mounted.current) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `revision_${id}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      setNotice("ZIP descargado: datos originales del fragmento, etiquetas y manifiesto de trazabilidad.");
    } catch (exception) { if (mounted.current) setError(exception.message); }
    finally { if (mounted.current) setExporting(null); }
  }

  return <section className="review-workspace" aria-label="Revisión de históricos y base de verdad">
    <header className="review-header"><div><span className="review-eyebrow">Base de verdad · Revisión manual</span><h2>Del histórico a la evidencia</h2><p>Examina un intervalo, contrasta entidades y documenta qué puedes afirmar sobre la medición.</p></div><span className="review-badge"><FileText size={15} /> Etiquetas por episodio</span></header>
    <div className="review-notice review-method-note"><strong>La curva orienta la revisión; la evidencia permite confirmar la causa.</strong> Una inspección visual se guarda como borrador. Para verificar una etiqueta, documenta mantenimiento, una acción controlada, un sensor de referencia o un registro independiente que la respalde. Usa «Incierto» cuando falte evidencia.</div>
    {error && <div className="review-alert" role="alert">{error}</div>}
    {notice && <div className="review-notice" role="status">{notice}</div>}

    <section className="review-panel"><div className="review-panel-title"><h3>1. Histórico de revisión</h3>{datasetsBusy && <Loader2 size={18} className="review-spin" aria-label="Cargando históricos" />}</div>
      <div className="review-import-row"><label className="review-file-input">Archivo CSV<input type="file" accept=".csv,text/csv" disabled={uploadBusy || saving} onChange={(event) => setUploadFile(event.target.files?.[0] || null)} /></label><label>Partición<select value={partition} disabled={uploadBusy || saving} onChange={(event) => setPartition(event.target.value)}>{Object.entries(PARTITIONS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><button className="review-button review-button-primary" onClick={upload} disabled={!uploadFile || uploadBusy || saving}>{uploadBusy ? <Loader2 size={17} className="review-spin" /> : <Upload size={17} />}{uploadProcessing ? "Indexando histórico…" : uploadBusy ? `Subiendo ${uploadProgress}%` : "Importar para revisión"}</button></div>
      {uploadBusy && <div className="review-upload-progress" role="status"><progress value={uploadProcessing ? undefined : uploadProgress} max="100" aria-label={uploadProcessing ? "Procesando CSV" : "Progreso de carga"} /><span>{uploadProcessing ? "Preparando el índice de entidades e intervalos. Puede tardar con históricos grandes." : "Cargando copia del CSV…"}</span></div>}
      <label className="review-dataset-select">Históricos guardados<select value={datasetId} disabled={datasetsBusy || saving || uploadBusy} onChange={(event) => setDatasetId(event.target.value)}><option value="" disabled>{datasets.length ? "Selecciona un histórico" : "Importa tu primer histórico"}</option>{datasets.map((dataset) => <option key={dataset.dataset_id} value={dataset.dataset_id}>{dataset.filename} · {PARTITIONS[dataset.partition] || dataset.partition} · {Number(dataset.indexed_rows ?? dataset.rows).toLocaleString("es-CL")} registros</option>)}</select></label>
      {selectedDataset && <div className="review-dataset-details"><span>{PARTITIONS[selectedDataset.partition] || selectedDataset.partition}</span><span>{entities.length} entidades</span><span>{selectedDataset.rejected_rows || 0} filas rechazadas</span><span>{timeLabel(selectedDataset.start)} → {timeLabel(selectedDataset.end)}</span><details><summary>Identidad del archivo</summary><code>SHA-256: {selectedDataset.sha256}</code>{selectedDataset.warnings?.length > 0 && <ul>{selectedDataset.warnings.map((warning, index) => <li key={index}>{typeof warning === "string" ? warning : JSON.stringify(warning)}</li>)}</ul>}</details></div>}
      <p className="review-help">Importa cada partición por separado. La revisión conserva el archivo fuente y sus etiquetas; no modifica la base de normalidad.</p>
    </section>

    {selectedDataset && <>
      <section className="review-panel"><div className="review-panel-title"><h3>2. Entidad e intervalo</h3><span className="review-badge">Todas las fechas en UTC</span></div>
        <fieldset className="review-fields" disabled={saving}><div className="review-field"><label>Buscar entidad<div className="review-search"><Search size={16} /><input value={entitySearch} onChange={(event) => setEntitySearch(event.target.value)} placeholder="Nombre o identificador" /></div></label><label>Entidad que recibirá la etiqueta<select value={entityId} onChange={(event) => { setEntityId(event.target.value); setContextIds((previous) => previous.filter((id) => id !== event.target.value)); setAnnotationId(null); setForm((previous) => ({ ...NEW_LABEL, reviewer: previous.reviewer })); }}><option value="" disabled>Selecciona una entidad</option>{targets.map((entity) => <option key={entity.entity_id} value={entity.entity_id}>{entity.entity_id}{entity.unit ? ` (${entity.unit})` : ""}</option>)}</select></label></div><PreciseDateInput label="Inicio" value={start} onChange={setStart} disabled={saving} /><PreciseDateInput label="Fin" value={end} onChange={setEnd} disabled={saving} /><label>Contexto antes y después (min)<input type="number" min="0" max="10080" step="1" value={padding} onChange={(event) => setPadding(event.target.value)} /></label></fieldset>
        {start && end && !validInterval && <p className="review-inline-error">Indica fechas válidas y un fin posterior al inicio.</p>}
        {!validPadding && <p className="review-inline-error">El contexto debe estar entre 0 y 10 080 minutos.</p>}
        <details className="review-context-picker"><summary>Entidades de contexto · {contextIds.length}/7 seleccionadas</summary><p className="review-help">Elige entidades que ayuden a contrastar lo ocurrido. Solo la entidad principal recibe la etiqueta.</p><label>Buscar entidad de contexto<input value={contextSearch} onChange={(event) => setContextSearch(event.target.value)} placeholder="Ej. temperatura, puerta o disponibilidad" /></label><div className="review-context-list">{contextOptions.map((entity) => <label key={entity.entity_id}><input type="checkbox" checked={contextIds.includes(entity.entity_id)} disabled={saving || (!contextIds.includes(entity.entity_id) && contextIds.length >= 7)} onChange={(event) => setContextIds((previous) => event.target.checked ? [...previous, entity.entity_id] : previous.filter((id) => id !== entity.entity_id))} /><span>{entity.entity_id}</span></label>)}{!contextOptions.length && <p className="review-help">No hay entidades que coincidan.</p>}</div></details>
        {contextIds.length > 0 && <div className="review-context-tags">{contextIds.map((id) => <span key={id}>{id}<button disabled={saving} aria-label={`Quitar ${id} del contexto`} onClick={() => setContextIds((previous) => previous.filter((item) => item !== id))}>×</button></span>)}</div>}
        <div className="review-actions"><button className="review-button review-button-primary" disabled={!entityId || !validInterval || !validPadding || fragmentBusy || saving} onClick={loadFragment}>{fragmentBusy ? <Loader2 size={17} className="review-spin" /> : <Search size={17} />}{fragmentBusy ? "Cargando fragmento…" : "Ver fragmento"}</button><button className="review-button" disabled={!validInterval || fragmentBusy || saving} onClick={() => moveWindow(-1)}><ArrowLeft size={16} /> Ventana anterior</button><button className="review-button" disabled={!validInterval || fragmentBusy || saving} onClick={() => moveWindow(1)}>Ventana siguiente <ArrowRight size={16} /></button></div>
      </section>

      <section className="review-panel"><div className="review-panel-title"><h3>3. Inspección del fragmento</h3><span className="review-chart-legend"><i /> Intervalo de la etiqueta</span></div>
        {fragmentBusy ? <div className="review-empty"><Loader2 size={23} className="review-spin" /><p>Cargando lecturas y contexto…</p></div> : !activeFragment ? <div className="review-empty"><Search size={25} /><p>Selecciona «Ver fragmento» para cargar el intervalo actual.</p><span>Si cambias la selección, vuelve a cargarla antes de guardar una etiqueta.</span></div> : <>
          <p className="review-help">Ejes temporales alineados en UTC. El sombreado delimita la etiqueta; el resto corresponde al contexto. Pasa el puntero por los puntos para consultar los valores originales.</p>
          <div className="review-charts">{activeFragment.series?.map((series) => <SeriesChart key={series.entity_id} series={series} fragment={activeFragment} target={series.entity_id === entityId} />)}</div>
          <p className="review-help">La ausencia de lecturas no demuestra una falla. Las líneas conectan observaciones y no constituyen mediciones adicionales. Los gráficos pueden muestrearse; la exportación conserva los registros a su resolución original.</p>
          <details className="review-raw"><summary>Lecturas originales · muestra de {activeFragment.rows?.length || 0} filas{activeFragment.preview_truncated ? " (limitada)" : ""}</summary><div className="review-table-scroll"><table><thead><tr><th>Fila fuente</th><th>Entidad</th><th>Marca temporal original</th><th>Estado original</th></tr></thead><tbody>{activeFragment.rows?.map((row, index) => <tr key={`${row.source_row}-${index}`}><td>{row.source_row}</td><td>{row.entity_id}</td><td><code>{row.timestamp}</code></td><td>{String(row.state ?? "")}</td></tr>)}</tbody></table>{!activeFragment.rows?.length && <p className="review-empty">No hay filas en esta ventana ampliada.</p>}</div></details>
        </>}
      </section>

      <section className="review-panel"><div className="review-panel-title"><h3>4. {annotationId ? "Editar anotación" : "Documentar y etiquetar"}</h3>{annotationId && <button className="review-text-button" disabled={saving} onClick={() => { setAnnotationId(null); setForm((previous) => ({ ...NEW_LABEL, reviewer: previous.reviewer })); setNotice("Nueva anotación para el intervalo seleccionado."); }}>Crear nueva anotación</button>}</div><p className="review-help">La etiqueta describe únicamente <strong>{entityId || "la entidad principal"}</strong> entre el inicio y el fin seleccionados. El contexto se exporta como evidencia.</p>
        <fieldset className="review-annotation-form" disabled={saving}><label>Clase esperada<select value={form.label} onChange={(event) => setFormField("label", event.target.value)}>{Object.entries(LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>{form.label === "sensor_fault" && <label>Tipo de falla<select value={form.fault_type || ""} onChange={(event) => setFormField("fault_type", event.target.value || null)}><option value="">Tipo indeterminado</option>{Object.entries(FAULT_TYPES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select><small>Selecciona un patrón respaldado por la evidencia. Las fallas sin tipo no entrenan el clasificador.</small></label>}<label>Tipo de evidencia<select value={form.evidence_type} onChange={(event) => setFormField("evidence_type", event.target.value)}>{Object.entries(EVIDENCE).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><label>Responsable de la revisión<input value={form.reviewer} onChange={(event) => setFormField("reviewer", event.target.value)} placeholder="Nombre o identificador del revisor" /></label><label>Estado de revisión<select value={form.status} onChange={(event) => setFormField("status", event.target.value)}><option value="draft">Borrador</option><option value="verified" disabled={!canVerify}>Verificado con evidencia</option></select></label><label className="review-wide">Evidencia y referencia verificable<textarea rows="3" value={form.evidence} onChange={(event) => setFormField("evidence", event.target.value)} placeholder="Qué ocurrió, cómo se comprobó, fecha del registro y dónde puede consultarse. Para ‘Normal’, explica qué respalda esa interpretación." /></label><label className="review-wide">Notas y límites de la interpretación<textarea rows="2" value={form.notes} onChange={(event) => setFormField("notes", event.target.value)} placeholder="Observaciones, dudas, cobertura del contexto o hipótesis que no pudiste confirmar." /></label></fieldset>
        <p className="review-help">«Incierto» permanece en borrador. Para habilitar «Verificado», selecciona evidencia independiente y completa su descripción y responsable. Verificar es una declaración del revisor, no una conclusión automática del sistema.</p>
        {!activeFragment && <p className="review-inline-error">Carga el fragmento actual para habilitar el guardado.</p>}
        <div className="review-actions"><button className="review-button review-button-primary" disabled={!activeFragment || fragmentBusy || saving || (form.status === "verified" && !canVerify)} onClick={saveAnnotation}>{saving ? <Loader2 size={17} className="review-spin" /> : <Save size={17} />}{saving ? "Guardando…" : annotationId ? "Guardar cambios" : "Guardar anotación"}</button>{annotationId && <button className="review-button" disabled={Boolean(exporting) || saving} onClick={() => exportAnnotation(annotationId)}>{exporting === annotationId ? <Loader2 size={17} className="review-spin" /> : <Download size={17} />}Descargar versión guardada</button>}</div>
      </section>

      <section className="review-panel"><div className="review-panel-title"><h3>Anotaciones del histórico</h3><span className="review-badge">{annotations.length} guardadas</span></div>{annotationsBusy ? <p className="review-empty">Cargando anotaciones…</p> : !annotations.length ? <p className="review-empty">Aún no hay anotaciones para este histórico.</p> : <div className="review-annotation-list">{annotations.map((annotation) => { const id = annotation.annotation_id || annotation.id; return <article key={id} className={`review-annotation-item ${annotationId === id ? "review-annotation-active" : ""}`}><div><div className="review-annotation-heading"><strong>{LABELS[annotation.label] || annotation.label}{annotation.label === "sensor_fault" ? ` · ${FAULT_TYPES[annotation.fault_type] || "Tipo indeterminado"}` : ""}</strong><span className={`review-status ${annotation.status === "verified" ? "review-verified" : ""}`}>{annotation.status === "verified" && <Check size={13} />}{annotation.status === "verified" ? "Verificado" : "Borrador"}</span></div><p>{annotation.entity_id}</p><small>{timeLabel(annotation.start)} → {timeLabel(annotation.end)}</small><small>{EVIDENCE[annotation.evidence_type] || annotation.evidence_type}{annotation.reviewer ? ` · ${annotation.reviewer}` : ""}{annotation.revision ? ` · revisión ${annotation.revision}` : ""}</small></div><div className="review-actions"><button className="review-button" disabled={saving} onClick={() => openAnnotation(annotation)}>Abrir y editar</button><button className="review-button" disabled={Boolean(exporting) || saving} aria-label={`Descargar ZIP de ${annotation.entity_id}, ${timeLabel(annotation.start)}`} onClick={() => exportAnnotation(id)}>{exporting === id ? <Loader2 size={16} className="review-spin" /> : <Download size={16} />} ZIP</button></div></article>; })}</div>}<p className="review-help">Cada ZIP reúne el fragmento original, etiquetas separadas y un manifiesto con la identidad del archivo. Las anotaciones de borrador o inciertas deben permanecer fuera de las métricas que exijan una clase confirmada.</p></section>
      <InjectionWorkspace key={selectedDataset.dataset_id} apiBaseUrl={apiBaseUrl} dataset={selectedDataset} annotations={annotations} lastSavedAnnotation={lastSavedAnnotation} />
    </>}
  </section>;
}
