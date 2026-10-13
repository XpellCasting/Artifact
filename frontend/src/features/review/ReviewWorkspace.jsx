import { request } from "../../shared/api.js";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, Check, Download, FileText, Loader2, Save, Search, Upload } from "lucide-react";
import { reviewUtcTimestamp as utcTimestamp, validReviewInterval, initialReviewRange, shiftReviewRange } from "./review-time.js";
import "./review-workspace.css";
import InjectionWorkspace from "../injections/InjectionWorkspace.jsx";
import HistoryExplorer from "../history/HistoryExplorer.jsx";
import PointLabeler from "./PointLabeler.jsx";
import ZoomableReviewChart from "./ZoomableReviewChart.jsx";
import { reviewMicros, formatReviewMicros } from "./review-time.js";
import { annotationForPoint, POINT_FAULTS, pointFaultKinds } from "./point-labels.js";
import { reviewDayRange } from "../history/exploration.js";
import { mergeAnnotations, reviewStatus, reviewStatusLabel, splitReviewAnnotations } from "./review-annotations.js";
const FAULT_TYPES = POINT_FAULTS;
const faultTypeForLabel = (label, kind) => label === "sensor_fault" && POINT_FAULTS[kind] ? kind : null;

const PARTITIONS = { development: "Desarrollo", validation: "Validación", test: "Prueba" };
const LABELS = { normal: "Normal", environmental_anomaly: "Anomalía", sensor_fault: "Falla del sensor", uncertain: "Incierto" };
const NEW_LABEL = { source_row: null, label: "uncertain", fault_type: null, status: "pending", evidence_type: "manual_review", evidence: "", reviewer: "", notes: "" };

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

export function SeriesChart(props) {
  if (props.exploration) return <SeriesPlot {...props} />;
  return <ZoomableReviewChart key={`${props.series.entity_id}:${props.fragment.view_start}:${props.fragment.view_end}`} {...props} Plot={SeriesPlot} />;
}

export function SeriesPlot({ series, fragment, target, exploration = false, onPoint, annotations = [], viewport, clipId = "review-static-plot" }) {
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const width = 1000;
  const height = 184;
  const left = 100;
  const right = 24;
  const top = 25;
  const bottom = 36;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const origin = reviewMicros(fragment.view_start);
  const total = Math.max(1, Number(reviewMicros(fragment.view_end) - origin));
  const [from, to] = viewport || [0, total];
  const span = Math.max(1, to - from);
  const offset = timestamp => Number(reviewMicros(timestamp) - origin);
  const x = timestamp => left + (offset(timestamp) - from) / span * plotWidth;
  const visible = point => offset(point.timestamp) >= from && offset(point.timestamp) <= to;
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
    <div className="review-chart-heading"><strong>{series.entity_id}</strong><div className="review-chart-meta"><span>{exploration ? "Exploración sin etiqueta" : target ? "Entidad principal" : "Entidad de contexto"}</span><span>{series.unit || (continuous ? "Continua" : "Estados")}</span><span>{(series.rows ?? points.length).toLocaleString("es-CL")} registros</span>{series.sampled && <span className="review-sampling">Vista muestreada · {points.length} puntos</span>}{series.invalid_rows > 0 && <span>{series.invalid_rows} valores no válidos</span>}</div></div>
    {!points.length ? <p className="review-empty">No hay lecturas de esta entidad en la ventana ampliada.</p> : <svg viewBox={`0 0 ${width} ${height}`} role={onPoint ? "group" : "img"} aria-label={`Serie de ${series.entity_id}. Intervalo a etiquetar sombreado; fechas UTC. Pasa el puntero por cada punto para ver la lectura original.`}>
      <defs><clipPath id={clipId}><rect x={left} y={top - 10} width={plotWidth} height={plotHeight + 16} /></clipPath></defs>
      <g clipPath={`url(#${clipId})`}><rect x={x(fragment.start)} y={top - 10} width={Math.max(1, x(fragment.end) - x(fragment.start))} height={plotHeight + 10} className="review-chart-selection" /></g>
      {axis.map((tick) => <g key={tick.value}><line x1={left} y1={y(tick.value)} x2={width - right} y2={y(tick.value)} className="review-chart-grid" /><text x={left - 10} y={y(tick.value) + 4} textAnchor="end" className="review-chart-axis"><title>{tick.text}</title>{tick.text.length > 13 ? `${tick.text.slice(0, 12)}…` : tick.text}</text></g>)}
      <g clipPath={`url(#${clipId})`}>
      <line x1={x(fragment.start)} x2={x(fragment.start)} y1={top - 10} y2={height - bottom} className="review-chart-boundary" />
      <line x1={x(fragment.end)} x2={x(fragment.end)} y1={top - 10} y2={height - bottom} className="review-chart-boundary" />
      <path d={path} className="review-chart-line" />
      {points.filter(visible).map((point, index) => {
        const annotation = annotationForPoint(annotations, series.entity_id, point);
        const label = annotation?.evaluable ? annotation.label : "unlabelled";
        const px = x(point.timestamp);
        const py = valid(point) ? y(numberOf(point)) : height - bottom - 3;
        const showPointText = Boolean(viewport && (to - from) < total * 0.2 && points.length <= 250);
        const pointTime = utcTimestamp(point.timestamp);
        const timeText = pointTime.includes("T") ? pointTime.slice(11, 19) : pointTime;
        const pointValue = String(point.state);
        return <g key={point.source_row ?? index}>
          <circle cx={px} cy={py} r={onPoint ? 4 : 3}
          role={onPoint ? "button" : undefined} tabIndex={onPoint ? 0 : undefined}
          aria-label={onPoint ? `Etiquetar ${point.timestamp}, valor ${point.state}, fila ${point.source_row}` : undefined}
          onClick={() => onPoint?.(series, point)} onKeyDown={event => { if (onPoint && ["Enter", " "].includes(event.key)) { event.preventDefault(); onPoint(series, point); } }}
          onPointerEnter={() => setHoveredPoint({ point, annotation })}
          onPointerLeave={() => setHoveredPoint((current) => current?.point === point ? null : current)}
          onFocus={() => setHoveredPoint({ point, annotation })}
          onBlur={() => setHoveredPoint((current) => current?.point === point ? null : current)}
          className={`${valid(point) ? "review-chart-point" : "review-chart-invalid"} review-label-${label}`} />
          {showPointText && <g className="review-point-inline-labels" aria-hidden="true">
            <text x={px} y={py - 8} textAnchor="middle" className="review-point-value-label">{pointValue.length > 12 ? `${pointValue.slice(0, 11)}…` : pointValue}</text>
            <text x={px} y={py + 17} textAnchor="middle" className="review-point-time-label">{timeText}</text>
          </g>}
        </g>;
      })}
      </g>
      {[0, 0.5, 1].map((position) => <text key={position} x={left + plotWidth * position} y={height - 9} textAnchor={position === 0 ? "start" : position === 1 ? "end" : "middle"} className="review-chart-axis">{formatReviewMicros(origin + BigInt(Math.round(from + span * position))).replace("T", " ")}</text>)}
    </svg>}
    {hoveredPoint && <div className="review-point-tooltip" role="status" aria-live="polite">
      <strong>{utcTimestamp(hoveredPoint.point.timestamp)}</strong>
      <span>Valor: {String(hoveredPoint.point.state)}</span>
      <span>Etiqueta: {hoveredPoint.annotation ? LABELS[hoveredPoint.annotation.label] : "Sin revisar"}{hoveredPoint.annotation?.fault_type ? ` · ${POINT_FAULTS[hoveredPoint.annotation.fault_type] || hoveredPoint.annotation.fault_type}` : ""}</span>
      {hoveredPoint.point.source_row != null && <span>Fila: {hoveredPoint.point.source_row}</span>}
    </div>}
    {onPoint && points.length > 0 && <label className="review-point-picker">Elegir lectura si los puntos se superponen<select value="" onChange={event => { const point = points.find(item => String(item.source_row) === event.target.value); if (point) onPoint(series, point); }}><option value="">Selecciona una lectura de la gráfica</option>{points.filter(visible).map(point => <option key={point.source_row} value={point.source_row}>{utcTimestamp(point.timestamp)} · {point.state} · fila {point.source_row}</option>)}</select></label>}
  </article>;
}

export default function ReviewWorkspace({ apiBaseUrl = "http://localhost:8000" }) {
  const [pointSelection, setPointSelection] = useState(null);
  const [pointError, setPointError] = useState("");
  const [annotationSaveError, setAnnotationSaveError] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [datasetsBusy, setDatasetsBusy] = useState(true);
  const [datasetId, setDatasetId] = useState("");
  const [partition, setPartition] = useState("development");
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [uploadProcessing, setUploadProcessing] = useState(false);
  const [entityId, setEntityId] = useState("");
  const [completedEntities, setCompletedEntities] = useState({});
  const [completedEntitiesDataset, setCompletedEntitiesDataset] = useState("");
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
  const [annotationsLoadedFor, setAnnotationsLoadedFor] = useState("");
  const { points: pointLabels, intervals: intervalAnnotations } = useMemo(() => splitReviewAnnotations(annotations), [annotations]);
  const labelsReady = annotationsLoadedFor === datasetId && !annotationsBusy;
  const [annotationId, setAnnotationId] = useState(null);
  const [form, setForm] = useState({ ...NEW_LABEL });
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [openPending, setOpenPending] = useState(false);
  const fragmentRequest = useRef(null);
  const intervalPanel = useRef(null);
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
  const canReview = form.evidence.trim().length >= 12 && form.reviewer.trim().length >= 2;
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
    try {
      const saved = window.localStorage.getItem(`review-completed-entities:${datasetId}`);
      setCompletedEntities(saved ? JSON.parse(saved) : {});
    } catch { setCompletedEntities({}); }
    setCompletedEntitiesDataset(datasetId);
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
    if (!datasetId || completedEntitiesDataset !== datasetId) return;
    try { window.localStorage.setItem(`review-completed-entities:${datasetId}`, JSON.stringify(completedEntities)); }
    catch { /* Completion marks remain usable for this session if storage is unavailable. */ }
  }, [datasetId, completedEntities, completedEntitiesDataset]);

  useEffect(() => {
    const controller = new AbortController();
    setAnnotations([]);
    setAnnotationsLoadedFor("");
    setLastSavedAnnotation(null);
    if (!datasetId) { setAnnotationsBusy(false); return () => controller.abort(); }
    setAnnotationsBusy(true);
    request(apiBaseUrl, `/review/annotations?dataset_id=${encodeURIComponent(datasetId)}`, { signal: controller.signal })
      .then((result) => { if (!controller.signal.aborted) { setAnnotations((current) => mergeAnnotations(current, result.annotations || [], datasetId)); setAnnotationsLoadedFor(datasetId); } })
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
      if (next.evidence.trim().length < 12 || next.reviewer.trim().length < 2) next.status = "pending";
      return next;
    });
  }

  function moveReviewDay(direction) {
    const current = utcTimestamp(start);
    if (!current || !selectedDataset) return;
    const date = new Date(current.slice(0, 10) + "T00:00:00Z");
    date.setUTCDate(date.getUTCDate() + direction);
    const next = reviewDayRange(date.toISOString().slice(0, 10), selectedDataset.start, selectedDataset.end);
    if (!next) return;
    setStart(next.start); setEnd(next.end); setAnnotationId(null); setPointSelection(null); setPointError("");
    setForm(previous => ({ ...NEW_LABEL, reviewer: previous.reviewer })); setOpenPending(true);
  }

  async function savePoint(label, faultType = null) {
    if (!pointSelection || pointSelection.key !== currentKey || saving || !labelsReady) return;
    const { series, point } = pointSelection;
    const existing = annotations.find(a => a.entity_id === series.entity_id && a.source_row === point.source_row);
    const timestamp = utcTimestamp(point.timestamp);
    const payload = { dataset_id: datasetId, entity_id: series.entity_id, source_row: point.source_row,
      start: timestamp, end: timestamp, padding_minutes: 0, context_entity_ids: [], label, fault_type: faultType,
      status: "reviewed", evidence_type: "manual_review", reviewer: form.reviewer.trim() || "Revisor local",
      evidence: "Etiqueta asignada manualmente al registro desde la gráfica.", notes: existing?.notes || "" };
    setSaving(true); setPointError("");
    try {
      const result = await request(apiBaseUrl, existing ? `/review/annotations/${existing.annotation_id}` : "/review/annotations",
        { method: existing ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (mounted.current && datasetIdRef.current === payload.dataset_id) {
        setAnnotations(previous => mergeAnnotations(previous, [result.annotation || result], payload.dataset_id));
        setNotice("Marca guardada en SQLite. Se recuperará al volver a abrir este histórico. Puedes continuar revisando.");
      }
    } catch (error) { if (mounted.current) setPointError(error.message); }
    finally { if (mounted.current) setSaving(false); }
  }

  async function saveAnnotation() {
    if (!activeFragment || saving || (form.status === "reviewed" && !canReview)) return;
    setSaving(true);
    setAnnotationSaveError(null);
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
    } catch (exception) { if (mounted.current) { setError(exception.message); setAnnotationSaveError({ key: savedKey, message: exception.message }); } }
    finally { if (mounted.current) setSaving(false); }
  }

  function openAnnotation(annotation) {
    setEntityId(annotation.entity_id);
    setContextIds(annotation.context_entity_ids || []);
    setStart(utcTimestamp(annotation.start));
    setEnd(utcTimestamp(annotation.end));
    setPadding(annotation.padding_minutes ?? 30);
    setAnnotationId(annotation.annotation_id || annotation.id);
    setForm({ ...Object.fromEntries(Object.keys(NEW_LABEL).map((key) => [key, annotation[key] ?? NEW_LABEL[key]])), status: reviewStatus(annotation) });
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
    <header className="review-header"><div><span className="review-eyebrow">Base de verdad · Revisión manual</span><h2>Construir la base de verdad</h2><p>Revisa cada entidad día por día, marca sus lecturas y descarga el histórico etiquetado.</p></div><span className="review-badge"><FileText size={15} /> Etiquetas por registro</span></header>
    <div className="review-notice review-method-note"><strong>Selecciona un día y pulsa una lectura para marcarla.</strong> Elige Normal, Anomalía o Falla con su tipo. La marca se guarda al pulsar la opción y puedes continuar revisando el mismo día. Los registros sin revisar mantienen sus etiquetas en NULL. Las marcas documentan tu interpretación del histórico.</div>
    {error && <div className="review-alert" role="alert">{error}</div>}
    {notice && <div className="review-notice" role="status">{notice}</div>}

    <section className="review-panel"><div className="review-panel-title"><h3>1. Histórico de revisión</h3>{datasetsBusy && <Loader2 size={18} className="review-spin" aria-label="Cargando históricos" />}</div>
      <div className="review-import-row"><label className="review-file-input">Histórico o ground_truth.csv etiquetado<input type="file" accept=".csv,text/csv" disabled={uploadBusy || saving} onChange={(event) => setUploadFile(event.target.files?.[0] || null)} /></label><label>Partición<select value={partition} disabled={uploadBusy || saving} onChange={(event) => setPartition(event.target.value)}>{Object.entries(PARTITIONS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><button className="review-button review-button-primary" onClick={upload} disabled={!uploadFile || uploadBusy || saving}>{uploadBusy ? <Loader2 size={17} className="review-spin" /> : <Upload size={17} />}{uploadProcessing ? "Indexando histórico…" : uploadBusy ? `Subiendo ${uploadProgress}%` : "Importar para revisión"}</button></div>
      {uploadBusy && <div className="review-upload-progress" role="status"><progress value={uploadProcessing ? undefined : uploadProgress} max="100" aria-label={uploadProcessing ? "Procesando CSV" : "Progreso de carga"} /><span>{uploadProcessing ? "Preparando el índice de entidades e intervalos. Puede tardar con históricos grandes." : "Cargando copia del CSV…"}</span></div>}
      <label className="review-dataset-select">Históricos guardados<select value={datasetId} disabled={datasetsBusy || saving || uploadBusy} onChange={(event) => setDatasetId(event.target.value)}><option value="" disabled>{datasets.length ? "Selecciona un histórico" : "Importa tu primer histórico"}</option>{datasets.map((dataset) => <option key={dataset.dataset_id} value={dataset.dataset_id}>{dataset.filename} · {PARTITIONS[dataset.partition] || dataset.partition} · {Number(dataset.indexed_rows ?? dataset.rows).toLocaleString("es-CL")} registros</option>)}</select></label>
      {selectedDataset && <div className="review-dataset-details"><span>{PARTITIONS[selectedDataset.partition] || selectedDataset.partition}</span><span>{entities.length} entidades</span><span>{selectedDataset.rejected_rows || 0} filas rechazadas</span><span>{timeLabel(selectedDataset.start)} → {timeLabel(selectedDataset.end)}</span><details><summary>Identidad del archivo</summary><code>SHA-256: {selectedDataset.sha256}</code>{selectedDataset.warnings?.length > 0 && <ul>{selectedDataset.warnings.map((warning, index) => <li key={index}>{typeof warning === "string" ? warning : JSON.stringify(warning)}</li>)}</ul>}</details></div>}
      <p className="review-help">Puedes importar entity_id,state,last_changed,es_sintetica,tipo,tipo_falla para continuar una base de verdad. Sus etiquetas se recuperan en la gráfica y se conservan al exportar. Para añadir una inyección, documenta una base normal de intervalo en el paso 4 y utiliza el paso 5. Importa cada partición por separado. La revisión conserva el archivo fuente y sus etiquetas; no modifica la base de normalidad.</p>
    </section>

    {selectedDataset && <>
      <HistoryExplorer key={`exploration-${datasetId}`} apiBaseUrl={apiBaseUrl} dataset={selectedDataset} Chart={SeriesChart} DateInput={PreciseDateInput} disabled={saving || uploadBusy} completedEntities={completedEntities} onToggleEntityComplete={(id, checked) => setCompletedEntities(previous => ({ ...previous, [id]: checked }))} onReview={(selection) => {
        if (selection.datasetId !== datasetIdRef.current) return;
        setEntityId(selection.entityId); setStart(selection.start); setEnd(selection.end);
        setContextIds(previous => previous.filter(id => id !== selection.entityId)); setAnnotationId(null);
        setForm(previous => ({ ...NEW_LABEL, reviewer: previous.reviewer }));
        setOpenPending(true);
        setNotice(selection.day ? `Día ${selection.day} seleccionado. ${selection.partialDay ? "Cobertura parcial del archivo." : "Jornada completa en UTC."} Revisa los datos antes de guardar una etiqueta.` : "Intervalo seleccionado para revisión.");
      }} />
      <section className="review-panel" ref={intervalPanel}><div className="review-panel-title"><h3>2. Revisión seleccionada</h3><span className="review-badge">Todas las fechas en UTC</span></div><p className="review-help"><strong>{entityId}</strong><br />{timeLabel(start)} → {timeLabel(end)}</p><p className="review-help">Selecciona un día en Actividad diaria para abrirlo directamente. Los ajustes siguientes permiten acotar un evento o añadir entidades de contexto.</p><details><summary>Ajustar entidad, intervalo y contexto</summary>
        <fieldset className="review-fields" disabled={saving}><div className="review-field"><label>Buscar entidad<div className="review-search"><Search size={16} /><input value={entitySearch} onChange={(event) => setEntitySearch(event.target.value)} placeholder="Nombre o identificador" /></div></label><label>Entidad que recibirá la etiqueta<select value={entityId} onChange={(event) => { setEntityId(event.target.value); setContextIds((previous) => previous.filter((id) => id !== event.target.value)); setAnnotationId(null); setForm((previous) => ({ ...NEW_LABEL, reviewer: previous.reviewer })); }}><option value="" disabled>Selecciona una entidad</option>{targets.map((entity) => <option key={entity.entity_id} value={entity.entity_id}>{completedEntities[entity.entity_id] ? "✓ Revisada · " : ""}{entity.entity_id}{entity.unit ? ` (${entity.unit})` : ""}</option>)}</select></label></div><PreciseDateInput label="Inicio" value={start} onChange={setStart} disabled={saving} /><PreciseDateInput label="Fin" value={end} onChange={setEnd} disabled={saving} /><label>Contexto antes y después (min)<input type="number" min="0" max="10080" step="1" value={padding} onChange={(event) => setPadding(event.target.value)} /></label></fieldset>
        {start && end && !validInterval && <p className="review-inline-error">Indica fechas válidas y un fin posterior al inicio.</p>}
        {!validPadding && <p className="review-inline-error">El contexto debe estar entre 0 y 10 080 minutos.</p>}
        <details className="review-context-picker"><summary>Entidades de contexto · {contextIds.length}/7 seleccionadas</summary><p className="review-help">Elige entidades que ayuden a contrastar lo ocurrido. Las anotaciones de intervalo afectan a la entidad principal. Una marca puntual afecta únicamente al registro que pulses.</p><label>Buscar entidad de contexto<input value={contextSearch} onChange={(event) => setContextSearch(event.target.value)} placeholder="Ej. temperatura, puerta o disponibilidad" /></label><div className="review-context-list">{contextOptions.map((entity) => <label key={entity.entity_id}><input type="checkbox" checked={contextIds.includes(entity.entity_id)} disabled={saving || (!contextIds.includes(entity.entity_id) && contextIds.length >= 7)} onChange={(event) => setContextIds((previous) => event.target.checked ? [...previous, entity.entity_id] : previous.filter((id) => id !== entity.entity_id))} /><span>{entity.entity_id}</span></label>)}{!contextOptions.length && <p className="review-help">No hay entidades que coincidan.</p>}</div></details>
        {contextIds.length > 0 && <div className="review-context-tags">{contextIds.map((id) => <span key={id}>{id}<button disabled={saving} aria-label={`Quitar ${id} del contexto`} onClick={() => setContextIds((previous) => previous.filter((item) => item !== id))}>×</button></span>)}</div>}
        <div className="review-actions"><button className="review-button review-button-primary" disabled={!entityId || !validInterval || !validPadding || fragmentBusy || saving} onClick={loadFragment}>{fragmentBusy ? <Loader2 size={17} className="review-spin" /> : <Search size={17} />}{fragmentBusy ? "Cargando fragmento…" : "Ver fragmento"}</button><button className="review-button" disabled={!validInterval || fragmentBusy || saving} onClick={() => moveWindow(-1)}><ArrowLeft size={16} /> Ventana anterior</button><button className="review-button" disabled={!validInterval || fragmentBusy || saving} onClick={() => moveWindow(1)}>Ventana siguiente <ArrowRight size={16} /></button></div>
      </details>
      </section>

      <section className="review-panel"><div className="review-panel-title"><h3>3. Revisar y marcar lecturas</h3><span className="review-chart-legend"><i /> Intervalo en revisión</span></div>
        {fragmentBusy ? <div className="review-empty"><Loader2 size={23} className="review-spin" /><p>Cargando lecturas y contexto…</p></div> : !activeFragment ? <div className="review-empty"><Search size={25} /><p>Selecciona «Ver fragmento» para cargar el intervalo actual.</p><span>Si cambias la selección, vuelve a cargarla antes de guardar una etiqueta.</span></div> : <>
          <p className="review-help">Ejes temporales alineados en UTC. El sombreado delimita el intervalo en revisión; cada marca puntual afecta solo a su lectura. Pasa el puntero por los puntos para consultar los valores originales.</p>
          <div className="review-actions"><button className="review-button" disabled={saving || fragmentBusy} onClick={() => moveReviewDay(-1)}>Día anterior</button><strong>{utcTimestamp(start).slice(0, 10)} · UTC</strong><button className="review-button" disabled={saving || fragmentBusy} onClick={() => moveReviewDay(1)}>Día siguiente</button></div>
          <p className="review-help" role="status">{annotationsBusy ? "Recuperando marcas guardadas en SQLite…" : labelsReady ? `${pointLabels.length} marcas nuevas y ${selectedDataset?.imported_labels || 0} etiquetas importadas guardadas en SQLite. Las marcas reaparecen al abrir este histórico, incluso después de cerrar la aplicación.` : "No se pudieron recuperar las marcas guardadas. Recarga la página antes de continuar etiquetando."}</p>
          <div className="review-charts">{activeFragment.series?.map((series) => <React.Fragment key={series.entity_id}><SeriesChart apiBaseUrl={apiBaseUrl} datasetId={datasetId} series={series} fragment={activeFragment} target={series.entity_id === entityId} annotations={annotations} onPoint={saving || !labelsReady ? undefined : (selectedSeries, point) => { setPointSelection({ series: selectedSeries, point, key: currentKey }); setPointError(""); }} />
            {pointSelection?.key === currentKey && pointSelection.series.entity_id === series.entity_id && <PointLabeler key={`${series.entity_id}:${pointSelection.point.source_row}`} selection={pointSelection} annotations={annotations} reviewer={form.reviewer} onReviewer={value => setFormField("reviewer", value)} onSave={savePoint} disabled={saving || !labelsReady} />}
          </React.Fragment>)}</div><p className="review-help">Pulsa un punto para etiquetar ese registro. Verde: normal. Ámbar: anomalía. Rojo: falla. Gris: sin etiqueta definitiva. Guardar una marca no ejecuta filtros ni clasificadores.</p>
          {pointError && <p className="review-alert" role="alert">{pointError}</p>}
          <p className="review-help">La ausencia de lecturas no demuestra una falla. Las líneas conectan observaciones y no constituyen mediciones adicionales. Los gráficos pueden muestrearse; la exportación conserva los registros a su resolución original.</p>
          <details className="review-raw"><summary>Lecturas originales · muestra de {activeFragment.rows?.length || 0} filas{activeFragment.preview_truncated ? " (limitada)" : ""}</summary><div className="review-table-scroll"><table><thead><tr><th>Fila fuente</th><th>Entidad</th><th>Marca temporal original</th><th>Estado original</th></tr></thead><tbody>{activeFragment.rows?.map((row, index) => <tr key={`${row.source_row}-${index}`}><td>{row.source_row}</td><td>{row.entity_id}</td><td><code>{row.timestamp}</code></td><td>{String(row.state ?? "")}</td></tr>)}</tbody></table>{!activeFragment.rows?.length && <p className="review-empty">No hay filas en esta ventana ampliada.</p>}</div></details>
        </>}
      </section>

      <details className="review-panel" open={annotationId ? true : undefined}><summary>4. Anotación de intervalo (opcional) y base para inyecciones</summary><div className="review-panel-title"><h3>{annotationId ? "Editar anotación" : "Documentar un intervalo"}</h3>{annotationId && <button className="review-text-button" disabled={saving} onClick={() => { setAnnotationId(null); setForm((previous) => ({ ...NEW_LABEL, reviewer: previous.reviewer })); setNotice("Nueva anotación para el intervalo seleccionado."); }}>Crear nueva anotación</button>}</div><p className="review-help">La etiqueta describe únicamente <strong>{entityId || "la entidad principal"}</strong> entre el inicio y el fin seleccionados. El contexto se exporta como evidencia.</p>
        <fieldset className="review-annotation-form" disabled={saving}><label>Clase asignada<select value={form.label} onChange={(event) => setFormField("label", event.target.value)}>{Object.entries(LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>{form.label === "sensor_fault" && <label>Tipo de falla<select value={form.fault_type || ""} onChange={(event) => setFormField("fault_type", event.target.value || null)}><option value="">Tipo indeterminado</option>{pointFaultKinds(entities.find(entity => entity.entity_id === entityId)?.kind).map(value => <option key={value} value={value}>{FAULT_TYPES[value]}</option>)}</select><small>El tipo de falla se incorporará a tipo_falla en el CSV. Si no puedes precisarlo, se conservará NULL.</small></label>}<label>Origen de la referencia<input value="Revisión del histórico" readOnly /><small>Se asigna automáticamente. Las inyecciones tendrán su propio origen.</small></label><label>Responsable de la revisión<input value={form.reviewer} onChange={(event) => setFormField("reviewer", event.target.value)} placeholder="Nombre o identificador del revisor" /></label><label>Estado de revisión<select value={form.status} onChange={(event) => setFormField("status", event.target.value)}><option value="pending">Pendiente</option><option value="reviewed" disabled={!canReview}>Revisado</option></select></label><label className="review-wide">Observaciones que respaldan la interpretación<textarea rows="3" value={form.evidence} onChange={(event) => setFormField("evidence", event.target.value)} placeholder="Describe los estados, cambios y contexto que observaste. Para Normal, explica por qué el fragmento parece compatible con el funcionamiento habitual. No necesitas registros externos." /></label><label className="review-wide">Notas y límites de la interpretación<textarea rows="2" value={form.notes} onChange={(event) => setFormField("notes", event.target.value)} placeholder="Observaciones, dudas, cobertura del contexto o hipótesis que no pudiste confirmar." /></label></fieldset>
        <p className="review-help">Completa responsable y observaciones para marcar Revisado. También puedes terminar una revisión como Incierto; sus etiquetas de fila se exportarán como NULL. Una referencia Normal revisada supone normalidad a partir del histórico y permite generar inyecciones conocidas.</p>
        {!activeFragment && <p className="review-inline-error">Carga el fragmento actual para habilitar el guardado.</p>}
        {annotationSaveError?.key === currentKey && <div className="review-alert" role="alert">{annotationSaveError.message}</div>}
        <div className="review-actions"><button className="review-button review-button-primary" disabled={!activeFragment || fragmentBusy || saving || (form.status === "reviewed" && !canReview)} onClick={saveAnnotation}>{saving ? <Loader2 size={17} className="review-spin" /> : <Save size={17} />}{saving ? "Guardando…" : annotationId ? "Guardar cambios" : "Guardar anotación"}</button>{annotationId && <button className="review-button" disabled={Boolean(exporting) || saving} onClick={() => exportAnnotation(annotationId)}>{exporting === annotationId ? <Loader2 size={17} className="review-spin" /> : <Download size={17} />}Descargar versión guardada</button>}</div>
      </details>

      <section className="review-panel"><div className="review-panel-title"><h3>Anotaciones del histórico</h3><span className="review-badge">{intervalAnnotations.length} guardadas</span></div>{annotationsBusy ? <p className="review-empty">Cargando anotaciones…</p> : !intervalAnnotations.length ? <p className="review-empty">No hay anotaciones de intervalo. Las marcas puntuales se conservan en SQLite y se muestran en las gráficas.</p> : <div className="review-annotation-list">{intervalAnnotations.map((annotation) => { const id = annotation.annotation_id || annotation.id; return <article key={id} className={`review-annotation-item ${annotationId === id ? "review-annotation-active" : ""}`}><div><div className="review-annotation-heading"><strong>{LABELS[annotation.label] || annotation.label}{annotation.label === "sensor_fault" ? ` · ${FAULT_TYPES[annotation.fault_type] || "Tipo indeterminado"}` : ""}</strong><span className={`review-status ${reviewStatus(annotation) === "reviewed" ? "review-verified" : ""}`}>{reviewStatus(annotation) === "reviewed" && <Check size={13} />}{reviewStatusLabel(annotation)}</span></div><p>{annotation.entity_id}</p><small>{timeLabel(annotation.start)} → {timeLabel(annotation.end)}</small><small>{"Revisión del histórico"}{annotation.externally_verified ? " · evidencia externa declarada en versión anterior" : ""}{annotation.reviewer ? ` · ${annotation.reviewer}` : ""}{annotation.revision ? ` · revisión ${annotation.revision}` : ""}</small></div><div className="review-actions"><button className="review-button" disabled={saving} onClick={() => openAnnotation(annotation)}>Abrir y editar</button><button className="review-button" disabled={Boolean(exporting) || saving} aria-label={`Descargar ZIP de ${annotation.entity_id}, ${timeLabel(annotation.start)}`} onClick={() => exportAnnotation(id)}>{exporting === id ? <Loader2 size={16} className="review-spin" /> : <Download size={16} />} ZIP</button></div></article>; })}</div>}<p className="review-help">Este listado contiene solo anotaciones de intervalo. Las marcas de puntos se guardan automáticamente en SQLite y se incluyen en el CSV completo. Cada ZIP reúne el fragmento original, etiquetas separadas y un manifiesto con la identidad del archivo. Las referencias revisadas se comparan como interpretaciones del histórico. Las inyecciones se evalúan por separado como intervenciones conocidas; ninguna categoría implica confirmar una falla real. Las anotaciones pendientes o inciertas no son referencias diagnósticas terminadas.</p></section>
      <InjectionWorkspace key={selectedDataset.dataset_id} apiBaseUrl={apiBaseUrl} dataset={selectedDataset} annotations={annotations} lastSavedAnnotation={lastSavedAnnotation} />
    </>}
  </section>;
}
