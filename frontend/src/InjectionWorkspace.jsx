import { reviewStatusLabel, reviewStatus } from "./review-annotations.js";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Download, FlaskConical, FolderOpen, Loader2, RefreshCw } from "lucide-react";
import { reviewMicros, reviewUtcTimestamp, validReviewInterval } from "./review-time.js";
import "./injection-workspace.css";
import { annotationKey, injectionBaselineProblems } from "./review-annotations.js";
import ScenarioWorkspace from "./ScenarioWorkspace.jsx";
import InjectionFragmentPicker from "./InjectionFragmentPicker.jsx";

const PARTITIONS = { development: "Desarrollo", validation: "Validación", test: "Prueba" };
const KINDS = {
  bias: { label: "Desplazamiento", help: "Suma la intensidad a cada lectura numérica válida del intervalo.", amount: "Desplazamiento" },
  drift: { label: "Deriva", help: "Añade un desplazamiento que crece linealmente desde cero hasta la intensidad indicada.", amount: "Desplazamiento final" },
  noise: { label: "Ruido", help: "Añade ruido gaussiano de media cero. La intensidad es su desviación estándar y la semilla permite repetirlo.", amount: "Desviación estándar" },
  spike: { label: "Pico", help: "Suma la intensidad solo a la primera lectura numérica válida del intervalo.", amount: "Amplitud del pico" },
  stuck: { label: "Valor congelado", help: "Repite la primera lectura válida del intervalo en las lecturas válidas siguientes." },
  unavailable: { label: "Indisponibilidad", help: "Escribe el estado unavailable en el intervalo. Conserva las filas y sus marcas de tiempo." },
  state_replace: { label: "Estado sustituido", help: "Sustituye los estados válidos del intervalo por un estado observado en esta entidad dentro del fragmento revisado." },
};
const CONTINUOUS_KINDS = ["bias", "drift", "noise", "spike", "stuck", "unavailable"];
const DISCRETE_KINDS = ["stuck", "unavailable", "state_replace"];
const INVALID_STATES = new Set(["", "unknown", "unavailable", "none", "null", "nan", "inf", "-inf", "infinity", "-infinity"]);
const LABELS = { normal: "Normal", environmental_anomaly: "Anomalía ambiental", sensor_fault: "Falla del sensor", uncertain: "Incierto" };
const timestampLabel = (value) => String(value || "—").replace("T", " ").replace(/\+00:00$/, "Z");
const finiteState = (value) => !INVALID_STATES.has(String(value ?? "").trim().toLowerCase()) && Number.isFinite(Number(value));
const validState = (value) => String(value).trim() === "NONE" || !INVALID_STATES.has(String(value ?? "").trim().toLowerCase());

async function request(base, path, options = {}) {
  const response = await fetch(`${base.replace(/\/$/, "")}${path}`, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === "string" ? payload.detail : payload.detail ? JSON.stringify(payload.detail) : `Error HTTP ${response.status}`);
  }
  return response.json();
}

export function Comparison({ run, continuous, onSelectTime, originalOnly = false, title = "Comparación del caso guardado" }) {
  const [changesOnly, setChangesOnly] = useState(false);
  const points = run.preview || [];
  const tablePoints = changesOnly ? points.filter((point) => point.changed) : points;
  const width = 1000;
  const height = 230;
  const left = 100;
  const right = 24;
  const top = 24;
  const bottom = 52;
  const times = points.map((point) => reviewMicros(point.timestamp));
  const validTimes = times.filter((value) => value !== null);
  const first = validTimes.reduce((value, time) => time < value ? time : value, validTimes[0] ?? 0n);
  const last = validTimes.reduce((value, time) => time > value ? time : value, validTimes[0] ?? 0n);
  const span = Number(last - first) || 1;
  const x = (time) => left + Number((time ?? first) - first) / span * (width - left - right);
  const states = [...new Set(points.flatMap((point) => [point.original_state, point.injected_state]).filter(validState).map(String))];
  const valueOf = (state) => continuous ? (finiteState(state) ? Number(state) : NaN) : (validState(state) ? states.indexOf(String(state)) : NaN);
  const values = points.flatMap((point) => [valueOf(point.original_state), valueOf(point.injected_state)]).filter(Number.isFinite);
  let low = values.length ? Math.min(...values) : 0;
  let high = values.length ? Math.max(...values) : 1;
  if (low === high) { low -= 0.5; high += 0.5; }
  const scale = Math.max(Math.abs(low), Math.abs(high), 1);
  const scaledLow = low / scale;
  const scaledHigh = high / scale;
  const margin = continuous ? (scaledHigh - scaledLow) * 0.12 : 0.35 / scale;
  const y = (value) => top + (scaledHigh + margin - value / scale) / (scaledHigh - scaledLow + 2 * margin || 1) * (height - top - bottom);
  function path(field) {
    let connected = false;
    let output = "";
    points.forEach((point, index) => {
      const value = valueOf(point[field]);
      if (!Number.isFinite(value) || times[index] === null) { connected = false; return; }
      output += connected ? (continuous ? ` L${x(times[index])},${y(value)}` : ` H${x(times[index])} V${y(value)}`) : ` M${x(times[index])},${y(value)}`;
      connected = true;
    });
    return output;
  }
  const ticks = continuous ? [...new Set([low, low / 2 + high / 2, high])].map((value) => ({ value, text: Number(value.toPrecision(5)).toString() })) : states.map((text, value) => ({ value, text })).filter((_, index) => states.length <= 5 || index % Math.ceil(states.length / 5) === 0 || index === states.length - 1);
  const start = reviewMicros(run.start);
  const end = reviewMicros(run.end);
  return <div className="injection-comparison">
    <div className="injection-comparison-heading"><h4>{title}</h4><div className="injection-legend"><span><i className="injection-original-swatch" />Original</span>{!originalOnly && <span><i className="injection-altered-swatch" />Con inyección</span>}<span><i className="injection-invalid-swatch" />Estado no válido</span></div></div>
    {points.length ? <svg viewBox={`0 0 ${width} ${height}`} role={onSelectTime ? "group" : "img"} aria-label={`Comparación de lecturas originales y alteradas de ${run.entity_id}. El intervalo de intervención aparece sombreado. Los valores exactos están en la tabla.`}>
      {start !== null && end !== null && <rect x={Math.max(left, Math.min(width - right, x(start)))} y={top - 8} width={Math.max(1, Math.min(width - right, x(end)) - Math.max(left, x(start)))} height={height - top - bottom + 16} className="injection-window" />}
      {ticks.map((tick) => <g key={tick.value}><line x1={left} x2={width - right} y1={y(tick.value)} y2={y(tick.value)} className="injection-grid-line" /><text x={left - 10} y={y(tick.value) + 4} textAnchor="end" className="injection-axis"><title>{tick.text}</title>{tick.text.length > 14 ? `${tick.text.slice(0, 13)}…` : tick.text}</text></g>)}
      <path d={path("original_state")} className="injection-original-line" />
      {!originalOnly && <path d={path("injected_state")} className="injection-altered-line" />}
      {points.flatMap((point, index) => (originalOnly ? ["original_state"] : ["original_state", "injected_state"]).map((field) => {
        const value = valueOf(point[field]);
        if (times[index] === null) return null;
        return <circle key={`${index}-${field}`} cx={x(times[index])} cy={Number.isFinite(value) ? y(value) : height - bottom + (field === "original_state" ? 14 : 24)} r={onSelectTime ? 4 : Number.isFinite(value) ? 2 : 3} role={onSelectTime ? "button" : undefined} tabIndex={onSelectTime ? 0 : undefined} aria-label={onSelectTime ? `Seleccionar ${reviewUtcTimestamp(point.timestamp)}, valor ${point[field]}` : undefined} onClick={() => onSelectTime?.(point.timestamp)} onKeyDown={event => { if (onSelectTime && ["Enter", " "].includes(event.key)) { event.preventDefault(); onSelectTime(point.timestamp); } }} className={Number.isFinite(value) ? field === "original_state" ? "injection-original-point" : "injection-altered-point" : "injection-invalid-point"}><title>{`${reviewUtcTimestamp(point.timestamp)} · ${point[field]}`}</title></circle>;
      }))}
      <text x={left} y={height - 9} className="injection-axis">{timestampLabel(points[0].timestamp)}</text><text x={width - right} y={height - 9} textAnchor="end" className="injection-axis">{timestampLabel(points[points.length - 1].timestamp)}</text>
    </svg> : <p className="review-help">No hay vista previa para este caso. Descarga el ZIP para inspeccionar sus datos.</p>}
    <p className="review-help">{originalOnly ? "Lecturas originales de la entidad; todavía no se han modificado. " : "La comparación muestra la entidad intervenida y su contexto temporal. "}{run.preview_sampled ? "La vista está muestreada; la inyección usa la resolución completa." : "Consulta las fechas y valores exactos."}</p>
    {!originalOnly && <details className="injection-preview-table"><summary>Lecturas originales y alteradas · {points.length} filas en la vista</summary><label className="injection-checkbox"><input type="checkbox" checked={changesOnly} onChange={(event) => setChangesOnly(event.target.checked)} />Mostrar solo filas modificadas en esta vista</label><div className="review-table-scroll"><table><thead><tr><th>Fila de origen</th><th>Fecha y hora (zona original)</th><th>Original</th><th>Con inyección</th><th>Modificada</th></tr></thead><tbody>{tablePoints.map((point, index) => <tr key={`${point.source_row}-${index}`} className={point.changed ? "injection-changed-row" : undefined}><td>{point.source_row}</td><td>{timestampLabel(point.timestamp)}</td><td>{String(point.original_state)}</td><td>{String(point.injected_state)}</td><td>{point.changed ? "Sí" : "No"}</td></tr>)}</tbody></table>{!tablePoints.length && <p className="review-empty">No hay filas que mostrar con este filtro.</p>}</div></details>}
  </div>;
}

export default function InjectionWorkspace({ apiBaseUrl = "http://localhost:8000", dataset, annotations = [], lastSavedAnnotation = null }) {
  const datasetId = dataset?.dataset_id || "";
  const [baseId, setBaseId] = useState("");
  const [kind, setKind] = useState("bias");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [amount, setAmount] = useState("1");
  const [seed, setSeed] = useState("42");
  const [replacementState, setReplacementState] = useState("");
  const [runs, setRuns] = useState([]);
  const [result, setResult] = useState(null);
  const [lastCreated, setLastCreated] = useState(null);
  const [listBusy, setListBusy] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [openingId, setOpeningId] = useState("");
  const [exportingId, setExportingId] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [refresh, setRefresh] = useState(0);
  const activeDataset = useRef(datasetId);
  activeDataset.current = datasetId;
  const mounted = useRef(false);
  const controllers = useRef(new Set());
  const openSequence = useRef(0);
  const currentBaseVersion = useRef("");
  const reviewedAnnotations = useMemo(() => annotations.filter((annotation) => annotation.dataset_id === datasetId), [annotations, datasetId]);
  const intervalAnnotations = useMemo(() => reviewedAnnotations.filter(annotation => annotation.source_row == null), [reviewedAnnotations]);
  const base = intervalAnnotations.find((annotation) => annotationKey(annotation) === baseId);
  const baselineProblems = injectionBaselineProblems(base);
  const entity = dataset?.entities?.find((item) => item.entity_id === base?.entity_id);
  const types = entity?.kind === "continuous" ? CONTINUOUS_KINDS : DISCRETE_KINDS;
  const baseVersion = base ? `${annotationKey(base)}:${base.revision}` : "";
  currentBaseVersion.current = baseVersion;

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; controllers.current.forEach((controller) => controller.abort()); controllers.current.clear(); };
  }, []);

  useEffect(() => {
    controllers.current.forEach((controller) => controller.abort());
    controllers.current.clear();
    openSequence.current += 1;
    setLastCreated(null);
    setBaseId(""); setRuns([]); setResult(null); setError(""); setNotice(""); setGenerating(false); setOpeningId(""); setExportingId("");
  }, [datasetId]);

  useEffect(() => {
    setBaseId((current) => intervalAnnotations.some((annotation) => annotationKey(annotation) === current) ? current : annotationKey(intervalAnnotations[0]));
  }, [intervalAnnotations]);

  useEffect(() => {
    if (lastSavedAnnotation?.dataset_id === datasetId) setBaseId(annotationKey(lastSavedAnnotation));
  }, [lastSavedAnnotation, datasetId]);

  useEffect(() => {
    setStart(base?.start || ""); setEnd(base?.end || "");
    setKind(entity?.kind === "continuous" ? "bias" : "stuck");
    setAmount("1"); setReplacementState("");
  }, [baseVersion, entity?.kind]);

  useEffect(() => {
    if (!datasetId) { setListBusy(false); return undefined; }
    const controller = new AbortController();
    controllers.current.add(controller);
    setListBusy(true);
    request(apiBaseUrl, `/review/injections?dataset_id=${encodeURIComponent(datasetId)}`, { signal: controller.signal }).then((payload) => {
      if (!controller.signal.aborted && activeDataset.current === datasetId) setRuns((current) => [...new Map([...current, ...(payload.injections || [])].map((run) => [run.injection_id, run])).values()].sort((a, b) => b.created_at.localeCompare(a.created_at)));
    }).catch((failure) => {
      if (failure.name !== "AbortError" && activeDataset.current === datasetId) setError(failure.message);
    }).finally(() => {
      controllers.current.delete(controller);
      if (!controller.signal.aborted && mounted.current && activeDataset.current === datasetId) setListBusy(false);
    });
    return () => { controller.abort(); controllers.current.delete(controller); };
  }, [apiBaseUrl, datasetId, refresh]);

  const a = reviewMicros(start);
  const b = reviewMicros(end);
  const baseStart = reviewMicros(base?.start);
  const baseEnd = reviewMicros(base?.end);
  const validInterval = validReviewInterval(start, end) && baseStart !== null && baseEnd !== null && a >= baseStart && b <= baseEnd;
  const needsAmount = Boolean(KINDS[kind]?.amount);
  const validAmount = !needsAmount || (amount.trim() !== "" && Number.isFinite(Number(amount)) && (kind === "noise" ? Number(amount) > 0 : Number(amount) !== 0));
  const validSeed = seed.trim() !== "" && Number.isInteger(Number(seed)) && Number(seed) >= 0 && Number(seed) <= 2147483647;
  const validReplacement = kind !== "state_replace" || (replacementState.trim().length > 0 && validState(replacementState));
  const validDriftDuration = kind !== "drift" || (a !== null && b !== null && a < b);
  const canGenerate = Boolean(base && !baselineProblems.length && entity && types.includes(kind) && validInterval && validDriftDuration && validAmount && validSeed && validReplacement && !generating);
  const duration = validInterval ? Number(b - a) / 1000000 : null;

  async function generate(event) {
    event.preventDefault();
    if (!canGenerate) return;
    const scope = datasetId;
    const version = baseVersion;
    const controller = new AbortController();
    controllers.current.add(controller);
    setGenerating(true); setError(""); setNotice("");
    const payload = { annotation_id: annotationKey(base), annotation_revision: base.revision, kind, start: reviewUtcTimestamp(start), end: reviewUtcTimestamp(end), amount: needsAmount ? Number(amount) : 1, seed: Number(seed), replacement_state: kind === "state_replace" ? replacementState.trim() : "" };
    try {
      const saved = await request(apiBaseUrl, "/review/injections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: controller.signal });
      if (controller.signal.aborted || !mounted.current || activeDataset.current !== scope) return;
      openSequence.current += 1;
      setOpeningId(""); setResult(saved);
      setLastCreated(saved);
      setRuns((current) => [saved, ...current.filter((run) => run.injection_id !== saved.injection_id)]);
      setNotice(`Caso guardado con ${saved.changed_rows} filas modificadas.${currentBaseVersion.current !== version ? " El resultado conserva la revisión y los parámetros utilizados al generar." : ""}`);
    } catch (failure) {
      if (failure.name !== "AbortError" && mounted.current && activeDataset.current === scope) setError(failure.message);
    } finally {
      controllers.current.delete(controller);
      if (!controller.signal.aborted && mounted.current && activeDataset.current === scope) setGenerating(false);
    }
  }

  async function openRun(id) {
    const scope = datasetId;
    const sequence = ++openSequence.current;
    const controller = new AbortController();
    controllers.current.add(controller);
    setOpeningId(id); setError(""); setNotice("");
    try {
      const saved = await request(apiBaseUrl, `/review/injections/${encodeURIComponent(id)}`, { signal: controller.signal });
      if (!controller.signal.aborted && mounted.current && activeDataset.current === scope && sequence === openSequence.current) setResult(saved);
    } catch (failure) {
      if (failure.name !== "AbortError" && mounted.current && activeDataset.current === scope && sequence === openSequence.current) setError(failure.message);
    } finally {
      controllers.current.delete(controller);
      if (!controller.signal.aborted && mounted.current && activeDataset.current === scope && sequence === openSequence.current) setOpeningId("");
    }
  }

  async function download(id) {
    if (exportingId) return;
    const scope = datasetId;
    const controller = new AbortController();
    controllers.current.add(controller);
    setExportingId(id); setError("");
    try {
      const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/review/injections/${encodeURIComponent(id)}/export`, { signal: controller.signal });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(typeof payload.detail === "string" ? payload.detail : `No se pudo descargar el caso: HTTP ${response.status}`);
      }
      const blob = await response.blob();
      if (controller.signal.aborted || !mounted.current || activeDataset.current !== scope) return;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url; anchor.download = `inyeccion-${id}.zip`; document.body.appendChild(anchor); anchor.click(); anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      setNotice("ZIP preparado: fragmento original, datos alterados y trazabilidad de la intervención.");
    } catch (failure) {
      if (failure.name !== "AbortError" && mounted.current && activeDataset.current === scope) setError(failure.message);
    } finally {
      controllers.current.delete(controller);
      if (!controller.signal.aborted && mounted.current && activeDataset.current === scope) setExportingId("");
    }
  }

  if (!datasetId) return null;
  return <section className="review-panel injection-workspace" aria-labelledby="injection-heading">
    <div className="review-panel-title"><h3 id="injection-heading">5. Generar inyecciones sobre fragmentos revisados</h3><span className="review-badge"><FlaskConical size={14} />Casos semisintéticos</span></div>
    <p className="review-help injection-intro">Usa una revisión guardada como Normal, elige una intervención y conserva una copia del resultado. Cada caso registra su semilla, su revisión de origen y la partición del histórico.</p>
    {error && <div className="review-alert injection-feedback" role="alert">{error}</div>}
    {notice && <div className="review-notice injection-feedback" role="status">{notice}</div>}
    {!intervalAnnotations.length ? <div className="review-empty injection-empty"><FlaskConical size={26} /><p>Guarda una anotación del histórico para cargarla aquí.</p><span>Para generar inyecciones, guarda una referencia Normal como Revisado, con responsable y observaciones del histórico. No se exige evidencia externa.</span></div> : <form onSubmit={generate}>
      <fieldset disabled={generating}>
        <label className="injection-base-label">Revisión base para la inyección<select value={baseId} onChange={(event) => { setBaseId(event.target.value); setError(""); }}>
          {intervalAnnotations.map((annotation) => <option key={annotationKey(annotation)} value={annotationKey(annotation)}>{annotation.entity_id} · {timestampLabel(annotation.start)} · {LABELS[annotation.label] || annotation.label} · revisión {annotation.revision} · {injectionBaselineProblems(annotation).length ? "requiere completar revisión" : reviewStatusLabel(annotation)}</option>)}
        </select></label>
        {base && (baselineProblems.length ? <div className="injection-baseline-note" role="status"><strong>Anotación cargada · pendiente para generar inyecciones</strong><span>Abre esta anotación en «Anotaciones del histórico», completa lo siguiente en el paso 4 y guarda los cambios:</span><ul>{baselineProblems.map((problem) => <li key={problem}>{problem}</li>)}</ul><small>Revisión {base.revision} · {timestampLabel(base.start)} → {timestampLabel(base.end)}</small></div> : <div className={`injection-baseline-note ${reviewStatus(base) === "reviewed" ? "injection-baseline-verified" : ""}`}><strong>{reviewStatus(base) === "reviewed" ? "Base normal revisada · inyección evaluable" : "Revisión pendiente · completa la anotación"}</strong><span>{reviewStatus(base) === "reviewed" ? "Origen: inyección controlada. La intervención será conocida; la normalidad del original se asume tras revisar el histórico." : "Guarda el fragmento como Revisado para usarlo como base."} Las etiquetas de la intervención no convierten el resto del histórico en datos normales.</span><small>{PARTITIONS[base.partition] || base.partition} · {base.reviewer} · intervalo revisado: {timestampLabel(base.start)} → {timestampLabel(base.end)}</small></div>)}
        <InjectionFragmentPicker apiBaseUrl={apiBaseUrl} base={base} entity={entity} start={start} end={end} onStart={setStart} onEnd={setEnd} disabled={generating} Comparison={Comparison} />
        <div className="injection-fields">
          <label>Tipo de inyección<select value={kind} onChange={(event) => setKind(event.target.value)}>{types.map((type) => <option key={type} value={type}>{KINDS[type].label}</option>)}</select></label>
          <label>Semilla de reproducción<input type="number" inputMode="numeric" min="0" max="2147483647" step="1" value={seed} onChange={(event) => setSeed(event.target.value)} /></label>
          {needsAmount && <label>{KINDS[kind].amount}{entity?.unit ? ` (${entity.unit})` : " (unidades de la entidad)"}<input type="number" step="any" value={amount} onChange={(event) => setAmount(event.target.value)} /></label>}
          {kind === "state_replace" && <label>Estado de sustitución<input type="text" value={replacementState} placeholder="Por ejemplo: on" onChange={(event) => setReplacementState(event.target.value)} /></label>}
          <label className="injection-time-label">Inicio de la inyección (ISO 8601, UTC)<input type="text" spellCheck={false} value={start} placeholder="2026-01-01T12:00:00.000000Z" onChange={(event) => setStart(event.target.value)} /></label>
          <label className="injection-time-label">Fin de la inyección (ISO 8601, UTC)<input type="text" spellCheck={false} value={end} placeholder="2026-01-01T12:30:00.000000Z" onChange={(event) => setEnd(event.target.value)} /></label>
        </div>
        <p className="review-help">{KINDS[kind]?.help} {duration !== null && <span>Duración seleccionada: <strong>{duration.toLocaleString("es-CL", { maximumFractionDigits: 6 })} s</strong>. Ambos extremos están incluidos.</span>}</p>
        {base && !validInterval && <p className="review-inline-error">Indica fechas con zona horaria dentro del intervalo revisado, con el inicio anterior o igual al fin. Se conservan hasta 6 decimales de segundo.</p>}
        {validInterval && !validDriftDuration && <p className="review-inline-error">La deriva requiere una duración mayor que cero.</p>}
        {!validAmount && <p className="review-inline-error">{kind === "noise" ? "La desviación estándar debe ser un número mayor que cero." : "La intensidad debe ser un número finito distinto de cero."}</p>}
        {!validSeed && <p className="review-inline-error">La semilla debe ser un entero entre 0 y 2147483647.</p>}
        {!validReplacement && <p className="review-inline-error">Indica un estado válido. Para simular unavailable, usa Indisponibilidad.</p>}
        <div className="review-actions"><button className="review-button review-button-primary" type="submit" disabled={!canGenerate}>{generating ? <Loader2 size={16} className="review-spin" /> : <FlaskConical size={16} />}{generating ? "Generando caso…" : "Generar y guardar caso"}</button><span className="injection-action-note">El CSV de origen se conserva intacto.</span></div>
      </fieldset>
    </form>}
    {result && <article className="injection-result" aria-label="Resultado de la inyección guardada">
      <div className="injection-result-heading"><div><span className="review-eyebrow">Caso guardado</span><h4>{KINDS[result.kind]?.label || result.kind} · {result.entity_id}</h4><small>{result.injection_id}</small></div><button className="review-button" type="button" onClick={() => download(result.injection_id)} disabled={Boolean(exportingId)}>{exportingId === result.injection_id ? <Loader2 size={15} className="review-spin" /> : <Download size={15} />}Descargar ZIP del caso</button></div>
      <div className="injection-stats"><div><span>Filas modificadas</span><strong>{Number(result.changed_rows || 0).toLocaleString("es-CL")}</strong></div><div><span>Filas exportadas</span><strong>{Number(result.exported_rows || 0).toLocaleString("es-CL")}</strong></div><div><span>Semilla</span><strong>{result.seed}</strong></div><div><span>Partición heredada</span><strong>{PARTITIONS[result.partition] || result.partition}</strong></div><div><span>Revisión de origen</span><strong>{result.annotation_revision}</strong></div></div>
      <p className="review-help"><span className={`review-status ${result.evaluable ? "review-verified" : ""}`}>{result.evaluable ? "Inyección controlada · evaluable" : "Exploratorio · no evaluable"}</span> {timestampLabel(result.start)} → {timestampLabel(result.end)}{KINDS[result.kind]?.amount ? ` · ${KINDS[result.kind].amount}: ${result.amount}` : ""}{result.kind === "state_replace" ? ` · estado: ${result.replacement_state}` : ""}</p>
      {(result.warnings || []).length > 0 && <ul className="injection-warnings">{result.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>}
      <Comparison key={result.injection_id} run={result} continuous={dataset?.entities?.find((item) => item.entity_id === result.entity_id)?.kind === "continuous"} />
    </article>}
    <div className="injection-saved-heading"><h4>Casos guardados de este histórico</h4><button className="review-button injection-refresh" type="button" onClick={() => setRefresh((value) => value + 1)} disabled={listBusy || generating}><RefreshCw size={14} className={listBusy ? "review-spin" : ""} />Actualizar casos</button></div>
    {listBusy ? <p className="review-help" role="status">Cargando casos guardados…</p> : !runs.length ? <p className="review-help">Todavía no hay inyecciones guardadas para este histórico.</p> : <div className="injection-run-list">{runs.map((run) => <article key={run.injection_id} className={`injection-run ${result?.injection_id === run.injection_id ? "injection-run-active" : ""}`}><div><strong>{KINDS[run.kind]?.label || run.kind} · {run.entity_id}</strong><p>{timestampLabel(run.start)} → {timestampLabel(run.end)}</p><small>{run.changed_rows} filas modificadas · semilla {run.seed} · revisión {run.annotation_revision} · {run.evaluable ? "evaluable" : "exploratorio"}</small></div><div className="review-actions"><button className="review-button" type="button" onClick={() => openRun(run.injection_id)} disabled={generating || openingId === run.injection_id} aria-label={`Abrir caso ${run.injection_id}`}>{openingId === run.injection_id ? <Loader2 size={14} className="review-spin" /> : <FolderOpen size={14} />}Abrir</button><button className="review-button" type="button" onClick={() => download(run.injection_id)} disabled={Boolean(exportingId)} aria-label={`Descargar ZIP del caso ${run.injection_id}`}>{exportingId === run.injection_id ? <Loader2 size={14} className="review-spin" /> : <Download size={14} />}ZIP</button></div></article>)}</div>}
    <ScenarioWorkspace key={datasetId} apiBaseUrl={apiBaseUrl} dataset={dataset} runs={runs} annotations={reviewedAnnotations} lastCreated={lastCreated} Comparison={Comparison} />
  </section>;
}
