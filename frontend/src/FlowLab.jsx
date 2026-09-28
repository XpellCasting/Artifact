import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ReactFlow, BaseEdge, Background, Controls, Handle, MarkerType, Position, ReactFlowProvider, useNodesState, useReactFlow } from "@xyflow/react";
import { AlertTriangle, ArrowLeft, ArrowRight, Check, Copy, GitBranch, Maximize2, Play, Plus, RotateCcw, Save, Trash2 } from "lucide-react";
import "@xyflow/react/dist/style.css";
import { NODE_LABELS, SIGNAL_LABELS, DECISIONS, SIGNALS, clone, connect, edgeLabel, fingerprint, flowPayload, labelNode, removeNode, templates, validateDraft } from "./flow-lab-model.js";
import "./flow-lab.css";

const PRESETS = templates();
const PAGE_SIZE = 50;
const DESCRIPTIONS = { input: "Lecturas del histórico seleccionado", statistical: "Sospechas temporales", statistical_focus: "Cambios secuenciales en residuos", contextual: "Coherencia entre entidades", classifier: "Compatibilidad con patrones", merge: "Reúne las evidencias de las ramas", decision: "Distribuye según la señal recibida", output: "Resultado final por lectura" };
async function request(base, path, options = {}) {
  const response = await fetch(`${base}${path}`, { ...options, headers: { "Content-Type": "application/json", ...options.headers } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `No se pudo completar la operación (${response.status}).`);
  return payload;
}
function FilterNode({ data, selected }) {
  const disabled = data.node.config?.enabled === false;
  return <div className={`fl-node fl-node-${data.node.type} ${selected ? "is-selected" : ""} ${disabled ? "is-disabled" : ""}`}>
    {data.node.type !== "input" && <Handle type="target" position={Position.Left} />}
    <span className="fl-node-kind">{NODE_LABELS[data.node.type]}</span><strong>{labelNode(data.node)}</strong>
    <span className="fl-node-detail">{disabled ? "Omitido · deja pasar la entrada" : DESCRIPTIONS[data.node.type]}</span>
    {data.count !== undefined && <span className="fl-node-count">{data.count.toLocaleString()} lecturas</span>}
    {data.node.type !== "output" && <Handle type="source" position={Position.Right} />}
  </div>;
}
const NODE_TYPES = { filter: FilterNode };
function RoutedEdge({ sourceX, sourceY, targetX, targetY, data, ...props }) {
  const bypass = targetX - sourceX > 200;
  const lane = Math.min(sourceY, targetY) - (bypass ? 170 + data.lane * 45 : 85);
  const path = `M ${sourceX} ${sourceY} L ${sourceX + 20} ${sourceY} L ${sourceX + 20} ${lane} L ${targetX - 20} ${lane} L ${targetX - 20} ${targetY} L ${targetX} ${targetY}`;
  return <BaseEdge id={props.id} style={props.style} markerEnd={props.markerEnd} markerStart={props.markerStart} label={props.label} labelStyle={props.labelStyle} labelBgStyle={props.labelBgStyle} labelBgPadding={props.labelBgPadding} labelBgBorderRadius={props.labelBgBorderRadius} path={path} labelX={(sourceX + targetX) / 2} labelY={lane} />;
}
const EDGE_TYPES = { routed: RoutedEdge };
function Signals({ value, onChange, legend = "Señales que pasan por esta conexión" }) {
  return <fieldset className="fl-signals"><legend>{legend}</legend>{SIGNALS.map((signal) => <label key={signal}><input type="checkbox" checked={value.includes(signal)} onChange={(event) => onChange(event.target.checked ? [...value, signal] : value.filter((item) => item !== signal))} />{SIGNAL_LABELS[signal]}</label>)}</fieldset>;
}
function FlowEvaluationProgress({ job, fallbackTotal }) {
  const processed = Math.max(0, Number(job?.processed) || 0);
  const total = Math.max(0, Number(job?.total) || Number(fallbackTotal) || 0);
  const progress = total ? Math.min(100, processed / total * 100) : 0;
  const count = (value) => Number(value).toLocaleString();
  const phase = job?.phase || "evaluation";
  const title = phase === "preparation" ? "Preparando evaluación" : phase === "saving" ? "Guardando ejecución" : "Evaluando flujo";
  const detail = job?.detail || (phase === "preparation" ? "Calibrando modelos y preparando estados causales…" : phase === "saving" ? "Persistiendo resultados…" : null);
  const groupStarted = job?.group_started_at ? Date.parse(job.group_started_at) : NaN;
  const groupSeconds = Number.isFinite(groupStarted) ? Math.max(0, Math.floor((Date.now() - groupStarted) / 1000)) : null;
  const lastCompleted = job?.last_completed_at ? Date.parse(job.last_completed_at) : NaN;
  const noRecentGroups = job?.recent_events_per_second == null && Number.isFinite(lastCompleted) && Date.now() - lastCompleted >= 30000;
  const rateDetail = job?.recent_events_per_second != null && Number.isFinite(Number(job.recent_events_per_second))
    ? `${Number(job.recent_events_per_second).toLocaleString()} lecturas/s (últimos 30 s)`
    : noRecentGroups ? "Sin grupos completados en los últimos 30 s" : "Calculando ritmo…";
  return <section className="history-load-progress fl-evaluation-progress" aria-live="polite">
    <div className="history-load-heading"><span>{title}</span><strong>{phase === "evaluation" ? `${Math.round(progress)}%` : "En curso"}</strong></div>
    <div className="progress-track" role="progressbar" aria-label="Progreso de evaluación del flujo" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(progress)} aria-valuetext={`${count(processed)} de ${total ? count(total) : "total por determinar"} lecturas procesadas`}>
      <span style={{ width: `${progress}%` }} />
    </div>
    <p>{detail || (total ? `Procesadas ${count(processed)} de ${count(total)} lecturas del conjunto de evaluación.` : `Preparando las lecturas que se evaluarán…`)}</p>
    <div className="history-load-meta"><span>Lecturas procesadas: {phase === "evaluation" ? count(processed) : "—"}</span><span>Total: {phase === "evaluation" && total ? count(total) : "—"}</span></div>
    {phase === "evaluation" && job?.current_timestamp && <div className="history-load-meta">
      <span>Grupo actual: {new Date(job.current_timestamp).toLocaleString()} · {count(job.current_batch_rows || 0)} lecturas</span>
      <span>{groupSeconds === null ? `Último grupo: ${Number(job.last_batch_ms || 0).toLocaleString()} ms` : `Procesando hace ${groupSeconds} s`}</span>
      <span>{rateDetail}</span>
    </div>}
  </section>;
}

function FlowLabContent({ apiBaseUrl, historyReady, replay, upload, uploadBusy, onUpload, onLabelledUpload }) {
  const [draft, setDraft] = useState(() => clone(PRESETS[0]));
  const [savedKey, setSavedKey] = useState("");
  const [flows, setFlows] = useState([]);
  const [runs, setRuns] = useState([]);
  const [selection, setSelection] = useState(null);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(() => Boolean(localStorage.getItem("flow-evaluation-job")));
  const [job, setJob] = useState(() => { const id = localStorage.getItem("flow-evaluation-job"); return id ? { job_id: id, status: "running", processed: 0, total: 0 } : null; });
  const [loaded, setLoaded] = useState(false);
  const [newType, setNewType] = useState("statistical");
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [activeRun, setActiveRun] = useState(null);
  const [inspecting, setInspecting] = useState(false);
  const [resultNode, setResultNode] = useState("output");
  const [page, setPage] = useState({ items: [], total: 0, offset: 0, next_offset: null });
  const [resultBusy, setResultBusy] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const pageRequest = useRef(0);
  const [layoutRevision, setLayoutRevision] = useState(0);
  const [canvasNodes, setCanvasNodes, onCanvasNodesChange] = useNodesState([]);
  const { fitView, screenToFlowPosition } = useReactFlow();
  const canvasRef = useRef(null);
  const graph = inspecting && activeRun ? activeRun.snapshot.flow : draft;
  const dirty = fingerprint(draft) !== savedKey;
  const errors = useMemo(() => validateDraft(draft), [draft]);
  const node = selection?.kind === "node" ? graph.nodes.find((item) => item.id === selection.id) : null;
  const edge = selection?.kind === "edge" ? graph.edges.find((item) => item.id === selection.id) : null;
  const refresh = useCallback(async () => {
    const [f, r] = await Promise.all([request(apiBaseUrl, "/flows"), request(apiBaseUrl, "/flow-runs")]);
    setFlows(f.items || []); setRuns(r.items || []); setLoaded(true);
  }, [apiBaseUrl]);
  useEffect(() => { refresh().catch((reason) => setError(reason.message)); }, [refresh]);
  useEffect(() => {
    if (!job?.job_id) return;
    let stopped = false;
    let timer;
    async function poll() {
      try {
        const state = await request(apiBaseUrl, `/flow-evaluations/${job.job_id}`);
        if (stopped) return;
        if (state.status === "failed") throw new Error(state.error);
        if (state.status === "completed") {
          localStorage.removeItem("flow-evaluation-job"); setJob(null); setBusy(false);
          setActiveRun(state.run); setInspecting(true); setResultNode("output"); setSelection(null);
          setMessage("Evaluación terminada. El gráfico muestra el flujo utilizado con todos los eventos futuros.");
          await refresh();
        } else { setJob(state); timer = setTimeout(poll, 1500); }
      } catch (reason) { if (!stopped) { setError(reason.message); setBusy(false); setJob(null); localStorage.removeItem("flow-evaluation-job"); } }
    }
    poll();
    return () => { stopped = true; clearTimeout(timer); };
  }, [job?.job_id, apiBaseUrl, refresh]);
  useEffect(() => { const frame = requestAnimationFrame(() => fitView({ padding: .18, duration: 200, maxZoom: 1 })); return () => cancelAnimationFrame(frame); }, [layoutRevision, inspecting, fitView]);
  useEffect(() => {
    const observer = new ResizeObserver(() => fitView({ padding: .18, maxZoom: 1 }));
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [fitView]);
  function change(mutator) { setDraft((current) => { const next = clone(current); mutator(next); return next; }); setMessage(""); }
  function choose(flow, isSaved = false) {
    setDraft(clone(flow)); setSavedKey(isSaved ? fingerprint(flow) : ""); setSelection(null); setSource(""); setTarget(""); setInspecting(false); setMessage(""); setError(""); setLayoutRevision((value) => value + 1);
  }
  function addNode() {
    const id = `${newType}_${crypto.randomUUID().slice(0, 8)}`;
    const bounds = canvasRef.current?.getBoundingClientRect();
    const position = bounds ? screenToFlowPosition({ x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2 }) : { x: 400, y: 100 };
    change((next) => next.nodes.push({ id, type: newType, position, config: newType === "statistical" ? { suspicion_threshold: .6, continuous_method: "temporal_quantile_v1" } : newType === "statistical_focus" ? { suspicion_threshold: .6, continuous_method: "residual_focus_v1", signal_policy: "detection_before_suppression" } : newType === "decision" ? { accept: ["positive"] } : newType === "merge" ? { mode: "any" } : {} }));
    setSelection({ kind: "node", id });
  }
  function addConnection({ source: origin, target: destination }) {
    const next = connect(draft, origin, destination);
    if (!next) { setError("Conexión no válida: revisa la dirección, los ciclos y las uniones de ramas."); return; }
    setDraft(next); setError(""); setMessage(""); setSelection({ kind: "edge", id: next.edges.at(-1).id });
  }
  function configure(update) { change((next) => { const item = next.nodes.find((item) => item.id === node.id); item.config = { ...item.config, ...update }; }); }
  function deleteSelection() {
    if (node) setDraft((current) => removeNode(current, node.id));
    if (edge) change((next) => { next.edges = next.edges.filter((item) => item.id !== edge.id); });
    setSelection(null);
  }
  async function persist() {
    const saved = await request(apiBaseUrl, draft.flow_id ? `/flows/${draft.flow_id}` : "/flows", { method: draft.flow_id ? "PUT" : "POST", body: JSON.stringify(flowPayload(draft)) });
    setDraft(saved); setSavedKey(fingerprint(saved)); return saved;
  }
  async function save() {
    setBusy(true); setError(""); setMessage("");
    try { await persist(); await refresh(); setMessage("Flujo guardado."); } catch (reason) { setError(reason.message); } finally { setBusy(false); }
  }
  async function execute() {
    setBusy(true); setError(""); setMessage("");
    try {
      // Guardar la configuración visible antes de ejecutar evita usar una versión anterior silenciosamente.
      const saved = dirty || !draft.flow_id ? await persist() : draft;
      const started = await request(apiBaseUrl, `/flows/${saved.flow_id}/runs`, { method: "POST", body: "{}" });
      localStorage.setItem("flow-evaluation-job", started.job_id); setJob(started);
    } catch (reason) { setError(reason.message); setBusy(false); }
  }
  const loadPage = useCallback(async (runId, nodeId, offset) => {
    const revision = ++pageRequest.current;
    setResultBusy(true); setSelectedResult(null); setPage({ items: [], total: 0, offset, next_offset: null });
    try { const payload = await request(apiBaseUrl, `/flow-runs/${runId}/results?node_id=${encodeURIComponent(nodeId)}&offset=${offset}&limit=${PAGE_SIZE}`); if (revision === pageRequest.current) setPage(payload); }
    catch (reason) { if (revision === pageRequest.current) setError(reason.message); }
    finally { if (revision === pageRequest.current) setResultBusy(false); }
  }, [apiBaseUrl]);
  useEffect(() => { if (activeRun) loadPage(activeRun.run_id, resultNode, 0); return () => { pageRequest.current += 1; }; }, [activeRun, resultNode, loadPage]);
  const visualNodes = useMemo(() => graph.nodes.map((item) => ({ id: item.id, type: "filter", position: item.position || { x: 0, y: 0 }, selected: selection?.id === item.id, deletable: !["input", "output"].includes(item.type), data: { node: item, count: inspecting ? activeRun?.summary?.node_counts?.[item.id] : undefined } })), [graph, selection?.id, inspecting, activeRun?.summary?.node_counts]);
  const visualEdges = useMemo(() => graph.edges.map((item, index) => ({ ...item, data: { lane: index % 3 }, selected: selection?.id === item.id, type: "routed", markerEnd: { type: MarkerType.ArrowClosed }, label: item.when === undefined ? undefined : edgeLabel(item), labelStyle: { fontSize: 11 }, labelBgPadding: [7, 4], labelBgBorderRadius: 5, labelBgStyle: { fill: "#161d29", fillOpacity: 1 } })), [graph, selection?.id]);
  useEffect(() => { setCanvasNodes(visualNodes); }, [visualNodes, setCanvasNodes]);

  return <section className="flow-lab" aria-label="Evaluación de flujos">
    <header className="fl-header"><div><span className="eyebrow">Evaluación causal</span><h2>Evaluación de flujos</h2><p>Diseña el recorrido de los filtros y comprueba qué identifica en los eventos futuros.</p></div><div className="fl-state"><span className={`fl-dot ${inspecting || !dirty ? "is-saved" : ""}`} />{inspecting ? "Inspección de una ejecución" : dirty ? "Cambios sin guardar" : "Todos los cambios guardados"}</div></header>
    {error && <div role="alert" className="fl-notice fl-error"><AlertTriangle size={18} /><span>{error}</span>{!loaded && <button onClick={() => { setError(""); refresh().catch((e) => setError(e.message)); }}>Reintentar</button>}</div>}
    {message && <div role="status" className="fl-notice"><Check size={16} />{message}</div>}
    <div className="fl-toolbar">{inspecting ? <>
      <button onClick={() => { setInspecting(false); setSelection(null); }}><ArrowLeft size={16} />Volver al diseño</button><strong>{graph.name}</strong><span className="fl-muted">Configuración de la ejecución · solo lectura</span><button onClick={() => choose({ ...clone(graph), flow_id: null, name: `${graph.name} (copia)` })}><Copy size={16} />Editar una copia</button>
    </> : <>
      <label className="fl-name">Nombre del flujo<input aria-label="Nombre del flujo" value={draft.name} disabled={busy} onChange={(event) => change((next) => { next.name = event.target.value; })} /></label>
      <label>Abrir guardado<select aria-label="Abrir flujo guardado" disabled={busy} value={draft.flow_id || ""} onChange={(event) => { const item = flows.find((item) => item.flow_id === event.target.value); if (item) choose(item, true); }}><option value="">Seleccionar…</option>{flows.map((item) => <option key={item.flow_id} value={item.flow_id}>{item.name}</option>)}</select></label>
      <label>Nuevo desde plantilla<select aria-label="Nueva plantilla" disabled={busy} value="" onChange={(event) => choose(PRESETS[Number(event.target.value)])}><option value="" disabled>Seleccionar…</option>{PRESETS.map((item, index) => <option key={item.name} value={index}>{item.name}</option>)}</select></label>
      <button disabled={busy} onClick={() => choose({ ...clone(draft), flow_id: null, name: `${draft.name} (copia)` })}><Copy size={16} />Duplicar</button><button className="fl-primary" disabled={busy || !!errors.length} onClick={save}><Save size={16} />Guardar</button>
    </>}</div>
    <div className="fl-editor"><div className="fl-canvas-column">
      <div className="fl-canvas-toolbar">{!inspecting && <><label className="fl-inline">Añadir<select aria-label="Tipo de nodo" value={newType} disabled={busy} onChange={(event) => setNewType(event.target.value)}>{Object.entries(NODE_LABELS).filter(([type]) => !["input", "output"].includes(type)).map(([type, label]) => <option key={type} value={type}>{label}</option>)}</select></label><button disabled={busy} onClick={addNode}><Plus size={16} />Nodo</button></>}<button className="fl-fit" onClick={() => fitView({ padding: .18, duration: 200 })}><Maximize2 size={16} />Ver todo</button></div>
      <div className="fl-canvas" ref={canvasRef}><ReactFlow nodes={canvasNodes} edges={visualEdges} nodeTypes={NODE_TYPES} edgeTypes={EDGE_TYPES} colorMode="dark" fitView minZoom={.15} maxZoom={2} nodesDraggable={!inspecting && !busy} nodesConnectable={!inspecting && !busy} deleteKeyCode={inspecting || busy ? null : ["Backspace", "Delete"]}
        onConnect={addConnection} onNodeClick={(_, item) => { setSelection({ kind: "node", id: item.id }); if (inspecting) setResultNode(item.id); }} onEdgeClick={(_, item) => setSelection({ kind: "edge", id: item.id })} onPaneClick={() => setSelection(null)}
        onNodesChange={(changes) => { onCanvasNodesChange(changes); if (inspecting || busy) return; const removed = changes.filter((item) => item.type === "remove").map((item) => item.id); if (removed.length) change((next) => { for (const id of removed) { const updated = removeNode(next, id); next.nodes = updated.nodes; next.edges = updated.edges; } }); }}
        onNodeDragStop={(_, moved) => { if (inspecting || busy) return; change((next) => { const item = next.nodes.find((node) => node.id === moved.id); if (item) item.position = moved.position; }); }}
        onEdgesChange={(changes) => { if (inspecting || busy) return; const removed = changes.filter((item) => item.type === "remove").map((item) => item.id); if (removed.length) change((next) => { next.edges = next.edges.filter((item) => !removed.includes(item.id)); }); }}>
        <Background gap={22} size={1} color="#334155" /><Controls showInteractive={false} />
      </ReactFlow></div>
      <p className="fl-canvas-caption">{inspecting ? "Selecciona un nodo para consultar sus resultados." : "Arrastra los nodos. Conecta sus puntos laterales. Usa la rueda para acercar y arrastra el fondo para desplazarte."}</p>
    </div><aside className="fl-inspector" aria-label="Inspector del flujo"><h3>{node ? labelNode(node) : edge ? "Configurar conexión" : "Inspector"}</h3>
      {node ? <fieldset disabled={busy || inspecting} className="fl-fields"><span className="fl-muted">{DESCRIPTIONS[node.type]}</span><label>Nombre visible<input value={node.config?.label || ""} placeholder={NODE_LABELS[node.type]} onChange={(event) => configure({ label: event.target.value })} /></label>
        {["statistical", "statistical_focus", "contextual", "classifier"].includes(node.type) && <label className="fl-check"><input type="checkbox" checked={node.config?.enabled !== false} onChange={(event) => configure({ enabled: event.target.checked })} />Filtro activado</label>}
        {node.type === "statistical" && <><label>Continuos<select value={node.config?.continuous_method || "temporal_quantile_v1"} onChange={(event) => configure({ continuous_method: event.target.value, ...(event.target.value === "residual_ewma_v1" ? { signal_policy: "detection_before_suppression" } : {}) })}><option value="temporal_quantile_v1">Cuantiles temporales</option><option value="residual_ewma_v1">Residuos + EWMA</option></select></label><span className="fl-hint">Los métodos por residuos usan la predicción central actual, normalizan residuos en Desarrollo y calibran un presupuesto empírico de hasta un episodio diario. No son probabilidades ni diagnósticos.</span><label>Umbral de sospecha<input type="number" min="0.01" max="0.99" step="0.01" value={node.config?.suspicion_threshold === null ? "" : node.config?.suspicion_threshold ?? .6} onChange={(event) => configure({ suspicion_threshold: event.target.value === "" ? null : Number(event.target.value) })} /><span className="fl-hint">Este umbral solo controla binarios y multivalóricos; el continuo nuevo se calibra en Desarrollo.</span></label></>}
        {node.type === "statistical_focus" && <><p><strong>Continuos:</strong> FOCuS sobre residuos normalizados</p><span className="fl-hint">Combina FOCuS con canales puntual, energía y constancia. Un aviso puede originarse en cualquiera de ellos; no confirma un tipo de falla.</span></>}
        {node.type === "merge" && <label>Señal de la unión<select value={node.config?.mode || "any"} onChange={(event) => configure({ mode: event.target.value })}><option value="any">Cualquiera positiva (O)</option><option value="all">Todas positivas (Y)</option></select><span className="fl-hint">La evidencia conserva su origen. Una rama no evaluada no cuenta como respaldo.</span></label>}
        {node.type === "decision" && <><Signals legend="Resultado verdadero cuando recibe" value={node.config?.accept || ["positive"]} onChange={(accept) => configure({ accept })} /><p className="fl-hint">Verdadero produce señal positiva; falso, sin señal. Una entrada no evaluable permanece desconocida salvo que la selecciones explícitamente.</p></>}
        {!["input", "output"].includes(node.type) && <button className="fl-danger" onClick={deleteSelection}><Trash2 size={15} />Eliminar nodo y conexiones</button>}
      </fieldset> : edge ? <fieldset disabled={busy || inspecting} className="fl-fields"><p className="fl-route">{labelNode(graph.nodes.find((item) => item.id === edge.source))}<ArrowRight size={15} />{labelNode(graph.nodes.find((item) => item.id === edge.target))}</p><Signals value={edge.when || SIGNALS} onChange={(when) => change((next) => { next.edges.find((item) => item.id === edge.id).when = when; })} /><button className="fl-danger" onClick={deleteSelection}><Trash2 size={15} />Eliminar conexión</button></fieldset> : <p className="fl-muted">Selecciona un filtro o una conexión para ver y editar sus propiedades.</p>}
      {!inspecting && <details className="fl-connect"><summary>Conectar mediante formulario</summary><fieldset disabled={busy} className="fl-fields"><label>Desde<select aria-label="Origen de conexión" value={source} onChange={(event) => setSource(event.target.value)}><option value="">Seleccionar…</option>{draft.nodes.filter((item) => item.type !== "output").map((item) => <option key={item.id} value={item.id}>{labelNode(item)}</option>)}</select></label><label>Hacia<select aria-label="Destino de conexión" value={target} onChange={(event) => setTarget(event.target.value)}><option value="">Seleccionar…</option>{draft.nodes.filter((item) => item.type !== "input").map((item) => <option key={item.id} value={item.id}>{labelNode(item)}</option>)}</select></label><button disabled={!source || !target} onClick={() => addConnection({ source, target })}><GitBranch size={15} />Conectar</button></fieldset></details>}
    </aside></div>
    {!inspecting && <>{!!errors.length && <div className="fl-validation" role="status"><strong>Completa el diseño para ejecutar</strong><ul>{errors.map((text) => <li key={text}>{text}</li>)}</ul></div>}
      <section className="fl-execution"><div><h3>Evaluación con eventos futuros</h3><p>El histórico prepara el detector. Se evalúa todo el archivo futuro en orden temporal, usando el flujo visible y sin consultar eventos posteriores. Cada ejecución empieza con una copia nueva del estado histórico.</p>
        <div className="fl-upload-actions"><label className="file-action">Cargar eventos futuros<input type="file" accept=".csv,text/csv" disabled={busy || uploadBusy || !historyReady} onChange={onUpload} /></label><label className="file-action">Cargar conjunto etiquetado<input type="file" accept=".csv,text/csv" disabled={busy || uploadBusy || !historyReady} onChange={onLabelledUpload} /></label></div>
        <p className="fl-hint">CSV etiquetado: entity_id, state, last_changed, es_sintetica, tipo, tipo_falla. Las etiquetas se usan solo para comprobar los resultados.</p>
        {replay?.loaded && <p><strong>{replay.filename}</strong> · {replay.rows} eventos · {replay.labelled ? "Con etiquetas" : "Sin etiquetas"}</p>}
        {upload && <p role="status">{upload.message}</p>}
        {!historyReady && <p className="fl-required">Carga un histórico desde Evaluación y espera a que se prepare el detector.</p>}
        {job && <FlowEvaluationProgress job={job} fallbackTotal={replay?.rows} />}
      </div><button className="fl-primary" disabled={busy || uploadBusy || !loaded || !historyReady || !replay?.loaded || !!errors.length} onClick={execute}><Play size={16} />{busy ? "Evaluando…" : dirty || !draft.flow_id ? "Guardar y evaluar" : "Evaluar flujo"}</button></section>
    </>}
    <section className="fl-comparison" aria-label="Comparación de conjuntos etiquetados"><h3>Conjuntos etiquetados evaluados</h3><p>Compara las detecciones de cada flujo. La huella identifica el contenido de los eventos: archivos con nombres iguales pueden contener datos diferentes.</p><div className="fl-table-scroll"><table><thead><tr><th>Conjunto · huella</th><th>Flujo · fecha</th><th>Fallas detectadas / total</th><th>Anomalías detectadas / total</th><th>Tipos identificados / etiquetados</th><th>Falsos positivos</th><th>Detalle</th></tr></thead><tbody>{runs.filter(r => r.summary.evaluation?.labelled).map(run => {
      const groups = run.summary.evaluation.groups;
      const sum = (kind, field) => groups.filter(g => !kind || g.expected === kind).reduce((total, g) => total + (g[field] || 0), 0);
      return <tr key={run.run_id}><td>{run.snapshot.source.filename}<small className="fl-muted"> · {run.snapshot.source.fingerprint.slice(0, 10)}</small></td><td>{run.snapshot.flow.name} · {new Date(run.created_at).toLocaleString()}</td><td>{sum("Falla", "detected")} / {sum("Falla", "total")}</td><td>{sum("anomalia", "detected")} / {sum("anomalia", "total")}</td><td>{sum("Falla", "type_identified")} / {groups.filter(g => g.expected === "Falla" && g.fault_type).reduce((n, g) => n + g.total, 0)}</td><td>{sum(null, "false_positives")}</td><td><button disabled={busy} onClick={() => { setActiveRun(run); setInspecting(true); setResultNode("output"); setSelection(null); }}>Ver resultados</button></td></tr>;
    })}</tbody></table></div>{!runs.some(r => r.summary.evaluation?.labelled) && <p className="fl-muted">Carga un conjunto etiquetado y evalúa un flujo para comenzar la comparación.</p>}</section>
    <section className="fl-results" aria-label="Resultados de flujos"><div className="fl-results-toolbar"><h3>Ejecuciones</h3><label>Consultar<select aria-label="Ejecución guardada" value={activeRun?.run_id || ""} disabled={busy} onChange={(event) => { const run = runs.find((item) => item.run_id === event.target.value); if (run) { setActiveRun(run); setInspecting(true); setResultNode("output"); setSelection(null); setError(""); setMessage(""); setLayoutRevision((value) => value + 1); } }}><option value="">Seleccionar ejecución…</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.snapshot?.flow?.name} · {run.snapshot.source?.filename || "Histórico"} · {new Date(run.created_at).toLocaleString()}</option>)}</select></label><button disabled={busy} aria-label="Actualizar ejecuciones" onClick={() => refresh().catch((e) => setError(e.message))}><RotateCcw size={16} /></button></div>
      {!activeRun ? <div className="fl-empty"><GitBranch size={28} /><strong>Tu primera ejecución aparecerá aquí</strong><p>Guarda un flujo y ejecútalo para revisar el recorrido y la evidencia.</p></div> : <>
        <p><strong>{activeRun.snapshot.source?.filename || "Ejecución anterior"}</strong> · {activeRun.snapshot.execution_mode === "causal_future" ? "Eventos futuros · evaluación causal" : "Análisis retrospectivo anterior"} · {activeRun.snapshot.baseline_id ? `Base ${activeRun.snapshot.baseline_id.slice(0, 16)}` : "Base sin procedencia registrada"}</p>
        {activeRun.snapshot.baseline_provenance?.continuous_models && <p className="fl-muted">Entrenamiento completado · {Object.values(activeRun.snapshot.baseline_provenance.continuous_models).flatMap((models) => Object.values(models || {})).filter((model) => model.eligible).length} sensores preparados · {Object.values(activeRun.snapshot.baseline_provenance.continuous_models).flatMap((models) => Object.values(models || {})).filter((model) => !model.eligible).length} no evaluables</p>}
        <div className="fl-metrics"><span><strong>{activeRun.summary.candidate_count}</strong>Lecturas de entrada</span><span><strong>{activeRun.summary.output_count}</strong>Resultados finales</span><span><strong>{Number(activeRun.summary.elapsed_ms).toLocaleString()} ms</strong>Tiempo total</span>{Object.entries(activeRun.summary.by_final_decision || {}).map(([key, value]) => <span key={key}><strong>{value}</strong>{DECISIONS[key] || key}</span>)}</div>
        <FlowLabelReport report={activeRun.summary.evaluation} />
        <StatisticalLabelReport reports={activeRun.summary.statistical_evaluation} version={activeRun.summary.statistical_evaluation_version} />
        <div className="fl-result-controls"><label>Resultados del nodo<select aria-label="Nodo de resultados" value={resultNode} onChange={(event) => setResultNode(event.target.value)}><option value="output">Salida final</option>{activeRun.snapshot.flow.nodes.filter((item) => item.id !== "output").map((item) => <option key={item.id} value={item.id}>{labelNode(item)}</option>)}</select></label><span className="fl-muted">{resultBusy ? "Cargando resultados…" : `${page.total ? page.offset + 1 : 0}–${page.offset + page.items.length} de ${page.total}`}</span><button disabled={resultBusy || page.offset === 0} onClick={() => loadPage(activeRun.run_id, resultNode, Math.max(0, page.offset - PAGE_SIZE))}><ArrowLeft size={15} />Anterior</button><button disabled={resultBusy || page.next_offset === null} onClick={() => loadPage(activeRun.run_id, resultNode, page.next_offset)}>Siguiente<ArrowRight size={15} /></button></div>
        <div className="fl-table-scroll"><table><thead><tr><th>Entidad</th><th>Fecha y hora UTC</th><th>Resultado</th><th>Etiqueta y detección</th><th>Evidencia</th></tr></thead><tbody>{page.items.map((item) => <tr key={item.observation_id}><td>{item.entity_id}</td><td>{item.timestamp}</td><td>{DECISIONS[item.final_decision] || SIGNAL_LABELS[item.signal] || "No evaluable"}</td><td>{item.evaluation ? `${item.evaluation.expected} · ${EVALUATION_LABELS[item.evaluation.outcome]}${item.evaluation.type_identified ? " · Tipo identificado" : ""}` : "—"}</td><td><button onClick={() => setSelectedResult(item)}>Ver {item.evidence?.length || 0} etapas</button></td></tr>)}</tbody></table>{!resultBusy && !page.items.length && <p className="fl-empty-row">Este nodo no recibió lecturas en la ejecución.</p>}</div>
        {selectedResult && <div className="fl-evidence"><h4>Recorrido de {selectedResult.entity_id}</h4><p className="fl-muted">{selectedResult.timestamp}</p>{(selectedResult.evidence || []).map((entry, index) => <details key={`${entry.node_id}-${index}`}><summary>{labelNode(activeRun.snapshot.flow.nodes.find((item) => item.id === entry.node_id))} · {SIGNAL_LABELS[entry.signal]} · {entry.status === "omitted" ? "Filtro omitido" : entry.status}</summary><pre>{JSON.stringify(entry.data, null, 2)}</pre></details>)}</div>}
      </>}
    </section>
  </section>;
}
export default function FlowLab(props) { return <ReactFlowProvider><FlowLabContent {...props} /></ReactFlowProvider>; }
const EVALUATION_LABELS = { detectado: "Detectado", no_detectado: "No detectado", falso_positivo: "Falso positivo", normal_correcto: "Normal correcto", no_evaluable: "No evaluable", sin_etiqueta: "Sin etiqueta" };
function FlowLabelReport({ report }) {
  if (!report?.labelled) return <p className="fl-muted">Esta ejecución no tiene etiquetas de referencia para medir detecciones y errores.</p>;
  return <section aria-label="Detección de datos etiquetados"><h3>Qué identifica este flujo</h3><p>Se cuenta como detección una sospecha, una falla compatible o una anomalía ambiental en la salida final. Identificar el tipo exige que el diagnóstico coincida con tipo_falla; una sospecha sola no identifica el tipo.</p><div className="fl-table-scroll"><table><thead><tr><th>Entidad</th><th>Etiqueta</th><th>Origen</th><th>Tipo esperado</th><th>Detectadas / total</th><th>No detectadas</th><th>Tipo identificado</th><th>Falsos positivos</th><th>Indeterminadas / sin ruta</th></tr></thead><tbody>{report.groups.map((row, i) => <tr key={i}><td>{row.entity_id}</td><td>{row.expected}</td><td>{row.synthetic ? "Inyectado" : "No inyectado"}</td><td>{row.fault_type || "—"}</td><td>{["Falla", "anomalia"].includes(row.expected) ? `${row.detected} / ${row.total}` : `— / ${row.total}`}</td><td>{["Falla", "anomalia"].includes(row.expected) ? row.missed : "—"}</td><td>{row.fault_type ? row.type_identified : "—"}</td><td>{row.false_positives}</td><td>{row.indeterminate}</td></tr>)}</tbody></table></div><p className="fl-hint">Las lecturas sin etiqueta no cuentan como normales. Consulta las filas de Salida final para localizar las detectadas y las omitidas.</p></section>;
}
function StatisticalLabelReport({ reports, version }) {
  if (version !== "pre_suppression_v1") return <p className="fl-muted">Evaluación anterior: conteo basado en alertas emitidas. La métrica previa a supresión no está disponible.</p>;
  return <section aria-label="Detección estadística por lectura"><h3>Detección estadística por lectura, antes de suprimir alertas</h3><p>Evaluación de lecturas recibidas por el nodo. Una lectura no evaluable no confirma normalidad ni se cuenta como falla detectada.</p>{Object.entries(reports || {}).map(([nodeId, report]) => { const groups = report.groups || []; const sum = field => groups.reduce((total, row) => total + (row[field] || 0), 0); return <div key={nodeId}><h4>{nodeId}</h4><div className="fl-metrics"><span><strong>{sum("detected")}</strong>Detecciones</span><span><strong>{sum("alerted_rows")}</strong>Alertas emitidas</span><span><strong>{sum("suppressed_rows")}</strong>Lecturas suprimidas</span><span><strong>{sum("non_evaluable")}</strong>No evaluables</span></div></div>; })}</section>;
}
