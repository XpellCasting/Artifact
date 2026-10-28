import { request } from "../../shared/api.js";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ReactFlow, BaseEdge, Background, Controls, Handle, MarkerType, Position, ReactFlowProvider, useNodesState, useReactFlow } from "@xyflow/react";
import { AlertTriangle, ArrowLeft, ArrowRight, Check, Copy, GitBranch, Maximize2, Play, Plus, RotateCcw, Save, Trash2 } from "lucide-react";
import "@xyflow/react/dist/style.css";
import { NODE_LABELS, SIGNAL_LABELS, DECISIONS, SIGNALS, clone, connect, defaultNodeConfig, edgeLabel, fingerprint, flowPayload, labelNode, nextNodeId, removeNode, templates, validateDraft } from "./flow-lab-model.js";
import "./flow-lab.css";
import FutureEventsSource from "./FutureEventsSource.jsx";
import { useFlowEvaluationPolling } from "./use-flow-evaluation-polling.js";

const PRESETS = templates();
const PAGE_SIZE = 50;
const DRAFT_STORAGE_KEY = "flow-lab-selected-draft-v1";
const DESCRIPTIONS = { input: "Lecturas del histórico seleccionado", statistical_focus: "Cambios secuenciales en residuos", contextual: "Coherencia entre entidades", classifier: "Compatibilidad con patrones", merge: "Combina señales con lógica O / Y", decision: "Evalúa una condición lógica sobre la señal", output: "Resultado final por lectura" };
function restoreFlowDraft() {
  try {
    const saved = JSON.parse(localStorage.getItem(DRAFT_STORAGE_KEY) || "null");
    if (saved?.draft?.nodes && saved?.draft?.edges) {
      return { draft: clone(saved.draft), savedKey: typeof saved.savedKey === "string" ? saved.savedKey : "" };
    }
  } catch { /* Una copia local incompleta se descarta y se abre la plantilla inicial. */ }
  return { draft: clone(PRESETS[0]), savedKey: "" };
}
function FilterNode({ data, selected }) {
  const disabled = data.node.config?.enabled === false;
  const description = data.node.type === "merge" ? (data.node.config?.mode === "all" ? "Y · Todas las ramas positivas" : "O · Cualquiera de las ramas positiva") : DESCRIPTIONS[data.node.type];
  return <div className={`fl-node fl-node-${data.node.type} ${selected ? "is-selected" : ""} ${disabled ? "is-disabled" : ""}`}>
    {data.node.type !== "input" && <Handle type="target" position={Position.Left} />}
    <span className="fl-node-kind">{NODE_LABELS[data.node.type]}</span><strong>{labelNode(data.node)}</strong>
    <span className="fl-node-detail">{disabled ? "Omitido · deja pasar la entrada" : description}</span>
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

function FlowLabContent({ apiBaseUrl, onSelectHistory, historyReady, continuousModels, replay, upload, uploadBusy, onLabelledUpload }) {
  const [draft, setDraft] = useState(() => restoreFlowDraft().draft);
  const [savedKey, setSavedKey] = useState(() => restoreFlowDraft().savedKey);
  const [flows, setFlows] = useState([]);
  const [runs, setRuns] = useState([]);
  const [runsLoaded, setRunsLoaded] = useState(false);
  const [classifierModels, setClassifierModels] = useState([]);
  const [activeClassifierModelId, setActiveClassifierModelId] = useState(null);
  const [selection, setSelection] = useState(null);
  const [error, setError] = useState("");
  const [syncError, setSyncError] = useState("");
  const visibleError = error || syncError;
  const [message, setMessage] = useState("");
  const [pending, setPending] = useState(false);
  const [synchronized, setSynchronized] = useState(false);
  const [job, setJob] = useState(null);
  const executing = job?.status === "running";
  const busy = pending || !synchronized || executing;
  const [loaded, setLoaded] = useState(false);
  const [newType, setNewType] = useState("statistical_focus");
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [activeRun, setActiveRun] = useState(null);
  const [inspecting, setInspecting] = useState(false);
  const [resultNode, setResultNode] = useState("output");
  const [page, setPage] = useState({ items: [], total: 0, offset: 0, next_offset: null });
  const [resultBusy, setResultBusy] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const pageRequest = useRef(0);
  const refreshRequest = useRef(0);
  const [layoutRevision, setLayoutRevision] = useState(0);
  const [canvasNodes, setCanvasNodes, onCanvasNodesChange] = useNodesState([]);
  const { fitView, screenToFlowPosition } = useReactFlow();
  const canvasRef = useRef(null);
  const graph = executing && job.flow ? job.flow : inspecting && activeRun ? activeRun.snapshot.flow : draft;
  const dirty = fingerprint(draft) !== savedKey;
  const errors = useMemo(() => validateDraft(draft), [draft]);
  const node = selection?.kind === "node" ? graph.nodes.find((item) => item.id === selection.id) : null;
  const edge = selection?.kind === "edge" ? graph.edges.find((item) => item.id === selection.id) : null;
  useEffect(() => {
    try { localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify({ draft, savedKey })); }
    catch { /* La edición sigue disponible en memoria si el navegador no admite almacenamiento local. */ }
  }, [draft, savedKey]);
  const refresh = useCallback(async () => {
    const revision = ++refreshRequest.current;
    const current = () => revision === refreshRequest.current;
    await Promise.all([
      request(apiBaseUrl, "/flows").then((f) => { if (current()) setFlows(f.items || []); }),
      request(apiBaseUrl, "/flow-runs?compact=true").then((r) => {
        if (!current()) return;
        setRuns(r.items || []);
        setRunsLoaded(true);
        setActiveRun((active) => (r.items || []).find((item) => item.run_id === active?.run_id) || active || r.items?.[0] || null);
      }),
      request(apiBaseUrl, "/diagnostics/classifier-models").then((models) => {
        if (current()) { setClassifierModels(models.models || []); setActiveClassifierModelId(models.active_model_id || null); }
      }),
    ]);
    if (current()) setLoaded(true);
  }, [apiBaseUrl]);
  const refreshClassifierModels = useCallback(async () => {
    const models = await request(apiBaseUrl, "/diagnostics/classifier-models");
    setClassifierModels(models.models || []);
    setActiveClassifierModelId(models.active_model_id || null);
  }, [apiBaseUrl]);
  useEffect(() => {
    if (node?.type === "classifier") refreshClassifierModels().catch((reason) => setError(reason.message));
  }, [node?.id, node?.type, refreshClassifierModels]);
  useEffect(() => {
    const updateModels = () => {
      if (document.visibilityState !== "hidden") refreshClassifierModels().catch((reason) => setError(reason.message));
    };
    window.addEventListener("focus", updateModels);
    document.addEventListener("visibilitychange", updateModels);
    return () => {
      window.removeEventListener("focus", updateModels);
      document.removeEventListener("visibilitychange", updateModels);
    };
  }, [refreshClassifierModels]);
  useFlowEvaluationPolling({
    apiBaseUrl, jobId: job?.job_id, discover: true,
    onReady: () => { setSynchronized(true); setSyncError(""); },
    onRunsChanged: () => refresh().catch((reason) => setError(reason.message)),
    onUpdate: (state) => {
      if (job?.job_id !== state.job_id) { setSelection(null); setMessage(""); setError(""); setLayoutRevision((value) => value + 1); }
      setJob(state); setInspecting(false);
    },
    onConnectionError: (detail) => { if (!job) setSyncError(detail); setJob((current) => current ? { ...current, polling_error: detail } : current); },
    onFailure: (detail) => { setError(detail); setJob(null); },
    onComplete: (state) => {
      setJob(null);
      setActiveRun(state.run); setInspecting(true); setResultNode("output"); setSelection(null);
      setMessage(state.mode === "home_assistant" ? "Recepción terminada. La ejecución y sus resultados quedaron guardados." : "Evaluación terminada. El gráfico muestra el flujo utilizado con todos los eventos futuros.");
      if (state.warning) setError(state.warning);
    },
  });
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
    const id = nextNodeId(draft, newType);
    const bounds = canvasRef.current?.getBoundingClientRect();
    const position = bounds ? screenToFlowPosition({ x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2 }) : { x: 400, y: 100 };
    change((next) => next.nodes.push({ id, type: newType, position, config: defaultNodeConfig(newType) }));
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
    const { loaded_existing: loadedExisting = false, ...saved } = await request(apiBaseUrl, draft.flow_id ? `/flows/${draft.flow_id}` : "/flows", { method: draft.flow_id ? "PUT" : "POST", body: JSON.stringify(flowPayload(draft)) });
    setDraft(saved); setSavedKey(fingerprint(saved));
    if (loadedExisting) {
      setSelection(null); setSource(""); setTarget(""); setInspecting(false); setLayoutRevision((value) => value + 1);
      setMessage(`Ya existe un flujo llamado «${saved.name}». Se cargó el guardado.`);
    }
    return { flow: saved, loadedExisting };
  }
  async function prepareExecution() {
    if (!dirty && draft.flow_id) return draft;
    const saved = await persist();
    if (saved.loadedExisting) {
      setMessage(`Ya existe un flujo llamado «${saved.flow.name}». Se cargó el guardado. Revisa el diseño y vuelve a iniciar la evaluación.`);
      await refresh();
      return null;
    }
    return saved.flow;
  }
  async function save() {
    setPending(true); setError(""); setMessage("");
    try { const saved = await persist(); await refresh(); if (!saved.loadedExisting) setMessage("Flujo guardado."); } catch (reason) { setError(reason.message); } finally { setPending(false); }
  }
  async function execute() {
    setPending(true); setError(""); setMessage("");
    try {
      const saved = await prepareExecution();
      if (!saved) return;
      const started = await request(apiBaseUrl, `/flows/${saved.flow_id}/runs`, { method: "POST", body: "{}" });
      setJob({ ...started, mode: "csv", flow: saved });
    } catch (reason) { setError(reason.message); }
    finally { setPending(false); }
  }
  async function startLive(connection) {
    setPending(true); setError(""); setMessage("");
    try {
      const saved = await prepareExecution();
      if (!saved) return false;
      const started = await request(apiBaseUrl, `/flows/${saved.flow_id}/live/start`, { method: "POST", body: JSON.stringify(connection) });
      setJob({ ...started, mode: "home_assistant", flow: saved });
      return true;
    } catch (reason) { setError(reason.message); return false; }
    finally { setPending(false); }
  }
  async function stopLive() {
    setError("");
    try { setJob(await request(apiBaseUrl, `/flow-evaluations/${job.job_id}/stop`, { method: "POST" })); }
    catch (reason) { setError(reason.message); }
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
  const visualEdges = useMemo(() => graph.edges.map((item, index) => ({ ...item, data: { lane: index % 3 }, selected: selection?.id === item.id, type: "routed", markerEnd: { type: MarkerType.ArrowClosed }, label: item.when === undefined ? undefined : edgeLabel(item), labelStyle: { fontSize: 11 }, labelBgPadding: [7, 4], labelBgBorderRadius: 5, labelBgStyle: { fill: "var(--ha-surface)", fillOpacity: 1 } })), [graph, selection?.id]);
  useEffect(() => { setCanvasNodes(visualNodes); }, [visualNodes, setCanvasNodes]);

  return <section className="flow-lab" aria-label="Evaluación de flujos">
    <header className="fl-header"><div className="fl-state"><span className={`fl-dot ${executing || inspecting || !dirty ? "is-saved" : ""}`} />{executing ? "Ejecución sincronizada entre dispositivos" : inspecting ? "Inspección de una ejecución" : dirty ? "Cambios sin guardar" : "Todos los cambios guardados"}</div></header>
    {onSelectHistory && <section className="fl-history-summary" aria-label="Historial de referencia"><div><strong>{historyReady ? "Historial de referencia listo" : "Prepara tu historial de referencia"}</strong>{!historyReady && <p>Elige el historial de Home Assistant o un archivo CSV en el Selector de Historial.</p>}</div><button type="button" disabled={busy} onClick={onSelectHistory}>Selector de Historial<ArrowRight size={16} /></button></section>}
    {visibleError && <div role="alert" className="fl-notice fl-error"><AlertTriangle size={18} /><span>{visibleError}</span>{!loaded && <button onClick={() => { setError(""); refresh().catch((e) => setError(e.message)); }}>Reintentar</button>}</div>}
    {message && <div role="status" className="fl-notice"><Check size={16} />{message}</div>}
    <div className="fl-design-heading"><span className="hs-step">Diseño del flujo</span></div>
    <div className="fl-toolbar">{executing ? <><strong>{graph.name}</strong><span className="fl-muted">Evaluación en curso · configuración de la ejecución</span></> : inspecting ? <>
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
        <Background gap={22} size={1} color="var(--ha-divider)" /><Controls showInteractive={false} />
      </ReactFlow></div>
      <p className="fl-canvas-caption">{inspecting ? "Selecciona un nodo para consultar sus resultados." : "Arrastra los nodos. Conecta sus puntos laterales. Usa la rueda para acercar y arrastra el fondo para desplazarte."}</p>
    </div><aside className="fl-inspector" aria-label="Inspector del flujo"><h3>{node ? labelNode(node) : edge ? "Configurar conexión" : "Inspector"}</h3>
      {node ? <fieldset disabled={busy || inspecting} className="fl-fields"><span className="fl-muted">{DESCRIPTIONS[node.type]}</span><label>Nombre visible<input value={node.config?.label || ""} placeholder={NODE_LABELS[node.type]} onChange={(event) => configure({ label: event.target.value })} /></label>
        {["statistical_focus", "contextual", "classifier"].includes(node.type) && <label className="fl-check"><input type="checkbox" checked={node.config?.enabled !== false} onChange={(event) => configure({ enabled: event.target.checked })} />Filtro activado</label>}
        {["merge", "decision"].includes(node.type) ? <p className="fl-hint">El resultado lógico determina la señal enviada. Las evidencias de los filtros se conservan para consultar su diagnóstico.</p> : <label>Uso de evidencias<select value={node.config?.evidence_policy || "legacy"} onChange={(event) => configure({ evidence_policy: event.target.value })}><option value="legacy">Comportamiento guardado</option><option value="independent">Evaluación propia</option><option value="combine_v1">Evaluación propia y combinada</option></select><span className="fl-hint">Recibe las evidencias de sus predecesores y conserva su procedencia.</span></label>}{node.type === "statistical_focus" && <><label>Umbral para binarios y multivalor<input type="number" min="0.01" max="0.99" step="0.01" value={node.config?.suspicion_threshold ?? .6} onChange={(event) => configure({ suspicion_threshold: Number(event.target.value) })} /></label><p><strong>Continuos:</strong> FOCuS sobre residuos normalizados</p><span className="fl-hint">Combina FOCuS con canales puntual, energía y constancia. Un aviso puede originarse en cualquiera de ellos; no confirma un tipo de falla.</span></>}
        {node.type === "classifier" && <label>Modelo<select aria-label="Modelo del clasificador" value={node.config?.model_id || ""} onChange={(event) => configure({ model_id: event.target.value || null })}>{node.config?.model_id && !classifierModels.some((item) => item.model_id === node.config.model_id) && <option value={node.config.model_id} disabled>Versión no disponible · selecciona otro modelo</option>}<option value="">Modelo activo{activeClassifierModelId ? ` · ${activeClassifierModelId.slice(0, 12)}` : " · ninguno"}</option>{classifierModels.map((model) => <option key={model.model_id} value={model.model_id}>{model.architecture_version === "gat_lstm_v3_segments_multiscale" ? "GAT-LSTM segmentos" : "GAT-LSTM v2"} · {model.model_id.slice(0, 12)}{model.active ? " (activo)" : ""}</option>)}</select><span className="fl-hint">La elección queda guardada en este nodo. “Modelo activo” usa el seleccionado en Diagnóstico.</span></label>}
        {node.type === "merge" && <label>Función lógica<select value={node.config?.mode || "any"} onChange={(event) => configure({ mode: event.target.value })}><option value="any">Cualquiera positiva (O)</option><option value="all">Todas positivas (Y)</option></select><span className="fl-hint">Combina los resultados de una misma lectura. Una rama ausente cuenta como no evaluable. Puedes encadenar uniones para comparar fórmulas como (A O B) Y C.</span></label>}
        {node.type === "decision" && <><Signals legend="Resultado verdadero cuando recibe" value={node.config?.accept || ["positive"]} onChange={(accept) => configure({ accept })} /><p className="fl-hint">Verdadero produce señal positiva; falso, sin señal. Selecciona solo «Sin señal» para negar una señal evaluable (NO). Una entrada no evaluable permanece desconocida salvo que la selecciones explícitamente. Configura en cada conexión qué resultados pueden pasar.</p></>}
        {!["input", "output"].includes(node.type) && <button className="fl-danger" onClick={deleteSelection}><Trash2 size={15} />Eliminar nodo y conexiones</button>}
      </fieldset> : edge ? <fieldset disabled={busy || inspecting} className="fl-fields"><p className="fl-route">{labelNode(graph.nodes.find((item) => item.id === edge.source))}<ArrowRight size={15} />{labelNode(graph.nodes.find((item) => item.id === edge.target))}</p><Signals value={edge.when || SIGNALS} onChange={(when) => change((next) => { next.edges.find((item) => item.id === edge.id).when = when; })} /><button className="fl-danger" onClick={deleteSelection}><Trash2 size={15} />Eliminar conexión</button></fieldset> : <p className="fl-muted">Selecciona un filtro o una conexión para ver y editar sus propiedades.</p>}
      {!inspecting && <details className="fl-connect"><summary>Conectar mediante formulario</summary><fieldset disabled={busy} className="fl-fields"><label>Desde<select aria-label="Origen de conexión" value={source} onChange={(event) => setSource(event.target.value)}><option value="">Seleccionar…</option>{draft.nodes.filter((item) => item.type !== "output").map((item) => <option key={item.id} value={item.id}>{labelNode(item)}</option>)}</select></label><label>Hacia<select aria-label="Destino de conexión" value={target} onChange={(event) => setTarget(event.target.value)}><option value="">Seleccionar…</option>{draft.nodes.filter((item) => item.type !== "input").map((item) => <option key={item.id} value={item.id}>{labelNode(item)}</option>)}</select></label><button disabled={!source || !target} onClick={() => addConnection({ source, target })}><GitBranch size={15} />Conectar</button></fieldset></details>}
    </aside></div>
    {!inspecting && <>{!!errors.length && <div className="fl-validation" role="status"><strong>Completa el diseño para ejecutar</strong><ul>{errors.map((text) => <li key={text}>{text}</li>)}</ul></div>}
      <section className="fl-execution"><div><span className="hs-step">Evaluación</span>
        <FutureEventsSource apiBaseUrl={apiBaseUrl} busy={busy} canRun={loaded && historyReady && !uploadBusy && !errors.length} canUpload={historyReady && !uploadBusy} dirty={dirty || !draft.flow_id} job={job} error={visibleError} replay={replay} upload={upload} onLabelledUpload={onLabelledUpload} onEvaluate={execute} onStartLive={startLive} onStopLive={stopLive} />
        {!historyReady && <p className="fl-required">Carga un historial desde el Selector de Historial y espera a que se prepare el detector.</p>}
        <CalibrationReport models={continuousModels} />
        {job && job.mode !== "home_assistant" && <>{job.polling_error && <p className="fl-required" role="status">No se pudo consultar el progreso. Se reintentará automáticamente. {job.polling_error}</p>}<FlowEvaluationProgress job={job} fallbackTotal={replay?.rows} /></>}
      </div></section>
    </>}
    <section className="fl-comparison" aria-label="Comparación de conjuntos etiquetados"><h3>Conjuntos etiquetados evaluados</h3><div className="fl-table-scroll"><table><thead><tr><th>Conjunto · huella</th><th>Flujo · fecha</th><th>Fallas detectadas / total</th><th>Anomalías detectadas / total</th><th>Tipos identificados / etiquetados</th><th>Falsos positivos</th><th>Detalle</th></tr></thead><tbody>{runs.filter(r => r.summary.evaluation?.labelled).map(run => {
      const groups = run.summary.evaluation.groups;
      const sum = (kind, field) => groups.filter(g => !kind || g.expected === kind).reduce((total, g) => total + (g[field] || 0), 0);
      return <tr key={run.run_id}><td>{run.snapshot.source.filename}<small className="fl-muted"> · {run.snapshot.source.fingerprint.slice(0, 10)}</small></td><td>{run.snapshot.flow.name} · {new Date(run.created_at).toLocaleString()}</td><td>{sum("Falla", "detected")} / {sum("Falla", "total")}</td><td>{sum("anomalia", "detected")} / {sum("anomalia", "total")}</td><td>{sum("Falla", "type_identified")} / {groups.filter(g => g.expected === "Falla" && g.fault_type).reduce((n, g) => n + g.total, 0)}</td><td>{sum(null, "false_positives")}</td><td><button disabled={busy} onClick={() => { setActiveRun(run); setInspecting(true); setResultNode("output"); setSelection(null); }}>Ver resultados</button></td></tr>;
    })}</tbody></table></div>{!runs.some(r => r.summary.evaluation?.labelled) && <p className="fl-muted">Carga un conjunto etiquetado y evalúa un flujo para comenzar la comparación.</p>}</section>
    <section className="fl-results" aria-label="Resultados de flujos"><div className="fl-results-toolbar"><h3>Ejecuciones</h3><label>Consultar<select aria-label="Ejecución guardada" value={activeRun?.run_id || ""} disabled={busy} onChange={(event) => { const run = runs.find((item) => item.run_id === event.target.value); if (run) { setActiveRun(run); setInspecting(true); setResultNode("output"); setSelection(null); setError(""); setMessage(""); setLayoutRevision((value) => value + 1); } }}><option value="">Seleccionar ejecución…</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.snapshot?.flow?.name} · {run.snapshot.source?.filename || "Histórico"} · {new Date(run.created_at).toLocaleString()}</option>)}</select></label><button disabled={busy} aria-label="Actualizar ejecuciones" onClick={() => refresh().catch((e) => setError(e.message))}><RotateCcw size={16} /></button></div>
      {!activeRun ? <div className="fl-empty"><GitBranch size={28} /><strong>{runsLoaded ? "Tu primera ejecución aparecerá aquí" : "Cargando ejecuciones guardadas…"}</strong><p>{runsLoaded ? "Guarda un flujo y ejecútalo para revisar el recorrido y la evidencia." : "Consultando las pruebas realizadas."}</p></div> : <>
        <p><strong>{activeRun.snapshot.source?.filename || "Ejecución anterior"}</strong> · {activeRun.snapshot.execution_mode === "causal_live" ? "Eventos de Home Assistant · evaluación en directo" : activeRun.snapshot.execution_mode === "causal_future" ? "Eventos futuros · evaluación causal" : "Análisis retrospectivo anterior"} · {activeRun.snapshot.baseline_id ? `Base ${activeRun.snapshot.baseline_id.slice(0, 16)}` : "Base sin procedencia registrada"}</p>
        <CalibrationReport models={Object.assign({}, ...Object.values(activeRun.snapshot.baseline_provenance?.continuous_models || {}))} />
        {activeRun.snapshot.baseline_provenance?.continuous_models && <p className="fl-muted">Entrenamiento completado · {Object.values(activeRun.snapshot.baseline_provenance.continuous_models).flatMap((models) => Object.values(models || {})).filter((model) => model.eligible).length} sensores preparados · {Object.values(activeRun.snapshot.baseline_provenance.continuous_models).flatMap((models) => Object.values(models || {})).filter((model) => !model.eligible).length} no evaluables</p>}
        <div className="fl-metrics"><span><strong>{activeRun.summary.candidate_count}</strong>Lecturas de entrada</span><span><strong>{activeRun.summary.output_count}</strong>Resultados finales</span><span><strong>{Number(activeRun.summary.elapsed_ms).toLocaleString()} ms</strong>Tiempo total</span>{Object.entries(activeRun.summary.by_final_decision || {}).map(([key, value]) => <span key={key}><strong>{value}</strong>{DECISIONS[key] || key}</span>)}</div>
        <FlowLabelReport report={activeRun.summary.evaluation} />
        <StatisticalLabelReport reports={activeRun.summary.statistical_evaluation} version={activeRun.summary.statistical_evaluation_version} />
        <div className="fl-result-controls"><label>Resultados del nodo<select aria-label="Nodo de resultados" value={resultNode} onChange={(event) => setResultNode(event.target.value)}><option value="output">Salida final</option>{activeRun.snapshot.flow.nodes.filter((item) => item.id !== "output").map((item) => <option key={item.id} value={item.id}>{labelNode(item)}</option>)}</select></label><span className="fl-muted">{resultBusy ? "Cargando resultados…" : `${page.total ? page.offset + 1 : 0}–${page.offset + page.items.length} de ${page.total}`}</span><button disabled={resultBusy || page.offset === 0} onClick={() => loadPage(activeRun.run_id, resultNode, Math.max(0, page.offset - PAGE_SIZE))}><ArrowLeft size={15} />Anterior</button><button disabled={resultBusy || page.next_offset === null} onClick={() => loadPage(activeRun.run_id, resultNode, page.next_offset)}>Siguiente<ArrowRight size={15} /></button></div>
        <div className="fl-table-scroll"><table><thead><tr><th>Entidad</th><th>Fecha y hora UTC</th><th>Resultado</th><th>Etiqueta y detección</th><th>Evidencia</th></tr></thead><tbody>{page.items.map((item) => <tr key={item.observation_id}><td>{item.entity_id}</td><td>{item.timestamp}</td><td>{DECISIONS[item.final_decision] || SIGNAL_LABELS[item.signal] || "No evaluable"}</td><td>{item.evaluation ? `${item.evaluation.expected} · ${EVALUATION_LABELS[item.evaluation.outcome]}${item.evaluation.type_identified ? " · Tipo identificado" : ""}` : "—"}</td><td><button onClick={() => setSelectedResult(item)}>Ver {item.evidence?.length || 0} etapas</button></td></tr>)}</tbody></table>{!resultBusy && !page.items.length && <p className="fl-empty-row">Este nodo no recibió lecturas en la ejecución.</p>}</div>
        {selectedResult && <div className="fl-evidence"><h4>Recorrido de {selectedResult.entity_id}</h4>
          {selectedResult.logical_evaluation && <p>Resultado lógico del flujo: {SIGNAL_LABELS[selectedResult.logical_evaluation.signal]}</p>}
          {selectedResult.own_evaluation && <p>Evaluación propia: {selectedResult.own_evaluation.label} · Combinada: {selectedResult.combined_evaluation?.label}{selectedResult.combined_evaluation?.conflict ? " · Conflicto entre evidencias" : ""}</p>}
          <p className="fl-muted">{selectedResult.timestamp}</p>{(selectedResult.evidence || []).map((entry, index) => <details key={`${entry.node_id}-${index}`}><summary>{labelNode(activeRun.snapshot.flow.nodes.find((item) => item.id === entry.node_id))} · {SIGNAL_LABELS[entry.signal]} · {entry.status === "omitted" ? "Filtro omitido" : entry.status}</summary><pre>{JSON.stringify({ propia: entry.own_evaluation, evidencias_utilizadas: entry.used_evidence_ids, datos: entry.data }, null, 2)}</pre></details>)}
        </div>}
      </>}
    </section>
  </section>;
}
export default function FlowLab(props) { return <ReactFlowProvider><FlowLabContent {...props} /></ReactFlowProvider>; }
const EVALUATION_LABELS = { detectado: "Detectado", no_detectado: "No detectado", falso_positivo: "Falso positivo", normal_correcto: "Normal correcto", no_evaluable: "No evaluable", sin_etiqueta: "Sin etiqueta" };
export function FlowLabelReport({ report }) {
  if (!report?.labelled) return <p className="fl-muted">Esta ejecución no tiene etiquetas de referencia para medir detecciones y errores.</p>;
  return <section aria-label="Detección de datos etiquetados"><h3>Qué identifica este flujo</h3><p>Se cuenta como detección una sospecha, una falla compatible o una anomalía ambiental en la salida final. Identificar el tipo exige que el diagnóstico coincida con tipo_falla; una sospecha sola no identifica el tipo.</p><div className="fl-table-scroll"><table><thead><tr><th>Entidad</th><th>Etiqueta</th><th>Origen</th><th>Tipo esperado</th><th>Detectadas / total</th><th>Evaluadas sin sospecha</th><th>Tipo identificado</th><th>Falsos positivos (lecturas)</th><th>Sin evaluación / sin conclusión</th></tr></thead><tbody>{report.groups.map((row, i) => <tr key={i}><td>{row.entity_id}</td><td>{row.expected}</td><td>{row.synthetic ? "Inyectado" : "No inyectado"}</td><td>{row.fault_type || "—"}</td><td>{["Falla", "anomalia"].includes(row.expected) ? `${row.detected} / ${row.total}` : `— / ${row.total}`}</td><td>{["Falla", "anomalia"].includes(row.expected) ? (row.missed_evaluable ?? Math.max(0, row.missed - row.indeterminate)) : "—"}</td><td>{row.fault_type ? row.type_identified : "—"}</td><td>{row.false_positives}</td><td>{row.indeterminate}</td></tr>)}</tbody></table></div><p className="fl-hint">Los falsos positivos cuentan lecturas, incluidas las repeticiones con alerta suprimida. El detalle estadístico separa las alertas emitidas. Las lecturas sin etiqueta no cuentan como normales. Los casos sin evaluación o sin conclusión se muestran aparte de las lecturas evaluadas sin sospecha. Consulta la Salida final para revisar el motivo.</p></section>;
}
export function StatisticalLabelReport({ reports, version }) {
  if (version !== "pre_suppression_v1") return <p className="fl-muted">Evaluación anterior: conteo basado en alertas emitidas. La métrica previa a supresión no está disponible.</p>;
  return <section aria-label="Detección estadística por lectura">
    <h3>Detección estadística por lectura, antes de suprimir alertas</h3>
    <p>Cada lectura sospechosa cuenta, aunque su alerta se suprima por repetición. Los falsos positivos por lectura y las alertas emitidas sobre lecturas normales se muestran por separado. Una lectura no evaluable no confirma normalidad.</p>
    {Object.entries(reports || {}).map(([nodeId, report]) => {
      const groups = report.groups || [];
      const sum = field => groups.reduce((total, row) => total + (row[field] || 0), 0);
      return <div key={nodeId}><h4>{nodeId}</h4>
        <div className="fl-metrics"><span><strong>{sum("detected")}</strong>Detecciones</span><span><strong>{sum("alerted_rows")}</strong>Alertas emitidas</span><span><strong>{sum("suppressed_rows")}</strong>Lecturas suprimidas</span><span><strong>{sum("non_evaluable")}</strong>No evaluables</span></div>
        <div className="fl-table-scroll"><table><thead><tr>
          <th>Entidad</th><th>Etiqueta / tipo</th><th>Origen</th><th>Evaluadas / recibidas</th><th>Falsos positivos: lecturas</th><th>Falsos positivos: alertas emitidas</th><th>Alertas emitidas</th><th>Lecturas suprimidas</th>
        </tr></thead><tbody>{groups.map((row, i) => <tr key={i}>
          <td>{row.entity_id}</td><td>{row.expected}{row.fault_type ? ` / ${row.fault_type}` : ""}</td><td>{row.synthetic ? "Inyectado" : "No inyectado"}</td>
          <td>{row.evaluable} / {row.total}</td><td>{row.false_positives}</td>
          <td>{row.expected === "normal" ? (row.false_positive_alerts ?? row.alerted_rows ?? "—") : "—"}</td>
          <td>{row.alerted_rows ?? "—"}</td><td>{row.suppressed_rows ?? "—"}</td>
        </tr>)}</tbody></table></div>
      </div>;
    })}
  </section>;
}

const CALIBRATION_REASONS = {
  insufficient_training_data: "Lecturas insuficientes para entrenar",
  insufficient_calendar_blocks: "Período insuficiente para separar entrenamiento y calibración",
  insufficient_residual_partition: "Lecturas insuficientes en una partición",
  insufficient_threshold_days: "Cobertura insuficiente para calibrar el umbral",
  calibration_budget_unmet: "No se pudo calibrar el límite de alertas",
};
export function CalibrationReport({ models }) {
  const entries = Object.entries(models || {});
  if (!entries.length) return null;
  const number = value => Number.isFinite(value) ? value.toFixed(2) : "—";
  return <details className="fl-calibration"><summary>Cobertura del detector · {entries.filter(([, model]) => model.eligible).length} de {entries.length} sensores continuos preparados</summary>
    <p>Particiones por bloques de 24 horas desde el inicio de cada serie. El fragmento final se conserva con su duración real y la fecha del último valor. El silencio de un sensor que reporta cambios no demuestra desconexión.</p>
    <div className="fl-table-scroll"><table><thead><tr><th>Entidad</th><th>Estado</th><th>Reporte</th><th>Días de calibración</th><th>Días de cobertura</th><th>Fragmento final (h)</th><th>Marcadas en calibración</th><th>Episodios/día en calibración</th><th>Último valor recibido</th></tr></thead><tbody>{entries.map(([entity, model]) => <tr key={entity}><td>{entity}</td><td>{model.eligible ? "Preparado" : CALIBRATION_REASONS[model.reason] || model.reason || "Sin referencia"}</td><td>{{ on_change: "Por cambios", periodic: "Periódico", unknown: "Sin declarar" }[model.reporting_policy?.mode] || "Sin declarar"}</td><td>{number(model.coverage?.calendar_days ?? model.partitions?.threshold_calendar_days)}</td><td>{number(model.coverage?.exposure_days ?? model.partitions?.threshold_days)}</td><td>{number(model.partitions?.final_partial_block_hours)}</td><td>{model.joint_calibration ? `${number(model.joint_calibration.flagged_row_rate * 100)} %` : "—"}</td><td>{number(model.joint_calibration?.episodes_per_day)}</td><td>{model.last_observed_value ?? "—"} · {model.last_observation_at || "Fecha sin registrar"}</td></tr>)}</tbody></table></div>
    <p>FOCuS reinicia su acumulación tras una detección de los canales residuales. La calibración utiliza el mismo reinicio y límites conjuntos de lecturas marcadas y episodios por día. La cobertura por cambios supone que el registro está completo; no confirma la conectividad. Los reportes periódicos requieren un intervalo máximo declarado.</p>
  </details>;
}
