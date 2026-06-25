import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  X,
  Database,
  Eye,
  FlaskConical,
  Loader2,
  Play,
  RotateCcw,
  Zap,
} from "lucide-react";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const RED_SUSPICION_SCORE = 0.60;
const MARKER_GROUP_GAP_MS = 15 * 60 * 1000;

const CONTINUOUS_FAULT_TYPES = ["random/spike", "malfunction", "bias", "drift", "missing"];
const BINARY_FAULT_TYPES = ["random", "malfunction", "bias", "missing"];

const EQUATIONS = [
  {
    id: "hampel",
    title: "Filtro Hampel",
    pattern: "isolated_spike",
    equation: "H_i(t) = |x_i(t) - mediana(W)| / (1.4826 * MAD(W) + epsilon)",
    score: "s_H = min(1, H_i(t) / 3.5)",
  },
  {
    id: "robust_rate_change",
    title: "Tasa de cambio robusta",
    pattern: "abrupt_change",
    equation: "R_i(t) = |x_i(t) - x_i(t-Delta t)| / Delta t",
    score: "s_R = min(1, Z_R_i(t) / 3.5)",
  },
  {
    id: "cusum_residual",
    title: "CUSUM sobre residuales",
    pattern: "persistent_shift",
    equation: "r_i(t) = x_i(t) - EWMA_i(t-Delta t); C+ y C- acumulan residuales",
    score: "s_C = min(1, max(C+, C-) / theta_C)",
  },
  {
    id: "constant_persistence",
    title: "Persistencia constante",
    pattern: "stuck_value",
    equation: "D_const mide el tiempo consecutivo con |x_i(t)-x_i(t-Delta t)| <= epsilon_const",
    score: "s_D_const = min(1, Z_D_const / 3.5)",
  },
  {
    id: "reading_gap",
    title: "Brecha temporal",
    pattern: "missing_gap",
    equation: "M_i(t) = t - t_i_last",
    score: "s_M = min(1, Z_M / 3.5)",
  },
];

const STATE_EQUATIONS = [
  {
    id: "activation_count",
    title: "Activaciones",
    pattern: "activation_burst",
    equation: "A_i(W) = sum 1[b_i(t_j-1)=0 and b_i(t_j)=1]",
    score: "s_A = min(1, Z_A / tau_A)",
  },
  {
    id: "state_change_count",
    title: "Cambios de estado",
    pattern: "activation_burst",
    equation: "C_i(W) = sum 1[b_i(t_j-1) != b_i(t_j)]",
    score: "s_C = min(1, Z_C / tau_C)",
  },
  {
    id: "active_time_ratio",
    title: "Proporcion activa",
    pattern: "state_persistence",
    equation: "P_on,i(W) = T_on,i(W) / |W|",
    score: "s_P = min(1, Z_P / tau_P)",
  },
  {
    id: "time_since_last_activation",
    title: "Tiempo desde ultima activacion",
    pattern: "absence_gap",
    equation: "G_i(t) = t - max{tau < t | b_i(tau-dt)=0 and b_i(tau)=1}",
    score: "s_G = min(1, Z_G / tau_G)",
  },
  {
    id: "report_gap",
    title: "Brecha temporal de reporte",
    pattern: "missing_gap",
    equation: "M_i(t) = t - t_i_last",
    score: "s_M = min(1, Z_M / tau_M)",
  },
];

const ALL_EQUATIONS = [...EQUATIONS, ...STATE_EQUATIONS];

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function App() {
  const [health, setHealth] = useState(null);
  const [historyStatus, setHistoryStatus] = useState(null);
  const [entities, setEntities] = useState([]);
  const [selectedEntityIds, setSelectedEntityIds] = useState([]);
  const [faults, setFaults] = useState([]);
  const [timelines, setTimelines] = useState([]);
  const [historicalResults, setHistoricalResults] = useState([]);
  const [realtime, setRealtime] = useState(null);
  const [runSummary, setRunSummary] = useState(null);
  const [tickSeconds, setTickSeconds] = useState(60);
  const [iterationCount, setIterationCount] = useState(10);
  const [iterationsDone, setIterationsDone] = useState(0);
  const [suspicionThreshold, setSuspicionThreshold] = useState(RED_SUSPICION_SCORE);
  const [chartStart, setChartStart] = useState("");
  const [chartEnd, setChartEnd] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [isIterating, setIsIterating] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const [normalizationPreview, setNormalizationPreview] = useState(null);
  const [activeModule, setActiveModule] = useState("estadistico");
  const [resultEntityFilterIds, setResultEntityFilterIds] = useState(null);

  const continuousEntities = useMemo(
    () => entities.filter((entity) => entity.sensor_kind === "continuous"),
    [entities],
  );

  const binaryEntities = useMemo(
    () => entities.filter((entity) => entity.sensor_kind === "binary"),
    [entities],
  );

  const analyzableEntities = useMemo(
    () => entities.filter((entity) => ["continuous", "binary"].includes(entity.sensor_kind)),
    [entities],
  );

  const selectedAnalyzable = useMemo(
    () => selectedEntityIds.filter((entityId) => analyzableEntities.some((entity) => entity.entity_id === entityId)),
    [selectedEntityIds, analyzableEntities],
  );

  const allAnalyzedResults = useMemo(() => {
    return [...historicalResults, ...(realtime?.results || [])]
      .filter((result) => ["continuous", "binary"].includes(result.sensor_kind));
  }, [historicalResults, realtime]);

  const resultEntityOptions = useMemo(() => {
    return [...new Set(allAnalyzedResults.map((result) => result.entity_id).filter(Boolean))]
      .sort((left, right) => left.localeCompare(right));
  }, [allAnalyzedResults]);

  const filteredAnalyzedResults = useMemo(() => {
    if (resultEntityFilterIds === null) {
      return allAnalyzedResults;
    }
    const selected = new Set(resultEntityFilterIds);
    return allAnalyzedResults.filter((result) => selected.has(result.entity_id));
  }, [allAnalyzedResults, resultEntityFilterIds]);

  const displayResults = useMemo(() => {
    return [...filteredAnalyzedResults]
      .sort(compareResultsByExecutionDate)
      .slice(0, 1000);
  }, [filteredAnalyzedResults]);

  const suspiciousCount = allAnalyzedResults.filter(isRedSuspicion).length;
  const activeFaults = faults.filter((fault) => fault.enabled && selectedAnalyzable.includes(fault.entity_id));

  useEffect(() => {
    runStep("init", async () => {
      setHealth(await api("/health"));
      const status = await api("/history/static/status");
      applyHistoryStatus(status, { keepSelection: false });
    });
  }, []);

  async function runStep(label, action) {
    setBusy(label);
    setError("");
    try {
      return await action();
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setBusy("");
    }
  }

  function applyHistoryStatus(status, { keepSelection = false } = {}) {
    setHistoryStatus(status);
    const nextEntities = status?.entities || [];
    setEntities(nextEntities);
    const nextAnalyzable = nextEntities.filter((entity) => ["continuous", "binary"].includes(entity.sensor_kind));
    if (!keepSelection) {
      setSelectedEntityIds(nextAnalyzable.map((entity) => entity.entity_id));
    }
    setFaults((current) => syncFaultRows(nextAnalyzable, current));
  }

  async function loadStaticHistory() {
    await runStep("static-history", async () => {
      const status = await api("/history/static/ensure", { method: "POST" });
      applyHistoryStatus(status);
      setHistoricalResults([]);
      setRealtime(null);
      setChartStart("");
      setChartEnd("");
      await loadTimeline("", "");
    });
  }

  async function loadRealHistory() {
    await runStep("real-history", async () => {
      const status = await api("/history/real/load", { method: "POST" });
      applyHistoryStatus(status);
      const analysis = await api("/history/analyze", {
        method: "POST",
        body: JSON.stringify({ suspicion_threshold: suspicionThreshold }),
      });
      setHistoricalResults((analysis.results || []).filter((result) => ["continuous", "binary"].includes(result.sensor_kind)));
      setRunSummary(analysis.summary || status.historical_summary || {});
      setRealtime(null);
      setChartStart("");
      setChartEnd("");
      await loadTimeline("", "");
    });
  }

  async function analyzeHistory() {
    await runStep("analyze-history", async () => {
      const payload = await api("/history/analyze", {
        method: "POST",
        body: JSON.stringify({ suspicion_threshold: suspicionThreshold }),
      });
      setHistoricalResults((payload.results || []).filter((result) => ["continuous", "binary"].includes(result.sensor_kind)));
      setRunSummary(payload.summary || {});
    });
  }

  async function openNormalizationPreview() {
    await runStep("normalization-preview", async () => {
      const payload = await api("/history/normalization/preview");
      setNormalizationPreview(payload);
    });
  }

  async function resetRealtime() {
    await runStep("reset", async () => {
      const payload = await api("/realtime/reset", { method: "POST" });
      setRealtime(payload);
      setRunSummary(null);
      setIterationsDone(0);
      await loadTimeline(chartStart, chartEnd);
    });
  }

  async function generateOneTick({ quiet = false } = {}) {
    const action = async () => {
      if (!historyStatus?.exists || historyStatus?.valid === false) {
        const status = await api("/history/static/ensure", { method: "POST" });
        applyHistoryStatus(status, { keepSelection: true });
      }
      const payload = await api("/realtime/tick", {
        method: "POST",
        body: JSON.stringify({
          selected_entities: selectedAnalyzable,
          tick_seconds: tickSeconds,
          faults: faults.map((fault) => ({
            ...fault,
            enabled: fault.enabled && selectedAnalyzable.includes(fault.entity_id),
          })),
          suspicion_threshold: suspicionThreshold,
        }),
      });
      setRealtime(payload);
      setRunSummary(payload.summary || {});
      await loadTimeline(chartStart, chartEnd);
      return payload;
    };
    if (quiet) {
      return action();
    }
    return runStep("tick", action);
  }

  async function runIterations() {
    if (!selectedAnalyzable.length) {
      setError("Selecciona al menos una entidad continua o binaria para generar iteraciones.");
      return;
    }
    setIsIterating(true);
    setBusy("iterations");
    setError("");
    setIterationsDone(0);
    try {
      const target = Math.max(1, Number(iterationCount) || 1);
      const payload = await api("/realtime/iterations", {
        method: "POST",
        body: JSON.stringify({
          selected_entities: selectedAnalyzable,
          tick_seconds: tickSeconds,
          iteration_count: target,
          faults: faults.map((fault) => ({
            ...fault,
            enabled: fault.enabled && selectedAnalyzable.includes(fault.entity_id),
          })),
          suspicion_threshold: suspicionThreshold,
        }),
      });
      setRealtime(payload);
      setRunSummary(payload.summary || {});
      setIterationsDone(payload.summary?.iterations_done || target);
      await loadTimeline(chartStart, chartEnd);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsIterating(false);
      setBusy("");
    }
  }

  async function loadTimeline(startValue = chartStart, endValue = chartEnd) {
    const payload = await api(`/realtime/timeline${timelineQueryString(startValue, endValue)}`);
    const nextTimelines = (payload.entities || []).filter((timeline) => ["continuous", "binary"].includes(timeline.sensor_kind));
    setTimelines(nextTimelines);
    if (!startValue || !endValue) {
      const range = timelineDateRange(nextTimelines);
      setChartStart(range.start);
      setChartEnd(range.end);
    }
  }

  function toggleEntity(entityId) {
    setSelectedEntityIds((current) =>
      current.includes(entityId)
        ? current.filter((item) => item !== entityId)
        : [...current, entityId],
    );
  }

  function updateFault(entityId, patch) {
    setFaults((current) =>
      current.map((fault) => (fault.entity_id === entityId ? { ...fault, ...patch } : fault)),
    );
  }

  function clearFaults() {
    setFaults((current) => current.map((fault) => ({ ...fault, enabled: false })));
  }

  function toggleResultEntityFilter(entityId) {
    setResultEntityFilterIds((current) => {
      const next = new Set(current === null ? resultEntityOptions : current);
      if (next.has(entityId)) {
        next.delete(entityId);
      } else {
        next.add(entityId);
      }
      return resultEntityOptions.filter((option) => next.has(option));
    });
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <span className="eyebrow">Artefacto de tesis</span>
          <h1>Filtro estadistico-temporal para entidades continuas y de estado</h1>
          <p>
            Flujo actual: cargar historico, seleccionar sensores continuos o binarios, inyectar fallas,
            generar iteraciones de tiempo real y revisar sospechas con sus ecuaciones.
          </p>
        </div>
        <div className="header-actions">
          <button className="primary-action" onClick={() => generateOneTick()} disabled={Boolean(busy) || !selectedAnalyzable.length}>
            {busy === "tick" ? <Loader2 className="spin" size={18} /> : <Zap size={18} />}
            Generar tick
          </button>
          {isIterating ? (
            <button className="secondary-action" disabled>
              <Loader2 className="spin" size={18} />
              Ejecutando
            </button>
          ) : (
            <button className="secondary-action" onClick={runIterations} disabled={Boolean(busy) || !selectedAnalyzable.length}>
              <Play size={18} />
              Ejecutar iteraciones
            </button>
          )}
        </div>
      </header>

      {error && (
        <section className="alert">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </section>
      )}

      <nav className="module-tabs" aria-label="Filtros del artefacto">
        <ModuleTab
          active={activeModule === "estadistico"}
          icon={<BarChart3 size={17} />}
          label="Estadistico"
          onClick={() => setActiveModule("estadistico")}
        />
        <ModuleTab
          active={activeModule === "contextual"}
          icon={<Activity size={17} />}
          label="Contextual"
          onClick={() => setActiveModule("contextual")}
        />
        <ModuleTab
          active={activeModule === "clasificador"}
          icon={<FlaskConical size={17} />}
          label="Clasificador"
          onClick={() => setActiveModule("clasificador")}
        />
      </nav>

      {activeModule === "estadistico" ? (
        <>
          <section className="metric-grid">
            <Metric icon={<Database />} label="Datos historicos" value={formatCount(historyStatus?.rows)} detail={formatHistoryRange(historyStatus)} />
            <Metric icon={<Activity />} label="Entidades historico" value={formatCount(historyStatus?.entity_count || entities.length)} detail={`${continuousEntities.length} continuas / ${binaryEntities.length} binarias`} />
            <Metric icon={<Database />} label="Base normalidad" value={formatCount(historyStatus?.normality_history?.rows)} detail={historyStatus?.normality_history?.exists ? "historico secundario" : "sin generar"} />
            <Metric icon={<Zap />} label="Estado realtime" value={formatCount(realtime?.events)} detail={`${realtime?.tick_index || 0} ticks generados`} />
            <Metric icon={<FlaskConical />} label="Fallas activas" value={activeFaults.length} detail={activeFaults.map((fault) => fault.fault_type).join(", ") || "sin fallas"} />
            <Metric icon={<BarChart3 />} label="Sospechas estadisticas" value={suspiciousCount} detail={`rojo score >= ${RED_SUSPICION_SCORE.toFixed(2)}`} />
          </section>

          <section className="workspace-grid">
            <Panel title="Historico">
              <div className="history-summary">
                <SummaryRow label="API" value={health?.status || "-"} />
                <SummaryRow label="Origen" value={historyStatus?.source_type || "-"} />
                <SummaryRow label="Filas" value={formatCount(historyStatus?.rows)} />
                <SummaryRow label="Filas normalidad" value={formatCount(historyStatus?.normality_history?.rows)} />
                <SummaryRow label="Entidades" value={formatCount(historyStatus?.entity_count || entities.length)} />
                <SummaryRow label="Rango" value={formatHistoryRange(historyStatus)} />
              </div>
              <div className="button-row">
                <StepButton id="real-history" busy={busy} onClick={loadRealHistory} icon={<Database size={17} />} label="Cargar historico real" />
                <StepButton id="static-history" busy={busy} onClick={loadStaticHistory} icon={<Database size={17} />} label="Crear/cargar historico canonico" />
                <StepButton id="analyze-history" busy={busy} onClick={analyzeHistory} icon={<BarChart3 size={17} />} label="Analizar historico" disabled={!historyStatus?.exists} />
                <StepButton id="normalization-preview" busy={busy} onClick={openNormalizationPreview} icon={<Eye size={17} />} label="Ver normalizacion" disabled={!historyStatus?.exists} />
              </div>
              <div className="entity-list compact">
                {analyzableEntities.map((entity) => (
                  <label className="entity-row" key={entity.entity_id}>
                    <input type="checkbox" checked={selectedAnalyzable.includes(entity.entity_id)} onChange={() => toggleEntity(entity.entity_id)} />
                    <span>
                      <strong>{entity.entity_id}</strong>
                      <small>{entityLabel(entity)} / {entity.unit_of_measurement || entity.unit || "sin unidad"}</small>
                    </span>
                  </label>
                ))}
                {!analyzableEntities.length && <Empty text="No hay entidades continuas o binarias cargadas." />}
              </div>
            </Panel>

            <Panel title="Tiempo real e inyeccion">
              <div className="control-grid">
                <label>
                  Segundos por tick
                  <input type="number" min="10" step="10" value={tickSeconds} onChange={(event) => setTickSeconds(Number(event.target.value))} />
                </label>
                <label>
                  Iteraciones a ejecutar
                  <input type="number" min="1" value={iterationCount} onChange={(event) => setIterationCount(Number(event.target.value))} />
                </label>
                <label>
                  Umbral sospecha: {Math.round(suspicionThreshold * 100)}%
                  <input
                    type="range"
                    min="1"
                    max="99"
                    value={Math.round(suspicionThreshold * 100)}
                    onChange={(event) => setSuspicionThreshold(Number(event.target.value) / 100)}
                  />
                </label>
              </div>
              <div className="iteration-bar">
                <div>
                  <span>Iteraciones realizadas</span>
                  <strong>{iterationsDone} / {Math.max(1, Number(iterationCount) || 1)}</strong>
                </div>
                <div className="progress-track">
                  <span style={{ width: `${iterationProgress(iterationsDone, iterationCount)}%` }} />
                </div>
              </div>
              <div className="button-row">
                <StepButton id="tick" busy={busy} onClick={() => generateOneTick()} icon={<Zap size={17} />} label="Generar 1 tick" disabled={!selectedAnalyzable.length} />
                <button className="secondary-action inline" onClick={resetRealtime} disabled={Boolean(busy)}>
                  <RotateCcw size={17} />
                  Reset realtime
                </button>
                <button className="secondary-action inline" onClick={clearFaults} disabled={Boolean(busy)}>
                  <RotateCcw size={17} />
                  Desactivar fallas
                </button>
              </div>
              <div className="fault-list">
                {faults.map((fault) => {
                  const selected = selectedAnalyzable.includes(fault.entity_id);
                  const entity = entityById(analyzableEntities, fault.entity_id);
                  const allowedFaults = faultOptionsForEntity(entity);
                  return (
                    <div className={`fault-row ${selected ? "" : "muted"}`} key={fault.entity_id}>
                      <label className="switch-row">
                        <input
                          type="checkbox"
                          checked={fault.enabled && selected}
                          disabled={!selected}
                          onChange={(event) => updateFault(fault.entity_id, { enabled: event.target.checked })}
                        />
                        <span>
                          <strong>{fault.entity_id}</strong>
                          <small>{selected ? `falla aplicable a ${kindText(entity?.sensor_kind)}` : "entidad no seleccionada"}</small>
                        </span>
                      </label>
                      <select
                        value={fault.fault_type}
                        disabled={!selected}
                        onChange={(event) => updateFault(fault.entity_id, { fault_type: event.target.value })}
                      >
                        {allowedFaults.map((faultType) => (
                          <option key={faultType} value={faultType}>{faultType}</option>
                        ))}
                      </select>
                    </div>
                  );
                })}
              </div>
            </Panel>
          </section>

          <section className="charts-panel">
            <div className="section-heading">
              <div>
                <h2>Charts del historico y realtime</h2>
                <span>{timelines.length} entidades graficadas</span>
              </div>
              <div className="date-controls">
                <label>
                  Fecha y hora inicial
                  <input type="datetime-local" value={chartStart} onChange={(event) => setChartStart(event.target.value)} />
                </label>
                <label>
                  Fecha y hora final
                  <input type="datetime-local" value={chartEnd} onChange={(event) => setChartEnd(event.target.value)} />
                </label>
                <button type="button" onClick={() => loadTimeline()} disabled={Boolean(busy)}>
                  Actualizar
                </button>
              </div>
            </div>
            <div className="chart-grid">
              {timelines.map((timeline) => (
                <TimelineChart
                  key={timeline.entity_id}
                  timeline={timeline}
                  results={allAnalyzedResults}
                  faults={realtime?.faults || []}
                />
              ))}
              {!timelines.length && <Empty text="Carga un historico o genera un tick para ver los charts." />}
            </div>
          </section>

          <section className="workspace-grid lower single">
            <Panel title="Resultados del filtro estadistico">
              <div className="summary-strip">
                <SummaryRow label="Tiempo total filtro" value={formatElapsed(runSummary?.elapsed_ms)} />
                <SummaryRow label="Tiempo continuas" value={formatElapsed(runSummary?.timings_by_kind_ms?.continuous)} />
                <SummaryRow label="Tiempo binarias" value={formatElapsed(runSummary?.timings_by_kind_ms?.binary)} />
                <SummaryRow label="Cache analisis" value={runSummary ? (runSummary.cache_hit ? "reutilizado" : "nuevo") : "-"} />
                <SummaryRow label="Paso a contextual" value={allAnalyzedResults.filter(isRedSuspicion).length} />
                <SummaryRow label="Resultados visibles" value={`${formatCount(displayResults.length)} / ${formatCount(allAnalyzedResults.length)}`} />
              </div>
              <ResultEntityFilters
                entities={resultEntityOptions}
                selectedIds={resultEntityFilterIds}
                onToggle={toggleResultEntityFilter}
                onShowAll={() => setResultEntityFilterIds(null)}
              />
              <ResultsTable results={displayResults} onResultClick={setSelectedResult} />
            </Panel>
          </section>
        </>
      ) : (
        <ProcessWindow
          title={activeModule === "contextual" ? "Filtro contextual" : "Clasificador"}
          description={
            activeModule === "contextual"
              ? "Ventana reservada para contrastar sospechas con entidades relacionadas del hogar."
              : "Ventana reservada para clasificar la sospecha luego del filtro estadistico y contextual."
          }
        />
      )}
      {selectedResult && (
        <CalculationModal
          result={selectedResult}
          onClose={() => setSelectedResult(null)}
        />
      )}
      {normalizationPreview && (
        <NormalizationModal
          preview={normalizationPreview}
          onClose={() => setNormalizationPreview(null)}
        />
      )}
    </main>
  );
}

function Metric({ icon, label, value, detail }) {
  return (
    <article className="metric">
      <div className="metric-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value ?? "-"}</strong>
        <small>{detail || "-"}</small>
      </div>
    </article>
  );
}

function ModuleTab({ active, icon, label, onClick }) {
  return (
    <button className={`module-tab ${active ? "active" : ""}`} type="button" onClick={onClick}>
      {icon}
      <span>{label}</span>
    </button>
  );
}

function ProcessWindow({ title, description }) {
  return (
    <section className="process-window" aria-label={title}>
      <div className="process-window-inner">
        <span className="eyebrow">Modulo reservado</span>
        <h2>{title}</h2>
        <strong>en proceso</strong>
        <p>{description}</p>
      </div>
    </section>
  );
}

function Panel({ title, children }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function StepButton({ id, busy, onClick, icon, label, disabled = false }) {
  return (
    <button className="secondary-action inline" onClick={onClick} disabled={disabled || Boolean(busy)}>
      {busy === id ? <Loader2 className="spin" size={17} /> : icon}
      {label}
    </button>
  );
}

function SummaryRow({ label, value }) {
  return (
    <div className="summary-row">
      <span>{label}</span>
      <strong>{value ?? "-"}</strong>
    </div>
  );
}

function Empty({ text }) {
  return <div className="empty">{text}</div>;
}

function ResultEntityFilters({ entities, selectedIds, onToggle, onShowAll }) {
  if (!entities.length) {
    return null;
  }
  const selectedCount = selectedIds === null ? entities.length : selectedIds.length;
  return (
    <div className="results-filter">
      <div className="results-filter-header">
        <span>Filtrar por entidad</span>
        <strong>{selectedCount} / {entities.length}</strong>
        <button type="button" className="secondary-action inline" onClick={onShowAll}>
          Todas
        </button>
      </div>
      <div className="results-filter-list">
        {entities.map((entityId) => (
          <label className="entity-filter-chip" key={entityId}>
            <input
              type="checkbox"
              checked={selectedIds === null || selectedIds.includes(entityId)}
              onChange={() => onToggle(entityId)}
            />
            <span>{entityId}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

function ResultsTable({ results, onResultClick }) {
  return (
    <div className="results-table">
      <div className="table-head">
        <span>Entidad</span>
        <span>Tiempo mayor-menor</span>
        <span>Score</span>
        <span>Raw</span>
        <span>Lectura</span>
        <span>Patron</span>
        <span>Indicadores</span>
        <span>Ecuacion</span>
      </div>
      {results.map((result, index) => (
        <button
          type="button"
          className="table-row clickable"
          key={`${result.entity_id}-${result.timestamp}-${index}`}
          onClick={() => onResultClick?.(result)}
          title="Ver calculo estadistico"
        >
          <span>{result.entity_id}</span>
          <TimeStack value={result.timestamp} />
          <span>{formatNumber(result.suspicion_score)}</span>
          <span>{formatNumber(result.raw_score)}</span>
          <span>{formatReading(result)}</span>
          <span>{result.suspected_pattern || result.pattern || "-"}</span>
          <span>{(result.triggered_indicators || []).join(", ") || "-"}</span>
          <span>{equationNames(result.triggered_indicators)}</span>
        </button>
      ))}
      {!results.length && <Empty text="No hay sospechas estadisticas para mostrar." />}
    </div>
  );
}

function TimeStack({ value }) {
  const parts = splitDateTime(value);
  return (
    <span className="time-stack">
      <strong>{parts.date}</strong>
      <small>{parts.time}</small>
    </span>
  );
}

function CalculationModal({ result, onClose }) {
  const equations = equationsForResult(result);
  const rows = calculationRows(result);
  const reading = result.reading || {};
  const baseline = result.baseline_summary || {};

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="calculation-modal" role="dialog" aria-modal="true" aria-label="Detalle del calculo estadistico" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <div>
            <span>Calculo estadistico</span>
            <h2>{result.entity_id}</h2>
            <small>{shortDate(result.timestamp)} / {kindText(result.sensor_kind)} / {result.suspected_pattern || result.pattern}</small>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Cerrar detalle">
            <X size={18} />
          </button>
        </header>

        <div className="modal-score-grid">
          <SummaryRow label="Score sospecha" value={formatNumber(result.suspicion_score)} />
          <SummaryRow label="Raw dominante" value={formatNumber(result.raw_score)} />
          <SummaryRow label="Evidencia" value={formatNumber(result.evidence_score)} />
          <SummaryRow label="Paso contextual" value={result.passed_to_contextual_filter ? "si" : "no"} />
        </div>

        <div className="modal-columns">
          <article className="modal-block">
            <h3>Lectura usada</h3>
            <KeyValueGrid rows={readingRows(result)} />
          </article>
          <article className="modal-block">
            <h3>Base de normalidad</h3>
            <KeyValueGrid rows={objectRows(baseline)} />
          </article>
        </div>

        <article className="modal-block">
          <h3>Recreacion del calculo</h3>
          <div className="calc-table">
            <div className="calc-head">
              <span>Indicador</span>
              <span>Valor crudo</span>
              <span>Evidencia</span>
              <span>Estado</span>
            </div>
            {rows.map((row) => (
              <div className={`calc-row ${row.active ? "active" : ""}`} key={row.id}>
                <span>{row.title}</span>
                <span>{row.raw}</span>
                <span>{row.score}</span>
                <span>{row.active ? "activado" : "sin activar"}</span>
              </div>
            ))}
          </div>
          <p className="calc-note">{calculationSummary(result)}</p>
        </article>

        <article className="modal-block">
          <h3>Ecuaciones usadas</h3>
          <div className="equation-list modal-equations">
            {equations.map((equation) => (
              <EquationCardWithValues equation={equation} result={result} key={equation.id} />
            ))}
          </div>
        </article>
      </section>
    </div>
  );
}

function NormalizationModal({ preview, onClose }) {
  const grouped = groupNormalizationSamples(preview?.samples || []);
  const fragmentsByEntity = new Map((preview?.fragments || []).map((fragment) => [fragment.entity_id, fragment]));

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="calculation-modal normalization-modal" role="dialog" aria-modal="true" aria-label="Vista de normalizacion" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <div>
            <span>Fragmentos procesados</span>
            <h2>Normalizacion del historial</h2>
            <small>{preview.source_type || "-"} / {formatCount(preview.rows)} filas / {formatCount(preview.entity_count)} entidades</small>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Cerrar vista de normalizacion">
            <X size={18} />
          </button>
        </header>

        <div className="normalization-list">
          {grouped.map(([entityId, samples]) => (
            <article className="modal-block" key={entityId}>
              <h3>{entityId}</h3>
              {fragmentsByEntity.has(entityId) && (
                <NormalizationFragmentChart timeline={fragmentsByEntity.get(entityId)} />
              )}
              <div className="normalization-table">
                <div className="normalization-head">
                  <span>Fila</span>
                  <span>Original recibido</span>
                  <span>Normalizado usado por el filtro</span>
                </div>
                {samples.map((sample) => (
                  <div className="normalization-row" key={`${entityId}-${sample.raw_index}`}>
                    <span>#{Number(sample.raw_index) + 1}</span>
                    <KeyValueGrid rows={compactRows(sample.original)} />
                    <KeyValueGrid rows={compactRows(sample.normalizado)} />
                  </div>
                ))}
              </div>
            </article>
          ))}
          {!grouped.length && <Empty text="No hay fragmentos de normalizacion disponibles." />}
        </div>
      </section>
    </div>
  );
}

function NormalizationFragmentChart({ timeline }) {
  const width = 720;
  const height = 190;
  const padding = { top: 14, right: 22, bottom: 30, left: 54 };
  const points = (timeline?.points || []).map((point, index) => ({
    ...point,
    value: Number(point.value),
    time: Date.parse(point.timestamp),
    index,
  }));
  const isBinary = timeline?.sensor_kind === "binary";
  const values = points.map((point) => point.value).filter(Number.isFinite);
  const min = isBinary ? 0 : (values.length ? Math.min(...values) : 0);
  const max = isBinary ? 100 : (values.length ? Math.max(...values) : 1);
  const range = max - min || 1;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const validTimes = points.length > 1 && points.every((point) => Number.isFinite(point.time));
  const firstTime = validTimes ? points[0].time : 0;
  const lastTime = validTimes ? points.at(-1).time : Math.max(points.length - 1, 1);
  const timeRange = Math.max(lastTime - firstTime, 1);
  const xFor = (point) => padding.left + ((validTimes ? point.time - firstTime : point.index) / (validTimes ? timeRange : Math.max(points.length - 1, 1))) * plotWidth;
  const yFor = (value) => height - padding.bottom - ((value - min) / range) * plotHeight;
  const linePath = points
    .filter((point) => Number.isFinite(point.value))
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(point).toFixed(1)} ${yFor(point.value).toFixed(1)}`)
    .join(" ");

  return (
    <div className="normalization-chart">
      <div className="chart-title compact">
        <div>
          <strong>Fragmento visual normalizado</strong>
          <small>{points.length} puntos / {kindText(timeline?.sensor_kind)}</small>
        </div>
        <span>{isBinary ? "0-100%" : `${formatAxisValue(min)} - ${formatAxisValue(max)}`}</span>
      </div>
      <svg className="timeline-svg" viewBox={`0 0 ${width} ${height}`}>
        {[0, 0.5, 1].map((tick) => {
          const y = padding.top + tick * plotHeight;
          const label = max - tick * range;
          return (
            <React.Fragment key={tick}>
              <line className="grid" x1={padding.left} y1={y} x2={width - padding.right} y2={y} />
              <text x="8" y={y + 4}>{isBinary ? `${Math.round(label)}%` : formatAxisValue(label)}</text>
            </React.Fragment>
          );
        })}
        <line className="axis" x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
        <line className="axis" x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
        {isBinary && points.map((point, index) => {
          const barWidth = Math.max(2, plotWidth / Math.max(points.length, 1) * 0.72);
          const y = yFor(point.value);
          return (
            <rect
              className="binary-bar"
              key={`${point.timestamp}-normalization-bar-${index}`}
              x={xFor(point) - barWidth / 2}
              y={y}
              width={barWidth}
              height={height - padding.bottom - y}
            />
          );
        })}
        {linePath && <path d={linePath} />}
        {points.map((point, index) => (
          index % Math.max(1, Math.floor(points.length / 28)) === 0
            ? <circle key={`${point.timestamp}-${index}`} cx={xFor(point)} cy={yFor(point.value)} r="2.4" />
            : null
        ))}
        <text x={padding.left} y={height - 8}>{points[0]?.timestamp ? shortDate(points[0].timestamp).slice(0, 16) : "-"}</text>
        <text x={width - 142} y={height - 8}>{points.at(-1)?.timestamp ? shortDate(points.at(-1).timestamp).slice(0, 16) : "-"}</text>
      </svg>
    </div>
  );
}

function EquationCardWithValues({ equation, result }) {
  const substituted = equationWithCalculationValues(equation, result);
  return (
    <article className="equation-card">
      <div>
        <strong>{equation.title}</strong>
        <span>{`${equation.id} -> ${equation.pattern}`}</span>
      </div>
      <code>{substituted.equation}</code>
      <code>{substituted.score}</code>
    </article>
  );
}

function KeyValueGrid({ rows }) {
  if (!rows.length) {
    return <Empty text="Sin datos disponibles." />;
  }
  return (
    <div className="key-value-grid">
      {rows.map((row) => (
        <React.Fragment key={row.label}>
          <span>{row.label}</span>
          <strong>{row.value}</strong>
        </React.Fragment>
      ))}
    </div>
  );
}

function TimelineChart({ timeline, results, faults }) {
  const width = 720;
  const height = 240;
  const padding = { top: 18, right: 24, bottom: 34, left: 58 };
  const basePoints = timeline.points || [];
  const points = mergeTimelineWithSuspiciousPoints(basePoints, results, timeline);
  const isBinary = timeline.sensor_kind === "binary" || timeline.chart_kind === "active_percentage";
  const values = points.map((point) => Number(point.value)).filter(Number.isFinite);
  const min = isBinary ? 0 : (values.length ? Math.min(...values) : 0);
  const max = isBinary ? 100 : (values.length ? Math.max(...values) : 1);
  const range = max - min || 1;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const parsed = points.map((point, index) => ({
    ...point,
    value: Number(point.value),
    time: Date.parse(point.timestamp),
    index,
  }));
  const validTimes = parsed.length > 1 && parsed.every((point) => Number.isFinite(point.time));
  const firstTime = validTimes ? parsed[0].time : 0;
  const lastTime = validTimes ? parsed[parsed.length - 1].time : Math.max(parsed.length - 1, 1);
  const timeRange = Math.max(lastTime - firstTime, 1);
  const xFor = (point) => padding.left + ((validTimes ? point.time - firstTime : point.index) / (validTimes ? timeRange : Math.max(parsed.length - 1, 1))) * plotWidth;
  const xForTime = (time) => padding.left + ((time - firstTime) / timeRange) * plotWidth;
  const yFor = (value) => height - padding.bottom - ((value - min) / range) * plotHeight;
  const linePath = parsed
    .filter((point) => Number.isFinite(point.value))
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(point).toFixed(1)} ${yFor(point.value).toFixed(1)}`)
    .join(" ");
  const markerCandidates = results
    .filter((result) => result.entity_id === timeline.entity_id && isRedSuspicion(result))
    .map((result) => {
      const time = Date.parse(result.timestamp);
      if (!validTimes || !Number.isFinite(time) || time < firstTime || time > lastTime) {
        return null;
      }
      const value = resultChartValue(result, timeline);
      if (!Number.isFinite(value)) {
        return null;
      }
      return {
        x: xForTime(time),
        y: yFor(value),
        time,
        score: Number(result.suspicion_score || 0),
        pattern: result.suspected_pattern || result.pattern || "sospecha",
        label: `${result.suspected_pattern || result.pattern} / score ${result.suspicion_score}`,
      };
    })
    .filter(Boolean);
  const markers = groupConsecutiveMarkers(markerCandidates);
  const faultBands = (faults || [])
    .filter((fault) => fault.entity_id === timeline.entity_id)
    .map((fault) => {
      const start = Date.parse(fault.start);
      const end = Date.parse(fault.end);
      if (!validTimes || !Number.isFinite(start) || !Number.isFinite(end) || end < firstTime || start > lastTime) {
        return null;
      }
      const x1 = Math.max(padding.left, xForTime(Math.max(start, firstTime)));
      const x2 = Math.min(width - padding.right, xForTime(Math.min(end, lastTime)));
      return { x: x1, width: Math.max(2, x2 - x1), label: fault.fault_type };
    })
    .filter(Boolean);

  return (
    <article className="chart-card">
      <div className="chart-title">
        <div>
          <strong>{timeline.entity_id}</strong>
          <small>{kindText(timeline.sensor_kind)} / {timeline.area_id || "sin area"}</small>
        </div>
        <span>{isBinary ? "activacion 0-100%" : `${formatAxisValue(min)} - ${formatAxisValue(max)}`}</span>
      </div>
      <svg className="timeline-svg" viewBox={`0 0 ${width} ${height}`}>
        {[0, 0.5, 1].map((tick) => {
          const y = padding.top + tick * plotHeight;
          const label = max - tick * range;
          return (
            <React.Fragment key={tick}>
              <line className="grid" x1={padding.left} y1={y} x2={width - padding.right} y2={y} />
              <text x="8" y={y + 4}>{isBinary ? `${Math.round(label)}%` : formatAxisValue(label)}</text>
            </React.Fragment>
          );
        })}
        {faultBands.map((band, index) => (
          <rect className="fault-band" key={`${band.label}-${index}`} x={band.x} y={padding.top} width={band.width} height={plotHeight}>
            <title>{band.label}</title>
          </rect>
        ))}
        <line className="axis" x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
        <line className="axis" x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
        {isBinary && parsed.map((point, index) => {
          const barWidth = Math.max(2, plotWidth / Math.max(parsed.length, 1) * 0.72);
          const y = yFor(point.value);
          return (
            <rect
              className="binary-bar"
              key={`${point.timestamp}-bar-${index}`}
              x={xFor(point) - barWidth / 2}
              y={y}
              width={barWidth}
              height={height - padding.bottom - y}
            />
          );
        })}
        {linePath && <path d={linePath} />}
        {parsed.map((point, index) => (
          index % Math.max(1, Math.floor(parsed.length / 36)) === 0
            ? <circle key={`${point.timestamp}-${index}`} cx={xFor(point)} cy={yFor(point.value)} r="2.4" />
            : null
        ))}
        {markers.map((marker, index) => (
          <circle className="suspicion" key={`${marker.label}-${index}`} cx={marker.x} cy={marker.y} r="5">
            <title>{marker.label}</title>
          </circle>
        ))}
        <text x={padding.left} y={height - 8}>{points[0]?.timestamp ? shortDate(points[0].timestamp).slice(0, 16) : "-"}</text>
        <text x={width - 142} y={height - 8}>{points.at(-1)?.timestamp ? shortDate(points.at(-1).timestamp).slice(0, 16) : "-"}</text>
      </svg>
      <div className="chart-legend">
        <span>{basePoints.length} puntos</span>
        {isBinary && <span>bucket: {formatBucketSeconds(timeline.bucket_seconds)}</span>}
        {points.length > basePoints.length && <span>{points.length - basePoints.length} sospechas reinsertadas en la curva</span>}
        <span>{`rojo: score >= ${RED_SUSPICION_SCORE.toFixed(2)} agrupado`}</span>
        {faultBands.length > 0 && <span>zona roja suave: falla activa</span>}
      </div>
    </article>
  );
}

function syncFaultRows(entities, currentFaults) {
  const currentById = Object.fromEntries(currentFaults.map((fault) => [fault.entity_id, fault]));
  return entities.map((entity) => ({
    entity_id: entity.entity_id,
    fault_type: validFaultTypeForEntity(entity, currentById[entity.entity_id]?.fault_type),
    enabled: Boolean(currentById[entity.entity_id]?.enabled),
    label: "sensor_fault",
  }));
}

function isRedSuspicion(result) {
  return Boolean(result?.is_suspicious) && Number(result?.suspicion_score || 0) >= RED_SUSPICION_SCORE;
}

function compareResultsByExecutionDate(left, right) {
  const leftTime = Date.parse(left?.timestamp || "");
  const rightTime = Date.parse(right?.timestamp || "");
  const safeLeft = Number.isFinite(leftTime) ? leftTime : 0;
  const safeRight = Number.isFinite(rightTime) ? rightTime : 0;
  if (safeRight !== safeLeft) {
    return safeRight - safeLeft;
  }
  return Number(right?.suspicion_score || 0) - Number(left?.suspicion_score || 0);
}

function mergeTimelineWithSuspiciousPoints(points, results, timeline) {
  const byTime = new Map();
  const entityId = timeline.entity_id;
  let firstVisibleTime = Infinity;
  let lastVisibleTime = -Infinity;

  for (const point of points) {
    const time = Date.parse(point.timestamp);
    if (Number.isFinite(time)) {
      firstVisibleTime = Math.min(firstVisibleTime, time);
      lastVisibleTime = Math.max(lastVisibleTime, time);
      byTime.set(time, { ...point, source: "timeline" });
    }
  }

  if (!Number.isFinite(firstVisibleTime) || !Number.isFinite(lastVisibleTime)) {
    return points;
  }

  for (const result of results || []) {
    if (result.entity_id !== entityId || !isRedSuspicion(result)) {
      continue;
    }
    const time = Date.parse(result.timestamp);
    const value = resultChartValue(result, timeline);
    if (
      !Number.isFinite(time)
      || !Number.isFinite(value)
      || time < firstVisibleTime
      || time > lastVisibleTime
      || byTime.has(time)
    ) {
      continue;
    }
    byTime.set(time, {
      timestamp: result.timestamp,
      value,
      state: String(value),
      source: "suspicion",
    });
  }

  return [...byTime.entries()]
    .sort(([leftTime], [rightTime]) => leftTime - rightTime)
    .map(([, point]) => point);
}

function resultChartValue(result, timeline) {
  const reading = result.reading || {};
  if (timeline?.sensor_kind === "binary" || timeline?.chart_kind === "active_percentage") {
    if (reading.active_ratio !== undefined && reading.active_ratio !== null) {
      return Number(reading.active_ratio) * 100;
    }
    if (reading.missing_ticks !== undefined) {
      return 100;
    }
    const state = String(result.observed_state ?? result.value ?? "").toLowerCase();
    return ["on", "open", "home", "active", "detected", "1", "true"].includes(state) ? 100 : 0;
  }
  return Number(reading.value ?? result.value ?? result.observed_state);
}

function groupConsecutiveMarkers(markers) {
  const ordered = [...markers].sort((left, right) => left.time - right.time);
  const groups = [];
  for (const marker of ordered) {
    const lastGroup = groups.at(-1);
    const lastMarker = lastGroup?.items.at(-1);
    const sameRun = lastMarker
      && marker.pattern === lastMarker.pattern
      && marker.time - lastMarker.time <= MARKER_GROUP_GAP_MS;
    if (sameRun) {
      lastGroup.items.push(marker);
    } else {
      groups.push({ items: [marker] });
    }
  }

  return groups.map((group) => {
    const representative = group.items.reduce(
      (best, item) => (item.score > best.score ? item : best),
      group.items[0],
    );
    if (group.items.length === 1) {
      return representative;
    }
    const start = shortDate(new Date(group.items[0].time).toISOString());
    const end = shortDate(new Date(group.items.at(-1).time).toISOString());
    return {
      ...representative,
      label: `${representative.pattern} / score ${representative.score.toFixed(4)} / ${group.items.length} eventos entre ${start} y ${end}`,
    };
  });
}

function defaultFaultType(entity) {
  if (entity?.sensor_kind === "binary") {
    return "random";
  }
  if (entity?.device_class === "humidity") {
    return "drift";
  }
  return "random/spike";
}

function faultOptionsForEntity(entity) {
  if (entity?.sensor_kind === "binary") {
    return BINARY_FAULT_TYPES;
  }
  return CONTINUOUS_FAULT_TYPES;
}

function validFaultTypeForEntity(entity, faultType) {
  const options = faultOptionsForEntity(entity);
  return options.includes(faultType) ? faultType : defaultFaultType(entity);
}

function entityById(entities, entityId) {
  return entities.find((entity) => entity.entity_id === entityId);
}

function entityLabel(entity) {
  return `${kindText(entity.sensor_kind)} / ${entity.device_class || "estado"}`;
}

function kindText(kind) {
  if (kind === "continuous") {
    return "continua";
  }
  if (kind === "binary") {
    return "estado binario";
  }
  return kind || "entidad";
}

function timelineDateRange(timelines) {
  const times = [];
  for (const timeline of timelines || []) {
    for (const point of timeline.points || []) {
      const time = Date.parse(point.timestamp);
      if (Number.isFinite(time)) {
        times.push(time);
      }
    }
  }
  if (!times.length) {
    return { start: "", end: "" };
  }
  return {
    start: toDatetimeLocalValue(Math.min(...times)),
    end: toDatetimeLocalValue(Math.max(...times)),
  };
}

function timelineQueryString(startValue, endValue) {
  const params = new URLSearchParams();
  const start = inputDateToIso(startValue);
  const end = inputDateToIso(endValue);
  if (start) {
    params.set("start", start);
  }
  if (end) {
    params.set("end", end);
  }
  params.set("max_points", "1000");
  return `?${params.toString()}`;
}

function inputDateToIso(value) {
  if (!value) {
    return "";
  }
  const time = new Date(value).getTime();
  return Number.isFinite(time) ? new Date(time).toISOString() : "";
}

function toDatetimeLocalValue(timestamp) {
  const date = new Date(timestamp);
  if (!Number.isFinite(date.getTime())) {
    return "";
  }
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

function equationNames(indicators = []) {
  const names = indicators
    .map((indicator) => ALL_EQUATIONS.find((equation) => equation.id === indicator)?.title)
    .filter(Boolean);
  return names.join(", ") || "-";
}

function equationsForResult(result) {
  const ids = result?.triggered_indicators || [];
  const found = ids
    .map((indicator) => ALL_EQUATIONS.find((equation) => equation.id === indicator))
    .filter(Boolean);
  return found.length ? found : ALL_EQUATIONS.filter((equation) => equation.pattern === (result?.suspected_pattern || result?.pattern));
}

function equationWithCalculationValues(equation, result) {
  if (result?.sensor_kind === "binary") {
    return binaryEquationWithValues(equation, result);
  }
  return continuousEquationWithValues(equation, result);
}

function continuousEquationWithValues(equation, result) {
  const reading = result?.reading || {};
  const raw = reading.raw_components || {};
  const components = reading.components || {};
  const base = result?.baseline_summary || {};
  const value = Number(reading.value ?? result?.value ?? 0);
  const delta = Number(reading.delta_value ?? 0);
  const gapSeconds = Number(reading.gap_seconds ?? 0);
  const deltaMinutes = gapSeconds > 0 ? gapSeconds / 60 : 0;
  const rate = Number(reading.rate_per_minute ?? 0);
  const residual = Number(reading.residual ?? 0);
  const prediction = value - residual;

  if (equation.id === "hampel") {
    return {
      equation: `H_i(t) = |${formatCalc(value)} - mediana(W)| / (1.4826 * MAD(W) + epsilon) = ${formatCalc(raw.H_i)}`,
      score: `s_H = min(1, ${formatCalc(raw.H_i)} / 3.5) = ${formatCalc(components.s_H)}`,
    };
  }
  if (equation.id === "robust_rate_change") {
    return {
      equation: `R_i(t) = |${formatCalc(delta)}| / ${formatCalc(deltaMinutes)}min = ${formatCalc(rate)}/min; Z_R_i(t) = ${formatCalc(raw.Z_R_i)}`,
      score: `s_R = min(1, ${formatCalc(raw.Z_R_i)} / 3.5) = ${formatCalc(components.s_R)}`,
    };
  }
  if (equation.id === "cusum_residual") {
    return {
      equation: `r_i(t) = ${formatCalc(value)} - ${formatCalc(prediction)} = ${formatCalc(residual)}; max(C+, C-) = ${formatCalc(raw.C_residual)}`,
      score: `s_C = min(1, ${formatCalc(raw.C_residual)} / ${formatCalc(base.cusum_theta)}) = ${formatCalc(components.s_C)}`,
    };
  }
  if (equation.id === "constant_persistence") {
    return {
      equation: `D_const = ${formatCalc(reading.constant_duration_seconds)}s con epsilon_const = ${formatCalc(base.epsilon_const)}; Z_D_const = ${formatCalc(raw.Z_D_const)}`,
      score: `s_D_const = min(1, ${formatCalc(raw.Z_D_const)} / 3.5) = ${formatCalc(components.s_D_const)}`,
    };
  }
  if (equation.id === "reading_gap") {
    return {
      equation: `M_i(t) = t - t_i_last = ${formatCalc(gapSeconds)}s; Z_M = ${formatCalc(raw.Z_M)}`,
      score: `s_M = min(1, ${formatCalc(raw.Z_M)} / 3.5) = ${formatCalc(components.s_M)}`,
    };
  }
  return { equation: equation.equation, score: equation.score };
}

function binaryEquationWithValues(equation, result) {
  const reading = result?.reading || {};
  const raw = reading.raw_components || {};
  const components = reading.components || {};
  const windowSeconds = Number(reading.window_seconds ?? 0);

  if (equation.id === "activation_count") {
    return {
      equation: `A_i(W) = cantidad de transiciones 0->1 en ${formatCalc(windowSeconds)}s = ${reading.activations ?? "-"}`,
      score: `s_A = min(1, Z_A / tau_A) = min(1, ${formatCalc(raw.Z_A_i)} / 3.5) = ${formatCalc(components.s_A)}`,
    };
  }
  if (equation.id === "state_change_count") {
    return {
      equation: `C_i(W) = cantidad de cambios de estado en ${formatCalc(windowSeconds)}s = ${reading.state_changes ?? "-"}`,
      score: `s_C = min(1, Z_C / tau_C) = min(1, ${formatCalc(raw.Z_C_i)} / 3.5) = ${formatCalc(components.s_C)}`,
    };
  }
  if (equation.id === "active_time_ratio") {
    return {
      equation: `P_on,i(W) = T_on,i(W) / |W| = ${formatCalc(reading.active_seconds)}s / ${formatCalc(windowSeconds)}s = ${formatCalc(reading.active_ratio)}`,
      score: `s_P = min(1, Z_P / tau_P) = min(1, ${formatCalc(raw.Z_P_on_i)} / 3.5) = ${formatCalc(components.s_P)}`,
    };
  }
  if (equation.id === "time_since_last_activation") {
    return {
      equation: `G_i(t) = t - ultima activacion = ${formatCalc(reading.gap_seconds)}s; Z_G = ${formatCalc(raw.Z_G_i)}`,
      score: `s_G = min(1, Z_G / tau_G) = min(1, ${formatCalc(raw.Z_G_i)} / 3.5) = ${formatCalc(components.s_G)}`,
    };
  }
  if (equation.id === "report_gap") {
    return {
      equation: `M_i(t) = t - t_i_last = ${formatCalc(reading.report_gap_seconds)}s; Z_M = ${formatCalc(raw.Z_M_i)}`,
      score: `s_M = min(1, Z_M / tau_M) = min(1, ${formatCalc(raw.Z_M_i)} / 3.5) = ${formatCalc(components.s_M)}`,
    };
  }
  return { equation: equation.equation, score: equation.score };
}

function calculationRows(result) {
  if (result?.sensor_kind === "binary") {
    return [
      calculationRow(result, "activation_count", "Activaciones A_i(W)", "Z_A_i", "s_A"),
      calculationRow(result, "state_change_count", "Cambios C_i(W)", "Z_C_i", "s_C"),
      calculationRow(result, "active_time_ratio", "Proporcion P_on,i(W)", "Z_P_on_i", "s_P"),
      calculationRow(result, "time_since_last_activation", "Tiempo G_i(t)", "Z_G_i", "s_G"),
      calculationRow(result, "report_gap", "Brecha M_i(t)", "Z_M_i", "s_M"),
    ];
  }
  return [
    calculationRow(result, "hampel", "Filtro Hampel H_i(t)", "H_i", "s_H"),
    calculationRow(result, "robust_rate_change", "Tasa robusta Z_R_i(t)", "Z_R_i", "s_R"),
    calculationRow(result, "cusum_residual", "CUSUM residual", "C_residual", "s_C"),
    calculationRow(result, "constant_persistence", "Persistencia D_const", "Z_D_const", "s_D_const"),
    calculationRow(result, "reading_gap", "Brecha M_i(t)", "Z_M", "s_M"),
  ];
}

function calculationRow(result, indicator, title, rawKey, scoreKey) {
  const raw = result?.reading?.raw_components?.[rawKey];
  const score = result?.reading?.components?.[scoreKey];
  const active = (result?.triggered_indicators || []).includes(indicator);
  return {
    id: indicator,
    title,
    raw: formatNumber(raw),
    score: formatNumber(score),
    active,
  };
}

function calculationSummary(result) {
  if (result?.sensor_kind === "binary") {
    return `S_bin(W) = promedio de s_A, s_C, s_P, s_G y s_M = ${formatNumber(result.evidence_score)}. Luego P_sos = S_bin / (S_bin + tau_i) = ${formatNumber(result.suspicion_score)}.`;
  }
  return `Evidencia continua = max(s_H, s_R, s_C, s_D_const, s_M) = ${formatNumber(result.evidence_score)}. Luego P_sos = evidencia / (evidencia + tau_i) = ${formatNumber(result.suspicion_score)}.`;
}

function readingRows(result) {
  const reading = result?.reading || {};
  if (result?.sensor_kind === "binary") {
    return [
      { label: "Estado observado", value: result.observed_state ?? "-" },
      { label: "A_i(W)", value: reading.activations ?? "-" },
      { label: "C_i(W)", value: reading.state_changes ?? "-" },
      { label: "T_on,i(W)", value: formatSeconds(reading.active_seconds) },
      { label: "P_on,i(W)", value: formatPercent(reading.active_ratio) },
      { label: "G_i(t)", value: formatSeconds(reading.gap_seconds) },
      { label: "M_i(t)", value: formatSeconds(reading.report_gap_seconds) },
      { label: "Ventana", value: formatSeconds(reading.window_seconds) },
    ];
  }
  return [
    { label: "Valor observado", value: reading.value ?? result?.value ?? "-" },
    { label: "Delta", value: formatSigned(reading.delta_value) },
    { label: "Tasa/min", value: reading.rate_per_minute !== undefined ? `${formatNumber(reading.rate_per_minute)}/min` : "-" },
    { label: "Brecha", value: formatSeconds(reading.gap_seconds) },
    { label: "Residual", value: formatNumber(reading.residual) },
    { label: "Duracion constante", value: formatSeconds(reading.constant_duration_seconds) },
  ];
}

function objectRows(object) {
  return Object.entries(object || {}).map(([key, value]) => ({
    label: key,
    value: typeof value === "number" ? formatNumber(value) : String(value ?? "-"),
  }));
}

function compactRows(object) {
  return Object.entries(object || {})
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => ({
      label: key,
      value: typeof value === "number" ? formatNumber(value) : String(value),
    }));
}

function groupNormalizationSamples(samples) {
  const grouped = new Map();
  for (const sample of samples || []) {
    const entityId = sample.entity_id || "sin entidad";
    grouped.set(entityId, [...(grouped.get(entityId) || []), sample]);
  }
  return [...grouped.entries()].sort(([left], [right]) => left.localeCompare(right));
}

function formatReading(result) {
  const reading = result.reading || {};
  if (reading.missing_ticks !== undefined) {
    return `missing ${reading.missing_ticks} ticks / M=${formatSeconds(reading.report_gap_seconds ?? reading.gap_seconds)}`;
  }
  if (result.sensor_kind === "binary") {
    return [
      `A=${reading.activations ?? "-"}`,
      `C=${reading.state_changes ?? "-"}`,
      `P=${formatPercent(reading.active_ratio)}`,
      `G=${formatSeconds(reading.gap_seconds)}`,
      `M=${formatSeconds(reading.report_gap_seconds)}`,
    ].join(" ");
  }
  const parts = [`valor ${reading.value ?? result.value ?? result.observed_state ?? "-"}`];
  if (reading.delta_value !== undefined) {
    parts.push(`delta ${formatSigned(reading.delta_value)}`);
  }
  if (reading.rate_per_minute !== undefined && reading.rate_per_minute !== null) {
    parts.push(`rate ${Number(reading.rate_per_minute).toFixed(4)}/min`);
  }
  return parts.join(" / ");
}

function formatPercent(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return `${Math.round(Number(value) * 100)}%`;
}

function formatHistoryRange(history) {
  if (!history?.exists) {
    return "sin historico cargado";
  }
  return `${shortDate(history.start).slice(0, 16)} -> ${shortDate(history.end).slice(0, 16)}`;
}

function formatElapsed(ms) {
  if (ms === undefined || ms === null) {
    return "-";
  }
  return Number(ms) < 1000 ? `${Number(ms).toFixed(1)} ms` : `${(Number(ms) / 1000).toFixed(3)} s`;
}

function formatCount(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return Number(value).toLocaleString("es-CL");
}

function formatAxisValue(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2);
}

function formatNumber(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return Number(value).toFixed(4);
}

function formatCalc(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  if (Math.abs(number) >= 1000) {
    return number.toFixed(0);
  }
  if (Math.abs(number) >= 100) {
    return number.toFixed(2);
  }
  if (Math.abs(number) >= 10) {
    return number.toFixed(3);
  }
  return number.toFixed(4);
}

function formatSigned(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `${number >= 0 ? "+" : ""}${number.toFixed(3)}`;
}

function formatSeconds(value) {
  if (value === undefined || value === null) {
    return "-";
  }
  return `${Number(value).toFixed(0)}s`;
}

function formatBucketSeconds(seconds) {
  if (!seconds) {
    return "-";
  }
  if (seconds < 60) {
    return `${seconds}s`;
  }
  if (seconds % 3600 === 0) {
    return `${seconds / 3600}h`;
  }
  if (seconds % 60 === 0) {
    return `${seconds / 60}min`;
  }
  return `${seconds}s`;
}

function shortDate(value) {
  return value ? String(value).replace("T", " ").slice(0, 19) : "-";
}

function splitDateTime(value) {
  const formatted = shortDate(value);
  if (formatted === "-") {
    return { date: "-", time: "-" };
  }
  const [date, time = ""] = formatted.split(" ");
  return { date, time: time.replace("Z", "") || "-" };
}

function iterationProgress(done, target) {
  const total = Math.max(1, Number(target) || 1);
  return Math.min(100, Math.max(0, (Number(done) / total) * 100));
}

createRoot(document.getElementById("root")).render(<App />);
