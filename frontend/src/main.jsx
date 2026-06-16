import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Filter,
  Loader2,
  Play,
  RotateCcw,
  Settings2,
  SlidersHorizontal,
} from "lucide-react";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const FAULT_TYPES = ["random/spike", "random", "malfunction", "bias", "drift", "stuck-at", "missing"];

const SIMULATION_ENTITIES = [
  { entity_id: "sensor.living_temperature", sensor_kind: "continuous", tipo_interno: "continua", area_id: "living" },
  { entity_id: "sensor.living_humidity", sensor_kind: "continuous", tipo_interno: "continua", area_id: "living" },
  { entity_id: "binary_sensor.living_motion", sensor_kind: "binary", tipo_interno: "estado_binario", area_id: "living" },
  { entity_id: "binary_sensor.main_door_contact", sensor_kind: "binary", tipo_interno: "estado_binario", area_id: "entrance" },
  { entity_id: "climate.living", sensor_kind: "multistate", tipo_interno: "estado_multivalor", area_id: "living" },
];

const DEFAULT_FAULTS = [
  ["sensor.living_temperature", "random/spike"],
  ["sensor.living_humidity", "drift"],
  ["binary_sensor.living_motion", "random"],
  ["binary_sensor.main_door_contact", "stuck-at"],
  ["climate.living", "random"],
].map(([entity_id, fault_type]) => ({
  entity_id,
  fault_type,
  enabled: false,
  label: "sensor_fault",
}));

function defaultFaultsForEntities(entities) {
  return (entities.length ? entities : SIMULATION_ENTITIES).map((entity) => ({
    entity_id: entity.entity_id,
    fault_type: defaultFaultForEntity(entity),
    enabled: false,
    label: "sensor_fault",
  }));
}

function defaultFaultForEntity(entity) {
  if (entity.sensor_kind === "continuous") {
    return entity.device_class === "humidity" ? "drift" : "random/spike";
  }
  if (entity.sensor_kind === "binary") {
    return entity.device_class === "motion" ? "random" : "stuck-at";
  }
  return "random";
}

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
  const [tickSeconds, setTickSeconds] = useState(60);
  const [suspicionThreshold, setSuspicionThreshold] = useState(0.5);
  const [faults, setFaults] = useState(DEFAULT_FAULTS);
  const [selectedEntityIds, setSelectedEntityIds] = useState(SIMULATION_ENTITIES.map((entity) => entity.entity_id));
  const [entities, setEntities] = useState([]);
  const [timelines, setTimelines] = useState([]);
  const [runSummary, setRunSummary] = useState(null);
  const [results, setResults] = useState([]);
  const [historicalResults, setHistoricalResults] = useState([]);
  const [realtime, setRealtime] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [chartStart, setChartStart] = useState("");
  const [chartEnd, setChartEnd] = useState("");
  const currentEntities = entities.length ? entities : SIMULATION_ENTITIES;

  useEffect(() => {
    runStep("health", async () => {
      setHealth(await api("/health"));
      const status = await api("/history/static/status");
      setHistoryStatus(status);
      if (status.entities?.length) {
        syncLoadedEntities(status.entities);
      }
    });
  }, []);

  useEffect(() => {
    if (!timelines.length || busy === "tick" || busy === "reset") {
      return undefined;
    }
    const handle = window.setTimeout(() => {
      loadTimelineForRange(chartStart, chartEnd).catch((err) => setError(err.message));
    }, 400);
    return () => window.clearTimeout(handle);
  }, [chartStart, chartEnd]);

  const kindCounts = useMemo(() => {
    return entities.reduce((acc, entity) => {
      acc[entity.sensor_kind] = (acc[entity.sensor_kind] || 0) + 1;
      return acc;
    }, {});
  }, [entities]);
  const displayResults = useMemo(
    () => [...historicalResults, ...results]
      .sort((left, right) => Number(right.suspicion_score || 0) - Number(left.suspicion_score || 0))
      .slice(0, 1000),
    [historicalResults, results],
  );
  const suspiciousCount = displayResults.filter((result) => result.is_suspicious).length;

  async function runStep(label, action) {
    setBusy(label);
    setError("");
    try {
      await action();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  function activeFaults() {
    return faults.map((fault) => ({
      ...fault,
      enabled: fault.enabled && selectedEntityIds.includes(fault.entity_id),
    }));
  }

  function resetRealtimeState({ clearHistorical = false } = {}) {
    setTimelines([]);
    setRunSummary(null);
    setResults([]);
    setRealtime(null);
    setChartStart("");
    setChartEnd("");
    if (clearHistorical) {
      setHistoricalResults([]);
    }
  }

  function syncLoadedEntities(nextEntities, options = {}) {
    const usableEntities = nextEntities || [];
    setEntities(usableEntities);
    setSelectedEntityIds(usableEntities.map((entity) => entity.entity_id));
    setFaults(defaultFaultsForEntities(usableEntities));
    resetRealtimeState(options);
  }

  async function ensureHistory() {
    await runStep("history", async () => {
      const payload = await api("/history/static/ensure", { method: "POST" });
      setHistoryStatus(payload);
      syncLoadedEntities(payload.entities || [], { clearHistorical: true });
    });
  }

  async function loadRealHistory() {
    await runStep("real-history", async () => {
      const payload = await api("/history/real/load", { method: "POST" });
      setHistoryStatus(payload);
      syncLoadedEntities(payload.entities || [], { clearHistorical: true });
      setHistoricalResults(payload.historical_results || []);
      setRunSummary(payload.historical_summary || {});
      await loadTimelineForRange("", "", true);
    });
  }

  async function analyzeHistoricalResults(nextThreshold = suspicionThreshold) {
    const payload = await api("/history/analyze", {
      method: "POST",
      body: JSON.stringify({ suspicion_threshold: nextThreshold }),
    });
    setHistoricalResults(payload.results || []);
    setRunSummary(payload.summary || {});
  }

  async function generateTick() {
    await runStep("tick", async () => {
      if (!historyStatus?.exists || historyStatus?.valid === false) {
        const history = await api("/history/static/ensure", { method: "POST" });
        setHistoryStatus(history);
        syncLoadedEntities(history.entities || []);
      }
      const payload = await api("/realtime/tick", {
        method: "POST",
        body: JSON.stringify({
          selected_entities: selectedEntityIds,
          tick_seconds: tickSeconds,
          faults: activeFaults(),
          suspicion_threshold: suspicionThreshold,
        }),
      });
      setRealtime(payload);
      setRunSummary(payload.summary || {});
      setResults(payload.results || []);
      await loadTimelineForRange(chartStart, chartEnd, true);
    });
  }

  async function loadTimelineForRange(startValue = chartStart, endValue = chartEnd, initializeRange = false) {
    const timeline = await api(`/realtime/timeline${timelineQueryString(startValue, endValue)}`);
    const timelineEntities = (timeline.entities || []).map((item) => ({
      ...item,
      realtime_start: timeline.realtime_start,
      realtime_end: timeline.realtime_end,
    }));
    setTimelines(timelineEntities);
    if (initializeRange) {
      const range = timelineDateRange(timelineEntities);
      setChartStart((current) => current || range.start);
      setChartEnd((current) => current || range.end);
    }
  }

  async function resetRealtime() {
    await runStep("reset", async () => {
      const payload = await api("/realtime/reset", { method: "POST" });
      setRealtime(payload);
      resetRealtimeState();
    });
  }

  function updateFault(index, patch) {
    setFaults((current) =>
      current.map((fault, itemIndex) => (itemIndex === index ? { ...fault, ...patch } : fault)),
    );
    setRunSummary(null);
    setResults([]);
  }

  function toggleEntity(entityId) {
    setSelectedEntityIds((current) =>
      current.includes(entityId)
        ? current.filter((item) => item !== entityId)
        : [...current, entityId],
    );
    setRunSummary(null);
    setResults([]);
  }

  function resetFaults() {
    setFaults((current) => current.map((fault) => ({ ...fault, enabled: false })));
    setRunSummary(null);
    setResults([]);
  }

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <h1>Filtro estadistico en tiempo real</h1>
          <p>Carga historico real o canonico y luego emite ticks nuevos con fallas sinteticas opcionales.</p>
        </div>
        <button className="primary-action" onClick={generateTick} disabled={Boolean(busy) || selectedEntityIds.length === 0}>
          {busy === "tick" ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          Generar tick
        </button>
      </section>

      {error && (
        <section className="alert">
          <AlertTriangle size={18} />
          {error}
        </section>
      )}

      <section className="metrics-grid">
        <Metric icon={<Settings2 />} label="API" value={health?.status || "-"} detail="Swagger en /docs" />
        <Metric icon={<SlidersHorizontal />} label="Historico" value={historyStatus?.rows || "-"} detail={formatHistoryStatus(historyStatus)} />
        <Metric icon={<CheckCircle2 />} label="Entidades" value={entities.length || selectedEntityIds.length || "-"} detail={formatKindCounts(kindCounts)} />
        <Metric icon={<BarChart3 />} label="Sospechas" value={suspiciousCount || "-"} detail={`${displayResults.length} resultados`} />
        <Metric icon={<Activity />} label="Tiempo real" value={realtime?.events || "-"} detail={realtime?.current_timestamp ? shortDate(realtime.current_timestamp) : "sin ticks"} />
      </section>

      <section className="content-grid simulation-grid">
        <Panel title="Historico y tiempo real">
          <div className="form-grid">
            <label>
              Tick segundos
              <input type="number" min="10" value={tickSeconds} onChange={(event) => setTickSeconds(Number(event.target.value))} />
            </label>
            <label>
              Umbral sospecha {Math.round(suspicionThreshold * 100)}%
              <input
                type="range"
                min="1"
                max="99"
                value={Math.round(suspicionThreshold * 100)}
                onChange={(event) => {
                  const nextThreshold = Number(event.target.value) / 100;
                  setSuspicionThreshold(nextThreshold);
                  setRunSummary(null);
                  setResults([]);
                  if (historicalResults.length) {
                    analyzeHistoricalResults(nextThreshold).catch((err) => setError(err.message));
                  }
                }}
              />
            </label>
          </div>
          <div className="workflow">
            <StepButton busy={busy} id="history" icon={<SlidersHorizontal />} label="Crear/cargar historico" onClick={ensureHistory} disabled={false} />
            <StepButton busy={busy} id="real-history" icon={<SlidersHorizontal />} label="Cargar historico real" onClick={loadRealHistory} disabled={false} />
            <StepButton busy={busy} id="tick" icon={<Filter />} label="Generar tick" onClick={generateTick} disabled={selectedEntityIds.length === 0} />
            <button className="step-button" onClick={resetRealtime} disabled={Boolean(busy)}>
              <RotateCcw size={18} />
              Reset realtime
            </button>
            <button className="step-button" onClick={resetFaults} disabled={Boolean(busy)}>
              <RotateCcw size={18} />
              Fallas off
            </button>
          </div>
        </Panel>

        <Panel title="Fallas activas">
          <div className="fault-list">
            {faults.map((fault, index) => {
              const entityEnabled = selectedEntityIds.includes(fault.entity_id);
              return (
                <div className={`fault-row ${entityEnabled ? "" : "disabled-row"}`} key={`${fault.entity_id}-${fault.fault_type}`}>
                  <label className="switch-row">
                    <input
                      type="checkbox"
                      checked={fault.enabled && entityEnabled}
                      disabled={!entityEnabled}
                      onChange={(event) => updateFault(index, { enabled: event.target.checked })}
                    />
                    <span>
                      <strong>{fault.entity_id}</strong>
                      <small>{entityEnabled ? fault.fault_type : "entidad desactivada"}</small>
                    </span>
                  </label>
                  <span className="fault-mode">por tick</span>
                  <select
                    value={fault.fault_type}
                    disabled={!entityEnabled || Boolean(busy)}
                    onChange={(event) => updateFault(index, { fault_type: event.target.value })}
                  >
                    {FAULT_TYPES.map((faultType) => (
                      <option key={faultType} value={faultType}>{faultType}</option>
                    ))}
                  </select>
                </div>
              );
            })}
          </div>
        </Panel>
      </section>

      <section className="content-grid">
        <Panel title="Entidades para tiempo real">
          <div className="entity-list">
            {currentEntities.map((entity) => (
              <label className="entity-row" key={entity.entity_id}>
                <input
                  type="checkbox"
                  checked={selectedEntityIds.includes(entity.entity_id)}
                  onChange={() => toggleEntity(entity.entity_id)}
                />
                <span>
                  <strong>{entity.entity_id}</strong>
                  <small>{entity.sensor_kind} / {entity.tipo_interno} / {entity.area_id || "sin area"}</small>
                </span>
              </label>
            ))}
          </div>
        </Panel>

        <Panel title="Estado realtime">
          {historyStatus?.exists ? (
            <div className="summary-list">
              <SummaryRow label="Historico filas" value={historyStatus.rows} />
              <SummaryRow label="Historico rango" value={formatHistoryRange(historyStatus)} />
              <SummaryRow label="Ticks generados" value={realtime?.tick_index || 0} />
              <SummaryRow label="Eventos realtime" value={realtime?.events || 0} />
              <SummaryRow label="Tiempo filtro" value={formatElapsed(runSummary?.elapsed_ms)} />
              <SummaryRow label="Tiempo por tipo" value={formatKindTiming(runSummary?.timings_by_kind_ms)} />
              <SummaryRow label="Ground truth realtime" value={formatRealtimeTruth(realtime)} />
            </div>
          ) : (
            <Empty text="Crea o carga el historico estatico antes de generar ticks." />
          )}
        </Panel>
      </section>

      <section className="charts-section">
        <div className="section-heading">
          <div>
            <h2>Historico reciente + tiempo real</h2>
            <span>{timelines.length ? `${timelines.length} entidades graficadas` : "sin ticks"}</span>
          </div>
          <div className="chart-date-controls">
            <label>
              Inicio
              <input type="datetime-local" value={chartStart} onChange={(event) => setChartStart(event.target.value)} disabled={!timelines.length} />
            </label>
            <label>
              Fin
              <input type="datetime-local" value={chartEnd} onChange={(event) => setChartEnd(event.target.value)} disabled={!timelines.length} />
            </label>
            <button type="button" onClick={() => {
              const range = timelineDateRange(timelines);
              setChartStart(range.start);
              setChartEnd(range.end);
            }} disabled={!timelines.length} title="Restablecer rango" aria-label="Restablecer rango">
              <RotateCcw size={15} />
            </button>
          </div>
        </div>
        <div className="chart-grid">
          {timelines.map((timeline) => (
            <TimelineChart
              key={timeline.entity_id}
              timeline={timeline}
              faults={realtime?.faults || []}
              results={displayResults}
              chartStart={chartStart}
              chartEnd={chartEnd}
            />
          ))}
          {!timelines.length && <Empty text="Genera un tick para ver el historico reciente junto a eventos nuevos." />}
        </div>
      </section>

      <section className="results-section">
        <div className="section-heading">
          <h2>Resultados del filtro estadistico</h2>
          <span>{displayResults.length ? `${suspiciousCount} sospechas marcadas` : "sin resultados"}</span>
        </div>
        <div className="results-table">
          <div className="table-head">
            <span>Entidad</span>
            <span>Tipo</span>
            <span>Tiempo</span>
            <span>Score</span>
            <span>Raw</span>
            <span>Lectura</span>
            <span>Patron</span>
            <span>Detalle</span>
            <span>Ground truth</span>
          </div>
          {displayResults.slice(0, 1000).map((result, index) => (
            <div className="table-row simulation-row" key={`${result.entity_id}-${result.timestamp}-${index}`}>
              <span>{result.entity_id}</span>
              <span className={`pill ${result.sensor_kind}`}>{result.sensor_kind}</span>
              <span>{shortDate(result.timestamp)}</span>
              <span>{result.suspicion_score}</span>
              <span>{result.raw_score}</span>
              <span>{formatReading(result)}</span>
              <span title={(result.triggered_indicators || []).join(", ")}>{result.suspected_pattern || result.pattern || "-"}</span>
              <span>{result.pattern_detail || "-"}</span>
              <span>{result.ground_truth_label || result.result_source || "-"}</span>
            </div>
          ))}
          {!displayResults.length && <Empty text="Carga historico real o genera ticks; las sospechas apareceran automaticamente." />}
        </div>
      </section>
    </main>
  );
}

function Metric({ icon, label, value, detail }) {
  return (
    <div className="metric">
      <div className="metric-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </div>
  );
}

function StepButton({ busy, id, icon, label, onClick, disabled }) {
  const isBusy = busy === id;
  return (
    <button className="step-button" onClick={onClick} disabled={disabled || Boolean(busy)}>
      {isBusy ? <Loader2 className="spin" size={18} /> : icon}
      {label}
    </button>
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

function SummaryRow({ label, value }) {
  return (
    <div className="summary-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Empty({ text }) {
  return <div className="empty">{text}</div>;
}

function timelineDateRange(timelines) {
  const times = [];
  for (const timeline of timelines || []) {
    for (const point of timeline.points || []) {
      const timestamp = Date.parse(point.timestamp);
      if (Number.isFinite(timestamp)) {
        times.push(timestamp);
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

function filterPointsByDateRange(points, startTime, endTime) {
  if (!points.length || (!Number.isFinite(startTime) && !Number.isFinite(endTime))) {
    return points;
  }
  const filtered = points.filter((point) => {
    const timestamp = Date.parse(point.timestamp);
    if (!Number.isFinite(timestamp)) {
      return true;
    }
    if (Number.isFinite(startTime) && timestamp < startTime) {
      return false;
    }
    if (Number.isFinite(endTime) && timestamp > endTime) {
      return false;
    }
    return true;
  });
  return filtered.length ? filtered : [];
}

function toDatetimeLocalValue(timestamp) {
  const date = new Date(timestamp);
  if (!Number.isFinite(date.getTime())) {
    return "";
  }
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

function inputDateTimeToMs(value) {
  if (!value) {
    return Number.NaN;
  }
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : Number.NaN;
}

function timelineQueryString(startValue, endValue) {
  const params = new URLSearchParams();
  const { startTime, endTime } = normalizeDateRange(startValue, endValue);
  if (Number.isFinite(startTime)) {
    params.set("start", new Date(startTime).toISOString());
  }
  if (Number.isFinite(endTime)) {
    params.set("end", new Date(endTime).toISOString());
  }
  params.set("max_points", "900");
  return `?${params.toString()}`;
}

function normalizeDateRange(startValue, endValue) {
  const startTime = inputDateTimeToMs(startValue);
  const endTime = inputDateTimeToMs(endValue);
  if (Number.isFinite(startTime) && Number.isFinite(endTime) && startTime > endTime) {
    return { startTime: endTime, endTime: startTime };
  }
  return { startTime, endTime };
}

function TimelineChart({ timeline, faults, results, chartStart, chartEnd }) {
  const width = 640;
  const height = 190;
  const padding = { top: 18, right: 18, bottom: 30, left: 54 };
  const points = timeline.points || [];
  const isBinary = timeline.sensor_kind === "binary";
  const { startTime, endTime } = normalizeDateRange(chartStart, chartEnd);
  const visiblePoints = filterPointsByDateRange(points, startTime, endTime);
  const values = visiblePoints.map((point) => Number(point.value)).filter((value) => Number.isFinite(value));
  const min = isBinary ? 0 : values.length ? Math.min(...values) : 0;
  const max = isBinary ? 100 : values.length ? Math.max(...values) : 1;
  const range = max - min || 1;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const parsedPoints = visiblePoints.map((point, index) => ({
    ...point,
    chartValue: Number(point.value),
    chartTime: Date.parse(point.timestamp),
    fallbackIndex: index,
  }));
  const hasValidTimes = parsedPoints.length > 1 && parsedPoints.every((point) => Number.isFinite(point.chartTime));
  const firstTime = hasValidTimes ? parsedPoints[0].chartTime : 0;
  const lastTime = hasValidTimes ? parsedPoints[parsedPoints.length - 1].chartTime : Math.max(parsedPoints.length - 1, 1);
  const timeRange = Math.max(lastTime - firstTime, 1);
  const xFor = (point, index) => {
    if (hasValidTimes) {
      return padding.left + ((point.chartTime - firstTime) / timeRange) * plotWidth;
    }
    return padding.left + (index / Math.max(parsedPoints.length - 1, 1)) * plotWidth;
  };
  const xForTime = (timestamp) => padding.left + ((timestamp - firstTime) / timeRange) * plotWidth;
  const yFor = (value) => height - padding.bottom - ((value - min) / range) * plotHeight;
  const path = parsedPoints
    .filter((point) => Number.isFinite(point.chartValue))
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(point, index).toFixed(1)} ${yFor(point.chartValue).toFixed(1)}`)
    .join(" ");
  const first = parsedPoints[0]?.timestamp ? shortTime(parsedPoints[0].timestamp) : "-";
  const last = parsedPoints[parsedPoints.length - 1]?.timestamp ? shortTime(parsedPoints[parsedPoints.length - 1].timestamp) : "-";
  const pointWidth = plotWidth / Math.max(parsedPoints.length, 1);
  const barWidth = Math.max(2, Math.min(18, pointWidth * 0.72));
  const entityFaults = faults.filter((fault) => fault.entity_id === timeline.entity_id && fault.enabled !== false);
  const categoryIndex = Object.fromEntries(
    Object.entries(timeline.categories || {}).map(([value, label]) => [String(label), Number(value)]),
  );
  const entityResults = (results || []).filter((result) => result.entity_id === timeline.entity_id && result.is_suspicious);
  const suspicionMarkers = hasValidTimes
    ? entityResults
        .map((result) => {
          const timestamp = Date.parse(result.timestamp);
          if (!Number.isFinite(timestamp) || timestamp < firstTime || timestamp > lastTime) {
            return null;
          }
          const value = chartValueForResult(result, timeline, categoryIndex);
          if (!Number.isFinite(value)) {
            return null;
          }
          return {
            x: xForTime(timestamp),
            y: yFor(Math.min(Math.max(value, min), max)),
            pattern: result.suspected_pattern || result.pattern || "sospecha",
            score: result.suspicion_score,
          };
        })
        .filter(Boolean)
    : [];
  const realtimeBand = hasValidTimes && timeline.realtime_start && timeline.realtime_end
    ? makeTimeBand(timeline.realtime_start, timeline.realtime_end, firstTime, lastTime, padding, width, xForTime)
    : null;
  const faultBands = hasValidTimes
    ? entityFaults
        .map((fault) => {
          const start = Date.parse(fault.start);
          const end = Date.parse(fault.end);
          if (!Number.isFinite(start) || !Number.isFinite(end) || end < firstTime || start > lastTime) {
            return null;
          }
          const x1 = Math.max(padding.left, xForTime(Math.max(start, firstTime)));
          const x2 = Math.min(width - padding.right, xForTime(Math.min(end, lastTime)));
          return { x: x1, width: Math.max(x2 - x1, 2), fault_type: fault.fault_type };
        })
        .filter(Boolean)
    : [];

  return (
    <div className="chart-card">
      <div className="chart-title">
        <div>
          <strong>{timeline.entity_id}</strong>
          <small>{timeline.sensor_kind} / {timeline.area_id || "sin area"}</small>
        </div>
        <span>{formatChartScale(timeline, min, max)}</span>
      </div>
      <div className="chart-range">
        <span>Inicio visible {first}</span>
        <span>Fin visible {last}</span>
        <span>{visiblePoints.length} puntos</span>
      </div>
      <svg className="timeline-svg" viewBox={`0 0 ${width} ${height}`} role="img">
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
        {[0, 0.5, 1].map((tick) => {
          const y = padding.top + tick * plotHeight;
          const label = max - tick * range;
          return (
            <React.Fragment key={tick}>
              <line className="grid-line" x1={padding.left} y1={y} x2={width - padding.right} y2={y} />
              <text className="axis-label" x="8" y={y + 4}>{isBinary ? `${Math.round(label)}%` : formatAxisValue(label)}</text>
            </React.Fragment>
          );
        })}
        {realtimeBand && (
          <rect className="realtime-band" x={realtimeBand.x} y={padding.top} width={realtimeBand.width} height={plotHeight} />
        )}
        {faultBands.map((band, index) => (
          <rect className="fault-band" key={`${band.fault_type}-${index}`} x={band.x} y={padding.top} width={band.width} height={plotHeight} />
        ))}
        {isBinary && parsedPoints.map((point, index) => {
          const value = Number.isFinite(point.chartValue) ? point.chartValue : 0;
          const y = yFor(value);
          return (
            <rect
              className="binary-activation-bar"
              key={`${point.timestamp}-${index}`}
              x={xFor(point, index) - barWidth / 2}
              y={y}
              width={barWidth}
              height={height - padding.bottom - y}
            />
          );
        })}
        {path && <path className={isBinary ? "binary-percentage-line" : ""} d={path} fill="none" />}
        {parsedPoints.map((point, index) =>
          index % Math.max(1, Math.floor(parsedPoints.length / 28)) === 0 ? (
            <circle key={`${point.timestamp}-${index}`} cx={xFor(point, index)} cy={yFor(Number(point.value))} r="2.2" />
          ) : null,
        )}
        {suspicionMarkers.map((marker, index) => (
          <circle
            className="suspicion-marker"
            key={`${marker.pattern}-${index}`}
            cx={marker.x}
            cy={marker.y}
            r="4"
          >
            <title>{`${marker.pattern} / score ${marker.score}`}</title>
          </circle>
        ))}
        <text x={padding.left} y={height - 7}>{first}</text>
        <text x={width - padding.right - 48} y={height - 7}>{last}</text>
      </svg>
      {isBinary && (
        <div className="chart-legend">
          <span>bucket: {formatBucketSeconds(timeline.bucket_seconds)}</span>
          <span>{visiblePoints.length} lapsos visibles</span>
          {realtimeBand && <span>realtime</span>}
          {suspicionMarkers.length > 0 && <span>sospechas: {suspicionMarkers.length}</span>}
          {entityFaults.map((fault) => <span key={`${fault.entity_id}-${fault.fault_type}`}>falla: {fault.fault_type}</span>)}
        </div>
      )}
      {!isBinary && (realtimeBand || suspicionMarkers.length > 0 || entityFaults.length > 0) && (
        <div className="chart-legend">
          {realtimeBand && <span>realtime</span>}
          {suspicionMarkers.length > 0 && <span>sospechas: {suspicionMarkers.length}</span>}
          {entityFaults.map((fault) => <span key={`${fault.entity_id}-${fault.fault_type}`}>falla: {fault.fault_type}</span>)}
        </div>
      )}
      {timeline.sensor_kind === "multistate" && (
        <div className="chart-legend">
          {Object.entries(timeline.categories || {}).map(([value, label]) => (
            <span key={value}>{value}: {label}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function formatKindCounts(counts) {
  return Object.entries(counts)
    .map(([kind, count]) => `${kind}: ${count}`)
    .join(" / ") || "sin datos";
}

function chartValueForResult(result, timeline, categoryIndex) {
  const reading = result.reading || {};
  if (timeline.sensor_kind === "binary") {
    if (reading.active_ratio !== undefined && reading.active_ratio !== null) {
      return Number(reading.active_ratio) * 100;
    }
    return Number(result.value) ? 100 : 0;
  }
  if (timeline.sensor_kind === "continuous") {
    return Number(reading.value ?? result.value ?? result.observed_state);
  }
  const state = String(reading.state ?? result.value ?? result.observed_state);
  return categoryIndex[state];
}

function makeTimeBand(startValue, endValue, firstTime, lastTime, padding, width, xForTime) {
  const start = Date.parse(startValue);
  const end = Date.parse(endValue);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < firstTime || start > lastTime) {
    return null;
  }
  const x1 = Math.max(padding.left, xForTime(Math.max(start, firstTime)));
  const x2 = Math.min(width - padding.right, xForTime(Math.min(end, lastTime)));
  return { x: x1, width: Math.max(x2 - x1, 2) };
}

function formatKindTiming(timings) {
  if (!timings || !Object.keys(timings).length) {
    return "-";
  }
  return Object.entries(timings)
    .map(([kind, ms]) => `${kind}: ${formatElapsed(ms)}`)
    .join(" / ");
}

function formatHistoryStatus(history) {
  if (!history?.exists) {
    return "sin historico";
  }
  if (history.valid === false) {
    return "historico desactualizado";
  }
  const days = history.metadata?.duration_days || "60";
  return `${days} dias / ${history.entity_count || 0} entidades`;
}

function formatHistoryRange(history) {
  if (!history?.exists) {
    return "-";
  }
  return `${shortDate(history.start).slice(0, 10)} a ${shortDate(history.end).slice(0, 10)}`;
}

function formatRealtimeTruth(realtime) {
  const labels = realtime?.ground_truth_labels || {};
  const entries = Object.entries(labels);
  if (!entries.length) {
    return "-";
  }
  return entries.map(([label, count]) => `${label}: ${count}`).join(" / ");
}

function formatElapsed(ms) {
  if (ms === undefined || ms === null) {
    return "-";
  }
  if (ms < 1000) {
    return `${Number(ms).toFixed(1)} ms`;
  }
  return `${(Number(ms) / 1000).toFixed(3)} s`;
}

function formatChartScale(timeline, min, max) {
  if (timeline.sensor_kind === "binary") {
    return `activacion 0-100% / ${formatBucketSeconds(timeline.bucket_seconds)}`;
  }
  if (timeline.sensor_kind === "multistate") {
    return "categorico";
  }
  return `${min.toFixed(1)} - ${max.toFixed(1)}`;
}

function formatAxisValue(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  if (Math.abs(value) >= 100) {
    return Number(value).toFixed(0);
  }
  return Number(value).toFixed(1);
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

function formatReading(result) {
  const reading = result.reading || {};
  if (reading.missing_ticks !== undefined) {
    return `missing ${reading.missing_ticks} ticks / gap ${formatSeconds(reading.gap_seconds)}`;
  }
  if (result.sensor_kind === "continuous") {
    const delta = reading.delta_value !== undefined ? ` d=${formatSigned(reading.delta_value)}` : "";
    const rate = reading.rate_per_minute !== undefined && reading.rate_per_minute !== null ? ` r=${Number(reading.rate_per_minute).toFixed(3)}/min` : "";
    const returned = reading.return_to_normal_seconds !== undefined && reading.return_to_normal_seconds !== null
      ? ` retorno=${formatSeconds(reading.return_to_normal_seconds)}`
      : "";
    return `valor ${reading.value ?? result.observed_state}${delta}${rate}${returned}`;
  }
  if (result.sensor_kind === "binary") {
    return `A=${reading.activations ?? "-"} P=${formatNumber(reading.active_ratio)} T=${formatSeconds(reading.active_seconds)}`;
  }
  if (result.sensor_kind === "multistate") {
    const previous = reading.previous_state ? `${reading.previous_state}->` : "";
    return `${previous}${reading.state ?? result.observed_state}`;
  }
  return result.observed_state || "-";
}

function formatNumber(value) {
  return value === undefined || value === null ? "-" : Number(value).toFixed(2);
}

function formatSigned(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `${number >= 0 ? "+" : ""}${number.toFixed(2)}`;
}

function formatSeconds(value) {
  return value === undefined || value === null ? "-" : `${Number(value).toFixed(0)}s`;
}

function shortDate(value) {
  return value ? value.replace("T", " ").slice(0, 19) : "-";
}

function shortTime(value) {
  const text = shortDate(value);
  return text.includes(" ") ? text.slice(11, 16) : text;
}

createRoot(document.getElementById("root")).render(<App />);
