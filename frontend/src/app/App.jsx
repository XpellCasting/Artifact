import { request } from "../shared/api.js";
import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Download,
  X,
  Database,
  Eye,
  FlaskConical,
  GitBranch,
  Loader2,
  RotateCcw,
  Zap,
} from "lucide-react";
import { downloadCsv } from "../shared/csv.js";
import ReviewWorkspace from "../features/review/ReviewWorkspace.jsx";
import DiagnosticWorkspace from "../features/diagnostics/DiagnosticWorkspace.jsx";
import FlowLab from "../features/flows/FlowLab.jsx";
import { isStatisticalSuspicion, statisticalCounts, episodeLabel } from "../shared/statistical-routing.js";
import { isEnvironmentalResult, contextRoutingLabel, contextRoutingReasonLabel, contextRoutingCounts, contextCoherenceLabel, contextObservationLabel, preciseContextTimestamp, contextCommonCause } from "../shared/context-routing.js";

import { mergeReplayResults, startHistoryUpload, startValidationUpload, HistoryLoadProgress } from "../features/history/history-runtime.jsx";
import { wait, downloadResultsCsv, compareResultsByExecutionDate, entityLabel, timelineDateRange, timelineQueryString, formatHistoryRange, formatSuspicionPercent, normalityDetail, formatElapsed, formatCount } from "../shared/analysis.jsx";
import { Metric, ModuleTab, Panel, StepButton, SummaryRow, Empty } from "../shared/components.jsx";
import { EpisodeGroupingControl, SuspicionThresholdControl, ContinuousConfigEditor, ResultEntityFilters, ResultsTable, CalculationModal, NormalizationModal, TimelineChart } from "../features/statistical/StatisticalWorkspace.jsx";
import { ContextualWorkspace } from "../features/contextual/ContextualWorkspace.jsx";
import { API_BASE_URL, DEFAULT_SUSPICION_THRESHOLD, MIN_SUSPICION_THRESHOLD, MAX_SUSPICION_THRESHOLD } from "../shared/analysis-constants.js";

export async function api(path, options = {}) {
  return request(API_BASE_URL, path, options);
}


export function App() {
  const [health, setHealth] = useState(null);
  const [historyStatus, setHistoryStatus] = useState(null);
  const [entities, setEntities] = useState([]);
  const [timelines, setTimelines] = useState([]);
  const [historicalResults, setHistoricalResults] = useState([]);
  const [runSummary, setRunSummary] = useState(null);
  const [suspicionThreshold, setSuspicionThreshold] = useState(DEFAULT_SUSPICION_THRESHOLD);
  const [suspicionQuietMinutes, setSuspicionQuietMinutes] = useState(30);
  const [chartStart, setChartStart] = useState("");
  const [chartEnd, setChartEnd] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [historyLoad, setHistoryLoad] = useState(null);
  const [selectedResult, setSelectedResult] = useState(null);
  const [normalizationPreview, setNormalizationPreview] = useState(null);
  const [activeModule, setActiveModule] = useState("estadistico");
  const [classifierVisited, setClassifierVisited] = useState(false);
  const [resultEntityFilterIds, setResultEntityFilterIds] = useState(null);
  const [continuousConfig, setContinuousConfig] = useState([]);
  const [contextMap, setContextMap] = useState(null);
  const [contextResults, setContextResults] = useState([]);
  const [contextSummary, setContextSummary] = useState(null);
  const [contextError, setContextError] = useState("");
  const [validationReplay, setValidationReplay] = useState(null);
  const [validationUpload, setValidationUpload] = useState(null);

  function updateSuspicionThreshold(percent) {
    const parsed = Number(percent);
    if (!Number.isFinite(parsed)) return;
    const normalized = Math.min(
      MAX_SUSPICION_THRESHOLD,
      Math.max(MIN_SUSPICION_THRESHOLD, parsed / 100),
    );
    setSuspicionThreshold(Math.round(normalized * 1000) / 1000);
  }

  const analyzedSuspicionThreshold = Number(runSummary?.suspicion_threshold);
  const visibleSuspicionThreshold = Number.isFinite(analyzedSuspicionThreshold)
    ? analyzedSuspicionThreshold
    : suspicionThreshold;
  const [validationAutoRunning, setValidationAutoRunning] = useState(false);
  const continuousEntities = useMemo(
    () => entities.filter((entity) => entity.sensor_kind === "continuous"),
    [entities],
  );

  const binaryEntities = useMemo(
    () => entities.filter((entity) => entity.sensor_kind === "binary"),
    [entities],
  );

  const analyzableEntities = useMemo(
    () => entities.filter((entity) => ["continuous", "binary", "multistate"].includes(entity.sensor_kind)),
    [entities],
  );

  const allAnalyzedResults = useMemo(() => {
    return historicalResults
      .filter((result) => ["continuous", "binary", "multistate"].includes(result.sensor_kind));
  }, [historicalResults]);

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

  const statisticalSummary = statisticalCounts(allAnalyzedResults, runSummary || {});
  const suspiciousCount = statisticalSummary.detected;

  useEffect(() => {
    runStep("init", async () => {
      setHealth(await api("/health"));
      const status = await api("/history/static/status?autoload=false");
      if (status.history_load && status.history_load.status !== "completed") {
        setActiveModule("evaluacion");
        const completed = await monitorHistoryLoad(status.history_load);
        await finishHistoryLoad(completed);
      } else {
        applyHistoryStatus(status);
        if (status.exists) await loadTimeline("", "");
      }
      const replay = await api("/validation/replay/status");
      applyValidationPipeline(replay);
      if (replay?.loaded) {
        await loadValidationResultPages();
      }
    });
  }, []);

  useEffect(() => {
    if (activeModule === "contextual") {
      loadContextualData();
    }
  }, [activeModule]);


  useEffect(() => {
    if (!validationAutoRunning) {
      return undefined;
    }
    let cancelled = false;
    let timer;
    const controller = new AbortController();

    const poll = async () => {
      try {
        const payload = await api("/validation/replay/status", {
          signal: controller.signal,
        });
        if (cancelled) {
          return;
        }
        applyValidationPipeline(payload);
        if (payload.status === "running") {
          timer = window.setTimeout(poll, 900);
        } else {
          setValidationAutoRunning(false);
          await loadValidationResultPages();
          if (payload.status === "failed") {
            setError(payload.last_error || "La reproduccion de Evaluacion fallo.");
          }
        }
      } catch (err) {
        if (!cancelled && err.name !== "AbortError") {
          setValidationAutoRunning(false);
          setError(err.message);
        }
      }
    };

    timer = window.setTimeout(poll, 500);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [validationAutoRunning]);

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

  function applyHistoryStatus(status) {
    setHistoryStatus(status);
    const nextEntities = status?.entities || [];
    setEntities(nextEntities);
  }

  function clearContextualData({ keepMap = false } = {}) {
    if (!keepMap) {
      setContextMap(null);
    }
    setContextResults([]);
    setContextSummary(null);
    setContextError("");
  }

  function applyContextualPayload(payload) {
    const embedded = payload?.contextual;
    const nextResults = Array.isArray(payload?.historical_contextual_results)
      ? payload.historical_contextual_results
      : Array.isArray(payload?.contextual_results)
        ? payload.contextual_results
      : Array.isArray(embedded?.results)
        ? embedded.results
        : null;
    if (nextResults !== null) {
      setContextResults(nextResults);
    }
    if (embedded?.summary) {
      setContextSummary(embedded.summary);
    } else if (payload?.contextual_summary) {
      setContextSummary(payload.contextual_summary);
    } else if (payload?.summary?.contextual) {
      setContextSummary(payload.summary.contextual);
    }
  }

  async function loadRealHistory() {
    await runStep("real-history", async () => {
      setHistoryLoad(null);
      const job = await api("/history/real/load/start", { method: "POST" });
      const completed = await monitorHistoryLoad(job);
      await finishHistoryLoad(completed);
    });
  }

  async function uploadRealHistory(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await runStep("real-upload", async () => {
      setHistoryLoad({
        status: "uploading",
        phase: "uploading",
        progress: 0,
        message: `Subiendo ${file.name}`,
        file_size_bytes: file.size,
      });
      const job = await startHistoryUpload(file, (loaded, total) => {
        setHistoryLoad({
          status: "uploading",
          phase: "uploading",
          progress: Math.min(8, (loaded / Math.max(1, total)) * 8),
          message: `Subiendo ${file.name}`,
          file_size_bytes: total,
          uploaded_bytes: loaded,
        });
      });
      const completed = await monitorHistoryLoad(job, { uploadOffset: true });
      await finishHistoryLoad(completed);
    });
    event.target.value = "";
  }

  async function monitorHistoryLoad(initialJob, { uploadOffset = false } = {}) {
    let current = initialJob;
    while (true) {
      const displayed = uploadOffset
        ? { ...current, progress: 8 + Math.min(91, Number(current.progress || 0) * 0.91) }
        : current;
      if (current.status === "failed") {
        setHistoryLoad(displayed);
        throw new Error(current.error || current.message || "La carga del historico fallo.");
      }
      if (current.status === "completed") {
        return displayed;
      }
      setHistoryLoad(displayed);
      await wait(750);
      current = await api(`/history/real/load/jobs/${current.job_id}`);
    }
  }

  async function finishHistoryLoad(completedJob) {
    const status = completedJob.history;
    if (!status?.exists) {
      throw new Error("La API finalizo la carga sin entregar un historico valido.");
    }
    applyHistoryStatus(status);
    setHistoricalResults([]);
    setValidationReplay(null);
    setValidationUpload(null);
    setValidationAutoRunning(false);
    setContinuousConfig([]);
    setRunSummary(null);
    clearContextualData();
    setChartStart("");
    setChartEnd("");
    setHistoryLoad({
      ...completedJob,
      status: "running",
      phase: "building_timeline",
      progress: 99,
      message: "Preparando la vista temporal",
    });
    await loadTimeline("", "");
    setHistoryLoad({
      ...completedJob,
      status: "completed",
      phase: "ready",
      progress: 100,
      message: "Historico listo para utilizar",
    });
  }

  async function analyzeHistory() {
    await runStep("analyze-history", async () => {
      const payload = await api("/history/analyze", {
        method: "POST",
        body: JSON.stringify({ suspicion_threshold: suspicionThreshold,
          suspicion_quiet_seconds: suspicionQuietMinutes * 60 }),
      });
      setHistoricalResults((payload.results || []).filter((result) => ["continuous", "binary", "multistate"].includes(result.sensor_kind)));
      setRunSummary(payload.summary || {});
      applyContextualPayload(payload);
    });
  }

  async function loadContextualData() {
    return runStep("context-load", async () => {
      setContextError("");
      const [mapResponse, resultsResponse] = await Promise.allSettled([
        api("/context/map"),
        api("/context/results"),
      ]);
      const messages = [];
      if (mapResponse.status === "fulfilled") {
        setContextMap(mapResponse.value);
      } else {
        messages.push(`Mapa: ${mapResponse.reason?.message || "no disponible"}`);
      }
      if (resultsResponse.status === "fulfilled") {
        setContextResults(resultsResponse.value?.results || []);
        setContextSummary(resultsResponse.value?.summary || {});
      } else {
        messages.push(`Resultados: ${resultsResponse.reason?.message || "no disponibles"}`);
      }
      setContextError(messages.join(" / "));
      if (mapResponse.status === "rejected" && resultsResponse.status === "rejected") {
        throw new Error(messages.join(" / "));
      }
      return {
        map: mapResponse.status === "fulfilled" ? mapResponse.value : null,
        results: resultsResponse.status === "fulfilled" ? resultsResponse.value : null,
      };
    });
  }

  async function rebuildContextMap() {
    return runStep("context-rebuild", async () => {
      setContextError("");
      const nextMap = await api("/context/map/rebuild", { method: "POST" });
      setContextMap(nextMap);
      try {
        const nextResults = await api("/context/results");
        setContextResults(nextResults.results || []);
        setContextSummary(nextResults.summary || {});
      } catch (err) {
        setContextError(`Mapa reconstruido; resultados no disponibles: ${err.message}`);
      }
      return nextMap;
    });
  }

  async function openNormalizationPreview() {
    await runStep("normalization-preview", async () => {
      const payload = await api("/history/normalization/preview");
      setNormalizationPreview(payload);
    });
  }

  async function loadContinuousConfig() {
    await runStep("continuous-config", async () => {
      const payload = await api("/config/continuous");
      setContinuousConfig(payload.configs || []);
    });
  }

  async function saveContinuousConfig() {
    await runStep("save-continuous-config", async () => {
      const payload = await api("/config/continuous", {
        method: "POST",
        body: JSON.stringify({ configs: continuousConfig }),
      });
      setContinuousConfig(payload.configs || continuousConfig);
      setHistoricalResults([]);
      setRunSummary(null);
      clearContextualData({ keepMap: true });
    });
  }

  function updateContinuousConfig(entityId, key, value) {
    setContinuousConfig((current) =>
      current.map((item) =>
        item.entity_id === entityId ? { ...item, values: { ...(item.values || {}), [key]: Number(value) } } : item,
      ),
    );
  }

  async function loadTimeline(startValue = chartStart, endValue = chartEnd) {
    const payload = await api(`/history/timeline${timelineQueryString(startValue, endValue)}`);
    const nextTimelines = (payload.entities || []).filter((timeline) => ["continuous", "binary", "multistate"].includes(timeline.sensor_kind));
    setTimelines(nextTimelines);
    if (!startValue || !endValue) {
      const range = timelineDateRange(nextTimelines);
      setChartStart(range.start);
      setChartEnd(range.end);
    }
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

  function applyValidationPipeline(payload) {
    setValidationReplay(payload);
    setValidationAutoRunning(payload?.status === "running");
    if (!payload?.loaded) {
      return;
    }
    const incomingStatistical = payload.new_results?.length
      ? payload.new_results
      : payload.recent_results?.length
        ? payload.recent_results
        : payload.results || [];
    const incomingContextual = payload.new_contextual_results?.length
      ? payload.new_contextual_results
      : payload.recent_contextual_results?.length
        ? payload.recent_contextual_results
        : payload.contextual_results || [];
    if (incomingStatistical.length) {
      setHistoricalResults((current) => mergeReplayResults(current, incomingStatistical));
    }
    if (incomingContextual.length) {
      setContextResults((current) => mergeReplayResults(current, incomingContextual));
    }
    setContextSummary(payload.contextual_summary || null);
    if (payload.last_batch) {
      setRunSummary({
        elapsed_ms: payload.last_batch.processing_elapsed_ms,
        timings_by_kind_ms: payload.last_batch.timings_by_kind_ms || {},
        cache_hit: false,
        source: "evaluation_replay",
        suspicion_threshold: payload.suspicion_threshold,
        suspicion_quiet_seconds: payload.suspicion_quiet_seconds,
        grouping: payload.pipeline_summary?.grouping || {
          detected: payload.detection_count,
          forwarded: payload.contextual_count,
          suppressed: 0,
        },
      });
    }
  }

  async function loadValidationResultPages() {
    const [statistical, contextual] = await Promise.all([
      api("/validation/replay/results?kind=statistical&offset=0&limit=500"),
      api("/validation/replay/results?kind=contextual&offset=0&limit=500"),
    ]);
    setHistoricalResults(statistical.items || []);
    setContextResults(contextual.items || []);
  }

  async function uploadLabelledEvaluation(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setValidationAutoRunning(false);
    await runStep("labelled-upload", async () => {
      const response = await fetch(`${API_BASE_URL}/validation/labelled/upload`, {
        method: "POST", headers: { "Content-Type": "text/csv", "X-Filename": encodeURIComponent(file.name) }, body: file,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "No se pudo cargar el conjunto etiquetado.");
      setHistoricalResults([]); setContextResults([]); setContextSummary(null);
      applyValidationPipeline(payload);
      setValidationUpload({ status: "completed", progress: 100, message: "Datos etiquetados cargados; eventos futuros preparados sin etiquetas", file_size_bytes: file.size });
      return payload;
    });
    event.target.value = "";
  }

  async function uploadValidationHistory(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setValidationAutoRunning(false);
    setHistoricalResults([]);
    setContextResults([]);
    setContextSummary(null);
    const completedUpload = await runStep("validation-upload", async () => {
      setValidationUpload({
        status: "uploading",
        progress: 0,
        message: `Subiendo ${file.name}`,
        file_size_bytes: file.size,
      });
      const payload = await startValidationUpload(
        file,
        (loaded, total) => setValidationUpload({
          status: "uploading",
          progress: Math.min(90, (loaded / Math.max(total, 1)) * 90),
          message: `Subiendo ${file.name}`,
          file_size_bytes: total,
          uploaded_bytes: loaded,
        }),
        () => setValidationUpload({
          status: "processing",
          progress: 94,
          message: "Archivo recibido; validando y ordenando eventos de Evaluacion",
          file_size_bytes: file.size,
          uploaded_bytes: file.size,
        }),
      );
      applyValidationPipeline(payload);
      setValidationUpload({
        status: "completed",
        progress: 100,
        message: "Evaluacion lista para reproducir",
        file_size_bytes: file.size,
      });
      return payload;
    });
    if (!completedUpload) {
      setValidationUpload((current) => ({
        ...(current || {}),
        status: "failed",
        message: "No se pudo cargar el CSV de Evaluacion",
      }));
    }
    event.target.value = "";
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <span className="eyebrow"></span>
          <h1>Filtro estadistico-temporal</h1>
        </div>
      </header>

      {error && (
        <section className="alert">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </section>
      )}

      <nav className="module-tabs" aria-label="Módulos del artefacto">
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
          label="Clasificador diagnóstico"
          onClick={() => { setClassifierVisited(true); setActiveModule("clasificador"); }}
        />
        <ModuleTab
          active={activeModule === "revision"}
          icon={<Eye size={17} />}
          label="Analisis y etiquetado"
          onClick={() => setActiveModule("revision")}
        />
        <ModuleTab active={activeModule === "evaluacion"} icon={<BarChart3 size={17} />} label="Evaluación" onClick={() => setActiveModule("evaluacion")} />
        <ModuleTab active={activeModule === "flujos"} icon={<GitBranch size={17} />} label="Evaluación de flujos" onClick={() => setActiveModule("flujos")} />
      </nav>

      {activeModule === "flujos" ? (
        <FlowLab apiBaseUrl={API_BASE_URL} historyReady={Boolean(historyStatus?.exists && historyStatus?.future_evaluation_ready)} replay={validationReplay} upload={validationUpload} uploadBusy={Boolean(busy)} onUpload={uploadValidationHistory} onLabelledUpload={uploadLabelledEvaluation} />
      ) : activeModule === "estadistico" ? (
        <>
          <section className="metric-grid">
            <Metric icon={<Database />} label="Datos historicos" value={formatCount(historyStatus?.rows)} detail={formatHistoryRange(historyStatus)} />
            <Metric icon={<Activity />} label="Entidades historico" value={formatCount(historyStatus?.entity_count || entities.length)} detail={`${continuousEntities.length} continuas / ${binaryEntities.length} binarias`} />
            <Metric icon={<Database />} label="Base normalidad" value={formatCount(historyStatus?.normality_history?.rows)} detail={normalityDetail(historyStatus?.normality_history)} />
            <Metric icon={<BarChart3 />} label="Sospechas estadisticas" value={suspiciousCount} detail={`score >= ${visibleSuspicionThreshold.toFixed(2)} (${formatSuspicionPercent(visibleSuspicionThreshold)})`} />
          </section>

          <SuspicionThresholdControl
            value={suspicionThreshold}
            analyzedValue={Number.isFinite(analyzedSuspicionThreshold) ? analyzedSuspicionThreshold : null}
            busy={Boolean(busy)}
            onChange={updateSuspicionThreshold}
          />
          <EpisodeGroupingControl
            minutes={suspicionQuietMinutes}
            onChange={setSuspicionQuietMinutes}
            busy={Boolean(busy)}
            summary={statisticalSummary}
            replay={validationReplay}
          />

          <section className="charts-panel">
            <div className="section-heading">
              <div>
                <h2>Datos reales por fecha</h2>
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
                  suspicionThreshold={visibleSuspicionThreshold}
                  onResultClick={setSelectedResult}
                />
              ))}
              {!timelines.length && <Empty text="Carga un conjunto de datos real para ver los charts." />}
            </div>
          </section>

          <section className="workspace-grid lower single">
            <Panel title="Resultados del filtro estadistico">
              <div className="summary-strip">
                <SummaryRow label="Tiempo total filtro" value={formatElapsed(runSummary?.elapsed_ms)} />
                <SummaryRow label="Tiempo continuas" value={formatElapsed(runSummary?.timings_by_kind_ms?.continuous)} />
                <SummaryRow label="Tiempo binarias" value={formatElapsed(runSummary?.timings_by_kind_ms?.binary)} />
                <SummaryRow label="Tiempo multistate" value={formatElapsed(runSummary?.timings_by_kind_ms?.multistate)} />
                <SummaryRow
                  label="Multistate elegibles"
                  value={runSummary ? `${runSummary.multistate_eligibility?.eligible_entities || 0} / ${runSummary.multistate_eligibility?.entities || 0}` : "-"}
                />
                <SummaryRow label="Cache analisis" value={runSummary ? (runSummary.cache_hit ? "reutilizado" : "nuevo") : "-"} />
                <SummaryRow label="Enviadas a contextual" value={formatCount(statisticalSummary.forwarded)} />
                <SummaryRow label="Repeticiones agrupadas" value={formatCount(statisticalSummary.suppressed)} />
                <SummaryRow label="Resultados visibles / total" value={`${formatCount(displayResults.length)} / ${formatCount(statisticalSummary.detected)}`} />
                <button className="secondary-action inline" onClick={() => downloadResultsCsv(displayResults)} disabled={!displayResults.length}>
                  <Download size={17} />
                  Descargar tabla en CSV
                </button>
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
      ) : activeModule === "evaluacion" ? (
        <>
          <section className="workspace-grid lower single">
            <Panel title="Conjunto de datos">
              <p>El histórico se conserva y se recupera automáticamente al reiniciar el servidor. Puedes subir otro archivo para reemplazarlo.</p>
              <div className="history-summary">
                <SummaryRow label="API" value={health?.status || "-"} />
                <SummaryRow label="Origen" value={historyStatus?.source_type || "-"} />
                <SummaryRow label="Filas" value={formatCount(historyStatus?.rows)} />
                <SummaryRow label="Filas normalidad" value={formatCount(historyStatus?.normality_history?.rows)} />
                <SummaryRow label="Entidades" value={formatCount(historyStatus?.entity_count || entities.length)} />
                <SummaryRow label="Base de normalidad" value={historyStatus?.baseline_id ? historyStatus.baseline_id.slice(0, 16) : "Sin procedencia registrada"} />
                <SummaryRow label="Rango" value={formatHistoryRange(historyStatus)} />
              </div>
              <div className="button-row">
                <StepButton id="real-history" busy={busy} onClick={loadRealHistory} icon={<Database size={17} />} label="Cargar historico" />
                <label className="file-action">
                  Subir historico
                  <input type="file" accept=".csv,text/csv" onChange={uploadRealHistory} disabled={Boolean(busy)} />
                </label>
                <StepButton id="analyze-history" busy={busy} onClick={analyzeHistory} icon={<BarChart3 size={17} />} label="Analizar historico" disabled={!historyStatus?.exists} />
                <StepButton id="normalization-preview" busy={busy} onClick={openNormalizationPreview} icon={<Eye size={17} />} label="Ver normalizacion" disabled={!historyStatus?.exists} />
                <StepButton id="continuous-config" busy={busy} onClick={loadContinuousConfig} icon={<Eye size={17} />} label="Parametros continuos" disabled={!historyStatus?.exists} />
              </div>
              {historyLoad && <HistoryLoadProgress job={historyLoad} />}
              {continuousConfig.length > 0 && (
                <ContinuousConfigEditor
                  configs={continuousConfig}
                  busy={busy}
                  onChange={updateContinuousConfig}
                  onSave={saveContinuousConfig}
                />
              )}
              <div className="entity-list compact">
                {analyzableEntities.map((entity) => (
                  <div className="entity-row" key={entity.entity_id}>
                    <span>
                      <strong>{entity.entity_id}</strong>
                      <small>{entityLabel(entity)} / {entity.unit_of_measurement || entity.unit || "sin unidad"}</small>
                    </span>
                  </div>
                ))}
                {!analyzableEntities.length && <Empty text="No hay entidades continuas, binarias o multivaloricas cargadas." />}
              </div>
            </Panel>
          </section>
        </>
      ) : activeModule === "contextual" ? (
        <ContextualWorkspace
          mapData={contextMap}
          results={contextResults}
          summary={contextSummary}
          busy={busy}
          error={contextError}
          onReload={loadContextualData}
          onRebuild={rebuildContextMap}
        />
      ) : activeModule === "revision" ? (
        <ReviewWorkspace apiBaseUrl={API_BASE_URL} />
      ) : (
        null
      )}
      <div hidden={activeModule !== "clasificador"}>
        {classifierVisited && <DiagnosticWorkspace apiBaseUrl={API_BASE_URL} />}
      </div>
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

