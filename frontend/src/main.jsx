import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Download,
  Droplets,
  X,
  Database,
  Eye,
  FlaskConical,
  Loader2,
  Play,
  Power,
  RotateCcw,
  Thermometer,
  Zap,
} from "lucide-react";
import { downloadCsv } from "./csv.js";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const RED_SUSPICION_SCORE = 0.60;
const MARKER_GROUP_GAP_MS = 15 * 60 * 1000;

const CONTINUOUS_FAULT_TYPES = ["random", "malfunction", "bias", "drift", "missing"];
const BINARY_FAULT_TYPES = ["random", "malfunction", "bias", "missing"];

const EQUATIONS = [
  {
    id: "hampel",
    title: "Hampel / limite operacional",
    pattern: "isolated_spike",
    equation: "H_i(t) = |x_i(t) - mediana(W)| / (1.4826 * MAD(W) + epsilon)",
    score: "s_H = max(phi_z(H_i), phi_z(Z_F_i))",
  },
  {
    id: "robust_rate_change",
    title: "Tasa de cambio robusta",
    pattern: "abrupt_change",
    equation: "R_i(t) = |x_i(t) - x_i(t-Delta t)| / Delta t",
    score: "s_R = phi_z(Z_R_i(t))",
  },
  {
    id: "cusum_residual",
    title: "CUSUM sobre residuales",
    pattern: "persistent_shift",
    equation: "r_i(t) = x_i(t) - pEWMA_i(t-Delta t); C+ y C- acumulan residuales",
    score: "s_C = phi_ratio(max(C+, C-) / theta_C)",
  },
  {
    id: "constant_persistence",
    title: "Persistencia constante",
    pattern: "stuck_value",
    equation: "D_const mide el tiempo consecutivo con |x_i(t)-x_i(t-Delta t)| <= epsilon_const",
    score: "s_D_const = max(phi_z(Z_D_const), phi_ratio(D_const / T_const))",
  },
  {
    id: "reading_gap",
    title: "Brecha temporal",
    pattern: "missing_gap",
    equation: "M_i(t) = t - t_i_last",
    score: "s_M = max(phi_z(Z_M), phi_ratio(M_i / T_M))",
  },
  {
    id: "dspot_extreme",
    title: "DSPOT extremo",
    pattern: "extreme_value",
    equation: "Z_DSPOT = distancia del valor contra el umbral extremo robusto",
    score: "s_DSPOT = phi_z(Z_DSPOT)",
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
  {
    id: "time_since_last_change",
    title: "Tiempo desde ultimo cambio",
    pattern: "state_persistence",
    equation: "TLC_i(t) = t - t_last_change",
    score: "s_TLC = min(1, Z_TLC / tau_TLC)",
  },
  {
    id: "z_duration_score",
    title: "Z-score de duracion",
    pattern: "state_persistence",
    equation: "Z_D_i = Z_D_on",
    score: "s_D = min(1, Z_D_i / tau_D)",
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

function downloadResultsCsv(results) {
  downloadCsv([
    ["Entidad", "Tiempo mayor-menor", "Score", "Raw", "Lectura", "Patron", "Indicadores", "Ecuacion"],
    ...results.map((result) => [
      result.entity_id,
      shortDate(result.timestamp),
      formatNumber(result.suspicion_score),
      formatNumber(result.raw_score),
      formatReading(result),
      [result.suspected_pattern || result.pattern || "-", timeWindowLabel(result.time_window)].filter(Boolean).join(" / "),
      (result.triggered_indicators || []).join(", ") || "-",
      equationNames(result.triggered_indicators),
    ]),
  ], "resultados_filtro_estadistico.csv");
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
  const [continuousConfig, setContinuousConfig] = useState([]);
  const [manualStates, setManualStates] = useState({});
  const [isManualRunning, setIsManualRunning] = useState(false);
  const [manualTickSeconds, setManualTickSeconds] = useState(3600);
  const [contextMap, setContextMap] = useState(null);
  const [contextResults, setContextResults] = useState([]);
  const [contextSummary, setContextSummary] = useState(null);
  const [contextError, setContextError] = useState("");
  const [classifierResults, setClassifierResults] = useState([]);
  const [classifierSummary, setClassifierSummary] = useState(null);
  const [classifierEvaluation, setClassifierEvaluation] = useState(null);
  const [classifierSpecification, setClassifierSpecification] = useState(null);
  const [classifierError, setClassifierError] = useState("");
  const isSimulatorRoute = window.location.pathname.startsWith("/simulador");

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

  const temperatureEntities = useMemo(
    () => continuousEntities.filter(isTemperatureEntity),
    [continuousEntities],
  );

  const humidityEntities = useMemo(
    () => continuousEntities.filter(isHumidityEntity),
    [continuousEntities],
  );

  const manualEntities = useMemo(
    () => [...temperatureEntities, ...humidityEntities, ...binaryEntities],
    [temperatureEntities, humidityEntities, binaryEntities],
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

  useEffect(() => {
    setManualStates((current) => syncManualStates(manualEntities, current));
  }, [manualEntities]);

  useEffect(() => {
    if (!isManualRunning) {
      return undefined;
    }
    const interval = window.setInterval(() => {
      sendManualPulse({ quiet: true });
    }, 1000);
    return () => window.clearInterval(interval);
  }, [isManualRunning, manualStates, manualTickSeconds, suspicionThreshold, historyStatus, entities]);

  useEffect(() => {
    if (isSimulatorRoute && historyStatus?.exists && manualEntities.length && !timelines.length) {
      loadTimeline("", "");
    }
  }, [isSimulatorRoute, historyStatus?.exists, manualEntities.length, timelines.length]);

  useEffect(() => {
    if (!isSimulatorRoute && activeModule === "contextual") {
      loadContextualData();
    }
  }, [activeModule, isSimulatorRoute]);

  useEffect(() => {
    if (!isSimulatorRoute && activeModule === "clasificador") {
      loadClassifierData();
    }
  }, [activeModule, isSimulatorRoute]);

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

  function clearContextualData({ keepMap = false } = {}) {
    if (!keepMap) {
      setContextMap(null);
    }
    setContextResults([]);
    setContextSummary(null);
    setContextError("");
    clearClassifierData();
  }

  function clearClassifierData({ keepSpecification = false } = {}) {
    setClassifierResults([]);
    setClassifierSummary(null);
    setClassifierEvaluation(null);
    setClassifierError("");
    if (!keepSpecification) {
      setClassifierSpecification(null);
    }
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

  function applyClassifierPayload(payload, { replaceResults = false, source = "" } = {}) {
    const embedded = payload?.classification;
    const nextResults = Array.isArray(payload?.historical_classification_results)
      ? payload.historical_classification_results
      : Array.isArray(payload?.classification_results)
        ? payload.classification_results
        : Array.isArray(embedded?.results)
          ? embedded.results
          : null;
    if (nextResults !== null) {
      const taggedResults = source
        ? nextResults.map((result) => result?.source ? result : { ...result, source })
        : nextResults;
      const shouldReplace = replaceResults || Array.isArray(payload?.historical_classification_results);
      setClassifierResults((current) => (
        shouldReplace ? taggedResults : mergeClassifierResults(current, taggedResults)
      ));
    }
    if (embedded?.summary) {
      setClassifierSummary(embedded.summary);
    } else if (payload?.classification_summary) {
      setClassifierSummary(payload.classification_summary);
    } else if (payload?.summary?.classification) {
      setClassifierSummary(payload.summary.classification);
    }
    if (embedded?.model) {
      setClassifierSpecification((current) => ({
        ...normalizeClassifierSpecificationPayload(current),
        ...normalizeClassifierSpecificationPayload(embedded.model),
      }));
    }
    if (embedded?.specification) {
      setClassifierSpecification((current) => ({
        ...normalizeClassifierSpecificationPayload(current),
        ...normalizeClassifierSpecificationPayload(embedded.specification),
      }));
    }
    if (embedded?.evaluation) {
      setClassifierEvaluation(embedded.evaluation);
    }
  }

  async function loadStaticHistory() {
    await runStep("static-history", async () => {
      const status = await api("/history/static/ensure", { method: "POST" });
      applyHistoryStatus(status);
      setHistoricalResults([]);
      setContinuousConfig([]);
      setRealtime(null);
      clearContextualData();
      setChartStart("");
      setChartEnd("");
      await loadTimeline("", "");
    });
  }

  async function loadRealHistory() {
    await runStep("real-history", async () => {
      const status = await api("/history/real/load", { method: "POST" });
      applyHistoryStatus(status);
      setHistoricalResults([]);
      setContinuousConfig([]);
      setRunSummary(null);
      setRealtime(null);
      clearContextualData();
      setChartStart("");
      setChartEnd("");
      await loadTimeline("", "");
    });
  }

  async function uploadRealHistory(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await runStep("real-upload", async () => {
      const status = await api("/history/real/upload", {
        method: "POST",
        headers: { "Content-Type": "text/csv" },
        body: await file.arrayBuffer(),
      });
      applyHistoryStatus(status);
      setHistoricalResults([]);
      setContinuousConfig([]);
      setRunSummary(null);
      setRealtime(null);
      clearContextualData();
      setChartStart("");
      setChartEnd("");
      await loadTimeline("", "");
    });
    event.target.value = "";
  }

  async function analyzeHistory() {
    await runStep("analyze-history", async () => {
      const payload = await api("/history/analyze", {
        method: "POST",
        body: JSON.stringify({ suspicion_threshold: suspicionThreshold }),
      });
      setHistoricalResults((payload.results || []).filter((result) => ["continuous", "binary"].includes(result.sensor_kind)));
      setRunSummary(payload.summary || {});
      applyContextualPayload(payload);
      applyClassifierPayload(payload, { replaceResults: true, source: "historical" });
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
      clearClassifierData();
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

  async function loadClassifierData() {
    return runStep("classifier-load", async () => {
      setClassifierError("");
      const [resultsResponse, evaluationResponse, specificationResponse, mapResponse] = await Promise.allSettled([
        api("/classifier/results"),
        api("/classifier/evaluation"),
        api("/classifier/specification"),
        api("/context/map"),
      ]);
      const messages = [];
      if (resultsResponse.status === "fulfilled") {
        const payload = resultsResponse.value || {};
        setClassifierResults(payload.results || []);
        setClassifierSummary(payload.summary || {});
        if (payload.model && specificationResponse.status !== "fulfilled") {
          setClassifierSpecification(normalizeClassifierSpecificationPayload(payload.model));
        }
      } else {
        messages.push(`Diagnosticos: ${resultsResponse.reason?.message || "no disponibles"}`);
      }
      if (evaluationResponse.status === "fulfilled") {
        setClassifierEvaluation(evaluationResponse.value || {});
      } else {
        messages.push(`Evaluacion: ${evaluationResponse.reason?.message || "no disponible"}`);
      }
      if (specificationResponse.status === "fulfilled") {
        setClassifierSpecification(normalizeClassifierSpecificationPayload(specificationResponse.value));
      } else {
        messages.push(`Modelo: ${specificationResponse.reason?.message || "no disponible"}`);
      }
      if (mapResponse.status === "fulfilled") {
        setContextMap(mapResponse.value || null);
      } else {
        messages.push(`Mapa contextual: ${mapResponse.reason?.message || "no disponible"}`);
      }
      setClassifierError(messages.join(" / "));
      if (
        resultsResponse.status === "rejected"
        && evaluationResponse.status === "rejected"
        && specificationResponse.status === "rejected"
        && mapResponse.status === "rejected"
      ) {
        throw new Error(messages.join(" / "));
      }
      return {
        results: resultsResponse.status === "fulfilled" ? resultsResponse.value : null,
        evaluation: evaluationResponse.status === "fulfilled" ? evaluationResponse.value : null,
        specification: specificationResponse.status === "fulfilled" ? specificationResponse.value : null,
        map: mapResponse.status === "fulfilled" ? mapResponse.value : null,
      };
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

  async function resetRealtime() {
    await runStep("reset", async () => {
      const payload = await api("/realtime/reset", { method: "POST" });
      setRealtime(payload);
      setRunSummary(null);
      setIterationsDone(0);
      clearClassifierData();
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
      applyContextualPayload(payload);
      applyClassifierPayload(payload, { source: "realtime" });
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
      applyContextualPayload(payload);
      applyClassifierPayload(payload, { source: "realtime" });
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

  function updateManualState(entityId, value) {
    setManualStates((current) => ({ ...current, [entityId]: value }));
  }

  async function sendManualPulse({ quiet = false } = {}) {
    const action = async () => {
      let sourceEntities = entities;
      if (!historyStatus?.exists || historyStatus?.valid === false) {
        const status = await api("/history/static/ensure", { method: "POST" });
        applyHistoryStatus(status, { keepSelection: true });
        sourceEntities = status.entities || [];
      }
      const controllable = controlledEntities(sourceEntities);
      if (!controllable.length) {
        throw new Error("No hay sensores de temperatura, humedad o binarios para simular.");
      }
      const states = syncManualStates(controllable, manualStates);
      setManualStates(states);
      const payload = await api("/manual/tick", {
        method: "POST",
        body: JSON.stringify({
          selected_entities: controllable.map((entity) => entity.entity_id),
          sensor_states: Object.fromEntries(controllable.map((entity) => [entity.entity_id, states[entity.entity_id]])),
          tick_seconds: manualTickSeconds,
          suspicion_threshold: suspicionThreshold,
        }),
      });
      setRealtime(payload);
      setRunSummary(payload.summary || {});
      applyContextualPayload(payload);
      applyClassifierPayload(payload, { source: "realtime" });
      await loadTimeline("", "");
      return payload;
    };
    if (quiet) {
      try {
        return await action();
      } catch (err) {
        setError(err.message);
        setIsManualRunning(false);
        return null;
      }
    }
    return runStep("manual-pulse", action);
  }

  if (isSimulatorRoute) {
    return (
      <main className="app-shell">
        <header className="app-header">
          <div>
            <span className="eyebrow">Simulador Home Assistant</span>
            <h1>Sensores detectados enviando senales al filtro estadistico</h1>
            <p>
              Ajusta temperatura, humedad y sensores binarios. El envio respeta la cadencia historica estimada por entidad.
            </p>
          </div>
          <div className="header-actions">
            <a className="secondary-action" href="/" target="_blank" rel="noreferrer">Abrir filtro</a>
            <button className="primary-action" onClick={() => sendManualPulse()} disabled={Boolean(busy) || !manualEntities.length}>
              {busy === "manual-pulse" ? <Loader2 className="spin" size={18} /> : <Zap size={18} />}
              Enviar pulso
            </button>
            <button
              className="secondary-action"
              onClick={() => setIsManualRunning((running) => !running)}
              disabled={!manualEntities.length}
            >
              {isManualRunning ? <RotateCcw size={18} /> : <Play size={18} />}
              {isManualRunning ? "Pausar" : "Iniciar envio"}
            </button>
          </div>
        </header>

        {error && (
          <section className="alert">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </section>
        )}

        <section className="metric-grid simulator-metrics">
          <Metric icon={<Database />} label="Historico" value={formatCount(historyStatus?.rows)} detail={formatHistoryRange(historyStatus)} />
          <Metric icon={<Thermometer />} label="Temperatura" value={temperatureEntities.length} detail="sensores continuos" />
          <Metric icon={<Droplets />} label="Humedad" value={humidityEntities.length} detail="sensores continuos" />
          <Metric icon={<Power />} label="Binarios" value={binaryEntities.length} detail="sensores de estado" />
          <Metric icon={<Zap />} label="Realtime" value={formatCount(realtime?.events)} detail={`${realtime?.tick_index || 0} pulsos`} />
          <Metric icon={<BarChart3 />} label="Sospechas" value={suspiciousCount} detail={`rojo score >= ${RED_SUSPICION_SCORE.toFixed(2)}`} />
        </section>

        <section className="workspace-grid simulator-grid">
          <Panel title="Controles manuales">
            <div className="control-grid simulator-controls">
              <label>
                Segundos por pulso simulado
                <input type="number" min="10" step="10" value={manualTickSeconds} onChange={(event) => setManualTickSeconds(Number(event.target.value))} />
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
            <div className="button-row">
              <StepButton id="static-history" busy={busy} onClick={loadStaticHistory} icon={<Database size={17} />} label="Crear/cargar historico" />
              <button className="secondary-action inline" onClick={resetRealtime} disabled={Boolean(busy)}>
                <RotateCcw size={17} />
                Reset realtime
              </button>
            </div>
            <ManualControlGroup
              title="Temperatura"
              icon={<Thermometer size={17} />}
              entities={temperatureEntities}
              states={manualStates}
              step={0.5}
              unit="C"
              onChange={updateManualState}
            />
            <ManualControlGroup
              title="Humedad"
              icon={<Droplets size={17} />}
              entities={humidityEntities}
              states={manualStates}
              step={1}
              unit="%"
              onChange={updateManualState}
            />
            <ManualBinaryGroup entities={binaryEntities} states={manualStates} onChange={updateManualState} />
          </Panel>

          <Panel title="Resultados realtime">
            <div className="summary-strip">
              <SummaryRow label="Insertadas ultimo pulso" value={runSummary?.manual_inserted ?? "-"} />
              <SummaryRow label="Omitidas por cadencia" value={runSummary?.manual_skipped_by_cadence ?? "-"} />
              <SummaryRow label="Tiempo total filtro" value={formatElapsed(runSummary?.elapsed_ms)} />
              <SummaryRow label="Sospechas visibles" value={formatCount(displayResults.length)} />
              <button className="secondary-action inline" onClick={() => downloadResultsCsv(displayResults)} disabled={!displayResults.length}>
                <Download size={17} />
                Descargar tabla en CSV
              </button>
            </div>
            <ResultsTable results={displayResults} onResultClick={setSelectedResult} />
          </Panel>
        </section>

        <section className="charts-panel">
          <div className="section-heading">
            <div>
              <h2>Charts en tiempo real</h2>
              <span>{timelines.length} entidades graficadas</span>
            </div>
            <button type="button" className="secondary-action inline" onClick={() => loadTimeline()} disabled={Boolean(busy)}>
              Actualizar charts
            </button>
          </div>
          <div className="chart-grid">
            {timelines.map((timeline) => (
              <TimelineChart
                key={timeline.entity_id}
                timeline={timeline}
                results={allAnalyzedResults}
                faults={realtime?.faults || []}
                onResultClick={setSelectedResult}
              />
            ))}
            {!timelines.length && <Empty text="Carga un historico y envia un pulso para ver los charts." />}
          </div>
        </section>

        {selectedResult && (
          <CalculationModal
            result={selectedResult}
            onClose={() => setSelectedResult(null)}
          />
        )}
      </main>
    );
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
          <a className="secondary-action" href="/simulador" target="_blank" rel="noreferrer">Abrir simulador</a>
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
            <Metric icon={<Database />} label="Base normalidad" value={formatCount(historyStatus?.normality_history?.rows)} detail={normalityDetail(historyStatus?.normality_history)} />
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
                <label className="file-action">
                  Subir historico real
                  <input type="file" accept=".csv,text/csv" onChange={uploadRealHistory} disabled={Boolean(busy)} />
                </label>
                <StepButton id="static-history" busy={busy} onClick={loadStaticHistory} icon={<Database size={17} />} label="Crear/cargar historico canonico" />
                <StepButton id="analyze-history" busy={busy} onClick={analyzeHistory} icon={<BarChart3 size={17} />} label="Analizar historico" disabled={!historyStatus?.exists} />
                <StepButton id="normalization-preview" busy={busy} onClick={openNormalizationPreview} icon={<Eye size={17} />} label="Ver normalizacion" disabled={!historyStatus?.exists} />
                <StepButton id="continuous-config" busy={busy} onClick={loadContinuousConfig} icon={<Eye size={17} />} label="Parametros continuos" disabled={!historyStatus?.exists} />
              </div>
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
                  onResultClick={setSelectedResult}
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
      ) : (
        <ClassifierWorkspace
          results={classifierResults}
          summary={classifierSummary}
          evaluation={classifierEvaluation}
          specification={classifierSpecification}
          mapData={contextMap}
          busy={busy}
          error={classifierError}
          onReload={loadClassifierData}
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

function ContextualWorkspace({ mapData, results, summary, busy, error, onReload, onRebuild }) {
  const [matrixMode, setMatrixMode] = useState("covariance");
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [selectedContextResult, setSelectedContextResult] = useState(null);
  const nodes = contextMapNodes(mapData);
  const edges = contextMapEdges(mapData);
  const mapSummary = mapData?.summary || {};
  const hasMap = contextMapHasData(mapData);
  const stale = contextMapIsStale(mapData);
  const staleReasons = contextMapStaleReasons(mapData);
  const acceptedEdges = contextAcceptedEdgeCount(mapData, edges);
  const coveredEntities = contextNumber(
    mapSummary.covered_entities,
    nodes.filter((node) => contextBoolean(node.covered)).length,
  );
  const coverage = contextNumber(
    mapSummary.coverage_ratio,
    mapSummary.coverage,
    summary?.coverage_ratio,
    summary?.coverage,
    nodes.length ? coveredEntities / nodes.length : null,
  );
  const outcomeCounts = contextOutcomeCounts(results);
  const isLoading = busy === "context-load";
  const isRebuilding = busy === "context-rebuild";

  return (
    <div className="contextual-workspace">
      <section className={`context-map-banner ${!hasMap ? "missing" : stale ? "stale" : "current"}`}>
        <div className="context-map-banner-icon">
          {stale || !hasMap ? <AlertTriangle size={21} /> : <Activity size={21} />}
        </div>
        <div className="context-map-banner-copy">
          <span className="eyebrow">Segundo filtro · validacion espacio-temporal</span>
          <h2>{!hasMap ? "Mapa contextual no disponible" : stale ? "Mapa contextual obsoleto" : "Mapa contextual vigente"}</h2>
          <p>
            {!hasMap
              ? "Construye el mapa desde el historico normalizado antes de evaluar las sospechas estadisticas."
              : stale
                ? `Las decisiones deben interpretarse con cautela${staleReasons.length ? `: ${staleReasons.join(", ")}` : "."}`
                : `Version ${mapData?.map_version || "-"} construida ${contextBuiltAtText(mapData)}. Las asociaciones son evidencia contextual, no causalidad.`}
          </p>
          {error && <small className="context-inline-error">{error}</small>}
        </div>
        <div className="context-map-actions">
          <button type="button" className="secondary-action inline" onClick={onReload} disabled={Boolean(busy)}>
            {isLoading ? <Loader2 className="spin" size={17} /> : <Eye size={17} />}
            Actualizar vista
          </button>
          <button type="button" className="primary-action" onClick={onRebuild} disabled={Boolean(busy)}>
            {isRebuilding ? <Loader2 className="spin" size={17} /> : <RotateCcw size={17} />}
            Reconstruir mapa
          </button>
        </div>
      </section>

      <section className="metric-grid context-metric-grid">
        <Metric
          icon={<Database />}
          label="Version del mapa"
          value={mapData?.map_version || "-"}
          detail={contextBuiltAtText(mapData)}
        />
        <Metric
          icon={<Activity />}
          label="Entidades cubiertas"
          value={formatCount(coveredEntities)}
          detail={coverage === null
            ? "cobertura no informada"
            : `${formatContextPercent(coverage)} de ${formatCount(contextNumber(mapSummary.nodes, nodes.length))}`}
        />
        <Metric
          icon={<Zap />}
          label="Aristas aceptadas"
          value={formatCount(acceptedEdges)}
          detail={`${formatCount(edges.length)} relaciones almacenadas`}
        />
        <Metric
          icon={<BarChart3 />}
          label="Evaluaciones"
          value={formatCount(contextNumber(summary?.result_count, summary?.results, results.length))}
          detail={`${outcomeCounts.respaldado} respaldadas / ${outcomeCounts.contradicho} contradichas`}
        />
        <Metric
          icon={<AlertTriangle />}
          label="Cobertura insuficiente"
          value={formatCount(outcomeCounts.cobertura_insuficiente)}
          detail="no equivale a contradiccion"
        />
        <Metric
          icon={<Eye />}
          label="Vigencia"
          value={!hasMap ? "Sin mapa" : stale ? "Obsoleto" : "Vigente"}
          detail={staleReasons.join(", ") || contextRefreshTrigger(mapData)}
        />
      </section>

      <section className="context-main-grid">
        <Panel title="Mapa de covarianza y dependencias">
          <div className="context-panel-intro">
            <p>
              La covarianza cruda se muestra para pares continuos. Cambia a asociacion normalizada para comparar
              relaciones continuas, de estado y mixtas en una escala comun.
            </p>
            <div className="context-matrix-switch" role="group" aria-label="Metrica visible en el mapa">
              <button
                type="button"
                className={matrixMode === "covariance" ? "active" : ""}
                aria-pressed={matrixMode === "covariance"}
                onClick={() => setMatrixMode("covariance")}
              >
                Covarianza
              </button>
              <button
                type="button"
                className={matrixMode === "association" ? "active" : ""}
                aria-pressed={matrixMode === "association"}
                onClick={() => setMatrixMode("association")}
              >
                Asociacion
              </button>
            </div>
          </div>
          <ContextMatrix mapData={mapData} mode={matrixMode} onSelectEdge={setSelectedEdge} />
        </Panel>

        <Panel title="Parametros reproducibles">
          <div className="context-map-metadata">
            <KeyValueGrid
              rows={[
                { label: "Version", value: mapData?.map_version || "-" },
                { label: "Construido", value: contextBuiltAtText(mapData) },
                { label: "Rango fuente", value: contextSourceRangeText(mapData?.source_range) },
                { label: "Estado", value: !hasMap ? "sin mapa" : stale ? "obsoleto" : "vigente" },
                { label: "Disparador", value: contextRefreshTrigger(mapData) },
                { label: "Motivos de obsolescencia", value: staleReasons.join(", ") || "ninguno" },
              ]}
            />
          </div>
          <div className="context-config-block">
            <h3>Regla de construccion</h3>
            {contextConfigRows(mapData?.config).length ? (
              <KeyValueGrid rows={contextConfigRows(mapData?.config)} />
            ) : (
              <Empty text="El backend aun no informa la configuracion del mapa." />
            )}
          </div>
          <p className="context-method-note">
            Una arista debe exponer observaciones, cobertura, fuerza y estabilidad antes de ingresar al top-k.
            El frontend presenta estos valores; no completa parametros ausentes ni presume causalidad.
          </p>
        </Panel>
      </section>

      <Panel title="Resultados del filtro contextual">
        <div className="context-results-summary" aria-label="Resumen de resultados contextuales">
          {["respaldado", "contradicho", "no_explicado", "cobertura_insuficiente"].map((outcome) => (
            <div className={`context-outcome-card ${outcome}`} key={outcome}>
              <span>{contextOutcomeLabel(outcome)}</span>
              <strong>{formatCount(outcomeCounts[outcome])}</strong>
            </div>
          ))}
          <div className="context-outcome-card common-cause">
            <span>Causa comun posible</span>
            <strong>{formatCount(results.filter((result) => contextCommonCause(result).possible).length)}</strong>
          </div>
        </div>
        <ContextResultsTable
          results={results}
          mapStale={stale}
          onResultClick={setSelectedContextResult}
        />
      </Panel>

      {selectedEdge && (
        <ContextEdgeModal
          edge={selectedEdge}
          mapVersion={mapData?.map_version}
          onClose={() => setSelectedEdge(null)}
        />
      )}
      {selectedContextResult && (
        <ContextResultModal
          result={selectedContextResult}
          fallbackMapStale={stale}
          onClose={() => setSelectedContextResult(null)}
        />
      )}
    </div>
  );
}

function ContextMatrix({ mapData, mode, onSelectEdge }) {
  const matrices = mapData?.matrices || {};
  const nodes = contextMapNodes(mapData);
  const edges = contextMapEdges(mapData);
  const entities = Array.isArray(matrices.entities) && matrices.entities.length
    ? matrices.entities.map((entity) => typeof entity === "string" ? entity : entity?.entity_id).filter(Boolean)
    : nodes.map((node) => node.entity_id || node.id).filter(Boolean);
  const values = mode === "association" ? matrices.association : matrices.covariance;
  const accepted = matrices.accepted;
  const numericValues = Array.isArray(values)
    ? values.flatMap((row, rowIndex) =>
      Array.isArray(row)
        ? row.map((value, columnIndex) => rowIndex === columnIndex ? null : contextFiniteNumber(value)).filter((value) => value !== null)
        : [],
    )
    : [];
  const scale = mode === "association"
    ? 1
    : Math.max(...numericValues.map((value) => Math.abs(value)), 0.0001);
  const edgeIndex = new Map();
  edges.forEach((edge) => {
    const source = contextEdgeSource(edge);
    const target = contextEdgeTarget(edge);
    if (source && target) {
      edgeIndex.set(`${source}\u0000${target}`, edge);
    }
  });

  if (!entities.length || !Array.isArray(values) || !values.length) {
    return (
      <div className="context-matrix-empty">
        <Empty text="No hay una matriz calculada. Reconstruye el mapa con un historico valido." />
      </div>
    );
  }

  return (
    <>
      <div className="context-matrix-legend" aria-label="Leyenda del mapa">
        <span><i className="negative" />Relacion inversa</span>
        <span><i className="neutral" />Relacion cercana a cero</span>
        <span><i className="positive" />Relacion directa</span>
        <span><b className="accepted" />Arista aceptada</span>
        <span><b className="rejected" />Arista no aceptada</span>
      </div>
      <div className="context-matrix-scroll">
        <table className="context-matrix">
          <thead>
            <tr>
              <th scope="col">Entidad</th>
              {entities.map((entityId) => (
                <th scope="col" title={entityId} key={entityId}>
                  <span>{compactEntityId(entityId)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entities.map((rowEntity, rowIndex) => (
              <tr key={rowEntity}>
                <th scope="row" title={rowEntity}>{compactEntityId(rowEntity)}</th>
                {entities.map((columnEntity, columnIndex) => {
                  const value = contextMatrixCell(values, rowIndex, columnIndex);
                  const numeric = contextFiniteNumber(value);
                  const covariance = contextMatrixCell(matrices.covariance, rowIndex, columnIndex);
                  const association = contextMatrixCell(matrices.association, rowIndex, columnIndex);
                  const matrixAccepted = contextMatrixCell(accepted, rowIndex, columnIndex);
                  const edge = edgeIndex.get(`${rowEntity}\u0000${columnEntity}`)
                    || edgeIndex.get(`${columnEntity}\u0000${rowEntity}`)
                    || null;
                  const isAccepted = matrixAccepted === null || matrixAccepted === undefined
                    ? contextEdgeAccepted(edge)
                    : contextBoolean(matrixAccepted);
                  const isDiagonal = rowIndex === columnIndex;
                  const available = numeric !== null;
                  const title = available
                    ? `${rowEntity} ↔ ${columnEntity}: ${formatContextMatrixValue(numeric, mode)} · ${isAccepted ? "aceptada" : "no aceptada"}`
                    : `${rowEntity} ↔ ${columnEntity}: no aplicable`;
                  const detail = edge || {
                    source_entity_id: rowEntity,
                    target_entity_id: columnEntity,
                    covariance,
                    association,
                    accepted: isAccepted,
                    relation_family: "matrix",
                  };
                  return (
                    <td key={columnEntity}>
                      <button
                        type="button"
                        className={[
                          "context-matrix-cell",
                          isDiagonal ? "diagonal" : "",
                          !available ? "unavailable" : "",
                          isAccepted ? "accepted" : "rejected",
                        ].filter(Boolean).join(" ")}
                        style={available ? { backgroundColor: contextHeatColor(numeric, scale) } : undefined}
                        disabled={isDiagonal || !available}
                        title={title}
                        aria-label={title}
                        onClick={() => onSelectEdge(detail)}
                      >
                        {available ? formatContextMatrixValue(numeric, mode) : "N/D"}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="context-matrix-scale">
        <span>{formatContextMatrixValue(-scale, mode)}</span>
        <div aria-hidden="true" />
        <span>0</span>
        <div aria-hidden="true" />
        <span>{formatContextMatrixValue(scale, mode)}</span>
      </div>
      <p className="context-matrix-note">
        {mode === "covariance"
          ? "N/D identifica pares cuya covarianza cruda no es aplicable, por ejemplo relaciones mixtas o entre estados. Haz clic en una celda para auditar la arista."
          : "La asociacion normalizada conserva el tipo de metrica calculada por el backend. Haz clic en una celda para ver muestra, desfase y estabilidad."}
      </p>
    </>
  );
}

function ContextResultsTable({ results, mapStale, onResultClick }) {
  return (
    <div className="context-results-table" role="table" aria-label="Resultados contextuales">
      <div className="context-results-head" role="row">
        <span role="columnheader">Entidad</span>
        <span role="columnheader">Tiempo</span>
        <span role="columnheader">Score estadistico</span>
        <span role="columnheader">Coherencia</span>
        <span role="columnheader">Resultado</span>
        <span role="columnheader">Cobertura</span>
        <span role="columnheader">Causa comun</span>
        <span role="columnheader">Mapa</span>
      </div>
      {(results || []).map((result, index) => {
        const outcome = contextOutcome(result);
        const coverage = contextCoverage(result);
        const commonCause = contextCommonCause(result);
        const stale = contextResultMapIsStale(result, mapStale);
        return (
          <button
            type="button"
            role="row"
            className="context-result-row"
            key={`${result.event_id || result.entity_id}-${result.timestamp}-${index}`}
            onClick={() => onResultClick(result)}
            title="Ver evidencia contextual"
          >
            <span role="cell">
              <strong>{result.entity_id || result.target_entity_id || "-"}</strong>
              <small>{kindText(result.sensor_kind)}</small>
            </span>
            <TimeStack value={result.timestamp} />
            <span role="cell">{formatNumber(contextStatScore(result))}</span>
            <span role="cell">{formatNumber(contextCoherenceScore(result))}</span>
            <span role="cell"><ContextStatusBadge value={outcome} /></span>
            <span role="cell">
              {coverage.ratio === null ? "-" : formatContextPercent(coverage.ratio)}
              <small>{coverage.detail}</small>
            </span>
            <span role="cell"><ContextStatusBadge value={commonCause.key} label={commonCause.label} /></span>
            <span role="cell"><ContextStatusBadge value={stale ? "stale" : "current"} label={stale ? "Obsoleto" : "Vigente"} /></span>
          </button>
        );
      })}
      {!results?.length && <Empty text="No hay sospechas evaluadas por el filtro contextual." />}
    </div>
  );
}

function ContextStatusBadge({ value, label }) {
  const key = contextClassKey(value);
  return <span className={`context-status-badge ${key}`}>{label || contextOutcomeLabel(value)}</span>;
}

function ContextEdgeModal({ edge, mapVersion, onClose }) {
  const accepted = contextEdgeAccepted(edge);
  const coverage = contextNumber(edge.coverage_ratio, edge.coverage);
  const stability = contextNumber(edge.stability_score, edge.temporal_stability, edge.stability);
  const rejectionReasons = contextArray(
    edge.rejection_reasons,
    edge.reasons,
    edge.rejected_by,
  );
  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="calculation-modal context-detail-modal" role="dialog" aria-modal="true" aria-label="Detalle de la arista contextual" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <div>
            <span>Arista del mapa · version {mapVersion || "-"}</span>
            <h2>{contextEdgeSource(edge)} ↔ {contextEdgeTarget(edge)}</h2>
            <small>Asociacion historica utilizada como evidencia contextual, no como atribucion causal.</small>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Cerrar detalle">
            <X size={18} />
          </button>
        </header>
        <div className="modal-score-grid">
          <SummaryRow label="Aceptada" value={accepted ? "si" : "no"} />
          <SummaryRow label="Covarianza" value={formatContextValue(edge.covariance ?? edge.raw_covariance)} />
          <SummaryRow label="Asociacion" value={formatContextValue(edge.association ?? edge.association_signed ?? edge.weight)} />
          <SummaryRow label="Estabilidad" value={stability === null ? "-" : formatContextPercent(stability)} />
        </div>
        <div className="modal-columns">
          <article className="modal-block">
            <h3>Relacion historica</h3>
            <KeyValueGrid rows={[
              { label: "Origen", value: contextEdgeSource(edge) },
              { label: "Destino", value: contextEdgeTarget(edge) },
              { label: "Familia", value: edge.relation_family || edge.relation_type || "-" },
              { label: "Metrica", value: edge.metric || edge.method || "-" },
              { label: "Signo esperado", value: edge.expected_sign || edge.sign || edge.expected_response || "-" },
              { label: "Desfase", value: formatSeconds(edge.lag_seconds ?? edge.lag) },
            ]} />
          </article>
          <article className="modal-block">
            <h3>Regla de aceptacion</h3>
            <KeyValueGrid rows={[
              { label: "Observaciones", value: formatCount(contextNumber(edge.n_observations, edge.observations, edge.sample_size)) },
              { label: "Cobertura", value: coverage === null ? "-" : formatContextPercent(coverage) },
              { label: "Estabilidad temporal", value: stability === null ? "-" : formatContextPercent(stability) },
              { label: "Valor p", value: formatContextValue(edge.p_value) },
              { label: "Estado", value: accepted ? "aceptada" : "no aceptada" },
              { label: "Motivo", value: rejectionReasons.join(", ") || (accepted ? "cumple la regla configurada" : "no informado") },
            ]} />
          </article>
        </div>
        <article className="modal-block">
          <h3>Estabilidad por periodos</h3>
          {contextArray(edge.stability_windows, edge.period_stability, edge.window_scores, edge.slice_associations).length ? (
            <div className="context-period-list">
              {contextArray(edge.stability_windows, edge.period_stability, edge.window_scores, edge.slice_associations).map((period, index) => (
                <div key={`${formatContextValue(period?.start)}-${index}`}>
                  <span>{typeof period === "object" ? `${shortDate(period.start || period.window_start)} → ${shortDate(period.end || period.window_end)}` : `Periodo ${index + 1}`}</span>
                  <strong>{typeof period === "object" ? formatContextValue(period.association ?? period.score ?? period.stability) : formatContextValue(period)}</strong>
                </div>
              ))}
            </div>
          ) : (
            <Empty text="No se informaron subventanas de estabilidad para esta arista." />
          )}
        </article>
      </section>
    </div>
  );
}

function ContextResultModal({ result, fallbackMapStale, onClose }) {
  const outcome = contextOutcome(result);
  const coverage = contextCoverage(result);
  const commonCause = contextCommonCause(result);
  const evidence = contextResultEvidence(result);
  const mapQuality = result.map_quality || result.map || {};
  const stale = contextResultMapIsStale(result, fallbackMapStale);
  const supporting = contextArray(result.supports, result.supporting_entities, result.supported_by);
  const contradictions = contextArray(result.contradictions, result.contradicted_by);
  const unknowns = contextArray(result.unknowns, result.missing_context, result.unavailable_entities);

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="calculation-modal context-detail-modal" role="dialog" aria-modal="true" aria-label="Detalle del filtro contextual" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <div>
            <span>Validacion contextual espacio-temporal</span>
            <h2>{result.entity_id || result.target_entity_id || "-"}</h2>
            <small>{shortDate(result.timestamp)} · mapa {mapQuality.version || result.map_version || "-"}</small>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Cerrar detalle">
            <X size={18} />
          </button>
        </header>

        <div className="modal-score-grid">
          <SummaryRow label="Score estadistico" value={formatNumber(contextStatScore(result))} />
          <SummaryRow label="Coherencia contextual" value={formatNumber(contextCoherenceScore(result))} />
          <SummaryRow label="Resultado" value={contextOutcomeLabel(outcome)} />
          <SummaryRow label="Efecto en sospecha" value={result.suspicion_effect || result.effect_on_suspicion || "-"} />
          <SummaryRow label="Cobertura" value={coverage.ratio === null ? "-" : formatContextPercent(coverage.ratio)} />
          <SummaryRow label="Vigencia del mapa" value={stale ? "obsoleto" : "vigente"} />
        </div>

        <div className="modal-columns">
          <article className="modal-block">
            <h3>Cobertura antes de decidir</h3>
            <KeyValueGrid rows={[
              { label: "Relaciones elegibles", value: formatCount(coverage.eligible) },
              { label: "Relaciones observadas", value: formatCount(coverage.observed) },
              { label: "Proporcion", value: coverage.ratio === null ? "-" : formatContextPercent(coverage.ratio) },
              { label: "Estado", value: coverage.status || (outcome === "cobertura_insuficiente" ? "insuficiente" : "suficiente") },
              { label: "Sin evidencia", value: formatCount(unknowns.length) },
            ]} />
            <p className="context-method-note">
              Falta de cobertura no se interpreta como contradiccion. Una contradiccion exige evidencia observable
              opuesta al signo, estado o secuencia historicamente esperada.
            </p>
          </article>
          <article className="modal-block">
            <h3>Causa comun</h3>
            <ContextStatusBadge value={commonCause.key} label={commonCause.label} />
            <KeyValueGrid rows={[
              { label: "Identificable", value: commonCause.identifiable ? "si" : "no" },
              { label: "Puntaje", value: formatContextValue(commonCause.score) },
              { label: "Evidencia independiente", value: commonCause.independentEvidence ? "disponible" : "no disponible" },
              { label: "Dominio", value: commonCause.domain || "-" },
              { label: "Detalle", value: commonCause.detail || "El historico correlacional no permite atribuir una causa comun por si solo." },
            ]} />
          </article>
        </div>

        <article className="modal-block">
          <h3>Aportes de entidades relacionadas</h3>
          {evidence.length ? (
            <div className="context-evidence-table">
              <div className="context-evidence-head">
                <span>Entidad</span>
                <span>Relacion</span>
                <span>Peso</span>
                <span>Desfase</span>
                <span>c_j normalizado</span>
                <span>Aporte w·c</span>
                <span>Estado</span>
              </div>
              {evidence.map((item, index) => {
                const status = contextEvidenceStatus(item);
                return (
                  <div className="context-evidence-row" key={`${contextEvidenceEntity(item)}-${index}`}>
                    <span>{contextEvidenceEntity(item)}</span>
                    <span>{item.relation_type || item.relation_family || item.metric || item.expected_sign || "-"}</span>
                    <span>{formatContextValue(item.weight ?? item.association)}</span>
                    <span>{formatSeconds(item.lag_seconds ?? item.lag)}</span>
                    <span>{formatContextValue(item.c_j_normalized ?? item.context_evidence_normalized ?? item.c_j)}</span>
                    <span>{formatContextValue(item.contribution ?? item.weighted_evidence ?? item.weighted_contribution)}</span>
                    <span><ContextStatusBadge value={status} label={contextEvidenceStatusLabel(status)} /></span>
                  </div>
                );
              })}
            </div>
          ) : (
            <Empty text="No se informaron aportes por entidad para esta evaluacion." />
          )}
        </article>

        <div className="modal-columns">
          <article className="modal-block">
            <h3>Resumen de evidencia</h3>
            <KeyValueGrid rows={[
              { label: "Respaldos", value: formatCount(supporting.length) },
              { label: "Contradicciones efectivas", value: formatCount(contradictions.length) },
              { label: "Sin cobertura", value: formatCount(unknowns.length) },
              { label: "Explicacion", value: result.explanation || result.reason || "-" },
            ]} />
          </article>
          <article className="modal-block">
            <h3>Calidad del mapa usado</h3>
            <KeyValueGrid rows={[
              { label: "Version", value: mapQuality.version || result.map_version || "-" },
              { label: "Vigencia", value: stale ? "obsoleto" : "vigente" },
              { label: "Edad", value: formatSeconds(mapQuality.age_seconds ?? result.map_age_seconds) },
              { label: "Estabilidad", value: formatContextValue(mapQuality.stability ?? result.map_stability) },
              { label: "Motivos", value: contextArray(mapQuality.stale_reasons, result.stale_reasons).join(", ") || "ninguno" },
            ]} />
          </article>
        </div>
      </section>
    </div>
  );
}

function contextMapNodes(mapData) {
  return Array.isArray(mapData?.nodes) ? mapData.nodes : [];
}

function contextMapEdges(mapData) {
  return Array.isArray(mapData?.edges) ? mapData.edges : [];
}

function contextMapHasData(mapData) {
  return Boolean(
    mapData
    && (
      mapData.map_version
      || mapData.built_at
      || contextMapNodes(mapData).length
      || mapData?.matrices?.entities?.length
    )
  );
}

function contextMapIsStale(mapData) {
  return contextBoolean(
    mapData?.refresh?.stale
    ?? mapData?.refresh?.is_stale
    ?? mapData?.summary?.stale
    ?? mapData?.stale,
  );
}

function contextMapStaleReasons(mapData) {
  return contextArray(
    mapData?.refresh?.stale_reasons,
    mapData?.refresh?.reasons,
    mapData?.stale_reasons,
  ).map((reason) => typeof reason === "object" ? formatContextValue(reason.reason || reason.code || reason) : String(reason));
}

function contextRefreshTrigger(mapData) {
  return formatContextValue(
    mapData?.refresh?.trigger
    ?? mapData?.refresh?.last_trigger
    ?? mapData?.refresh?.status
    ?? "sin disparador informado",
  );
}

function contextBuiltAtText(mapData) {
  if (!mapData?.built_at) {
    return "sin fecha de construccion";
  }
  return shortDate(mapData.built_at);
}

function contextSourceRangeText(sourceRange) {
  if (Array.isArray(sourceRange)) {
    return `${shortDate(sourceRange[0])} → ${shortDate(sourceRange[1])}`;
  }
  if (sourceRange && typeof sourceRange === "object") {
    return `${shortDate(sourceRange.start || sourceRange.from)} → ${shortDate(sourceRange.end || sourceRange.to)}`;
  }
  return sourceRange || "-";
}

function contextConfigRows(config = {}) {
  if (!config || typeof config !== "object") {
    return [];
  }
  const used = new Set();
  const definitions = [
    [["k", "top_k"], "Relaciones maximas (k)"],
    [["window_seconds", "window_size_seconds"], "Ventana temporal"],
    [["lag_candidates_seconds", "lag_candidates"], "Desfases evaluados"],
    [["max_lag_seconds"], "Desfase maximo"],
    [["association_threshold", "min_association", "association_min"], "Asociacion minima"],
    [["min_observations", "minimum_observations"], "Observaciones minimas"],
    [["min_sign_consistency", "stability_threshold", "min_stability"], "Consistencia de signo minima"],
    [["stability_slices", "stability_windows", "temporal_splits"], "Subventanas de estabilidad"],
    [["min_stable_slices"], "Subventanas estables minimas"],
    [["min_pair_coverage", "coverage_threshold", "min_coverage"], "Cobertura historica minima"],
    [["runtime_coverage_threshold"], "Cobertura minima en ejecucion"],
    [["max_map_age_days"], "Vigencia maxima (dias)"],
    [["max_age_seconds", "stale_after_seconds"], "Vigencia maxima"],
    [["acceptance_rule"], "Regla de aceptacion"],
  ];
  const rows = [];
  definitions.forEach(([keys, label]) => {
    const key = keys.find((candidate) => config[candidate] !== undefined && config[candidate] !== null);
    if (!key) {
      return;
    }
    used.add(key);
    const isSeconds = key.endsWith("_seconds") && !key.includes("candidates");
    rows.push({
      label,
      value: isSeconds ? formatBucketSeconds(Number(config[key])) : formatContextValue(config[key]),
    });
  });
  Object.entries(config).forEach(([key, value]) => {
    if (!used.has(key)) {
      rows.push({ label: key, value: formatContextValue(value) });
    }
  });
  return rows;
}

function contextAcceptedEdgeCount(mapData, edges) {
  const summaryCount = contextNumber(
    mapData?.summary?.accepted_edges,
    mapData?.summary?.accepted_edge_count,
  );
  if (summaryCount !== null) {
    return summaryCount;
  }
  if (edges.length) {
    return edges.filter(contextEdgeAccepted).length;
  }
  const accepted = mapData?.matrices?.accepted;
  if (!Array.isArray(accepted)) {
    return 0;
  }
  let count = 0;
  accepted.forEach((row, rowIndex) => {
    if (!Array.isArray(row)) {
      return;
    }
    row.forEach((value, columnIndex) => {
      if (columnIndex > rowIndex && contextBoolean(value)) {
        count += 1;
      }
    });
  });
  return count;
}

function contextMatrixCell(matrix, rowIndex, columnIndex) {
  if (!Array.isArray(matrix) || !Array.isArray(matrix[rowIndex])) {
    return null;
  }
  return matrix[rowIndex][columnIndex] ?? null;
}

function contextFiniteNumber(value) {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function contextNumber(...values) {
  for (const value of values) {
    const number = contextFiniteNumber(value);
    if (number !== null) {
      return number;
    }
  }
  return null;
}

function contextBoolean(value) {
  if (typeof value === "string") {
    return ["true", "1", "yes", "si", "accepted", "stale", "obsoleto"].includes(value.trim().toLowerCase());
  }
  return Boolean(value);
}

function contextArray(...values) {
  for (const value of values) {
    if (Array.isArray(value)) {
      return value;
    }
  }
  return [];
}

function compactEntityId(entityId) {
  const value = String(entityId || "-");
  const withoutDomain = value.includes(".") ? value.split(".").slice(1).join(".") : value;
  return withoutDomain.length > 22 ? `${withoutDomain.slice(0, 20)}…` : withoutDomain;
}

function contextHeatColor(value, scale) {
  const numeric = contextFiniteNumber(value);
  if (numeric === null) {
    return "rgba(148, 163, 184, 0.05)";
  }
  const intensity = Math.min(1, Math.abs(numeric) / Math.max(scale, 0.0001));
  const alpha = 0.10 + intensity * 0.72;
  if (numeric > 0) {
    return `rgba(45, 212, 191, ${alpha.toFixed(3)})`;
  }
  if (numeric < 0) {
    return `rgba(167, 139, 250, ${alpha.toFixed(3)})`;
  }
  return "rgba(148, 163, 184, 0.12)";
}

function formatContextMatrixValue(value, mode) {
  const number = contextFiniteNumber(value);
  if (number === null) {
    return "N/D";
  }
  if (mode === "association") {
    return number.toFixed(2);
  }
  if (Math.abs(number) >= 10000 || (Math.abs(number) > 0 && Math.abs(number) < 0.001)) {
    return number.toExponential(2);
  }
  return formatCalc(number);
}

function contextEdgeSource(edge = {}) {
  return edge.source_entity_id || edge.source || edge.entity_i || edge.entity_id || "-";
}

function contextEdgeTarget(edge = {}) {
  return edge.target_entity_id || edge.target || edge.entity_j || edge.related_entity_id || "-";
}

function contextEdgeAccepted(edge) {
  if (!edge) {
    return false;
  }
  return contextBoolean(edge.accepted ?? edge.is_accepted ?? edge.status);
}

function contextOutcome(result = {}) {
  const raw = result.context_outcome
    || result.contextual_status
    || result.context_result
    || result.contextual_result
    || result.outcome
    || result.status
    || "no_explicado";
  const key = contextClassKey(raw);
  if (["respaldado", "supported", "backed", "coherent"].includes(key)) {
    return "respaldado";
  }
  if (["contradicho", "contradiction", "contradicted", "incoherent"].includes(key)) {
    return "contradicho";
  }
  if (["cobertura_insuficiente", "insufficient_coverage", "low_coverage", "not_applicable"].includes(key)) {
    return "cobertura_insuficiente";
  }
  return "no_explicado";
}

function contextOutcomeLabel(value) {
  const key = contextClassKey(value);
  const labels = {
    respaldado: "Respaldado",
    contradicho: "Contradicho",
    no_explicado: "No explicado",
    cobertura_insuficiente: "Cobertura insuficiente",
    supported: "Respaldado",
    contradicted: "Contradicho",
    possible: "Posible",
    identifiable: "Identificable",
    unidentifiable: "No identificable",
    unknown: "No identificable",
    stale: "Obsoleto",
    current: "Vigente",
    supports: "Respalda",
    contradicts: "Contradice",
    missing: "Sin cobertura",
    neutral: "Neutral",
  };
  return labels[key] || String(value || "No explicado");
}

function contextClassKey(value) {
  return String(value || "unknown")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
}

function contextOutcomeCounts(results = []) {
  const counts = {
    respaldado: 0,
    contradicho: 0,
    no_explicado: 0,
    cobertura_insuficiente: 0,
  };
  results.forEach((result) => {
    counts[contextOutcome(result)] += 1;
  });
  return counts;
}

function contextCoverage(result = {}) {
  const coverage = result.coverage;
  const eligible = contextNumber(
    typeof coverage === "object" ? coverage.eligible : null,
    typeof coverage === "object" ? coverage.eligible_relations : null,
    typeof coverage === "object" ? coverage.eligible_neighbors : null,
    result.eligible_relations,
  );
  const observed = contextNumber(
    typeof coverage === "object" ? coverage.observed : null,
    typeof coverage === "object" ? coverage.observed_relations : null,
    typeof coverage === "object" ? coverage.observed_neighbors : null,
    result.observed_relations,
  );
  let ratio = contextNumber(
    typeof coverage === "number" ? coverage : null,
    typeof coverage === "object" ? coverage.ratio : null,
    typeof coverage === "object" ? coverage.coverage_ratio : null,
    result.coverage_ratio,
  );
  if (ratio === null && eligible && observed !== null) {
    ratio = observed / eligible;
  }
  const status = typeof coverage === "object"
    ? coverage.status || coverage.coverage_status
    : result.coverage_status;
  return {
    eligible,
    observed,
    ratio,
    status: status || "",
    detail: eligible === null || observed === null ? (status || "") : `${formatCount(observed)} / ${formatCount(eligible)} relaciones`,
  };
}

function contextCommonCause(result = {}) {
  const raw = result.common_cause;
  const object = raw && typeof raw === "object" ? raw : {};
  const explicitCandidate = object.candidate ?? object.possible ?? object.detected ?? result.common_cause_possible;
  const status = contextClassKey(
    object.status
    || object.result
    || (typeof raw === "string" ? raw : "")
    || (contextBoolean(explicitCandidate) ? "possible" : "")
    || (result.common_cause_identifiable === false ? "unidentifiable" : "unknown"),
  );
  const possible = ["possible", "probable", "detected", "common_cause_possible", "unidentifiable_without_independent_evidence"].includes(status)
    || contextBoolean(explicitCandidate);
  const independentEvidence = contextBoolean(
    object.independent_signal_present
    ?? object.independent_evidence
    ?? object.independent_evidence_available
    ?? result.independent_signal_present,
  );
  const identifiable = contextBoolean(object.identifiable ?? result.common_cause_identifiable)
    || (possible && independentEvidence);
  const explicitlyNotIndicated = explicitCandidate !== undefined
    && explicitCandidate !== null
    && !contextBoolean(explicitCandidate);
  return {
    key: possible && !identifiable
      ? "possible_unidentifiable"
      : possible
        ? "possible"
        : identifiable
          ? "identifiable"
          : explicitlyNotIndicated
            ? "not_indicated"
            : "unknown",
    label: possible && !identifiable
      ? "Posible · no identificable"
      : possible
        ? "Posible"
        : identifiable
          ? "Identificable"
          : explicitlyNotIndicated
            ? "No indicada"
            : "No evaluada",
    possible,
    identifiable,
    score: object.score ?? result.common_cause_score,
    independentEvidence,
    domain: object.domain
      || object.failure_domain
      || (object.independent_domains_required ? formatContextValue(object.independent_domains_required) : null)
      || result.common_failure_domain,
    detail: object.detail || object.explanation || object.interpretation || result.common_cause_explanation,
  };
}

function contextResultMapIsStale(result, fallback) {
  const value = result?.map_quality?.stale
    ?? result?.map?.stale
    ?? result?.map_stale;
  return value === undefined || value === null ? Boolean(fallback) : contextBoolean(value);
}

function contextStatScore(result = {}) {
  return contextNumber(result.stat_score, result.statistical_score, result.suspicion_score);
}

function contextCoherenceScore(result = {}) {
  return contextNumber(result.coherence_score, result.contextual_score, result.context_score, result.score);
}

function contextResultEvidence(result = {}) {
  return contextArray(
    result.evidence,
    result.evaluated_relations,
    result.related_evidence,
    result.relations,
  );
}

function contextEvidenceEntity(item = {}) {
  return item.neighbor_id
    || item.related_entity_id
    || item.entity_id
    || item.target_entity_id
    || item.source_entity_id
    || "-";
}

function contextEvidenceStatus(item = {}) {
  const key = contextClassKey(item.observation_status || item.status || item.result || "neutral");
  if (["support", "supports", "supported", "respaldado", "coherent"].includes(key)) {
    return "supports";
  }
  if (["contradiction", "contradicts", "contradicted", "contradicho", "incoherent"].includes(key)) {
    return "contradicts";
  }
  if (["missing", "unavailable", "insufficient_coverage", "sin_cobertura"].includes(key)) {
    return "missing";
  }
  return "neutral";
}

function contextEvidenceStatusLabel(status) {
  return contextOutcomeLabel(status);
}

function formatContextPercent(value) {
  const number = contextFiniteNumber(value);
  if (number === null) {
    return "-";
  }
  return number > 1 ? `${number.toFixed(1)}%` : formatPercent(number);
}

function formatContextValue(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  if (typeof value === "boolean") {
    return value ? "si" : "no";
  }
  if (typeof value === "number") {
    return formatCalc(value);
  }
  if (Array.isArray(value)) {
    return value.map(formatContextValue).join(", ");
  }
  if (typeof value === "object") {
    return Object.entries(value).map(([key, item]) => `${key}: ${formatContextValue(item)}`).join(" · ");
  }
  return String(value);
}

function ClassifierWorkspace({ results, summary, evaluation, specification, mapData, busy, error, onReload }) {
  const [activeView, setActiveView] = useState("diagnosticos");
  const modelVersion = classifierModelVersion(specification, summary);
  const modelType = specification?.model_type || specification?.type || summary?.model_type || "-";
  const specificationStatus = classifierSpecificationStatus(specification);
  const mapAvailable = contextMapHasData(mapData) || Boolean(summary?.map_version);
  const mapStale = summary?.map_stale !== undefined && summary?.map_stale !== null
    ? contextBoolean(summary.map_stale)
    : contextMapHasData(mapData)
      ? contextMapIsStale(mapData)
      : null;
  const mapVersion = mapData?.map_version || summary?.map_version || "-";
  const isLoading = busy === "classifier-load";
  const mapStatus = !mapAvailable ? "no disponible" : mapStale ? "obsoleto" : "vigente";

  return (
    <div className="classifier-workspace">
      <section className={`classifier-readiness-banner ${specificationStatus.key} ${mapStale ? "map-stale" : ""} ${!mapAvailable ? "map-missing" : ""}`}>
        <div className="classifier-readiness-icon">
          {specificationStatus.key === "ready" && mapStale === false
            ? <FlaskConical size={22} />
            : <AlertTriangle size={22} />}
        </div>
        <div className="classifier-readiness-copy">
          <span className="eyebrow">Tercer filtro · decision diagnostica</span>
          <h2>{specificationStatus.label}</h2>
          <p>
            Modelo {modelVersion} ({modelType}). Mapa contextual {mapVersion}: {mapStatus}.
            La confianza expresa el criterio informado por el modelo; no se presenta como probabilidad si no esta calibrada.
          </p>
          {error && <small className="classifier-inline-error">{error}</small>}
        </div>
        <button type="button" className="secondary-action inline" onClick={onReload} disabled={Boolean(busy)}>
          {isLoading ? <Loader2 className="spin" size={17} /> : <Eye size={17} />}
          Actualizar clasificador
        </button>
      </section>

      <nav className="classifier-subtabs" aria-label="Vistas del clasificador">
        {[
          ["diagnosticos", "Diagnosticos"],
          ["evaluacion", "Evaluacion E2E"],
          ["modelo", "Modelo y umbrales"],
        ].map(([id, label]) => (
          <button
            type="button"
            className={activeView === id ? "active" : ""}
            aria-pressed={activeView === id}
            onClick={() => setActiveView(id)}
            key={id}
          >
            {label}
          </button>
        ))}
      </nav>

      {activeView === "diagnosticos" && (
        <ClassifierDiagnosticsView
          results={results}
          summary={summary}
          fallbackMapStale={mapStale}
        />
      )}
      {activeView === "evaluacion" && (
        <ClassifierEvaluationView evaluation={evaluation} />
      )}
      {activeView === "modelo" && (
        <ClassifierSpecificationView specification={specification} />
      )}
    </div>
  );
}

function ClassifierDiagnosticsView({ results = [], summary = {}, fallbackMapStale = false }) {
  const [classFilter, setClassFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [entityQuery, setEntityQuery] = useState("");
  const [minimumConfidence, setMinimumConfidence] = useState(0);
  const [selectedDiagnosis, setSelectedDiagnosis] = useState(null);
  const counts = classifierClassCounts(results, summary?.by_class);
  const filteredResults = useMemo(() => {
    const query = entityQuery.trim().toLowerCase();
    return (results || []).filter((result) => {
      const diagnosticClass = classifierResultClass(result);
      const confidence = classifierConfidence(result);
      const source = String(result.source || result.origin || "unknown").toLowerCase();
      if (classFilter !== "all" && diagnosticClass !== classFilter) {
        return false;
      }
      if (sourceFilter !== "all" && source !== sourceFilter) {
        return false;
      }
      if (query && !String(result.entity_id || "").toLowerCase().includes(query)) {
        return false;
      }
      if (minimumConfidence > 0 && (confidence === null || normalizedConfidence(confidence) < minimumConfidence)) {
        return false;
      }
      return true;
    });
  }, [results, classFilter, sourceFilter, entityQuery, minimumConfidence]);
  const evaluated = contextNumber(summary?.evaluated, summary?.classified, results.length);
  const lowConfidence = contextNumber(summary?.low_confidence, results.filter(classifierIsLowConfidence).length);
  const meanConfidence = contextNumber(summary?.confidence_mean, classifierMeanConfidence(results));

  return (
    <>
      <section className="metric-grid classifier-metric-grid">
        <Metric
          icon={<BarChart3 />}
          label="Evaluados"
          value={formatCount(evaluated)}
          detail="casos que llegaron al clasificador"
        />
        <Metric
          icon={<Activity />}
          label="Normal"
          value={formatCount(counts.normal)}
          detail="candidatos reclasificados"
        />
        <Metric
          icon={<Zap />}
          label="Anomalia ambiental"
          value={formatCount(counts.environmental_anomaly)}
          detail="evento coherente del entorno"
        />
        <Metric
          icon={<AlertTriangle />}
          label="Posible falla de sensor"
          value={formatCount(counts.possible_sensor_fault)}
          detail="requiere revision o alerta"
        />
        <Metric
          icon={<Eye />}
          label="Indeterminado"
          value={formatCount(counts.indeterminate)}
          detail="evidencia insuficiente o conflictiva"
        />
        <Metric
          icon={<FlaskConical />}
          label="Confianza"
          value={formatClassifierConfidence(meanConfidence)}
          detail={`${formatCount(lowConfidence)} bajo umbral`}
        />
      </section>

      <Panel title="Diagnosticos del tercer filtro">
        <div className="classifier-scope-note">
          <AlertTriangle size={17} />
          <span>
            Esta tabla contiene solo casos recibidos por el clasificador. Los casos descartados antes aparecen
            exclusivamente en Evaluacion E2E como <strong>not_reached</strong>, que no es una clase diagnostica.
          </span>
        </div>
        <div className="classifier-filters">
          <label>
            Clase
            <select value={classFilter} onChange={(event) => setClassFilter(event.target.value)}>
              <option value="all">Todas</option>
              <option value="normal">Normal</option>
              <option value="environmental_anomaly">Anomalia ambiental</option>
              <option value="possible_sensor_fault">Posible falla de sensor</option>
              <option value="indeterminate">Indeterminado</option>
            </select>
          </label>
          <label>
            Origen
            <select value={sourceFilter} onChange={(event) => setSourceFilter(event.target.value)}>
              <option value="all">Historico y realtime</option>
              <option value="historical">Historico</option>
              <option value="realtime">Realtime</option>
            </select>
          </label>
          <label>
            Buscar entidad
            <input
              type="search"
              value={entityQuery}
              placeholder="sensor..."
              onChange={(event) => setEntityQuery(event.target.value)}
            />
          </label>
          <label className="classifier-confidence-filter">
            Confianza minima: {Math.round(minimumConfidence * 100)}%
            <input
              type="range"
              min="0"
              max="100"
              value={Math.round(minimumConfidence * 100)}
              onChange={(event) => setMinimumConfidence(Number(event.target.value) / 100)}
            />
          </label>
          <button
            type="button"
            className="secondary-action inline"
            onClick={() => {
              setClassFilter("all");
              setSourceFilter("all");
              setEntityQuery("");
              setMinimumConfidence(0);
            }}
          >
            <RotateCcw size={16} />
            Limpiar
          </button>
          <div className="classifier-filter-count">
            <span>Visibles</span>
            <strong>{formatCount(filteredResults.length)} / {formatCount(results.length)}</strong>
          </div>
        </div>
        <ClassifierResultsTable
          results={filteredResults}
          fallbackMapStale={fallbackMapStale}
          onResultClick={setSelectedDiagnosis}
        />
      </Panel>

      {selectedDiagnosis && (
        <ClassifierResultModal
          result={selectedDiagnosis}
          fallbackMapStale={fallbackMapStale}
          onClose={() => setSelectedDiagnosis(null)}
        />
      )}
    </>
  );
}

function ClassifierResultsTable({ results, fallbackMapStale, onResultClick }) {
  return (
    <div className="classifier-results-table" role="table" aria-label="Resultados del clasificador">
      <div className="classifier-results-head" role="row">
        <span role="columnheader">Entidad</span>
        <span role="columnheader">Tiempo</span>
        <span role="columnheader">Clase</span>
        <span role="columnheader">Confianza</span>
        <span role="columnheader">Subtipo</span>
        <span role="columnheader">S. estadistico</span>
        <span role="columnheader">Contexto / cobertura</span>
        <span role="columnheader">Causa comun</span>
        <span role="columnheader">Mapa</span>
        <span role="columnheader">Accion</span>
      </div>
      {(results || []).map((result, index) => {
        const confidence = classifierConfidence(result);
        const coverage = classifierContextCoverage(result);
        const commonCause = classifierCommonCause(result);
        const stale = classifierResultMapStale(result, fallbackMapStale);
        return (
          <button
            type="button"
            role="row"
            className="classifier-result-row"
            key={result.diagnosis_id || `${result.entity_id}-${result.timestamp}-${index}`}
            onClick={() => onResultClick(result)}
            title="Ver traza completa del diagnostico"
          >
            <span role="cell">
              <strong>{result.entity_id || "-"}</strong>
              <small>{kindText(result.sensor_kind)}</small>
            </span>
            <TimeStack value={result.timestamp} />
            <span role="cell"><DiagnosticClassBadge value={classifierResultClass(result)} /></span>
            <span role="cell">
              {formatClassifierConfidence(confidence)}
              <small>{classifierConfidenceKindLabel(result.confidence_kind)}</small>
            </span>
            <span role="cell">{classifierSubtype(result)}</span>
            <span role="cell">{formatNumber(classifierStatisticalScore(result))}</span>
            <span role="cell">
              {contextOutcomeLabel(classifierContextOutcome(result))}
              <small>{coverage.ratio === null ? "cobertura -" : `cobertura ${formatContextPercent(coverage.ratio)}`}</small>
            </span>
            <span role="cell">
              <ContextStatusBadge value={commonCause.key} label={commonCause.label} />
            </span>
            <span role="cell">
              <ContextStatusBadge
                value={stale === null ? "unknown" : stale ? "stale" : "current"}
                label={stale === null ? "No informado" : stale ? "Obsoleto" : "Vigente"}
              />
            </span>
            <span role="cell">{classifierActionLabel(result.suggested_action)}</span>
          </button>
        );
      })}
      {!results?.length && <Empty text="No hay diagnosticos que coincidan con los filtros." />}
    </div>
  );
}

function DiagnosticClassBadge({ value }) {
  const key = classifierClassKey(value);
  return (
    <span className={`diagnostic-class-badge ${key}`}>
      {classifierClassLabel(key)}
      {key === "not_reached" && <small>perdida del pipeline</small>}
    </span>
  );
}

function ClassifierResultModal({ result, fallbackMapStale, onClose }) {
  const diagnosticClass = classifierResultClass(result);
  const confidence = classifierConfidence(result);
  const pipeline = result.pipeline || {};
  const statistical = pipeline.statistical || result.statistical || {};
  const contextual = pipeline.contextual || result.contextual || {};
  const classifier = pipeline.classifier || result.classifier || {};
  const coverage = classifierContextCoverage(result);
  const commonCause = classifierCommonCause(result);
  const mapQuality = classifierMapQuality(result);
  const stale = classifierResultMapStale(result, fallbackMapStale);
  const evidenceRows = classifierEvidenceRows(result);
  const reasonCodes = contextArray(result.reason_codes, classifier.reason_codes);
  const limitations = contextArray(result.limitations, classifier.limitations);
  const groundTruth = result.ground_truth || result.ground_truth_label;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section
        className="calculation-modal classifier-detail-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Detalle del diagnostico final"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="modal-header">
          <div>
            <span>Traza completa del tercer filtro</span>
            <h2>{result.entity_id || "-"}</h2>
            <small>{shortDate(result.timestamp)} · {result.diagnosis_id || result.event_id || "sin id de diagnostico"}</small>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Cerrar detalle">
            <X size={18} />
          </button>
        </header>

        <div className="classifier-decision-header">
          <DiagnosticClassBadge value={diagnosticClass} />
          <div>
            <span>Confianza</span>
            <strong>{formatClassifierConfidence(confidence)}</strong>
            <small>{classifierConfidenceKindLabel(result.confidence_kind)}</small>
          </div>
          <div>
            <span>Subtipo</span>
            <strong>{classifierSubtype(result)}</strong>
          </div>
          <div>
            <span>Accion sugerida</span>
            <strong>{classifierActionLabel(result.suggested_action)}</strong>
          </div>
          <div>
            <span>Base de verdad</span>
            <strong>{groundTruth ? classifierEvaluationLabel(groundTruth) : "Sin GT"}</strong>
          </div>
        </div>

        {diagnosticClass === "not_reached" && (
          <div className="classifier-not-reached-warning">
            <AlertTriangle size={18} />
            <span><strong>not_reached no es una clase.</strong> Este registro representa un caso perdido antes de ejecutar el clasificador.</span>
          </div>
        )}

        <article className="modal-block">
          <h3>Traza del flujo secuencial</h3>
          <div className="classifier-pipeline-trace">
            <ClassifierTraceStage
              step="1"
              title="Estadistico-temporal"
              status={classifierStageStatus(statistical)}
              rows={[
                { label: "Score", value: formatNumber(statistical.score ?? statistical.suspicion_score) },
                { label: "Umbral", value: formatNumber(statistical.threshold) },
                { label: "Paso", value: classifierPassedText(statistical.passed) },
                { label: "Patron", value: statistical.pattern || statistical.suspected_pattern || "-" },
                { label: "Indicadores", value: formatContextValue(statistical.indicators || statistical.triggered_indicators) },
                { label: "Tiempo", value: formatElapsed(statistical.elapsed_ms) },
              ]}
            />
            <ClassifierTraceStage
              step="2"
              title="Contextual"
              status={contextOutcome({ context_outcome: contextual.outcome || contextual.context_outcome })}
              rows={[
                { label: "Resultado", value: contextOutcomeLabel(classifierContextOutcome(result)) },
                { label: "Coherencia", value: formatNumber(classifierContextScore(result)) },
                { label: "Cobertura", value: coverage.ratio === null ? "-" : formatContextPercent(coverage.ratio) },
                { label: "Observadas", value: `${formatCount(coverage.observed)} / ${formatCount(coverage.eligible)}` },
                { label: "Causa comun", value: commonCause.label },
                { label: "Tiempo", value: formatElapsed(contextual.elapsed_ms) },
              ]}
            />
            <ClassifierTraceStage
              step="3"
              title="Clasificador"
              status={diagnosticClass}
              rows={[
                { label: "Modelo", value: classifier.model_version || result.model_version || "-" },
                { label: "Regla", value: result.decision_rule_id || classifier.decision_rule_id || "-" },
                { label: "Clase", value: classifierClassLabel(diagnosticClass) },
                { label: "Confianza", value: formatClassifierConfidence(confidence) },
                { label: "Subtipo", value: classifierSubtype(result) },
                { label: "Tiempo", value: formatElapsed(classifier.elapsed_ms) },
              ]}
            />
          </div>
        </article>

        <div className="modal-columns">
          <article className="modal-block">
            <h3>Cobertura y mapa utilizados</h3>
            <KeyValueGrid rows={[
              { label: "Relaciones elegibles", value: formatCount(coverage.eligible) },
              { label: "Relaciones observadas", value: formatCount(coverage.observed) },
              { label: "Cobertura", value: coverage.ratio === null ? "-" : formatContextPercent(coverage.ratio) },
              { label: "Version del mapa", value: mapQuality.version || mapQuality.map_version || "-" },
              { label: "Vigencia al decidir", value: stale === null ? "no informada" : stale ? "obsoleto" : "vigente" },
              { label: "Motivos", value: contextArray(mapQuality.stale_reasons).join(", ") || "ninguno" },
            ]} />
          </article>
          <article className="modal-block">
            <h3>Causa comun y limite causal</h3>
            <ContextStatusBadge value={commonCause.key} label={commonCause.label} />
            <KeyValueGrid rows={[
              { label: "Candidata", value: commonCause.possible ? "si" : "no" },
              { label: "Identificable", value: commonCause.identifiable ? "si" : "no" },
              { label: "Evidencia independiente", value: commonCause.independentEvidence ? "disponible" : "no disponible" },
              { label: "Dominio", value: commonCause.domain || "-" },
              { label: "Interpretacion", value: commonCause.detail || "El historico asociativo no permite atribuir una causa raiz." },
            ]} />
          </article>
        </div>

        <article className="modal-block">
          <h3>Evidencia y variables usadas</h3>
          {evidenceRows.length ? (
            <div className="classifier-evidence-table">
              <div className="classifier-evidence-head">
                <span>Variable / evidencia</span>
                <span>Fuente</span>
                <span>Valor</span>
                <span>Aporte</span>
                <span>Direccion / veredicto</span>
                <span>Disponibilidad</span>
              </div>
              {evidenceRows.map((item, index) => (
                <div className="classifier-evidence-row" key={`${item.name}-${index}`}>
                  <span>{item.name}</span>
                  <span>{item.source}</span>
                  <span>{formatContextValue(item.value)}</span>
                  <span>{formatContextValue(item.contribution)}</span>
                  <span>{item.direction}</span>
                  <span>{item.available ? "disponible" : "ausente"}</span>
                </div>
              ))}
            </div>
          ) : (
            <Empty text="El backend no informo variables ni aportes para este diagnostico." />
          )}
        </article>

        <div className="modal-columns">
          <article className="modal-block">
            <h3>Justificacion</h3>
            <KeyValueGrid rows={[
              { label: "Regla", value: result.decision_rule_id || classifier.decision_rule_id || "-" },
              { label: "Codigos de razon", value: reasonCodes.length ? reasonCodes.join(", ") : "-" },
              { label: "Explicacion", value: result.explanation || classifier.explanation || "-" },
            ]} />
          </article>
          <article className="modal-block">
            <h3>Limitaciones aplicadas</h3>
            {limitations.length ? (
              <ul className="classifier-limitation-list">
                {limitations.map((limitation, index) => <li key={`${formatContextValue(limitation)}-${index}`}>{formatContextValue(limitation)}</li>)}
              </ul>
            ) : (
              <Empty text="No se informaron limitaciones para este resultado." />
            )}
          </article>
        </div>
      </section>
    </div>
  );
}

function ClassifierTraceStage({ step, title, status, rows }) {
  return (
    <article className={`classifier-trace-stage ${contextClassKey(status)}`}>
      <header>
        <span>{step}</span>
        <div>
          <strong>{title}</strong>
          <small>{classifierStageLabel(status)}</small>
        </div>
      </header>
      <KeyValueGrid rows={rows} />
    </article>
  );
}

function ClassifierEvaluationView({ evaluation }) {
  const payload = evaluation || {};
  const available = classifierEvaluationAvailable(payload);
  const warnings = contextArray(payload.warnings);
  const dataset = payload.dataset || {};
  const policy = payload.metric_policy || {};

  if (!available) {
    return (
      <section className="classifier-evaluation-unavailable" aria-label="Evaluacion extremo a extremo no disponible">
        <div className="classifier-evaluation-unavailable-icon"><AlertTriangle size={26} /></div>
        <span className="eyebrow">Fuente actual · mes real</span>
        <h2>Evaluacion E2E no disponible</h2>
        <strong>Pendiente: 1 mes disponible de 5 y sin etiquetas reales suficientes</strong>
        <p>
          No se calculan matriz de confusion, precision, recall, F1 ni falsos negativos con denominadores
          incompletos. La evaluacion se habilitara con fallas controladas en realtime o con futura verdad real suficiente.
        </p>
        <div className="classifier-validation-facts">
          <SummaryRow label="Fuente actual" value="Mes real" />
          <SummaryRow label="Cobertura temporal" value="1 mes de 5" />
          <SummaryRow label="Etiquetas reales" value="insuficientes" />
          <SummaryRow label="Metricas E2E" value="no disponibles" />
        </div>
        {warnings.length > 0 && (
          <div className="classifier-evaluation-warnings">
            <h3>Advertencias informadas</h3>
            <ul>
              {warnings.map((warning, index) => (
                <li key={`${formatContextValue(warning)}-${index}`}>{formatContextValue(warning)}</li>
              ))}
            </ul>
          </div>
        )}
      </section>
    );
  }

  const funnelStages = classifierFunnelStages(payload.funnel);
  const losses = Array.isArray(payload.preclassifier_losses) ? payload.preclassifier_losses : [];
  const endToEnd = payload.end_to_end || {};
  const conditional = payload.classifier_conditional || {};
  const fnBeforeClassifier = classifierPreclassifierFalseNegatives(losses, endToEnd);
  const labeledCases = contextNumber(dataset.labeled_cases, dataset.total_labeled, funnelStages[0]?.value);
  const reachedClassifier = classifierFunnelValue(payload.funnel, "reached_classifier");
  const faultMetrics = classifierClassMetrics(endToEnd, "possible_sensor_fault");
  const macroF1 = contextNumber(endToEnd.macro_f1, endToEnd.metrics?.macro_f1);
  const abstentionRate = contextNumber(endToEnd.abstention_rate, endToEnd.metrics?.abstention_rate);

  return (
    <div className="classifier-evaluation-view">
      <section className="evaluation-scope-banner">
        <div>
          <span className="eyebrow">Metricas del flujo completo</span>
          <h2>Evaluacion extremo a extremo</h2>
          <p>
            Incluye todos los casos reales etiquetados del conjunto evaluable. Los casos que no llegaron al
            clasificador se conservan como <strong>not_reached</strong>; no se convierten en una clase diagnostica.
          </p>
        </div>
        <ContextStatusBadge value="current" label="GT disponible para esta ejecucion" />
      </section>

      <section className="metric-grid classifier-metric-grid">
        <Metric icon={<Database />} label="Casos etiquetados" value={formatOptionalCount(labeledCases)} detail="denominador E2E" />
        <Metric icon={<Activity />} label="Llegaron al clasificador" value={formatOptionalCount(reachedClassifier)} detail="subconjunto condicional" />
        <Metric icon={<AlertTriangle />} label="FN antes del clasificador" value={formatOptionalCount(fnBeforeClassifier)} detail="perdidas de gates previos" />
        <Metric icon={<BarChart3 />} label="Macro F1 E2E" value={formatOptionalMetric(macroF1)} detail="todos los casos etiquetados" />
        <Metric icon={<Zap />} label="Recall posible falla" value={formatOptionalPercent(faultMetrics?.recall)} detail="incluye not_reached" />
        <Metric icon={<Eye />} label="Abstencion" value={formatOptionalPercent(abstentionRate)} detail="prediccion indeterminada" />
      </section>

      <section className="classifier-evaluation-grid">
        <Panel title="Embudo del pipeline">
          <ClassifierFunnel stages={funnelStages} />
        </Panel>
        <Panel title="Fuente y politica de metricas">
          <KeyValueGrid rows={[
            { label: "Fuente actual", value: "Mes real" },
            { label: "Cobertura temporal", value: dataset.months_available && dataset.months_required ? `${dataset.months_available} de ${dataset.months_required} meses` : "periodo real evaluable" },
            { label: "Conjunto", value: dataset.id || dataset.dataset_id || "-" },
            { label: "Particion", value: dataset.split || "-" },
            { label: "Casos inciertos excluidos", value: formatOptionalCount(contextNumber(dataset.uncertain_excluded)) },
            { label: "Politica indeterminado", value: formatContextValue(policy.indeterminate_policy) },
            { label: "Politica gate", value: formatContextValue(policy.gate_rejection_policy || "not_reached") },
            { label: "Umbrales fijos", value: policy.fixed_thresholds === undefined ? "-" : policy.fixed_thresholds ? "si" : "no" },
          ]} />
        </Panel>
      </section>

      <Panel title="Matriz de confusion del pipeline E2E">
        <ClassifierConfusionMatrix matrix={payload.confusion_matrix_e2e} />
      </Panel>

      <Panel title="Falsos negativos descartados antes del clasificador">
        <ClassifierPreclassifierLossTable losses={losses} />
        <p className="classifier-method-note">
          FN E2E(c) = FN antes del clasificador(c) + FN dentro del clasificador(c).
          El recall E2E usa todos los casos reales etiquetados de la clase como denominador.
        </p>
      </Panel>

      <section className="classifier-metric-scope-grid">
        <ClassifierMetricScopePanel
          title="Desempeno E2E"
          scope="Todos los casos etiquetados, incluidos not_reached"
          metrics={endToEnd}
          highlight
        />
        <ClassifierMetricScopePanel
          title="Desempeno condicional"
          scope="Solo casos que recibio el clasificador"
          metrics={conditional}
        />
      </section>

      {warnings.length > 0 && (
        <section className="classifier-evaluation-warnings">
          <h3>Advertencias de evaluacion</h3>
          <ul>
            {warnings.map((warning, index) => (
              <li key={`${formatContextValue(warning)}-${index}`}>{formatContextValue(warning)}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function ClassifierFunnel({ stages }) {
  if (!stages.length) {
    return <Empty text="No se informaron conteos por etapa." />;
  }
  const base = stages.find((stage) => stage.value !== null)?.value;
  return (
    <div className="classifier-funnel">
      {stages.map((stage, index) => {
        const ratio = base && stage.value !== null ? stage.value / base : null;
        return (
          <React.Fragment key={stage.id}>
            <article className={`classifier-funnel-stage ${stage.id}`}>
              <span>{stage.label}</span>
              <strong>{formatOptionalCount(stage.value)}</strong>
              <small>{ratio === null ? "proporcion no informada" : formatContextPercent(ratio)}</small>
            </article>
            {index < stages.length - 1 && <div className="classifier-funnel-arrow" aria-hidden="true">→</div>}
          </React.Fragment>
        );
      })}
    </div>
  );
}

function ClassifierConfusionMatrix({ matrix }) {
  const normalized = normalizeClassifierConfusionMatrix(matrix);
  if (!normalized.truthLabels.length || !normalized.predictedLabels.length || !normalized.values.length) {
    return <Empty text="La matriz de confusion no esta disponible para esta ejecucion." />;
  }
  const maximum = Math.max(
    ...normalized.values.flatMap((row) => row.map((value) => contextFiniteNumber(value) ?? 0)),
    1,
  );
  return (
    <>
      <div className="classifier-confusion-scroll">
        <table className="classifier-confusion-matrix">
          <thead>
            <tr>
              <th scope="col">Real ↓ / Predicho →</th>
              {normalized.predictedLabels.map((label) => (
                <th scope="col" className={classifierClassKey(label)} key={label}>
                  {classifierEvaluationLabel(label)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {normalized.truthLabels.map((truth, rowIndex) => (
              <tr key={truth}>
                <th scope="row">{classifierEvaluationLabel(truth)}</th>
                {normalized.predictedLabels.map((predicted, columnIndex) => {
                  const value = contextFiniteNumber(normalized.values[rowIndex]?.[columnIndex]);
                  return (
                    <td
                      className={classifierClassKey(predicted)}
                      style={value === null ? undefined : { backgroundColor: classifierConfusionCellColor(truth, predicted, value, maximum) }}
                      title={`${classifierEvaluationLabel(truth)} → ${classifierEvaluationLabel(predicted)}: ${formatOptionalCount(value)}`}
                      key={predicted}
                    >
                      {formatOptionalCount(value)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="classifier-method-note">
        <strong>not_reached</strong> es una columna de perdida del pipeline: identifica casos reales que no
        alcanzaron el tercer filtro. <strong>Indeterminado</strong> si es una salida diagnostica de abstencion.
      </p>
    </>
  );
}

function ClassifierPreclassifierLossTable({ losses }) {
  if (!losses.length) {
    return <Empty text="No se informaron perdidas por etapa." />;
  }
  return (
    <div className="classifier-loss-table">
      <div className="classifier-loss-head">
        <span>Gate</span>
        <span>Clase real</span>
        <span>Soporte antes</span>
        <span>Descartados</span>
        <span>FN atribuibles</span>
        <span>Motivo</span>
      </div>
      {losses.map((loss, index) => (
        <div className="classifier-loss-row" key={`${loss.gate}-${loss.truth_class}-${index}`}>
          <span>{classifierStageLabel(loss.gate)}</span>
          <span>{classifierClassLabel(loss.truth_class)}</span>
          <span>{formatOptionalCount(contextNumber(loss.support_before_gate, loss.support))}</span>
          <span>{formatOptionalCount(contextNumber(loss.rejected, loss.not_reached))}</span>
          <span className="danger-value">{formatOptionalCount(contextNumber(loss.false_negatives, loss.fn))}</span>
          <span>{formatContextValue(loss.reason || loss.reason_code)}</span>
        </div>
      ))}
    </div>
  );
}

function ClassifierMetricScopePanel({ title, scope, metrics, highlight = false }) {
  const payload = metrics?.metrics || metrics || {};
  const byClass = metrics?.by_class || payload.by_class || {};
  const summaryRows = [
    ["Accuracy", payload.accuracy],
    ["Macro F1", payload.macro_f1],
    ["F1 ponderado", payload.weighted_f1],
    ["Balanced accuracy", payload.balanced_accuracy],
    ["Tasa de abstencion", payload.abstention_rate],
  ].filter(([, value]) => contextFiniteNumber(value) !== null);
  return (
    <section className={`panel classifier-metric-scope ${highlight ? "highlight" : ""}`}>
      <div className="classifier-metric-scope-heading">
        <div>
          <h2>{title}</h2>
          <span>{scope}</span>
        </div>
        <ContextStatusBadge value={highlight ? "current" : "neutral"} label={highlight ? "Metrica principal" : "Vista condicional"} />
      </div>
      {summaryRows.length ? (
        <div className="classifier-scope-metrics">
          {summaryRows.map(([label, value]) => (
            <SummaryRow
              label={label}
              value={label.toLowerCase().includes("tasa") ? formatOptionalPercent(value) : formatOptionalMetric(value)}
              key={label}
            />
          ))}
        </div>
      ) : (
        <Empty text="No se informaron metricas agregadas para este alcance." />
      )}
      <ClassifierClassMetricsTable byClass={byClass} conditional={!highlight} />
    </section>
  );
}

function ClassifierClassMetricsTable({ byClass, conditional }) {
  const entries = Object.entries(byClass || {}).filter(([, value]) => value && typeof value === "object");
  if (!entries.length) {
    return null;
  }
  return (
    <div className="classifier-class-metrics">
      <div className="classifier-class-metrics-head">
        <span>Clase</span>
        <span>Soporte</span>
        <span>TP</span>
        <span>FP</span>
        <span>FN</span>
        {!conditional && <span>FN antes</span>}
        <span>Precision</span>
        <span>Recall</span>
        <span>F1</span>
      </div>
      {entries.map(([diagnosticClass, metrics]) => (
        <div className="classifier-class-metrics-row" key={diagnosticClass}>
          <span><DiagnosticClassBadge value={diagnosticClass} /></span>
          <span>{formatOptionalCount(contextNumber(metrics.support_all, metrics.support))}</span>
          <span>{formatOptionalCount(contextNumber(metrics.tp))}</span>
          <span>{formatOptionalCount(contextNumber(metrics.fp))}</span>
          <span>{formatOptionalCount(contextNumber(metrics.fn_e2e, metrics.fn))}</span>
          {!conditional && <span className="danger-value">{formatOptionalCount(contextNumber(metrics.fn_before_classifier))}</span>}
          <span>{formatOptionalPercent(metrics.precision)}</span>
          <span>{formatOptionalPercent(metrics.recall)}</span>
          <span>{formatOptionalMetric(metrics.f1)}</span>
        </div>
      ))}
    </div>
  );
}

function ClassifierSpecificationView({ specification }) {
  const spec = normalizeClassifierSpecificationPayload(specification);
  const status = classifierSpecificationStatus(spec);
  const config = spec.config && typeof spec.config === "object" ? spec.config : {};
  const features = normalizeClassifierFeatures(spec.features);
  const decisions = normalizeClassifierDecisionTable(spec.decision_table);
  const protocol = spec.validation_protocol && typeof spec.validation_protocol === "object"
    ? spec.validation_protocol
    : {};
  const hasSpecification = Boolean(
    spec.model_version
    || spec.version
    || spec.model_type
    || Object.keys(config).length
    || features.length
    || decisions.length,
  );

  if (!hasSpecification) {
    return (
      <section className="classifier-specification-empty">
        <FlaskConical size={28} />
        <h2>Especificacion no disponible</h2>
        <p>El backend aun no informa version, variables, reglas ni umbrales del clasificador.</p>
      </section>
    );
  }

  return (
    <div className="classifier-specification-view">
      <section className={`classifier-specification-banner ${status.key}`}>
        <div>
          <span className="eyebrow">Especificacion versionada</span>
          <h2>{classifierModelVersion(spec)} · {spec.model_type || spec.type || "tipo no informado"}</h2>
          <p>{status.label}. La interfaz muestra las reglas del backend y no completa umbrales ausentes.</p>
        </div>
        <DiagnosticClassBadge value={status.key === "ready" ? "normal" : "indeterminate"} />
      </section>

      <section className="classifier-specification-grid">
        <Panel title="Configuracion del modelo">
          <KeyValueGrid rows={[
            { label: "Version", value: classifierModelVersion(spec) },
            { label: "Tipo", value: spec.model_type || spec.type || "-" },
            { label: "Estado", value: status.label },
            { label: "Esquema", value: spec.schema_version || "-" },
            { label: "Confianza", value: spec.confidence_kind || config.confidence_kind || "-" },
            ...Object.entries(config).map(([key, value]) => ({ label: key, value: formatContextValue(value) })),
          ]} />
        </Panel>
        <Panel title="Estado de validacion">
          <div className="classifier-validation-state">
            <AlertTriangle size={18} />
            <strong>Pendiente: 1 mes disponible de 5 y sin etiquetas reales suficientes</strong>
          </div>
          <KeyValueGrid rows={[
            { label: "Fuente actual", value: "Mes real" },
            { label: "Cobertura temporal", value: "1 mes de 5" },
            { label: "Evaluacion E2E", value: "no disponible" },
            { label: "Protocolo", value: formatContextValue(protocol.status || status.label) },
            { label: "Umbrales congelados", value: protocol.fixed_thresholds === undefined ? "-" : protocol.fixed_thresholds ? "si" : "no" },
          ]} />
          <p className="classifier-method-note">
            La validacion cuantitativa se habilitara con fallas controladas en realtime o futura verdad real suficiente.
          </p>
        </Panel>
      </section>

      <Panel title="Variables de entrada">
        <ClassifierFeaturesTable features={features} />
      </Panel>

      <Panel title="Tabla de decision">
        <ClassifierDecisionTable decisions={decisions} />
      </Panel>
    </div>
  );
}

function ClassifierFeaturesTable({ features }) {
  if (!features.length) {
    return <Empty text="No se informaron variables del modelo." />;
  }
  return (
    <div className="classifier-features-table">
      <div className="classifier-features-head">
        <span>Variable</span>
        <span>Fuente</span>
        <span>Tipo</span>
        <span>Requerida</span>
        <span>Normalizacion / dominio</span>
      </div>
      {features.map((feature, index) => (
        <div className="classifier-features-row" key={`${feature.name}-${index}`}>
          <span>{feature.name}</span>
          <span>{feature.source}</span>
          <span>{feature.type}</span>
          <span>{feature.required === null ? "-" : feature.required ? "si" : "no"}</span>
          <span>{feature.detail}</span>
        </div>
      ))}
    </div>
  );
}

function ClassifierDecisionTable({ decisions }) {
  if (!decisions.length) {
    return <Empty text="No se informaron reglas o nodos de decision." />;
  }
  return (
    <div className="classifier-decision-table">
      <div className="classifier-decision-head">
        <span>Regla</span>
        <span>Condiciones</span>
        <span>Clase</span>
        <span>Subtipo</span>
        <span>Confianza / umbral</span>
        <span>Accion</span>
      </div>
      {decisions.map((decision, index) => (
        <div className="classifier-decision-row" key={`${decision.id}-${index}`}>
          <span>{decision.id}</span>
          <span>{decision.conditions}</span>
          <span>{decision.diagnosticClass ? <DiagnosticClassBadge value={decision.diagnosticClass} /> : "-"}</span>
          <span>{decision.subtype}</span>
          <span>{decision.confidence}</span>
          <span>{classifierActionLabel(decision.action)}</span>
        </div>
      ))}
    </div>
  );
}

function classifierModelVersion(specification = {}, summary = {}) {
  if (typeof specification === "string" && specification.trim()) {
    return specification;
  }
  const model = specification?.model;
  const summaryModel = summary?.model;
  if (typeof model === "string" && model.trim()) {
    return model;
  }
  if (typeof summaryModel === "string" && summaryModel.trim()) {
    return summaryModel;
  }
  return specification?.model_version
    || specification?.version
    || model?.version
    || summary?.model_version
    || summaryModel?.version
    || "-";
}

function normalizeClassifierSpecificationPayload(value) {
  if (!value) {
    return {};
  }
  if (typeof value === "string" || typeof value === "number") {
    return { model_version: String(value) };
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value;
}

function classifierSpecificationStatus(specification = {}) {
  const spec = normalizeClassifierSpecificationPayload(specification);
  const hasSpecification = Boolean(
    spec
    && (
      spec.model_version
      || spec.version
      || spec.model_type
      || spec.type
      || spec.config
      || spec.features
      || spec.decision_table
    )
  );
  if (!hasSpecification) {
    return { key: "unavailable", label: "Especificacion no disponible" };
  }
  const rawStatus = typeof spec.status === "object"
    ? spec.status.key || spec.status.status || spec.status.value
    : spec.status;
  const key = contextClassKey(rawStatus || "pending");
  if (["ready", "active", "configured", "operational", "validated"].includes(key)) {
    return { key: "ready", label: "Modelo configurado" };
  }
  if (["invalid", "error", "blocked", "disabled"].includes(key)) {
    return { key: "invalid", label: "Especificacion requiere correccion" };
  }
  return { key: "pending", label: "Especificacion pendiente de validacion" };
}

function classifierClassKey(value) {
  const rawValue = value && typeof value === "object"
    ? value.diagnostic_class || value.class || value.label || value.value
    : value;
  const key = contextClassKey(rawValue || "unknown");
  const aliases = {
    normal_operation: "normal",
    healthy: "normal",
    ambiental_anomaly: "environmental_anomaly",
    anomalia_ambiental: "environmental_anomaly",
    environmental_event: "environmental_anomaly",
    environment_anomaly: "environmental_anomaly",
    possible_fault: "possible_sensor_fault",
    possible_device_fault: "possible_sensor_fault",
    device_fault: "possible_sensor_fault",
    possible_sensor_failure: "possible_sensor_fault",
    possible_failure: "possible_sensor_fault",
    falla_posible: "possible_sensor_fault",
    posible_falla: "possible_sensor_fault",
    posible_falla_de_sensor: "possible_sensor_fault",
    uncertain: "indeterminate",
    inconclusive: "indeterminate",
    insufficient_evidence: "indeterminate",
    no_determinado: "indeterminate",
    no_reached: "not_reached",
    not_evaluated: "not_reached",
    rejected_before_classifier: "not_reached",
    pipeline_loss: "not_reached",
  };
  return aliases[key] || key;
}

function classifierClassLabel(value) {
  const key = classifierClassKey(value);
  const labels = {
    normal: "Normal",
    environmental_anomaly: "Anomalia ambiental",
    possible_sensor_fault: "Posible falla de sensor",
    sensor_fault: "Falla de sensor",
    indeterminate: "Indeterminado",
    not_reached: "No alcanzo el clasificador",
    unknown: "No informado",
  };
  if (labels[key]) {
    return labels[key];
  }
  return String(value || "Indeterminado")
    .replace(/[_-]+/g, " ")
    .replace(/^\w/, (letter) => letter.toUpperCase());
}

function classifierResultClass(result = {}) {
  const key = classifierClassKey(
    result.diagnostic_class
    || result.classification
    || result.predicted_class
    || result.class
    || result.pipeline?.classifier?.diagnostic_class
    || result.pipeline?.classifier?.predicted_class
    || "unknown",
  );
  return key === "sensor_fault" ? "possible_sensor_fault" : key;
}

function classifierClassCounts(results = [], summaryByClass = {}) {
  const computed = {
    normal: 0,
    environmental_anomaly: 0,
    possible_sensor_fault: 0,
    indeterminate: 0,
  };
  (results || []).forEach((result) => {
    const key = classifierResultClass(result);
    if (Object.prototype.hasOwnProperty.call(computed, key)) {
      computed[key] += 1;
    }
  });
  const reported = {};
  if (Array.isArray(summaryByClass)) {
    summaryByClass.forEach((item) => {
      if (!item || typeof item !== "object") {
        return;
      }
      const key = classifierResultClass({
        diagnostic_class: item.diagnostic_class || item.class || item.label,
      });
      const count = contextNumber(item.count, item.value, item.total);
      if (count !== null) {
        reported[key] = (reported[key] || 0) + count;
      }
    });
  } else if (summaryByClass && typeof summaryByClass === "object") {
    Object.entries(summaryByClass).forEach(([diagnosticClass, value]) => {
      const key = classifierResultClass({ diagnostic_class: diagnosticClass });
      const count = contextNumber(
        typeof value === "object" ? value.count : value,
        typeof value === "object" ? value.total : null,
      );
      if (count !== null) {
        reported[key] = (reported[key] || 0) + count;
      }
    });
  }
  return Object.fromEntries(
    Object.entries(computed).map(([key, value]) => [
      key,
      Object.prototype.hasOwnProperty.call(reported, key) ? reported[key] : value,
    ]),
  );
}

function mergeClassifierResults(current = [], incoming = []) {
  const merged = new Map();
  [...(current || []), ...(incoming || [])].forEach((result, index) => {
    const compositeKey = [
      result?.entity_id,
      result?.timestamp,
      classifierResultClass(result || {}),
    ].filter(Boolean).join("|");
    const key = result?.diagnosis_id
      || result?.event_id
      || compositeKey
      || `result-${index}`;
    merged.set(key, result);
  });
  return [...merged.values()].sort((left, right) => {
    const leftTime = Date.parse(left?.timestamp || "");
    const rightTime = Date.parse(right?.timestamp || "");
    if (!Number.isFinite(leftTime) || !Number.isFinite(rightTime)) {
      return 0;
    }
    return rightTime - leftTime;
  });
}

function classifierConfidence(result = {}) {
  return contextNumber(
    result.confidence,
    result.pipeline?.classifier?.confidence,
    result.classifier?.confidence,
  );
}

function normalizedConfidence(value) {
  const number = contextFiniteNumber(value);
  if (number === null || number < 0) {
    return null;
  }
  const normalized = number > 1 ? number / 100 : number;
  return normalized >= 0 && normalized <= 1 ? normalized : null;
}

function classifierIsLowConfidence(result = {}) {
  if (result.low_confidence !== undefined && result.low_confidence !== null) {
    return contextBoolean(result.low_confidence);
  }
  const confidence = normalizedConfidence(classifierConfidence(result));
  const threshold = normalizedConfidence(contextNumber(
    result.confidence_threshold,
    result.low_confidence_threshold,
    result.pipeline?.classifier?.confidence_threshold,
  ));
  return confidence !== null && threshold !== null ? confidence < threshold : false;
}

function classifierMeanConfidence(results = []) {
  const values = (results || [])
    .map(classifierConfidence)
    .map(normalizedConfidence)
    .filter((value) => value !== null);
  if (!values.length) {
    return null;
  }
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function formatClassifierConfidence(value) {
  const normalized = normalizedConfidence(value);
  return normalized === null ? "-" : `${(normalized * 100).toFixed(1)}%`;
}

function classifierSubtype(result = {}) {
  const value = result.subtype
    || result.diagnostic_subtype
    || result.pipeline?.classifier?.subtype
    || result.classifier?.subtype;
  return value ? formatContextValue(value) : "-";
}

function classifierContextBlock(result = {}) {
  return result.pipeline?.contextual || result.contextual || {};
}

function classifierContextCoverage(result = {}) {
  const contextual = classifierContextBlock(result);
  const coverage = contextual.coverage
    ?? contextual.context_coverage
    ?? result.context_coverage
    ?? result.coverage;
  return contextCoverage({
    coverage,
    eligible_relations: contextual.eligible_relations ?? result.eligible_relations,
    observed_relations: contextual.observed_relations ?? result.observed_relations,
    coverage_ratio: contextual.coverage_ratio ?? result.coverage_ratio,
    coverage_status: contextual.coverage_status ?? result.coverage_status,
  });
}

function classifierCommonCause(result = {}) {
  const contextual = classifierContextBlock(result);
  const raw = contextual.common_cause ?? result.common_cause;
  return contextCommonCause({
    common_cause: raw,
    common_cause_possible: contextual.common_cause_possible ?? result.common_cause_possible,
    common_cause_identifiable: contextual.common_cause_identifiable ?? result.common_cause_identifiable,
    independent_signal_present: contextual.independent_signal_present ?? result.independent_signal_present,
    common_cause_score: contextual.common_cause_score ?? result.common_cause_score,
    common_failure_domain: contextual.common_failure_domain ?? result.common_failure_domain,
    common_cause_explanation: contextual.common_cause_explanation ?? result.common_cause_explanation,
  });
}

function classifierMapQuality(result = {}) {
  const contextual = classifierContextBlock(result);
  const quality = contextual.map_quality
    || contextual.map
    || result.map_quality
    || result.map
    || {};
  if (!quality || typeof quality !== "object") {
    return {};
  }
  return {
    ...quality,
    version: quality.version || quality.map_version || contextual.map_version || result.map_version,
    stale: quality.stale ?? contextual.map_stale ?? result.map_stale,
    stale_reasons: contextArray(quality.stale_reasons, contextual.stale_reasons, result.stale_reasons),
  };
}

function classifierResultMapStale(result = {}, fallback = null) {
  const quality = classifierMapQuality(result);
  const value = quality.stale
    ?? result.pipeline?.contextual?.map_stale
    ?? result.contextual?.map_stale
    ?? result.map_stale;
  if (value === undefined || value === null) {
    return fallback === undefined || fallback === null ? null : Boolean(fallback);
  }
  return contextBoolean(value);
}

function classifierStatisticalScore(result = {}) {
  const statistical = result.pipeline?.statistical || result.statistical || {};
  return contextNumber(
    statistical.score,
    statistical.suspicion_score,
    result.statistical_score,
    result.stat_score,
    result.suspicion_score,
  );
}

function classifierContextOutcome(result = {}) {
  const contextual = classifierContextBlock(result);
  return contextOutcome({
    context_outcome: contextual.outcome
      || contextual.context_outcome
      || contextual.status
      || result.context_outcome
      || result.contextual_status,
  });
}

function classifierContextScore(result = {}) {
  const contextual = classifierContextBlock(result);
  return contextNumber(
    contextual.coherence_score,
    contextual.contextual_score,
    contextual.score,
    result.coherence_score,
    result.contextual_score,
    result.context_score,
  );
}

function classifierActionLabel(action) {
  if (action === undefined || action === null || action === "") {
    return "-";
  }
  if (typeof action === "object") {
    return formatContextValue(action.label || action.action || action.code || action);
  }
  const key = contextClassKey(action);
  const labels = {
    none: "Sin accion",
    no_action: "Sin accion",
    monitor: "Monitorear",
    continue_monitoring: "Continuar monitoreo",
    inspect: "Inspeccionar",
    inspect_sensor: "Inspeccionar sensor",
    review_sensor: "Revisar sensor",
    maintenance: "Derivar a mantenimiento",
    create_alert: "Emitir alerta",
    alert: "Emitir alerta",
    suppress_alert: "No alertar",
    request_more_context: "Solicitar mas contexto",
    collect_more_evidence: "Recopilar mas evidencia",
    accept_candidate_as_normal: "Aceptar candidato como normal",
    observe_environmental_event: "Observar evento ambiental",
    alert_and_hold_sensitive_automation_recommended: "Alertar; retencion automatica recomendada",
    inspect_sensor_before_automation: "Inspeccionar sensor antes de automatizar",
    collect_context_or_independent_evidence: "Recopilar contexto o evidencia independiente",
  };
  return labels[key] || String(action).replace(/[_-]+/g, " ");
}

function classifierConfidenceKindLabel(value) {
  const key = contextClassKey(value);
  const labels = {
    evidence_strength_not_calibrated_probability: "Fuerza de evidencia; no es probabilidad calibrada",
  };
  return labels[key] || value || "Tipo no informado";
}

function classifierEvidenceRows(result = {}) {
  const classifier = result.pipeline?.classifier || result.classifier || {};
  const sources = [
    ["evidencia", result.evidence ?? classifier.evidence],
    ["variable", result.features_used ?? classifier.features_used],
  ];
  const rows = [];
  const append = (sourceLabel, item, fallbackName = "") => {
    if (item === undefined || item === null) {
      return;
    }
    if (typeof item !== "object" || Array.isArray(item)) {
      rows.push({
        name: fallbackName || String(item),
        source: sourceLabel,
        value: fallbackName ? item : "-",
        contribution: "-",
        direction: "-",
        available: true,
      });
      return;
    }
    const available = item.available === undefined && item.missing === undefined
      ? true
      : item.available !== undefined
        ? contextBoolean(item.available)
        : !contextBoolean(item.missing);
    rows.push({
      name: item.name
        || item.feature
        || item.indicator
        || item.entity_id
        || item.related_entity_id
        || fallbackName
        || "evidencia",
      source: item.source || item.stage || item.origin || sourceLabel,
      value: item.value ?? item.observed_value ?? item.score ?? item.c_j ?? "-",
      contribution: item.contribution
        ?? item.weighted_contribution
        ?? item.weight
        ?? item.importance
        ?? "-",
      direction: formatContextValue(
        item.direction
        ?? item.verdict
        ?? item.status
        ?? item.effect
        ?? "-",
      ),
      available,
    });
  };
  sources.forEach(([sourceLabel, collection]) => {
    if (Array.isArray(collection)) {
      collection.forEach((item) => append(sourceLabel, item));
      return;
    }
    if (!collection || typeof collection !== "object") {
      if (collection !== undefined && collection !== null) {
        append(sourceLabel, collection);
      }
      return;
    }
    const items = Array.isArray(collection.items) ? collection.items : null;
    if (items) {
      items.forEach((item) => append(sourceLabel, item));
      return;
    }
    Object.entries(collection).forEach(([name, item]) => append(sourceLabel, item, name));
  });
  const unique = new Map();
  rows.forEach((row) => {
    const key = `${contextClassKey(row.name)}|${contextClassKey(row.source)}`;
    if (!unique.has(key)) {
      unique.set(key, row);
    }
  });
  return [...unique.values()];
}

function classifierStageStatus(stage = {}) {
  if (typeof stage === "string") {
    return contextClassKey(stage);
  }
  const raw = stage.status || stage.outcome || stage.result;
  if (raw) {
    return contextClassKey(raw);
  }
  if (stage.passed === true) {
    return "passed";
  }
  if (stage.passed === false) {
    return "rejected";
  }
  return "unknown";
}

function classifierPassedText(value) {
  if (value === undefined || value === null) {
    return "-";
  }
  return contextBoolean(value) ? "si" : "no";
}

function classifierStageLabel(status) {
  const key = contextClassKey(status);
  const labels = {
    passed: "Paso",
    pass: "Paso",
    accepted: "Aceptado",
    rejected: "Descartado",
    failed: "No paso",
    not_reached: "No alcanzado",
    statistical: "Estadistico-temporal",
    statistical_temporal: "Estadistico-temporal",
    contextual: "Contextual",
    classifier: "Clasificador",
    respaldado: "Respaldado",
    contradicho: "Contradicho",
    no_explicado: "No explicado",
    cobertura_insuficiente: "Cobertura insuficiente",
    unknown: "Estado no informado",
  };
  return labels[key] || classifierClassLabel(key);
}

function classifierEvaluationAvailable(payload = {}) {
  if (payload.available !== true) {
    return false;
  }
  const matrix = normalizeClassifierConfusionMatrix(payload.confusion_matrix_e2e);
  const hasMatrixValue = matrix.values.some((row) => row.some((value) => contextFiniteNumber(value) !== null));
  const hasFunnelValue = classifierFunnelStages(payload.funnel)
    .some((stage) => contextFiniteNumber(stage.value) !== null);
  const metricBlocks = [
    payload.end_to_end,
    payload.end_to_end?.metrics,
    payload.classifier_conditional,
    payload.classifier_conditional?.metrics,
  ];
  const hasMetric = metricBlocks.some((block) => (
    block
    && typeof block === "object"
    && Object.entries(block).some(([key, value]) => key !== "by_class" && contextFiniteNumber(value) !== null)
  ));
  const hasClassMetric = [payload.end_to_end, payload.classifier_conditional].some((block) => {
    const byClass = block?.by_class || block?.metrics?.by_class;
    return byClass && typeof byClass === "object" && Object.values(byClass).some((metrics) => (
      metrics
      && typeof metrics === "object"
      && Object.values(metrics).some((value) => contextFiniteNumber(value) !== null)
    ));
  });
  return hasMatrixValue || hasFunnelValue || hasMetric || hasClassMetric;
}

function classifierFunnelStages(funnel) {
  const rawStages = Array.isArray(funnel)
    ? funnel
    : Array.isArray(funnel?.stages)
      ? funnel.stages
      : null;
  if (rawStages) {
    return rawStages.map((stage, index) => {
      if (stage && typeof stage === "object") {
        const id = contextClassKey(stage.id || stage.key || stage.stage || `stage_${index + 1}`);
        return {
          id,
          label: stage.label || classifierFunnelStageLabel(id),
          value: contextNumber(stage.value, stage.count, stage.total),
        };
      }
      return {
        id: `stage_${index + 1}`,
        label: `Etapa ${index + 1}`,
        value: contextFiniteNumber(stage),
      };
    });
  }
  if (!funnel || typeof funnel !== "object") {
    return [];
  }
  const definitions = [
    ["all_labeled", ["all_labeled", "labeled_cases", "total_labeled"]],
    ["passed_statistical", ["passed_statistical", "statistical_passed"]],
    ["contextual_evaluated", ["contextual_evaluated", "passed_contextual", "contextual_passed"]],
    ["reached_classifier", ["reached_classifier", "classifier_inputs"]],
    ["classified", ["classified", "classifier_outputs"]],
    ["indeterminate", ["indeterminate", "abstained"]],
  ];
  return definitions.flatMap(([id, keys]) => {
    const key = keys.find((candidate) => funnel[candidate] !== undefined && funnel[candidate] !== null);
    if (!key) {
      return [];
    }
    return [{
      id,
      label: classifierFunnelStageLabel(id),
      value: contextFiniteNumber(funnel[key]),
    }];
  });
}

function classifierFunnelStageLabel(value) {
  const key = contextClassKey(value);
  const labels = {
    all_labeled: "Casos reales etiquetados",
    labeled_cases: "Casos reales etiquetados",
    passed_statistical: "Pasan filtro estadistico",
    contextual_evaluated: "Evaluados por contexto",
    reached_classifier: "Llegan al clasificador",
    classified: "Salida diagnostica",
    indeterminate: "Salida indeterminada",
  };
  return labels[key] || classifierStageLabel(key);
}

function classifierFunnelValue(funnel, target) {
  const targetKey = contextClassKey(target);
  const stage = classifierFunnelStages(funnel).find((item) => item.id === targetKey);
  return stage ? contextFiniteNumber(stage.value) : null;
}

function classifierPreclassifierFalseNegatives(losses = [], endToEnd = {}) {
  const values = [];
  (losses || []).forEach((loss) => {
    const value = contextNumber(loss?.false_negatives, loss?.fn);
    if (value !== null) {
      values.push(value);
    }
  });
  if (!values.length) {
    const byClass = endToEnd?.by_class || endToEnd?.metrics?.by_class;
    if (byClass && typeof byClass === "object") {
      Object.values(byClass).forEach((metrics) => {
        const value = contextFiniteNumber(metrics?.fn_before_classifier);
        if (value !== null) {
          values.push(value);
        }
      });
    }
  }
  return values.length ? values.reduce((total, value) => total + value, 0) : null;
}

function classifierClassMetrics(metrics = {}, targetClass) {
  const byClass = metrics?.by_class || metrics?.metrics?.by_class || {};
  const entries = Array.isArray(byClass)
    ? byClass.map((item) => [item?.class || item?.label || item?.diagnostic_class, item])
    : Object.entries(byClass);
  const target = classifierClassKey(targetClass);
  const aliases = target === "possible_sensor_fault"
    ? ["possible_sensor_fault", "sensor_fault", "possible_fault"]
    : target === "sensor_fault"
      ? ["sensor_fault", "possible_sensor_fault", "possible_fault"]
      : [target];
  for (const alias of aliases) {
    const match = entries.find(([key]) => contextClassKey(key) === contextClassKey(alias));
    if (match?.[1] && typeof match[1] === "object") {
      return match[1];
    }
  }
  return null;
}

function formatOptionalCount(value) {
  const number = contextFiniteNumber(value);
  return number === null ? "-" : formatCount(number);
}

function formatOptionalMetric(value) {
  const number = contextFiniteNumber(value);
  return number === null ? "-" : number.toFixed(3);
}

function formatOptionalPercent(value) {
  const number = contextFiniteNumber(value);
  if (number === null) {
    return "-";
  }
  return number > 1 ? `${number.toFixed(1)}%` : `${(number * 100).toFixed(1)}%`;
}

function normalizeClassifierConfusionMatrix(matrix) {
  if (!matrix) {
    return { truthLabels: [], predictedLabels: [], values: [] };
  }
  if (Array.isArray(matrix)) {
    if (!matrix.length) {
      return { truthLabels: [], predictedLabels: [], values: [] };
    }
    if (Array.isArray(matrix[0])) {
      const labels = matrix.map((_, index) => `class_${index + 1}`);
      return { truthLabels: labels, predictedLabels: labels, values: matrix };
    }
    const rows = matrix.filter((row) => row && typeof row === "object");
    const predictedLabels = [...new Set(rows.flatMap((row) => (
      row.predictions && typeof row.predictions === "object"
        ? Object.keys(row.predictions)
        : Object.keys(row).filter((key) => !["truth", "truth_class", "actual", "label"].includes(key))
    )))];
    return {
      truthLabels: rows.map((row, index) => row.truth || row.truth_class || row.actual || row.label || `class_${index + 1}`),
      predictedLabels,
      values: rows.map((row) => predictedLabels.map((label) => (
        row.predictions?.[label] ?? row[label] ?? null
      ))),
    };
  }
  if (typeof matrix !== "object") {
    return { truthLabels: [], predictedLabels: [], values: [] };
  }
  const values = matrix.values || matrix.matrix || matrix.counts;
  const truthLabels = matrix.truth_labels || matrix.actual_labels || matrix.rows || matrix.labels;
  const predictedLabels = matrix.predicted_labels || matrix.prediction_labels || matrix.columns || matrix.labels;
  if (Array.isArray(values) && Array.isArray(truthLabels) && Array.isArray(predictedLabels)) {
    return { truthLabels, predictedLabels, values };
  }
  const nested = matrix.by_truth || matrix.data || matrix.cells;
  const objectRows = nested && typeof nested === "object" && !Array.isArray(nested)
    ? nested
    : Object.fromEntries(
      Object.entries(matrix).filter(([, value]) => value && typeof value === "object" && !Array.isArray(value)),
    );
  const nestedEntries = Object.entries(objectRows);
  if (!nestedEntries.length) {
    return { truthLabels: [], predictedLabels: [], values: [] };
  }
  const nestedPredicted = [...new Set(nestedEntries.flatMap(([, row]) => Object.keys(row)))];
  return {
    truthLabels: nestedEntries.map(([truth]) => truth),
    predictedLabels: nestedPredicted,
    values: nestedEntries.map(([, row]) => nestedPredicted.map((predicted) => row[predicted] ?? null)),
  };
}

function classifierEvaluationLabel(label) {
  const rawKey = contextClassKey(label);
  if (rawKey === "sensor_fault") {
    return "Falla de sensor";
  }
  if (classifierClassKey(rawKey) === "not_reached") {
    return "No alcanzo el clasificador";
  }
  return classifierClassLabel(rawKey);
}

function classifierConfusionCellColor(truth, predicted, value, maximum) {
  const truthKey = classifierClassKey(truth) === "sensor_fault"
    ? "possible_sensor_fault"
    : classifierClassKey(truth);
  const predictedKey = classifierClassKey(predicted) === "sensor_fault"
    ? "possible_sensor_fault"
    : classifierClassKey(predicted);
  const ratio = Math.min(1, Math.max(0, Number(value) / Math.max(Number(maximum), 1)));
  const alpha = (0.08 + ratio * 0.68).toFixed(3);
  if (truthKey === predictedKey) {
    return `rgba(45, 212, 191, ${alpha})`;
  }
  if (predictedKey === "not_reached") {
    return `rgba(244, 63, 94, ${alpha})`;
  }
  if (predictedKey === "indeterminate") {
    return `rgba(245, 158, 11, ${alpha})`;
  }
  return `rgba(167, 139, 250, ${alpha})`;
}

function normalizeClassifierFeatures(features) {
  if (!features) {
    return [];
  }
  const entries = Array.isArray(features)
    ? features.map((feature, index) => [feature?.name || feature?.id || `feature_${index + 1}`, feature])
    : typeof features === "object"
      ? Object.entries(features)
      : [];
  return entries.map(([fallbackName, feature]) => {
    if (typeof feature === "string") {
      return {
        name: feature,
        source: "-",
        type: "-",
        required: null,
        detail: "-",
      };
    }
    if (!feature || typeof feature !== "object") {
      return {
        name: fallbackName,
        source: "-",
        type: typeof feature,
        required: null,
        detail: formatContextValue(feature),
      };
    }
    const required = feature.required === undefined || feature.required === null
      ? null
      : contextBoolean(feature.required);
    return {
      name: feature.name || feature.id || fallbackName,
      source: feature.source || feature.stage || feature.origin || "-",
      type: feature.type || feature.data_type || feature.kind || "-",
      required,
      detail: formatContextValue(
        feature.normalization
        ?? feature.domain
        ?? feature.range
        ?? feature.description
        ?? "-",
      ),
    };
  });
}

function normalizeClassifierDecisionTable(decisionTable) {
  if (!decisionTable) {
    return [];
  }
  const entries = Array.isArray(decisionTable)
    ? decisionTable.map((decision, index) => [decision?.id || decision?.rule_id || `rule_${index + 1}`, decision])
    : typeof decisionTable === "object"
      ? Object.entries(decisionTable)
      : [];
  return entries.map(([fallbackId, decision]) => {
    const payload = decision && typeof decision === "object"
      ? decision
      : { conditions: decision };
    const rawClass = payload.diagnostic_class
      || payload.predicted_class
      || payload.class
      || payload.output;
    const diagnosticClass = rawClass
      ? classifierResultClass({ diagnostic_class: rawClass })
      : "";
    const confidence = contextNumber(
      payload.confidence,
      payload.confidence_threshold,
      payload.min_confidence,
      payload.threshold,
    );
    return {
      id: payload.id || payload.rule_id || fallbackId,
      conditions: formatContextValue(payload.conditions ?? payload.when ?? payload.condition ?? "-"),
      diagnosticClass,
      subtype: formatContextValue(payload.subtype ?? payload.diagnostic_subtype ?? "-"),
      confidence: confidence === null ? "-" : formatClassifierConfidence(confidence),
      action: payload.suggested_action ?? payload.action,
    };
  });
}

function ContinuousConfigEditor({ configs, busy, onChange, onSave }) {
  const fields = [
    ["physical_low", "hist min"],
    ["physical_high", "hist max"],
    ["extreme_low", "ext min"],
    ["extreme_high", "ext max"],
    ["value_floor", "piso valor"],
    ["rate_floor_per_minute", "piso tasa"],
    ["const_epsilon_floor", "epsilon"],
    ["min_gap_seconds", "gap s"],
    ["min_constant_seconds", "const s"],
  ];
  return (
    <div className="config-editor">
      <div className="config-editor-head">
        <span>Parametros continuos editables</span>
        <button type="button" className="secondary-action inline" onClick={onSave} disabled={Boolean(busy)}>
          Guardar conf.txt
        </button>
      </div>
      <div className="config-table">
        <div className="config-row config-header">
          <span>Entidad</span>
          {fields.map(([, label]) => <span key={label}>{label}</span>)}
        </div>
        {configs.map((config) => (
          <div className="config-row" key={config.entity_id}>
            <strong title={config.entity_id}>{config.entity_id}</strong>
            {fields.map(([key]) => (
              <input
                key={key}
                type="number"
                step="0.000001"
                value={config.values?.[key] ?? ""}
                onChange={(event) => onChange(config.entity_id, key, event.target.value)}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
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

function ManualControlGroup({ title, icon, entities, states, step, unit, onChange }) {
  return (
    <div className="manual-group">
      <h3>{icon}{title}</h3>
      {entities.map((entity) => {
        const value = Number(states[entity.entity_id] ?? defaultManualValue(entity));
        return (
          <div className="manual-row" key={entity.entity_id}>
            <div>
              <strong>{entity.entity_id}</strong>
              <small>{entityLabel(entity)}</small>
            </div>
            <div className="manual-stepper">
              <button type="button" onClick={() => onChange(entity.entity_id, roundManual(value - step))}>-</button>
              <input
                type="number"
                step={step}
                value={value}
                onChange={(event) => onChange(entity.entity_id, Number(event.target.value))}
              />
              <button type="button" onClick={() => onChange(entity.entity_id, roundManual(value + step))}>+</button>
              <span>{unit}</span>
            </div>
          </div>
        );
      })}
      {!entities.length && <Empty text={`No hay sensores de ${title.toLowerCase()} detectados.`} />}
    </div>
  );
}

function ManualBinaryGroup({ entities, states, onChange }) {
  return (
    <div className="manual-group">
      <h3><Power size={17} />Sensores binarios</h3>
      {entities.map((entity) => {
        const active = Boolean(states[entity.entity_id]);
        return (
          <div className="manual-row" key={entity.entity_id}>
            <div>
              <strong>{entity.entity_id}</strong>
              <small>{entityLabel(entity)}</small>
            </div>
            <button
              type="button"
              className={`manual-toggle ${active ? "active" : ""}`}
              onClick={() => onChange(entity.entity_id, !active)}
            >
              {active ? "on" : "off"}
            </button>
          </div>
        );
      })}
      {!entities.length && <Empty text="No hay sensores binarios detectados." />}
    </div>
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
          <span>
            {result.suspected_pattern || result.pattern || "-"}
            <small>{timeWindowLabel(result.time_window)}</small>
          </span>
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

function timeWindowLabel(value) {
  if (value === "month_repetition") {
    return "lapso: repeticion mensual";
  }
  if (value === "hour_of_day") {
    return "lapso: hora del dia";
  }
  if (value === "local") {
    return "lapso: local";
  }
  return "";
}

function CalculationModal({ result, onClose }) {
  const equations = equationsForResult(result);
  const rows = calculationRows(result);
  const reading = result.reading || {};
  const baseline = result.baseline_summary || {};
  const temporalContext = result.temporal_context || {};

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
          <SummaryRow label="Version score" value={result.score_version || "-"} />
          <SummaryRow
            label="Naturaleza"
            value={result.score_kind === "statistical_evidence_strength_not_probability" ? "fuerza de evidencia" : (result.score_kind || "-")}
          />
          <SummaryRow label="Consistencia temporal" value={temporalConsistencyText(temporalContext.temporal_consistency)} />
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
          <h3>Contexto por lapsos</h3>
          <KeyValueGrid rows={temporalContextRows(result)} />
        </article>

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
  const fragments = preview?.fragments || [];
  const normality = preview?.normality || {};
  const [selectedEntityId, setSelectedEntityId] = useState(fragments[0]?.entity_id || "");
  const selected = fragments.find((fragment) => fragment.entity_id === selectedEntityId) || fragments[0];

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

        <label className="normalization-selector">
          Entidad
          <select value={selected?.entity_id || ""} onChange={(event) => setSelectedEntityId(event.target.value)}>
            {fragments.map((fragment) => (
              <option value={fragment.entity_id} key={fragment.entity_id}>
                {fragment.entity_id} / {kindText(fragment.sensor_kind)}
              </option>
            ))}
          </select>
        </label>

        {selected ? (
          <NormalizationEntityPlane fragment={selected} normality={normality.entities?.[selected.entity_id]} />
        ) : (
          <Empty text="No hay fragmentos de normalizacion disponibles." />
        )}
      </section>
    </div>
  );
}

function NormalizationEntityPlane({ fragment, normality }) {
  const width = 900;
  const height = 360;
  const padding = { top: 22, right: 28, bottom: 42, left: 64 };
  const points = normalizedEntityPoints(fragment, normality);
  const yValues = points.map((point) => point.y).filter(Number.isFinite);
  const yMin = Math.min(-1, ...yValues);
  const yMax = Math.max(1, ...yValues);
  const yRange = yMax - yMin || 1;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xFor = (hour) => padding.left + (hour / 24) * plotWidth;
  const yFor = (value) => height - padding.bottom - ((value - yMin) / yRange) * plotHeight;
  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(point.hour).toFixed(1)} ${yFor(point.y).toFixed(1)}`)
    .join(" ");

  return (
    <article className={`normalization-plane entity-plane ${fragment.sensor_kind}`}>
      <div className="chart-title compact">
        <div>
          <strong>{fragment.entity_id}</strong>
          <small>{kindText(fragment.sensor_kind)} / X: hora del dia / Y: {fragment.sensor_kind === "continuous" ? "z robusto historico" : "valor normalizado"}</small>
        </div>
        <span>{points.length} lecturas</span>
      </div>
      <svg className="normalization-plane-svg" viewBox={`0 0 ${width} ${height}`}>
        {[0, 6, 12, 18, 24].map((hour) => (
          <React.Fragment key={hour}>
            <line className="grid" x1={xFor(hour)} y1={padding.top} x2={xFor(hour)} y2={height - padding.bottom} />
            <text x={xFor(hour) - 8} y={height - 14}>{hour}</text>
          </React.Fragment>
        ))}
        {[yMin, 0, yMax].map((tick) => {
          const y = yFor(tick);
          return (
            <React.Fragment key={tick}>
              <line className="grid" x1={padding.left} y1={y} x2={width - padding.right} y2={y} />
              <text x="12" y={y + 4}>{formatAxisValue(tick)}</text>
            </React.Fragment>
          );
        })}
        <line className="axis" x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
        <line className="axis" x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
        <text x={width - 94} y={height - 14}>hora del dia</text>
        <text x="12" y="18">normalizado</text>
        {linePath && <path className="normal-line" d={linePath} />}
        {points.map((point) => (
          <circle className="plane-point" key={`${point.timestamp}-${point.index}`} cx={xFor(point.hour)} cy={yFor(point.y)} r="4">
            <title>{`${fragment.entity_id}\nhora ${formatAxisValue(point.hour)}\noriginal ${point.original}\nnormalizado ${formatAxisValue(point.y)}${point.baseline ? `\nbase ${point.baseline}` : ""}`}</title>
          </circle>
        ))}
      </svg>
    </article>
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

function TimelineChart({ timeline, results, faults, onResultClick }) {
  const width = 720;
  const height = 240;
  const padding = { top: 18, right: 24, bottom: 34, left: 58 };
  const basePoints = timeline.points || [];
  const points = mergeTimelineWithSuspiciousPoints(basePoints, results, timeline);
  const isBinary = timeline.sensor_kind === "binary" || timeline.chart_kind === "active_percentage";
  const values = points.map((point) => numericPointValue(point.value)).filter(Number.isFinite);
  const min = isBinary ? 0 : (values.length ? Math.min(...values) : 0);
  const max = isBinary ? 100 : (values.length ? Math.max(...values) : 1);
  const range = max - min || 1;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const parsed = points.map((point, index) => ({
    ...point,
    value: numericPointValue(point.value),
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
  let lineStarted = false;
  const linePath = parsed
    .map((point) => {
      if (!Number.isFinite(point.value)) {
        lineStarted = false;
        return null;
      }
      const command = lineStarted ? "L" : "M";
      lineStarted = true;
      return `${command} ${xFor(point).toFixed(1)} ${yFor(point.value).toFixed(1)}`;
    })
    .filter(Boolean)
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
        result,
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
  const gapBands = (timeline.gaps || [])
    .map((gap) => {
      const start = Date.parse(gap.start);
      const end = Date.parse(gap.end);
      if (!validTimes || !Number.isFinite(start) || !Number.isFinite(end) || end < firstTime || start > lastTime) {
        return null;
      }
      const x1 = Math.max(padding.left, xForTime(Math.max(start, firstTime)));
      const x2 = Math.min(width - padding.right, xForTime(Math.min(end, lastTime)));
      return { x: x1, width: Math.max(2, x2 - x1), label: gap.label || "sin lectura valida" };
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
        {gapBands.map((band, index) => (
          <rect className="gap-band" key={`${band.label}-${index}`} x={band.x} y={padding.top} width={band.width} height={plotHeight}>
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
          Number.isFinite(point.value) && index % Math.max(1, Math.floor(parsed.length / 36)) === 0
            ? <circle key={`${point.timestamp}-${index}`} cx={xFor(point)} cy={yFor(point.value)} r="2.4" />
            : null
        ))}
        {markers.map((marker, index) => (
          <circle
            className="suspicion clickable"
            key={`${marker.label}-${index}`}
            cx={marker.x}
            cy={marker.y}
            r="5"
            role="button"
            tabIndex="0"
            onClick={() => marker.result && onResultClick?.(marker.result)}
            onKeyDown={(event) => {
              if ((event.key === "Enter" || event.key === " ") && marker.result) {
                event.preventDefault();
                onResultClick?.(marker.result);
              }
            }}
          >
            <title>{marker.label}</title>
          </circle>
        ))}
        <text x={padding.left} y={height - 8}>{points[0]?.timestamp ? shortDate(points[0].timestamp).slice(0, 16) : "-"}</text>
        <text x={width - 142} y={height - 8}>{points.at(-1)?.timestamp ? shortDate(points.at(-1).timestamp).slice(0, 16) : "-"}</text>
      </svg>
      <div className="chart-legend">
        <span>{basePoints.length} puntos</span>
        {isBinary && <span>bucket: {formatBucketSeconds(timeline.bucket_seconds)}</span>}
        {points.length > basePoints.length && <span>{points.length - basePoints.length} sospechas encontradas</span>}
        <span>{`rojo: score >= ${RED_SUSPICION_SCORE.toFixed(2)} agrupado`}</span>
        {faultBands.length > 0 && <span>zona roja suave: falla activa</span>}
        {gapBands.length > 0 && <span>zona punteada: sin lectura valida</span>}
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

function controlledEntities(entities) {
  return (entities || []).filter((entity) =>
    entity.sensor_kind === "binary"
    || isTemperatureEntity(entity)
    || isHumidityEntity(entity)
  );
}

function syncManualStates(entities, current) {
  return Object.fromEntries(
    controlledEntities(entities).map((entity) => [
      entity.entity_id,
      current[entity.entity_id] ?? defaultManualValue(entity),
    ]),
  );
}

function defaultManualValue(entity) {
  if (entity?.sensor_kind === "binary") {
    return false;
  }
  if (isHumidityEntity(entity)) {
    return 55;
  }
  return 20;
}

function isTemperatureEntity(entity) {
  const text = `${entity?.entity_id || ""} ${entity?.device_class || ""}`.toLowerCase();
  return entity?.sensor_kind === "continuous" && (text.includes("temperature") || text.includes("temp"));
}

function isHumidityEntity(entity) {
  const text = `${entity?.entity_id || ""} ${entity?.device_class || ""}`.toLowerCase();
  return entity?.sensor_kind === "continuous" && text.includes("humidity");
}

function roundManual(value) {
  return Math.round(Number(value) * 100) / 100;
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
      equation: `H_i(t) = |${formatCalc(value)} - mediana(W)| / (1.4826 * MAD(W) + epsilon) = ${formatCalc(raw.H_i)}; Z_F_i = ${formatCalc(raw.Z_F_i)}`,
      score: `s_H = max(phi_z(${formatCalc(raw.H_i)}), phi_z(${formatCalc(raw.Z_F_i)})) = ${formatCalc(components.s_H)}`,
    };
  }
  if (equation.id === "robust_rate_change") {
    return {
      equation: `R_i(t) = |${formatCalc(delta)}| / ${formatCalc(deltaMinutes)}min = ${formatCalc(rate)}/min; Z_R_i(t) = ${formatCalc(raw.Z_R_i)}`,
      score: `s_R = phi_z(${formatCalc(raw.Z_R_i)}) = ${formatCalc(components.s_R)}`,
    };
  }
  if (equation.id === "cusum_residual") {
    return {
      equation: `r_i(t) = ${formatCalc(value)} - ${formatCalc(prediction)} = ${formatCalc(residual)}; max(C+, C-) = ${formatCalc(raw.C_residual)}`,
      score: `s_C = phi_ratio(${formatCalc(raw.C_residual)} / ${formatCalc(base.cusum_theta)}) = ${formatCalc(components.s_C)}`,
    };
  }
  if (equation.id === "constant_persistence") {
    return {
      equation: `D_const = ${formatCalc(reading.constant_duration_seconds)}s con epsilon_const = ${formatCalc(base.epsilon_const)}; Z_D_const = ${formatCalc(raw.Z_D_const)}`,
      score: `s_D_const = max(phi_z(${formatCalc(raw.Z_D_const)}), phi_ratio(${formatCalc(reading.constant_duration_seconds)} / ${formatCalc(base.constant_alert_seconds)})) = ${formatCalc(components.s_D_const)}`,
    };
  }
  if (equation.id === "reading_gap") {
    return {
      equation: `M_i(t) = t - t_i_last = ${formatCalc(gapSeconds)}s; Z_M = ${formatCalc(raw.Z_M)}`,
      score: `s_M = max(phi_z(${formatCalc(raw.Z_M)}), phi_ratio(${formatCalc(gapSeconds)} / ${formatCalc(base.gap_alert_seconds)})) = ${formatCalc(components.s_M)}`,
    };
  }
  if (equation.id === "dspot_extreme") {
    return {
      equation: `Z_DSPOT = distancia de ${formatCalc(value)} contra [${formatCalc(base.dspot_lower)}, ${formatCalc(base.dspot_upper)}] = ${formatCalc(raw.Z_DSPOT)}`,
      score: `s_DSPOT = phi_z(${formatCalc(raw.Z_DSPOT)}) = ${formatCalc(components.s_DSPOT)}`,
    };
  }
  return { equation: equation.equation, score: equation.score };
}

function numericPointValue(value) {
  if (value === null || value === undefined || value === "") {
    return NaN;
  }
  return Number(value);
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
  if (equation.id === "time_since_last_change") {
    return {
      equation: `TLC_i(t) = t - ultimo cambio = ${formatCalc(reading.time_since_last_change_seconds)}s; Z_TLC = ${formatCalc(raw.Z_TLC_i)}`,
      score: `s_TLC = min(1, ${formatCalc(raw.Z_TLC_i)} / 3.5) = ${formatCalc(components.s_TLC)}`,
    };
  }
  if (equation.id === "z_duration_score") {
    return {
      equation: `Z_D_i = Z_D_on = ${formatCalc(raw.Z_D_i)}`,
      score: `s_D = min(1, ${formatCalc(raw.Z_D_i)} / 3.5) = ${formatCalc(components.s_D)}`,
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
      calculationRow(result, "time_since_last_change", "Tiempo TLC_i(t)", "Z_TLC_i", "s_TLC"),
      calculationRow(result, "z_duration_score", "Z-score duracion Z_D_i", "Z_D_i", "s_D"),
    ];
  }
  return [
    calculationRow(result, "hampel", "Filtro Hampel H_i(t)", "H_i", "s_H"),
    calculationRow(result, "robust_rate_change", "Tasa robusta Z_R_i(t)", "Z_R_i", "s_R"),
    calculationRow(result, "cusum_residual", "CUSUM residual", "C_residual", "s_C"),
    calculationRow(result, "constant_persistence", "Persistencia D_const", "Z_D_const", "s_D_const"),
    calculationRow(result, "reading_gap", "Brecha M_i(t)", "Z_M", "s_M"),
    calculationRow(result, "dspot_extreme", "DSPOT extremo", "Z_DSPOT", "s_DSPOT"),
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
    return `S_bin(W) usa doble horizonte: base larga y base reciente. El score mostrado es el menor entre ambas para evitar falsos positivos por cambios de rutina = ${formatNumber(result.evidence_score)}. Luego P_sos = S_bin / (S_bin + tau_i) = ${formatNumber(result.suspicion_score)}.`;
  }
  return `Version ${result.score_version || "v2"}: phi(u)=ln(1+u)/(ln(1+u)+lambda), con phi(1)=0.60. Evidencia continua = max(s_H, s_R, s_C, s_D_const, s_M, s_DSPOT) = ${formatNumber(result.evidence_score)}. El score de sospecha conserva ese mismo valor; expresa fuerza de evidencia, no una probabilidad = ${formatNumber(result.suspicion_score)}.`;
}

function readingRows(result) {
  const reading = result?.reading || {};
  if (result?.sensor_kind === "binary") {
    return [
      { label: "Estado observado", value: result.observed_state ?? "-" },
      { label: "A_i(W)", value: reading.activations ?? "-" },
      { label: "C_i(W)", value: reading.state_changes ?? "-" },
      { label: "T_on,i(W)", value: formatSeconds(reading.active_seconds) },
      { label: "D_off,i(W)", value: formatSeconds(reading.inactive_seconds) },
      { label: "P_on,i(W)", value: formatPercent(reading.active_ratio) },
      { label: "G_i(t)", value: formatSeconds(reading.gap_seconds) },
      { label: "TLC_i(t)", value: formatSeconds(reading.time_since_last_change_seconds) },
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
    {
      label: "Fuente s_H",
      value: reading.component_sources?.s_H === "operational_limit" ? "limite operacional" : "Hampel",
    },
    { label: "Score limite operacional", value: formatNumber(reading.operational_limit_score) },
  ];
}

function objectRows(object) {
  return Object.entries(object || {}).map(([key, value]) => ({
    label: key,
    value: typeof value === "number" ? formatNumber(value) : String(value ?? "-"),
  }));
}

function temporalContextRows(result) {
  const context = result.temporal_context || {};
  return [
    { label: "Lapso observado", value: timeWindowLabel(result.time_window) || "local" },
    { label: "Consistencia temporal", value: temporalConsistencyText(context.temporal_consistency) },
    { label: "Score local", value: formatNumber(context.local_score) },
    { label: "Score hora del dia", value: formatNumber(context.hour_of_day_score) },
    { label: "Patron repetido", value: context.repeated_pattern ? "si" : "no" },
    { label: "Dias similares", value: context.similar_days_in_month ?? "-" },
  ];
}

function temporalConsistencyText(value) {
  if (value === "alta") {
    return "alta: patron repetido";
  }
  if (value === "media") {
    return "media: confirmado por franja horaria";
  }
  if (value === "baja") {
    return "baja: sospecha principalmente local";
  }
  return "-";
}

function normalizedEntityPoints(fragment, normality) {
  const points = (fragment?.points || []).map((point, index) => ({
    ...point,
    index,
    hour: hourOfDay(point.timestamp),
    numeric: Number(point.value),
  }));
  if (fragment?.sensor_kind === "continuous") {
    const values = points.map((point) => point.numeric).filter(Number.isFinite);
    const median = Number(normality?.median);
    const scale = Number(normality?.scaled_mad);
    const fallbackMean = average(values);
    const fallbackScale = standardDeviation(values, fallbackMean) || 1;
    const center = Number.isFinite(median) ? median : fallbackMean;
    const robustScale = Number.isFinite(scale) && scale > 0 ? scale : fallbackScale;
    return points
      .filter((point) => Number.isFinite(point.numeric))
      .map((point) => ({
        ...point,
        y: (point.numeric - center) / robustScale,
        original: point.state,
        baseline: `mediana=${formatNumber(center)} escala=${formatNumber(robustScale)}`,
      }))
      .sort((left, right) => left.hour - right.hour);
  }
  if (fragment?.sensor_kind === "binary") {
    return points.map((point) => ({
      ...point,
      y: Number(point.numeric) >= 50 ? 1 : 0,
      original: point.state,
    })).sort((left, right) => left.hour - right.hour);
  }
  const states = [...new Set(points.map((point) => String(point.state || "")).filter(Boolean))];
  const indexByState = new Map(states.map((state, index) => [state, index]));
  return points.map((point) => ({
    ...point,
    y: indexByState.get(String(point.state || "")) ?? 0,
    original: point.state,
  })).sort((left, right) => left.hour - right.hour);
}

function hourOfDay(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) {
    return 0;
  }
  return date.getHours() + date.getMinutes() / 60;
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function standardDeviation(values, mean = average(values)) {
  if (values.length < 2) {
    return 0;
  }
  return Math.sqrt(average(values.map((value) => (value - mean) ** 2)));
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
      `TLC=${formatSeconds(reading.time_since_last_change_seconds)}`,
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

function normalityDetail(normality) {
  if (!normality?.exists) {
    return "sin generar";
  }
  return normality.mode === "persisted" ? "persistida en cache" : "en memoria";
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
