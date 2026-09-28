import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
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
import { downloadCsv } from "./csv.js";
import ReviewWorkspace from "./ReviewWorkspace.jsx";
import DiagnosticWorkspace from "./DiagnosticWorkspace.jsx";
import FlowLab from "./FlowLab.jsx";
import { isStatisticalSuspicion, statisticalCounts, episodeLabel } from "./statistical-routing.js";
import { isEnvironmentalResult, contextRoutingLabel, contextRoutingReasonLabel, contextRoutingCounts, contextCoherenceLabel, contextObservationLabel, preciseContextTimestamp, contextCommonCause } from "./context-routing.js";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const DEFAULT_SUSPICION_THRESHOLD = 0.60;
const MIN_SUSPICION_THRESHOLD = 0.01;
const MAX_SUSPICION_THRESHOLD = 0.99;
const MARKER_GROUP_GAP_MS = 15 * 60 * 1000;

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

const MULTISTATE_EQUATIONS = [
  {
    id: "state_change_count",
    title: "Cambios multivalor",
    pattern: "transition_burst",
    equation: "C_i(W) = sum 1[s_i(t_j-1) != s_i(t_j)]",
    score: "s_C = min(1, Z_C / tau_C)",
  },
  {
    id: "state_probability",
    title: "Rareza del estado",
    pattern: "rare_state",
    equation: "P(s) = frecuencia historica del estado observado",
    score: "s_estado = max(0, (p_min - P(s)) / p_min)",
  },
  {
    id: "transition_probability",
    title: "Rareza de la transicion",
    pattern: "rare_transition",
    equation: "P(s_j|s_i) = N(s_i -> s_j) / N(s_i -> *)",
    score: "s_trans = max(0, (p_min - P(s_j|s_i)) / p_min)",
  },
  {
    id: "state_duration",
    title: "Persistencia del estado",
    pattern: "state_persistence",
    equation: "D_s(t) = t - instante del ultimo cambio de estado",
    score: "s_D = min(1, Z_D / tau_D)",
  },
  {
    id: "report_gap",
    title: "Brecha multivalor",
    pattern: "missing_gap",
    equation: "M_i(t) = t - t_i_last",
    score: "s_M = min(1, Z_M / tau_M)",
  },
];

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

function replayResultKey(result) {
  return result?.event_id || `${result?.entity_id || ""}@${result?.timestamp || ""}`;
}

function mergeReplayResults(current, incoming, limit = 500) {
  const merged = new Map();
  [...(incoming || []), ...(current || [])].forEach((item) => {
    merged.set(replayResultKey(item), item);
  });
  return [...merged.values()].sort(compareResultsByExecutionDate).slice(0, limit);
}

function startHistoryUpload(file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE_URL}/history/real/upload/start`);
    request.setRequestHeader("Content-Type", "text/csv");
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded, event.total);
      }
    };
    request.onerror = () => reject(new Error("No se pudo conectar con la API para subir el historico."));
    request.onload = () => {
      const payload = (() => {
        try {
          return JSON.parse(request.responseText || "{}");
        } catch {
          return {};
        }
      })();
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(payload.detail || `HTTP ${request.status}`));
        return;
      }
      resolve(payload);
    };
    request.send(file);
  });
}

function startValidationUpload(file, onProgress, onProcessing) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE_URL}/validation/replay/upload`);
    request.setRequestHeader("Content-Type", "text/csv");
    request.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded, event.total);
      }
    };
    request.upload.onload = () => onProcessing?.();
    request.onerror = () => reject(new Error("No se pudo conectar con la API para subir Evaluacion."));
    request.onload = () => {
      const payload = (() => {
        try {
          return JSON.parse(request.responseText || "{}");
        } catch {
          return {};
        }
      })();
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(payload.detail || `HTTP ${request.status}`));
        return;
      }
      resolve(payload);
    };
    request.send(file);
  });
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function downloadResultsCsv(results) {
  downloadCsv([
    ["Entidad", "Tiempo mayor-menor", "Score", "Raw", "Lectura", "Patron", "Indicadores", "Ecuacion", "Episodio", "Agrupada", "Apariciones"],
    ...results.map((result) => [
      result.entity_id,
      shortDate(result.timestamp),
      formatNumber(result.suspicion_score),
      formatNumber(result.raw_score),
      formatReading(result),
      [result.suspected_pattern || result.pattern || "-", timeWindowLabel(result.time_window)].filter(Boolean).join(" / "),
      (result.triggered_indicators || []).join(", ") || "-",
      equationNames(result.triggered_indicators, result.sensor_kind),
      result.episode?.episode_id || "",
      result.suppressed_by_episode ? "si" : "no",
      result.episode?.occurrences ?? "",
    ]),
  ], "resultados_filtro_estadistico.csv");
}

function App() {
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

function EpisodeGroupingControl({ minutes, onChange, busy, summary, replay }) {
  const started = Number(replay?.processed_rows || 0) > 0;
  const activeMinutes = Number(replay?.suspicion_quiet_seconds || 0) / 60;
  return (
    <section className="episode-grouping-panel" aria-labelledby="episode-grouping-title">
      <div>
        <span className="eyebrow">Salida del filtro estadistico</span>
        <h2 id="episode-grouping-title">Agrupar sospechas repetidas</h2>
      </div>
      <label>
        Tiempo sin repeticion para abrir otro episodio
        <select value={minutes} onChange={(event) => onChange(Number(event.target.value))} disabled={busy}>
          <option value="0">Desactivado · enviar todas</option>
          <option value="5">5 minutos</option>
          <option value="15">15 minutos</option>
          <option value="30">30 minutos</option>
          <option value="60">1 hora</option>
          <option value="120">2 horas</option>
          <option value="1440">24 horas</option>
        </select>
      </label>
      <div className="summary-strip">
        <SummaryRow label="Sospechas detectadas" value={formatCount(summary.detected)} />
        <SummaryRow label="Enviadas a F2" value={formatCount(summary.forwarded)} />
        <SummaryRow label="Repeticiones agrupadas" value={formatCount(summary.suppressed)} />
      </div>
      {started && activeMinutes !== minutes && (
        <p className="episode-grouping-note">
          Esta ejecucion conserva {activeMinutes ? `${activeMinutes} minutos` : "la agrupacion desactivada"}.
        </p>
      )}
    </section>
  );
}

function SuspicionThresholdControl({ value, analyzedValue, busy, onChange }) {
  const percent = Math.round(value * 1000) / 10;
  const pendingAnalysis = analyzedValue !== null && analyzedValue !== value;

  return (
    <section className="suspicion-threshold-panel" aria-labelledby="suspicion-threshold-title">
      <div className="suspicion-threshold-copy">
        <span className="eyebrow">Filtro 1 · estadistico-temporal</span>
        <h2 id="suspicion-threshold-title">Umbral de sospecha</h2>
      </div>
      <div className="suspicion-threshold-inputs">
        <output aria-live="polite">{formatSuspicionPercent(value)}</output>
        <label className="suspicion-range-label">
          <span className="sr-only">Ajustar umbral de sospecha</span>
          <input
            type="range"
            min="1"
            max="99"
            step="0.1"
            value={percent}
            onChange={(event) => onChange(event.target.value)}
            disabled={busy}
          />
          <small><span>1 %</span><span>99 %</span></small>
        </label>
        <label className="suspicion-number-label">
          <span>Valor (%)</span>
          <input
            type="number"
            min="1"
            max="99"
            step="0.1"
            inputMode="decimal"
            value={percent}
            onChange={(event) => onChange(event.target.value)}
            disabled={busy}
          />
        </label>
        <button
          type="button"
          className="secondary-action inline"
          onClick={() => onChange(DEFAULT_SUSPICION_THRESHOLD * 100)}
          disabled={busy || value === DEFAULT_SUSPICION_THRESHOLD}
        >
          Restaurar 60 %
        </button>
      </div>
      {pendingAnalysis && (
        <p className="suspicion-threshold-note">
          Resultados mostrados calculados con {formatSuspicionPercent(analyzedValue)}. Pulsa “Analizar historico” para aplicar {formatSuspicionPercent(value)}.
        </p>
      )}
    </section>
  );
}

function ContextualWorkspace({ mapData, results, summary, busy, error, onReload, onRebuild }) {
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
  const outcomeCounts = contextOutcomeCounts(results, summary);
  const routingCounts = contextRoutingCounts(results, summary);
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
              ? "Sin mapa generado."
              : stale
                ? (staleReasons.length ? staleReasons.join(", ") : "Mapa desactualizado.")
                : `Version ${mapData?.map_version || "-"} (${contextBuiltAtText(mapData)})`}
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
          detail=""
        />
        <Metric
          icon={<Eye />}
          label="Vigencia"
          value={!hasMap ? "Sin mapa" : stale ? "Obsoleto" : "Vigente"}
          detail={staleReasons.join(", ") || contextRefreshTrigger(mapData)}
        />
      </section>

      <Panel title="Salida del filtro contextual">
        <div className="context-routing-summary" aria-label="Decisiones de la compuerta contextual">
          <div className="context-outcome-card respaldado">
            <span>Anomalias ambientales</span>
            <strong>{formatCount(routingCounts.environmental)}</strong>
            <small>Filtro 2</small>
          </div>
          <div className="context-outcome-card no_explicado">
            <span>Pendientes de clasificacion</span>
            <strong>{formatCount(routingCounts.pending)}</strong>
            <small>Filtro 3</small>
          </div>
        </div>
      </Panel>

      <section className="context-main-grid">
        <Panel title="Mapa de asociaciones">
          <ContextMatrix mapData={mapData} onSelectEdge={setSelectedEdge} />
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
        </Panel>
      </section>

      <Panel title="Resultados del filtro contextual">
        <div className="context-results-summary" aria-label="Resumen de resultados contextuales">
          {["respaldado", "contradicho", "no_explicado", "cobertura_insuficiente", "sin_relaciones"].map((outcome) => (
            <div className={`context-outcome-card ${outcome}`} key={outcome}>
              <span>{contextOutcomeLabel(outcome)}</span>
              <strong>{formatCount(outcomeCounts[outcome])}</strong>
            </div>
          ))}
          <div className="context-outcome-card common-cause">
            <span>Coincidencia contextual</span>
            <strong>{formatCount(summary?.contextual_coincidences ?? results.filter((result) => contextCommonCause(result).coincidence).length)}</strong>
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

function ContextMatrix({ mapData, onSelectEdge }) {
  const matrices = mapData?.matrices || {};
  const nodes = contextMapNodes(mapData);
  const edges = contextMapEdges(mapData);
  const entities = Array.isArray(matrices.entities) && matrices.entities.length
    ? matrices.entities.map((entity) => typeof entity === "string" ? entity : entity?.entity_id).filter(Boolean)
    : nodes.map((node) => node.entity_id || node.id).filter(Boolean);
  const values = matrices.association;
  const accepted = matrices.accepted;
  const scale = 1;
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
                    ? `${rowEntity} ↔ ${columnEntity}: ${formatContextMatrixValue(numeric)} · ${isAccepted ? "aceptada" : "no aceptada"}`
                    : `${rowEntity} ↔ ${columnEntity}: no aplicable`;
                  const detail = edge || {
                    source_entity_id: rowEntity,
                    target_entity_id: columnEntity,
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
                        {available ? formatContextMatrixValue(numeric) : "N/D"}
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
        <span>{formatContextMatrixValue(-scale)}</span>
        <div aria-hidden="true" />
        <span>0</span>
        <div aria-hidden="true" />
        <span>{formatContextMatrixValue(scale)}</span>
      </div>
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
            <span role="cell">{contextCoherenceLabel(result)}</span>
            <span role="cell">
              <ContextStatusBadge value={outcome} />
              <small>{contextRoutingLabel(result)}</small>
            </span>
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
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Cerrar detalle">
            <X size={18} />
          </button>
        </header>
        <div className="modal-score-grid">
          <SummaryRow label="Aceptada" value={accepted ? "si" : "no"} />
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
              { label: "Ventana", value: formatSeconds(edge.window_seconds) },
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
        {edge.window_candidates?.length > 0 && (
          <article className="modal-block">
            <h3>Ventanas evaluadas</h3>
            {edge.window_candidates.map((candidate) => (
              <div key={candidate.window_seconds}>
                <strong>{formatSeconds(candidate.window_seconds)}{candidate.window_seconds === edge.window_seconds ? " · seleccionada" : ""}</strong>
                <KeyValueGrid rows={[
                  { label: "Observaciones", value: formatCount(candidate.n_observations) },
                  { label: "Cobertura", value: formatContextPercent(candidate.coverage_ratio) },
                  { label: "Asociacion", value: formatContextValue(candidate.association_signed) },
                  { label: "Desfase", value: formatSeconds(candidate.lag_seconds) },
                  { label: "Controles", value: candidate.threshold_eligible ? "cumple" : candidate.rejection_reasons.join(", ") },
                ]} />
              </div>
            ))}
          </article>
        )}
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
            <small>{preciseContextTimestamp(result.timestamp)} · mapa {mapQuality.version || result.map_version || "-"}</small>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Cerrar detalle">
            <X size={18} />
          </button>
        </header>

        <article className="modal-block">
          <h3>{contextRoutingLabel(result)}</h3>
          <p>{contextRoutingReasonLabel(result)}</p>
        </article>

        <div className="modal-score-grid">
          <SummaryRow label="Score estadistico" value={formatNumber(contextStatScore(result))} />
          <SummaryRow label="Coherencia contextual" value={contextCoherenceLabel(result)} />
          <SummaryRow label="Resultado" value={contextOutcomeLabel(outcome)} />
          <SummaryRow label="Efecto en sospecha" value={result.suspicion_effect || result.effect_on_suspicion || "-"} />
          <SummaryRow label="Cobertura" value={coverage.ratio === null ? "-" : formatContextPercent(coverage.ratio)} />
          <SummaryRow label="Vigencia del mapa" value={stale ? "obsoleto" : "vigente"} />
        </div>

        {result.decision_method && (
          <article className="modal-block">
            <h3>Decisión contextual calibrada</h3>
            <KeyValueGrid rows={[
              { label: "Método", value: result.decision_method },
              { label: "Familias disponibles", value: (result.available_families || []).join(", ") || "ninguna" },
              { label: "Puntuación determinante", value: formatContextValue(result.decision_score) },
              { label: "Umbral calibrado", value: formatContextValue(result.decision_threshold) },
              { label: "Calibración", value: result.decision_calibration?.status || "no disponible" },
              { label: "Exclusiones", value: formatCount((result.decision_exclusions || []).length) },
            ]} />
            <KeyValueGrid rows={Object.entries(result.family_scores || {}).map(([family, score]) => ({
              label: `${family}: puntuación / cobertura`,
              value: `${formatContextValue(score?.score)} / ${formatContextPercent(score?.coverage)}`,
            }))} />
          </article>
        )}

        <div className="modal-columns">
          <article className="modal-block">
            <h3>Cobertura antes de decidir</h3>
            <KeyValueGrid rows={[
              { label: "Relaciones elegibles", value: formatCount(coverage.eligible) },
              { label: "Relaciones observadas", value: formatCount(coverage.observed) },
              { label: "Ultimos valores reutilizados", value: formatCount(result.coverage?.carried_neighbors ?? 0) },
              { label: "Proporcion", value: coverage.ratio === null ? "-" : formatContextPercent(coverage.ratio) },
              { label: "Estado", value: coverage.status || (outcome === "cobertura_insuficiente" ? "insuficiente" : "suficiente") },
              { label: "Sin evidencia", value: formatCount(unknowns.length) },
            ]} />
          </article>
          <article className="modal-block">
            <h3>Causa comun</h3>
            <ContextStatusBadge value={commonCause.key} label={commonCause.label} />
            <KeyValueGrid rows={[
              { label: "Identificable", value: commonCause.identifiable ? "si" : "no" },
              { label: "Coincidencia contextual", value: commonCause.coincidence ? "Detectada" : "No detectada" },
              { label: "Ventana pasada", value: formatSeconds(commonCause.windowSeconds) },
              { label: "Entidades relacionadas observadas", value: commonCause.entities.join(", ") || "-" },
              { label: "Evidencia independiente", value: commonCause.independentEvidence ? "disponible" : "no disponible" },
              { label: "Dominio identificado", value: commonCause.domain || "Ninguno" },
              { label: "Detalle", value: commonCause.detail || "-" },
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
                    <span>
                      {contextEvidenceEntity(item)}
                      {item.observation_method && (
                        <small className="context-observation-detail">
                          {contextObservationLabel(item)}
                          {item.observed_timestamp && <><br />Lectura: {preciseContextTimestamp(item.observed_timestamp)}</>}
                          <br />Referencia: {preciseContextTimestamp(item.reference_timestamp)}
                          <br />Antiguedad: {formatSeconds(item.observation_age_seconds)} · limite: {formatSeconds(item.max_age_seconds)}
                          <br />Ventana de relacion: {formatSeconds(item.window_seconds)}
                          {item.observed_value != null && <><br />Valor utilizado: {formatContextValue(item.observed_value)}</>}
                          {item.magnitude_check?.expected_value != null && <>
                            <br />Rango esperado: {formatContextValue(item.magnitude_check.lower)} a {formatContextValue(item.magnitude_check.upper)}
                            <br />Magnitud: {item.magnitude_check.status === "compatible" ? "compatible" : "incompatible"}
                            {item.magnitude_check.extrapolated && <> · extrapolacion fuera del rango de ajuste</>}
                          </>}
                        </small>
                      )}
                    </span>
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
    [["lag_candidates_seconds", "lag_candidates"], "Desfases en ventana base"],
    [["lag_candidates_by_window"], "Desfases por ventana (segundos)"],
    [["window_candidates_seconds"], "Ventanas candidatas (segundos)"],
    [["common_cause_window_seconds"], "Ventana de coincidencia"],
    [["max_lag_seconds"], "Desfase maximo"],
    [["association_threshold", "min_association", "association_min"], "Asociacion minima"],
    [["min_observations", "minimum_observations"], "Observaciones minimas"],
    [["min_sign_consistency", "stability_threshold", "min_stability"], "Consistencia de signo minima"],
    [["stability_slices", "stability_windows", "temporal_splits"], "Subventanas de estabilidad"],
    [["min_stable_slices"], "Subventanas estables minimas"],
    [["min_pair_coverage", "coverage_threshold", "min_coverage"], "Cobertura historica minima"],
    [["runtime_coverage_threshold"], "Cobertura minima en ejecucion"],
    [["continuous_max_age_seconds"], "Antiguedad maxima continua"],
    [["continuous_max_age_seconds_by_entity"], "Antiguedad por sensor (segundos)"],
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

function formatContextMatrixValue(value) {
  const number = contextFiniteNumber(value);
  if (number === null) {
    return "N/D";
  }
  return number.toFixed(2);
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
  if (["no_learned_relations", "sin_relaciones"].includes(key)) return "sin_relaciones";
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
    sin_relaciones: "Sin relaciones aprendidas",
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

function contextOutcomeCounts(results = [], summary = {}) {
  const counts = {
    respaldado: 0,
    contradicho: 0,
    no_explicado: 0,
    cobertura_insuficiente: 0,
    sin_relaciones: 0,
  };
  results.forEach((result) => {
    counts[contextOutcome(result)] += 1;
  });
  if (summary?.by_outcome) {
    Object.keys(counts).forEach((key) => { counts[key] = 0; });
    Object.entries(summary.by_outcome).forEach(([outcome, count]) => {
      counts[contextOutcome({ context_outcome: outcome })] += Number(count) || 0;
    });
  }
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


function contextResultMapIsStale(result, fallback) {
  const value = result?.map_quality?.stale
    ?? result?.map?.stale
    ?? result?.map_stale;
  return value === undefined || value === null ? Boolean(fallback) : contextBoolean(value);
}

function contextStatScore(result = {}) {
  return contextNumber(result.stat_score, result.statistical_score, result.suspicion_score);
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

function HistoryLoadProgress({ job }) {
  const progress = Math.min(100, Math.max(0, Number(job?.progress || 0)));
  const ready = job?.status === "completed";
  const failed = job?.status === "failed";
  return (
    <section
      className={`history-load-progress ${ready ? "ready" : ""} ${failed ? "failed" : ""}`}
      aria-live="polite"
    >
      <div className="history-load-heading">
        <span>{ready ? "Historico listo" : failed ? "Carga fallida" : historyLoadPhaseLabel(job?.phase)}</span>
        <strong>{Math.round(progress)}%</strong>
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-label="Progreso de carga del historico"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={Math.round(progress)}
      >
        <span style={{ width: `${progress}%` }} />
      </div>
      <p>{job?.error || job?.message || "Preparando carga"}</p>
      <div className="history-load-meta">
        <span>Transcurrido: {formatDuration(job?.elapsed_seconds)}</span>
        <span>Restante: {ready ? "0 s" : formatDuration(job?.eta_seconds, "estimando")}</span>
        <span>Archivo: {formatBytes(job?.file_size_bytes)}</span>
      </div>
    </section>
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
            {result.episode && <small>{episodeLabel(result)}</small>}
            <small>
              {timeWindowLabel(result.time_window)}
              {result.replay ? ` / retraso ${formatDuration(result.replay.detection_delay_seconds)}` : ""}
            </small>
          </span>
          <span>{(result.triggered_indicators || []).join(", ") || "-"}</span>
          <span>{equationNames(result.triggered_indicators, result.sensor_kind)}</span>
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
        {result.episode && <article className="modal-block">
          <h3>Episodio de sospecha</h3>
          <KeyValueGrid rows={[
            { label: "Decision de envio", value: episodeLabel(result) },
            { label: "Identificador", value: result.episode.episode_id },
            { label: "Primera sospecha", value: preciseContextTimestamp(result.episode.first_seen) },
            { label: "Ultima aparicion registrada", value: preciseContextTimestamp(result.episode.last_seen) },
            { label: "Apariciones hasta este evento", value: formatCount(result.episode.occurrences) },
            { label: "Score maximo hasta este evento", value: formatNumber(result.episode.max_score) },
            { label: "Lapso para abrir otro", value: formatSeconds(result.episode.quiet_seconds) },
          ]} />
        </article>}

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

function TimelineChart({ timeline, results, suspicionThreshold, onResultClick }) {
  const width = 720;
  const height = 240;
  const padding = { top: 18, right: 24, bottom: 34, left: 58 };
  const basePoints = timeline.points || [];
  const points = mergeTimelineWithSuspiciousPoints(
    basePoints, results, timeline, suspicionThreshold,
  );
  const isBinary = timeline.sensor_kind === "binary" || timeline.chart_kind === "active_percentage";
  const isMultistate = timeline.sensor_kind === "multistate" || timeline.chart_kind === "categorical_state";
  const categoryEntries = Object.entries(timeline.categories || {})
    .sort((left, right) => Number(left[1]) - Number(right[1]));
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
  const axisTicks = isMultistate && categoryEntries.length > 0 && categoryEntries.length <= 6
    ? categoryEntries.map(([label, value]) => ({
      key: `state-${label}`,
      y: yFor(Number(value)),
      label,
    }))
    : [0, 0.5, 1].map((tick) => ({
      key: `tick-${tick}`,
      y: padding.top + tick * plotHeight,
      label: isBinary ? `${Math.round(max - tick * range)}%` : formatAxisValue(max - tick * range),
    }));

  return (
    <article className="chart-card">
      <div className="chart-title">
        <div>
          <strong>{timeline.entity_id}</strong>
          <small>{kindText(timeline.sensor_kind)} / {timeline.area_id || "sin area"}</small>
        </div>
        <span>
          {isBinary
            ? "activacion 0-100%"
            : isMultistate
              ? (categoryEntries.length ? categoryEntries.map(([state]) => state).join(" / ") : "alta cardinalidad")
              : `${formatAxisValue(min)} - ${formatAxisValue(max)}`}
        </span>
      </div>
      <svg className="timeline-svg" viewBox={`0 0 ${width} ${height}`}>
        {axisTicks.map((tick) => (
          <React.Fragment key={tick.key}>
            <line className="grid" x1={padding.left} y1={tick.y} x2={width - padding.right} y2={tick.y} />
            <text x="8" y={tick.y + 4}>{tick.label}</text>
          </React.Fragment>
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
        {isMultistate && <span>{multistateStatusText(timeline.analysis_status)}</span>}
        {points.length > basePoints.length && <span>{points.length - basePoints.length} sospechas encontradas</span>}
        <span>{`rojo: score >= ${suspicionThreshold.toFixed(2)} agrupado`}</span>
        {gapBands.length > 0 && <span>zona punteada: sin lectura valida</span>}
      </div>
    </article>
  );
}

function isRedSuspicion(result, suspicionThreshold) {
  return isStatisticalSuspicion(result);
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

function mergeTimelineWithSuspiciousPoints(points, results, timeline, suspicionThreshold) {
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
    if (result.entity_id !== entityId || !isRedSuspicion(result, suspicionThreshold)) {
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
  if (timeline?.sensor_kind === "multistate" || timeline?.chart_kind === "categorical_state") {
    const categories = timeline?.categories || {};
    const state = String(reading.state ?? result.observed_state ?? "");
    if (Object.prototype.hasOwnProperty.call(categories, state)) {
      return Number(categories[state]);
    }
    const matched = Object.entries(categories).find(
      ([candidate]) => candidate.toLocaleLowerCase() === state.toLocaleLowerCase(),
    );
    return matched ? Number(matched[1]) : NaN;
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
  if (kind === "multistate") {
    return "estado multivalor";
  }
  return kind || "entidad";
}

function multistateStatusText(status) {
  if (status === "eligible") {
    return "multistate analizable";
  }
  if (status === "high_cardinality") {
    return "no analizable: alta cardinalidad";
  }
  if (status === "insufficient_states") {
    return "no analizable: falta variedad de estados";
  }
  if (status === "insufficient_history") {
    return "no analizable: historial insuficiente";
  }
  return "multistate";
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

function equationNames(indicators = [], sensorKind = "") {
  const catalogue = sensorKind === "binary"
    ? STATE_EQUATIONS
    : sensorKind === "multistate"
      ? MULTISTATE_EQUATIONS
      : EQUATIONS;
  const names = indicators
    .map((indicator) => catalogue.find((equation) => equation.id === indicator)?.title)
    .filter(Boolean);
  return names.join(", ") || "-";
}

function equationsForResult(result) {
  const ids = result?.triggered_indicators || [];
  const catalogue = result?.sensor_kind === "binary"
    ? STATE_EQUATIONS
    : result?.sensor_kind === "multistate"
      ? MULTISTATE_EQUATIONS
      : EQUATIONS;
  const found = ids
    .map((indicator) => catalogue.find((equation) => equation.id === indicator))
    .filter(Boolean);
  return found.length ? found : catalogue.filter((equation) => equation.pattern === (result?.suspected_pattern || result?.pattern));
}

function equationWithCalculationValues(equation, result) {
  if (result?.sensor_kind === "binary") {
    return binaryEquationWithValues(equation, result);
  }
  if (result?.sensor_kind === "multistate") {
    return multistateEquationWithValues(equation, result);
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

function multistateEquationWithValues(equation, result) {
  const reading = result?.reading || {};
  const raw = reading.raw_components || {};
  const components = reading.components || {};
  if (equation.id === "state_change_count") {
    return {
      equation: `C_i(W) = ${reading.state_changes ?? "-"} cambios en ${formatSeconds(reading.window_seconds)}; Z_C = ${formatCalc(raw.state_change_count)}`,
      score: `s_C = ${formatCalc(components.state_change_count)}`,
    };
  }
  if (equation.id === "state_probability") {
    return {
      equation: `P(${reading.state ?? result.observed_state ?? "s"}) = ${formatCalc(reading.state_probability)}; sorpresa = ${formatCalc(raw.state_probability)}`,
      score: `s_estado = ${formatCalc(components.state_probability)}`,
    };
  }
  if (equation.id === "transition_probability") {
    return {
      equation: `P(${reading.state ?? "s_j"}|${reading.previous_state ?? "s_i"}) = ${formatCalc(reading.transition_probability)}; sorpresa = ${formatCalc(raw.transition_probability)}`,
      score: `s_trans = ${formatCalc(components.transition_probability)}`,
    };
  }
  if (equation.id === "state_duration") {
    return {
      equation: `D_s(t) = ${formatSeconds(reading.state_duration_seconds)}; Z_D = ${formatCalc(raw.state_duration)}`,
      score: `s_D = ${formatCalc(components.state_duration)}`,
    };
  }
  if (equation.id === "report_gap") {
    return {
      equation: `M_i(t) = ${formatSeconds(reading.report_gap_seconds)}; Z_M = ${formatCalc(raw.report_gap)}`,
      score: `s_M = ${formatCalc(components.report_gap)}`,
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
  if (result?.sensor_kind === "multistate") {
    return [
      calculationRow(result, "state_change_count", "Cambios C_i(W)", "state_change_count", "state_change_count"),
      calculationRow(result, "state_probability", "Rareza P(s)", "state_probability", "state_probability"),
      calculationRow(result, "transition_probability", "Rareza P(s_j|s_i)", "transition_probability", "transition_probability"),
      calculationRow(result, "state_duration", "Duracion D_s(t)", "state_duration", "state_duration"),
      calculationRow(result, "report_gap", "Brecha M_i(t)", "report_gap", "report_gap"),
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
  if (result?.sensor_kind === "multistate") {
    return [
      { label: "Estado observado", value: reading.state ?? result.observed_state ?? "-" },
      { label: "Estado anterior", value: reading.previous_state ?? "-" },
      { label: "C_i(W)", value: reading.state_changes ?? "-" },
      { label: "P(s)", value: formatNumber(reading.state_probability) },
      { label: "P(s_j|s_i)", value: formatNumber(reading.transition_probability) },
      { label: "D_s(t)", value: formatSeconds(reading.state_duration_seconds) },
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
  const rows = [
    { label: "Lapso observado", value: timeWindowLabel(result.time_window) || "local" },
    { label: "Consistencia temporal", value: temporalConsistencyText(context.temporal_consistency) },
    { label: "Score local", value: formatNumber(context.local_score) },
    { label: "Score hora del dia", value: formatNumber(context.hour_of_day_score) },
    { label: "Patron repetido", value: context.repeated_pattern ? "si" : "no" },
    { label: "Dias similares", value: context.similar_days_in_month ?? "-" },
  ];
  if (result.replay) {
    rows.push(
      { label: "Reloj de deteccion simulado", value: shortDate(result.replay.detected_at_simulated) },
      { label: "Retraso por lote", value: formatDuration(result.replay.detection_delay_seconds) },
      { label: "Evaluacion causal", value: result.replay.causal ? "si, sin filas futuras" : "no" },
    );
  }
  return rows;
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
    return `missing ${reading.missing_ticks} lecturas / M=${formatSeconds(reading.report_gap_seconds ?? reading.gap_seconds)}`;
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
  if (result.sensor_kind === "multistate") {
    return [
      `estado ${reading.state ?? result.observed_state ?? "-"}`,
      `C=${reading.state_changes ?? "-"}`,
      `P(s)=${formatNumber(reading.state_probability)}`,
      `P(trans)=${formatNumber(reading.transition_probability)}`,
      `D=${formatSeconds(reading.state_duration_seconds)}`,
      `M=${formatSeconds(reading.report_gap_seconds)}`,
    ].join(" / ");
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

function formatSuspicionPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  return `${(Math.round(numeric * 1000) / 10).toLocaleString("es-CL", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 1,
  })} %`;
}

function normalityDetail(normality) {
  if (!normality?.exists) {
    return "sin generar";
  }
  return normality.mode === "persisted" ? "persistida en cache" : "en memoria";
}

function historyLoadPhaseLabel(phase) {
  const labels = {
    queued: "Carga en espera",
    uploading: "Subiendo archivo",
    saving_upload: "Guardando archivo",
    upload_saved: "Archivo recibido",
    preparing: "Preparando carga",
    reading_csv: "Leyendo CSV",
    validating_columns: "Validando columnas",
    normalizing: "Normalizando estructura",
    normalizing_timestamps: "Normalizando fechas",
    classifying_entities: "Clasificando entidades",
    normalizing_values: "Normalizando valores",
    sorting_history: "Ordenando historico",
    building_catalog: "Construyendo catalogo",
    history_ready: "Historico normalizado",
    resetting_state: "Reiniciando estado",
    persisting_normality: "Persistiendo normalidad",
    finalizing: "Finalizando carga",
    building_timeline: "Preparando visualizacion",
    ready: "Historico listo",
    failed: "Carga fallida",
  };
  return labels[phase] || "Procesando historico";
}

function formatDuration(value, fallback = "-") {
  if (value === undefined || value === null || value === "") {
    return fallback;
  }
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0) {
    return fallback;
  }
  if (seconds < 60) {
    return `${Math.round(seconds)} s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes} min ${remainingSeconds} s`;
}

function formatBytes(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "-";
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
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

createRoot(document.getElementById("root")).render(<App />);
