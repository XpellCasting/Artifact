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
import { downloadCsv } from "./csv.js";
import ReviewWorkspace from "../features/review/ReviewWorkspace.jsx";
import DiagnosticWorkspace from "../features/diagnostics/DiagnosticWorkspace.jsx";
import FlowLab from "../features/flows/FlowLab.jsx";
import { isStatisticalSuspicion, statisticalCounts, episodeLabel } from "./statistical-routing.js";
import { isEnvironmentalResult, contextRoutingLabel, contextRoutingReasonLabel, contextRoutingCounts, contextCoherenceLabel, contextObservationLabel, preciseContextTimestamp, contextCommonCause } from "./context-routing.js";

import { MARKER_GROUP_GAP_MS, EQUATIONS, STATE_EQUATIONS, MULTISTATE_EQUATIONS } from "./analysis-constants.js";

export function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}


export function downloadResultsCsv(results) {
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


export function compactEntityId(entityId) {
  const value = String(entityId || "-");
  const withoutDomain = value.includes(".") ? value.split(".").slice(1).join(".") : value;
  return withoutDomain.length > 22 ? `${withoutDomain.slice(0, 20)}…` : withoutDomain;
}


export function timeWindowLabel(value) {
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


export function isRedSuspicion(result, suspicionThreshold) {
  return isStatisticalSuspicion(result);
}


export function compareResultsByExecutionDate(left, right) {
  const leftTime = Date.parse(left?.timestamp || "");
  const rightTime = Date.parse(right?.timestamp || "");
  const safeLeft = Number.isFinite(leftTime) ? leftTime : 0;
  const safeRight = Number.isFinite(rightTime) ? rightTime : 0;
  if (safeRight !== safeLeft) {
    return safeRight - safeLeft;
  }
  return Number(right?.suspicion_score || 0) - Number(left?.suspicion_score || 0);
}


export function mergeTimelineWithSuspiciousPoints(points, results, timeline, suspicionThreshold) {
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


export function resultChartValue(result, timeline) {
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


export function groupConsecutiveMarkers(markers) {
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


export function entityLabel(entity) {
  return `${kindText(entity.sensor_kind)} / ${entity.device_class || "estado"}`;
}


export function kindText(kind) {
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


export function multistateStatusText(status) {
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


export function timelineDateRange(timelines) {
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


export function timelineQueryString(startValue, endValue) {
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


export function inputDateToIso(value) {
  if (!value) {
    return "";
  }
  const time = new Date(value).getTime();
  return Number.isFinite(time) ? new Date(time).toISOString() : "";
}


export function toDatetimeLocalValue(timestamp) {
  const date = new Date(timestamp);
  if (!Number.isFinite(date.getTime())) {
    return "";
  }
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}


export function equationNames(indicators = [], sensorKind = "") {
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


export function equationsForResult(result) {
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


export function equationWithCalculationValues(equation, result) {
  if (result?.sensor_kind === "binary") {
    return binaryEquationWithValues(equation, result);
  }
  if (result?.sensor_kind === "multistate") {
    return multistateEquationWithValues(equation, result);
  }
  return continuousEquationWithValues(equation, result);
}


export function continuousEquationWithValues(equation, result) {
  const reading = result?.reading || {};
  if (equation.id === "residual_sequential") return {
    equation: `Residuo = ${formatCalc(reading.residual)}; z = ${formatCalc(reading.z)}; FOCuS = ${formatCalc(reading.channels?.D)}`,
    score: `S = ${formatCalc(reading.joint_score)}; umbral calibrado = ${formatCalc(reading.threshold)}`,
  };
  if (equation.id === "constant_persistence") return {
    equation: equation.equation, score: `Evidencia de constancia = ${formatCalc(reading.constant_score)}`,
  };
  return { equation: equation.equation, score: equation.score };
}

export function numericPointValue(value) {
  if (value === null || value === undefined || value === "") {
    return NaN;
  }
  return Number(value);
}


export function binaryEquationWithValues(equation, result) {
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


export function multistateEquationWithValues(equation, result) {
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


export function calculationRows(result) {
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
  const reading = result?.reading || {};
  return ["P", "D", "N"].map(channel => ({
    id: channel, title: { P: "Residuo puntual", D: "FOCuS", N: "Energía residual" }[channel],
    raw: formatNumber(reading.channels?.[channel]),
    score: formatNumber(reading.channel_q99?.[channel] ? reading.channels?.[channel] / reading.channel_q99[channel] : null),
    active: reading.dominant_channel === channel && (result?.detection_sources || []).includes("residual_sequential"),
  }));
}


export function calculationRow(result, indicator, title, rawKey, scoreKey) {
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


export function readingRows(result) {
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
    { label: "Valor observado", value: result?.value ?? "-" },
    { label: "Predicción", value: formatNumber(reading.expected_value) },
    { label: "Residuo", value: formatNumber(reading.residual) },
    { label: "Residuo normalizado", value: formatNumber(reading.z) },
    { label: "Estadístico FOCuS", value: formatNumber(reading.focus_statistic) },
    { label: "Puntaje conjunto", value: formatNumber(reading.joint_score) },
    { label: "Umbral calibrado", value: formatNumber(reading.threshold) },
    { label: "Evidencia de constancia", value: formatNumber(reading.constant_score) },
  ];
}


export function objectRows(object) {
  return Object.entries(object || {}).map(([key, value]) => ({
    label: key,
    value: typeof value === "number" ? formatNumber(value) : String(value ?? "-"),
  }));
}


export function temporalContextRows(result) {
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


export function temporalConsistencyText(value) {
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


export function normalizedEntityPoints(fragment, normality) {
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


export function hourOfDay(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) {
    return 0;
  }
  return date.getHours() + date.getMinutes() / 60;
}


export function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}


export function standardDeviation(values, mean = average(values)) {
  if (values.length < 2) {
    return 0;
  }
  return Math.sqrt(average(values.map((value) => (value - mean) ** 2)));
}


export function formatReading(result) {
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


export function formatPercent(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return `${Math.round(Number(value) * 100)}%`;
}


export function formatHistoryRange(history) {
  if (!history?.exists) {
    return "sin historico cargado";
  }
  return `${shortDate(history.start).slice(0, 16)} -> ${shortDate(history.end).slice(0, 16)}`;
}


export function formatSuspicionPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  return `${(Math.round(numeric * 1000) / 10).toLocaleString("es-CL", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 1,
  })} %`;
}


export function normalityDetail(normality) {
  if (!normality?.exists) {
    return "sin generar";
  }
  return normality.mode === "persisted" ? "persistida en cache" : "en memoria";
}


export function formatDuration(value, fallback = "-") {
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


export function formatBytes(value) {
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


export function formatElapsed(ms) {
  if (ms === undefined || ms === null) {
    return "-";
  }
  return Number(ms) < 1000 ? `${Number(ms).toFixed(1)} ms` : `${(Number(ms) / 1000).toFixed(3)} s`;
}


export function formatCount(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return Number(value).toLocaleString("es-CL");
}


export function formatAxisValue(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2);
}


export function formatNumber(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return Number(value).toFixed(4);
}


export function formatCalc(value) {
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


export function formatSigned(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `${number >= 0 ? "+" : ""}${number.toFixed(3)}`;
}


export function formatSeconds(value) {
  if (value === undefined || value === null) {
    return "-";
  }
  return `${Number(value).toFixed(0)}s`;
}


export function formatBucketSeconds(seconds) {
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


export function shortDate(value) {
  return value ? String(value).replace("T", " ").slice(0, 19) : "-";
}


export function splitDateTime(value) {
  const formatted = shortDate(value);
  if (formatted === "-") {
    return { date: "-", time: "-" };
  }
  const [date, time = ""] = formatted.split(" ");
  return { date, time: time.replace("Z", "") || "-" };
}

