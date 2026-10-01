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
import { downloadCsv } from "../../shared/csv.js";
import ReviewWorkspace from "../review/ReviewWorkspace.jsx";
import DiagnosticWorkspace from "../diagnostics/DiagnosticWorkspace.jsx";
import FlowLab from "../flows/FlowLab.jsx";
import { isStatisticalSuspicion, statisticalCounts, episodeLabel } from "../../shared/statistical-routing.js";
import { isEnvironmentalResult, contextRoutingLabel, contextRoutingReasonLabel, contextRoutingCounts, contextCoherenceLabel, contextObservationLabel, preciseContextTimestamp, contextCommonCause } from "../../shared/context-routing.js";

import { SummaryRow, Empty, TimeStack, KeyValueGrid } from "../../shared/components.jsx";
import { timeWindowLabel, isRedSuspicion, mergeTimelineWithSuspiciousPoints, resultChartValue, groupConsecutiveMarkers, kindText, multistateStatusText, equationNames, equationsForResult, equationWithCalculationValues, numericPointValue, calculationRows, readingRows, objectRows, temporalContextRows, temporalConsistencyText, normalizedEntityPoints, formatReading, formatSuspicionPercent, formatDuration, formatCount, formatAxisValue, formatNumber, formatSeconds, formatBucketSeconds, shortDate } from "../../shared/analysis.jsx";
import { DEFAULT_SUSPICION_THRESHOLD } from "../../shared/analysis-constants.js";

export function EpisodeGroupingControl({ minutes, onChange, busy, summary, replay }) {
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


export function SuspicionThresholdControl({ value, analyzedValue, busy, onChange }) {
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


export function ContinuousConfigEditor({ configs, busy, onChange, onSave }) {
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


export function ResultEntityFilters({ entities, selectedIds, onToggle, onShowAll }) {
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


export function ResultsTable({ results, onResultClick }) {
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


export function CalculationModal({ result, onClose }) {
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


export function NormalizationModal({ preview, onClose }) {
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


export function NormalizationEntityPlane({ fragment, normality }) {
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


export function EquationCardWithValues({ equation, result }) {
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


export function TimelineChart({ timeline, results, suspicionThreshold, onResultClick }) {
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

