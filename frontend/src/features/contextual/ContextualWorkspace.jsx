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

import { Metric, Panel, SummaryRow, Empty, TimeStack, KeyValueGrid } from "../../shared/components.jsx";
import { compactEntityId, kindText, formatPercent, formatCount, formatNumber, formatCalc, formatSeconds, formatBucketSeconds, shortDate } from "../../shared/analysis.jsx";

export function ContextualWorkspace({ mapData, results, summary, busy, error, onReload, onRebuild }) {
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


export function ContextMatrix({ mapData, onSelectEdge }) {
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


export function ContextResultsTable({ results, mapStale, onResultClick }) {
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


export function ContextStatusBadge({ value, label }) {
  const key = contextClassKey(value);
  return <span className={`context-status-badge ${key}`}>{label || contextOutcomeLabel(value)}</span>;
}


export function ContextEdgeModal({ edge, mapVersion, onClose }) {
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


export function ContextResultModal({ result, fallbackMapStale, onClose }) {
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


export function contextMapNodes(mapData) {
  return Array.isArray(mapData?.nodes) ? mapData.nodes : [];
}


export function contextMapEdges(mapData) {
  return Array.isArray(mapData?.edges) ? mapData.edges : [];
}


export function contextMapHasData(mapData) {
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


export function contextMapIsStale(mapData) {
  return contextBoolean(
    mapData?.refresh?.stale
    ?? mapData?.refresh?.is_stale
    ?? mapData?.summary?.stale
    ?? mapData?.stale,
  );
}


export function contextMapStaleReasons(mapData) {
  return contextArray(
    mapData?.refresh?.stale_reasons,
    mapData?.refresh?.reasons,
    mapData?.stale_reasons,
  ).map((reason) => typeof reason === "object" ? formatContextValue(reason.reason || reason.code || reason) : String(reason));
}


export function contextRefreshTrigger(mapData) {
  return formatContextValue(
    mapData?.refresh?.trigger
    ?? mapData?.refresh?.last_trigger
    ?? mapData?.refresh?.status
    ?? "sin disparador informado",
  );
}


export function contextBuiltAtText(mapData) {
  if (!mapData?.built_at) {
    return "sin fecha de construccion";
  }
  return shortDate(mapData.built_at);
}


export function contextSourceRangeText(sourceRange) {
  if (Array.isArray(sourceRange)) {
    return `${shortDate(sourceRange[0])} → ${shortDate(sourceRange[1])}`;
  }
  if (sourceRange && typeof sourceRange === "object") {
    return `${shortDate(sourceRange.start || sourceRange.from)} → ${shortDate(sourceRange.end || sourceRange.to)}`;
  }
  return sourceRange || "-";
}


export function contextConfigRows(config = {}) {
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


export function contextAcceptedEdgeCount(mapData, edges) {
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


export function contextMatrixCell(matrix, rowIndex, columnIndex) {
  if (!Array.isArray(matrix) || !Array.isArray(matrix[rowIndex])) {
    return null;
  }
  return matrix[rowIndex][columnIndex] ?? null;
}


export function contextFiniteNumber(value) {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}


export function contextNumber(...values) {
  for (const value of values) {
    const number = contextFiniteNumber(value);
    if (number !== null) {
      return number;
    }
  }
  return null;
}


export function contextBoolean(value) {
  if (typeof value === "string") {
    return ["true", "1", "yes", "si", "accepted", "stale", "obsoleto"].includes(value.trim().toLowerCase());
  }
  return Boolean(value);
}


export function contextArray(...values) {
  for (const value of values) {
    if (Array.isArray(value)) {
      return value;
    }
  }
  return [];
}


export function contextHeatColor(value, scale) {
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


export function formatContextMatrixValue(value) {
  const number = contextFiniteNumber(value);
  if (number === null) {
    return "N/D";
  }
  return number.toFixed(2);
}


export function contextEdgeSource(edge = {}) {
  return edge.source_entity_id || edge.source || edge.entity_i || edge.entity_id || "-";
}


export function contextEdgeTarget(edge = {}) {
  return edge.target_entity_id || edge.target || edge.entity_j || edge.related_entity_id || "-";
}


export function contextEdgeAccepted(edge) {
  if (!edge) {
    return false;
  }
  return contextBoolean(edge.accepted ?? edge.is_accepted ?? edge.status);
}


export function contextOutcome(result = {}) {
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


export function contextOutcomeLabel(value) {
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


export function contextClassKey(value) {
  return String(value || "unknown")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
}


export function contextOutcomeCounts(results = [], summary = {}) {
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


export function contextCoverage(result = {}) {
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



export function contextResultMapIsStale(result, fallback) {
  const value = result?.map_quality?.stale
    ?? result?.map?.stale
    ?? result?.map_stale;
  return value === undefined || value === null ? Boolean(fallback) : contextBoolean(value);
}


export function contextStatScore(result = {}) {
  return contextNumber(result.stat_score, result.statistical_score, result.suspicion_score);
}


export function contextResultEvidence(result = {}) {
  return contextArray(
    result.evidence,
    result.evaluated_relations,
    result.related_evidence,
    result.relations,
  );
}


export function contextEvidenceEntity(item = {}) {
  return item.neighbor_id
    || item.related_entity_id
    || item.entity_id
    || item.target_entity_id
    || item.source_entity_id
    || "-";
}


export function contextEvidenceStatus(item = {}) {
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


export function contextEvidenceStatusLabel(status) {
  return contextOutcomeLabel(status);
}


export function formatContextPercent(value) {
  const number = contextFiniteNumber(value);
  if (number === null) {
    return "-";
  }
  return number > 1 ? `${number.toFixed(1)}%` : formatPercent(number);
}


export function formatContextValue(value) {
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

