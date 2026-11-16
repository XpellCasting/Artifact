import React, { lazy, Suspense, useEffect, useId, useState } from "react";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import { request } from "../../shared/api.js";
import { DECISIONS, NODE_LABELS } from "./flow-lab-model.js";

const TABS = [
  ["summary", "Resumen"], ["diagnosis", "Diagnóstico"], ["matrix", "Matriz de confusión"],
  ["detection", "Detección"], ["timings", "Tiempos"], ["detail", "Detalle"],
];
const CLASS_LABELS = { normal: "Normal", anomalia: "Anomalía ambiental", Falla: "Falla de sensor" };
const CLASSES = Object.keys(CLASS_LABELS);
const DIAGNOSES = [...CLASSES, "no_diagnosis"];
const SCORES = [["precision", "Precisión"], ["recall", "Recall"], ["f1", "F1"]];
const DETECTION_LABELS = [
  ["detected", "Detectadas"], ["no_signal", "Sin señal de alteración"], ["not_evaluated", "No evaluadas"],
];
const FILTERS = ["statistical_focus", "contextual", "classifier"];
const COUNT = new Intl.NumberFormat();
const PERCENT = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 2 });
const MILLISECONDS = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
const count = (value) => value == null ? "—" : COUNT.format(value);
const percent = (value) => value == null ? "—" : PERCENT.format(value);
const duration = (value) => value == null ? "—" : `${MILLISECONDS.format(value)} ms`;
const Bars = lazy(() => import("./FlowMetricBars.jsx"));

function Card({ label, value, detail }) {
  return <div className="fl-run-card">
    <dt>{label}</dt>
    <dd><strong>{value}</strong>{detail && <small>{detail}</small>}</dd>
  </div>;
}

function Score({ value, reason, numerator, denominator }) {
  return <>
    <strong>{percent(value)}</strong>
    {denominator != null && <small>{count(numerator)} / {count(denominator)}</small>}
    {reason && <small>{reason}</small>}
  </>;
}

function MetricBars({ percentage, ...props }) {
  const axis = percentage ? { max: 100, tickNumber: 5, valueFormatter: (value) => percent(value / 100) } : { valueFormatter: count };
  return <Suspense fallback={<p role="status" className="fl-muted">Cargando gráfico…</p>}>
    <Bars {...props} axis={axis} />
  </Suspense>;
}

function SummaryPanel({ metrics }) {
  const { diagnosis, support, reason } = metrics;
  const macro = diagnosis?.macro_f1;
  const missing = macro?.missing_classes ?? CLASSES.filter((key) => support.by_class?.[key] === 0);
  const missingLabels = missing.map((key) => CLASS_LABELS[key]).join(", ");
  const macroDetail = missingLabels ? `Clases reales ausentes · ${missingLabels}` : macro?.reason;
  return <>
    {reason && <p className="fl-muted" role="status">{reason}</p>}
    <dl className="fl-run-cards">
      <Card label="F1 macro" value={percent(macro?.value)} detail={macroDetail} />
      <Card label="Cobertura de diagnóstico" value={percent(diagnosis?.coverage.value)}
        detail={diagnosis && `${count(diagnosis.coverage.numerator)} / ${count(support.labelled)} lecturas etiquetadas`} />
      <Card label="Sin diagnóstico" value={percent(diagnosis?.no_diagnosis.value)}
        detail={diagnosis && `${count(diagnosis.no_diagnosis.numerator)} / ${count(support.labelled)} lecturas etiquetadas`} />
      <Card label="Lecturas etiquetadas" value={count(support.labelled)} />
    </dl>
    <p className="fl-muted">Lecturas sin etiqueta <strong>{count(support.unlabelled)}</strong></p>
    {diagnosis?.no_diagnosis.numerator > 0 && <div className="fl-table-scroll">
      <table aria-label="Desglose de lecturas sin diagnóstico"><thead><tr><th scope="col">Decisión sin diagnóstico</th><th scope="col">Lecturas</th></tr></thead>
        <tbody>{Object.entries(diagnosis.no_diagnosis.by_decision).map(([key, value]) =>
          <tr key={key}><th scope="row">{DECISIONS[key]}</th><td>{count(value)}</td></tr>)}</tbody>
      </table>
    </div>}
  </>;
}

function DiagnosisPanel({ diagnosis }) {
  const { by_class: scores } = diagnosis;
  const colors = ["var(--ha-primary)", "var(--ha-success)", "var(--ha-warning)"];
  const series = SCORES.map(([key, label], index) => ({
    label, color: colors[index], valueFormatter: (value) => value == null ? "Sin valor" : percent(value / 100),
    data: CLASSES.map((kind) => scores[kind][key].value == null ? null : scores[kind][key].value * 100),
  }));
  return <>
    <MetricBars title="Precisión, recall y F1 por clase" labels={Object.values(CLASS_LABELS)} series={series} percentage />
    <div className="fl-table-scroll"><table aria-label="Métricas de diagnóstico por clase">
      <thead><tr><th scope="col">Clase real</th>{SCORES.map(([, label]) => <th scope="col" key={label}>{label}</th>)}<th scope="col">Lecturas reales</th></tr></thead>
      <tbody>{CLASSES.map((kind) => <tr key={kind}>
        <th scope="row">{CLASS_LABELS[kind]}</th>
        {SCORES.map(([key]) => <td key={key} className="fl-run-value">
          <strong>{percent(scores[kind][key].value)}</strong>
          {scores[kind][key].reason && <small>{scores[kind][key].reason}</small>}
        </td>)}
        <td>{count(scores[kind].support)}</td>
      </tr>)}</tbody>
    </table></div>
  </>;
}

function MatrixPanel({ diagnosis, mode, onModeChange }) {
  return <>
    <label className="fl-matrix-mode">Mostrar matriz en
      <select value={mode} onChange={(event) => onModeChange(event.target.value)}>
        <option value="count">Cantidades</option>
        <option value="percent">Porcentajes por fila</option>
      </select>
    </label>
    <div className="fl-table-scroll"><table className="fl-matrix" aria-label="Clase real frente a diagnóstico final">
      <thead><tr><th scope="col">Clase real / Diagnóstico final</th>{DIAGNOSES.map((key) =>
        <th scope="col" key={key}>{CLASS_LABELS[key] || "Sin diagnóstico"}</th>)}<th scope="col">Lecturas reales</th></tr></thead>
      <tbody>{CLASSES.map((kind) => {
        const total = diagnosis.by_class[kind].support;
        return <tr key={kind}><th scope="row">{CLASS_LABELS[kind]}</th>
          {DIAGNOSES.map((key) => {
            const value = diagnosis.matrix[kind][key];
            const fraction = total ? value / total : null;
            const color = key === kind ? "var(--ha-primary)" : key === "no_diagnosis" ? "var(--ha-muted)" : "var(--ha-warning)";
            const backgroundColor = fraction ? `color-mix(in srgb, ${color} ${8 + 24 * fraction}%, var(--fl-surface))` : undefined;
            return <td key={key} style={{ backgroundColor }}>
              {mode === "count" ? count(value) : fraction == null ? <abbr title="Sin lecturas de esta clase">—</abbr> : percent(fraction)}
            </td>;
          })}
          <td>{count(total)}</td>
        </tr>;
      })}</tbody>
    </table></div>
  </>;
}

function DetectionPanel({ detection }) {
  const classes = ["anomalia", "Falla"];
  const groups = detection.by_class;
  const colors = ["var(--ha-primary)", "var(--ha-muted)", "var(--ha-warning)"];
  const series = DETECTION_LABELS.map(([key, label], index) => ({
    label, color: colors[index], stack: "lecturas", data: classes.map((kind) => groups[kind][key]),
    valueFormatter: count,
  }));
  return <>
    <MetricBars title="Estado de detección por clase" labels={classes.map((key) => CLASS_LABELS[key])} series={series} height={200} />
    <div className="fl-table-scroll"><table aria-label="Detección por clase">
      <thead><tr><th scope="col">Clase real</th>{DETECTION_LABELS.map(([, label]) =>
        <th scope="col" key={label}>{label}</th>)}<th scope="col">Recall de detección</th></tr></thead>
      <tbody>{classes.map((kind) => <tr key={kind}><th scope="row">{CLASS_LABELS[kind]}</th>
        {DETECTION_LABELS.map(([key]) => <td key={key}>{count(groups[kind][key])}</td>)}
        <td className="fl-run-value"><Score {...groups[kind].recall} /></td>
      </tr>)}</tbody>
    </table></div>
    <div className="fl-table-scroll"><table aria-label="Tasas de falsas alarmas">
      <thead><tr><th scope="col">Falsas alarmas</th><th scope="col">Tasa y conteos</th></tr></thead>
      <tbody>
        <tr><th scope="row">Normales señaladas como alteración</th><td className="fl-run-value"><Score {...detection.normal_as_alteration} /></td></tr>
        <tr><th scope="row">Lecturas válidas diagnosticadas como falla</th><td className="fl-run-value"><Score {...detection.valid_as_fault} /></td></tr>
      </tbody>
    </table></div>
  </>;
}

function TimingsPanel({ timings }) {
  const { total, filters } = timings;
  const series = [{
    label: "Tiempo acumulado", color: "var(--ha-primary)", data: FILTERS.map((key) => filters[key].ms),
    valueFormatter: duration,
  }];
  return <>
    <dl className="fl-run-cards">
      <Card label={total.source === "elapsed_ms" ? "Tiempo de evaluación del flujo" : "Tiempo total registrado"}
        value={duration(total.ms)} detail={total.reason} />
    </dl>
    {series[0].data.some((value) => value != null) &&
      <MetricBars title="Tiempos acumulados de los filtros en milisegundos"
        labels={FILTERS.map((key) => NODE_LABELS[key])} series={series} height={200} />}
    <div className="fl-table-scroll"><table aria-label="Tiempos acumulados por filtro">
      <thead><tr><th scope="col">Filtro</th><th scope="col">Tiempo acumulado</th></tr></thead>
      <tbody>{FILTERS.map((key) => <tr key={key}>
        <th scope="row">{NODE_LABELS[key]}</th><td className="fl-run-value">
          <strong>{duration(filters[key].ms)}</strong>{filters[key].reason && <small>{filters[key].reason}</small>}
        </td>
      </tr>)}</tbody>
    </table></div>
  </>;
}

function MetricsPanel({ tab, metrics, mode, onModeChange }) {
  if (tab === "summary") return <SummaryPanel metrics={metrics} />;
  if (tab === "timings") return <TimingsPanel timings={metrics.timings} />;
  if (!metrics.diagnosis) return <>
    <p className="fl-muted" role="status">{metrics.reason}</p>
    {tab === "diagnosis" && metrics.support.by_class && <div className="fl-table-scroll">
      <table aria-label="Conteos de referencia guardados">
        <thead><tr><th scope="col">Clase real</th><th scope="col">Lecturas reales</th></tr></thead>
        <tbody>{CLASSES.map((key) => <tr key={key}>
          <th scope="row">{CLASS_LABELS[key]}</th><td>{count(metrics.support.by_class[key])}</td>
        </tr>)}</tbody>
      </table>
    </div>}
  </>;
  if (tab === "diagnosis") return <DiagnosisPanel diagnosis={metrics.diagnosis} />;
  if (tab === "matrix") return <MatrixPanel diagnosis={metrics.diagnosis} mode={mode} onModeChange={onModeChange} />;
  return <DetectionPanel detection={metrics.detection} />;
}

export default function FlowRunResults({ apiBaseUrl, run, activeTab, onTabChange, children }) {
  const id = useId();
  const [requested, setRequested] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [response, setResponse] = useState(null);
  const [mode, setMode] = useState("count");
  useEffect(() => {
    if (activeTab !== "detail") setRequested(true);
  }, [activeTab]);
  useEffect(() => {
    if (!requested) return;
    const controller = new AbortController();
    request(apiBaseUrl, `/flow-runs/${encodeURIComponent(run.run_id)}/metrics`, { signal: controller.signal })
      .then((metrics) => { if (!controller.signal.aborted) setResponse({ metrics }); })
      .catch((error) => { if (!controller.signal.aborted) setResponse({ error: error.message }); });
    return () => controller.abort();
  }, [apiBaseUrl, attempt, requested, run.run_id]);

  const { source, flow } = run.snapshot;
  return <div className="fl-run-results">
    <header className="fl-run-header">
      <h3>{source?.filename || (run.snapshot.execution_mode === "causal_live" ? "Eventos de Home Assistant" : "Histórico")}</h3>
      <p className="fl-muted">
        <span>{flow.name}</span>
        <time dateTime={run.created_at}>{new Date(run.created_at).toLocaleString()}</time>
        {source?.fingerprint && <span>Huella <code title={source.fingerprint}>{source.fingerprint.slice(0, 16)}</code></span>}
      </p>
    </header>
    <Tabs className="fl-run-tabs" aria-label="Resultados de la ejecución" value={activeTab} variant="scrollable"
      scrollButtons="auto" allowScrollButtonsMobile onChange={(_, tab) => onTabChange(tab)}
      slotProps={{ startScrollButton: { "aria-label": "Desplazar pestañas a la izquierda" },
        endScrollButton: { "aria-label": "Desplazar pestañas a la derecha" } }}>
      {TABS.map(([key, label]) => <Tab key={key} id={`${id}-tab-${key}`} aria-controls={`${id}-panel-${key}`}
        label={label} value={key} disableRipple />)}
    </Tabs>
    {TABS.map(([key]) => <div key={key} className="fl-run-panel" role="tabpanel"
      id={`${id}-panel-${key}`} aria-labelledby={`${id}-tab-${key}`} hidden={key !== activeTab} tabIndex={0}>
      {key === "detail" ? children : key === activeTab && (
        !response ? <p role="status" className="fl-muted">Cargando métricas…</p>
          : response.error ? <div className="fl-notice fl-error" role="alert">
            <span>{response.error}</span>
            <button onClick={() => { setResponse(null); setAttempt((value) => value + 1); }}>Reintentar</button>
          </div>
            : <MetricsPanel tab={key} metrics={response.metrics} mode={mode} onModeChange={setMode} />
      )}
    </div>)}
  </div>;
}
