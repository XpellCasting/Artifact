import React, { lazy, Suspense, useEffect, useId, useState } from "react";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import { request } from "../../shared/api.js";
import { DECISIONS, NODE_LABELS, labelNode } from "./flow-lab-model.js";
const ConfidenceHistory = lazy(() => import("./FlowConfidenceHistory.jsx"));

const TABS = [
  ["summary", "Resumen"], ["diagnosis", "Diagnóstico"], ["matrix", "Matriz de confusión"],
  ["detection", "Detección"], ["timings", "Tiempos"], ["confidence", "Confianza"], ["detail", "Detalle"],
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
const BRIER = new Intl.NumberFormat(undefined, { maximumFractionDigits: 6 });
const count = (value) => value == null ? "—" : COUNT.format(value);
const percent = (value) => value == null ? "—" : PERCENT.format(value);
const duration = (value) => value == null ? "—" : `${MILLISECONDS.format(value)} ms`;
const seconds = (value) => value == null ? "—" : `${MILLISECONDS.format(value)} s`;
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

function DiagnosisPanel({ diagnosis, brier, labelled, nodes }) {
  const { by_class: scores } = diagnosis;
  const nodeNames = new Map(nodes.map((node) => [node.id, labelNode(node)]));
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
    {brier && <section aria-label="Brier multiclase">
      <h3>Brier multiclase</h3>
      {brier.reason ? <p className="fl-muted">{brier.reason}</p> : <>
        <p className="fl-hint">Va de 0 a 2. Cero indica probabilidades perfectas. Se usan las probabilidades de cada clasificador conservadas en Salida.</p>
        <div className="fl-table-scroll"><table aria-label="Brier por clasificador">
          <thead><tr><th scope="col">Clasificador</th><th scope="col">Brier</th><th scope="col">Lecturas con probabilidades / etiquetadas</th></tr></thead>
          <tbody>{Object.entries(brier.by_node).map(([id, score]) => <tr key={id}>
            <th scope="row">{nodeNames.get(id) ?? id}</th>
            <td>{BRIER.format(score.value)}</td><td>{count(score.count)} / {count(labelled)}</td>
          </tr>)}</tbody>
        </table></div>
      </>}
    </section>}
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

function DetectionPanel({ detection, measurements, sensor, onSensorChange }) {
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
    {measurements && <DetectionMeasurements measurements={measurements} sensor={sensor} onSensorChange={onSensorChange} />}
  </>;
}

function DetectionMeasurements({ measurements, sensor, onSensorChange }) {
  const delay = measurements.fault_delay_seconds;
  const alerts = measurements.false_alerts_by_sensor_day;
  const sensors = [...new Set(alerts.days.map((row) => row.sensor))];
  const selectedSensor = sensors.includes(sensor) ? sensor : sensors[0];
  const sensorDays = alerts.days.filter((row) => row.sensor === selectedSensor);
  return <>
    <section aria-label="Retraso hasta detectar una falla">
      <h3>Retraso hasta detectar una falla</h3>
      <p className="fl-hint">Cada secuencia de lecturas de falla del mismo sensor y tipo forma un episodio. El retraso se mide desde la primera lectura de esa secuencia hasta la primera señal de alteración.</p>
      <dl className="fl-run-cards">
        <Card label="Episodios detectados" value={delay.episodes == null ? "—" : `${count(delay.detected)} / ${count(delay.episodes)}`}
          detail={delay.missed == null ? null : `${count(delay.missed)} sin detectar`} />
        <Card label="Retraso mediano" value={seconds(delay.median)} />
        <Card label="Retraso P95" value={seconds(delay.p95)} />
      </dl>
      {delay.reason && <p className="fl-muted">{delay.reason}</p>}
    </section>
    <section aria-label="Falsas alertas por sensor y día">
      <h3>Falsas alertas por sensor y día</h3>
      {alerts.reason ? <p className="fl-muted">{alerts.reason}</p> : <>
        <p className="fl-hint">Cuenta alertas sobre lecturas normales y alertas de falla sobre lecturas ambientales. Cada día UTC con lecturas de esas clases aporta al promedio.</p>
        <dl className="fl-run-cards">
          <Card label="Falsas alertas" value={count(alerts.count)} />
          <Card label="Falsas alertas de falla" value={count(alerts.fault_count)} />
          <Card label="Promedio por sensor y día" value={MILLISECONDS.format(alerts.mean)}
            detail={`${count(alerts.days.length)} días observados por sensor en total`} />
        </dl>
        <label className="fl-matrix-mode">Sensor
          <select value={selectedSensor} onChange={(event) => onSensorChange(event.target.value)}>
            {sensors.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
        </label>
        <div className="fl-table-scroll"><table aria-label="Falsas alertas diarias del sensor">
          <thead><tr><th scope="col">Día UTC</th><th scope="col">Falsas alertas</th><th scope="col">De falla</th><th scope="col">Lecturas normales o ambientales</th></tr></thead>
          <tbody>{sensorDays.map((row) => <tr key={row.day}>
            <th scope="row">{row.day}</th><td>{count(row.false_alerts)}</td>
            <td>{count(row.false_fault_alerts)}</td><td>{count(row.valid_readings)}</td>
          </tr>)}</tbody>
        </table></div>
      </>}
    </section>
  </>;
}

function TimingsPanel({ timings, measurements }) {
  const { total, filters } = timings;
  const latency = measurements?.latency_ms;
  const resources = measurements?.resources;
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
    {latency && <section aria-label="Latencia por lectura">
      <h3>Latencia por lectura</h3>
      <dl className="fl-run-cards">
        <Card label="Mediana" value={duration(latency.median)} />
        <Card label="P95" value={duration(latency.p95)} />
      </dl>
      <p className="fl-hint">{latency.reason ||
        `${count(latency.count)} lecturas medidas desde que empieza a procesarse su grupo hasta que está disponible la decisión de salida. Las lecturas simultáneas comparten el tiempo del grupo.`}</p>
    </section>}
    {measurements && <section aria-label="Consumo de CPU y RAM">
      <h3>CPU y RAM</h3>
      <dl className="fl-run-cards">
        <Card label="Tiempo de CPU" value={seconds(resources?.cpu_seconds)} />
        <Card label="CPU media" value={resources?.cpu_average_percent == null ? "—" : percent(resources.cpu_average_percent / 100)} />
        <Card label="Pico de RAM (RSS)" value={resources?.peak_rss_bytes == null ? "—" :
          `${MILLISECONDS.format(resources.peak_rss_bytes / 1048576)} MiB`} />
      </dl>
      <p className="fl-hint">{resources
        ? `Proceso de la API durante la evaluación. El 100 % de CPU equivale a un núcleo lógico. RAM muestreada cada ${resources.rss_interval_ms} ms.`
        : "La ejecución no registró el consumo de CPU y RAM."}</p>
    </section>}
  </>;
}

function MetricsPanel({ tab, metrics, nodes, mode, onModeChange, sensor, onSensorChange }) {
  if (tab === "summary") return <SummaryPanel metrics={metrics} />;
  if (tab === "timings") return <TimingsPanel timings={metrics.timings} measurements={metrics.measurements} />;
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
  if (tab === "diagnosis") return <DiagnosisPanel diagnosis={metrics.diagnosis} brier={metrics.measurements?.brier}
    labelled={metrics.support.labelled} nodes={nodes} />;
  if (tab === "matrix") return <MatrixPanel diagnosis={metrics.diagnosis} mode={mode} onModeChange={onModeChange} />;
  return <DetectionPanel detection={metrics.detection} measurements={metrics.measurements} sensor={sensor} onSensorChange={onSensorChange} />;
}

export default function FlowRunResults({ apiBaseUrl, run, activeTab, onTabChange, children }) {
  const id = useId();
  const [requested, setRequested] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [response, setResponse] = useState(null);
  const [mode, setMode] = useState("count");
  const [sensor, setSensor] = useState("");
  useEffect(() => {
    if (!["detail", "confidence"].includes(activeTab)) setRequested(true);
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
        {source?.context_replay && <span>Contexto · {source.context_replay.filename} · {source.context_replay.applied.toLocaleString()} eventos</span>}
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
      {key === "detail" ? children : key === "confidence" && key === activeTab ?
        <Suspense fallback={<p role="status">Cargando historial de confianza…</p>}><ConfidenceHistory apiBaseUrl={apiBaseUrl} run={run} /></Suspense>
        : key === activeTab && (
        !response ? <p role="status" className="fl-muted">Cargando métricas…</p>
          : response.error ? <div className="fl-notice fl-error" role="alert">
            <span>{response.error}</span>
            <button onClick={() => { setResponse(null); setAttempt((value) => value + 1); }}>Reintentar</button>
          </div>
            : <MetricsPanel tab={key} metrics={response.metrics} nodes={flow.nodes ?? []} mode={mode}
              onModeChange={setMode} sensor={sensor} onSensorChange={setSensor} />
      )}
    </div>)}
  </div>;
}
