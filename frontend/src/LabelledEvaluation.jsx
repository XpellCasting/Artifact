import React, { useEffect, useState } from "react";

export default function LabelledEvaluation({ apiBaseUrl, replay, busy, historyReady, onUpload }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBaseUrl}/validation/labelled/metrics`, { signal: controller.signal })
      .then(async response => { const value = await response.json(); if (!response.ok) throw new Error(value.detail || "No se pudieron cargar las métricas."); return value; })
      .then(value => { if (!controller.signal.aborted) { setData(value); setError(""); } }).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [apiBaseUrl, replay?.processed_rows, replay?.filename, replay?.status, replay?.labelled, replay?.diagnostic_run_id]);
  return <section className="review-panel" aria-label="Conjunto de datos etiquetados">
    <h2>Conjunto de datos etiquetados</h2>
    <p>Importa entity_id,state,last_changed,es_sintetica,tipo,tipo_falla. El archivo aparece también en Evaluación con eventos futuros, con solo entity_id,state,last_changed. La carga no inicia los filtros.</p>
    <label className="file-action">Cargar conjunto etiquetado<input type="file" accept=".csv,text/csv" disabled={Boolean(busy) || !historyReady} onChange={onUpload} /></label>
    {!historyReady && <p>Carga primero el histórico de Desarrollo y espera a que se prepare la base del detector.</p>}
    {error && <p role="alert">{error}</p>}
    {data?.loaded && data.run_id === replay?.diagnostic_run_id && <>
      <LabelledEvaluationReport data={data} />
    </>}
  </section>;
}

export function LabelledEvaluationReport({ data }) {
  const ratio = (a, b) => `${a} / ${b}`;
  return <>
      <p><strong>{data.filename}</strong> · {data.status === "completed" ? "Evaluación terminada" : "Resultados parciales: avanza los eventos futuros para evaluar"}</p>
      <div className="review-table-scroll"><table><thead><tr><th>Sensor</th><th>Datos normales</th><th>Fallas no inyectadas detectadas / total</th><th>Fallas inyectadas detectadas / total</th><th>Anomalías detectadas / total</th><th>Falsos positivos</th><th>Evaluables</th><th>No evaluables</th><th>Procesados / total</th><th>Sin etiqueta</th></tr></thead>
        <tbody>{data.entities.map(row => <tr key={row.entity_id}><td>{row.entity_id}</td><td>{row.normal}</td><td>{ratio(row.real_detected, row.real_faults)}</td><td>{ratio(row.synthetic_detected, row.synthetic_faults)}</td><td>{ratio(row.anomalies_detected, row.anomalies)}</td><td>{row.false_positives}</td><td>{row.evaluable ?? "—"}</td><td>{row.non_evaluable ?? "—"} {row.non_evaluable_invalid_state ? `(estado inválido: ${row.non_evaluable_invalid_state})` : ""} {row.non_evaluable_missing_baseline ? `(sin base: ${row.non_evaluable_missing_baseline})` : ""}</td><td>{ratio(row.processed, row.total)}</td><td>{row.unlabelled}</td></tr>)}</tbody></table></div>
      <p>{data.statistical_evaluation_version === "pre_suppression_v1" ? "Las detecciones se cuentan por lectura después del umbral final y antes de suprimir alertas repetidas." : "Evaluación anterior: conteo basado en alertas emitidas."} Una lectura no evaluable no equivale a una falla detectada ni a normalidad confirmada. Las etiquetas NULL no se consideran normales y los denominadores conservan todo el archivo.</p>
      {data.unmatched_detections > 0 && <p>{data.unmatched_detections} alertas no se pudieron asociar a una lectura inequívoca y se excluyen del conteo.</p>}
  </>;
}
