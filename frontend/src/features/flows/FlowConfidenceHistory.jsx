import React, { useEffect, useState } from "react";
import { LineChart } from "@mui/x-charts/LineChart";
import { request } from "../../shared/api.js";
import { confidenceChartData } from "./confidence-history-model.js";

export default function FlowConfidenceHistory({ apiBaseUrl, run }) {
  const [selection, setSelection] = useState({ runId: run.run_id, entity: "", offset: 0 });
  const selected = selection.runId === run.run_id ? selection : { runId: run.run_id, entity: "", offset: 0 };
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const key = `${run.run_id}:${selected.entity}:${selected.offset}:${attempt}`;
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    const params = new URLSearchParams({ offset: String(selected.offset), limit: "1000" });
    if (selected.entity) params.set("entity_id", selected.entity);
    request(apiBaseUrl, `/flow-runs/${encodeURIComponent(run.run_id)}/confidence?${params}`, { signal: controller.signal })
      .then((data) => { if (!controller.signal.aborted) setResponse({ key, data }); })
      .catch((failure) => { if (!controller.signal.aborted) setError(failure.message); });
    return () => controller.abort();
  }, [apiBaseUrl, run.run_id, selected.entity, selected.offset, attempt]);
  const data = response?.key === key ? response.data : null;
  const series = confidenceChartData(data?.points);
  return <section aria-label="Historial de confianza de la ejecución">
    <h3>Entidades de confianza</h3>
    <p className="fl-hint">Índice operativo por lectura, de 0 % a 100 %. Los huecos corresponden a lecturas no evaluables.
      {run.snapshot.execution_mode === "causal_future" && " El gráfico usa las fechas originales de las lecturas del CSV."}</p>
    {error ? <div className="fl-notice fl-error" role="alert"><span>{error}</span>
      <button onClick={() => setAttempt((value) => value + 1)}>Reintentar</button></div>
      : !data ? <p role="status">Cargando historial de confianza…</p>
        : !data.entities.length ? <p role="status">Esta ejecución no contiene entidades de confianza. Ejecuta un flujo con la versión actual.</p>
          : <>
            <label>Sensor de origen<select aria-label="Sensor del historial de confianza" value={data.source_entity_id || ""}
              onChange={(event) => setSelection({ runId: run.run_id, entity: event.target.value, offset: 0 })}>
              {data.entities.map((item) => <option key={item.source_entity_id} value={item.source_entity_id}>{item.source_entity_id}</option>)}
            </select></label>
            <p><strong>{data.entity_id}</strong></p>
            <div className="fl-run-chart">
              {series.timestamps.length ? <LineChart height={320}
                xAxis={[{ data: series.timestamps, scaleType: "time", label: "Fecha y hora" }]}
                yAxis={[{ min: 0, max: 100, label: "Confianza (%)", valueFormatter: (value) => `${value} %` }]}
                series={[{ data: series.values, label: "Confianza", curve: "stepAfter", showMark: false,
                  connectNulls: false, valueFormatter: (value) => value == null ? "No evaluable" : `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} %` }]}
                grid={{ horizontal: true }} /> : <p>No hay puntos en esta página.</p>}
            </div>
            <div className="fl-result-controls"><span>{data.total ? data.offset + 1 : 0}–{data.offset + data.points.length} de {data.total} lecturas</span>
              <button disabled={data.offset === 0} onClick={() => setSelection({ runId: run.run_id, entity: data.source_entity_id, offset: Math.max(0, data.offset - 1000) })}>Anterior</button>
              <button disabled={data.next_offset == null} onClick={() => setSelection({ runId: run.run_id, entity: data.source_entity_id, offset: data.next_offset })}>Siguiente</button>
            </div>
          </>}
  </section>;
}
