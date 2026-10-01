import React, { useEffect, useState } from "react";
import { reviewUtcTimestamp } from "../review/review-time.js";

export default function InjectionFragmentPicker({ apiBaseUrl, base, entity, start, end, onStart, onEnd, disabled, Comparison }) {
  const [view, setView] = useState(null), [error, setError] = useState("");
  const [edge, setEdge] = useState("start");
  const key = JSON.stringify([base?.dataset_id, base?.annotation_id, base?.revision]);
  useEffect(() => {
    if (!base) return;
    const controller = new AbortController();
    setView(null); setError("");
    const parameters = new URLSearchParams({ entity_ids: base.entity_id, start: base.start, end: base.end, padding_minutes: "0", max_points: "600" });
    fetch(`${apiBaseUrl.replace(/\/$/, "")}/review/datasets/${encodeURIComponent(base.dataset_id)}/fragment?${parameters}`, { signal: controller.signal })
      .then(async response => { const data = await response.json(); if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "No se pudo cargar el fragmento."); return data; })
      .then(data => { if (!controller.signal.aborted) setView({ key, data }); })
      .catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [apiBaseUrl, key]);
  if (!base) return null;
  const series = view?.key === key ? view.data.series[0] : null;
  function select(timestamp) {
    if (disabled) return;
    const exact = reviewUtcTimestamp(timestamp);
    if (edge === "start") { onStart(exact); setEdge("end"); } else onEnd(exact);
  }
  return <section className="injection-fragment-picker">
    <h4>Elegir las fechas sobre el fragmento original</h4>
    <p className="review-help">Selecciona qué límite quieres fijar y pulsa una lectura en el gráfico. Después de elegir el inicio, el siguiente clic fija el fin. También puedes usar la lista o editar las fechas exactas debajo.</p>
    <div className="review-actions"><button type="button" className="review-button" aria-pressed={edge === "start"} disabled={disabled} onClick={() => setEdge("start")}>Fijar inicio</button><button type="button" className="review-button" aria-pressed={edge === "end"} disabled={disabled} onClick={() => setEdge("end")}>Fijar fin</button><button type="button" className="review-button" disabled={disabled} onClick={() => { onStart(reviewUtcTimestamp(base.start)); onEnd(reviewUtcTimestamp(base.end)); }}>Usar todo el fragmento revisado</button></div>
    <p className="review-help" aria-live="polite">Próximo clic: <strong>{edge === "start" ? "inicio" : "fin"}</strong><br />Inicio UTC: <strong>{start}</strong><br />Fin UTC: <strong>{end}</strong></p>
    {error && <p role="alert" className="review-inline-error">{error}</p>}
    {!series && !error && <p className="review-help">Cargando fragmento original…</p>}
    {series && <>
      <Comparison continuous={entity?.kind === "continuous"} onSelectTime={disabled ? undefined : select} title="Fragmento original · intervalo a intervenir sombreado"
        run={{ entity_id: base.entity_id, start, end, preview_sampled: series.sampled, preview: series.points.map((point, i) => ({ source_row: i, timestamp: point.timestamp, original_state: point.state, injected_state: point.state, changed: false })) }} originalOnly />
      <label>Lectura para fijar {edge === "start" ? "inicio" : "fin"} (UTC)<select value="" disabled={disabled} onChange={e => select(e.target.value)}><option value="" disabled>Selecciona una lectura</option>{series.points.map((point, i) => <option key={i} value={point.timestamp}>{reviewUtcTimestamp(point.timestamp)} · {point.state}</option>)}</select></label>
      {series.sampled && <p className="review-help">Vista muestreada: la lista y el gráfico contienen lecturas representativas. Los campos ISO permiten indicar cualquier instante; la inyección usa todos los registros del intervalo.</p>}
    </>}
  </section>;
}
