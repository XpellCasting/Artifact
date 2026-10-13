import React, { useEffect, useMemo, useState } from "react";
import { firstObservationDay, explorationRows, reviewDayRange } from "./exploration.js";
import DailyActivity from "./DailyActivity.jsx";
import { validReviewInterval } from "../review/review-time.js";
import "./history-explorer.css";

const kinds = { binary: "Binaria", continuous: "Continua", multistate: "Multiestado" };
const number = value => value == null ? "—" : Number(value).toLocaleString("es-CL", { maximumFractionDigits: 3 });
const date = value => value ? value.replace("T", " ").replace(/\+00:00$/, "Z").replace(/(?:\.\d+)?Z$/, " UTC") : "—";
const gap = value => value == null ? "—" : value >= 86400 ? `${number(value / 86400)} días` : value >= 3600 ? `${number(value / 3600)} h` : `${number(value)} s`;
const columns = [["entity_id", "Entidad"], ["kind", "Tipo"], ["rows", "Registros"], ["valid_rows", "Válidos"], ["invalid_rows", "Inválidos"], ["start", "Primera observación UTC"], ["end", "Última observación UTC"], ["distinct_states", "Estados distintos"], ["transitions", "Cambios"], ["max_report_gap_seconds", "Mayor intervalo"]];

async function get(base, path, signal) {
  const response = await fetch(`${base.replace(/\/$/, "")}${path}`, { signal });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Error HTTP ${response.status}`);
  return data;
}

export default function HistoryExplorer({ apiBaseUrl, dataset, onReview, disabled, Chart, DateInput, completedEntities = {}, onToggleEntityComplete }) {
  const [summary, setSummary] = useState(null), [error, setError] = useState("");
  const [search, setSearch] = useState(""), [kind, setKind] = useState("");
  const [sort, setSort] = useState({ field: "entity_id", direction: 1 });
  const [selected, setSelected] = useState(null), [range, setRange] = useState({ start: "", end: "" });
  const [detail, setDetail] = useState(null), [busy, setBusy] = useState(false);
  const [query, setQuery] = useState(null);
  const [selectedDay, setSelectedDay] = useState("");
  const base = `/review/datasets/${encodeURIComponent(dataset.dataset_id)}`;
  const selectionKey = JSON.stringify([selected?.entity_id, range.start, range.end]);
  const visibleDetail = detail?.key === selectionKey ? detail.data : null;
  useEffect(() => {
    const controller = new AbortController();
    get(apiBaseUrl, `${base}/exploration`, controller.signal).then(setSummary)
      .catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [apiBaseUrl, base]);
  useEffect(() => {
    if (!query) return;
    const controller = new AbortController();
    setBusy(true); setError("");
    const parameters = new URLSearchParams({ entity_id: query.entity, start: query.start, end: query.end });
    get(apiBaseUrl, `${base}/exploration/entity?${parameters}`, controller.signal)
      .then(data => { if (!controller.signal.aborted) setDetail({ key: query.key, data }); })
      .catch(e => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [apiBaseUrl, base, query]);
  const rows = useMemo(() => explorationRows(summary?.entities || [], search, kind, sort.field, sort.direction), [summary, search, kind, sort]);
  function select(entity) {
    const initial = firstObservationDay(entity.start, entity.end);
    setSelected(entity); setRange(initial); setDetail(null);
    setSelectedDay("");
    setQuery({ entity: entity.entity_id, ...initial, key: JSON.stringify([entity.entity_id, initial.start, initial.end]) });
  }
  function selectDay(day) {
    const next = reviewDayRange(day, dataset.start, dataset.end);
    if (!selected || disabled || !next) return;
    setSelectedDay(day); setRange({ start: next.start, end: next.end });
    setQuery({ entity: selected.entity_id, start: next.start, end: next.end, key: JSON.stringify([selected.entity_id, next.start, next.end]) });
    onReview({ datasetId: dataset.dataset_id, entityId: selected.entity_id, start: next.start, end: next.end, day, partialDay: next.partial });
  }
  function sortBy(field) { setSort(s => ({ field, direction: s.field === field ? -s.direction : 1 })); }
  const activity = selected && detail?.data?.entity_id === selected.entity_id ? detail.data.daily : [];
  return <section className="review-panel history-explorer" aria-label="Explorar histórico">
    <div className="review-panel-title"><div><span className="review-eyebrow">Exploración manual</span><h3>Explorar histórico</h3></div><span className="review-badge">{summary ? `${summary.entities.length} entidades · ${number(summary.indexed_rows)} registros` : "Preparando resumen…"}</span></div>
    <p className="review-help">Compara todo el archivo y elige dónde revisar. Un intervalo sin reportes no confirma una desconexión. Los cambios no se cuentan a través de un reporte inválido.</p>
    {error && <p className="review-alert" role="alert">{error}</p>}
    {summary?.rejected_rows > 0 && <p className="review-notice">{number(summary.rejected_rows)} filas no indexables; están conservadas en el original y excluidas del resumen por entidad.</p>}
    <div className="exploration-tools"><label>Buscar entidad<input value={search} onChange={e => setSearch(e.target.value)} placeholder="Nombre de la entidad" /></label><label>Tipo<select value={kind} onChange={e => setKind(e.target.value)}><option value="">Todos los tipos</option>{Object.entries(kinds).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label></div>
    {summary && <div className="review-table-scroll exploration-table"><table><thead><tr>{columns.map(([field, label]) => <th key={field} aria-sort={sort.field === field ? sort.direction === 1 ? "ascending" : "descending" : "none"}><button type="button" onClick={() => sortBy(field)}>{label}{sort.field === field ? sort.direction === 1 ? " ↑" : " ↓" : ""}</button></th>)}<th>Acción</th></tr></thead><tbody>{rows.map(row => <tr key={row.entity_id} className={row.entity_id === selected?.entity_id ? "exploration-selected" : ""}>{columns.map(([field]) => <td key={field} title={field === "max_report_gap_seconds" && row.gap_start ? `${date(row.gap_start)} → ${date(row.gap_end)}` : undefined}>{field === "entity_id" ? row[field] : field === "kind" ? kinds[row.kind] : field === "start" || field === "end" ? date(row[field]) : field === "max_report_gap_seconds" ? gap(row[field]) : number(row[field])}</td>)}<td className="exploration-actions"><button className="review-button" disabled={disabled} onClick={() => select(row)}>Explorar<span className="exploration-sr"> {row.entity_id}</span></button><label className="exploration-complete" title="Marcar entidad revisada por completo"><input type="checkbox" aria-label={`Entidad ${row.entity_id} revisada por completo`} checked={Boolean(completedEntities[row.entity_id])} disabled={disabled} onChange={event => onToggleEntityComplete?.(row.entity_id, event.target.checked)} /><span>✓</span></label></td></tr>)}</tbody></table>{!rows.length && <p className="review-empty">No hay entidades que coincidan con la búsqueda.</p>}</div>}
    {selected && <div className="exploration-detail">
      <div className="review-panel-title"><h3>{selected.entity_id}</h3><span className="review-badge">{kinds[selected.kind]}</span></div>
      {!!activity?.length && <DailyActivity activity={activity} selectedDay={selectedDay} onSelect={selectDay} disabled={disabled} />}
      {selectedDay && reviewDayRange(selectedDay, dataset.start, dataset.end)?.partial && <p className="review-help">Día parcial: el intervalo se limita a la cobertura temporal del archivo.</p>}
      <details><summary>Ajustar el intervalo de exploración</summary>
      <div className="exploration-range"><DateInput label="Inicio de exploración" value={range.start} onChange={start => setRange(r => ({ ...r, start }))} disabled={disabled} /><DateInput label="Fin de exploración" value={range.end} onChange={end => setRange(r => ({ ...r, end }))} disabled={disabled} /><button className="review-button" disabled={disabled || busy || !validReviewInterval(range.start, range.end)} onClick={() => setQuery({ entity: selected.entity_id, ...range, key: selectionKey })}>Actualizar intervalo</button></div>
      </details>
      {busy && <p role="status" className="review-help">Calculando detalle…</p>}
      {!visibleDetail && !busy && <p className="review-help">Actualiza el intervalo para ver sus datos.</p>}
      {visibleDetail && <><p className="review-help">Intervalo seleccionado: {number(visibleDetail.rows)} registros · {number(visibleDetail.valid_rows)} válidos · {number(visibleDetail.invalid_rows)} inválidos.</p>
        {selected.kind === "continuous" ? <div className="exploration-statistics">{[["Mínimo", "min"], ["Mediana", "median"], ["Máximo", "max"]].map(([label, field]) => <div key={field}><span>{label}</span><strong>{number(visibleDetail.numeric[field])} {selected.unit}</strong></div>)}</div> : <><h4>Frecuencia por categoría · intervalo seleccionado</h4><p className="review-help">Porcentaje de los registros válidos del intervalo, no del tiempo. Estados originales conservados.</p><div className="review-table-scroll exploration-frequency"><table><thead><tr><th>Estado original</th><th>Registros</th><th>% de registros válidos</th></tr></thead><tbody>{visibleDetail.frequencies.slice(0, 100).map(item => <tr key={item.state}><td>{item.state}</td><td>{number(item.rows)}</td><td>{number(item.percent)} %</td></tr>)}</tbody></table>{visibleDetail.frequencies.length > 100 && <p>Se muestran las 100 categorías más frecuentes de {number(visibleDetail.frequencies.length)}.</p>}{!visibleDetail.frequencies.length && <p className="review-help">Sin categorías válidas en este intervalo.</p>}</div></>}
        {visibleDetail.fragment.series.map(series => <Chart key={series.entity_id} series={series} fragment={visibleDetail.fragment} target={false} exploration />)}
        {visibleDetail.fragment.series.some(series => series.sampled) && <p className="review-help">Gráfico muestreado. Los resúmenes usan todas las filas del intervalo. {visibleDetail.fragment.sampling_note}</p>}
        <button className="review-button review-button-primary" disabled={disabled} onClick={() => onReview({ datasetId: dataset.dataset_id, entityId: selected.entity_id, start: visibleDetail.start, end: visibleDetail.end })}>Revisar este intervalo</button>
      </>}
    </div>}
  </section>;
}
