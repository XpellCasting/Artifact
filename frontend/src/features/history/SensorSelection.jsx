import React, { useMemo, useState } from "react";

export const SENSOR_PAGE_SIZE = 50;
const KINDS = { continuous: "Continuo", binary: "Binario", multistate: "Multivalor" };
const count = (value) => Number(value).toLocaleString("es-CL");

export default function SensorSelection({ catalog, busy, onPrepare, onBack }) {
  const [excluded, setExcluded] = useState(() => new Set());
  const [search, setSearch] = useState("");
  const [domain, setDomain] = useState("");
  const [kind, setKind] = useState("");
  const [visibility, setVisibility] = useState("");
  const [page, setPage] = useState(0);
  const entities = catalog.entities;
  const domains = useMemo(() => [...new Set(entities.map((entity) => entity.domain))].filter(Boolean).sort(), [entities]);
  const selected = useMemo(() => entities.filter((entity) => !excluded.has(entity.entity_id)), [entities, excluded]);
  const filtered = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return entities.filter((entity) => (!query || `${entity.entity_id} ${entity.friendly_name || ""}`.toLocaleLowerCase().includes(query))
      && (!domain || entity.domain === domain) && (!kind || entity.sensor_kind === kind)
      && (!visibility || excluded.has(entity.entity_id) === (visibility === "inactive")));
  }, [entities, search, domain, kind, visibility, excluded]);
  const pages = Math.max(1, Math.ceil(filtered.length / SENSOR_PAGE_SIZE));
  const current = Math.min(page, pages - 1);
  const rows = filtered.slice(current * SENSOR_PAGE_SIZE, (current + 1) * SENSOR_PAGE_SIZE);
  function setActive(ids, active) {
    setExcluded((previous) => {
      const next = new Set(previous);
      for (const id of ids) { if (active) next.delete(id); else next.add(id); }
      return next;
    });
  }
  function filter(setter, value) { setter(value); setPage(0); }

  return <div className="hs-sensors" aria-label="Selección de sensores">
    <div className="hs-sensors-heading"><div><h4>Elige los sensores que se analizarán</h4><p>{catalog.source === "home_assistant" ? "Sensores disponibles en Home Assistant. El historial se descargará después de confirmar." : catalog.filename}</p></div><button type="button" disabled={busy} onClick={onBack}>Volver al origen</button></div>
    {catalog.request && <p>Período: {new Date(catalog.request.start).toLocaleString()} a {new Date(catalog.request.end).toLocaleString()}.</p>}
    <p className="hs-selection-summary" role="status"><strong>{count(selected.length)} de {count(entities.length)} sensores activos</strong>. Todos están activos inicialmente. Los desactivados quedan fuera de las bases estadística y contextual.</p>
    {catalog.invalid_rows > 0 && <p>{count(catalog.invalid_rows)} filas del CSV no tienen entidad o fecha válidas y se omitirán.</p>}
    <fieldset disabled={busy}>
      <div className="hs-sensor-filters">
        <label>Buscar sensores<input type="search" value={search} onChange={(event) => filter(setSearch, event.target.value)} placeholder="Nombre o entity_id" /></label>
        <label>Dominio<select value={domain} onChange={(event) => filter(setDomain, event.target.value)}><option value="">Todos los dominios</option>{domains.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        <label>Tipo<select value={kind} onChange={(event) => filter(setKind, event.target.value)}><option value="">Todos los tipos</option>{Object.entries(KINDS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label>Estado de selección<select value={visibility} onChange={(event) => filter(setVisibility, event.target.value)}><option value="">Activos e inactivos</option><option value="active">Solo activos</option><option value="inactive">Solo inactivos</option></select></label>
      </div>
      <div className="hs-bulk-actions"><button type="button" disabled={!excluded.size} onClick={() => setExcluded(new Set())}>Activar todos</button><button type="button" disabled={!selected.length} onClick={() => setExcluded(new Set(entities.map((entity) => entity.entity_id)))}>Desactivar todos</button><button type="button" disabled={!filtered.length} onClick={() => setActive(filtered.map((entity) => entity.entity_id), true)}>Activar resultados</button><button type="button" disabled={!filtered.length} onClick={() => setActive(filtered.map((entity) => entity.entity_id), false)}>Desactivar resultados</button></div>
      <small>Las acciones sobre resultados afectan a los {count(filtered.length)} sensores que coinciden con los filtros, incluidas las otras páginas.</small>
      <ul className="hs-sensor-list" aria-label="Sensores disponibles">{rows.map((entity) => <li key={entity.entity_id} data-testid="history-sensor-row"><label className="hs-sensor-row"><input type="checkbox" checked={!excluded.has(entity.entity_id)} aria-label={`Analizar ${entity.entity_id}`} onChange={(event) => setActive([entity.entity_id], event.target.checked)} /><span><strong>{entity.friendly_name || entity.entity_id}</strong>{entity.friendly_name && <small>{entity.entity_id}</small>}<small>{KINDS[entity.sensor_kind] || "Por determinar"} · {entity.domain}{entity.unit_of_measurement ? ` · ${entity.unit_of_measurement}` : ""}{entity.rows != null ? ` · ${count(entity.rows)} lecturas` : ""}</small></span></label></li>)}</ul>
      {!rows.length && <p>No hay sensores que coincidan con estos filtros.</p>}
      <div className="hs-pagination"><button type="button" disabled={current === 0} onClick={() => setPage(current - 1)}>Página anterior</button><span>Página {count(current + 1)} de {count(pages)} · {count(filtered.length)} resultados</span><button type="button" disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}>Página siguiente</button></div>
      {selected.length >= 500 && <p className="hs-capacity" role="status">Has seleccionado {count(selected.length)} sensores. El filtro contextual puede comparar hasta {count(selected.length * (selected.length - 1) / 2)} pares, antes de considerar retardos. Prepararlos todos puede requerir mucho tiempo y memoria. Puedes reducir la selección por tipo, dominio o búsqueda.</p>}
      <div className="hs-actions"><button type="button" className="hs-primary" disabled={!selected.length} onClick={() => onPrepare(selected.map((entity) => entity.entity_id))}>{busy ? "Preparando filtros…" : `Preparar filtros con ${count(selected.length)} sensores`}</button><small>La preparación empieza únicamente al pulsar este botón. Mostrar el catálogo no crea las bases.</small></div>
      {!selected.length && <p className="hs-error" role="status">Activa al menos un sensor para continuar.</p>}
    </fieldset>
  </div>;
}
