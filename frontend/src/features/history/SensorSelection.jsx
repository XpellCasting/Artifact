import React, { useId, useMemo, useState } from "react";

export const SENSOR_PAGE_SIZE = 50;
const KINDS = { continuous: "Continuo", binary: "Binario", multistate: "Multivalor" };
const count = (value) => Number(value).toLocaleString("es-CL");
const normalize = (value) => String(value || "").normalize("NFD").replace(/\p{M}/gu, "").toLocaleLowerCase();
const UNASSIGNED = "__unassigned__";

function SensorList({ entities, selectedIds, setActive, page, setPage, label }) {
  const pages = Math.max(1, Math.ceil(entities.length / SENSOR_PAGE_SIZE));
  const current = Math.min(page, pages - 1);
  const rows = entities.slice(current * SENSOR_PAGE_SIZE, (current + 1) * SENSOR_PAGE_SIZE);
  return <>
    <ul className="hs-sensor-list" aria-label={label}>{rows.map((entity) => <li key={entity.entity_id} data-testid="history-sensor-row"><label className="hs-sensor-row"><input type="checkbox" checked={selectedIds.has(entity.entity_id)} aria-label={`Analizar ${entity.entity_id}`} onChange={(event) => setActive([entity.entity_id], event.target.checked)} /><span><strong>{entity.friendly_name || entity.entity_id}</strong>{entity.friendly_name && <small>{entity.entity_id}</small>}<small>{KINDS[entity.sensor_kind] || "Por determinar"} · {entity.domain}{entity.unit_of_measurement ? ` · ${entity.unit_of_measurement}` : ""}{entity.rows != null ? ` · ${count(entity.rows)} lecturas` : ""}</small></span></label></li>)}</ul>
    {!rows.length && <p>No hay sensores que coincidan con estos filtros.</p>}
    <div className="hs-pagination"><button type="button" aria-label={`Página anterior de ${label}`} disabled={current === 0} onClick={() => setPage(current - 1)}>Página anterior</button><span>Página {count(current + 1)} de {count(pages)} · {count(entities.length)} resultados</span><button type="button" aria-label={`Página siguiente de ${label}`} disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}>Página siguiente</button></div>
  </>;
}

export default function SensorSelection({ catalog, busy, prepareDisabled, onPrepare, onBack }) {
  const entities = catalog.entities;
  const homeAssistant = catalog.source === "home_assistant";
  const areasAvailable = homeAssistant && catalog.areas_available === true;
  const [selectedIds, setSelectedIds] = useState(() => new Set(homeAssistant ? [] : entities.map((entity) => entity.entity_id)));
  const [search, setSearch] = useState("");
  const [area, setArea] = useState("");
  const [domain, setDomain] = useState("");
  const [kind, setKind] = useState("");
  const [visibility, setVisibility] = useState("");
  const [pages, setPages] = useState({});
  const [openAreas, setOpenAreas] = useState(() => new Set());
  const panelId = useId();
  const domains = useMemo(() => [...new Set(entities.map((entity) => entity.domain))].filter(Boolean).sort(), [entities]);
  const areas = useMemo(() => {
    if (!areasAvailable) return [];
    const groups = new Map();
    for (const entity of entities) {
      if (!entity.area_id) continue;
      if (!groups.has(entity.area_id)) groups.set(entity.area_id, { id: entity.area_id, name: entity.area_name, entities: [] });
      groups.get(entity.area_id).entities.push(entity);
    }
    return [...groups.values()].sort((a, b) => a.name.localeCompare(b.name, "es"));
  }, [entities, areasAvailable]);
  const selected = useMemo(() => entities.filter((entity) => selectedIds.has(entity.entity_id)), [entities, selectedIds]);
  const filtered = useMemo(() => {
    const query = normalize(search.trim());
    return entities.filter((entity) => (!query || normalize(`${entity.entity_id} ${entity.friendly_name || ""} ${areasAvailable ? entity.area_name || "" : ""}`).includes(query))
      && (!area || (area === UNASSIGNED ? !entity.area_id : entity.area_id === area))
      && (!domain || entity.domain === domain) && (!kind || entity.sensor_kind === kind)
      && (!visibility || selectedIds.has(entity.entity_id) === (visibility === "active")));
  }, [entities, search, area, domain, kind, visibility, selectedIds, areasAvailable]);
  const filteredAreas = useMemo(() => {
    const groups = new Map();
    for (const entity of filtered) {
      const id = entity.area_id || UNASSIGNED;
      if (!groups.has(id)) groups.set(id, []);
      groups.get(id).push(entity);
    }
    return groups;
  }, [filtered]);
  function setActive(ids, active) {
    setSelectedIds((previous) => {
      const next = new Set(previous);
      for (const id of ids) { if (active) next.add(id); else next.delete(id); }
      return next;
    });
  }
  function filter(setter, value) { setter(value); setPages({}); }
  function list(rows, id, label) {
    return <SensorList entities={rows} selectedIds={selectedIds} setActive={setActive} page={pages[id] || 0} setPage={(value) => setPages((previous) => ({ ...previous, [id]: value }))} label={label} />;
  }

  return <div className="hs-sensors" aria-label="Selección de sensores">
    <div className="hs-sensors-heading"><div><h4>Elige los sensores que se analizarán</h4><p>{homeAssistant ? "Sensores disponibles en Home Assistant. El historial se descargará después de confirmar." : catalog.filename}</p></div><button type="button" disabled={busy} onClick={onBack}>Volver al origen</button></div>
    {catalog.request && <p>Período: {new Date(catalog.request.start).toLocaleString()} a {new Date(catalog.request.end).toLocaleString()}.</p>}
    <p className="hs-selection-summary" role="status"><strong>{count(selected.length)} de {count(entities.length)} sensores activos</strong>. {homeAssistant ? "Elige al menos un sensor para continuar." : "Todos están activos inicialmente."} Los desactivados quedan fuera de las bases estadística y contextual.</p>
    {homeAssistant && !areasAvailable && <p className="hs-capacity" role="status">No se pudieron consultar las áreas. Se muestra el catálogo sin agrupación; actualiza la integración Artifact y revisa su conexión para disponer de estos metadatos.</p>}
    {catalog.invalid_rows > 0 && <p>{count(catalog.invalid_rows)} filas del CSV no tienen entidad o fecha válidas y se omitirán.</p>}
    <fieldset disabled={busy}>
      <div className="hs-sensor-filters">
        <label>Buscar sensores<input type="search" value={search} onChange={(event) => filter(setSearch, event.target.value)} placeholder={areasAvailable ? "Nombre, entity_id o área" : "Nombre o entity_id"} /></label>
        {homeAssistant && <label>Área<select aria-label="Área" value={area} disabled={!areasAvailable} onChange={(event) => filter(setArea, event.target.value)}><option value="">Todas</option>{areas.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}{areasAvailable && <option value={UNASSIGNED}>Sin área asignada</option>}</select></label>}
        <label>Dominio<select value={domain} onChange={(event) => filter(setDomain, event.target.value)}><option value="">Todos los dominios</option>{domains.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        <label>Tipo<select value={kind} onChange={(event) => filter(setKind, event.target.value)}><option value="">Todos los tipos</option>{Object.entries(KINDS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label>Estado de selección<select value={visibility} onChange={(event) => filter(setVisibility, event.target.value)}><option value="">Activos e inactivos</option><option value="active">Solo activos</option><option value="inactive">Solo inactivos</option></select></label>
      </div>
      <div className="hs-bulk-actions"><button type="button" disabled={selected.length === entities.length} onClick={() => setSelectedIds(new Set(entities.map((entity) => entity.entity_id)))}>Activar todos</button><button type="button" disabled={!selected.length} onClick={() => setSelectedIds(new Set())}>Desactivar todos</button><button type="button" disabled={!filtered.length} onClick={() => setActive(filtered.map((entity) => entity.entity_id), true)}>Activar resultados</button><button type="button" disabled={!filtered.length} onClick={() => setActive(filtered.map((entity) => entity.entity_id), false)}>Desactivar resultados</button></div>
      <small>Las acciones sobre resultados afectan a los {count(filtered.length)} sensores que coinciden con los filtros, incluidas las otras páginas.</small>
      {areasAvailable && areas.length > 0 ? <>
        {areas.map((item) => {
          const matches = filteredAreas.get(item.id) || [];
          if (!matches.length) return null;
          const open = openAreas.has(item.id);
          const active = item.entities.reduce((total, entity) => total + Number(selectedIds.has(entity.entity_id)), 0);
          return <section className="hs-area" key={item.id} aria-label={`Área ${item.name}`}>
            <button type="button" className="hs-area-toggle" aria-expanded={open} aria-controls={`${panelId}-${item.id}`} onClick={() => setOpenAreas((previous) => { const next = new Set(previous); if (open) next.delete(item.id); else next.add(item.id); return next; })}><span aria-hidden="true">{open ? "▾" : "▸"}</span><strong>{item.name}</strong><span>{count(active)} de {count(item.entities.length)} seleccionados · {count(matches.length)} resultados</span></button>
            {open && <div className="hs-area-content" id={`${panelId}-${item.id}`}>
              <div className="hs-bulk-actions"><button type="button" onClick={() => setActive(matches.map((entity) => entity.entity_id), true)}>Activar resultados de {item.name}</button><button type="button" onClick={() => setActive(matches.map((entity) => entity.entity_id), false)}>Desactivar resultados de {item.name}</button></div>
              {list(matches, item.id, `Sensores de ${item.name}`)}
            </div>}
          </section>;
        })}
        {!!filteredAreas.get(UNASSIGNED)?.length && <section className="hs-unassigned" aria-label="Sin área asignada"><h4>Sin área asignada</h4>{list(filteredAreas.get(UNASSIGNED), UNASSIGNED, "Sensores sin área asignada")}</section>}
        {!filtered.length && <p>No hay sensores que coincidan con estos filtros.</p>}
      </> : <>{areasAvailable && <h4>Sin área asignada</h4>}{list(filtered, "all", "Sensores disponibles")}</>}
      {selected.length >= 500 && <p className="hs-capacity" role="status">Has seleccionado {count(selected.length)} sensores. El filtro contextual puede comparar hasta {count(selected.length * (selected.length - 1) / 2)} pares, antes de considerar retardos. Prepararlos todos puede requerir mucho tiempo y memoria. Puedes reducir la selección por área, tipo, dominio o búsqueda.</p>}
      <div className="hs-actions"><button type="button" className="hs-primary" disabled={!selected.length || prepareDisabled} onClick={() => onPrepare(selected.map((entity) => entity.entity_id))}>{busy ? "Preparando filtros…" : `Preparar filtros con ${count(selected.length)} sensores`}</button><small>La preparación empieza únicamente al pulsar este botón. Mostrar el catálogo no crea las bases.</small></div>
      {!selected.length && <p className="hs-error" role="status">Activa al menos un sensor para continuar.</p>}
    </fieldset>
  </div>;
}
