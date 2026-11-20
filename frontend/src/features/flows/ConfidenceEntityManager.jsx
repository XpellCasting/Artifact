import React, { useEffect, useState } from "react";
import { request } from "../../shared/api.js";

const PAGE_SIZE = 50;
const keyFor = (item) => `${item.source_mode}:${item.source_entity_id}`;

export default function ConfidenceEntityManager({ apiBaseUrl }) {
  const [open, setOpen] = useState(false);
  return <section className="fl-confidence-manager" aria-label="Gestión de sensores de confianza">
    <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
      {open ? "Cerrar gestión de sensores de confianza" : "Gestionar sensores de confianza"}
    </button>
    {open && <ConfidenceEntityList apiBaseUrl={apiBaseUrl} />}
  </section>;
}

function ConfidenceEntityList({ apiBaseUrl }) {
  const [entities, setEntities] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    request(apiBaseUrl, "/confidence/entities?compact=true", { signal: controller.signal })
      .then((data) => { if (!controller.signal.aborted) setEntities(data.entities); })
      .catch((failure) => { if (!controller.signal.aborted) setError(failure.message); });
    return () => controller.abort();
  }, [apiBaseUrl, attempt]);

  function refresh() {
    setEntities(null);
    setSelected(new Set());
    setPage(0);
    setError("");
    setAttempt((value) => value + 1);
  }

  const query = filter.trim().toLowerCase();
  const matches = (entities || []).filter((item) => item.source_entity_id.includes(query)
    || item.entity_id.includes(query));
  const visible = matches.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const selection = (entities || []).filter((item) => selected.has(keyFor(item)) && !item.available);

  function toggle(item, checked) {
    const key = keyFor(item);
    const next = new Set(selected);
    if (checked) next.add(key);
    else next.delete(key);
    setSelected(next);
  }

  async function remove() {
    setDeleting(true);
    setError("");
    setStatus("");
    try {
      const result = await request(apiBaseUrl, "/confidence/entities", {
        method: "DELETE",
        body: JSON.stringify({ entities: selection.map(({ source_entity_id, source_mode, revision }) => ({
          source_entity_id, source_mode, revision,
        })) }),
      });
      let message = result.removed === 1 ? "Se retiró un sensor de confianza."
        : `Se retiraron ${result.removed} sensores de confianza.`;
      if (result.synchronization === "pending") message += " La eliminación en Home Assistant está pendiente de sincronización.";
      if (result.synchronization === "integration_update_required") message += " Actualiza la integración Paranoid en Home Assistant para completar la eliminación.";
      setStatus(message);
      refresh();
    } catch (failure) {
      setError(failure.message);
    } finally {
      setConfirming(false);
      setDeleting(false);
    }
  }

  return <>
    {status && <p role="status">{status}</p>}
    {error && <p className="fl-required" role="alert">{error}</p>}
    {!entities && !error && <p role="status">Cargando sensores de confianza…</p>}
    <fieldset disabled={confirming || deleting} className="fl-fields">
      <div className="fl-confidence-actions">
        {entities?.length > 0 && <label>Buscar sensor<input type="search" value={filter}
          onChange={(event) => { setFilter(event.target.value); setPage(0); }} /></label>}
        <button type="button" onClick={refresh} disabled={!entities && !error}>Actualizar lista</button>
      </div>
      {entities?.length === 0 && <p>El catálogo de confianza está vacío.</p>}
      {entities?.length > 0 && <>
        <div className="fl-table-scroll">
          <table aria-label="Sensores de confianza">
            <thead><tr><th>Selección</th><th>Sensor de origen</th><th>Modo</th><th>Estado</th></tr></thead>
            <tbody>{visible.map((item) => {
              const mode = item.source_mode === "test" ? "Test" : "En directo";
              return <tr key={keyFor(item)}>
                <td><input type="checkbox" aria-label={`Seleccionar ${item.source_entity_id} · ${mode}`}
                  disabled={item.available} checked={selected.has(keyFor(item))}
                  onChange={(event) => toggle(item, event.target.checked)} /></td>
                <td><code>{item.source_entity_id}</code></td><td>{mode}</td>
                <td>{item.available ? "En evaluación" : "Inactivo"}</td>
              </tr>;
            })}</tbody>
          </table>
          {!matches.length && <p className="fl-empty-row">Ningún sensor coincide con la búsqueda.</p>}
        </div>
        <div className="fl-confidence-actions">
          <button type="button" disabled={!matches.some((item) => !item.available)}
            onClick={() => setSelected((previous) => new Set([...previous, ...matches.filter((item) => !item.available).map(keyFor)]))}>
            Seleccionar resultados inactivos
          </button>
          <button type="button" disabled={!selection.length} onClick={() => setSelected(new Set())}>Limpiar selección</button>
          {matches.length > PAGE_SIZE && <>
            <button type="button" disabled={page === 0} onClick={() => setPage(page - 1)}>Anterior</button>
            <span>{page * PAGE_SIZE + 1}–{page * PAGE_SIZE + visible.length} de {matches.length}</span>
            <button type="button" disabled={(page + 1) * PAGE_SIZE >= matches.length} onClick={() => setPage(page + 1)}>Siguiente</button>
          </>}
          <button className="fl-danger" type="button" disabled={!selection.length}
            onClick={() => setConfirming(true)}>Borrar selección ({selection.length})</button>
        </div>
      </>}
    </fieldset>
    {confirming && <div className="fl-confidence-confirm" role="group" aria-label="Confirmar borrado">
      <p>Se retirará la selección de Paranoid y Home Assistant. Los resultados de las ejecuciones guardadas seguirán disponibles.</p>
      <div className="fl-confidence-actions">
        <button className="fl-danger" type="button" disabled={deleting} onClick={remove}>
          {deleting ? "Borrando…" : "Confirmar borrado"}
        </button>
        <button type="button" disabled={deleting} onClick={() => setConfirming(false)}>Cancelar</button>
      </div>
    </div>}
  </>;
}
