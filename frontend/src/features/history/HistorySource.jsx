import React, { useEffect, useState } from "react";
import { CheckCircle2, Database, FileSpreadsheet, House, Loader2, Upload } from "lucide-react";
import { request } from "../../shared/api.js";
import { homeAssistantConnectionLabel, useHomeAssistantConnection } from "./use-home-assistant-connection.js";
import { formatCount, formatHistoryRange } from "../../shared/analysis.jsx";
import { HistoryLoadProgress, previewHistoryUpload } from "./history-runtime.jsx";
import SensorSelection from "./SensorSelection.jsx";
import "./history-source.css";

function localDateTime(date) {
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

export default function HistorySource({ apiBaseUrl, status, job, busy, onImport, onPrepareCsv }) {
  const [source, setSource] = useState("home_assistant");
  const [editing, setEditing] = useState(!status?.exists);
  const connection = useHomeAssistantConnection(apiBaseUrl);
  const connected = Boolean(connection?.bridge_connected);
  const [start, setStart] = useState(() => localDateTime(new Date(Date.now() - 30 * 86400000)));
  const [end, setEnd] = useState(() => localDateTime(new Date()));
  const [error, setError] = useState("");
  const [working, setWorking] = useState(false);
  const [progress, setProgress] = useState("");
  const [catalog, setCatalog] = useState(null);
  const loading = working || busy || ["queued", "running", "uploading"].includes(job?.status);
  const ready = Boolean(status?.exists && status?.future_evaluation_ready);
  useEffect(() => { if (status?.exists) setEditing(false); }, [status?.exists, status?.baseline_id]);
  useEffect(() => { if (status?.selection_required) { setSource("csv"); setEditing(true); } }, [status?.selection_required]);

  async function inspectHomeAssistant(event) {
    event.preventDefault();
    if (!connected || loading) return;
    const from = new Date(start), until = new Date(end);
    if (!Number.isFinite(from.getTime()) || !Number.isFinite(until.getTime()) || from >= until) {
      setError("La fecha final debe ser posterior a la inicial."); return;
    }
    if (until > new Date()) { setError("Selecciona un período de historial que termine en el pasado."); return; }
    const input = { start: from.toISOString(), end: until.toISOString(), entity_ids: [] };
    setWorking(true); setError(""); setProgress("Consultando sensores disponibles…");
    try {
      const preview = await request(apiBaseUrl, "/history/home-assistant/entities", { method: "POST", body: JSON.stringify(input) });
      setCatalog({ ...preview, request: input });
    } catch (reason) { setError(reason.message); }
    finally { setWorking(false); setProgress(""); }
  }

  async function inspectCsv(event) {
    const input = event.target;
    const file = input.files?.[0];
    if (!file) return;
    setWorking(true); setError(""); setProgress(`Subiendo ${file.name}…`);
    try {
      const preview = await previewHistoryUpload(file, (loaded, total) => setProgress(loaded >= total
        ? "Identificando sensores del CSV, sin preparar filtros…" : `Subiendo CSV: ${Math.round(100 * loaded / Math.max(1, total))} %`), apiBaseUrl);
      setCatalog(preview);
    } catch (reason) { setError(reason.message); }
    finally { input.value = ""; setWorking(false); setProgress(""); }
  }

  async function inspectSaved() {
    setWorking(true); setError(""); setProgress("Identificando sensores del CSV guardado…");
    try { setCatalog(await request(apiBaseUrl, "/history/selection/saved", { method: "POST" })); }
    catch (reason) { setError(reason.message); }
    finally { setWorking(false); setProgress(""); }
  }

  async function prepare(entityIds) {
    if (!entityIds.length || loading || (catalog.source === "home_assistant" && !connected)) return;
    setWorking(true); setError("");
    try {
      const result = catalog.source === "home_assistant"
        ? await onImport({ ...catalog.request, entity_ids: entityIds })
        : await onPrepareCsv(catalog.candidate_id, entityIds);
      if (!result) { setError("No se pudieron preparar los filtros. Revisa el aviso de la carga e inténtalo de nuevo."); return; }
      setCatalog(null); setEditing(false);
    } catch (reason) { setError(reason.message); }
    finally { setWorking(false); }
  }

  return <section className="history-source" aria-label="Origen del historial de referencia">
    <div className="hs-heading"><div><span className="hs-step">Datos de referencia</span><h3>Selector de Historial</h3><p>Elige el historial y los sensores antes de preparar las bases estadística y contextual.</p></div>
      {status?.exists && <button type="button" className="hs-secondary" disabled={loading} onClick={() => setEditing(!editing)} aria-expanded={editing}>{editing ? "Cerrar selección" : "Cambiar historial"}</button>}
    </div>
    {status?.exists && <div className={`hs-current ${ready ? "is-ready" : ""}`} role="status">
      {ready ? <CheckCircle2 size={20} /> : <Database size={20} />}<div><strong>{status.history_origin === "home_assistant" ? "Historial de Home Assistant" : "Historial CSV"}</strong><span>{formatCount(status.rows)} lecturas · {formatCount(status.entity_count)} entidades · {formatHistoryRange(status)}</span></div><span className="hs-badge">{ready ? "Listo para evaluar" : "Preparando filtros"}</span>
    </div>}
    {status?.selection_required && <p className="hs-capacity" role="status">Hay un CSV guardado pendiente de selección. Revisa sus sensores y confirma cuáles estarán activos antes de preparar las bases de los filtros.</p>}
    {editing && <>
      <fieldset className="hs-options" disabled={loading}><legend>Origen de los datos</legend>
        <label className={`hs-option ${source === "home_assistant" ? "is-selected" : ""}`}><input type="radio" name="history-source" value="home_assistant" checked={source === "home_assistant"} onChange={() => { setSource("home_assistant"); setCatalog(null); setError(""); }} /><House size={24} /><span><strong>Home Assistant</strong><small>Consulta tus sensores y elige cuáles importar.</small></span></label>
        <label className={`hs-option ${source === "csv" ? "is-selected" : ""}`}><input type="radio" name="history-source" value="csv" checked={source === "csv"} onChange={() => { setSource("csv"); setCatalog(null); setError(""); }} /><FileSpreadsheet size={24} /><span><strong>Archivo CSV</strong><small>Revisa los sensores de un histórico existente.</small></span></label>
      </fieldset>
      {source === "home_assistant" && <div className={`hs-current ${connected ? "is-ready" : ""}`} role="status"><House size={20} /><div><strong>{homeAssistantConnectionLabel(connection)}</strong><span>{connected ? "La integración proporciona los sensores y el historial." : "Configura la integración Artifact en Home Assistant y comprueba su conexión para continuar."}</span></div></div>}
      {catalog ? <SensorSelection catalog={catalog} busy={loading} prepareDisabled={catalog.source === "home_assistant" && !connected} onPrepare={prepare} onBack={() => { setCatalog(null); setError(""); }} /> : source === "home_assistant" ? <form className="hs-form" onSubmit={inspectHomeAssistant}>
        <fieldset disabled={loading}>
          <div className="hs-fields"><label>Desde<input type="datetime-local" required value={start} onChange={(event) => setStart(event.target.value)} /></label><label>Hasta<input type="datetime-local" required value={end} onChange={(event) => setEnd(event.target.value)} /></label></div>
          <small>Fechas según la zona horaria de tu navegador. El catálogo muestra las entidades compatibles disponibles actualmente; las lecturas dependen del período elegido.</small>
          <div className="hs-actions"><button className="hs-primary" type="submit" disabled={!connected}><Database size={16} />Consultar sensores</button><small>Primero eliges los sensores; después confirmas la preparación.</small></div>
        </fieldset>
      </form> : <div className="hs-csv"><FileSpreadsheet size={30} /><div><strong>Selecciona tu histórico en CSV</strong><p>Columnas requeridas: <code>entity_id</code>, <code>state</code> y <code>last_changed</code>.</p><small>Se leerá el catálogo de sensores. Los filtros se prepararán después de confirmar tu selección.</small></div><label className="hs-file"><Upload size={16} /><span>Seleccionar CSV</span><input type="file" accept=".csv,text/csv" aria-label="Seleccionar CSV del historial" disabled={loading} onChange={inspectCsv} /></label>
        {(status?.real_history_available || status?.exists) && <button type="button" className="hs-secondary" disabled={loading} onClick={inspectSaved}><Database size={16} />Revisar CSV guardado en el servidor</button>}
      </div>}
      {status?.exists && <p className="hs-replace">La referencia activa se reemplazará únicamente al confirmar «Preparar filtros».</p>}
    </>}
    {progress && <p className="hs-preview-progress" role="status"><Loader2 size={16} className="hs-spin" />{progress}</p>}
    {error && <p className="hs-error" role="alert">{error}</p>}
    {job && <HistoryLoadProgress job={job} />}
  </section>;
}
