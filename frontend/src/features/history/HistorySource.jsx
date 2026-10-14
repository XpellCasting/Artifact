import React, { useEffect, useState } from "react";
import { CheckCircle2, Database, FileSpreadsheet, House, Loader2, Upload } from "lucide-react";
import { useHomeAssistantConnection } from "./use-home-assistant-connection.js";
import { formatCount, formatHistoryRange } from "../../shared/analysis.jsx";
import { HistoryLoadProgress } from "./history-runtime.jsx";
import "./history-source.css";

function localDateTime(date) {
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

export default function HistorySource({ apiBaseUrl, status, job, busy, onImport, onUpload, onRestore }) {
  const [source, setSource] = useState("home_assistant");
  const [editing, setEditing] = useState(!status?.exists);
  const [url, setUrl] = useState("http://homeassistant.local:8123");
  const [token, setToken] = useState("");
  const [tokenConfigured, setTokenConfigured] = useState(false);
  const [configuredUrl, setConfiguredUrl] = useState("");
  const connection = useHomeAssistantConnection(apiBaseUrl);
  const integration = Boolean(connection?.bridge_configured || connection?.bridge_connected);
  const [start, setStart] = useState(() => localDateTime(new Date(Date.now() - 30 * 86400000)));
  const [end, setEnd] = useState(() => localDateTime(new Date()));
  const [entities, setEntities] = useState("");
  const [error, setError] = useState("");
  const loading = busy || ["queued", "running", "uploading"].includes(job?.status);
  const ready = Boolean(status?.exists && status?.future_evaluation_ready);
  const useConfiguredToken = tokenConfigured && url.trim().replace(/\/$/, "") === configuredUrl.replace(/\/$/, "");

  useEffect(() => {
    if (connection?.url) {
      setUrl(connection.url);
      setConfiguredUrl(connection.url);
      setTokenConfigured(connection.token_configured);
    }
  }, [connection?.url, connection?.token_configured]);
  useEffect(() => { if (status?.exists) setEditing(false); }, [status?.exists, status?.baseline_id]);

  async function importHistory(event) {
    event.preventDefault();
    const from = new Date(start), until = new Date(end);
    if (!Number.isFinite(from.getTime()) || !Number.isFinite(until.getTime()) || from >= until) {
      setError("La fecha final debe ser posterior a la inicial.");
      return;
    }
    if (until > new Date()) {
      setError("Selecciona un período de historial que termine en el pasado.");
      return;
    }
    setError("");
    const imported = await onImport({ ...(integration ? { connection_mode: "integration", url: "", token: "" } : { url: url.trim(), token: token.trim() }), start: from.toISOString(), end: until.toISOString(),
      entity_ids: entities.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean) });
    if (imported) { setToken(""); setEditing(false); }
  }

  return <section className="history-source" aria-label="Origen del historial de referencia">
    <div className="hs-heading"><div><span className="hs-step">Datos de referencia</span><h3>Selector de Historial</h3><p>Usa los datos guardados en Home Assistant o carga un CSV para preparar los filtros del flujo.</p></div>
      {status?.exists && <button type="button" className="hs-secondary" disabled={loading} onClick={() => setEditing(!editing)} aria-expanded={editing}>{editing ? "Cerrar selección" : "Cambiar historial"}</button>}
    </div>
    {status?.exists && <div className={`hs-current ${ready ? "is-ready" : ""}`} role="status">
      {ready ? <CheckCircle2 size={20} /> : <Database size={20} />}<div><strong>{status.history_origin === "home_assistant" ? "Historial de Home Assistant" : "Historial CSV"}</strong><span>{formatCount(status.rows)} lecturas · {formatCount(status.entity_count)} entidades · {formatHistoryRange(status)}</span></div><span className="hs-badge">{ready ? "Listo para evaluar" : "Preparando filtros"}</span>
    </div>}
    {editing && <>
      <fieldset className="hs-options" disabled={loading}><legend>Origen de los datos</legend>
        <label className={`hs-option ${source === "home_assistant" ? "is-selected" : ""}`}><input type="radio" name="history-source" value="home_assistant" checked={source === "home_assistant"} onChange={() => { setSource("home_assistant"); setError(""); }} /><House size={24} /><span><strong>Home Assistant</strong><small>Importa el historial de tus entidades.</small></span></label>
        <label className={`hs-option ${source === "csv" ? "is-selected" : ""}`}><input type="radio" name="history-source" value="csv" checked={source === "csv"} onChange={() => { setSource("csv"); setError(""); }} /><FileSpreadsheet size={24} /><span><strong>Archivo CSV</strong><small>Usa un histórico que ya tengas.</small></span></label>
      </fieldset>
      {source === "home_assistant" ? <form className="hs-form" onSubmit={importHistory}>
        <fieldset disabled={loading}>
          {integration ? <div className={`hs-current ${connection.bridge_connected ? "is-ready" : ""}`} role="status"><House size={20} /><div><strong>{connection.bridge_connected ? "Conectado mediante la integración de Home Assistant" : "Esperando la conexión de Home Assistant"}</strong><span>{connection.bridge_connected ? "La integración proporciona el historial automáticamente." : "Comprueba que la integración Artifact esté cargada en Home Assistant."}</span></div></div> : <details className="hs-connection" open={!useConfiguredToken || undefined}><summary>Conexión con Home Assistant{useConfiguredToken ? " · configurada en el servidor" : ""}</summary><div className="hs-fields">
            <label>URL de Home Assistant<input type="url" required value={url} onChange={(event) => setUrl(event.target.value)} placeholder="http://homeassistant.local:8123" /></label>
            <label>Token de acceso<input type="password" autoComplete="off" required={!useConfiguredToken} value={token} onChange={(event) => setToken(event.target.value)} placeholder={useConfiguredToken ? "Se usará el token configurado" : "Token de larga duración"} /><small>Disponible en tu perfil de Home Assistant → Seguridad. El token introducido se usa para esta importación.</small></label>
          </div></details>}
          <div className="hs-fields"><label>Desde<input type="datetime-local" required value={start} onChange={(event) => setStart(event.target.value)} /></label><label>Hasta<input type="datetime-local" required value={end} onChange={(event) => setEnd(event.target.value)} /></label></div>
          <small>Fechas según la zona horaria de tu navegador.</small>
          <label>Entidades (opcional)<input value={entities} onChange={(event) => setEntities(event.target.value)} placeholder="sensor.temperatura, binary_sensor.puerta" /><small>Separa los identificadores con comas. Si lo dejas vacío, se importan todas las entidades compatibles.</small></label>
          <div className="hs-actions"><button className="hs-primary" type="submit" disabled={integration && !connection.bridge_connected}>{loading ? <Loader2 size={16} className="hs-spin" /> : <Database size={16} />}{loading ? "Importando historial…" : "Importar historial"}</button><small>Se consulta el historial disponible para el período seleccionado.</small></div>
        </fieldset>
      </form> : <div className="hs-csv"><FileSpreadsheet size={30} /><div><strong>Carga tu histórico en CSV</strong><p>Columnas requeridas: <code>entity_id</code>, <code>state</code> y <code>last_changed</code>.</p><small>Este archivo se usará como referencia para preparar los filtros. Los eventos futuros se cargan en el paso de evaluación.</small></div><label className="hs-file"><Upload size={16} /><span>Seleccionar CSV</span><input type="file" accept=".csv,text/csv" aria-label="Seleccionar CSV del historial" disabled={loading} onChange={onUpload} /></label>
        {status?.real_history_available && !status?.exists && <button type="button" className="hs-secondary" disabled={loading} onClick={onRestore}><Database size={16} />Usar CSV guardado en el servidor</button>}
      </div>}
      {status?.exists && <p className="hs-replace">Al importar otro historial se reemplaza la referencia activa y se preparan de nuevo los filtros.</p>}
    </>}
    {error && <p className="hs-error" role="alert">{error}</p>}
    {job && <HistoryLoadProgress job={job} />}
  </section>;
}
