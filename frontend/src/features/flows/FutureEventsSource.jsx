import React, { useEffect, useState } from "react";
import { FileSpreadsheet, House, Loader2, Play, Radio, Square } from "lucide-react";
import { useHomeAssistantConnection } from "../history/use-home-assistant-connection.js";
import { DECISIONS } from "./flow-lab-model.js";

export default function FutureEventsSource({ apiBaseUrl, busy, canRun, canUpload = canRun, dirty, job, replay, upload, onUpload, onLabelledUpload, onEvaluate, onStartLive, onStopLive }) {
  const [source, setSource] = useState("csv");
  const [url, setUrl] = useState("http://homeassistant.local:8123");
  const [configuredUrl, setConfiguredUrl] = useState("");
  const [tokenConfigured, setTokenConfigured] = useState(false);
  const [token, setToken] = useState("");
  const [entities, setEntities] = useState("");
  const connection = useHomeAssistantConnection(apiBaseUrl);
  const integration = Boolean(connection?.bridge_configured || connection?.bridge_connected);
  const live = job?.mode === "home_assistant" && job?.status === "running";
  const stopping = live && ["stopping", "saving"].includes(job?.phase);
  const configured = tokenConfigured && url.trim().replace(/\/$/, "") === configuredUrl.replace(/\/$/, "");
  useEffect(() => {
    if (connection?.url) {
      setUrl(connection.url); setConfiguredUrl(connection.url); setTokenConfigured(connection.token_configured);
    }
  }, [connection?.url, connection?.token_configured]);
  useEffect(() => { if (live) setSource("home_assistant"); }, [live]);

  async function start(event) {
    event.preventDefault();
    const started = await onStartLive({ ...(integration ? { connection_mode: "integration", url: "", token: "" } : { url: url.trim(), token: token.trim() }),
      entity_ids: entities.split(/[,\s]+/).map((value) => value.trim()).filter(Boolean) });
    if (started) setToken("");
  }

  return <div className="fl-future-source" aria-label="Origen de los eventos futuros">
    <fieldset className="fl-event-sources" disabled={busy}><legend>Elige cómo recibir los eventos</legend>
      <label className={`fl-event-option ${source === "csv" ? "is-selected" : ""}`}><input type="radio" name="future-event-source" checked={source === "csv"} onChange={() => setSource("csv")} /><FileSpreadsheet size={22} /><span><strong>Archivo CSV</strong><small>Evalúa un conjunto de eventos existente.</small></span></label>
      <label className={`fl-event-option ${source === "home_assistant" ? "is-selected" : ""}`}><input type="radio" name="future-event-source" checked={source === "home_assistant"} onChange={() => setSource("home_assistant")} /><House size={22} /><span><strong>Home Assistant en directo</strong><small>Evalúa los cambios de estado que lleguen.</small></span></label>
    </fieldset>
    {source === "csv" ? <div className="fl-future-csv">
      <p>Se recorre todo el CSV en orden temporal. Cada ejecución comienza con una copia nueva del estado del historial de referencia.</p>
      <div className="fl-upload-actions"><label className="file-action">Cargar eventos futuros<input type="file" accept=".csv,text/csv" disabled={busy || !canUpload} onChange={onUpload} /></label><label className="file-action">Cargar conjunto etiquetado<input type="file" accept=".csv,text/csv" disabled={busy || !canUpload} onChange={onLabelledUpload} /></label></div>
      <p className="fl-hint">Columnas: entity_id, state, last_changed. Un conjunto etiquetado añade es_sintetica, tipo y tipo_falla; las etiquetas sirven para comprobar los resultados.</p>
      {replay?.loaded && <p><strong>{replay.filename}</strong> · {replay.rows} eventos · {replay.labelled ? "Con etiquetas" : "Sin etiquetas"}</p>}
      {upload && <p role="status">{upload.message}</p>}
      <button className="fl-primary" type="button" disabled={busy || !canRun || !replay?.loaded} onClick={onEvaluate}><Play size={16} />{busy ? "Evaluando…" : dirty ? "Guardar y evaluar CSV" : "Evaluar CSV"}</button>
    </div> : <form className="fl-live-form" onSubmit={start}>
      <p>Recibe los cambios de las entidades del historial de referencia y evalúalos a medida que llegan. El flujo conserva su estado entre eventos. Al detener la recepción se guarda la ejecución.</p>
      {!live && <fieldset disabled={busy}>
        {integration ? <div className="fl-live-status" role="status"><House size={18} /><strong>{connection.bridge_connected ? "Conectado mediante la integración de Home Assistant" : "Esperando la conexión de Home Assistant"}</strong></div> : <details open={!configured || undefined}><summary>Conexión con Home Assistant{configured ? " · configurada en el servidor" : ""}</summary><div className="fl-live-fields">
          <label>URL de Home Assistant<input type="url" required value={url} onChange={(event) => setUrl(event.target.value)} /></label>
          <label>Token de acceso<input type="password" autoComplete="off" required={!configured} value={token} onChange={(event) => setToken(event.target.value)} placeholder={configured ? "Se usará el token configurado" : "Token de larga duración"} /></label>
        </div></details>}
        <label>Entidades (opcional)<input value={entities} onChange={(event) => setEntities(event.target.value)} placeholder="sensor.temperatura, binary_sensor.puerta" /><span className="fl-hint">Deja vacío para recibir todas las entidades compatibles del historial de referencia.</span></label>
      </fieldset>}
      {live ? <>
        <div className="fl-live-status" role="status" aria-live="polite"><Radio size={17} /><strong>{stopping ? "Deteniendo y guardando…" : job.connection?.status === "connected" ? "Recibiendo eventos de Home Assistant" : "Conectando con Home Assistant…"}</strong><span>{job.connection?.received || 0} recibidos · {job.processed || 0} evaluados</span></div>
        {job.connection?.last_event_at && <p className="fl-hint">Último evento: {new Date(job.connection.last_event_at).toLocaleString()}</p>}
        {job.connection?.late > 0 && <p className="fl-required">{job.connection.late} eventos llegaron fuera de orden y quedaron fuera de la evaluación.</p>}
        <button type="button" disabled={stopping} onClick={onStopLive}>{stopping ? <Loader2 size={16} /> : <Square size={16} />}{stopping ? "Guardando ejecución…" : "Detener recepción y guardar"}</button>
        {!!job.recent_results?.length && <div className="fl-table-scroll"><table aria-label="Últimos resultados en directo"><thead><tr><th>Entidad</th><th>Fecha y hora</th><th>Resultado</th></tr></thead><tbody>{job.recent_results.slice(-20).map((item) => <tr key={item.observation_id}><td>{item.entity_id}</td><td>{new Date(item.timestamp).toLocaleString()}</td><td>{DECISIONS[item.final_decision] || "Sin conclusión"}</td></tr>)}</tbody></table></div>}
      </> : <button className="fl-primary" type="submit" disabled={busy || !canRun || (integration && !connection.bridge_connected)}><Radio size={16} />{dirty ? "Guardar e iniciar recepción" : "Iniciar recepción"}</button>}
    </form>}
  </div>;
}
