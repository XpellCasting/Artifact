import React, { useEffect, useState } from "react";
import { AlertTriangle, FileSpreadsheet, House, Loader2, Play, Radio, Square } from "lucide-react";
import { homeAssistantConnectionLabel, useHomeAssistantConnection } from "../history/use-home-assistant-connection.js";
import { DECISIONS } from "./flow-lab-model.js";

export default function FutureEventsSource({ apiBaseUrl, busy, canRun, canUpload = canRun, dirty, job, error, replay, upload, onLabelledUpload, onEvaluate, onStartLive, onStopLive }) {
  const [source, setSource] = useState("csv");
  const [entities, setEntities] = useState("");
  const connection = useHomeAssistantConnection(apiBaseUrl);
  const connected = Boolean(connection?.bridge_connected);
  const live = job?.mode === "home_assistant" && job?.status === "running";
  const stopping = live && ["stopping", "saving"].includes(job?.phase);
  useEffect(() => { if (live) setSource("home_assistant"); }, [live]);

  async function start(event) {
    event.preventDefault();
    if (!connected || busy || !canRun || live) return;
    await onStartLive({
      entity_ids: entities.split(/[,\s]+/).map((value) => value.trim()).filter(Boolean) });
  }

  return <div className="fl-future-source" aria-label="Origen de los eventos futuros">
    <fieldset className="fl-event-sources" aria-label="Origen de los eventos" disabled={busy}>
      <label className={`fl-event-option ${source === "csv" ? "is-selected" : ""}`}><input type="radio" name="future-event-source" checked={source === "csv"} onChange={() => setSource("csv")} /><FileSpreadsheet size={22} /><span><strong>Archivo CSV</strong><small>Evalúa un conjunto etiquetado.</small></span></label>
      <label className={`fl-event-option ${source === "home_assistant" ? "is-selected" : ""}`}><input type="radio" name="future-event-source" checked={source === "home_assistant"} onChange={() => setSource("home_assistant")} /><House size={22} /><span><strong>Home Assistant en directo</strong><small>Evalúa los cambios de estado que lleguen.</small></span></label>
    </fieldset>
    {source === "csv" ? <div className="fl-future-csv">
      <div className="fl-upload-actions"><label className="file-action">Cargar conjunto etiquetado<input type="file" accept=".csv,text/csv" disabled={busy || !canUpload} onChange={onLabelledUpload} /></label></div>
      {replay?.loaded && <p><strong>{replay.filename}</strong> · {replay.rows} eventos · {replay.labelled ? "Con etiquetas" : "Sin etiquetas"}</p>}
      {upload && <p role="status">{upload.message}</p>}
      <button className="fl-primary" type="button" disabled={busy || !canRun || !replay?.loaded || !replay?.labelled} onClick={onEvaluate}><Play size={16} />{busy ? "Evaluando…" : dirty ? "Guardar y evaluar CSV" : "Evaluar CSV"}</button>
    </div> : <form className="fl-live-form" onSubmit={start}>
      <p>Recibe los cambios de las entidades del historial de referencia y evalúalos a medida que llegan. La recepción permanece activa aunque no lleguen eventos y reintenta la conexión si se corta. Al detenerla se guarda la ejecución.</p>
      {!live && <fieldset disabled={busy}>
        <div className="fl-live-status" role="status"><House size={18} /><strong>{homeAssistantConnectionLabel(connection)}</strong></div>
        {!connected && <p className="fl-hint">Configura la integración Artifact en Home Assistant y comprueba su conexión para iniciar la recepción.</p>}
        <label>Entidades (opcional)<input value={entities} onChange={(event) => setEntities(event.target.value)} placeholder="sensor.temperatura, binary_sensor.puerta" /><span className="fl-hint">Deja vacío para recibir todas las entidades compatibles del historial de referencia.</span></label>
      </fieldset>}
      {live ? <>
        <div className="fl-live-status" role="status" aria-live="polite"><Radio size={17} /><strong>{stopping ? "Deteniendo y guardando…" : job.polling_error ? "Recuperando comunicación con Artifact…" : job.connection?.status === "reconnecting" ? "Reconectando con Home Assistant…" : job.connection?.status === "connected" ? "Recibiendo eventos de Home Assistant" : job.phase === "preparation" ? "Preparando recepción…" : "Conectando con Home Assistant…"}</strong><span>{job.connection?.received || 0} recibidos · {job.processed || 0} evaluados</span></div>
        {job.polling_error && <p className="fl-required" role="status">No se pudo consultar el estado. Se conserva la sesión y se reintentará automáticamente. {job.polling_error}</p>}
        {!stopping && job.connection?.status === "reconnecting" && <p className="fl-required" role="status">{job.connection.last_error} Se reintentará automáticamente. La sesión continúa activa hasta que la detengas.</p>}
        {job.connection?.status === "connected" && !job.connection?.received && <p className="fl-hint">Esperando cambios de estado. La falta de eventos no detiene la recepción.</p>}
        {job.connection?.interruptions > 0 && <p className="fl-required">Se registraron {job.connection.interruptions} cortes de conexión. Puede haber eventos del corte que no se hayan recibido.</p>}
        {job.connection?.last_event_at && <p className="fl-hint">Último evento: {new Date(job.connection.last_event_at).toLocaleString()}</p>}
        {job.connection?.late > 0 && <p className="fl-required">{job.connection.late} eventos llegaron fuera de orden y quedaron fuera de la evaluación.</p>}
        <button type="button" disabled={stopping} onClick={onStopLive}>{stopping ? <Loader2 size={16} /> : <Square size={16} />}{stopping ? "Guardando ejecución…" : "Detener recepción y guardar"}</button>
        {!!job.recent_results?.length && <div className="fl-table-scroll"><table aria-label="Últimos resultados en directo"><thead><tr><th>Entidad</th><th>Fecha y hora</th><th>Resultado</th></tr></thead><tbody>{job.recent_results.slice(-20).map((item) => <tr key={item.observation_id}><td>{item.entity_id}</td><td>{new Date(item.timestamp).toLocaleString()}</td><td>{DECISIONS[item.final_decision] || "Sin conclusión"}</td></tr>)}</tbody></table></div>}
      </> : <button className="fl-primary" type="submit" disabled={busy || !canRun || !connected}><Radio size={16} />{dirty ? "Guardar e iniciar recepción" : "Iniciar recepción"}</button>}
    </form>}
    {error && <div className="fl-notice fl-error" role="alert"><AlertTriangle size={18} /><span>{error}</span></div>}
  </div>;
}
