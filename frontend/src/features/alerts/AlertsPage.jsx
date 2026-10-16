import React, { useEffect, useRef, useState } from "react";
import { Bell, RotateCcw } from "lucide-react";
import { request } from "../../shared/api.js";
import FlowOutputResult from "../flows/FlowOutputResult.jsx";
import { flowOutputAlert } from "../flows/flow-output-alert.js";
import { connectHomeAssistantTheme } from "./home-assistant-theme.js";
import "../flows/flow-lab.css";
import "./alerts-page.css";

export default function AlertsPage({ apiBaseUrl }) {
  const [page, setPage] = useState({ items: [], next_before_id: null });
  const [beforeId, setBeforeId] = useState(null);
  const [severity, setSeverity] = useState("all");
  const executionId = useRef(undefined);
  const [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => connectHomeAssistantTheme(), []);

  useEffect(() => {
    const previousTitle = document.title;
    document.title = "Artifact-Alertas";
    return () => { document.title = previousTitle; };
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer;
    const controller = new AbortController();
    setLoading(true);
    setPage({ items: [], next_before_id: null });
    setError("");
    async function refresh() {
      try {
        const data = await request(apiBaseUrl, `/flow-alerts?limit=50${beforeId ? `&before_id=${beforeId}` : ""}${severity === "all" ? "" : `&severity=${severity}`}`, { signal: controller.signal });
        if (stopped) return;
        if (data.execution_id !== executionId.current) {
          const changed = executionId.current !== undefined;
          executionId.current = data.execution_id;
          if (changed && beforeId !== null) {
            setPage({ items: [], next_before_id: null });
            setBeforeId(null);
            return;
          }
        }
        setPage(data); setError("");
      } catch (reason) {
        if (stopped) return;
        setError(reason.message);
      } finally {
        if (!stopped) {
          setLoading(false);
          timer = setTimeout(refresh, 3000);
        }
      }
    }
    refresh();
    return () => { stopped = true; controller.abort(); clearTimeout(timer); };
  }, [apiBaseUrl, beforeId, revision, severity]);

  return <main className="artifact-alerts">
    <header className="artifact-alerts-header"><div><h1>Artifact-Alertas</h1>
      <p>Alertas de la última ejecución del flujo.</p></div>
      <button type="button" disabled={loading} onClick={() => { setBeforeId(null); setRevision((value) => value + 1); }}><RotateCcw size={17} />Actualizar</button>
    </header>
    <div className="artifact-alerts-toolbar">
      <label className="artifact-alerts-filter">Mostrar<select aria-label="Tipo de alerta" value={severity} onChange={(event) => { setBeforeId(null); setSeverity(event.target.value); }}>
        <option value="all">Ambas</option><option value="anomaly">Anomalías</option><option value="fault">Alertas de falla</option>
      </select></label>
      <span>{beforeId === null ? "Actualización automática cada 3 segundos" : "Consultando alertas anteriores"}</span>
      {beforeId !== null && <button type="button" disabled={loading} onClick={() => setBeforeId(null)}>Volver a las recientes</button>}
    </div>
    {error && <p className="artifact-alerts-error" role="alert">No se pudieron actualizar las alertas. {error}</p>}
    {loading && <p role="status">Cargando alertas…</p>}
    {!loading && !error && !page.items.length && <div className="artifact-alerts-empty"><Bell size={30} /><p>{severity === "anomaly" ? "No hay anomalías." : severity === "fault" ? "No hay alertas de falla." : "No hay alertas de salida."}</p></div>}
    <div className="artifact-alerts-list" aria-label="Alertas de salida">
      {page.items.map(({ alert_id, result, created_at }) => {
        const alert = flowOutputAlert(result);
        if (!alert) return null;
        return <article key={alert_id} className={`artifact-alert-card artifact-alert-card-${alert.severity}`}>
          <div className="artifact-alert-heading"><h2>{result.entity_id}</h2><time dateTime={result.timestamp || created_at}>{new Date(result.timestamp || created_at).toLocaleString()}</time></div>
          <FlowOutputResult item={result} />
        </article>;
      })}
    </div>
    {page.next_before_id !== null && <button className="artifact-alerts-older" type="button" disabled={loading} onClick={() => setBeforeId(page.next_before_id)}>Ver alertas anteriores</button>}
  </main>;
}
