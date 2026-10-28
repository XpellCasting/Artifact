import { useEffect, useState } from "react";
import { request } from "../../shared/api.js";

export function homeAssistantConnectionLabel(connection) {
  if (connection === null) return "Comprobando conexión";
  if (connection.bridge_connected) return "Conectada";
  return connection.bridge_configured === false ? "Integración no configurada" : "Desconectada";
}

export function useHomeAssistantConnection(apiBaseUrl) {
  const [connection, setConnection] = useState(null);
  useEffect(() => {
    let cancelled = false;
    let timer;
    setConnection(null);
    async function refresh() {
      try {
        const config = await request(apiBaseUrl, "/history/home-assistant/config");
        if (!cancelled) setConnection(config);
      } catch {
        if (!cancelled) setConnection((current) => ({ ...current, bridge_connected: false }));
      } finally {
        if (!cancelled) timer = setTimeout(refresh, 10000);
      }
    }
    refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [apiBaseUrl]);
  return connection;
}
