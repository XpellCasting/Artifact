import { useEffect, useState } from "react";
import { request } from "../../shared/api.js";

export function useHomeAssistantConnection(apiBaseUrl) {
  const [connection, setConnection] = useState(null);
  useEffect(() => {
    let cancelled = false;
    let timer;
    async function refresh() {
      try {
        const config = await request(apiBaseUrl, "/history/home-assistant/config");
        if (!cancelled) setConnection(config);
      } catch {
        if (!cancelled) setConnection((current) => current ? { ...current, bridge_connected: false } : null);
      } finally {
        if (!cancelled) timer = setTimeout(refresh, 10000);
      }
    }
    refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [apiBaseUrl]);
  return connection;
}
