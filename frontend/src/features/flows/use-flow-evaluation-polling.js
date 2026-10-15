import { useEffect, useRef } from "react";
import { request } from "../../shared/api.js";

export function useFlowEvaluationPolling({ apiBaseUrl, jobId, onUpdate, onComplete, onFailure, onConnectionError }) {
  const handlers = useRef({});
  useEffect(() => { handlers.current = { onUpdate, onComplete, onFailure, onConnectionError }; });
  useEffect(() => {
    if (!jobId) return;
    let stopped = false;
    let timer;
    async function poll() {
      let state;
      try {
        state = await request(apiBaseUrl, `/flow-evaluations/${jobId}`);
      } catch (reason) {
        if (stopped) return;
        if (reason.status === 404) {
          handlers.current.onFailure("La API ya no conserva esta recepción. Comprueba si Artifact se reinició antes de iniciar otra sesión.");
        } else {
          handlers.current.onConnectionError(reason.message);
          timer = setTimeout(poll, 3000);
        }
        return;
      }
      if (stopped) return;
      if (state.status === "failed") handlers.current.onFailure(state.error);
      else if (state.status === "completed") handlers.current.onComplete(state);
      else {
        handlers.current.onUpdate(state);
        timer = setTimeout(poll, 1500);
      }
    }
    poll();
    return () => { stopped = true; clearTimeout(timer); };
  }, [jobId, apiBaseUrl]);
}
