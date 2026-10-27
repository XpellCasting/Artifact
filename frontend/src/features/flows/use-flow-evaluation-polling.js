import { useEffect, useRef } from "react";
import { request } from "../../shared/api.js";

export function useFlowEvaluationPolling({ apiBaseUrl, jobId, discover = false, onUpdate, onComplete, onFailure, onConnectionError, onReady, onRunsChanged }) {
  const handlers = useRef({});
  useEffect(() => { handlers.current = { jobId, onUpdate, onComplete, onFailure, onConnectionError, onReady, onRunsChanged }; });
  const trackedJobId = discover ? null : jobId;
  useEffect(() => {
    if (!discover && !trackedJobId) return;
    let stopped = false;
    let timer;
    let handledTerminal;
    let runningId;
    let latestRunId;
    async function poll() {
      let state;
      const expectedJobId = handlers.current.jobId;
      try {
        const payload = await request(apiBaseUrl, discover ? "/flow-evaluations" : `/flow-evaluations/${trackedJobId}`, { cache: "no-store" });
        if (stopped) return;
        // A start/stop response may have arrived while this older snapshot was in flight.
        if (discover && expectedJobId !== handlers.current.jobId) {
          timer = setTimeout(poll, 1500);
          return;
        }
        handlers.current.onReady?.();
        if (discover) {
          state = payload.job;
          if (payload.latest_run_id !== latestRunId) {
            latestRunId = payload.latest_run_id;
            handlers.current.onRunsChanged?.(latestRunId);
          }
          if (state?.status === "completed" && !state.run && handledTerminal !== `${state.job_id}:${state.status}`) {
            // Large saved snapshots are fetched once, rather than on every discovery poll.
            state = await request(apiBaseUrl, `/flow-evaluations/${state.job_id}`, { cache: "no-store" });
          }
        } else state = payload;
      } catch (reason) {
        if (stopped) return;
        if (!discover && reason.status === 404) {
          handlers.current.onFailure("La API ya no conserva esta recepción. Comprueba si Artifact se reinició antes de iniciar otra sesión.");
        } else {
          handlers.current.onConnectionError(reason.message);
          timer = setTimeout(poll, 3000);
        }
        return;
      }
      if (stopped) return;
      if (discover && expectedJobId !== handlers.current.jobId) {
        timer = setTimeout(poll, 1500);
        return;
      }
      if (!state) {
        if (runningId || expectedJobId) handlers.current.onFailure("La API ya no conserva esta evaluación. Comprueba si Artifact se reinició; las ejecuciones terminadas siguen guardadas.");
        runningId = null;
      } else if (state.status === "failed" || state.status === "completed") {
        runningId = null;
        const terminalKey = `${state.job_id || trackedJobId}:${state.status}`;
        if (handledTerminal !== terminalKey) {
          handledTerminal = terminalKey;
          if (state.status === "failed") handlers.current.onFailure(state.error);
          else handlers.current.onComplete(state);
        }
      } else {
        runningId = state.job_id;
        handlers.current.onUpdate(state);
      }
      if (discover || state?.status === "running") timer = setTimeout(poll, 1500);
    }
    poll();
    return () => { stopped = true; clearTimeout(timer); };
  }, [trackedJobId, apiBaseUrl, discover]);
}
