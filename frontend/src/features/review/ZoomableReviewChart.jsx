import React, { useEffect, useId, useRef, useState } from "react";
import { reviewMicros, formatReviewMicros } from "./review-time.js";
import { zoomRange } from "../../shared/chart-zoom.js";

export default function ZoomableReviewChart({ Plot, series, fragment, apiBaseUrl, datasetId, ...props }) {
  const origin = reviewMicros(fragment.view_start);
  const total = Math.max(1, Number(reviewMicros(fragment.view_end) - origin));
  const [range, setRange] = useState([0, total]);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const root = useRef(null);
  const drag = useRef(null);
  const [dragging, setDragging] = useState(false);
  const clipId = useId().replace(/:/g, "");
  const zoomed = range[0] > 0 || range[1] < total;
  const key = range.join(":");
  useEffect(() => {
    const element = root.current;
    function wheel(event) {
      const svg = event.target.closest?.("svg");
      if (!svg || !event.deltaY) return;
      const matrix = svg.getScreenCTM();
      if (!matrix) return;
      const cursor = svg.createSVGPoint();
      cursor.x = event.clientX; cursor.y = event.clientY;
      const position = cursor.matrixTransform(matrix.inverse());
      if (position.x < 100 || position.x > 976 || position.y < 15 || position.y > 148) return;
      event.preventDefault();
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 184 : 1);
      setRange(previous => zoomRange(previous, total, (position.x - 100) / 876, delta));
    }
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, [total]);
  useEffect(() => {
    function move(event) {
      if (!drag.current) return;
      const { startX, range: startRange, width } = drag.current;
      const shift = (event.clientX - startX) / width * (startRange[1] - startRange[0]);
      const span = startRange[1] - startRange[0];
      const start = Math.max(0, Math.min(total - span, startRange[0] - shift));
      setRange([start, start + span]);
    }
    function end() {
      if (!drag.current) return;
      drag.current = null;
      setDragging(false);
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
    window.addEventListener("pointercancel", end);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", end);
      window.removeEventListener("pointercancel", end);
    };
  }, [total]);
  function startPan(event) {
    if (event.button !== 0 || event.target.closest?.('[role="button"]')) return;
    const svg = event.target.closest?.("svg");
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    drag.current = { startX: event.clientX, range, width: rect.width * 876 / 1000 };
    setDragging(true);
    event.preventDefault();
  }
  useEffect(() => {
    if (!zoomed || !series.sampled || !datasetId || !apiBaseUrl) { setLoading(false); setError(""); return; }
    const controller = new AbortController();
    setLoading(true); setError("");
    const timer = setTimeout(async () => {
      try {
        const params = new URLSearchParams({ entity_ids: series.entity_id,
          start: formatReviewMicros(origin + BigInt(Math.floor(range[0]))),
          end: formatReviewMicros(origin + BigInt(Math.ceil(range[1]))), padding_minutes: "0", max_points: "1200" });
        const response = await fetch(`${apiBaseUrl.replace(/\/$/, "")}/review/datasets/${datasetId}/fragment?${params}`, { signal: controller.signal });
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "No se pudieron cargar las lecturas del zoom.");
        if (!controller.signal.aborted) setDetail({ key, series: data.series.find(item => item.entity_id === series.entity_id) });
      } catch (exception) { if (!controller.signal.aborted) setError(exception.message); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    }, 220);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [key, zoomed, series.sampled, series.entity_id, datasetId, apiBaseUrl, origin]);
  return <div ref={root} className={`review-zoomable ${dragging ? "is-dragging" : ""}`}>
    <div className="review-zoom-toolbar"><span>Rueda: acercar o alejar · Arrastra la gráfica con el mouse para desplazarte lateralmente.</span><button type="button" className="review-button" disabled={!zoomed} onClick={() => setRange([0, total])}>Ver intervalo completo</button></div>
    <div className="review-pan-surface" onPointerDown={startPan}>
      <Plot {...props} series={zoomed && detail?.key === key && detail.series ? { ...detail.series, states: series.states } : series} fragment={fragment} viewport={range} clipId={clipId} />
    </div>
    {loading && <p className="review-help" role="status">Cargando lecturas del área ampliada…</p>}
    {error && <p className="review-inline-error" role="alert">{error} Se muestra la muestra disponible.</p>}
  </div>;
}
