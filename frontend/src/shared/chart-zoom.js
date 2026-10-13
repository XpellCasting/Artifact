// Relative microseconds avoid losing nearby observations to epoch rounding.
export function zoomRange(range, total, anchor, delta) {
  const ratio = Math.max(0, Math.min(1, anchor));
  const span = range[1] - range[0];
  const nextSpan = Math.max(Math.min(1, total), Math.min(total, span * Math.exp(Math.max(-1, Math.min(1, delta * 0.002)))));
  const at = range[0] + span * ratio;
  const start = Math.max(0, Math.min(total - nextSpan, at - nextSpan * ratio));
  return [start, start + nextSpan];
}
