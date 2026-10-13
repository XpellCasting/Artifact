// Selection and export boundaries retain the source's microsecond precision.
export function reviewMicros(value) {
  const match = String(value || "").match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})(?::(\d{2}))?(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})$/i);
  if (!match || /[1-9]/.test((match[4] || "").slice(6))) return null;
  const calendar = `${match[1]}T${match[2]}:${match[3] || "00"}`;
  const calendarDate = new Date(`${calendar}Z`);
  if (!Number.isFinite(calendarDate.getTime()) || calendarDate.toISOString().slice(0, 19) !== calendar) return null;
  const seconds = Date.parse(`${calendar}${match[5]}`);
  if (!Number.isFinite(seconds)) return null;
  return BigInt(seconds) * 1000n + BigInt((match[4] || "").slice(0, 6).padEnd(6, "0"));
}

export function formatReviewMicros(value) {
  const remainder = ((value % 1000000n) + 1000000n) % 1000000n;
  const seconds = (value - remainder) / 1000000n;
  return `${new Date(Number(seconds * 1000n)).toISOString().slice(0, 19)}.${String(remainder).padStart(6, "0")}Z`;
}

export function reviewUtcTimestamp(value) {
  if (!value) return "";
  const input = /(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : `${value}Z`;
  const micros = reviewMicros(input);
  return micros === null ? "" : formatReviewMicros(micros);
}

export function validReviewInterval(start, end) {
  const a = reviewMicros(start);
  const b = reviewMicros(end);
  return a !== null && b !== null && a <= b;
}

export function initialReviewRange(start, end) {
  const a = reviewMicros(start);
  const b = reviewMicros(end);
  if (a === null || b === null || a > b) return { start: "", end: "" };
  return { start: formatReviewMicros(a), end: formatReviewMicros(a + 3600000000n < b ? a + 3600000000n : b) };
}

export function shiftReviewRange(start, end, lower, upper, direction) {
  if (!validReviewInterval(start, end) || !validReviewInterval(lower, upper)) return null;
  const a = reviewMicros(start);
  const b = reviewMicros(end);
  const lo = reviewMicros(lower);
  const hi = reviewMicros(upper);
  const duration = b - a;
  if (a < lo || b > hi || duration === 0n) return null;
  const proposed = a + BigInt(direction) * duration;
  const next = proposed < lo ? lo : proposed + duration > hi ? hi - duration : proposed;
  if (next === a) return null;
  return { start: formatReviewMicros(next), end: formatReviewMicros(next + duration) };
}
