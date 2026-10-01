import { reviewMicros, formatReviewMicros } from "../review/review-time.js";

export function reviewDayRange(day, lower, upper) {
  const start = reviewMicros(`${day}T00:00:00Z`), lo = reviewMicros(lower), hi = reviewMicros(upper);
  if (start === null || lo === null || hi === null || lo > hi) return null;
  const end = start + 86400000000n - 1n;
  if (end < lo || start > hi) return null;
  return { start: formatReviewMicros(start < lo ? lo : start), end: formatReviewMicros(end > hi ? hi : end), partial: start < lo || end > hi };
}

export function firstObservationDay(start, end) {
  const a = reviewMicros(start), b = reviewMicros(end);
  if (a === null || b === null || a > b) return { start: "", end: "" };
  const day = 86400000000n;
  const midnight = a - ((a % day + day) % day);
  const last = midnight + day - 1n;
  return { start: formatReviewMicros(a), end: formatReviewMicros(last < b ? last : b) };
}

export function explorationRows(rows, search, kind, field, direction) {
  return rows.filter(row => row.entity_id.toLowerCase().includes(search.toLowerCase()) && (!kind || row.kind === kind))
    .sort((a, b) => {
      if (a[field] == null) return b[field] == null ? 0 : 1;
      if (b[field] == null) return -1;
      const order = typeof a[field] === "number" ? a[field] - b[field] : String(a[field]).localeCompare(String(b[field]));
      return order * direction;
    });
}
