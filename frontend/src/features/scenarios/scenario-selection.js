import { reviewMicros } from "../review/review-time.js";

export function scenarioConflicts(cases) {
  const entities = new Set();
  cases.forEach((item, index) => {
    const start = reviewMicros(item.start), end = reviewMicros(item.end);
    cases.slice(0, index).forEach(other => {
      const a = reviewMicros(other.start), b = reviewMicros(other.end);
      if (item.entity_id === other.entity_id && start !== null && end !== null && a !== null && b !== null && start <= b && a <= end) entities.add(item.entity_id);
    });
  });
  return [...entities];
}
