export function isStatisticalSuspicion(result = {}) {
  return result.is_suspicious === true || result.passed_to_contextual_filter === true;
}

export function statisticalCounts(results = [], summary = {}) {
  const detected = results.filter(isStatisticalSuspicion);
  const suppressed = detected.filter((item) => item.suppressed_by_episode === true).length;
  return {
    detected: summary.grouping?.detected ?? summary.suspicious ?? detected.length,
    forwarded: summary.grouping?.forwarded ?? detected.length - suppressed,
    suppressed: summary.grouping?.suppressed ?? suppressed,
  };
}

export function episodeLabel(result = {}) {
  if (result.suppressed_by_episode) return `Repeticion agrupada · aparicion ${result.episode?.occurrences ?? "-"}`;
  if (result.episode) return "Primera sospecha · enviada a F2";
  return "";
}
