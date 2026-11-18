export function confidencePercent(value) {
  return Number.isFinite(value) && value >= 0 && value <= 1
    ? `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 2 })} %`
    : "No evaluable";
}

export function confidenceChartData(points = []) {
  const ordered = points.map((point, index) => ({ ...point, time: new Date(point.timestamp), index }))
    .filter((point) => Number.isFinite(point.time.getTime()))
    .sort((left, right) => left.time - right.time || left.index - right.index);
  return {
    timestamps: ordered.map((point) => point.time),
    values: ordered.map((point) => Number.isFinite(point.value) && point.value >= 0 && point.value <= 100 ? point.value : null),
  };
}
