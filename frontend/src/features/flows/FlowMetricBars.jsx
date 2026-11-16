import React from "react";
import { BarChart } from "@mui/x-charts/BarChart";

const tickLabelStyle = { fontFamily: "var(--ha-font-family-body, Roboto, Noto, Arial, sans-serif)" };

export default function FlowMetricBars({ title, labels, series, axis, height = 244 }) {
  return <div className="fl-run-chart">
    <BarChart
      title={title}
      layout="horizontal"
      height={height}
      skipAnimation
      grid={{ vertical: true }}
      margin={{ left: 0, right: 26, top: 4, bottom: 4 }}
      yAxis={[{ scaleType: "band", data: labels, width: 126, tickLabelStyle }]}
      xAxis={[{ min: 0, ...axis, tickLabelStyle }]}
      series={series}
    />
  </div>;
}
