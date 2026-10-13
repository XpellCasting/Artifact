import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Download,
  X,
  Database,
  Eye,
  FlaskConical,
  GitBranch,
  Loader2,
  RotateCcw,
  Zap,
} from "lucide-react";
import { downloadCsv } from "./csv.js";
import ReviewWorkspace from "../features/review/ReviewWorkspace.jsx";
import DiagnosticWorkspace from "../features/diagnostics/DiagnosticWorkspace.jsx";
import FlowLab from "../features/flows/FlowLab.jsx";
import { isStatisticalSuspicion, statisticalCounts, episodeLabel } from "./statistical-routing.js";
import { isEnvironmentalResult, contextRoutingLabel, contextRoutingReasonLabel, contextRoutingCounts, contextCoherenceLabel, contextObservationLabel, preciseContextTimestamp, contextCommonCause } from "./context-routing.js";

import { splitDateTime } from "./analysis.jsx";

export function Metric({ icon, label, value, detail }) {
  return (
    <article className="metric">
      <div className="metric-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value ?? "-"}</strong>
        <small>{detail || "-"}</small>
      </div>
    </article>
  );
}


export function ModuleTab({ active, icon, label, onClick }) {
  return (
    <button className={`module-tab ${active ? "active" : ""}`} type="button" onClick={onClick}>
      {icon}
      <span>{label}</span>
    </button>
  );
}


export function Panel({ title, children }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {children}
    </section>
  );
}


export function StepButton({ id, busy, onClick, icon, label, disabled = false }) {
  return (
    <button className="secondary-action inline" onClick={onClick} disabled={disabled || Boolean(busy)}>
      {busy === id ? <Loader2 className="spin" size={17} /> : icon}
      {label}
    </button>
  );
}


export function SummaryRow({ label, value }) {
  return (
    <div className="summary-row">
      <span>{label}</span>
      <strong>{value ?? "-"}</strong>
    </div>
  );
}


export function Empty({ text }) {
  return <div className="empty">{text}</div>;
}


export function TimeStack({ value }) {
  const parts = splitDateTime(value);
  return (
    <span className="time-stack">
      <strong>{parts.date}</strong>
      <small>{parts.time}</small>
    </span>
  );
}


export function KeyValueGrid({ rows }) {
  if (!rows.length) {
    return <Empty text="Sin datos disponibles." />;
  }
  return (
    <div className="key-value-grid">
      {rows.map((row) => (
        <React.Fragment key={row.label}>
          <span>{row.label}</span>
          <strong>{row.value}</strong>
        </React.Fragment>
      ))}
    </div>
  );
}

