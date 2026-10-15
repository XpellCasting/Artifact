import React from "react";
import { AlertTriangle } from "lucide-react";
import { DECISIONS, SIGNAL_LABELS } from "./flow-lab-model.js";
import { flowOutputAlert } from "./flow-output-alert.js";

export default function FlowOutputResult({ item, alertEnabled = true }) {
  const alert = alertEnabled ? flowOutputAlert(item) : null;
  if (!alert) return <>{DECISIONS[item.final_decision] || SIGNAL_LABELS[item.signal] || "No evaluable"}</>;
  return <span className={`fl-output-alert fl-output-alert-${alert.severity}`} role="alert" aria-atomic="true">
    <AlertTriangle size={17} aria-hidden="true" /><strong>{alert.message}</strong>
  </span>;
}
