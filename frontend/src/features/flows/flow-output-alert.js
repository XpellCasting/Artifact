import { FAULT_TYPES } from "../diagnostics/diagnostic-data.js";

export function flowOutputAlert(item) {
  if (["environmental", "suspicion"].includes(item.final_decision)) {
    return { severity: "anomaly", message: "Se encontró una anomalía." };
  }
  if (item.final_decision !== "fault_compatible") return null;

  const diagnoses = (item.evidence || []).filter((entry) =>
    ["compatible", "no_fault_evidence", "indeterminate"].includes(entry.status));
  const usedIds = item.combined_evaluation?.used_evidence_ids;
  const relevant = item.evidence_policy === "combine_v1"
    ? diagnoses.filter((entry) => !usedIds || usedIds.includes(entry.evidence_id))
    : diagnoses.slice(-1);
  const types = [...new Set(relevant.filter((entry) => entry.status === "compatible")
    .map((entry) => entry.data || {})
    .filter((data) => !data.fault_type_status || data.fault_type_status === "identified")
    .map((data) => FAULT_TYPES[data.fault_type]).filter(Boolean))];
  return { severity: "fault", message: types.length ? `Falla · Posible falla: ${types.join(", ")}` : "Falla" };
}
