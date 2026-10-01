export const FAULT_TYPES = {
  spike: "Pico",
  bias: "Desplazamiento",
  drift: "Deriva",
  noise: "Ruido",
  stuck: "Valor congelado",
  unavailable: "Indisponibilidad",
  state_replace: "Estado sustituido",
};

export const DIAGNOSTIC_PARTITIONS = { development: "Desarrollo", validation: "Validación", test: "Prueba" };

export function annotationDiagnosticProblem(annotation, includeDrafts = false) {
  if (!["normal", "sensor_fault", "environmental_anomaly"].includes(annotation.label)) return "Se requiere normalidad, una falla identificada o una anomalía ambiental.";
  if (annotation.label === "sensor_fault" && !FAULT_TYPES[annotation.fault_type]) return "Falta asignar el tipo de falla en Revisión y etiquetado.";
  if (!["reviewed", "verified"].includes(annotation.status) && !includeDrafts) return "La revisión está pendiente; habilita el uso exploratorio para seleccionarlo.";
  if ((annotation.reviewer || "").trim().length < 2 || (annotation.evidence || "").trim().length < 12) return "Completa el responsable y las observaciones en Revisión y etiquetado.";
  return "";
}

export function faultTypeForLabel(label, faultType) {
  return label === "sensor_fault" && FAULT_TYPES[faultType] ? faultType : null;
}

export function diagnosticLabel(value) {
  return FAULT_TYPES[value] || ({ normal: "Funcionamiento normal", environmental_anomaly: "Anomalía ambiental", sensor_fault: "Falla sin tipo", uncertain: "Incierto", indeterminate: "Tipo indeterminado" })[value] || value || "Tipo indeterminado";
}
