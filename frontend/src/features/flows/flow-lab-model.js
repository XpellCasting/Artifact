export const NODE_LABELS = { input: "Entrada", statistical_focus: "Estadístico FOCuS", contextual: "Contextual", classifier: "Clasificador", merge: "Unión", decision: "Decisión", output: "Salida" };
export const SIGNAL_LABELS = { positive: "Señal positiva", negative: "Sin señal", unknown: "No evaluable" };
export const DECISIONS = { fault_compatible: "Compatible con falla", environmental: "Anomalía ambiental", suspicion: "Sospecha", normal: "Sin evidencia de falla", no_alert: "Sin alerta", indeterminate: "Indeterminado", not_routed: "Sin ruta de salida" };
export const SIGNALS = Object.keys(SIGNAL_LABELS);
export const clone = (value) => JSON.parse(JSON.stringify(value));
export const flowPayload = (flow) => ({ name: flow.name, nodes: flow.nodes.map(({ id, type, position, config }) => ({ id, type, position, config: config || {} })), edges: flow.edges.map(({ id, source, target, when }) => ({ id, source, target, ...(when === undefined ? {} : { when }) })) });
export const fingerprint = (flow) => JSON.stringify(flowPayload(flow));
export const labelNode = (node) => node?.config?.label || NODE_LABELS[node?.type] || node?.id || "Nodo";
export const edgeLabel = (edge) => edge.when === undefined ? "Todas las señales" : edge.when.map((signal) => SIGNAL_LABELS[signal]).join(" · ");

export function nextNodeId(flow, type) {
  const usedIds = new Set(flow.nodes.map((node) => node.id));
  let suffix = 1;
  while (usedIds.has(`${type}_${suffix}`)) suffix += 1;
  return `${type}_${suffix}`;
}

export function validateDraft(flow) {
  if (!flow) return [];
  const errors = [];
  const nodes = new Map(flow.nodes.map((node) => [node.id, node]));
  if (flow.nodes.filter((node) => node.type === "input").length !== 1) errors.push("El flujo necesita una Entrada.");
  if (flow.nodes.filter((node) => node.type === "output").length !== 1) errors.push("El flujo necesita una Salida.");
  const incoming = new Map(flow.nodes.map((node) => [node.id, 0]));
  const outgoing = new Map(flow.nodes.map((node) => [node.id, []]));
  for (const edge of flow.edges) {
    if (!nodes.has(edge.source) || !nodes.has(edge.target)) { errors.push("Hay una conexión sin nodo de origen o destino."); continue; }
    if (nodes.get(edge.source).type === "output" || nodes.get(edge.target).type === "input") errors.push("Las conexiones deben partir de Entrada y terminar en Salida.");
    if (edge.when !== undefined && (!edge.when.length || edge.when.some((value) => !SIGNALS.includes(value)))) errors.push("Selecciona al menos una señal válida en cada conexión.");
    incoming.set(edge.target, incoming.get(edge.target) + 1);
    outgoing.get(edge.source).push(edge.target);
  }
  for (const node of flow.nodes) {
    if (!Object.hasOwn(NODE_LABELS, node.type)) errors.push("El tipo de nodo fue retirado o no es reconocido; usa Estadístico FOCuS.");
    if (node.type !== "input" && !incoming.get(node.id)) errors.push(`${labelNode(node)} no tiene entrada.`);
    if (node.type !== "output" && !outgoing.get(node.id).length) errors.push(`${labelNode(node)} no tiene salida.`);
    if (incoming.get(node.id) > 1 && !["merge", "output"].includes(node.type)) errors.push(`Usa una Unión antes de ${labelNode(node)} para reunir ramas.`);
    const threshold = node.config?.suspicion_threshold === undefined ? .6 : node.config.suspicion_threshold;
    if (node.type === "statistical_focus" && (!Number.isFinite(threshold) || threshold < .01 || threshold > .99)) errors.push("El umbral estadístico debe estar entre 0,01 y 0,99.");
    if (node.type === "statistical_focus" && node.config?.continuous_method && node.config.continuous_method !== "residual_focus_v1") errors.push("Estadístico FOCuS usa residual_focus_v1.");
    if (node.type === "decision" && node.config?.accept && !node.config.accept.length) errors.push("Selecciona una señal para la decisión.");
  }
  const ready = [...incoming].filter(([, count]) => count === 0).map(([id]) => id);
  let visited = 0;
  while (ready.length) {
    const id = ready.shift(); visited += 1;
    for (const next of outgoing.get(id)) { incoming.set(next, incoming.get(next) - 1); if (incoming.get(next) === 0) ready.push(next); }
  }
  if (visited !== nodes.size) errors.push("El flujo contiene un ciclo. Elimina la conexión que vuelve hacia atrás.");
  return [...new Set(errors)];
}

export function connect(flow, source, target) {
  if (!source || !target || source === target || flow.edges.some((edge) => edge.source === source && edge.target === target)) return null;
  const proposal = { ...flow, edges: [...flow.edges, { id: `${source}-${target}`, source, target }] };
  const blocking = validateDraft(proposal).filter((message) => message.includes("ciclo") || message.includes("terminar en Salida") || message.includes("Usa una Unión"));
  return blocking.length ? null : proposal;
}

export function removeNode(flow, id) {
  if (["input", "output"].includes(flow.nodes.find((node) => node.id === id)?.type)) return flow;
  return { ...flow, nodes: flow.nodes.filter((node) => node.id !== id), edges: flow.edges.filter((edge) => edge.source !== id && edge.target !== id) };
}

export function templates() {
  const make = (name, types, edges) => ({ name, nodes: types.map(([id, type, x, y]) => ({ id, type, position: { x, y }, config: type === "statistical_focus" ? { evidence_policy: "combine_v1", suspicion_threshold: .6, continuous_method: "residual_focus_v1", signal_policy: "detection_before_suppression" } : type === "merge" ? { mode: "any", evidence_policy: "combine_v1" } : type === "decision" ? { accept: ["positive"], evidence_policy: "combine_v1" } : { evidence_policy: "combine_v1" } })), edges: edges.map(([source, target, when]) => ({ id: `${source}-${target}`, source, target, ...(when ? { when } : {}) })) });
  const serial = (name, filters) => {
    const chain = ["input", ...filters, "output"];
    return make(name, chain.map((id, index) => [id, id, index * 260, 80]), chain.slice(1).map((id, index) => [chain[index], id]));
  };
  const reference = serial("Secuencial con compuertas", ["statistical_focus", "contextual", "classifier"]);
  reference.edges = [
    ["input", "statistical_focus"], ["statistical_focus", "contextual", ["positive", "unknown"]], ["statistical_focus", "output", ["negative"]],
    ["contextual", "classifier", ["positive", "unknown"]], ["contextual", "output", ["negative"]], ["classifier", "output"],
  ].map(([source, target, when]) => ({ id: `${source}-${target}`, source, target, ...(when ? { when } : {}) }));
  return [reference,
    make("Tres filtros en paralelo", [["input", "input", 0, 180], ["statistical", "statistical_focus", 280, 0], ["contextual", "contextual", 280, 180], ["classifier", "classifier", 280, 360], ["merge", "merge", 560, 180], ["output", "output", 840, 180]], [["input", "statistical"], ["input", "contextual"], ["input", "classifier"], ["statistical", "merge"], ["contextual", "merge"], ["classifier", "merge"], ["merge", "output"]]),
    serial("Solo estadístico FOCuS", ["statistical_focus"]),
    serial("Solo contextual", ["contextual"]), serial("Solo clasificador", ["classifier"]),
    serial("Estadístico y clasificador", ["statistical_focus", "classifier"]), serial("Contextual antes del estadístico", ["contextual", "statistical_focus", "classifier"]),
    make("Decisión sobre el estadístico", [["input", "input", 0, 80], ["statistical", "statistical_focus", 260, 80], ["decision", "decision", 520, 80], ["classifier", "classifier", 780, 0], ["output", "output", 1040, 80]], [["input", "statistical"], ["statistical", "decision"], ["decision", "classifier", ["positive"]], ["decision", "output", ["negative", "unknown"]], ["classifier", "output"]]),
    serial("Flujo vacío", []),
  ];
}
