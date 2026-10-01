export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
export const DEFAULT_SUSPICION_THRESHOLD = 0.60;
export const MIN_SUSPICION_THRESHOLD = 0.01;
export const MAX_SUSPICION_THRESHOLD = 0.99;
export const MARKER_GROUP_GAP_MS = 15 * 60 * 1000;

export const EQUATIONS = [
  { id: "residual_sequential", title: "FOCuS sobre residuos", pattern: "residual_sequential_deviation",
    equation: "z = (valor - predicción - centro) / escala; D = FOCuS(z)",
    score: "S = max(P/q99(P), D/q99(D), N/q99(N)); detección si S >= umbral calibrado" },
  { id: "constant_persistence", title: "Persistencia constante", pattern: "stuck_value",
    equation: "Tiempo consecutivo con variación <= epsilon_const",
    score: "Comprobación independiente de constancia" },
];

export const STATE_EQUATIONS = [
  {
    id: "activation_count",
    title: "Activaciones",
    pattern: "activation_burst",
    equation: "A_i(W) = sum 1[b_i(t_j-1)=0 and b_i(t_j)=1]",
    score: "s_A = min(1, Z_A / tau_A)",
  },
  {
    id: "state_change_count",
    title: "Cambios de estado",
    pattern: "activation_burst",
    equation: "C_i(W) = sum 1[b_i(t_j-1) != b_i(t_j)]",
    score: "s_C = min(1, Z_C / tau_C)",
  },
  {
    id: "active_time_ratio",
    title: "Proporcion activa",
    pattern: "state_persistence",
    equation: "P_on,i(W) = T_on,i(W) / |W|",
    score: "s_P = min(1, Z_P / tau_P)",
  },
  {
    id: "time_since_last_activation",
    title: "Tiempo desde ultima activacion",
    pattern: "absence_gap",
    equation: "G_i(t) = t - max{tau < t | b_i(tau-dt)=0 and b_i(tau)=1}",
    score: "s_G = min(1, Z_G / tau_G)",
  },
  {
    id: "report_gap",
    title: "Brecha temporal de reporte",
    pattern: "missing_gap",
    equation: "M_i(t) = t - t_i_last",
    score: "s_M = min(1, Z_M / tau_M)",
  },
  {
    id: "time_since_last_change",
    title: "Tiempo desde ultimo cambio",
    pattern: "state_persistence",
    equation: "TLC_i(t) = t - t_last_change",
    score: "s_TLC = min(1, Z_TLC / tau_TLC)",
  },
  {
    id: "z_duration_score",
    title: "Z-score de duracion",
    pattern: "state_persistence",
    equation: "Z_D_i = Z_D_on",
    score: "s_D = min(1, Z_D_i / tau_D)",
  },
];

export const MULTISTATE_EQUATIONS = [
  {
    id: "state_change_count",
    title: "Cambios multivalor",
    pattern: "transition_burst",
    equation: "C_i(W) = sum 1[s_i(t_j-1) != s_i(t_j)]",
    score: "s_C = min(1, Z_C / tau_C)",
  },
  {
    id: "state_probability",
    title: "Rareza del estado",
    pattern: "rare_state",
    equation: "P(s) = frecuencia historica del estado observado",
    score: "s_estado = max(0, (p_min - P(s)) / p_min)",
  },
  {
    id: "transition_probability",
    title: "Rareza de la transicion",
    pattern: "rare_transition",
    equation: "P(s_j|s_i) = N(s_i -> s_j) / N(s_i -> *)",
    score: "s_trans = max(0, (p_min - P(s_j|s_i)) / p_min)",
  },
  {
    id: "state_duration",
    title: "Persistencia del estado",
    pattern: "state_persistence",
    equation: "D_s(t) = t - instante del ultimo cambio de estado",
    score: "s_D = min(1, Z_D / tau_D)",
  },
  {
    id: "report_gap",
    title: "Brecha multivalor",
    pattern: "missing_gap",
    equation: "M_i(t) = t - t_i_last",
    score: "s_M = min(1, Z_M / tau_M)",
  },
];

