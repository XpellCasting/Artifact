// Lovelace card. All evidence is rendered as text from a single recorded state.
const terms = { supports: "respalda", contradicted: "contradicción", contradicts: "contradice", supported: "respaldo",
  observed_neutral: "neutral", unknown: "desconocido", missing: "ausente", evaluated: "evaluado",
  level: "nivel", innovation: "innovación", trend: "tendencia", normal: "normal", sensor_fault: "falla de sensor",
  environmental_anomaly: "anomalía ambiental", P: "desviación puntual", D: "cambio secuencial", N: "energía residual" };
function display(value) {
  if (value == null) return "sin dato";
  if (typeof value === "number") return value.toLocaleString("es", { maximumFractionDigits: 6 });
  if (typeof value === "boolean") return value ? "sí" : "no";
  return terms[value] || String(value);
}

function comparison(label, before, after) {
  if (before == null && after == null) return "";
  return label + " " + (before == null ? display(after) : display(before) + " → " + display(after));
}

function filterDetails(change) {
  const list = document.createElement("ul");
  const names = { contextual: "Contextual", statistical_focus: "Estadístico", classifier: "Clasificador" };
  for (const filter of change.filters || []) {
    const item = document.createElement("li");
    item.textContent = (names[filter.module_type] || filter.node_id) + " · " + comparison("sospecha", filter.before, filter.after);
    if (filter.impact_pp != null) item.append(" · " + display(filter.impact_pp) + " puntos de confianza");
    const lines = document.createElement("ul");
    function line(parts) {
      const text = parts.filter(Boolean).join(" · ");
      if (text) {
        const li = document.createElement("li");
        li.textContent = text;
        lines.append(li);
      }
    }
    const a = filter.previous_evidence || {}, b = filter.evidence || {};
    if (filter.module_type === "contextual") {
      const prior = new Map((a.neighbors || []).map(n => [n.neighbor_id, n]));
      const now = new Map((b.neighbors || []).map(n => [n.neighbor_id, n]));
      line([comparison("Evaluación", a.context_outcome, b.context_outcome),
        comparison("familia decisiva", a.decision_family, b.decision_family),
        comparison("estadístico", a.decision_score, b.decision_score), comparison("umbral", a.decision_threshold, b.decision_threshold)]);
      for (const id of new Set([...now.keys(), ...prior.keys()])) {
        const old = prior.get(id) || {}, next = now.get(id) || {};
        line([id, comparison("valor observado", old.observed_value, next.observed_value),
          comparison("relación", old.observation_status, next.observation_status),
          comparison("aporte contextual", old.weighted_contribution, next.weighted_contribution),
          comparison("parte de la contradicción total", old.contradiction_term, next.contradiction_term),
          comparison("término del estadístico", old.decision_term, next.decision_term),
          next.expected_direction_or_state && "se esperaba " + next.expected_direction_or_state,
          next.carried_forward && "se usó la última observación disponible",
          !now.has(id) && "sin detalle actual para esta entidad"]);
        for (const family of new Set([...Object.keys(old.checks || {}), ...Object.keys(next.checks || {})])) {
          const x = old.checks?.[family] || {}, y = next.checks?.[family] || {};
          line([display(family), comparison("estado", x.status, y.status),
            comparison("valor del vecino", x.predictor_value, y.predictor_value),
            comparison("esperado para el sensor evaluado", x.expected_value, y.expected_value),
            comparison("desviación/tolerancia", x.ratio, y.ratio)]);
        }
      }
      if (b.total_neighbors > now.size) line(["Se muestran " + now.size + " de " + b.total_neighbors + " entidades."]);
    } else if (filter.module_type === "statistical_focus") {
      const labels = { expected_value: "Valor esperado", residual: "Desviación de la lectura", dominant_channel: "Canal dominante",
        joint_score: "Estadístico", threshold: "Umbral", pattern_detail: "Indicador", suspected_pattern: "Patrón",
        state_changes: "Cambios de estado", activations: "Activaciones", state_duration_seconds: "Duración del estado (s)",
        report_gap_seconds: "Intervalo entre lecturas (s)", transition_probability: "Probabilidad de la transición" };
      for (const [key, title] of Object.entries(labels)) line([comparison(title, a[key], b[key])]);
      for (const key of ["P", "D", "N"]) line([comparison(display(key), a.channel_scores?.[key], b.channel_scores?.[key])]);
      line([comparison("Persistencia detectada", a.persistence?.detected, b.persistence?.detected),
        comparison("duración (s)", a.persistence?.duration_seconds, b.persistence?.duration_seconds),
        comparison("umbral (s)", a.persistence?.threshold_seconds, b.persistence?.threshold_seconds)]);
    } else if (filter.module_type === "classifier") {
      const prior = new Map((a.window_evidence?.windows || []).map(w => [w.seconds, w]));
      for (const w of b.window_evidence?.windows || []) {
        const old = prior.get(w.seconds) || {};
        const impact = filter.window_fault_delta_pp?.[w.seconds];
        line(["Ventana de " + w.seconds / 3600 + " h del sensor de origen",
          impact != null && display(impact) + " puntos en la puntuación de falla",
          comparison("utilizada", old.usable, w.usable), comparison("lecturas en segmentos utilizados", old.records, w.records),
          comparison("cambios de estado en esos segmentos", old.changes, w.changes),
          comparison("último valor", old.last_segment?.last_value, w.last_segment?.last_value),
          comparison("fracción de estado dominante del último segmento", old.last_segment?.dominant_fraction, w.last_segment?.dominant_fraction),
          comparison("fracción inválida del último segmento", old.last_segment?.invalid_fraction, w.last_segment?.invalid_fraction),
          comparison("transiciones por intervalo del último segmento", old.last_segment?.transition_ratio, w.last_segment?.transition_ratio)]);
      }
      line([comparison("Clase seleccionada", a.predicted_label, b.predicted_label),
        comparison("tipo de falla", a.fault_type, b.fault_type)]);
      for (const key of ["normal", "environmental_anomaly", "sensor_fault"]) {
        line([comparison("Puntuación de " + display(key), a.class_scores?.[key], b.class_scores?.[key])]);
      }
      if (!b.window_evidence && filter.after != null) line(["Esta lectura carece de atribución por ventana del modelo."]);
    }
    if (filter.after == null) line(["Filtro sin aporte en la lectura actual", change.excluded_filters?.[filter.node_id]]);
    else if (!lines.children.length) line([filter.before === filter.after ? "El aporte del filtro se mantuvo." :
      "Esta lectura carece de evidencia adicional sobre el cambio."]);
    item.append(lines);
    list.append(item);
  }
  return list;
}

class ConfidenceChanges extends HTMLElement {
  constructor() {
    super();
    const root = this.attachShadow({ mode: "open" });
    root.innerHTML = `
      <style>
        ha-card { display:block; padding:16px; color:var(--primary-text-color) }
        h2 { font-size:1.2em; margin:0 0 12px }
        .controls { display:flex; gap:12px; flex-wrap:wrap; align-items:end }
        label { display:grid; gap:4px; max-width:100% }
        select,input,button { font:inherit; max-width:100%; padding:5px; color:inherit;
          border:1px solid var(--divider-color); border-radius:4px; background:var(--card-background-color) }
        summary,button { cursor:pointer } details { padding:8px 0; border-top:1px solid var(--divider-color) }
        pre { white-space:pre-wrap; overflow-wrap:anywhere } [hidden] { display:none }
      </style>
      <ha-card><h2></h2><div class="controls">
        <label>Entidad<select></select></label>
        <label>Desde (hora local de publicación)<input type="datetime-local" required></label>
        <button class="load">Consultar una hora</button>
      </div><p class="current"></p><details class="current-evidence" hidden><summary>Qué cambió en los filtros</summary>
      <div></div></details><p role="status"></p><div class="rows"></div>
      <button class="more" hidden>Ver más cambios</button></ha-card>`;
    this.select = root.querySelector("select");
    this.start = root.querySelector("input");
    this.status = root.querySelector('[role="status"]');
    this.rows = root.querySelector(".rows");
    this.more = root.querySelector(".more");
    const start = new Date(Date.now() - 59 * 60 * 1000);
    this.start.value = new Date(start - start.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
    this.requestId = 0;
    root.querySelector(".load").onclick = () => this.load();
    this.select.onchange = () => { this.showCurrent(); this.load(); };
    this.start.onchange = () => this.load();
    this.more.onclick = () => this.appendPage();
  }

  setConfig(config) {
    const selected = this.select.value;
    this.shadowRoot.querySelector("h2").textContent = config.title;
    this.select.replaceChildren(...config.entities.map(id => new Option(id, id)));
    if (config.entities.includes(selected)) this.select.value = selected;
    this.showCurrent();
    this.reset();
  }

  set hass(value) {
    this._hass = value;
    this.showCurrent();
  }

  showCurrent() {
    const attrs = this._hass?.states[this.select.value]?.attributes;
    this.shadowRoot.querySelector(".current").textContent = attrs?.confidence_change_reason
      ? "Último cambio · " + attrs.confidence_change_reason : "";
    const detail = attrs?.confidence_change_detail;
    const section = this.shadowRoot.querySelector(".current-evidence");
    section.hidden = !detail;
    if (detail && this.lastDetail !== detail) section.querySelector("div").replaceChildren(filterDetails(detail));
    this.lastDetail = detail;
  }

  reset() {
    this.requestId++;
    this.rows.replaceChildren();
    this.more.hidden = true;
    this.status.textContent = "";
  }

  async load() {
    this.reset();
    if (!this.start.reportValidity() || !this._hass) return;
    const id = this.requestId;
    const start = new Date(this.start.value);
    const query = new URLSearchParams({ filter_entity_id: this.select.value, skip_initial_state: "1",
      end_time: new Date(start.getTime() + 3600000).toISOString() });
    this.status.textContent = "Consultando historial…";
    try {
      const history = await this._hass.callApi("GET", "history/period/" + start.toISOString() + "?" + query);
      if (id !== this.requestId) return;
      const seen = new Set();
      const evidenceAt = new Map();
      this.changes = (history[0] || []).filter(state => {
        const a = state.attributes;
        const change = a?.confidence_change_detail;
        if (!change || a.activity !== "receiving" ||
            a.evaluated_at !== change.evaluated_at || !Number.isFinite(parseFloat(state.state))) return false;
        const key = JSON.stringify([a.execution_id, a.observation_id, change.evaluated_at]);
        if (seen.has(key)) return false;
        seen.add(key);
        // Older states have current evidence only. Join strictly to the recorded
        // predecessor in this run; a nearby graph point may skip a plateau.
        const previousKey = JSON.stringify([a.execution_id, change.previous_observation_id]);
        const prior = a.execution_id && change.previous_observation_id && evidenceAt.get(previousKey);
        if (prior) {
          for (const filter of change.filters || []) {
            filter.previous_evidence ??= prior.get(filter.node_id);
          }
        }
        if (a.execution_id && a.observation_id === change.observation_id) {
          evidenceAt.set(JSON.stringify([a.execution_id, a.observation_id]),
            new Map((change.filters || []).map(f => [f.node_id, f.evidence])));
        }
        return ["increase", "decrease"].includes(change.kind);
      }).reverse();
      this.status.textContent = this.changes.length ? this.changes.length + " cambios en esta hora."
        : "No hay alzas o bajas con motivo registrado en esta hora.";
      this.offset = 0;
      this.appendPage();
    } catch {
      if (id === this.requestId) this.status.textContent = "No se pudo leer el historial. Comprueba History y Recorder y vuelve a consultar.";
    }
  }

  appendPage() {
    for (const state of this.changes.slice(this.offset, this.offset + 25)) {
      const attrs = state.attributes;
      const change = attrs.confidence_change_detail;
      const row = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = new Date(state.last_changed).toLocaleString() + " · " + attrs.confidence_change_reason;
      const source = document.createElement("p");
      source.textContent = "Fecha de la lectura · " + change.source_timestamp;
      const value = comparison("Lectura de origen", change.previous_source_state ?? change.previous_source_value,
        change.source_state ?? change.source_value);
      if (value) source.append(" · " + value);
      const filters = filterDetails(change);
      const evidence = document.createElement("details");
      const caption = document.createElement("summary");
      caption.textContent = "Evidencia de los filtros (JSON)";
      const detail = document.createElement("pre");
      detail.textContent = JSON.stringify(change, null, 2);
      evidence.append(caption, detail);
      row.append(summary, source, filters, evidence);
      this.rows.append(row);
    }
    this.offset += 25;
    this.more.hidden = this.offset >= this.changes.length;
  }
}

if (!customElements.get("paranoid-confidence-changes")) {
  customElements.define("paranoid-confidence-changes", ConfidenceChanges);
}
