// Lovelace card. All evidence is rendered as text from a single recorded state.
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
      </div><p class="current"></p><p role="status"></p><div class="rows"></div>
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
      this.changes = (history[0] || []).filter(state => {
        const a = state.attributes;
        const change = a?.confidence_change_detail;
        if (!change || !["increase", "decrease"].includes(change.kind) || a.activity !== "receiving" ||
            a.evaluated_at !== change.evaluated_at || !Number.isFinite(parseFloat(state.state))) return false;
        const key = JSON.stringify([a.execution_id, a.observation_id, change.evaluated_at]);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
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
      const filters = document.createElement("ul");
      const names = { contextual: "Contextual", statistical_focus: "Estadístico", classifier: "Clasificador" };
      for (const filter of change.filters) {
        const item = document.createElement("li");
        const score = n => n == null ? "sin aporte" : n.toLocaleString(undefined, { maximumFractionDigits: 6 });
        item.textContent = (names[filter.module_type] || filter.node_id) + " · sospecha " +
          score(filter.before) + " → " + score(filter.after);
        if (filter.impact_pp != null) item.append(" · " + score(filter.impact_pp) + " puntos de confianza");
        filters.append(item);
      }
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
