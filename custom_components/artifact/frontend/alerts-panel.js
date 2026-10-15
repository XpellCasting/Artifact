class ArtifactAlertsPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: grid; grid-template-rows: auto minmax(0, 1fr); width: 100%; height: 100vh; height: 100dvh; overflow: hidden; background: var(--primary-background-color, #000); }
        header { display: flex; align-items: center; gap: 16px; min-height: 56px; padding: 0 16px; color: var(--primary-text-color); border-bottom: 1px solid var(--divider-color); }
        button { border: 0; background: transparent; color: inherit; font: inherit; font-size: 24px; cursor: pointer; }
        strong { flex: 1; }
        a { color: var(--primary-color, #03a9f4); font-size: 13px; }
        iframe { display: block; width: 100%; height: 100%; min-height: 0; border: 0; }
      </style>
      <header><button type="button" aria-label="Abrir menú">☰</button><strong>Artifact-Alertas</strong><a target="_blank" rel="noopener noreferrer">Abrir en otra ventana</a></header>
      <iframe title="Artifact-Alertas" referrerpolicy="no-referrer"></iframe>`;
    this.shadowRoot.querySelector("button").addEventListener("click", () => {
      this.dispatchEvent(new Event("hass-toggle-menu", { bubbles: true, composed: true }));
    });
  }

  set panel(value) {
    const url = value?.config?.url;
    if (!url || url === this._url) return;
    const target = new URL(url);
    if (!["http:", "https:"].includes(target.protocol)) return;
    this._url = target.href;
    this.shadowRoot.querySelector("a").href = target.href;
    this.shadowRoot.querySelector("iframe").src = target.href;
  }
}

if (!customElements.get("artifact-alerts-panel")) customElements.define("artifact-alerts-panel", ArtifactAlertsPanel);
