const ARTIFACT_THEME_PROPERTIES = [
  "--primary-background-color", "--secondary-background-color", "--card-background-color",
  "--ha-card-background", "--primary-text-color", "--secondary-text-color", "--divider-color",
  "--primary-color", "--text-primary-color", "--error-color", "--warning-color", "--success-color",
  "--ha-card-border-color", "--ha-card-border-radius", "--ha-card-box-shadow",
];

class ArtifactAlertsPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; width: 100%; height: 100vh; height: 100dvh; overflow: hidden; font-family: var(--ha-font-family-body, Roboto, Noto, sans-serif); background: var(--primary-background-color, #fafafa); }
        iframe { display: block; width: 100%; height: 100%; min-height: 0; border: 0; }
      </style>
      <iframe title="Artifact-Alertas" referrerpolicy="no-referrer"></iframe>`;
    this.shadowRoot.querySelector("iframe").addEventListener("load", () => this._sendTheme(true));
    this._onThemeReady = (event) => {
      if (this._url && event.origin === new URL(this._url).origin
        && event.source === this.shadowRoot.querySelector("iframe").contentWindow
        && event.data?.type === "artifact-alerts-theme-ready") this._sendTheme(true);
    };
  }

  connectedCallback() {
    window.addEventListener("message", this._onThemeReady);
    this._themeObserver = new MutationObserver(() => this._queueTheme());
    // HA applies its active theme on ancestors, including shadow hosts.
    for (let element = this; element; element = element.parentElement || element.getRootNode().host) {
      this._themeObserver.observe(element, { attributes: true, attributeFilter: ["style", "class"] });
    }
    this._queueTheme();
  }

  disconnectedCallback() {
    window.removeEventListener("message", this._onThemeReady);
    this._themeObserver?.disconnect();
    cancelAnimationFrame(this._themeFrame);
    this._themeFrame = null;
  }

  set hass(value) {
    const changed = value?.themes !== this._hass?.themes || value?.selectedTheme !== this._hass?.selectedTheme;
    this._hass = value;
    if (changed) this._queueTheme();
  }

  _queueTheme() {
    if (!this.isConnected || this._themeFrame != null) return;
    this._themeFrame = requestAnimationFrame(() => { this._themeFrame = null; this._sendTheme(); });
  }

  _sendTheme(force = false) {
    if (!this._url || !this.isConnected) return;
    const style = getComputedStyle(this);
    const variables = Object.fromEntries(ARTIFACT_THEME_PROPERTIES.map((key) => [key, style.getPropertyValue(key).trim()]));
    variables["--ha-font-family-body"] = style.getPropertyValue("--ha-font-family-body").trim() || style.fontFamily;
    const theme = { type: "artifact-alerts-theme", dark: Boolean(this._hass?.themes?.darkMode), variables };
    const signature = JSON.stringify(theme);
    if (!force && signature === this._themeSignature) return;
    this._themeSignature = signature;
    this.shadowRoot.querySelector("iframe").contentWindow?.postMessage(theme, new URL(this._url).origin);
  }

  set panel(value) {
    const url = value?.config?.url;
    if (!url || url === this._url) return;
    const target = new URL(url);
    if (!["http:", "https:"].includes(target.protocol)) return;
    this._url = target.href;
    target.searchParams.set("ha_origin", window.location.origin);
    this.shadowRoot.querySelector("iframe").src = target.href;
  }
}

if (!customElements.get("artifact-alerts-panel")) customElements.define("artifact-alerts-panel", ArtifactAlertsPanel);
