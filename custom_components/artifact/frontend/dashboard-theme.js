// Runs in Home Assistant; only configured Artifact frames receive theme values.
const configuredUrl = new URL(new URL(import.meta.url).searchParams.get("web_url"));
const themeProperties = [
  "--primary-background-color", "--secondary-background-color", "--card-background-color", "--ha-card-background",
  "--primary-text-color", "--secondary-text-color", "--divider-color", "--primary-color", "--text-primary-color",
  "--error-color", "--warning-color", "--success-color", "--ha-card-border-color", "--ha-card-border-radius", "--ha-card-box-shadow",
];
const clients = new Map();

function findFrame(root, source) {
  for (const frame of root.querySelectorAll("iframe")) if (frame.contentWindow === source) return frame;
  for (const element of root.querySelectorAll("*")) {
    if (element.shadowRoot) {
      const frame = findFrame(element.shadowRoot, source);
      if (frame) return frame;
    }
  }
  return null;
}

window.addEventListener("message", (event) => {
  if (event.origin !== configuredUrl.origin || event.data?.type !== "artifact-alerts-theme-ready") return;
  for (const [source, client] of clients) {
    if (!client.frame.isConnected) { client.observer.disconnect(); clients.delete(source); }
  }
  const frame = findFrame(document, event.source);
  if (!frame) return;
  const url = new URL(frame.src);
  const basePath = configuredUrl.pathname.replace(/\/$/, "");
  if (url.origin !== configuredUrl.origin || (basePath && url.pathname !== basePath && !url.pathname.startsWith(`${basePath}/`))) return;
  if (clients.has(event.source)) { clients.get(event.source).send(true); return; }
  let signature;
  const send = (force = false) => {
    if (!frame.isConnected) return;
    const style = getComputedStyle(frame);
    const variables = Object.fromEntries(themeProperties.map((key) => [key, style.getPropertyValue(key).trim()]));
    variables["--ha-font-family-body"] = style.getPropertyValue("--ha-font-family-body").trim() || style.fontFamily;
    const theme = { type: "artifact-alerts-theme", variables,
      dark: Boolean(document.querySelector("home-assistant")?.hass?.themes?.darkMode ?? matchMedia("(prefers-color-scheme: dark)").matches) };
    const next = JSON.stringify(theme);
    if (force || next !== signature) { signature = next; event.source.postMessage(theme, configuredUrl.origin); }
  };
  let scheduled = false;
  const observer = new MutationObserver(() => {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(() => { scheduled = false; send(); });
    }
  });
  for (let element = frame; element; element = element.parentElement || element.getRootNode().host) {
    observer.observe(element, { attributes: true, attributeFilter: ["style", "class"] });
  }
  clients.set(event.source, { frame, observer, send });
  send(true);
});
