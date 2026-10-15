const THEME_PROPERTIES = [
  "--primary-background-color", "--secondary-background-color", "--card-background-color",
  "--ha-card-background", "--primary-text-color", "--secondary-text-color", "--divider-color",
  "--primary-color", "--error-color", "--warning-color", "--ha-card-border-color", "--ha-card-border-radius",
  "--ha-font-family-body",
];

export function connectHomeAssistantTheme(target = window) {
  const root = target.document.documentElement;
  const previous = new Map([...THEME_PROPERTIES, "color-scheme"].map((key) => [key, root.style.getPropertyValue(key)]));
  const hadClass = root.classList.contains("artifact-alerts-document");
  root.classList.add("artifact-alerts-document");
  const parentOrigin = new URLSearchParams(target.location.search).get("ha_origin");
  let trustedOrigin;
  try {
    const url = new URL(parentOrigin);
    if (["http:", "https:"].includes(url.protocol) && url.origin === parentOrigin) trustedOrigin = parentOrigin;
  } catch { /* A standalone alerts page follows the browser's light/dark preference. */ }

  function receiveTheme(event) {
    if (!trustedOrigin || event.origin !== trustedOrigin || event.source !== target.parent
      || event.data?.type !== "artifact-alerts-theme") return;
    for (const key of THEME_PROPERTIES) {
      const value = event.data.variables?.[key];
      if (typeof value === "string" && value.trim() && value.length < 256) root.style.setProperty(key, value);
      else root.style.removeProperty(key);
    }
    root.style.colorScheme = event.data.dark ? "dark" : "light";
  }
  target.addEventListener("message", receiveTheme);
  if (trustedOrigin && target.parent !== target) {
    target.parent.postMessage({ type: "artifact-alerts-theme-ready" }, trustedOrigin);
  }
  return () => {
    target.removeEventListener("message", receiveTheme);
    if (!hadClass) root.classList.remove("artifact-alerts-document");
    for (const [key, value] of previous) {
      if (value) root.style.setProperty(key, value);
      else root.style.removeProperty(key);
    }
  };
}
