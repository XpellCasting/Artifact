const THEME_PROPERTIES = [
  "--primary-background-color", "--secondary-background-color", "--card-background-color",
  "--ha-card-background", "--primary-text-color", "--secondary-text-color", "--divider-color",
  "--primary-color", "--text-primary-color", "--error-color", "--warning-color", "--success-color",
  "--ha-card-border-color", "--ha-card-border-radius", "--ha-card-box-shadow",
  "--ha-font-family-body",
];

export function connectHomeAssistantTheme(target = window) {
  const root = target.document.documentElement;
  const previous = new Map([...THEME_PROPERTIES, "color-scheme"].map((key) => [key, root.style.getPropertyValue(key)]));
  const classes = ["artifact-ha-theme", ...(target.location.pathname.replace(/\/$/, "").endsWith("/alertas") ? ["artifact-alerts-document"] : [])];
  const previousClasses = classes.filter((name) => root.classList.contains(name));
  root.classList.add(...classes);
  const parentOrigin = new URLSearchParams(target.location.search).get("ha_origin");
  let trustedOrigin;
  try {
    const url = new URL(parentOrigin);
    if (["http:", "https:"].includes(url.protocol) && url.origin === parentOrigin) trustedOrigin = parentOrigin;
  } catch { /* A standalone alerts page follows the browser's light/dark preference. */ }
  if (!trustedOrigin && target.parent !== target) {
    try { trustedOrigin = new URL(target.document.referrer || target.location.ancestorOrigins?.[0]).origin; }
    catch { /* A referrer-free dashboard can still identify itself through its parent window. */ }
  }
  const storageKey = "artifact-home-assistant-theme-v1";
  function applyTheme(data) {
    for (const key of THEME_PROPERTIES) {
      const value = data.variables?.[key];
      if (typeof value === "string" && value.trim() && value.length < 256) root.style.setProperty(key, value);
      else root.style.removeProperty(key);
    }
    root.style.colorScheme = data.dark ? "dark" : "light";
  }
  function restoreTheme(raw) {
    try {
      if (!raw || raw.length > 8192 || target.parent === target) return;
      const cached = JSON.parse(raw);
      if (/^https?:\/\//.test(cached.origin) && (!trustedOrigin || cached.origin === trustedOrigin)) applyTheme(cached);
    } catch { /* A missing or unavailable cache does not prevent theme synchronization. */ }
  }
  try { restoreTheme(target.localStorage?.getItem(storageKey)); } catch { /* Private browsing can disable storage. */ }
  function receiveStorage(event) { if (event.key === storageKey) restoreTheme(event.newValue); }

  function receiveTheme(event) {
    if (target.parent === target || (trustedOrigin && event.origin !== trustedOrigin) || event.source !== target.parent
      || event.data?.type !== "artifact-alerts-theme") return;
    if (!/^https?:\/\//.test(event.origin)) return;
    trustedOrigin ||= event.origin;
    applyTheme(event.data);
    try {
      const variables = Object.fromEntries(THEME_PROPERTIES.map((key) => [key, root.style.getPropertyValue(key)]));
      target.localStorage?.setItem(storageKey, JSON.stringify({ origin: trustedOrigin, dark: Boolean(event.data.dark), variables }));
    } catch { /* The active theme still works when browser storage is unavailable. */ }
  }
  target.addEventListener("message", receiveTheme);
  target.addEventListener("storage", receiveStorage);
  if (target.parent !== target) {
    target.parent.postMessage({ type: "artifact-alerts-theme-ready" }, trustedOrigin || "*");
  }
  return () => {
    target.removeEventListener("message", receiveTheme);
    target.removeEventListener("storage", receiveStorage);
    for (const name of classes) if (!previousClasses.includes(name)) root.classList.remove(name);
    for (const [key, value] of previous) {
      if (value) root.style.setProperty(key, value);
      else root.style.removeProperty(key);
    }
  };
}
