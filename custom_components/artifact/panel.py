"""Register Artifact's alerts as a separate Home Assistant sidebar panel."""

from asyncio import Lock
from pathlib import Path
from urllib.parse import urlencode

from homeassistant.components import frontend, panel_custom
from homeassistant.components.http import StaticPathConfig

from .const import CONF_WEB_URL, DOMAIN

PANEL_MODULE_URL = "/artifact-static/alerts-panel.js"
DASHBOARD_THEME_URL = "/artifact-static/dashboard-theme.js"


def alerts_panel_path(entry):
    """Keep each configured server's panel independent."""
    return f"artifact-alertas-{entry.entry_id}"


async def async_setup_alerts_panel(hass, entry):
    data = hass.data.setdefault(DOMAIN, {})
    async with data.setdefault("panel_static_lock", Lock()):
        if not data.get("alerts_module_registered"):
            await hass.http.async_register_static_paths([
                StaticPathConfig(PANEL_MODULE_URL, str(Path(__file__).parent / "frontend" / "alerts-panel.js"), False),
                StaticPathConfig(DASHBOARD_THEME_URL, str(Path(__file__).parent / "frontend" / "dashboard-theme.js"), False),
            ])
            data["alerts_module_registered"] = True
    dashboard_module = f"{DASHBOARD_THEME_URL}?{urlencode({'v': '0.3.3', 'web_url': entry.data[CONF_WEB_URL]})}"
    modules = data.setdefault("dashboard_theme_modules", {})
    if old_module := modules.get(entry.entry_id):
        frontend.remove_extra_js_url(hass, old_module)
    frontend.add_extra_js_url(hass, dashboard_module)
    modules[entry.entry_id] = dashboard_module
    await panel_custom.async_register_panel(
        hass,
        frontend_url_path=alerts_panel_path(entry),
        webcomponent_name="artifact-alerts-panel",
        sidebar_title="Artifact-Alertas",
        sidebar_icon="mdi:bell-alert",
        module_url=f"{PANEL_MODULE_URL}?v=0.3.3",
        embed_iframe=False,
        config={"url": f"{entry.data[CONF_WEB_URL].rstrip('/')}/alertas"},
    )


def async_remove_alerts_panel(hass, entry):
    frontend.async_remove_panel(hass, alerts_panel_path(entry), warn_if_unknown=False)
    modules = hass.data.get(DOMAIN, {}).get("dashboard_theme_modules", {})
    if module := modules.pop(entry.entry_id, None):
        frontend.remove_extra_js_url(hass, module)
