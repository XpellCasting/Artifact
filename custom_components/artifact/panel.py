"""Register Paranoid's application and alerts in the Home Assistant sidebar."""

from asyncio import Lock
from pathlib import Path
from urllib.parse import urlencode

from homeassistant.components import frontend, panel_custom
from homeassistant.components.http import StaticPathConfig

from .const import CONF_WEB_URL, DOMAIN

PANEL_MODULE_URL = "/artifact-static/alerts-panel.js"
DASHBOARD_THEME_URL = "/artifact-static/dashboard-theme.js"
PANEL_ASSET_VERSION = "0.4.2"


def app_panel_path(entry):
    """Give each configured server its own application panel."""
    return f"paranoid-{entry.entry_id}"


def alerts_panel_path(entry):
    """Keep each configured server's panel independent."""
    return f"artifact-alertas-{entry.entry_id}"


async def async_setup_panels(hass, entry):
    data = hass.data.setdefault(DOMAIN, {})
    async with data.setdefault("panel_static_lock", Lock()):
        if not data.get("alerts_module_registered"):
            await hass.http.async_register_static_paths([
                StaticPathConfig(PANEL_MODULE_URL, str(Path(__file__).parent / "frontend" / "alerts-panel.js"), False),
                StaticPathConfig(DASHBOARD_THEME_URL, str(Path(__file__).parent / "frontend" / "dashboard-theme.js"), False),
            ])
            data["alerts_module_registered"] = True
    web_url = entry.data[CONF_WEB_URL].rstrip("/")
    dashboard_query = urlencode({"v": PANEL_ASSET_VERSION, "web_url": web_url})
    dashboard_module = f"{DASHBOARD_THEME_URL}?{dashboard_query}"
    modules = data.setdefault("dashboard_theme_modules", {})
    if old_module := modules.get(entry.entry_id):
        frontend.remove_extra_js_url(hass, old_module)
    frontend.add_extra_js_url(hass, dashboard_module)
    modules[entry.entry_id] = dashboard_module
    panels = (
        (app_panel_path(entry), "Paranoid", "mdi:monitor-dashboard", web_url),
        (alerts_panel_path(entry), "Paranoid-Alertas", "mdi:bell-alert", f"{web_url}/alertas"),
    )
    for path, title, icon, url in panels:
        await panel_custom.async_register_panel(
            hass,
            frontend_url_path=path,
            webcomponent_name="artifact-alerts-panel",
            sidebar_title=title,
            sidebar_icon=icon,
            module_url=f"{PANEL_MODULE_URL}?v={PANEL_ASSET_VERSION}",
            embed_iframe=False,
            config={"url": url, "title": title},
        )


def async_remove_panels(hass, entry):
    frontend.async_remove_panel(hass, app_panel_path(entry), warn_if_unknown=False)
    frontend.async_remove_panel(hass, alerts_panel_path(entry), warn_if_unknown=False)
    modules = hass.data.get(DOMAIN, {}).get("dashboard_theme_modules", {})
    if module := modules.pop(entry.entry_id, None):
        frontend.remove_extra_js_url(hass, module)
