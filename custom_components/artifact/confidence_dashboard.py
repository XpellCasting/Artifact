"""Managed native Lovelace dashboard; no custom chart renderer or user edits."""
from homeassistant.components import frontend
from homeassistant.components.lovelace import const as lovelace_const
from homeassistant.components.lovelace.dashboard import LovelaceConfig
from homeassistant.helpers.json import json_dumps, json_fragment

from .const import DOMAIN


def confidence_panel_path(entry):
    return f"paranoid-confianza-{entry.entry_id}"


def _dashboards(hass):
    data = hass.data[getattr(lovelace_const, "LOVELACE_DATA", "lovelace")]
    return data["dashboards"] if isinstance(data, dict) else data.dashboards


class ConfidenceDashboard(LovelaceConfig):
    def __init__(self, hass, path, hub):
        super().__init__(hass, path, {"title": "Paranoid-Confianza", "show_in_sidebar": True,
                                     "require_admin": False, "icon": "mdi:shield-check"})
        self.hub = hub

    @property
    def mode(self):
        return "yaml"

    async def async_get_info(self):
        return {"mode": self.mode, "views": 1}

    async def async_load(self, force):
        cards = [{"type": "markdown", "content": "## Confianza por lectura\nÍndice operativo de 0 % a 100 %. Historial nativo de Home Assistant mediante Recorder."}]
        if not self.hub.entity_ids:
            cards.append({"type": "markdown", "content": "Evalúa un CSV o inicia un flujo en directo desde Paranoid para crear los sensores de confianza."})
        for mode, title in (("live", "En directo"), ("test", "Test")):
            entities = [entity for key, entity in sorted(self.hub.entity_ids.items())
                        if self.hub.entities.get(key, {}).get("source_mode", "live") == mode]
            for offset in range(0, len(entities), 8):
                group = entities[offset:offset + 8]
                cards.extend([
                    {"type": "entities", "title": f"Sensores de confianza ({title})", "entities": group, "show_header_toggle": False},
                    {"type": "history-graph", "title": f"Historial de confianza (%) · {title}", "entities": group,
                     "hours_to_show": 24, "min_y_axis": 0, "max_y_axis": 100, "fit_y_data": False},
                ])
        return {"title": "Paranoid-Confianza", "views": [{"title": "Confianza", "path": "confianza", "cards": cards}]}

    async def async_json(self, force):
        return json_fragment(json_dumps(await self.async_load(force)))


def async_setup_confidence_dashboard(hass, entry, hub):
    managed = hass.data.setdefault(DOMAIN, {}).setdefault("confidence_dashboards", {})
    dashboards = _dashboards(hass)
    if entry.entry_id in managed:
        async_remove_confidence_dashboard(hass, entry)
    base = confidence_panel_path(entry)
    path, suffix = base, 2
    while path in dashboards or (frontend.async_panel_exists(hass, path) if hasattr(frontend, "async_panel_exists")
                                 else path in hass.data.get(frontend.DATA_PANELS, {})):
        path, suffix = f"{base}-{suffix}", suffix + 1
    dashboard = ConfidenceDashboard(hass, path, hub)
    dashboards[path] = dashboard
    managed[entry.entry_id] = dashboard
    hub.dashboard_update = dashboard._config_updated
    frontend.async_register_built_in_panel(hass, "lovelace", sidebar_title="Paranoid-Confianza",
                                          sidebar_icon="mdi:shield-check", frontend_url_path=path,
                                          config={"mode": "yaml"}, require_admin=False)


def async_remove_confidence_dashboard(hass, entry):
    managed = hass.data.get(DOMAIN, {}).get("confidence_dashboards", {})
    if dashboard := managed.pop(entry.entry_id, None):
        dashboard.hub.dashboard_update = None
        if _dashboards(hass).get(dashboard.url_path) is dashboard:
            _dashboards(hass).pop(dashboard.url_path)
            frontend.async_remove_panel(hass, dashboard.url_path, warn_if_unknown=False)
