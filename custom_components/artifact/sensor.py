"""Native percentage sensors: Recorder and Lovelace own their history/graphs."""
from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.const import PERCENTAGE
from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo

from .const import CONF_WEB_URL, DOMAIN


async def async_setup_entry(hass, entry, async_add_entities):
    hub = entry.runtime_data.bridge.confidence
    known = set()

    @callback
    def discover():
        new_keys = sorted(set(hub.entities) - known)
        if new_keys:
            known.update(new_keys)
            async_add_entities([ConfidenceSensor(hub, entry, key) for key in new_keys])

    entry.async_on_unload(hub.listen(discover))
    discover()


class ConfidenceSensor(SensorEntity):
    _attr_translation_key = "confidence"
    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 2
    _attr_icon = "mdi:shield-check"

    def __init__(self, hub, entry, key):
        self.hub, self.entry, self.key = hub, entry, key
        item = hub.entities[key]
        source = item["source_entity_id"]
        test = item.get("source_mode") == "test"
        self.entity_id = f"sensor.confianza_{key.split('.', 1)[1]}"
        self._attr_unique_id = f"{entry.entry_id}_confidence_{'test_' if test else ''}{source}"
        self._attr_name = f"Confianza test {source}" if test else f"Confianza {source}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)}, name=entry.title, manufacturer="XpellCasting",
            model="Paranoid", entry_type=DeviceEntryType.SERVICE, configuration_url=entry.data[CONF_WEB_URL])
        self._signature = None

    @property
    def available(self):
        return self.hub.connected and self.hub.synchronized and self.hub.entities[self.key]["available"]

    @property
    def native_value(self):
        return self.hub.entities[self.key].get("value")

    @property
    def extra_state_attributes(self):
        item = self.hub.entities[self.key]
        keys = ("entity_id", "source_entity_id", "source_value", "source_state", "source_timestamp",
                "evaluated_at", "observation_id", "flow_id", "execution_id", "config_hash",
                "confidence_kind", "confidence_calibrated", "evidence_coverage", "method",
                "active_filters", "available_filters", "reason", "activity",
                "confidence_change_reason", "confidence_change_detail")
        attributes = {key: item.get(key) for key in keys}
        attributes["confidence_entity_id"] = attributes.pop("entity_id")
        attributes["source_mode"] = item.get("source_mode", "live")
        return {**attributes, "paranoid_confidence": True, "paranoid_entry_id": self.entry.entry_id}

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        self.async_on_remove(self.hub.listen(self._handle_update))
        self.hub.register_entity(self.key, self.entity_id)

    @callback
    def _handle_update(self):
        item = self.hub.entities[self.key]
        signature = (item.get("revision"), self.available)
        if item.get("source_mode", "live") == "live":
            # A reading can refresh timestamps without changing the confidence
            # or its explanation. Keep such updates out of HA state history.
            signature = (self.available, *(item.get(key) for key in (
                "store_id", "session_id", "flow_id", "execution_id", "config_hash",
                "activity", "value", "reason", "method", "active_filters",
                "available_filters", "evidence_coverage", "confidence_change_detail")))
        if signature != self._signature:
            self._signature = signature
            self.async_write_ha_state()
