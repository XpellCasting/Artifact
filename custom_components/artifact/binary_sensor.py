"""Expose API connectivity and a link to Artifact's web interface."""

from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import CONF_WEB_URL, DOMAIN
from .coordinator import ArtifactCoordinator


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Add the connection indicator."""
    async_add_entities([ArtifactConnection(entry.runtime_data, entry)])


class ArtifactConnection(CoordinatorEntity[ArtifactCoordinator], BinarySensorEntity):
    """A healthy connection; unavailable if the API stops responding."""

    _attr_has_entity_name = True
    _attr_translation_key = "connection"
    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: ArtifactCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_connection"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="XpellCasting",
            model="Artifact",
            entry_type=DeviceEntryType.SERVICE,
            configuration_url=entry.data[CONF_WEB_URL],
        )

    @property
    def is_on(self) -> bool:
        """Read only the coordinator's cached response."""
        return self.coordinator.data is not None and self.coordinator.data["status"] == "ok"
