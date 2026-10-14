"""Connect Home Assistant to an independently running Artifact application."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntryNotReady
from aiohttp import ClientError

from .coordinator import ArtifactCoordinator
from .api import InvalidArtifactResponse
from .bridge import ArtifactBridge

PLATFORMS = [Platform.BINARY_SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Validate the connection before loading the entity platform."""
    coordinator = ArtifactCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator
    coordinator.bridge = ArtifactBridge(hass, entry, coordinator.client)
    try:
        await coordinator.bridge.async_start()
    except (ClientError, TimeoutError, InvalidArtifactResponse) as err:
        raise ConfigEntryNotReady("Unable to establish the Artifact bridge") from err
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except BaseException:
        await coordinator.bridge.async_close()
        raise
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Remove entities; the coordinator is shut down by the config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await entry.runtime_data.bridge.async_close()
    return unloaded
