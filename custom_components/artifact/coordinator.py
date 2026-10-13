"""Coordinate a single inexpensive health request for each Artifact instance."""

import logging
from typing import Any

from aiohttp import ClientError
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import ArtifactClient, InvalidArtifactResponse
from .const import DOMAIN, UPDATE_INTERVAL

_LOGGER = logging.getLogger(__name__)


class ArtifactCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Poll only health; Artifact retains its own application lifecycle."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=UPDATE_INTERVAL,
            always_update=False,
        )
        self.client = ArtifactClient(async_get_clientsession(hass), entry.data[CONF_URL])

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            return await self.client.async_health()
        except (ClientError, TimeoutError, InvalidArtifactResponse) as err:
            raise UpdateFailed("Unable to obtain Artifact health") from err
