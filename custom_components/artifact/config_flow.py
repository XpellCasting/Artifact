"""Configure Artifact and its web interface from Home Assistant's UI."""

from typing import Any

from aiohttp import ClientError
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.const import CONF_URL
from homeassistant.helpers.aiohttp_client import async_get_clientsession

# Home Assistant replaced voluptuous with probatio in newer releases.
try:
    import probatio as schema
except ImportError:
    import voluptuous as schema

from .api import ArtifactClient, InvalidArtifactResponse, normalize_url
from .const import CONF_WEB_URL, DOMAIN

STEP_SCHEMA = schema.Schema(
    {
        schema.Required(CONF_URL): str,
        schema.Required(CONF_WEB_URL): str,
    }
)


class ArtifactConfigFlow(ConfigFlow, domain=DOMAIN):
    """Configure one or more Artifact servers without using URLs as unique IDs."""

    VERSION = 1

    async def _validate(self, user_input: dict[str, Any]) -> dict[str, str]:
        """Validate both addresses and the actual API response."""
        data = {
            CONF_URL: normalize_url(user_input[CONF_URL]),
            CONF_WEB_URL: normalize_url(user_input[CONF_WEB_URL]),
        }
        await ArtifactClient(
            async_get_clientsession(self.hass), data[CONF_URL]
        ).async_health()
        return data

    async def _async_step_connection(
        self, step_id: str, user_input: dict[str, Any] | None
    ) -> ConfigFlowResult:
        """Share validation between initial setup and reconfiguration."""
        entry = self._get_reconfigure_entry() if step_id == "reconfigure" else None
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                data = await self._validate(user_input)
            except ValueError:
                errors["base"] = "invalid_url"
            except InvalidArtifactResponse:
                errors["base"] = "invalid_response"
            except (ClientError, TimeoutError):
                errors["base"] = "cannot_connect"
            else:
                for current in self._async_current_entries():
                    if (
                        (entry is None or current.entry_id != entry.entry_id)
                        and current.data[CONF_URL] == data[CONF_URL]
                    ):
                        return self.async_abort(reason="already_configured")
                if entry is not None:
                    return self.async_update_reload_and_abort(entry, data_updates=data)
                return self.async_create_entry(title="Artifact", data=data)
        return self.async_show_form(
            step_id=step_id,
            data_schema=self.add_suggested_values_to_schema(
                STEP_SCHEMA, user_input or (entry.data if entry else None)
            ),
            errors=errors,
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Set up a server."""
        return await self._async_step_connection("user", user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change addresses without recreating entities or the device."""
        return await self._async_step_connection("reconfigure", user_input)
