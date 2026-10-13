"""Small asynchronous client for Artifact's existing health endpoint."""

from typing import Any

from aiohttp import ClientSession, ClientTimeout, ContentTypeError
from yarl import URL

from .const import REQUEST_TIMEOUT


class InvalidArtifactResponse(Exception):
    """The endpoint did not return Artifact's health contract."""


def normalize_url(value: str) -> str:
    """Validate an HTTP(S) base URL, preserving reverse-proxy subpaths."""
    value = value.strip()
    if any(character.isspace() for character in value):
        raise ValueError("The URL cannot contain whitespace")
    url = URL(value)
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.user is not None
        or url.query_string
        or url.fragment
        or url.port is None
        or not 1 <= url.port <= 65535
    ):
        raise ValueError("Use an HTTP(S) URL without credentials, query or fragment")
    return str(url).rstrip("/")


class ArtifactClient:
    """Use the session owned by Home Assistant; never close it here."""

    def __init__(self, session: ClientSession, base_url: str) -> None:
        self._session = session
        self.base_url = normalize_url(base_url)

    async def async_health(self) -> dict[str, Any]:
        """Check connectivity without loading histories or running analyses."""
        async with self._session.get(
            f"{self.base_url}/health",
            timeout=ClientTimeout(total=REQUEST_TIMEOUT),
        ) as response:
            response.raise_for_status()
            try:
                payload = await response.json()
            except (ValueError, ContentTypeError) as err:
                raise InvalidArtifactResponse("The API did not return JSON") from err
        if (
            not isinstance(payload, dict)
            or payload.get("status") != "ok"
            or "static_history_dir" not in payload
            or "real_history_path" not in payload
        ):
            raise InvalidArtifactResponse("The API is not a healthy Artifact instance")
        # Internal filesystem paths are not needed in Home Assistant.
        return {"status": "ok"}
