"""Constants for the Artifact integration."""

from datetime import timedelta

DOMAIN = "artifact"
CONF_WEB_URL = "web_url"
DEFAULT_API_URL = "http://100.72.168.116:8000"
DEFAULT_WEB_URL = "http://100.72.168.116:5173"
UPDATE_INTERVAL = timedelta(seconds=30)
REQUEST_TIMEOUT = 10
