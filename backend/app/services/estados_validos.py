"""NONE es una categoría literal; los marcadores de ausencia siguen inválidos."""
import math

VALIDITY_VERSION = 2
INVALID_STATES = {"", "unknown", "unavailable", "none", "null", "nan", "inf", "-inf", "+inf", "infinity", "-infinity"}


def valid_state(state):
    if state is None:
        return False
    text = str(state).strip()
    if text == "NONE":
        return True
    if text.lower() in INVALID_STATES:
        return False
    try:
        return math.isfinite(float(text))
    except (ValueError, OverflowError):
        return True
