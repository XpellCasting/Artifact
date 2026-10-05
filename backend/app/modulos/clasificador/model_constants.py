"""Clases de fallas y tipos compartidos por el modelo de segmentos."""
FAULT_TYPES = ("spike", "bias", "drift", "noise", "stuck", "unavailable", "state_replace")
KIND_TO_ID = {"continuous": 0, "binary": 1, "multistate": 2}
