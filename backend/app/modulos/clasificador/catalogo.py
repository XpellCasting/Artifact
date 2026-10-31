"""Taxonomía compartida de fallas, independiente del clasificador."""
CATALOG = [
    {"fault_type": "spike", "label": "Pico", "description": "Cambio abrupto seguido de recuperación observada.", "entity_kinds": ["continuous"]},
    {"fault_type": "bias", "label": "Desplazamiento", "description": "Desviación sostenida del nivel de referencia, sin tendencia dominante.", "entity_kinds": ["continuous"]},
    {"fault_type": "drift", "label": "Deriva", "description": "Separación progresiva y consistente en tiempo real.", "entity_kinds": ["continuous"]},
    {"fault_type": "noise", "label": "Ruido", "description": "Fluctuaciones y cambios de dirección superiores a la referencia.", "entity_kinds": ["continuous"]},
    {"fault_type": "stuck", "label": "Valor congelado", "description": "Lecturas repetidas durante una constancia anormalmente prolongada.", "entity_kinds": ["continuous", "binary", "multistate"]},
    {"fault_type": "unavailable", "label": "Indisponibilidad", "description": "Estados explícitos unavailable o unknown, sin atribuir una causa física; el silencio por sí solo no confirma indisponibilidad.", "entity_kinds": ["continuous", "binary", "multistate"]},
    {"fault_type": "state_replace", "label": "Estado sustituido", "description": "Sustitución por otro estado válido de la referencia, pero incorrecto para la observación.", "entity_kinds": ["binary", "multistate"]},
]
