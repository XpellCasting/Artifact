"""Read row-level ground truth without inventing labels for unreviewed rows."""
FIELDS = ("es_sintetica", "tipo", "tipo_falla")

FAULT_NAMES = {
    "bias": "Desplazamiento",
    "drift": "Deriva",
    "noise": "Ruido",
    "spike": "Pico",
    "stuck": "Valor congelado",
    "unavailable": "Indisponibilidad",
    "state_replace": "Estado sustituido",
}

FAULT_ALIASES = {
    # bias
    "bias": "bias",
    "desplazamiento": "bias",
    # drift
    "drift": "drift",
    "deriva": "drift",
    # noise
    "noise": "noise",
    "ruido": "noise",
    "malfunction": "noise",
    # spike
    "spike": "spike",
    "pico": "spike",
    "random": "spike",
    # stuck
    "stuck": "stuck",
    "valor congelado": "stuck",
    "valor_congelado": "stuck",
    "valor atascado": "stuck",
    "valor_atascado": "stuck",
    "congelado": "stuck",
    "atascado": "stuck",
    # unavailable
    "unavailable": "unavailable",
    "indisponibilidad": "unavailable",
    "indisponibilidad_entidad": "unavailable",
    "estado_no_valido": "unavailable",
    "indisponible": "unavailable",
    # state_replace
    "state_replace": "state_replace",
    "estado sustituido": "state_replace",
    "estado_sustituido": "state_replace",
    "conmutacion_excesiva": "state_replace",
    "conmutación_excesiva": "state_replace",
    "sustituido": "state_replace",
}

NULL_FAULT_VALUES = {"", "null", "none", "nan", "no_especificada"}


def labelled_columns(headers):
    normalized = [name.strip().lower().lstrip("\ufeff") for name in headers]
    present = [name in normalized for name in FIELDS]
    if any(present) and not all(present):
        raise ValueError("El CSV etiquetado requiere es_sintetica, tipo y tipo_falla juntas.")
    return [normalized.index(name) for name in FIELDS] if all(present) else None


def read_label(raw, columns):
    if columns is None:
        return None
    if any(i >= len(raw) for i in columns):
        return None
    synthetic, kind, fault = (raw[i].strip() for i in columns)

    synth_clean = synthetic.strip().lower()
    if synth_clean in {"true", "1", "t"}:
        synth_flag = "true"
    elif synth_clean in {"false", "0", "f", "", "null", "none"}:
        synth_flag = "false"
    else:
        raise ValueError("es_sintetica debe ser true o false.")

    kinds = {
        "normal": "normal",
        "anomalia": "anomalia",
        "anomalía": "anomalia",
        "environmental_anomaly": "anomalia",
        "falla": "Falla",
        "sensor_fault": "Falla",
        "null": "NULL",
        "": "NULL",
        "none": "NULL",
        "nan": "NULL",
        "uncertain": "NULL",
        "incierto": "NULL",
    }
    kind_clean = kind.strip().lower()
    if kind_clean not in kinds:
        raise ValueError("tipo debe ser normal, anomalia, Falla o NULL.")
    kind = kinds[kind_clean]

    fault_clean = fault.strip().lower()
    fault_code = FAULT_ALIASES.get(fault_clean)

    if fault_clean not in NULL_FAULT_VALUES and (kind != "Falla" or fault_code is None):
        raise ValueError("tipo_falla debe ser una falla del catálogo, o NULL cuando no aplica.")

    display_fault = FAULT_NAMES.get(fault_code, "NULL") if kind == "Falla" else "NULL"
    label_map = {
        "normal": "normal",
        "anomalia": "environmental_anomaly",
        "Falla": "sensor_fault",
        "NULL": "uncertain",
    }

    return {
        "es_sintetica": synth_flag,
        "tipo": kind,
        "tipo_falla": display_fault,
        "label": label_map[kind],
        "fault_type": fault_code,
        "evaluable": kind != "NULL",
        "imported": True,
    }

