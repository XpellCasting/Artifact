# Filtro estadistico para entidades continuas

Este documento describe la implementacion actual del filtro estadistico-temporal
para entidades continuas del historico real, por ejemplo sensores de
temperatura, humedad, voltaje u otras entidades numericas.

## Alcance actual

El filtro analiza entidades con `sensor_kind = continuous`. La entrada no es el
CSV original de Home Assistant directamente, sino el historico ya normalizado por
`CargadorHistorial`.

La implementacion esta en:

```text
backend/app/Filtro_Estadistico/Continuos.py
```

El orquestador general la llama desde:

```text
backend/app/Filtro_Estadistico/Filtro.py
```

## Indicadores implementados

El filtro continuo usa los cinco indicadores definidos en `Tesis.Tex`:

```text
hampel                 -> isolated_spike
robust_rate_change     -> abrupt_change
cusum_residual         -> persistent_shift
constant_persistence   -> stuck_value
reading_gap            -> missing_gap
```

No se usan como indicadores finales `iqr`, `robust_z`, `ewma`,
`window_variability`, `local_slope` ni `drift_slope`.

## Base de normalidad

Para cada entidad continua se construye una base robusta desde su propio
historico. El objetivo es evitar que el rango minimo-maximo del sensor defina la
normalidad, porque esos extremos pueden contener lecturas atipicas.

La base conserva, entre otros:

```text
mediana
MAD
MAD escalado = 1.4826 * MAD
percentiles robustos
banda normal robusta
resumen de tasas de cambio
resumen de brechas temporales
resumen de duraciones constantes
escala de residuales para CUSUM
```

Tambien se aplican pisos minimos por tipo de sensor para evitar saturaciones por
`MAD = 0`:

```text
humedad      -> value_floor = 2.0
temperatura  -> value_floor = 0.25
otros        -> value_floor = 0.05
```

## Reglas fisicas

El filtro utiliza reglas fisicas basicas para reforzar sospechas evidentes:

```text
humedad < 5% o humedad >= 98%       -> extremo fisico
temperatura <= 0 C o >= 45 C        -> extremo fisico en contexto indoor
```

Estas reglas no reemplazan los indicadores estadisticos. Solo elevan la evidencia
del indicador `hampel` cuando la lectura es fisicamente inusual.

## Calculo por lectura

Para cada lectura `x_i(t)` se calcula:

```text
H_i(t)       -> filtro Hampel local
Z_R_i(t)     -> tasa de cambio robusta
C_residual   -> CUSUM sobre residuales
Z_D_const    -> persistencia de valor constante
Z_M          -> brecha temporal de lectura
```

Luego se generan componentes normalizados:

```text
s_H
s_R
s_C
s_D_const
s_M
```

La evidencia continua se calcula como:

```text
evidence_score = max(s_H, s_R, s_C, s_D_const, s_M)
```

Despues se calcula:

```text
suspicion_score = evidence_score / (evidence_score + tau_i)
```

Actualmente se usa:

```text
tau_i = 0.4
```

## Diferencia entre raw, evidencia y sospecha

La salida separa tres valores:

```text
raw_score       -> valor crudo del indicador dominante
evidence_score  -> evidencia normalizada entre 0 y 1
suspicion_score -> puntaje probabilistico entre 0 y 1
```

Ejemplo:

```text
raw_score = 4.52
evidence_score = 1.0
suspicion_score = 0.7143
```

Esto significa que el indicador dominante supero ampliamente el umbral
estadistico. El `raw_score` no es porcentaje; es la magnitud cruda del indicador.

## Patrones reportados

El patron se determina desde el indicador dominante:

```text
hampel                 -> isolated_spike
robust_rate_change     -> abrupt_change
cusum_residual         -> persistent_shift
constant_persistence   -> stuck_value
reading_gap            -> missing_gap
```

## Salida esperada

Cada sospecha continua conserva el contrato comun del filtro estadistico:

```json
{
  "entity_id": "sensor.bathroom_temperature",
  "timestamp": "2026-05-01T12:00:00Z",
  "sensor_kind": "continuous",
  "raw_score": 4.52,
  "evidence_score": 1.0,
  "suspicion_score": 0.7143,
  "suspected_pattern": "abrupt_change",
  "triggered_indicators": ["robust_rate_change"],
  "filter": "statistical_temporal",
  "passed_to_contextual_filter": true
}
```

El resultado sigue siendo una sospecha inicial. No clasifica por si solo si el
evento es normal, anomalia ambiental o falla de sensor.

## Limitaciones actuales

- La base de normalidad se calcula desde el historico disponible, por lo que
  historicos con muchas anomalias pueden afectar los umbrales.
- El filtro es univariado: evalua una entidad contra su propio comportamiento.
- La explicacion contextual todavia no esta implementada.
- El clasificador final todavia no esta implementado.
