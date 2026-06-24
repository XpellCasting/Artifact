# Filtro estadistico para entidades de estado binario

Este documento describe la implementacion actual del filtro estadistico-temporal
para entidades binarias del historico real, por ejemplo sensores PIR, contactos
magneticos, puertas y ventanas con estados `on/off`, `open/closed` o
`detected/clear`.

## Alcance actual

El filtro implementado analiza solo entidades con `sensor_kind = binary`.
Las entidades multivaloricas quedan fuera de esta etapa.

La implementacion esta en:

```text
backend/app/Filtro_Estadistico/Estados_Binarios.py
```

El orquestador general la llama desde:

```text
backend/app/Filtro_Estadistico/Filtro.py
```

## Variables de la tesis

Para cada evento historico de una entidad binaria se construye una ventana hacia
atras de una hora. Esa ventana se usa porque el historico real de Home Assistant
no siempre entrega muestreo fijo; muchas entidades binarias solo reportan cuando
cambian de estado.

Dentro de la ventana `W` se calculan:

```text
A_i(W) = cantidad de activaciones 0 -> 1
C_i(W) = cantidad total de cambios de estado
T_on,i(W) = tiempo acumulado en estado activo
P_on,i(W) = T_on,i(W) / |W|
G_i(t) = tiempo desde la ultima activacion 0 -> 1
M_i(t) = tiempo entre el reporte actual y el reporte anterior
```

En la salida del codigo esos valores aparecen como:

```text
A_i(W)    -> reading.activations
C_i(W)    -> reading.state_changes
T_on,i(W) -> reading.active_seconds
P_on,i(W) -> reading.active_ratio
G_i(t)    -> reading.gap_seconds
M_i(t)    -> reading.report_gap_seconds
```

## Base de normalidad

La base de normalidad se calcula desde el historico cargado para cada entidad
binaria. Para cada metrica se guarda:

```text
mediana
MAD
MAD escalado = 1.4826 * MAD
p90
p95
p99
```

Cuando `MAD = 0`, el filtro usa pisos minimos para evitar divisiones explosivas
o saturaciones artificiales:

```text
activations      -> piso MAD 1.0
state_changes    -> piso MAD 1.0
active_ratio     -> piso MAD 0.05
gap/report_gap   -> piso MAD 60 segundos
```

Para `G_i(t)` y `M_i(t)` se agregan umbrales robustos de alerta basados en
percentiles. Esto evita marcar como falla cada periodo largo sin movimiento o
sin apertura, que puede ser normal en sensores de estado.

## Normalizacion de indicadores

Cada indicador se transforma en un score parcial:

```text
s_A -> activation_count
s_C -> state_change_count
s_P -> active_time_ratio
s_G -> time_since_last_activation
s_M -> report_gap
```

Para una metrica binaria `M_i(W)` se usa z-score robusto de cola superior:

```text
Z_M(W) = max(0, M_i(W) - mediana(M_i^N)) /
         (1.4826 * MAD(M_i^N) + epsilon)
```

Luego:

```text
s_M = min(1, Z_M(W) / tau_M)
```

## Puntaje binario

El puntaje de evidencia sigue la forma de `Tesis.Tex`:

```text
S_bin(W) = min(1, (s_A + s_C + s_P + s_G + s_M) / 5)
```

En la salida:

```text
raw_score       -> valor crudo del indicador dominante
evidence_score  -> S_bin(W)
suspicion_score -> S_bin(W) / (S_bin(W) + tau_i)
```

Para binarios se usa `tau_i = 0.10`, porque `S_bin(W)` es un promedio de cinco
componentes. Con esto, una senal fuerte en una sola dimension puede convertirse
en sospecha inicial sin exigir que las cinco variables fallen al mismo tiempo.

## Indicadores y patrones

Los indicadores finales son:

```text
activation_count
state_change_count
active_time_ratio
time_since_last_activation
report_gap
```

El patron reportado se interpreta asi:

```text
activation_count o state_change_count -> activation_burst
active_time_ratio                     -> state_persistence
time_since_last_activation            -> absence_gap
report_gap                            -> missing_gap
```

## Salida esperada

Cada sospecha binaria conserva el contrato comun del filtro estadistico:

```json
{
  "entity_id": "binary_sensor.living_motion",
  "timestamp": "2026-05-01T12:00:00Z",
  "sensor_kind": "binary",
  "raw_score": 3.7,
  "evidence_score": 0.4,
  "suspicion_score": 0.8,
  "suspected_pattern": "activation_burst",
  "triggered_indicators": ["activation_count", "state_change_count"],
  "filter": "statistical_temporal",
  "passed_to_contextual_filter": true
}
```

El resultado sigue siendo una sospecha inicial. No clasifica por si solo si el
evento es normal, anomalia ambiental o falla de sensor.
