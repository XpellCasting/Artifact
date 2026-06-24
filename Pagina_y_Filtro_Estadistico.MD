# Funciones de la pagina y funcionamiento del filtro estadistico

Este documento describe que hace la pagina del artefacto y como se conecta con
el filtro estadistico-temporal implementado en el backend.

## Objetivo de la pagina

La pagina funciona como interfaz local del artefacto. Su objetivo principal es
permitir:

- cargar o crear un historial;
- revisar las entidades disponibles;
- visualizar el historico en charts;
- seleccionar entidades;
- ejecutar el filtro estadistico;
- revisar sospechas detectadas;
- abrir el detalle de calculo de una sospecha.

La pagina no implementa todavia la integracion real con Home Assistant. Trabaja
contra la API local del artefacto.

## Flujo principal

El flujo esperado de uso es:

```text
1. Abrir la pagina web.
2. Consultar el estado del backend.
3. Cargar historial real o crear/cargar historial canonico.
4. Revisar cantidad total de datos y entidades disponibles.
5. Ajustar el umbral global de sospecha.
6. Ejecutar el filtro estadistico.
7. Revisar charts y resultados sospechosos.
8. Clickear una fila de resultado para ver el calculo usado.
```

## Funciones visibles de la pagina

### Estado del sistema

Muestra si la API esta disponible y si existe un historial cargado. Esta seccion
se apoya en:

```text
GET /health
GET /history/static/status
```

### Cargar historial

Permite cargar el historial real disponible en el backend. El archivo usado por
defecto se define mediante la variable de entorno `REAL_HISTORY_PATH`.

Endpoint usado:

```text
POST /history/real/load
```

Cuando se carga el historial, el backend normaliza los datos hacia el modelo
interno del artefacto.

### Crear o cargar historial canonico

Permite crear un historial canonico minimo si no existe, o cargarlo si ya esta
creado.

Endpoint usado:

```text
POST /history/static/ensure
```

Este flujo sirve como respaldo cuando no se quiere usar el historial real.

### Resumen del historial

Muestra:

- cantidad total de filas;
- cantidad de entidades;
- fecha inicial;
- fecha final;
- tipo de fuente usada;
- lista de entidades disponibles.

La informacion viene desde:

```text
GET /history/static/status
```

### Seleccion de entidades

La pagina permite marcar entidades para enfocar los charts o el flujo realtime.
La seleccion no modifica el historial original.

### Charts historicos

Los charts muestran la serie temporal de cada entidad. Para entidades continuas
se muestra la linea de valores numericos. Para entidades binarias se muestran
activaciones o porcentaje de tiempo activo por lapsos.

Endpoint usado:

```text
GET /realtime/timeline
```

Parametros relevantes:

```text
start       -> fecha/hora inicial visible
end         -> fecha/hora final visible
max_points  -> maximo de puntos retornados por entidad
```

Los puntos rojos representan sospechas estadisticas cuyo `suspicion_score`
supera el umbral visual configurado en la pagina.

### Umbral global de sospecha

La pagina permite ajustar el umbral de sospecha. Ese valor se envia al backend
cuando se ejecuta el filtro.

Endpoint usado:

```text
POST /history/analyze
```

Body esperado:

```json
{
  "suspicion_threshold": 0.6
}
```

Un resultado se considera sospechoso si:

```text
suspicion_score >= suspicion_threshold
```

### Resultados del filtro estadistico

La tabla de resultados muestra, entre otros:

- entidad;
- tipo de sensor;
- timestamp;
- `suspicion_score`;
- `raw_score`;
- lectura;
- patron sospechado;
- indicadores activados;
- descripcion del calculo.

Los resultados provienen de:

```text
POST /history/analyze
```

### Ventana de calculo por sospecha

Al clickear una fila de la tabla de sospechas, la pagina abre una ventana con el
detalle del calculo. Esta ventana no recalcula todo el historial; solo reconstruye
la explicacion usando los valores que ya vienen en el resultado seleccionado.

La ventana muestra:

- ecuacion usada;
- valores reemplazados;
- indicador activado;
- patron resultante;
- lectura de la entidad;
- puntaje final.

## Simulacion realtime actual

La pagina tambien puede generar ticks manuales sobre entidades seleccionadas. En
la version actual del backend, el tick copia el ultimo estado historico de la
entidad seleccionada y avanza el timestamp.

Endpoints usados:

```text
POST /realtime/tick
POST /realtime/reset
GET /realtime/current
```

Este flujo todavia no reemplaza el analisis del historial real. Sirve como base
para una simulacion posterior.

## Funcionamiento general del filtro estadistico

El filtro estadistico esta orquestado por:

```text
backend/app/Filtro_Estadistico/Filtro.py
```

Ese archivo separa el analisis por tipo de entidad:

```text
continuous -> backend/app/Filtro_Estadistico/Continuos.py
binary     -> backend/app/Filtro_Estadistico/Estados_Binarios.py
multistate -> pendiente en esta etapa
```

El flujo interno es:

```text
1. Recibir eventos normalizados y entidades.
2. Separar entidades por tipo.
3. Construir base de normalidad por entidad.
4. Evaluar cada lectura o ventana temporal.
5. Calcular indicadores estadisticos.
6. Convertir la evidencia en suspicion_score.
7. Filtrar resultados bajo el umbral global.
8. Ordenar sospechas por score descendente.
9. Devolver maximo 1000 resultados principales.
```

## Filtro para entidades continuas

El filtro continuo usa los indicadores definidos en la tesis:

```text
hampel
robust_rate_change
cusum_residual
constant_persistence
reading_gap
```

Cada indicador genera un componente:

```text
s_H        -> evidencia por pico aislado
s_R        -> evidencia por cambio brusco
s_C        -> evidencia por desplazamiento persistente
s_D_const  -> evidencia por valor constante
s_M        -> evidencia por brecha temporal
```

La evidencia continua se calcula como:

```text
evidence_score = max(s_H, s_R, s_C, s_D_const, s_M)
```

Luego se transforma a puntaje probabilistico:

```text
suspicion_score = evidence_score / (evidence_score + tau_i)
```

Patrones posibles:

```text
hampel                 -> isolated_spike
robust_rate_change     -> abrupt_change
cusum_residual         -> persistent_shift
constant_persistence   -> stuck_value
reading_gap            -> missing_gap
```

## Filtro para entidades binarias

El filtro binario usa variables de ventana temporal:

```text
A_i(W)       -> cantidad de activaciones
C_i(W)       -> cantidad de cambios de estado
P_on,i(W)    -> proporcion de tiempo activo
G_i(t)       -> tiempo desde ultima activacion
M_i(t)       -> brecha de reporte
```

Cada variable produce un componente:

```text
s_A -> activaciones excesivas
s_C -> cambios excesivos
s_P -> permanencia activa
s_G -> ausencia prolongada de activacion
s_M -> falta de reportes
```

La evidencia binaria se calcula como media:

```text
S_bin = min(1, (s_A + s_C + s_P + s_G + s_M) / 5)
```

Luego se transforma a puntaje probabilistico:

```text
P_sos = S_bin / (S_bin + tau_i)
```

Patrones posibles:

```text
activation_burst
state_persistence
absence_gap
missing_gap
```

## Salida comun del filtro

Los resultados mantienen una salida comun para que la pagina pueda mostrarlos en
la misma tabla:

```json
{
  "entity_id": "sensor.bathroom_temp_bathroom_humidity",
  "timestamp": "2026-05-03T12:01:16Z",
  "sensor_kind": "continuous",
  "raw_score": 13.5,
  "evidence_score": 1.0,
  "suspicion_score": 0.7143,
  "suspected_pattern": "abrupt_change",
  "triggered_indicators": ["robust_rate_change"],
  "filter": "statistical_temporal",
  "passed_to_contextual_filter": true
}
```

## Interpretacion del puntaje

`raw_score` no es porcentaje. Representa la magnitud cruda del indicador
dominante.

`evidence_score` esta normalizado entre 0 y 1.

`suspicion_score` tambien esta entre 0 y 1 y es el valor que se compara contra el
umbral global de la interfaz.

## Limitaciones actuales

- El filtro multivalorico no esta activo en esta etapa.
- El filtro contextual todavia no clasifica la causa final.
- El resultado es una sospecha inicial, no un diagnostico definitivo.
- El realtime actual es una base para simulacion futura, no una reproduccion
  completa de eventos reales.
