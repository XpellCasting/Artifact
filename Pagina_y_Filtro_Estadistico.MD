# Funciones de la pagina y funcionamiento del filtro estadistico

Este documento describe que hace la pagina del artefacto y como se conecta con
el filtro estadistico-temporal implementado en el backend.

## Objetivo de la pagina

La pagina funciona como interfaz local del artefacto. Su objetivo principal es
permitir:

- cargar un historial real;
- revisar las entidades disponibles;
- visualizar el historico en charts;
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
3. Cargar o subir el historial real.
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
GET /history/static/status?autoload=false
```

### Cargar historial

Permite cargar el historial real disponible en el backend. El archivo usado por
defecto se define mediante la variable de entorno `REAL_HISTORY_PATH`.

Endpoint usado:

```text
POST /history/real/load/start
POST /history/real/upload/start
GET /history/real/load/jobs/{job_id}
```

Cuando se carga el historial, el backend normaliza los datos hacia el modelo
interno del artefacto.

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

### Charts historicos

Los charts muestran la serie temporal de cada entidad. Para entidades continuas
se muestra la linea de valores numericos. Para entidades binarias se muestran
activaciones o porcentaje de tiempo activo por lapsos. Para entidades
multivaloricas se muestra cada estado real como una categoria en el eje Y; no se
asigna significado numerico a la categoria.

Endpoint usado:

```text
GET /history/timeline
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

### Evaluacion con eventos futuros

El control se encuentra dentro de la vista Estadistico y recibe el CSV posterior
al historico cargado. El estado inicial de CUSUM, las ventanas binarias, las
transiciones multistate y el mapa contextual se preparan al cargar el historico.
Subir Evaluacion solo guarda, valida y ordena ese CSV: no entrena ni reinicializa
detectores. La copia del estado congelado se activa recien al pulsar `Avanzar`.
No crea otro modulo de resultados; al avanzar reproduce los eventos
en orden global de `last_changed` y envía cada sospecha a las vistas Contextual
y Clasificador existentes. Las filas futuras nunca se usan para puntuar ni
contextualizar el evento actual.

Controles disponibles:

- subir los eventos futuros de `Evaluacion`;
- elegir entre 100 y 10.000 eventos por avance;
- avanzar una vez, reproducir automaticamente, pausar o reiniciar;
- cambiar el umbral antes de continuar;
- observar reloj, porcentaje, rendimiento y avance por el pipeline existente.

Endpoints usados:

```text
POST /validation/replay/upload
GET  /validation/replay/status
POST /validation/replay/advance
POST /validation/replay/reset
```

Cada resultado incluye:

```json
{
  "replay": {
    "detected_at_simulated": "2026-05-21T01:10:10.209000+00:00",
    "detection_delay_seconds": 120.0,
    "causal": true
  }
}
```

El mismo metadato causal se conserva al pasar a Contextual y Clasificador. Las
tablas que muestran estos resultados son las vistas ya existentes de ambos
filtros; el control de Evaluacion solo informa el progreso y el embudo.

El retraso depende del lote: con avance de un evento se aproxima a cero; con
lotes grandes aumenta porque la interfaz publica las detecciones cuando termina
el lote. `keeps_up_with_event_time` compara el tiempo de procesamiento contra el
intervalo temporal real cubierto por ese lote.

## Funcionamiento general del filtro estadistico

El filtro estadistico esta orquestado por:

```text
backend/app/Filtro_Estadistico/Filtro.py
```

Ese archivo separa el analisis por tipo de entidad:

```text
continuous -> backend/app/Filtro_Estadistico/Continuos.py
binary     -> backend/app/Filtro_Estadistico/Estados_Binarios.py
multistate -> backend/app/Filtro_Estadistico/Estados_Multivalor.py
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

## Filtro para entidades multivaloricas

El filtro multivalorico trabaja solo con los estados y transiciones observados
en el CSV real. Una entidad es elegible cuando tiene al menos 24 lecturas
validas, entre 2 y 32 estados y no presenta una proporcion de valores unicos
propia de campos libres o timestamps.

Indicadores:

```text
C_i(W)       -> cambios de estado dentro de una hora
P(s)         -> frecuencia historica del estado observado
P(s_j|s_i)   -> frecuencia condicional de la transicion observada
D_s(t)       -> tiempo transcurrido en el estado actual
M_i(t)       -> brecha entre reportes validos
```

El puntaje conserva la fuerza de la evidencia estadistica:

```text
suspicion_score = max(s_C, s_estado, s_transicion, s_D, s_M)
```

Patrones posibles:

```text
transition_burst
rare_state
rare_transition
state_persistence
missing_gap
```

Las entidades con un solo estado se informan como `insufficient_states`; las
que contienen un valor casi siempre unico se informan como `high_cardinality`.
En el conjunto Desarrollo actual son elegibles
`sensor.hall_review_status` y `sensor.sittingroom_review_status`, ambas con los
estados reales `ALERT` y `DETECTION`.

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

- El perfil multivalorico se estima desde el historial que se encuentra cargado;
  todavia no se exporta como un modelo entrenado en Desarrollo para reutilizarlo
  sin recalibracion en Validacion y Prueba.
- Una persistencia o brecha es una sospecha estadistica; en historicos basados
  en cambios de estado puede representar una permanencia real y requiere el
  filtro contextual antes de interpretarse como falla.
- El resultado es una sospecha inicial, no un diagnostico definitivo.
- La linea temporal reproduce las fechas del CSV real; no genera observaciones
  ni timestamps artificiales.
