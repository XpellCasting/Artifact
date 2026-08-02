# Artefacto estadistico para diagnostico de sensores

Prototipo local para crear un historico canonico normal de 60 dias, simular
eventos nuevos por tick manual y ejecutar en secuencia el filtro
estadistico-temporal, el segundo filtro contextual espacio-temporal y el
clasificador diagnostico definidos en la tesis.

El alcance actual cubre:

- HU-03 historico canonico persistido en CSV.
- HU-04 validacion del historico.
- HU-05 seleccion de entidades.
- HU-06 clasificacion continua, binaria y multivalorica.
- HU-07 normalizacion y alineacion temporal.
- HU-08 identificacion de segmentos aptos.
- HU-10 construccion de base de normalidad.
- HU-15 manejo de datos faltantes y estados invalidos.
- HU-16 deteccion temporal en entidades continuas.
- HU-17 deteccion temporal en entidades de estado binarias y multivaloricas.
- HU-22 visualizacion web de resultados del filtro estadistico.
- Construccion reproducible del mapa de dependencias contextuales.
- Relaciones continuas, de estado y mixtas con desfase temporal.
- Validacion contextual con respaldo, contradiccion, no explicado y cobertura
  insuficiente como estados separados.
- Mapa de covarianza calculado y visible en la web.
- Clasificador determinista y versionado con salida normal, anomalia ambiental,
  posible falla de sensor o diagnostico indeterminado.
- Evaluacion separada del clasificador condicional y del flujo extremo a
  extremo, incluidos los falsos negativos descartados antes de clasificar.

## Ejecucion con Docker

Desde esta carpeta:

```powershell
docker compose up --build
```

Servicios:

- API: <http://localhost:8000>
- Swagger UI: <http://localhost:8000/docs>
- OpenAPI JSON: <http://localhost:8000/openapi.json>
- Web: <http://localhost:5173>

Smoke test post-Docker:

```powershell
python scripts/post_docker_test.py
```

El test espera la API, crea/carga el historico canonico, ejecuta analisis,
simula ticks para cada tipo de falla soportado en entidades continuas y
binarias, imprime si cada falla fue inyectada y si genero sospecha realtime, y
verifica que la web responda.

El endpoint `GET /health` queda documentado en Swagger UI bajo el grupo
`health` para revisar rapidamente que la API este operativa y que rutas de
datos esta usando.

El `docker-compose.yml` monta `./data/static_history` dentro del contenedor API.
Los CSV canonicos se crean ahi la primera vez que se llama
`POST /history/static/ensure` y luego se reutilizan sin sobrescribir. El dataset
HA externo queda fuera del flujo Docker principal.

## Ejecucion local del backend

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

El backend usa `pandas` para leer historicos grandes, validar columnas,
normalizar entidades y calcular ventanas/base de normalidad. El codigo del
pipeline esta comentado en los puntos donde se toman decisiones del artefacto:
seleccion de entidades, normalizacion, construccion de base y filtros.

## Ejecucion local del frontend

```powershell
cd frontend
npm install
npm run dev
```

## Flujo recomendado

1. `POST /history/static/ensure`
2. Seleccionar entidades en la web.
3. Activar o desactivar fallas sinteticas.
4. `POST /realtime/tick`
5. `GET /realtime/current`
6. `GET /realtime/timeline`
7. Abrir la pestana `Contextual` para consultar `GET /context/map` y
   `GET /context/results`.
8. Abrir la pestana `Clasificador` para revisar diagnosticos, trazabilidad,
   especificacion congelada y disponibilidad de la evaluacion extremo a
   extremo.

Cada tick genera eventos nuevos despues del ultimo timestamp del historico
estatico, aplica fallas solo a esos eventos nuevos y ejecuta automaticamente el
filtro estadistico. El historico externo queda disponible solo como
compatibilidad tecnica mediante `POST /datasets/load`.

## Historico y tiempo real

El modelo canonico usa estos CSV en `data/static_history/`:

- `entities.csv`
- `events.csv`
- `ground_truth.csv`
- `metadata.csv`

El catalogo de entidades y fallas vive en `backend/app/domain/`. Tanto el
historico canonico como la simulacion compatible usan esa misma fuente:

- `domain/entities.py`
- `domain/fault_types.py`
- `domain/schema.py`

Los nombres antiguos de fallas se aceptan como alias, pero internamente se
normalizan a `random/spike`, `random`, `malfunction`, `bias`, `drift`,
`stuck-at` y `missing`.

Si ya existian CSV generados con un catalogo anterior, el backend los marca como
desactualizados para evitar mezclar entidades viejas con el dominio actual.

Las utilidades estadisticas viven en `backend/app/stat_utils.py` para evitar
conflictos con la libreria estandar `statistics` de Python. La exportacion CSV
esta separada en `backend/app/exporting.py` para reducir responsabilidades de
`pipeline.py`.

El historico contiene solo comportamiento normal. Las fallas `random/spike`,
`malfunction`, `bias`, `drift`, `stuck-at` y `missing` se inyectan solo en los
ticks de tiempo real. `missing` se representa como fila ausente en eventos y como
registro en `ground_truth`.

La web ejecuta este flujo desde botones y muestra entidades, linea temporal,
zona real de falla, sospechas estadisticas, validaciones contextuales y tiempo
de ejecucion. La pestana contextual muestra una matriz de covarianza cruda para
entidades continuas y una matriz de asociacion normalizada que incorpora
relaciones continuas, de estado y mixtas.

El flujo de simulacion controlada compatible con endpoints `/simulation/*` usa
72 horas minimas y una referencia de ciclos completos para entidades
multivaloricas. Esto evita que estados normales que aparecen solo despues del
mediodia queden fuera del baseline.

En entidades continuas, `raw_score` conserva la magnitud cruda del indicador
dominante. La version `statistical-continuous-score-v2.0.0` transforma cada
indicador con una curva logaritmica suave y agrega mediante el maximo. Los
campos `evidence_score` y `suspicion_score` contienen la misma fuerza de
evidencia en `[0,1)`; no son una probabilidad de falla. Esta correccion se
verifico solo con el mes real disponible y no usa la base simulada. La
especificacion y el contraste antes/despues estan en
`Puntaje_Estadistico_Continuo_v2.MD`.

La respuesta de `POST /filters/statistical/run` incluye `elapsed_ms`,
`elapsed_seconds` y `timings_by_kind_ms`. Estos campos miden solo el tiempo de
ejecucion del filtro estadistico sobre datos ya normalizados, por lo que sirven
para estimar el costo del algoritmo de deteccion y no el costo de carga del CSV.

## Filtro contextual espacio-temporal

El mapa usa por defecto ventanas de una hora, desfases de 0 a 2 horas, un minimo
de 30 observaciones pareadas, cobertura historica de 0.60, asociacion absoluta
de 0.35, estabilidad en al menos 3 de 4 periodos y `top_k=5`. Todos estos
valores, la regla de aceptacion y los motivos de rechazo se devuelven en
`GET /context/map`.

Endpoints:

- `GET /context/map`
- `POST /context/map/rebuild`
- `POST /context/analyze`
- `GET /context/results`

La falta de una observacion relacionada reduce la cobertura y nunca se cuenta
automaticamente como contradiccion. Un patron simultaneo en tres o mas
entidades queda advertido como posible causa comun no identificable hasta
contar con evidencia independiente sobre red, alimentacion o integracion; en
ese estado el filtro conserva el puntaje estadistico sin ajustarlo.

La especificacion, formulas, criterios de actualizacion y trazabilidad de las
seis debilidades metodologicas estan en
`Filtro_Contextual_Espacio_Temporal.MD`.

## Clasificador diagnostico reproducible

El tercer filtro implementa la lista de decision
`diagnostic-rule-list-v1.0.0`. Sus variables proceden exclusivamente de los
resultados estadisticos y contextuales; las etiquetas de verdad no son una
entrada del diagnostico. La salida `indeterminate` representa abstencion cuando
la cobertura no alcanza el minimo, el mapa esta obsoleto o una posible causa
comun no puede distinguirse sin telemetria independiente.

Endpoints:

- `GET /classifier/specification`
- `POST /classifier/analyze`
- `GET /classifier/results`
- `GET /classifier/evaluation`
- `POST /classifier/evaluate`

El artefacto opera provisionalmente sobre el mes real disponible en
`data/datosreales/history.csv` y no usa una base simulada para seleccionar el
modelo ni para atribuirle rendimiento. Las reglas y umbrales quedan fijados a
priori, pero la validacion final permanece pendiente mientras no existan los
cinco meses planificados y etiquetas reales suficientes. Si no existe
`data/datosreales/ground_truth.csv` ni hay fallas controladas en tiempo real,
la API informa que la evaluacion no esta disponible; no reemplaza la ausencia
de verdad por metricas iguales a cero.

La justificacion metodologica, la tabla de reglas y la forma de contabilizar
los falsos negativos previos al clasificador se documentan en
`Clasificador_Diagnostico_Reproducible.MD`.
