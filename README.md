# Artefacto estadistico para diagnostico de sensores

Prototipo local para crear un historico canonico normal de 60 dias, simular
eventos nuevos por tick manual y ejecutar el primer filtro estadistico-temporal
definido en la tesis.

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
zona real de falla, sospechas estadisticas y tiempo de ejecucion. El resultado
aun no es diagnostico final: solo marca sospechas para filtros posteriores.

El flujo de simulacion controlada compatible con endpoints `/simulation/*` usa
72 horas minimas y una referencia de ciclos completos para entidades
multivaloricas. Esto evita que estados normales que aparecen solo despues del
mediodia queden fuera del baseline.

El campo `raw_score` conserva la suma ponderada calculada por las ecuaciones del
filtro. El campo `suspicion_score` es la version normalizada entre `0` y `1` para
visualizacion: `0` representa comportamiento cercano a la normalidad historica y
valores cercanos a `1` representan sospecha estadistica alta.

La respuesta de `POST /filters/statistical/run` incluye `elapsed_ms`,
`elapsed_seconds` y `timings_by_kind_ms`. Estos campos miden solo el tiempo de
ejecucion del filtro estadistico sobre datos ya normalizados, por lo que sirven
para estimar el costo del algoritmo de deteccion y no el costo de carga del CSV.
