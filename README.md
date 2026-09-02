# Artefacto estadistico para diagnostico de sensores

Prototipo local para cargar datos reales ordenados cronologicamente y ejecutar
en secuencia el filtro estadistico-temporal, el segundo filtro contextual
espacio-temporal y el clasificador diagnostico definidos en la tesis.

El alcance actual cubre:

- HU-03 historico canonico persistido en CSV.
- HU-04 validacion del historico.
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

El test espera la API, carga el historico real cuando esta disponible, verifica
su linea temporal y comprueba que la web responda. No genera eventos
artificiales.

El endpoint `GET /health` queda documentado en Swagger UI bajo el grupo
`health` para revisar rapidamente que la API este operativa y que rutas de
datos esta usando.

### Carga del historico con progreso

La interfaz inicia la carga real en segundo plano y consulta su avance hasta
que tambien queda preparada la vista temporal. La API expone:

- `POST /history/real/load/start`: carga el archivo configurado en
  `REAL_HISTORY_PATH`.
- `POST /history/real/upload/start`: guarda el CSV recibido e inicia su carga.
- `GET /history/real/load/jobs/{job_id}`: informa estado, fase, porcentaje,
  tiempo transcurrido, ETA y resultado final.

La ETA inicial usa una calibracion local de `0.22 s/MiB`, obtenida a partir de
la lectura, normalizacion y persistencia de normalidad del CSV de desarrollo de
1.959.825 filas. Puede ajustarse con la variable
`HISTORY_LOAD_SECONDS_PER_MIB`; despues de cada carga completada, el proceso
actualiza su tasa en memoria con el tiempo realmente observado. Esta cifra es
una referencia operacional y no una garantia para otro hardware.

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

1. Cargar o subir el CSV real desde la web.
2. Esperar a que la barra de progreso informe que el historico esta listo.
3. Ejecutar `POST /history/analyze` desde `Analizar historico`.
4. Consultar `GET /history/timeline` para visualizar los datos respetando su
   fecha original.
5. Abrir la pestana `Contextual` para consultar `GET /context/map` y
   `GET /context/results`.
6. Abrir la pestana `Clasificador` para revisar diagnosticos, trazabilidad,
   especificacion congelada y disponibilidad de la evaluacion extremo a
   extremo.
7. En el modulo `Estadistico`, usar `Evaluacion con eventos futuros` para subir
   el CSV posterior al historico y reproducirlo por lotes.

El artefacto no incluye generadores de eventos ni inyeccion de fallas. Los
controles incrementales existentes pertenecen solamente a la continuacion del
historico con eventos reales de Evaluacion: no crean estados ni timestamps y no
modifican la base de normalidad aprendida desde el historico.

## Evaluacion causal con eventos futuros

El control vive dentro del modulo estadistico; no existe una cuarta vista ni
tablas duplicadas. Al cargar el historico se construyen y congelan la normalidad,
el estado del detector y la referencia contextual. Subir Evaluacion solamente
guarda, valida y ordena su CSV; no entrena, recalibra ni inicia CUSUM. El estado
congelado se copia al comenzar el primer avance. Cada avance entrega el siguiente
lote cronologico y actualiza las tres vistas existentes:

1. el filtro estadistico detecta sospechas;
2. esas sospechas pasan inmediatamente al filtro contextual con observaciones
   disponibles hasta su timestamp;
3. el clasificador produce el diagnostico final.

Endpoints:

- `POST /validation/replay/upload`: recibe los eventos futuros y solamente
  guarda, valida y ordena el CSV. Usa una copia del estado ya preparado al cargar
  el historico.
- `GET /validation/replay/status`: devuelve reloj, progreso, rendimiento y los
  resultados acumulados de Estadistico, Contextual y Clasificador.
- `POST /validation/replay/advance`: procesa entre 1 y 20.000 eventos a traves
  del pipeline completo sin leer filas futuras de Evaluacion.
- `POST /validation/replay/reset`: vuelve al inicio y restaura el estado causal.

El retraso de cada deteccion es la diferencia entre el timestamp del evento y
el reloj de Evaluacion al cerrar su lote. Los tiempos anteriores del filtro
estadistico aislado no representan ahora el costo total, porque cada sospecha
recorre tambien Contextual y Clasificador.

## Historico real y reproduccion temporal

El modelo canonico usa estos CSV en `data/static_history/`:

- `entities.csv`
- `events.csv`
- `ground_truth.csv`
- `metadata.csv`

El catalogo canonico de entidades vive en `backend/app/domain/`:

- `domain/entities.py`
- `domain/schema.py`

Si ya existian CSV generados con un catalogo anterior, el backend los marca como
desactualizados para evitar mezclar entidades viejas con el dominio actual.

Las utilidades estadisticas viven en `backend/app/stat_utils.py` para evitar
conflictos con la libreria estandar `statistics` de Python. La exportacion CSV
esta separada en `backend/app/exporting.py` para reducir responsabilidades de
`pipeline.py`.

La web muestra entidades, linea temporal real, sospechas estadisticas,
validaciones contextuales y tiempo de ejecucion. La pestana contextual muestra
una matriz de covarianza cruda para
entidades continuas y una matriz de asociacion normalizada que incorpora
relaciones continuas, de estado y mixtas.

En entidades continuas, `raw_score` conserva la magnitud cruda del indicador
dominante. La version `statistical-continuous-score-v2.0.0` transforma cada
indicador con una curva logaritmica suave y agrega mediante el maximo. Los
campos `evidence_score` y `suspicion_score` contienen la misma fuerza de
evidencia en `[0,1)`; no son una probabilidad de falla. Esta correccion se
verifico solo con el mes real disponible y no usa la base simulada. La
especificacion y el contraste antes/despues estan en
`Puntaje_Estadistico_Continuo_v2.MD`.

Las entidades `multistate` tambien se analizan desde el historial real. El
filtro usa cambios por ventana, rareza de estado, rareza de transicion,
persistencia y brechas, sin generar estados artificiales. Exige al menos 24
lecturas validas y dos estados; ademas excluye series de alta cardinalidad como
`sensor.weatherstation_last_rain_2`, cuyo valor es practicamente un timestamp
distinto por fila. La respuesta incluye `multistate_eligibility` para explicar
que entidades fueron analizadas y cuales se omitieron.

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

El artefacto opera sobre el conjunto real cargado y no usa una base simulada
para seleccionar el modelo ni para atribuirle rendimiento. Desarrollo,
validacion y prueba deben conservar su separacion cronologica. Las reglas y
umbrales se fijan con desarrollo; la seleccion se revisa con validacion y las
metricas finales se informan una sola vez sobre prueba. Si el conjunto evaluado
no contiene etiquetas reales suficientes, la API informa que la evaluacion no
esta disponible; no reemplaza la ausencia de verdad por metricas iguales a
cero.

La justificacion metodologica, la tabla de reglas y la forma de contabilizar
los falsos negativos previos al clasificador se documentan en
`Clasificador_Diagnostico_Reproducible.MD`.
