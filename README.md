# Artefacto estadistico para diagnostico de sensores

Prototipo local para cargar datos reales ordenados cronologicamente y ejecutar
en secuencia el filtro estadistico-temporal y el segundo filtro contextual
espacio-temporal definidos en la tesis. La tercera etapa compara las sospechas
pendientes con referencias históricas y patrones de falla. Incluye preparación
de ejemplos, entrenamiento de prototipos supervisados y versiones evaluables.

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
- Cierre de cambios respaldados como anomalia ambiental, con alerta informativa.
- Diagnóstico de los casos pendientes de F2 mediante catálogo o modelo
  supervisado, con abstención cuando la evidencia no alcanza.
- Revision manual de fragmentos reales, anotaciones persistentes y exportacion
  de datos con su base de verdad en la pestaña `Revisión y etiquetado`.
- Generacion reproducible de fallas sobre fragmentos revisados como normales,
  con comparacion antes/despues y exportacion de intervenciones y etiquetas.
- Construcción de referencias normales y ventanas etiquetadas, entrenamiento
  con Desarrollo, evaluación temporal separada y activación explícita de F3.

## Revisión y etiquetado de fragmentos reales

La nueva pestaña permite importar un CSV con su partición (Desarrollo,
Validación o Prueba), elegir una entidad y otras de contexto, revisar un
intervalo y guardar una anotación. Los tiempos se introducen en UTC; la vista
incluye un margen antes y después del intervalo etiquetado. No ejecuta filtros
ni modifica su normalidad, parámetros o datos de reproducción.

Las etiquetas iniciales son `Incierto` y `Borrador`. Para declarar un caso
verificado se necesita una clase conocida, el nombre del revisor y una
descripción de evidencia independiente: mantenimiento, acción controlada,
sensor de referencia o registro del sistema. La revisión visual por sí sola se
conserva como borrador. La declaración del revisor no supone una comprobación
automática de la causa por parte del software.

Cada anotación se puede editar y exportar como ZIP con `events.csv` (filas
originales del fragmento y su contexto, a resolución completa),
`ground_truth.csv` (solo la entidad y el intervalo etiquetados) y
`manifest.json` (procedencia, huella del archivo, evidencia y configuración de
extracción). Las entidades de contexto y los intervalos sin anotación no se
etiquetan automáticamente como normales.

Los originales importados, el índice de consulta y las anotaciones se guardan
en `data/review`, configurable mediante `REVIEW_DATA_DIR` y persistido por un
volumen Docker. Los datos de revisión se excluyen de Git. El CSV debe contener
`entity_id`, `state` y `last_changed` o `timestamp`, con zona horaria explícita
y precisión máxima de microsegundos. Los identificadores deben coincidir
exactamente con la fuente, sin espacios exteriores.
La vista informa las filas que no se pudieron indexar para no tratarlas como
observaciones válidas ni ocultar problemas de calidad.

Procedimiento de revisión y criterios de etiquetado:
[Revision_Etiquetado.MD](Revision_Etiquetado.MD).

En esa misma pestaña, **Generar inyecciones** permite elegir una revisión normal
guardada, el intervalo de intervención, el tipo de falla, su intensidad y una
semilla. Conserva cada caso generado y descarga un ZIP con original, datos
alterados, etiquetas, cambios por fila y manifiesto. Los casos basados en una
revisión visual en borrador se identifican como exploratorios. Las variantes
heredan la partición de su fuente y no se incorporan automáticamente al detector.
Procedimiento y alcance: [Inyeccion_Fragmentos.MD](Inyeccion_Fragmentos.MD).

## Clasificador diagnóstico y preparación del investigador

En **Clasificador diagnóstico → Preparación del investigador**, selecciona
anotaciones normales, fallas documentadas e inyecciones guardadas. Configura la
ventana y el paso del barrido para construir referencias y ejemplos trazables.
El ZIP exporta las características y etiquetas; las variantes conservan el
grupo y la partición de su fragmento original.

El entrenamiento utiliza únicamente Desarrollo. La evaluación requiere
Validación o Prueba posteriores, sin grupos o períodos compartidos, y vuelve a
extraer características con las referencias fijadas en entrenamiento. La
interfaz muestra matriz de confusión, abstenciones y métricas de ventanas de
F3; estas métricas no representan todavía el rendimiento de extremo a extremo.

Sin modelo activo, F3 utiliza un catálogo explícito de patrones. Una versión
entrenada utiliza prototipos supervisados con escalado robusto; su activación
es explícita y se identifica como experimental si falta evaluación completa.
La compatibilidad expresa semejanza, no probabilidad calibrada ni confirmación
de una avería física. F1 y F2 conservan su funcionamiento y sus salidas no se
utilizan automáticamente como verdad. El usuario final no necesita etiquetar
ni reentrenar durante la operación.

Procedimiento, criterios y límites: [Diagnostico_Patrones.MD](Diagnostico_Patrones.MD).

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
6. En el modulo `Estadistico`, usar `Evaluacion con eventos futuros` para subir
   el CSV posterior al historico y reproducirlo por lotes.
7. Consultar las sospechas pendientes procesadas por F3 en **Clasificador
   diagnóstico → Diagnósticos**. Preparar referencias y, cuando existan ejemplos
   suficientes, entrenar y evaluar una versión desde sus vistas de investigador.

El generador de inyecciones está en `Revisión y etiquetado`, donde produce
copias de fragmentos y registra las intervenciones. Los controles incrementales
del módulo Estadístico reproducen el CSV de Evaluación recibido: no crean
estados ni timestamps ni modifican la base de normalidad aprendida.

## Evaluacion causal con eventos futuros

El control de reproducción vive dentro del modulo estadistico. La vista de
revisión manual utiliza sus propias fuentes y anotaciones. Al cargar el
historico en el modulo estadistico se construyen y congelan la normalidad,
el estado del detector y la referencia contextual. Subir Evaluacion solamente
guarda, valida y ordena su CSV; no entrena, recalibra ni inicia CUSUM. El estado
congelado se copia al comenzar el primer avance. Cada avance entrega el siguiente
lote cronologico y ejecuta las etapas implementadas:

1. el filtro estadistico detecta sospechas;
2. esas sospechas pasan inmediatamente al filtro contextual con observaciones
   disponibles hasta su timestamp;
3. los cambios respaldados terminan como ambientales; los demás llegan a F3,
   que conserva los indeterminados pendientes y registra su versión y razones.

Endpoints:

- `POST /validation/replay/upload`: recibe los eventos futuros y solamente
  guarda, valida y ordena el CSV. Usa una copia del estado ya preparado al cargar
  el historico.
- `GET /validation/replay/status`: devuelve un estado liviano con reloj,
  progreso, rendimiento, contadores y hasta 25 resultados del ultimo lote. Su
  tamano no crece con toda la ejecucion y se lee desde una instantanea, sin
  esperar a que termine el lote activo.
- `GET /validation/replay/results`: pagina los resultados acumulados por tipo
  (`statistical`, `contextual`, `environmental`, `pending` o `classifier`) con `offset` y
  `limit` (maximo 500).
- `POST /validation/replay/start`: inicia o reanuda un worker del backend que
  procesa lotes aunque la pestana del navegador no mantenga un bucle activo.
- `POST /validation/replay/pause`: solicita la pausa y responde cuando termina
  el lote que estaba en curso; no se inicia un lote posterior.
- `POST /validation/replay/advance`: procesa entre 1 y 20.000 eventos a traves
  del pipeline completo sin leer filas futuras de Evaluacion. Se conserva para
  el avance manual.
- `POST /validation/replay/reset`: vuelve al inicio y restaura el estado causal.

Los resultados detallados se escriben incrementalmente en
`data/cache/validation_replay_results.sqlite3`. En memoria solo se conservan los
500 mas recientes de cada filtro. De este modo una Evaluacion larga no acumula
todos los diccionarios contextuales en el proceso de la API, pero cualquier
pagina anterior sigue disponible mediante `/validation/replay/results`.

El retraso de cada deteccion es la diferencia entre el timestamp del evento y
el reloj de Evaluacion al cerrar su lote. Los tiempos anteriores del filtro
estadistico aislado no representan ahora el costo total, porque cada sospecha
recorre tambien Contextual y su compuerta de salida. El tiempo total actual
del lote incluye la ejecución de F3 y la persistencia de sus decisiones.

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
automaticamente como contradiccion. En el mapa v2, la coincidencia requiere tres
entidades relacionadas, el mismo patron y una ventana pasada de 300 segundos.
Es descriptiva: no identifica una causa comun ni veta por si sola la salida
ambiental. La atribucion causal permanece no evaluada sin telemetria independiente.
El estado de coincidencia se conserva entre lotes y se limpia al reiniciar.

Las ventanas y TTL se calibran con Desarrollo y se congelan en el mapa. Cada
desfase debe aprobar calidad antes de competir por fuerza de asociacion.
Para continuos relacionados se comprueba ademas el residuo respecto a un rango
condicional aprendido; una magnitud incompatible no permite cerrar como ambiental.
No se fuerzan relaciones: una entidad aislada queda como `no_learned_relations`,
diferente de `insufficient_coverage` cuando faltan lecturas de vecinos existentes.

La especificacion, formulas, criterios de actualizacion y trazabilidad de las
seis debilidades metodologicas estan en
`Filtro_Contextual_Espacio_Temporal.MD`.

## Compuerta contextual y tercer filtro diagnóstico

### Agrupacion de sospechas repetidas en la salida de F1

La pestaña Estadistico permite configurar un lapso sin repeticion para abrir un
nuevo episodio (30 minutos inicialmente; 0 desactiva la agrupacion). La primera
sospecha de una entidad y patron se envia inmediatamente a F2. Las siguientes
sospechas del mismo patron actualizan su episodio y no vuelven a enviarse hasta
que transcurra el lapso configurado desde la ultima sospecha. Otro patron o entidad
abre su propio episodio. La ausencia de repeticiones no es un diagnostico de recuperacion.

La agrupacion se aplica a las sospechas emitidas por F1, despues de su supresion
breve de repetidos ya existente. Desactivar esta agrupacion no desactiva esa
supresion interna. F1 continua procesando todas las lecturas. Cada sospecha que
llega a esta etapa conserva score, fecha e
indicadores originales, ademas de `episode` (identificador, primera/ultima fecha,
apariciones, duracion y score maximo hasta ese evento), `suppressed_by_episode` y
`passed_to_contextual_filter`. El representante conserva su decision contextual;
las repeticiones se pueden relacionar con el mediante `episode_id`. Los datos de
F2 no se recalculan retrospectivamente con observaciones posteriores del episodio.

La API acepta `suspicion_quiet_seconds` entre 0 y 86400 en `/history/analyze`,
`/validation/replay/start` y `/validation/replay/advance`. El replay fija la
configuracion al primer avance, conserva episodios entre lotes y pausas y los
limpia al reiniciar o cargar otro archivo. Usa tiempo de los datos, no del reloj
del ordenador. Una ejecucion antigua sin esta configuracion conserva la agrupacion
desactivada hasta reiniciarse. El cache historico incluye politica y lapso.

`pipeline_summary.grouping` distingue `detected`, `forwarded` y `suppressed`.
`detected = forwarded + suppressed`; las repeticiones no se convierten en
pendientes por faltarles resultado de F2. El almacenamiento de estadisticos
conserva todas las salidas de F1 que recibe, incluidas las agrupadas, y se consulta paginado.
La tabla y su CSV incluyen la referencia al episodio. La UI usa totales globales
para estos contadores y los resultados contextuales, aunque muestre 500 registros.

El lapso es una decision operacional ajustable; agrupar reduce envios repetidos,
pero no corrige posibles falsas sospechas en F1 ni valida un diagnostico de falla.
F3 analiza los representantes pendientes de F2 con la ventana disponible en
ese instante. La agrupación no espera a que termine el episodio para consultar
datos posteriores ni garantiza que F3 disponga ya de un patrón completo.

Un cambio con `context_outcome="supported"` y mapa vigente termina en el segundo
filtro como `environmental_anomaly`. El unico
umbral de respaldo es `ConfiguracionContextual.decision_threshold` (0.35 por
defecto). El puntaje ajustado se conserva como evidencia; no decide el destino.

Cada resultado contextual incluye:

- `contextual_class`: `environmental_anomaly` o `null`.
- `routing.terminal`: si F2 resolvio el evento.
- `routing.send_to_classifier`: elegibilidad para F3; la ejecución se refleja
  en `diagnosis` y `diagnosis_mode` cuando existe una decisión calculada.
- `routing.reason`: respaldo, contradiccion, falta de cobertura, ausencia de
  relaciones, contexto no explicado o mapa no vigente.
- `alert`: alerta informativa con `delivery="recorded_only"` para ambientales;
  `null` para los demas. No se envian notificaciones a Home Assistant.

El pipeline expone `environmental_results`, `pending_classification_results`
y `classifier_results`, con sus contadores. `pipeline_summary.classifier`
cuenta las decisiones calculadas de F3, incluidas sus abstenciones;
`indeterminate` y `no_fault` se identifican por separado. `classifier_status`
informa `ready`, `catalogue` o `active_model`. Un evento sin resultado contextual
o con diagnóstico indeterminado se conserva pendiente. Estos pendientes son
resultados de la ejecución actual, no una cola automática de reentrenamiento.

La implementación anterior del clasificador, su evaluador y los endpoints
`/classifier/*` fueron retirados. La versión actual utiliza `/diagnostics/*`,
un catálogo explícito y un modelo supervisado de prototipos con escalado
robusto, seleccionable tras preparación y evaluación. El anexo
`Clasificador_Diagnostico_Reproducible.MD` se conserva únicamente como referencia
histórica; la especificación vigente está en
[Diagnostico_Patrones.MD](Diagnostico_Patrones.MD).

La evaluación de extremo a extremo debe reconocer los
ambientales resueltos por F2 como decisiones finales sin llegada al clasificador.
Una falla real cerrada incorrectamente como ambiental sera un falso negativo
del filtro contextual, no una deteccion exitosa.

## Lecturas contextuales recientes y precision temporal

Las fechas normalizadas conservan los subsegundos y la zona UTC. El instante
usado para evaluar una sospecha no se trunca a segundos. La cache de analisis
es version 20; el cambio de implementacion invalida tambien el mapa anterior.

Para cada relacion continua, la referencia es `t - lag_seconds`. Se utiliza la
mediana de las lecturas validas de la ventana de esa arista hasta esa referencia.
Si no hay una lectura en esa ventana, se admite el ultimo valor conocido solo
dentro de su limite de antiguedad. No se usan lecturas posteriores a la
referencia, aunque ya esten presentes en el archivo o lote.

El limite inicial es **3600 segundos**. Es una politica conservadora inicial,
no un umbral calibrado con los resultados de Evaluacion. Se configura al
iniciar el backend; en Docker, los valores estan en `docker-compose.yml`:

```yaml
CONTEXT_CONTINUOUS_MAX_AGE_SECONDS: "3600"
CONTEXT_CONTINUOUS_MAX_AGE_BY_ENTITY: '{"sensor.ejemplo_temperatura": 1800}'
```

El objeto por entidad reemplaza el limite general para esos sensores. Los
valores deben ser finitos y no negativos; cero exige una lectura exactamente
en el instante de referencia. No se aplica este arrastre al aprendizaje del
mapa ni se cambia la politica de estados binarios/multivalor.

Un reporte `unknown`, `unavailable` o no numerico bloquea el rescate de una
lectura anterior. Entre lotes se conserva el ultimo reporte original, incluso
si es invalido: su fecha nunca se actualiza artificialmente. Los aportes
exponen `observed_timestamp`, `reference_timestamp`, `observation_age_seconds`,
`max_age_seconds`, `observation_method`, `carried_forward` y `missing_reason`.
La interfaz distingue lectura caducada, reporte invalido y ausencia de datos.

Sin peso contextual observado, `context_score` y `coherence_score` son `null`,
`context_evaluable=false` y la interfaz muestra **No evaluable**, no 0.5000.
Con evidencia parcial puede mostrarse una coherencia numerica; la cobertura
minima sigue siendo obligatoria para cerrar un evento como ambiental.

Tras actualizar el servicio, hay que recargar Desarrollo y Evaluacion y volver
a ejecutar los lotes: los resultados ya calculados no se modifican en vivo.

## Pruebas de la compuerta

Desde `backend`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Desde `frontend`:

```powershell
node --test src/context-routing.test.js
npm run build
```

Las pruebas usan fixtures controlados para verificar el flujo, no para estimar
el rendimiento diagnostico de la tesis. Cubren cierre ambiental, causa comun,
mapa obsoleto, pendientes, replay causal, caché, diagnóstico por patrones,
separación temporal del aprendizaje y conservación de versiones. Los endpoints
antiguos `/classifier/*` permanecen retirados; la API vigente es `/diagnostics/*`.
