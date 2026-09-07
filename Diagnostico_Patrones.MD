# Diagnóstico por patrones y preparación del clasificador F3

F3 compara las sospechas pendientes con el comportamiento histórico revisado
del sensor y propone un patrón de falla. La preparación corresponde al
investigador: el usuario final no necesita etiquetar cada sospecha ni entrenar
el modelo durante la operación.

La pestaña **Clasificador diagnóstico** contiene cuatro vistas:

- **Diagnósticos**: decisiones calculadas, características, razones y referencia.
- **Preparación del investigador**: selección de fragmentos e inyecciones,
  construcción de referencias y conjuntos de ejemplos.
- **Modelos y evaluación**: entrenamiento, evaluación y activación de versiones.
- **Catálogo y referencias**: patrones disponibles y referencias persistidas.

## Recorrido de las observaciones

1. F1 procesa las observaciones y detecta sospechas con sus reglas actuales.
2. La agrupación de episodios conserva su funcionamiento: las repeticiones
   agrupadas no se convierten en nuevos envíos a F2.
3. F2 conserva sus decisiones. Los ambientales terminales se registran y no
   llegan a F3.
4. F3 recibe los casos pendientes, la evidencia de los filtros y la ventana
   temporal del sensor disponible hasta cada decisión.
5. F3 registra `compatible`, `no_fault_evidence` o `indeterminate`.

`compatible` significa semejanza con un patrón conocido; no confirma una causa
física. `no_fault_evidence` indica semejanza con un control aprendido o ausencia
de evidencia concluyente bajo las condiciones del motor. No constituye una
certificación de funcionamiento correcto. `indeterminate` conserva el caso
pendiente porque faltan referencias, observaciones, separación entre clases o
un patrón conocido suficientemente cercano.

Los puntajes se identifican como `compatibility_not_probability`: no son
probabilidades calibradas. La falta de respaldo contextual no demuestra una
falla. Las salidas de F1 y F2 aportan evidencia operacional; no se convierten
automáticamente en etiquetas de verdad.

## Preparar las referencias y ejemplos

### 1. Revisar períodos reales

En **Revisión y etiquetado**, importa las fuentes indicando su partición
temporal: Desarrollo, Validación o Prueba. Guarda anotaciones normales sobre
períodos respaldados como funcionamiento correcto. Incluye diferentes horarios
y condiciones representativas; una ventana estable no representa por sí sola
toda la operación normal del sensor.

Para fallas reales, utiliza la clase **Falla del sensor** y el campo **Tipo de
falla**, conservando la evidencia del diagnóstico. También se pueden incorporar
anotaciones de anomalía ambiental como clase distinta de falla. Los casos
inciertos no se convierten en etiquetas conocidas.

Las anotaciones verificadas exigen los criterios de evidencia de la revisión.
Los borradores requieren habilitar expresamente **Incluir borradores y casos
exploratorios** al construir el conjunto. El conjunto y sus modelos conservan
esa condición; no adquieren verificación por haber sido procesados.

### 2. Incorporar casos de falla

Selecciona fallas reales documentadas o casos guardados por el generador de
inyecciones. Una inyección conserva la revisión normal utilizada, aunque esa
anotación se edite después. Su tipo de intervención aporta la etiqueta
semisintética, con procedencia separada de las observaciones reales.

El constructor incorpora una vez el control normal de cada revisión utilizada,
sin duplicarlo por cada variante sintética. Para un episodio real etiquetado,
debes seleccionar también una referencia normal anterior de la misma entidad.

| Etiqueta | Patrón observable | Entidades |
|---|---|---|
| `spike` | Salto breve y recuperación observada | Continuas |
| `bias` | Separación sostenida de nivel respecto de la referencia | Continuas |
| `drift` | Separación progresiva en el tiempo | Continuas |
| `noise` | Fluctuaciones y cambios de dirección superiores a los habituales | Continuas |
| `stuck` | Constancia prolongada en reportes efectivamente recibidos | Todos los tipos |
| `unavailable` | Reportes explícitos con estado no válido | Todos los tipos |
| `state_replace` | Estado persistente ausente de la referencia revisada | Binarias y multivalóricas |
| `normal` | Control revisado como funcionamiento correcto | Todos los tipos |
| `environmental_anomaly` | Evento ambiental documentado | Todos los tipos |

Un estado legítimo no representado en la referencia puede parecer desconocido.
Un sensor que informa solo cuando cambia no debe interpretarse como congelado
o indisponible por permanecer en silencio.

### 3. Construir el conjunto

En **Preparación del investigador**, selecciona fragmentos e inyecciones,
configura la **Ventana de análisis** y el **Paso del barrido**, y pulsa
**Construir referencias y ejemplos**. Los valores iniciales son una ventana de
3600 segundos y un paso de 1800 segundos; son configuraciones de partida, no
duraciones validadas para todas las entidades.

El constructor recorre los intervalos seleccionados. Cada ejemplo conserva
entidad, instante de decisión, partición, anotación y revisión, grupo, referencia
y procedencia. Las ventanas de un episodio son dependientes. Los picos conservan
su instante intervenido; una decisión temprana puede resultar indeterminada si
todavía no se observó recuperación.

Las referencias resumen nivel y dispersión robusta, deltas, tasas de cambio,
cadencia observada, constancia y transiciones. Las características comparan la
ventana con esas referencias: desviación de nivel, fluctuación, tendencia,
persistencia, estados inválidos, recuperación y transiciones, entre otras.

La preparación de controles y variantes sintéticas permite describir el
fragmento respecto de su propia referencia normal revisada. Es una preparación
retrospectiva del conjunto, explícitamente identificada como tal; no equivale a
una evaluación prospectiva del pipeline. Las observaciones de cada ventana
terminan siempre en su instante de decisión.

El ZIP del conjunto contiene `examples.csv` (trazabilidad, etiqueta `label` y
columnas `feature.*`) y `manifest.json` (configuración, recuentos y advertencias).
Solo las características permitidas y disponibles alimentan el aprendizaje.
Identificadores, partición, etiqueta, semilla y tipo de inyección son metadatos
o respuestas esperadas, no entradas. Tampoco se proporciona una diferencia con
el CSV original que no existiría en operación.

## Entrenar, evaluar y activar

1. Selecciona un conjunto y pulsa **Entrenar versión candidata**. Solo se
   utilizan sus ejemplos de Desarrollo. Cada tipo de entidad necesita controles
   normales y al menos una clase de falla compatible, además de características
   numéricas comunes suficientes. El modelo conserva exclusiones y soporte.
2. Prepara un conjunto de Validación con la misma duración de ventana y períodos
   posteriores e independientes. Evalúa desde **Modelos y evaluación**. Prueba
   se reserva para la comprobación final y no sustituye la validación.
3. Revisa matriz de confusión, exactitud, macro F1 y cobertura de decisiones
   aceptadas. Las abstenciones se contabilizan como `indeterminate`; no se
   eliminan para mejorar la exactitud. Las métricas de casos verificables se
   identifican por separado de las exploratorias.
4. Pulsa **Activar versión evaluada** cuando hayas revisado sus resultados. Si
   falta validación completa, la interfaz ofrece explícitamente **Activar
   versión experimental**. Entrenar o guardar una etiqueta nunca activa una
   versión automáticamente.

La evaluación vuelve a extraer las características usando exclusivamente las
referencias fijadas en el entrenamiento; no normaliza el conjunto reservado con
su propio período normal. Rechaza grupos compartidos y períodos solapados de la
misma entidad, incluso cuando cambie el archivo o su huella. El intervalo
evaluado debe ser posterior al último período usado para entrenar y disponer
de una referencia anterior de esa entidad.

La marca de evaluación disponible verifica procedencia y cobertura de clases;
no garantiza calidad ni aplica por sí sola un umbral mínimo de exactitud. Los
criterios de aceptación científica deben definirse para el experimento.

Al iniciar una reproducción se fija una instantánea de la versión y las
referencias. Activar otra versión no altera una reproducción en curso ni
reescribe decisiones antiguas. Inicia otro análisis o reinicia la reproducción
para utilizar la nueva versión desde el comienzo.

## Dos modos del motor

**Catálogo de patrones:** funciona sin un modelo entrenado activo. Emplea
condiciones explícitas sobre evolución y referencia, con umbrales de ingeniería
que deben validarse. No repite simplemente `suspected_pattern` de F1. Sin
referencia anterior suficiente, se abstiene.

**Modelo supervisado:** utiliza prototipos por clase y tipo de entidad mediante
centroides y escalado robusto aprendido únicamente con Desarrollo. Se abstiene
cuando una ventana queda fuera de la región de aceptación, dos clases resultan
demasiado cercanas o faltan características. No requiere dependencias nuevas
de machine learning.

El constructor actual entrena sobre características de las ventanas reales o
alteradas y sus referencias. No reproduce automáticamente todos esos casos por
F1 y F2 para obtener puntajes de entrenamiento. En operación, las salidas de
ambos filtros acompañan las decisiones y sus razones; las columnas ausentes
del conjunto de aprendizaje no se convierten en entradas obligatorias del modelo.

No hay entrenamiento automático después de cada evento ni aprendizaje activo
que obligue al usuario final a etiquetar. El investigador puede preparar nuevos
ejemplos y ejecutar otro ciclo de entrenamiento y evaluación.

## Alcance de la evaluación

Las métricas corresponden a **ventanas etiquetadas de F3**
(`classifier_windows_only_not_end_to_end`). No miden cuántas fallas deja pasar
F1, cuáles cierra incorrectamente F2 como ambientales ni el retraso total hasta
un diagnóstico. Muchas ventanas del mismo episodio no equivalen a muchas
fallas independientes.

La evaluación de extremo a extremo debe incluir esas omisiones y respetar la
agrupación de episodios. Un error ambiental terminal de F2 no es una detección
exitosa de F3. El análisis histórico se identifica como retrospectivo; la
reproducción posterior controla que modelo y referencia no hayan usado períodos
futuros o simultáneos al evento diagnosticado.

## API y persistencia

Los recursos se guardan bajo `REVIEW_DATA_DIR/diagnostics` (por defecto
`data/review/diagnostics`): referencias, conjuntos, modelos, trabajos y una base
SQLite con instantáneas y decisiones. Comparten el volumen de revisión y su
exclusión de Git. Los originales importados permanecen intactos.

| Método y ruta | Función |
|---|---|
| `GET /diagnostics/status` | Catálogo, referencias, conjuntos y versiones |
| `GET /diagnostics/sources` | Anotaciones e inyecciones disponibles |
| `POST /diagnostics/build` | Construcción por selección, ventana y paso |
| `GET /diagnostics/jobs/{job_id}` | Progreso o resultado del trabajo |
| `GET /diagnostics/datasets/{dataset_id}` | Resumen y vista previa del conjunto |
| `GET /diagnostics/datasets/{dataset_id}/export` | Descargar conjunto ZIP |
| `POST /diagnostics/train` | Entrenar versión candidata con `dataset_id` |
| `POST /diagnostics/models/{model_id}/evaluate` | Evaluar con `dataset_id` y `partition` |
| `POST /diagnostics/models/{model_id}/activate` | Activar; `allow_exploratory` permite el modo experimental |
| `GET /diagnostics/results` | Decisiones persistidas por origen y paginadas |

Construcción, entrenamiento y evaluación son trabajos con progreso. Consultar
resultados lee decisiones guardadas; no vuelve a ejecutar el clasificador.

El motor está en `backend/app/Diagnostico_Patrones.py`; la preparación y
persistencia, en `backend/app/Diagnostico_Trabajo.py`; la integración, en
`backend/app/main.py`; y la interfaz, en `frontend/src/DiagnosticWorkspace.jsx`.
El anexo anterior `Clasificador_Diagnostico_Reproducible.MD` se conserva como
historia del proyecto y no describe esta implementación.
