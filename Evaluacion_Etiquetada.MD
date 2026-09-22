# Evaluación con datos etiquetados

La pestaña **Evaluación** agrupa Conjunto de datos real, Conjunto de datos
etiquetados y Evaluación con eventos futuros. Carga primero Desarrollo y espera
la preparación de la base del detector. Después carga un CSV etiquetado de
fechas posteriores. Las etiquetas no se usan para preparar ni ajustar filtros.

El servidor valida las seis columnas, conserva la referencia en SQLite y entrega
al cargador de eventos un CSV con exactamente `entity_id,state,last_changed`.
Ese archivo queda listo en Evaluación con eventos futuros. Avanzar por lotes o
iniciar la reproducción ejecuta el flujo causal existente.

La tabla agrupa por sensor y muestra datos normales; fallas no inyectadas,
fallas inyectadas y anomalías detectadas/total; falsos positivos; procesados/total
y filas sin etiqueta. Detección significa sospecha emitida por el filtro
estadístico, no clasificación correcta de su causa. Las rutas contextual y
clasificadora existentes conservan sus propios resultados en sus pestañas.

Cada fila cuenta como máximo una vez. Un falso positivo es una fila etiquetada
normal que emitió una sospecha. NULL no se interpreta como normal. Los
 denominadores abarcan todo el archivo; durante la reproducción los conteos son
parciales. Alertas sin una lectura de origen inequívoca se informan separadamente
y no se asignan a todas las lecturas de una ventana. El conteo usa todos los
resultados, independientemente del límite de 500 resultados visibles.

Reiniciar conserva la referencia y pone las detecciones a cero. Cargar otro CSV
reemplaza la referencia activa; cargar eventos sin etiquetas desactiva la tabla
etiquetada. Endpoints: POST `/validation/labelled/upload` y GET
`/validation/labelled/metrics`. La referencia por fila vive en `evaluation_truth`
en `data/cache/validation_replay_results.sqlite3`, separada de los eventos.
