# Evaluación de flujos

Esta sección permite componer estadístico, contextual y clasificador como nodos independientes. Incluye plantillas secuenciales, paralelas, filtros individuales y decisiones. Arrastra los nodos y conecta sus puntos laterales; selecciona una conexión para escoger las señales que admite. El inspector permite desactivar filtros, ajustar el umbral estadístico y configurar uniones o decisiones.

Una unión «cualquiera» confirma una señal positiva si alguna rama la aporta. Una unión «todas» exige señales positivas de todas las ramas. Una rama ausente cuenta como no evaluable. Desactivar un filtro deja pasar la señal recibida. Las conexiones sin condiciones dejan pasar todas las señales. Los ciclos y las convergencias sin una Unión explícita se rechazan; Salida reúne los resultados por identidad de lectura.

Guardar y ejecutar conserva primero el borrador actual. Al consultar una ejecución, el lienzo muestra su configuración guardada, en modo de consulta. Volver al borrador conserva la edición actual. Los resultados se pueden consultar por nodo, recorrer por páginas y desplegar para inspeccionar las evidencias. Las lecturas sin ruta de salida se conservan con ese estado.

## Alcance de la ejecución

Carga primero el histórico de Desarrollo. En Evaluación de flujos, carga los eventos futuros o un CSV etiquetado y pulsa Guardar y evaluar. La evaluación recorre todo el archivo futuro en orden temporal, sin el límite retrospectivo de 20.000 filas. Los eventos simultáneos comparten reloj; ningún filtro ve un evento posterior. El estadístico utiliza el motor causal y una copia independiente del estado histórico por nodo; contexto y diagnóstico reciben ventanas pasadas. El clasificador conserva una instantánea de su modelo y referencias durante la ejecución.

La evaluación se ejecuta en segundo plano. Puedes cambiar de sección y volver para consultar el avance. Los resultados terminados persisten en SQLite y sobreviven a la recreación de Docker. Si el servicio se reinicia durante una evaluación, esa tarea incompleta debe ejecutarse de nuevo.

El CSV etiquetado utiliza entity_id,state,last_changed,es_sintetica,tipo,tipo_falla. Las etiquetas se separan antes de cargar los datos y se incorporan a los resultados únicamente después de ejecutar los filtros. La tabla compara conjuntos y flujos; el detalle agrupa por entidad, etiqueta, origen y tipo de falla. Una detección es una sospecha, falla compatible o anomalía ambiental en la salida final. Identificar el tipo exige un diagnóstico compatible que coincida con tipo_falla. Las lecturas sin etiqueta no se consideran normales; las lecturas indeterminadas y sin ruta no se cuentan como detectadas. Los resultados retrospectivos antiguos siguen consultables y se distinguen de las evaluaciones causales.

## Verificación

Desde `frontend`: `node --test src/flow-lab-model.test.js` y `npm run build`.

Desde `backend`: `.venv/Scripts/python -m unittest discover -s tests -p test_flow*.py`, además de `test_labelled_evaluation.py`.
