# Plan de implementación: GAT-LSTM por entidad para Home Assistant

Documento de trabajo para Terra con razonamiento alto. Fecha: 23 de septiembre de 2026.

## Estado y propósito

- Origen: análisis del paper arXiv:2410.11923v1 y revisión del código de esta tesis.
- Modo: planificación de implementación y experimento; no se ejecutó entrenamiento durante la elaboración del plan.
- Versión del plan: `gat_lstm_plan_v1`.
- Estado de verificación: puntos de integración inspeccionados; arquitectura propuesta aún no implementada ni validada experimentalmente.
- Objetivo técnico: incorporar un motor `gat_lstm` seleccionable en F3, con entrenamiento, evaluación, persistencia e interfaz, manteniendo disponibles los motores anteriores.
- Hipótesis experimental: las secuencias de varias entidades relacionadas podrían mejorar el recall de fallas y la cobertura respecto de los prototipos actuales. No presuponer una mejora ni utilizar los porcentajes del paper como objetivo.
- Unidad de predicción: estado diagnóstico de una entidad en un instante, utilizando únicamente su pasado y el de sus vecinos.

Este documento contiene las decisiones iniciales. El ejecutor debe implementarlas por etapas, resolver detalles rutinarios y verificar cada etapa antes de continuar. No necesita volver a preguntar por decisiones ya fijadas aquí. Si faltan etiquetas o dependencias para ejecutar un experimento real, debe terminar el código, los controles y las pruebas que sí pueda completar, y registrar exactamente la limitación.

## 1. Alcance y decisiones fijadas

1. Construir una **variante inspirada en el paper**, no anunciar una reproducción exacta. En el paper los nodos son segmentos; aquí serán entidades de Home Assistant.
2. Primera versión: tres clases diagnósticas `normal`, `environmental_anomaly`, `sensor_fault`. `indeterminate` es abstención, nunca una cuarta clase aprendida.
3. Conservar los subtipos originales en los metadatos, pero agruparlos como `sensor_fault` para esta primera cabeza. La clasificación fina de deriva, sesgo, ruido, etc., queda fuera de esta versión.
4. Soportar entidades continuas, binarias y multivalor mediante codificación diferenciada. La clasificación de tipo de entidad no constituye etiqueta diagnóstica.
5. Entrenar primero un prototipo supervisado; después incorporar preentrenamiento predictivo opcional para aprovechar los históricos sin etiquetas. Ambos caminos forman parte de la entrega, pero se verifican por separado.
6. Mantener F1, el enrutamiento de F2 y el catálogo/prototipos existentes. GAT-LSTM atenderá los pendientes que ya llegan a F3. Un modo sombra separado permitirá evaluar todas las ventanas etiquetadas sin alterar el enrutamiento.
7. Añadir entropía y DTW como características explícitas y evaluables por ablación. No convertir maximizar entropía en una regla automática para seleccionar ventanas.
8. CPU como ejecución inicial; GPU opcional. Usar PyTorch y una capa GAT densa pequeña, sin añadir PyTorch Geometric en esta versión. Límite inicial configurable: 32 entidades por modelo; el histórico seleccionado tiene 14.
9. Grafo y preprocesamiento congelados al terminar el ajuste. No aprender relaciones nuevas con validación, prueba o datos futuros durante la inferencia.
10. No activar automáticamente ningún candidato. Entrenamiento y evaluación guardan versiones; la acción existente de activación sigue siendo explícita en la interfaz.

## 2. Mapa del código existente

Raíz de trabajo: `C:/Users/BlindWizard/Desktop/Tesis/Artifact`.

| Archivo existente | Elemento que se debe conservar o ampliar |
|---|---|
| `backend/app/Diagnostico_Patrones.py` | `diagnose`, `fit_model`, `predict_features`; motores antiguos y contrato de resultados. |
| `backend/app/Diagnostico_Trabajo.py` | `BuildInput`, `DiagnosticWorkbench`, trabajos, datasets, modelos, evaluación, activación y rutas `/diagnostics/*`. |
| `backend/app/Diagnostico_Trabajo.py` | `DiagnosticRunSession`, `_run_snapshot`, `enrich_contextual`: contexto causal e integración de inferencia. |
| `backend/app/Revision_Fragmentos.py` | `ReviewStore`, anotaciones, puntos, intervalos, fuentes y evidencia. |
| `backend/app/Referencia_Revision.py` y `Procedencia_Normalidad.py` | Procedencia y significado de referencias/normalidad. |
| `backend/app/Inyeccion_Fragmentos.py` | Inyecciones reproducibles y vínculo con el episodio original. |
| `backend/app/Filtro_Contextual/Contextual.py` | `FiltroContextualEspacioTemporal.construir_mapa`, aristas y matrices contextuales. |
| `backend/app/Tipos_Entidad.py` y `Estados_Validos.py` | Semántica de tipos y estados; reutilizarla. `NONE` es distinto de `none`. |
| `backend/app/main.py`, `Reproduccion_Validacion.py`, `Evaluacion_Flujos.py` | Llamadas de integración, reproducción causal y medición del flujo. |
| `frontend/src/DiagnosticWorkspace.jsx` | Preparación, selección de motor, entrenamiento, evaluación y visualización. |
| `frontend/src/diagnostic-data.js` y su prueba | Validación de fuentes y etiquetas visibles. |
| `backend/requirements.txt`, `backend/Dockerfile`, `docker-compose.yml` | Dependencias y construcción del servicio. |

Atención a estos detalles comprobados:

- `/diagnostics/train` hoy solo acepta `dataset_id`; no basta con enviarle `method` desde React: hay que ampliar la validación del backend conservando el payload antiguo.
- `diagnose` aplica requisitos de referencias y reportes propios del motor antiguo. El despacho al motor neuronal debe suceder antes de esos requisitos, aplicando los suyos.
- `enrich_contextual` filtra actualmente eventos a entidades pendientes y entrega una ventana de una sola entidad. Para GAT deben estar disponibles también vecinos que F1 no marcó.
- `DiagnosticRunSession` captura una versión por ejecución. Conservar esa inmutabilidad al cargar pesos y preprocesamiento.
- La evaluación antigua tiene alcance de ventanas F3, no de flujo completo. No reutilizar su nombre como si demostrara detección extremo a extremo.
- El README menciona `backend/tests`, pero ese directorio no apareció en el árbol inspeccionado. Verificarlo al comenzar y crearlo si falta. Una ejecución con cero pruebas no cuenta como validación.
- Al preparar este plan, Git de `Artifact` mostraba cambios previos en `data/datosreales/history.csv` y dos bases de caché no rastreadas. No revertirlos ni incluirlos incidentalmente en cambios del modelo.

## 3. Datos, particiones y etiquetas

### 3.1 Fuente temporal canónica

Leer `../HistoricoPuro/particiones_seleccionadas_resumen.json` y comprobar los hashes de sus archivos. No elegir una fuente por parecerse su nombre a “desarrollo”.

| Partición UTC | Intervalo | Uso |
|---|---|---|
| Desarrollo | `[2026-02-01, 2026-05-21)` | Ajuste y subdivisión interna para selección. |
| Validación | `[2026-05-21, 2026-06-26)` | Comparación de candidatos ya configurados. |
| Prueba | `[2026-06-26, 2026-08-01)` | Evaluación final con configuración congelada. |
| Reserva | Agosto | Excluida del entrenamiento y de esta evaluación inicial. |

Estas fechas describen el manifiesto actual, no deben quedar codificadas dentro de la red. El manifiesto de cada experimento debe declararlas.

Dentro de Desarrollo, separar cronológicamente `train_fit` y `dev_tune`: corte al 80 % de su duración temporal como valor inicial, nunca división aleatoria de filas. Reservar en torno al corte las ventanas cuyo contexto o objetivo atraviese el límite. Un episodio y todas sus variantes deben quedar en una sola subdivisión; descartar grupos que crucen el corte y registrar el motivo. Si alguna clase falta, emitir diagnóstico de soporte; no recolocar episodios silenciosamente.

Todos los ajustes usan `train_fit`: vocabularios, escalas, límites de histograma, referencias DTW, grafo, pesos de clase y preentrenamiento. `dev_tune` sirve para parada temprana y política de abstención. Validación y Prueba no suministran información para esos ajustes.

Comprobar independencia por fechas, entidad, identidad del registro original y `parent_episode_id`, incluso cuando cambie el hash por reserializar un CSV. En un grafo, esta comprobación incluye **todas las entidades de contexto**, no solo la entidad etiquetada.

### 3.2 Auditoría previa obligatoria

Crear un comando de auditoría que trabaje en solo lectura sobre las fuentes y produzca un JSON con:

- Archivos, hashes, rangos reales, columnas, filas y entidades.
- Duplicados, conflictos de valores para un mismo registro, faltantes e irregularidad temporal.
- Conteos por etiqueta, estado de revisión, origen, partición real y episodios independientes.
- Etiquetas de punto frente a intervalo; conflictos entre anotaciones.
- Cobertura etiquetada por clase y por entidad en `train_fit`, `dev_tune`, validación y prueba.
- Fuentes importadas cuyo atributo `partition` contradiga sus fechas.
- Aptitud para construir tensores, preentrenar, ajustar tres clases y evaluar. Son cuatro resultados distintos.

Como referencia del estado inspeccionado: 351 anotaciones revisadas, solo 7 de falla; el modelo antiguo activo es exploratorio, con inyecciones y sin evaluaciones guardadas. Estos conteos deben recalcularse, nunca usarse como constantes.

### 3.3 Política de etiquetas

- Admitir `reviewed` y `verified`, conservando la diferencia de evidencia. Los borradores no entran en el ajuste supervisado ordinario.
- Un punto etiqueta exclusivamente la entidad y el instante anotados. Construir su entrada causal terminada exactamente en ese instante; no expandirlo a una hora normal o fallada.
- Un intervalo etiqueta los instantes objetivo que pertenezcan al intervalo. Usar la convención inclusiva/exclusiva real de `ReviewStore`, normalizarla en el manifiesto y probar los extremos. No alterar anotaciones originales.
- La etiqueta describe el estado en el instante objetivo; el contexto previo puede contener otra condición. No asignar la misma etiqueta a cada paso anterior de la secuencia.
- `uncertain`, ausencia de anotación y conflictos: `label_mask=false`. Nunca convertirlos en `normal`.
- Una ventana sin etiqueta puede contribuir al preentrenamiento, no a la pérdida de clasificación.
- Normal y ambiental conservan su significado; una falla con subtipo se transforma a `sensor_fault` únicamente en el objetivo grueso.
- Inyecciones: incluirlas solo con opción explícita en la configuración del experimento, heredar la partición y el grupo del original, y medirlas separadas de datos reales. Una base no revisada obliga a marcar el experimento exploratorio.
- Para inyecciones multivariadas, reconstruir el escenario desde los eventos originales y aplicar las intervenciones correspondientes; el resto de vecinos mantiene sus valores originales. No proporcionar al modelo valores “antes de inyectar”, intensidad, semilla o etiqueta.
- Si no existen ejemplos supervisados de las tres clases en `train_fit`, completar el preentrenamiento y el software, pero no producir un clasificador de tres clases con salidas sin entrenar.

## 4. Contrato temporal y de características

### 4.1 Configuración inicial

Valores de ingeniería para comenzar, sujetos a comparación posterior en Desarrollo:

```json
{
  "method": "gat_lstm",
  "schema_version": "gat-lstm-entities-v1",
  "task": "diagnostic_3class",
  "grid_seconds": 60,
  "sequence_steps": 60,
  "lookback_seconds": 3600,
  "feature_window_seconds": 900,
  "labelled_stride_seconds": 300,
  "graph_top_k": 4,
  "max_nodes": 32,
  "entropy_bins": 16,
  "dtw_reference_count": 3,
  "dtw_band_fraction": 0.1,
  "gat_heads": 4,
  "gat_head_dim": 8,
  "lstm_hidden": 32,
  "dropout": 0.1,
  "batch_size": 16,
  "learning_rate": 0.001,
  "max_epochs": 30,
  "patience": 5,
  "seed": 42,
  "device": "cpu",
  "pretrain_enabled": false
}
```

El contexto bruto necesario incluye `lookback_seconds + feature_window_seconds`: las características del primer paso también miran hacia atrás. El margen de particiones y las consultas deben cubrir esta dependencia, además del horizonte predictivo cuando corresponda.

### 4.2 Una sola función causal para entrenamiento e inferencia

Para un instante objetivo `t`, construir 60 pasos ordenados terminados en `t`, separados por 60 segundos. Para etiquetas puntuales, `t` es exacto, no redondeado al minuto. En barridos sin etiquetas se utiliza una rejilla UTC fija. En cada paso se incluyen solo eventos con timestamp menor o igual al paso; resolver empates de forma estable usando la identidad de fila cuando el protocolo sea por evento.

Reutilizar la semántica contextual de observación: conservar fecha original, edad, ausencia, invalidez y caducidad. No hacer interpolación que mire al futuro, relleno hacia atrás ni prolongar una lectura a través de un reporte inválido. La retención causal del último estado debe respetar la política por entidad/tipo y quedar congelada en el preprocesador.

No inferir indisponibilidad a partir de silencio. Un estado `unavailable` sí es una observación explícita y debe conservarse como característica, aunque su valor numérico sea inválido.

Para la primera versión usar, por paso y nodo:

- Continuas: valor vigente escalado, diferencia respecto del anterior válido sin atravesar invalidez, media/mínimo/máximo/desviación de reportes válidos del intervalo pasado.
- Categóricas: ID de categoría con embedding por entidad; vocabulario aprendido solo en `train_fit`, con tokens distintos `MISSING`, `INVALID` y `OOV`. Los IDs nunca son magnitudes euclidianas. Límite inicial de 64 categorías aprendidas por entidad; exceso pasa a OOV y se informa.
- Comunes: tipo de entidad, número de reportes originales, máscara de valor válido, máscara de reporte nuevo, presencia de invalidez, edad del último reporte y codificación seno/coseno de hora UTC.
- Entropía y máscara de disponibilidad; resumen de distancia DTW y máscara de disponibilidad según la sección siguiente.

Escalar continuas por entidad usando mediana e IQR del ajuste; si IQR es cero, usar escala 1 y registrar condición constante. Valores faltantes se rellenan con cero **después de escalar**, siempre acompañados por sus máscaras. No recortar automáticamente amplitudes extremas que podrían ser fallas.

Congelar `entity_ids`, tipos, vocabularios, orden de características y sus dimensiones. No usar el orden accidental de un CSV. Una entidad nueva es `unsupported_entity` hasta reentrenar; un vecino conocido ausente se representa con máscaras.

### 4.3 Entropía y DTW de esta variante

**Entropía:** `H = -sum(p * log(p))`. Para continuas fijar 16 intervalos sobre el rango de ajuste; extender los extremos a infinito. Si todos los valores de ajuste son iguales, declarar un caso constante reproducible. Para categóricas usar el vocabulario congelado. Contar reportes originales válidos del intervalo causal, no multiplicar observaciones retenidas por cada minuto sin evento. Entropía con menos de tres reportes válidos: máscara falsa. Normalizar por `log(K)` del alfabeto fijo cuando `K>1`; si `K=1`, valor 0. Esta normalización es una adaptación explícita, diferente de dividir por `log(w)` en el paper.

**DTW:** primera versión solo para continuas y frente a referencias normales de la **misma entidad**, para evitar comparar unidades o semánticas incompatibles. Elegir hasta tres ventanas normales revisadas de `train_fit`, sin solapamiento, mediante una regla determinista: primera, central y última entre las elegibles, deduplicadas. Escalar con el preprocesador congelado, no normalizar cada ventana por su propia media, porque eso borraría sesgos.

Distancia local absoluta en una dimensión; caminos con extremos fijados y movimientos `(1,0)`, `(0,1)`, `(1,1)` dentro de banda `ceil(0.1 * longitud)`. Normalizar el costo acumulado por longitud del camino, con desempate determinista. Características: mínimo y mediana de distancias, y `1/(1+d_min)`. Si hay algún paso inválido en una secuencia comparada, excluir esa comparación en v1; si no quedan referencias, máscara falsa. No sustituir una distancia ausente por similitud perfecta.

DTW no construye aquí todas las aristas: el grafo viene de F2. Documentar esta diferencia respecto del paper. Evaluar entropía y DTW mediante banderas `use_entropy` y `use_dtw`; sus algoritmos deben ser independientes de la red y reutilizables después por otros filtros. No modificar todavía los puntajes de F1/F2.

## 5. Grafo y arquitectura exacta

### 5.1 Grafo congelado

Construir un mapa F2 nuevo en memoria usando exclusivamente eventos de `train_fit`, sin sobrescribir el mapa operativo. Guardar en el modelo un snapshot y su huella. Verificar que las subdivisiones internas de `construir_mapa` tampoco accedan a eventos exteriores al rango recibido.

De sus relaciones aceptadas, elegir hasta cuatro fuentes por destino por fuerza absoluta de asociación, con desempate por ID. Interpretación del tensor: `A[i,j]=true` permite al destino `i` recibir información de la fuente `j`. Agregar autoarista a todo nodo. Conservar signo, retardo y procedencia como metadatos; v1 usa la topología, no multiplica automáticamente atención por correlaciones con signo. El retardo de F2 no se aplica como desplazamiento adicional en esta versión.

Un nodo aislado conserva su autoarista. Si todo el grafo tiene solo autoaristas, permitir pruebas técnicas pero marcar `spatial_evidence=false`; no describirlo como evidencia de que el contexto espacial mejora el diagnóstico.

### 5.2 Tensores

```text
X:              float32 [B, T, N, F]  características numéricas y máscaras
state_ids:      int64   [B, T, N]     categorías por entidad
adjacency:      bool    [N, N]        topología congelada
node_present:   bool    [B, T, N]     disponibilidad informativa, no solo valor válido
y:              int64   [B, N]        clase en el instante final; -1 sin etiqueta
label_mask:     bool    [B, N]        posiciones supervisadas
logits:         float32 [B, N, 3]
```

`node_present` puede ser verdadero ante un reporte explícito inválido; de otro modo se eliminaría la evidencia para indisponibilidad. Distinguirlo de `valid_value_mask`. Precisar esa regla en pruebas de presencia/ausencia y retención causal.

Metadatos por ejemplo: `sample_id`, objetivo temporal exacto, fuente/hashes, `entity_ids`, rangos completos de contexto, etiqueta y anotación por objetivo, `parent_episode_id`, partición, origen de evidencia y `injection_id` si corresponde. Estos metadatos no son características predictoras.

### 5.3 Red

```text
codificación diferenciada + proyección a 32 dimensiones
    -> GAT de 4 cabezas x 8 dimensiones por instante
    -> ELU + dropout externo
    -> reorganizar [B,T,N,32] a [B*N,T,32] sin alterar el tiempo
    -> LSTM unidireccional compartida, hidden=32, una capa, batch_first=True
    -> último estado [B*N,32]
    -> capa lineal a 3 logits
    -> reorganizar a [B,N,3]
```

No usar pooling global entre entidades. Reiniciar estados LSTM entre ejemplos independientes; no mantener un estado oculto compartido entre lotes o ejecuciones.

Implementar atención GAT con `e_ij=LeakyReLU(a_src^T W h_i + a_dst^T W h_j)` y softmax sobre vecinos admitidos. Esta forma aditiva es equivalente a aplicar un vector a la concatenación. Enmascarar vecinos ausentes, conservar un camino seguro para el propio nodo y manejar filas totalmente ausentes sin softmax de todos `-inf`. Esas filas producen representación neutra y rechazo informativo, no NaN. Comprobar gradientes y equivalencia de una capa con un cálculo manual pequeño.

El dropout de la LSTM de una sola capa no implementa el dropout entre capas que aquí interesa: usar el módulo externo indicado. No añadir una LSTM bidireccional.

## 6. Entrenamiento y abstención

### 6.1 Ajuste supervisado

- Pérdida de entropía cruzada sobre `label_mask`; pasar logits, no probabilidades ya normalizadas.
- Peso por clase basado en episodios etiquetados de `train_fit`, limitado a un máximo 5; normalizar pesos. Compensar también el número de ventanas de cada episodio para que una anotación larga no domine solo por su duración.
- Minilotes con cero posiciones supervisadas: omitir sin producir NaN ni gradiente falso. Si toda la época está vacía, terminar con `insufficient_labels`.
- Adam, tasa 0.001, máximo 30 épocas, paciencia 5, clipping de norma del gradiente a 1.0. Semilla 42 para pruebas; evaluación experimental posterior con 42, 43 y 44.
- Elegir checkpoint por pérdida supervisada en `dev_tune`; no usar Prueba. Si `dev_tune` no tiene etiquetas, ejecutar solo smoke o preentrenamiento y dejar pendiente el ajuste diagnóstico real.
- Registrar pérdidas, conteos supervisados, memoria, tiempo, semilla, dispositivo y versiones. Cualquier pérdida no finita debe terminar ese trabajo con un error explícito.

### 6.2 Preentrenamiento opcional con datos sin etiquetas

Reutilizar el mismo encoder GAT-LSTM y añadir cabezas temporales auxiliares por tipo. Predecir el valor/estado del siguiente paso a partir de una secuencia que termina antes de ese objetivo. Continuas: Huber sobre observaciones futuras válidas; categóricas: entropía cruzada sobre estados futuros conocidos. Promediar por entidad y tipo para no favorecer al sensor que más reporta.

Evaluar la pérdida únicamente donde exista una observación original válida del objetivo; no generar millones de objetivos a partir de estados copiados. Excluir conflictos y, si están identificadas, fallas supervisadas de los ejemplos destinados a aprender comportamiento habitual. Los casos no anotados pueden contener anomalías: registrarlo como limitación, no declarar que todo el histórico es normal.

Primer horizonte: 60 segundos. Si un tipo carece de objetivos suficientes a ese horizonte, informar cobertura y omitir su término auxiliar; no inventar observaciones. El encoder de ese tipo todavía puede aprender durante ajuste supervisado. Si no hay objetivos auxiliares para ningún tipo, preentrenamiento no ejecutable con esa configuración.

Usar solo `train_fit`, incluyendo el objetivo futuro dentro de sus límites. Seleccionar checkpoint por pérdida predictiva de `dev_tune`. Transferir únicamente el encoder, inicializar la cabeza diagnóstica y luego ajustar con etiquetas.

Un encoder preentrenado sin cabeza diagnóstica ajustada no puede activarse como clasificador. Su error predictivo tampoco se transforma por sí solo en una etiqueta de falla física.

### 6.3 Abstención y compatibilidad de salida

Rechazos estructurales: entidad desconocida, fuente/entidades incompatibles con el manifiesto, ausencia total de información del objetivo, historia requerida no disponible, checkpoint incompatible o dependencia neuronal ausente. Una invalidez explícita no equivale a falta total de información.

Para probabilidades softmax usar `confidence_kind="uncalibrated_softmax"`. No presentarlas como probabilidad calibrada de avería. No reutilizar el campo `compatibility` del catálogo con un significado distinto: mantenerlo `null` para GAT y añadir `confidence` y `class_scores`.

Política de rechazo inicial: `max_score >= threshold` y diferencia entre las dos mejores clases `>= margin`. Seleccionar ambos en `dev_tune` sobre una malla fija: threshold en `{0.4,0.5,0.6,0.7,0.8,0.9}`, margin en `{0,0.05,0.1,0.2}`. Maximizar F1 macro contando abstenciones como no aciertos; desempatar por mayor cobertura y luego menor umbral. Guardar la política. Sin soporte de las tres clases en `dev_tune`, se pueden emitir resultados exploratorios, pero `validated=false` y `rejection_policy_validated=false`.

Contrato de salida:

```json
{
  "entity_id": "...",
  "timestamp": "...",
  "status": "compatible | no_fault_evidence | indeterminate",
  "predicted_label": "normal | environmental_anomaly | sensor_fault | null",
  "fault_type": null,
  "method": "gat_lstm",
  "compatibility": null,
  "confidence": 0.0,
  "confidence_kind": "uncalibrated_softmax",
  "class_scores": {},
  "needs_review": true,
  "reasons": [],
  "model_version": "...",
  "preprocessor_version": "...",
  "graph_version": "...",
  "input_coverage": {},
  "evaluation_scope": "..."
}
```

Mapeo: falla aceptada -> `compatible`; normal/ambiental aceptado -> `no_fault_evidence`; rechazo -> `indeterminate`, `predicted_label=null`. Mostrar la etiqueta real en UI, para que ambiental no se pierda detrás de un texto genérico. `fault_type=null` expresa que la versión no identifica subtipo.

No atribuir explicaciones causales a los pesos de atención. Las razones verificables son datos disponibles, clases comparadas, rechazo, versión y vecinos utilizados.

## 7. Archivos nuevos y responsabilidades

Crear el paquete `backend/app/GAT_LSTM/` con módulos pequeños:

| Módulo nuevo | Responsabilidad |
|---|---|
| `__init__.py` | Importación ligera; no cargar PyTorch al iniciar el backend ordinario. |
| `config.py` | Configuración validada y versiones de esquemas. |
| `audit.py` | Fuentes, rangos, etiquetas, grupos y aptitud del experimento. |
| `temporal.py` | Una función causal de ventanas utilizada tanto offline como online. |
| `features.py` | Escalas, estados, entropía y DTW, con ajuste/transformación separados. |
| `graph.py` | Snapshot F2, orientación, top-k, autoaristas y hashes. |
| `dataset.py` | Índice de ejemplos, máscaras, procedencia y lectura por lotes. |
| `network.py` | GAT densa, LSTM y cabezas; importación opcional de PyTorch. |
| `training.py` | Ajuste supervisado, preentrenamiento y selección de checkpoints. |
| `evaluation.py` | Predicciones, métricas, cobertura y comparación homogénea. |
| `runtime.py` | Carga, inferencia, validación de entradas y adaptación al contrato F3. |

Funciones públicas esperadas, con tipos/dataclasses concretados en código:

```python
audit_sources(config, review_store) -> AuditReport
fit_preprocessor(train_fit_events, annotations, config) -> PreprocessorState
build_graph(train_fit_events, entity_catalog, config) -> GraphState
build_causal_sample(event_provider, target_time, preprocessor, graph, config) -> Sample
build_sequence_dataset(audit, preprocessor, graph, config) -> DatasetManifest
pretrain_encoder(dataset_manifest, config, progress) -> CandidateManifest
train_classifier(dataset_manifest, config, encoder=None, progress=None) -> CandidateManifest
evaluate_candidate(model_id, dataset_id, partition, progress) -> EvaluationReport
predict_at(snapshot, event_provider, entity_id, timestamp) -> DiagnosticResult
```

Persistencia propuesta, bajo `REVIEW_DATA_DIR/diagnostics`:

- Mantener `models/{id}.json` como manifiesto consultable por el workbench.
- Añadir `weights/{id}.pt`, `preprocessors/{id}.json`, `graphs/{id}.json`.
- Datasets: índice y arrays numéricos en `sequences/{id}/`; permitir memoria mapeada y lectura por lotes. No guardar tensores gigantes dentro del JSON de estado ni volver a leer todo el histórico en cada predicción.
- Añadir `experiments/{id}/audit.json`, `metrics.jsonl`, `predictions.csv`, `report.json` y resumen Markdown.
- Exportación: configuración, índices, procedencia, grafo y preprocesador. Nunca cargar arrays con `allow_pickle=True`.
- Pesos: `state_dict`; carga local con `weights_only=True`, `map_location="cpu"`, verificación de hash y compatibilidad de dimensiones. No aceptar rutas arbitrarias desde la API.
- Publicar primero archivos auxiliares completos, después manifiesto completo con escrituras atómicas. Un trabajo cancelado no debe aparecer como modelo válido.
- Mantener `active.json` y los JSON antiguos legibles. `method` ausente debe continuar por el motor antiguo.

Dependencias: crear `backend/requirements-gat.txt` con versión exacta de PyTorch comprobada para Python 3.12 y el host. No escribir `latest` ni asumir CUDA. Añadir una opción de construcción Docker documentada para instalar ese conjunto. Backend sin PyTorch: arranca normalmente, anuncia `gat_lstm.available=false` y devuelve un error claro al solicitar operaciones neuronales.

## 8. Etapas ejecutables y criterios de aceptación

Realizar una etapa a la vez. Actualizar un registro `GAT_LSTM_PROGRESO.md` con archivos cambiados, pruebas realmente ejecutadas, resultados y próxima etapa. No marcar una etapa terminada por tener solo esqueletos o mocks. Cada etapa puede dejar el candidato sin activar.

### P0. Inventario y auditoría

**Acciones:** revisar estado Git, instrucciones locales, archivos de la tabla anterior, dependencias, datos y manifiestos. Implementar `audit.py` y `scripts/gat_lstm_audit.py`. Verificar el árbol real de pruebas.

**Aceptación:** informe reproducible con aptitud por operación; detecta partición declarada que contradice fechas y grupos compartidos. No escribe en CSV originales, anotaciones, bases operativas ni `active.json`.

### P1. Contratos y preparación causal

**Acciones:** `config.py`, `temporal.py`, `features.py`, `dataset.py`; tres clases, máscaras, vocabulario y transformaciones congeladas. Trabajar inicialmente con fixtures pequeños.

**Aceptación:** cambiar eventos posteriores a `t` no cambia su entrada; una etiqueta puntual produce un único objetivo; un vecino sin etiqueta conserva máscara falsa; modificar valores de Prueba no altera escalas ni vocabularios. `NONE` mantiene su semántica. Entropía y DTW tienen pruebas matemáticas pequeñas.

### P2. Grafo y red mínima supervisada

**Acciones:** `graph.py`, `network.py`, dependencia opcional. Probar primero grafo pequeño controlado; después snapshot de F2 de `train_fit`.

**Aceptación:** forward/backward finitos para continuas y categóricas, nodo aislado, vecino faltante y observación inválida; las dimensiones finales son `[B,N,3]`. Cambiar un vecino conectado cambia la representación del objetivo en una prueba con pesos controlados; un vecino desconectado no la cambia. El orden temporal llega intacto a LSTM. No hay pooling de entidades.

### P3. Entrenamiento, persistencia y evaluación offline

**Acciones:** `training.py`, `evaluation.py`, guardado/carga y comandos `scripts/gat_lstm_train.py` y `scripts/gat_lstm_evaluate.py`. Entrenamiento sintético mínimo para validar el mecanismo, sin presentarlo como evidencia de tesis.

**Aceptación:** la red puede sobreajustar un fixture diminuto con relación conocida; las predicciones son iguales antes/después de recargar dentro de tolerancia; lotes sin etiquetas no aportan pérdida; no se entrenan clases ausentes; configuración e identidad de cada muestra quedan guardadas. Matriz de confusión con abstenciones y denominadores correctos.

### P4. Preentrenamiento opcional

**Acciones:** cabezas predictivas y CLI `scripts/gat_lstm_pretrain.py`; transferencia del encoder y separación de checkpoints predictivo/diagnóstico.

**Aceptación:** no entra ningún objetivo posterior al límite permitido; copiar estados no crea objetivos observados; la pérdida ignora posiciones faltantes; no se puede activar un encoder sin cabeza diagnóstica. Comparación posterior puede desactivar preentrenamiento con la misma arquitectura.

### P5. Integración backend sin regresiones

**Acciones:** ampliar `BuildInput`/payloads con esquemas discriminados por `method`, despacho del workbench y capacidad neuronal; mantener payloads antiguos. Reutilizar los trabajos asíncronos y el seguimiento por `job_id`.

**Aceptación:** una solicitud antigua sigue entrenando prototipos; un dataset neuronal no puede enviarse a un motor incompatible; entrenamiento falla con mensaje específico si falta PyTorch; jobs registran error sin publicar candidato incompleto; entrenamiento no modifica `active.json`.

API que debe quedar documentada y probada:

```text
GET  /diagnostics/status                         añade capabilities y método por modelo
POST /diagnostics/build                         method opcional; ausente = comportamiento antiguo
POST /diagnostics/train                         dataset_id + method/config/encoder_id opcionales
POST /diagnostics/pretrain                      nuevo, mismo protocolo asíncrono
POST /diagnostics/models/{id}/evaluate           conserva ruta; despacha por método
POST /diagnostics/models/{id}/activate           comprueba checkpoint y dependencias
POST /diagnostics/models/{id}/deactivate         conserva funcionamiento
GET  /diagnostics/jobs/{id}                      conserva seguimiento
```

El build neuronal necesita IDs de fuentes históricas además de anotaciones: así puede recuperar vecinos y datos no etiquetados. Definirlos en el esquema, comprobarlos contra `ReviewStore` y no recibir rutas locales libres. Reutilizar la auditoría para establecer los rangos reales.

Para v1 permitir un solo trabajo neuronal pesado simultáneo por servicio. Mantener API de estado receptiva; registrar progreso por época, `heartbeat_at` y tiempo límite configurable del trabajo. Un trabajo sin completar permanece fallido/incompleto, nunca “validado”. No iniciar automáticamente entrenamientos largos sobre todos los datos para demostrar que la UI funciona.

### P6. Inferencia causal dentro del flujo

**Acciones:** integrar `runtime.py` con `DiagnosticRunSession`, `_run_snapshot` y `enrich_contextual`. Auditar también todos sus llamadores en `main.py` y reproducción: agregar vecinos dentro del método no basta si el llamador ya los descartó.

**Aceptación:** el motor recibe entidades no sospechosas que sirven de contexto; no usa eventos futuros aunque el lote los contenga; F2 terminal conserva su resultado; una ejecución mantiene el mismo checkpoint aunque cambie el modelo activo; historia, replay y dataset usan el mismo constructor causal. No reconstruye grafo ni carga pesos por cada fila.

Mantener un buffer temporal acotado por ejecución para contexto y un registro de IDs que se pode con él; evitar crecimiento ilimitado de `events`/`seen` en el nuevo camino. Agrupar decisiones por instante para reutilizar un forward multientidad y conservar la semántica por evento cuando existan empates. Una selección de nuevo motor no recalcula resultados antiguos silenciosamente.

### P7. Interfaz

**Acciones:** ampliar `DiagnosticWorkspace.jsx`, `diagnostic-data.js` y estilos existentes. Añadir selector de motor en preparación/modelos, opciones de preentrenamiento, resumen de etiquetas y soporte, progreso y origen de evaluación.

**Aceptación:** la interfaz explica tres clases y abstención; para GAT permite falla sin subtipo sin debilitar la validación antigua de prototipos; distingue puntuación softmax de compatibilidad; muestra fallo por dependencia ausente, clases insuficientes y versión experimental. Modelo activo antiguo sigue visible y operable. No enseñar tensores o parámetros internos en la vista del usuario final; reservarlos a detalles del investigador.

### P8. Comparación y documentación

**Acciones:** `scripts/gat_lstm_compare.py`, reporte de resultados, README y documento de método. Pruebas de regresión backend/frontend y smoke Docker si el entorno lo permite. Ejecutar un smoke real acotado solo si los datos reúnen las condiciones de la auditoría; registrar por qué no se ejecuta una etapa de entrenamiento real cuando falten etiquetas.

**Aceptación:** resultados comparables por IDs de muestra, periodos y clases; pérdidas F1/F2 y abstenciones diferenciadas; reporte no confunde fixtures técnicos con resultados empíricos. Documento final separa funcionamiento comprobado, métricas observadas y validación pendiente. Entrega completa de código e integración aunque la mejora empírica quede sin demostrar.

## 9. Pruebas mínimas que detectan errores reales

Crear en `backend/tests/` utilizando `unittest` y fixtures pequeños, más las pruebas JS existentes. No levantar servicios contra bases reales en las pruebas.

| Archivo propuesto | Casos esenciales |
|---|---|
| `test_gat_lstm_data.py` | Futuro invariante, etiquetas de punto, conflictos, límite de intervalo, retención causal, fuente con partición falsa, OOV y orden de entidades. |
| `test_gat_lstm_features.py` | Entropía constante/binaria, bins congelados, DTW idéntica, desplazamiento acotado, huecos y ausencia de referencia. |
| `test_gat_lstm_graph.py` | Orientación source/target, top-k determinista, autoaristas, nodo sin vecinos y grafo aprendido solo con ajuste. |
| `test_gat_lstm_network.py` | Formas, cálculo manual GAT, máscara de vecinos, ausencia completa, efecto temporal, gradientes y fixture aprendible. |
| `test_gat_lstm_training.py` | Pérdida solo etiquetada, cero objetivos, pesos por episodio, preentrenamiento causal y checkpoint recargable. |
| `test_gat_lstm_integration.py` | Payload antiguo, dependencia ausente, snapshot por ejecución, F2 terminal, contexto completo y no autoactivación. |
| `test_gat_lstm_evaluation.py` | Ejemplo calculable a mano: TP/FP/FN, rechazos, clase ausente, falsa alarma ambiental y particiones solapadas. |
| `frontend/src/diagnostic-gat.test.js` | Selección de motor, clases gruesas, estados experimental/incompleto, puntuaciones y compatibilidad antigua. |

Cuando PyTorch no esté instalado, solo las pruebas neuronales pueden aparecer omitidas con motivo explícito. Ejecutarlas en el entorno neuronal antes de dar P2-P6 por verificadas; una suite con skips no demuestra esos componentes.

Comandos desde `Artifact/backend`, una vez creadas las pruebas y preparado su entorno:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_gat_lstm*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Desde `Artifact/frontend`:

```powershell
$gatJsTests = @(Get-ChildItem -LiteralPath './src' -Filter '*.test.js' | ForEach-Object { $_.FullName })
node --test $gatJsTests
npm run build
```

Los comandos de auditoría/preentrenamiento/entrenamiento/evaluación/comparación son entregables nuevos. Cada uno debe tener `--help`, configuración JSON validada y salida explícita; documentar las invocaciones exactas solo después de implementar sus argumentos. `scripts/post_docker_test.py` es un smoke operativo, no una prueba de mejora diagnóstica.

## 10. Comparación experimental y criterio de éxito

Comparar sobre el mismo conjunto de objetivos y la misma procedencia:

| Experimento | Propósito |
|---|---|
| B0: catálogo/prototipos actuales | Referencia operativa; proyectar subtipos a las tres clases al medir. |
| B1: prototipos con entropía y DTW | Aislar el aporte de características; versión nueva, sin cambiar antiguos modelos. |
| B2: encoder por entidad + LSTM sin vecinos | Aislar el aporte temporal con la misma preparación. |
| B3: GAT-LSTM supervisado | Medir el aporte de relaciones entre entidades. |
| B4: GAT-LSTM con preentrenamiento | Medir el aporte del histórico sin etiquetas. |

Para B1, usar un esquema de características nuevo y su escalador; no modificar la lista que esperan los modelos antiguos. Para B2, sustituir GAT por la proyección por nodo y conservar dimensiones, cabeza y protocolo. Registrar presupuesto de entrenamiento por variante.

Medir dos alcances separados:

1. **Clasificador aislado/sombra:** todos los objetivos etiquetados; cada motor recibe exactamente el contexto que su contrato permite. Si no puede decidir, se registra abstención. No eliminar esos ejemplos para favorecerlo.
2. **Flujo completo:** reproducción F1 -> F2 -> F3 con umbrales y agrupación fijos entre variantes. Fallas que F1/F2 no envían a F3 cuentan como fallas no detectadas; registrar también detección por episodio y demora. No confundir una lectura agrupada por F1 con una falla nueva omitida.

Métricas: recall y precisión de `sensor_fault`, F1 macro sobre el vocabulario fijo de tres clases, matriz con columna de abstención, cobertura, precisión entre aceptadas, falsas alarmas sobre normales y ambientales por separado, latencia p50/p95 y costo de preparación. Informar soporte por clase y episodios, no solo cantidad de ventanas. Para clases sin soporte, métricas no estimables `null` y evaluación incompleta; nunca mostrarlas como perfectas.

La selección en `dev_tune` y el informe final son distintos. En el informe mostrar métricas con abstenciones y métricas entre aceptadas. La cobertura es `aceptadas / objetivos elegibles`; rechazos estructurales también se informan con su denominador, sin desaparecer de la comparación.

Separar resultados de revisión histórica, verificación independiente, inyección sobre normal revisado e inyección exploratoria. Si hay episodios suficientes, intervalos de incertidumbre por remuestreo de episodios, no de ventanas correlacionadas. No fijar un número de ventanas como garantía de validez.

**Éxito de implementación:** entrenamiento/inferencia/persistencia/integración funcionan con pruebas y trazabilidad; los motores anteriores conservan comportamiento. **Éxito empírico:** mejora observada del recall/F1 de falla a una precisión y carga de falsas alarmas comparables, con cobertura e incertidumbre reportadas. No inventar un umbral de 95 % o 99 % para declarar éxito.

`validated` significa evaluación completa bajo el protocolo definido, no superioridad demostrada ni confirmación física. Añadir `evaluation_status`, `label_support` y `evidence_origins`; mantener `validated=false` ante clases ausentes, fuga detectada, ajuste exploratorio o política sin validar. No modificar retroactivamente el significado de versiones antiguas.

## 11. Fallos previsibles y respuesta esperada

| Situación | Acción del ejecutor |
|---|---|
| No hay etiquetas suficientes de las tres clases | Entregar preparación, preentrenamiento si procede, pruebas e informe de soporte; no fabricar etiquetas ni entrenar una cabeza sin clases. |
| Base importada “desarrollo” contiene meses posteriores | Rechazarla para ajuste en ese rango y seleccionar rangos/fuentes correctos, sin modificar el archivo original. |
| Etiquetas antiguas no se pueden asociar inequívocamente a filas actuales | Excluirlas del ajuste y emitir lista de revisión; no aproximar IDs/fechas silenciosamente. |
| No hay referencias normales para DTW | Máscara de distancia ausente y aviso de cobertura; GAT-LSTM puede funcionar sin esa característica. |
| Grafo solo con autoaristas | Continuar pruebas, informar ausencia de contexto espacial útil y comparar con B2. |
| PyTorch no se puede instalar | Conservar servicio convencional, informar operación no disponible y dejar reproducible el paso pendiente; no simular métricas. |
| Datos irregulares o entidad desconocida | Aplicar máscaras o rechazo documentado; no rellenar datos como normales. |
| Falla un test significativo | Corregir la etapa antes de continuar; no bajar la exigencia del test para ocultar el defecto. |
| GAT-LSTM no supera el baseline | Entregar el resultado observado, mantener candidato inactivo y explicar qué métricas no mejoraron. |

## 12. Lista de entrega

- [ ] Auditoría y manifiestos verificables, sin modificar fuentes originales.
- [ ] Preparador causal común a entrenamiento e inferencia, con pruebas contra fuga.
- [ ] Entropía/DTW reutilizables, sus máscaras y documentación de diferencias con el paper.
- [ ] Grafo congelado, GAT-LSTM por entidad y soporte de los tres tipos de nodo.
- [ ] Ajuste supervisado y preentrenamiento opcional implementados y diferenciados.
- [ ] Pesos, preprocesador, grafo, política de abstención y versiones recargables.
- [ ] API compatible, UI funcional y trabajos con progreso/error.
- [ ] Inferencia con vecinos, snapshot estable y decisión causal.
- [ ] Comparadores y reportes por ventana, episodio y procedencia.
- [ ] Pruebas ejecutadas con resultados reales, incluyendo las neuronales cuando la dependencia esté disponible.
- [ ] README, comandos, limitaciones y registro de avance actualizado.
- [ ] Modelo candidato guardado sin reemplazar automáticamente el activo.

## 13. Prompt de inicio para Terra alto

Copiar el siguiente texto en la tarea de implementación después de seleccionar Terra y razonamiento alto. El documento es autosuficiente; no requiere leer la conversación anterior.

> Implementa en `C:/Users/BlindWizard/Desktop/Tesis/Artifact` la variante descrita en `PLAN_IMPLEMENTACION_GAT_LSTM.md`. Lee el documento completo y las instrucciones locales antes de editar. Usa sus decisiones como contrato: nodo=entidad, diagnóstico en tres clases por instante, contexto causal, GAT seguido de LSTM por entidad, etiquetas parciales y preentrenamiento opcional. Trabaja secuencialmente de P0 a P8 y lleva `GAT_LSTM_PROGRESO.md` con pruebas y resultados reales. No sustituyas el diseño por un clasificador tabular presentado como GAT ni dejes integraciones simuladas. Conserva los motores y payloads antiguos; termina cada etapa y sus verificaciones antes de continuar. No modifiques históricos, anotaciones ni cambios previos del usuario, y no actives automáticamente modelos. Si faltan etiquetas para un experimento concluyente, termina el código y las pruebas técnicas y entrega el informe de soporte pendiente, sin inventar resultados. Resuelve los detalles rutinarios dentro del plan; pregunta solo si una contradicción real impide continuar una parte necesaria. Al terminar, entrega archivos cambiados, comandos ejecutados, pruebas aprobadas, limitaciones y estado del candidato.

Para reanudar otra sesión: leer este plan y `GAT_LSTM_PROGRESO.md`, verificar estado real del código y continuar desde la primera etapa pendiente; no dar por ejecutados comandos solo porque aparezcan documentados.

## Fuentes y límites de atribución

- Singh et al. (2024), [Spatial-Temporal Bearing Fault Detection Using Graph Attention Networks and LSTM](https://arxiv.org/abs/2410.11923): entropía, ventanas, DTW y atención. La selección de nodos por entidad, cabeza gruesa, máscaras, preentrenamiento y protocolo local de este plan son decisiones propias de adaptación.
- Zhao et al. (2020), [Multivariate Time-series Anomaly Detection via Graph Attention Network](https://arxiv.org/abs/2009.02040): antecedente de predicción/reconstrucción autosupervisadas; este plan no pretende reproducir MTAD-GAT.
- [PyTorch LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html): forma batch/secuencia/característica, estado final y dropout entre capas.
- [PyTorch torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html): carga de pesos y selección de dispositivo.
- [PyTorch reproducibilidad](https://docs.pytorch.org/docs/2.14/notes/randomness.html): registrar versiones y plataforma; una misma semilla no garantiza identidad entre CPU/GPU o versiones.

La revisión de APIs no demuestra que esa versión de PyTorch ya esté instalada o sea el paquete elegido para el proyecto. El ejecutor debe comprobar compatibilidad e instalar una versión exacta en el entorno del proyecto. Este plan fue elaborado con asistencia de IA y revisión de los archivos disponibles; no constituye un resultado experimental de la tesis.
