# Clasificador diagnóstico reproducible basado en reglas versionadas

## Estado del documento

Este anexo define la especificación a priori `diagnostic-rule-list-v1.0.0`
para el tercer filtro del flujo de diagnóstico. Su propósito es reemplazar la
función conceptual \(f\) de la sección **Filtro de clasificación** por una
lista de reglas determinista, auditable y directamente implementable.

La especificación no constituye un resultado experimental. Las reglas y los
umbrales se fijan antes de disponer de una base de verdad real suficiente y,
por tanto, todavía no se encuentran validados como estimadores del desempeño
en operación. El histórico real disponible en
`Artifact/data/datosreales/history.csv` cubre del 1 al 31 de mayo de 2026 y
contiene solamente `entity_id`, `state` y `last_changed`. No contiene una
etiqueta independiente que permita saber si cada episodio corresponde a una
lectura normal, una anomalía ambiental o una falla.

El mes disponible permite comprobar la compatibilidad del esquema, ejecutar
el flujo en modo sombra, medir cobertura y costo computacional y revisar la
trazabilidad de cada decisión. No permite seleccionar reglas por desempeño ni
estimar sensibilidad, precisión, \(F_2\) o una matriz de confusión diagnóstica.
No se utilizará la base simulada para seleccionar, ajustar ni validar el
clasificador; provisionalmente se trabaja únicamente con este mes real.
La validación definitiva queda pendiente hasta disponer de al menos cinco
meses de histórico y episodios etiquetados mediante evidencia real
independiente o fallas controladas ejecutadas en tiempo real. El umbral de
cinco meses es una decisión metodológica a priori de este anexo y es más
exigente que el alcance mínimo actualmente descrito en `Tesis.Tex`, líneas
214--228.

## Trazabilidad con la tesis

La tesis ya establece los elementos conceptuales que sustentan el contrato:

- El clasificador integra la evidencia estadístico-temporal y contextual
  (`Tesis.Tex`, líneas 1362--1376).
- Detectar una lectura atípica no equivale a conocer su origen
  (`Tesis.Tex`, líneas 1364--1369).
- Las entradas previstas incluyen la sospecha estadística, los indicadores y
  el resultado contextual (`Tesis.Tex`, líneas 1378--1391).
- Las tres categorías diagnósticas originales se describen en las líneas
  1393--1403, mientras que la propia tabla conceptual ya reconoce un caso
  incierto en las líneas 1445--1447.
- La función de decisión se deja abierta entre reglas, árbol de decisión o
  modelo supervisado (`Tesis.Tex`, líneas 1405--1415 y 1496--1502).
- La categoría posible falla no confirma una causa raíz
  (`Tesis.Tex`, líneas 1367--1369 y 1471--1477).
- La metodología exige separar desarrollo, validación y prueba respetando el
  orden temporal (`Tesis.Tex`, líneas 1703--1731).
- La evaluación incremental del flujo se plantea en las líneas 1844--1867,
  pero todavía no establece cómo contabilizar fallas descartadas antes del
  clasificador.

Las referencias `jan2021distributed` y `yasaei2020iot`, ya presentes en
`Referencias.bib`, respaldan respectivamente el uso de métodos de
clasificación de fallas y de asociaciones contextuales. No establecen los
umbrales numéricos de este contrato ni eliminan la necesidad de evidencia
independiente ante una posible causa común. Las referencias
`sharma2010sensor`, `de2016benchmark` y `baljak2013fault` respaldan la
distinción entre detección, diagnóstico, taxonomías de fallas y evaluación con
etiquetas conocidas. Ninguna de estas referencias se utiliza aquí para
atribuir resultados aún no observados.

## Debilidades que resuelve el contrato

| Debilidad de la formulación conceptual | Resolución en `diagnostic-rule-list-v1.0.0` |
|---|---|
| La función \(f\) no está definida. | Se fija una lista ordenada de nueve reglas, `R00` a `R08`; la primera regla satisfecha determina la salida. |
| El texto declara tres clases, pero menciona casos inciertos. | Se definen cuatro salidas, incluida `indeterminate` como abstención explícita. |
| Falta de contexto y contradicción pueden confundirse. | Un mapa obsoleto o cobertura menor que 0,50 obliga a abstenerse antes de evaluar contradicción. |
| Una falla común puede parecer una anomalía ambiental coherente. | Una causa común candidata sin evidencia independiente obliga a `indeterminate`. |
| La confianza no tiene definición operacional. | Se define una fuerza de evidencia reproducible y se prohíbe interpretarla como probabilidad. |
| No se indica cómo versionar o congelar la decisión. | Cada ejecución registra versión, parámetros, hashes y la regla activada. |
| La evaluación puede ignorar fallas que no llegan al clasificador. | Se separan \(FN_{\mathrm{gate}}\) y \(FN_{\mathrm{post}}\), y se informa una matriz extremo a extremo. |
| El histórico real actual no permite estimar desempeño. | Se restringe su uso a factibilidad y modo sombra; la validación diagnóstica queda pendiente. |

## Elección definitiva del enfoque

### Alternativas consideradas

La sección conceptual permite usar reglas, un árbol de decisión o un modelo
supervisado liviano (`Tesis.Tex`, líneas 1410--1413 y 1496--1502). Para la
versión inicial se selecciona una lista de reglas por las siguientes razones:

1. Las variables necesarias ya son producidas por los dos filtros anteriores.
2. Cada salida puede reconstruirse mostrando valores, umbrales y regla
   activada.
3. La abstención por cobertura o causa común puede imponerse como una
   restricción dura.
4. No requiere ajustar parámetros mediante etiquetas que el mes real
   disponible no contiene.
5. Su costo y comportamiento son estables y fáciles de someter a pruebas
   unitarias.

Un árbol aprendido o un modelo supervisado queda como alternativa futura.
Solo deberá considerarse cuando exista una cantidad suficiente de episodios
reales etiquetados para entrenar, validar y probar sin reutilizar períodos. La
mera disponibilidad de un algoritmo de aprendizaje no resuelve la ausencia de
etiquetas ni la no identificabilidad de una causa común.

### Compromiso epistemológico

`diagnostic-rule-list-v1.0.0` es una hipótesis operacional pre-registrada. El
anexo especifica qué hará el prototipo, no afirma que las reglas alcancen una
exactitud determinada. Los umbrales solo podrán conservarse, revisarse o
reemplazarse después de una validación temporal con base de verdad real. Todo
cambio producirá una versión nueva y deberá evaluarse sin consultar el
conjunto final de prueba.

## Variables de entrada disponibles

### Evidencia del filtro estadístico-temporal

El resultado estadístico ya expone los siguientes campos:

| Campo | Tipo | Uso en el clasificador |
|---|---:|---|
| `entity_id` | cadena | Identidad de la entidad. |
| `timestamp` | fecha-hora | Unión temporal y construcción de episodios. |
| `sensor_kind` | `continuous`, `binary` o `multistate` | Trazabilidad por tipo de entidad. |
| `observed_state` | cadena | Estado que originó la sospecha. |
| `raw_score` | número | Evidencia cruda explicativa; no decide la clase directamente. |
| `evidence_score` | \([0,1]\) | Resumen de componentes estadísticos. |
| `suspicion_score` | \([0,1]\) | Sospecha inicial enviada al segundo filtro. |
| `suspected_pattern` | cadena | Patrón dominante sugerido. |
| `triggered_indicators` | lista | Indicadores que superaron su umbral interno. |
| `reading.components` | mapa \([0,1]\) | Componentes normalizados usados para calcular la fuerza del patrón específico. |
| `passed_to_contextual_filter` | booleano | Permite auditar el primer punto de descarte. |
| `severity` | cadena | Etiqueta descriptiva; no reemplaza los puntajes. |

Se deriva:

\[
S_{\mathrm{component}}
=
\max_{q\in\texttt{reading.components}}q.
\]

En la implementación, esta característica aparece como
`indicator_component_max` en la especificación y como `pattern_score` en
`features_used`. Su componente dominante aparece como `pattern_component`.
Cuando dos componentes tienen el mismo valor máximo se selecciona el nombre
lexicográficamente mayor, porque la implementación maximiza el par
`(valor, nombre)`.
Si el mapa `reading.components` existe pero está vacío, se usa
`evidence_score` como respaldo y, para identificar el componente, el primer
elemento lexicográfico de `triggered_indicators`; si tampoco existe un
indicador se registra `unavailable`. La entrada solo es inválida cuando
`reading.components` no existe o no es un mapa.

### Evidencia del filtro contextual espacio-temporal

El segundo filtro ya produce:

| Campo | Tipo | Uso en el clasificador |
|---|---:|---|
| `event_id` | cadena | Identificador reproducible del evento. |
| `statistical_score` | \([0,1]\) | Copia de la sospecha inicial. |
| `context_score` | \([-1,1]\) | Balance contextual con signo. |
| `coherence_score` | \([0,1]\) | Representación normalizada de coherencia. |
| `support_score` | \([0,1]\) | Masa relativa de evidencia que respalda el evento. |
| `contradiction_score` | \([0,1]\) | Masa relativa de contradicciones efectivas. |
| `adjusted_suspicion_score` | \([0,1]\) | Sospecha después del efecto contextual permitido. |
| `context_outcome` | cadena | `supported`, `contradicted`, `unexplained` o `insufficient_coverage`. |
| `coverage.eligible_neighbors` | entero | Relaciones que podían evaluarse. |
| `coverage.observed_neighbors` | entero | Relaciones efectivamente observadas. |
| `coverage.ratio` | \([0,1]\) | Proporción ponderada de contexto disponible. |
| `coverage.status` | cadena | Debe ser `sufficient` para permitir una decisión determinada. |
| `common_cause.candidate` | booleano | Patrón multientidad compatible con causa común. |
| `common_cause.independent_evidence_available` | booleano | Disponibilidad de telemetría independiente. |
| `common_cause.concurrent_entities` | lista | Entidades afectadas de manera concurrente. |
| `map_quality.version` | cadena | Versión del mapa contextual. |
| `map_quality.stale` | booleano | Indica si el mapa perdió vigencia. |
| `map_quality.stale_reasons` | lista | Explicación de obsolescencia. |
| `evidence`, `supports`, `contradictions`, `unknowns` | listas | Trazabilidad por arista contextual. |

La decisión ejecutable toma \(S_{\mathrm{stat}}\) de
`statistical.suspicion_score`. Aunque la especificación publicada denomina
esta característica `statistical_score` y el resultado contextual conserva un
campo con ese nombre, el clasificador no lee
`contextual.statistical_score`. Esta diferencia nominal se registra para no
atribuir a la decisión una variable que el código no consume.

El clasificador no infiere que una asociación sea causal. Cuando
`common_cause.candidate` es verdadero y no existe evidencia independiente, el
histórico no permite distinguir entre una alteración física compartida y una
falla compartida de red, alimentación o integración.

### Validación del esquema

Una entrada es inválida si ocurre al menos una de estas condiciones:

- falta `entity_id`, `timestamp`, `sensor_kind` o `suspicion_score` en el
  resultado estadístico;
- `suspicion_score` no pertenece a \([0,1]\);
- falta `reading.components` o no es un mapa;
- no existe el resultado contextual correspondiente;
- falta alguno de `event_id`, `adjusted_suspicion_score`, `context_score`,
  `support_score`, `contradiction_score` o `context_outcome` en el resultado
  contextual;
- `adjusted_suspicion_score`, `support_score` o `contradiction_score` no
  pertenece a \([0,1]\), o `context_score` no pertenece a \([-1,1]\);
- `context_outcome` no pertenece al conjunto cerrado de estados;
- falta `coverage`, `coverage.ratio` o `coverage.status`, o la razón se
  encuentra fuera de \([0,1]\);
- falta `common_cause.candidate` o
  `common_cause.independent_evidence_available`;
- falta `map_quality.stale`.

Las entradas inválidas activan `R00_INVALID_OR_MISSING_INPUT`; nunca se
clasifican como normales.

## Salidas diagnósticas

| Código | Nombre para interfaz | Interpretación |
|---|---|---|
| `normal` | Lectura normal | La sospecha ajustada no alcanza el umbral operativo y ninguna regla de mayor precedencia obliga a otra salida. |
| `environmental_anomaly` | Anomalía ambiental | Existe una desviación inicialmente sospechosa con respaldo contextual suficiente. |
| `possible_sensor_fault` | Posible falla de sensor | Existe contradicción contextual suficiente o un patrón estadístico fuerte que permanece sin explicación. No equivale a causa raíz confirmada. |
| `indeterminate` | Indeterminada | La evidencia no permite asignar responsablemente una de las otras clases. |

`indeterminate` es una abstención del clasificador. La causa se conserva en
`reason_codes`, mediante uno o más de los siguientes códigos:

```text
missing_statistical_field:<campo>
missing_contextual_field:<campo>
missing_contextual_result
out_of_range:<campo>
invalid_value:context_outcome
possible_common_cause_without_independent_evidence
stale_context_map
insufficient_context_coverage
unexplained_or_conflicting_evidence
```

Una salida indeterminada no se cuenta como acierto de falla, aunque el sistema
recomiende `collect_context_or_independent_evidence`.

## Lista de reglas `diagnostic-rule-list-v1.0.0`

### Constantes

```text
statistical_threshold = 0.60
adjusted_normal_threshold = 0.60
minimum_context_coverage = 0.50
environmental_support_threshold = 0.50
contradiction_threshold = 0.35
fault_pattern_threshold = 0.70
low_confidence_threshold = 0.70
alert_confidence_threshold = 0.75
reject_stale_map = true
require_independent_evidence_for_common_cause = true
```

`statistical_threshold` se registra en la traza del primer filtro.
`low_confidence_threshold` se usa en el resumen agregado y
`alert_confidence_threshold` selecciona la acción de una posible falla. Los
cinco umbrales restantes y las dos guardas booleanas intervienen directamente
en las reglas.

### Semántica de ejecución

Las reglas se evalúan de arriba hacia abajo. La primera condición verdadera
determina la clase, la razón, la fuerza de evidencia y el identificador de
regla. No se combinan votos ni se permite que una regla posterior anule una
abstención anterior.

| Precedencia | `decision_rule_id` | Condición exacta | Salida | `reason_codes` |
|---:|---|---|---|---|
| 0 | `R00_INVALID_OR_MISSING_INPUT` | Algún campo requerido es inválido o falta el resultado contextual correspondiente. | `indeterminate` | Códigos dinámicos `missing_*`, `out_of_range:*` o `invalid_value:context_outcome`. |
| 1 | `R01_AMBIGUOUS_COMMON_CAUSE` | `common_cause.candidate == true` y `common_cause.independent_evidence_available == false`. | `indeterminate` | `possible_common_cause_without_independent_evidence` |
| 2 | `R02_STALE_CONTEXT_MAP` | `reject_stale_map == true` y `map_quality.stale == true`. | `indeterminate` | `stale_context_map` |
| 3 | `R03_INSUFFICIENT_CONTEXT_COVERAGE` | `coverage.ratio < 0.50`, o `coverage.status != "sufficient"`, o `context_outcome == "insufficient_coverage"`. | `indeterminate` | `insufficient_context_coverage` |
| 4 | `R04_STRONG_CONTEXTUAL_SUPPORT` | `context_outcome == "supported"` y `support_score >= 0.50`. | `environmental_anomaly` | `strong_contextual_support` |
| 5 | `R05_CONTEXT_REJECTS_SUSPICION` | `adjusted_suspicion_score < 0.60`. | `normal` | `adjusted_suspicion_below_classification_threshold` |
| 6 | `R06_CONTEXTUAL_CONTRADICTION` | `context_outcome == "contradicted"` y `contradiction_score >= 0.35`. | `possible_sensor_fault` | `contextual_contradiction` |
| 7 | `R07_UNEXPLAINED_SPECIFIC_PATTERN` | `context_outcome == "unexplained"` y `indicator_component_max >= 0.70`. | `possible_sensor_fault` | `unexplained_specific_sensor_pattern` y `dominant_component:<componente>` |
| 8 | `R08_UNRESOLVED_EVIDENCE` | Ninguna regla anterior se cumple. | `indeterminate` | `unexplained_or_conflicting_evidence` |

Esta precedencia forma parte de la versión. Modificar el orden, incluso sin
cambiar los números, crea un clasificador distinto.

## Fuerza de evidencia

El campo `confidence` representa fuerza de evidencia de la regla
activada. No representa una probabilidad calibrada, una frecuencia esperada de
acierto ni la probabilidad de que exista una falla. La interfaz debe mostrar
“fuerza de evidencia” y no “probabilidad”.

Sean \(S_{\mathrm{stat}}=\texttt{suspicion\_score}\),
\(S_{\mathrm{adj}}=\texttt{adjusted\_suspicion\_score}\),
\(S_{\mathrm{sup}}=\texttt{support\_score}\),
\(S_{\mathrm{con}}=\texttt{contradiction\_score}\),
\(C=\texttt{coverage.ratio}\) y
\(P=\texttt{indicator\_component\_max}\). Se define
\(\operatorname{clip}(x)=\min(1,\max(0,x))\). La implementación redondea a seis
decimales después de aplicar `clip`.

\[
\operatorname{confidence}_{\mathrm{ambiental}}
=
\operatorname{clip}
\left(
0{,}40S_{\mathrm{stat}}
+0{,}35S_{\mathrm{sup}}
+0{,}25C
\right),
\]

\[
\operatorname{confidence}_{\mathrm{normal}}
=
\operatorname{clip}
\left(
0{,}45(1-S_{\mathrm{adj}})
+0{,}35S_{\mathrm{sup}}
+0{,}20C
\right),
\]

\[
\operatorname{confidence}_{\mathrm{R06}}
=
\operatorname{clip}
\left(
0{,}45S_{\mathrm{stat}}
+0{,}35S_{\mathrm{con}}
+0{,}20C
\right),
\]

\[
\operatorname{confidence}_{\mathrm{R07}}
=
\operatorname{clip}
\left(
0{,}45S_{\mathrm{stat}}
+0{,}35P
+0{,}20C
\right).
\]

Para `indeterminate`, `confidence` es `null`. La razón de abstención,
la cobertura y la vigencia del mapa se presentan por separado. Esta decisión
evita transformar la falta de información en una confianza numérica
engañosa.

Cada salida incluye:

```text
confidence_kind = "evidence_strength_not_calibrated_probability"
confidence
decision_rule_id
model_version
feature_schema_version
features_used
evidence
```

Una calibración probabilística solo podrá añadirse en una versión posterior
con suficientes episodios reales etiquetados y una partición de calibración
separada.

## Subtipos sugeridos

El campo único `subtype` solo se informa cuando `diagnostic_class` es
`possible_sensor_fault`. Describe el patrón de lectura sugerido y no una causa
raíz. La implementación evalúa los indicadores en el orden siguiente; la
primera intersección determina el subtipo.

| `triggered_indicators` que activa la correspondencia | `subtype` |
|---|---|
| `reading_gap` o `report_gap` | `communication_gap` |
| `constant_persistence` | `stuck_value` |
| `active_time_ratio`, `time_since_last_change` o `z_duration_score` | `stuck_state_or_abnormal_duration` |
| `activation_count` o `state_change_count` | `repeated_activation_or_interference` |
| `time_since_last_activation` o `inactive_duration` | `omission_or_expected_inactivity` |
| `cusum_residual` | `persistent_shift` |
| `hampel`, `robust_rate_change` o `dspot_extreme` | `spike_or_extreme` |
| Ninguna correspondencia ni patrón utilizable | `unspecified_sensor_pattern` |

El catálogo canónico publicado por la especificación es:
`communication_gap`, `stuck_value`,
`stuck_state_or_abnormal_duration`,
`repeated_activation_or_interference`,
`omission_or_expected_inactivity`, `persistent_shift`, `spike_or_extreme` y
`unspecified_sensor_pattern`. La fuerza \(P\) utilizada por `R07` no se
convierte automáticamente en subtipo. Como compatibilidad con el primer
filtro, si ningún grupo de indicadores coincide, la función ejecutable retorna
primero `suspected_pattern`, luego `pattern` y solo si ambos están vacíos
retorna `unspecified_sensor_pattern`. Por ello, cualquier valor heredado fuera
del catálogo debe conservarse como tal en la auditoría y no renombrarse
silenciosamente.

## Registro de salida

`analizar` devuelve un sobre con `schema_version = "classifier-results-v1"`,
un objeto `model`, un `summary` agregado y la lista `results`. Cada elemento
de `results` contiene exactamente:

`model` contiene `version`, `type`, `feature_schema_version`,
`confidence_kind`, `selection_status` y `trained_at`; este último permanece
en `null` porque la lista no se entrena. `summary` contiene
`input_candidates`, `evaluated`, `classified`, `abstained`, `by_class`,
`low_confidence`, `confidence_mean` y `elapsed_ms`.

| Campo | Contenido |
|---|---|
| `diagnosis_id` | `diag-` seguido de los primeros 20 caracteres hexadecimales de SHA-256 sobre `model_version|event_id`. |
| `event_id`, `entity_id`, `timestamp` | Identificación y tiempo del evento. |
| `window_start`, `window_end` | Ventana heredada del resultado estadístico. |
| `sensor_kind`, `source` | Tipo de entidad y origen de ejecución; `source` usa `historical` por defecto. |
| `diagnostic_class`, `subtype` | Clase canónica y subtipo único o `null`. |
| `confidence`, `confidence_kind` | Fuerza de evidencia y literal `evidence_strength_not_calibrated_probability`. |
| `abstained` | Verdadero únicamente para `indeterminate`. |
| `decision_rule_id`, `reason_codes` | Primera regla satisfecha y sus razones reproducibles. |
| `suggested_action` | Acción determinada por la clase y, para una posible falla, por `confidence`. |
| `features_used`, `evidence` | Dos campos con la misma lista de trazas `{name, value, source, available, contribution, direction}`. |
| `limitations`, `explanation` | Límites epistemológicos y explicación textual de la decisión. |
| `model_version`, `feature_schema_version` | `diagnostic-rule-list-v1.0.0` y `diagnostic-features-v1`. |
| `input_provenance` | Identificadores estadístico y contextual, más `ground_truth_used_as_feature = false`. |
| `pipeline` | Trazas anidadas `statistical`, `contextual` y `classifier`. |

La forma estructural, sin valores observacionales, es:

```text
{
  diagnosis_id,
  event_id,
  entity_id,
  timestamp,
  window_start,
  window_end,
  sensor_kind,
  source,
  diagnostic_class,
  subtype,
  confidence,
  confidence_kind,
  abstained,
  decision_rule_id,
  reason_codes,
  suggested_action,
  features_used: [
    {name, value, source, available, contribution, direction}
  ],
  evidence: [
    {name, value, source, available, contribution, direction}
  ],
  limitations,
  explanation,
  model_version,
  feature_schema_version,
  input_provenance: {
    statistical_event_id,
    contextual_event_id,
    ground_truth_used_as_feature
  },
  pipeline: {
    statistical: {
      score, threshold, passed, window_start, window_end, pattern, indicators
    },
    contextual: {
      outcome, context_score, coherence_score, support_score,
      contradiction_score, adjusted_suspicion_score, coverage,
      common_cause, map
    },
    classifier: {
      model_version, decision_rule_id, elapsed_ms, reason_codes, explanation
    }
  }
}
```

Las acciones canónicas son:

| Condición | `suggested_action` |
|---|---|
| `normal` | `accept_candidate_as_normal` |
| `environmental_anomaly` | `observe_environmental_event` |
| `possible_sensor_fault` y `confidence >= 0.75` | `alert_and_hold_sensitive_automation_recommended` |
| `possible_sensor_fault` y `confidence < 0.75` | `inspect_sensor_before_automation` |
| `indeterminate` | `collect_context_or_independent_evidence` |

## Protocolo de uso con el histórico real actual

El mes real disponible se utiliza únicamente para:

1. comprobar que todas las variables pueden calcularse sin fuga temporal;
2. construir una primera base de normalidad y un mapa contextual experimental;
3. medir cobertura, tasa de abstención y causas de datos inválidos;
4. medir tiempo de ejecución, memoria y latencia;
5. revisar manualmente la explicación de una muestra de decisiones;
6. ejecutar el clasificador en modo sombra, sin presentarlo como mecanismo
   validado ni modificar automatizaciones.

No se utilizan las salidas del propio clasificador como etiquetas. La revisión
manual de una decisión tampoco se considera base de verdad si no existe una
fuente independiente, como un registro de mantenimiento, una acción documentada
del habitante, telemetría de red o alimentación, o una falla controlada en
tiempo real.

## Protocolo temporal para la validación futura

### Requisitos de entrada

La validación definitiva comenzará cuando se cumplan simultáneamente:

- al menos cinco meses cronológicos de histórico real;
- etiquetas de episodios obtenidas sin utilizar la salida del clasificador;
- presencia de episodios de lectura normal, anomalía ambiental y falla;
- trazabilidad del origen, inicio, fin y evidencia independiente de cada
  etiqueta.

Si una partición no contiene episodios positivos de falla, no se calcula
sensibilidad, \(F_2\) ni una conclusión de desempeño para esa partición. La
ausencia de positivos es una limitación de evaluación, no un indicador de
buen desempeño.

### Partición 60/20/20

El eje temporal completo se divide antes de observar resultados:

- 60 % inicial: desarrollo de la base de normalidad y del mapa contextual;
- 20 % siguiente: validación y selección entre versiones pre-registradas;
- 20 % final: prueba única, excluida de cualquier ajuste.

La división se realiza por tiempo y no por número de filas. Una frontera que
atraviese un episodio se desplaza hasta el final de ese episodio y el cambio se
registra antes de ejecutar el flujo. Ninguna observación posterior al instante
evaluado puede usarse para construir sus características.

### Pre-registro y selección

Antes de abrir las etiquetas de prueba se registra:

- versiones candidatas y orden de sus reglas;
- todos los umbrales;
- definición de episodio y ventana de correspondencia;
- tratamiento de `indeterminate` como no verdadero positivo;
- tratamiento de `not_reached` como predicción implícita normal en la métrica
  extremo a extremo;
- métricas, restricciones duras y desempates;
- semillas de cualquier procedimiento aleatorio auxiliar;
- código y configuración exactos.

Las guardas de entrada inválida, causa común, mapa obsoleto y cobertura son
invariantes y no pueden eliminarse para mejorar una métrica. Una versión
candidata solo es elegible si satisface simultáneamente las guardas
pre-registradas de la implementación:

| Guarda de elegibilidad | Umbral |
|---|---:|
| `fault_recall_end_to_end_min` | \(0{,}80\) |
| `environmental_recall_end_to_end_min` | \(0{,}70\) |
| `determinate_coverage_min` | \(0{,}70\) |
| `ambiguous_common_cause_as_environmental_max` | \(0{,}05\) |
| `legitimate_normal_hold_rate_max` | \(0{,}05\) |

Entre candidatas elegibles se maximiza
`fault_f2_end_to_end`. Los desempates se resuelven, en este orden, mediante
`macro_f1_end_to_end`, `determinate_coverage` y `latency_p95`. La regla de
parsimonia conserva la lista de decisión si su macro-\(F_1\) extremo a extremo
queda a no más de \(0{,}02\) del mejor candidato elegible. La selección aún no
se ha ejecutado, porque el histórico real actual carece de las etiquetas
requeridas. La prueba final no participa en la selección.

### Congelamiento y hashes

Antes de la prueba final se genera un manifiesto con SHA-256 de:

```text
code_hash
configuration_hash
feature_schema_version
split_boundaries
context_map_version
```

El manifiesto registra además fecha, versión y parámetros. Cualquier cambio
posterior invalida la prueba como evaluación final de esa versión. Si se
requiere modificar una regla después de observar el test, debe reservarse un
nuevo período futuro como conjunto final.

## Evaluación por episodios

### Unidad de análisis

La unidad primaria es un episodio, no una fila. Cada episodio contiene:

```text
episode_id
entity_id o conjunto de entidades
ground_truth_class
start_time
end_time
evidence_source
expected_diagnostic_deadline
```

Dos filas consecutivas pertenecen al mismo episodio cuando corresponden a la
misma alteración documentada y no existe un retorno a la condición nominal
entre ellas. La definición se fija antes de la prueba.

Para un episodio, la primera salida no normal emitida antes de
`expected_diagnostic_deadline` se toma como predicción. Si solo existen salidas
normales, la predicción es `normal`. Si ninguna fila alcanza el clasificador,
la predicción operativa implícita es `normal`, pero se conserva
`audit_state = "not_reached"` para atribuir el error a la compuerta previa.

### Falsos negativos de la compuerta

Sea \(\mathcal F\) el conjunto de episodios cuya clase real es falla. Se define:

\[
G(e)=
\begin{cases}
1, & \text{si el episodio }e\text{ alcanza el clasificador antes del plazo},\\
0, & \text{en otro caso}.
\end{cases}
\]

\[
FN_{\mathrm{gate}}
=
\sum_{e\in\mathcal F}\mathbb{1}[G(e)=0].
\]

Para los episodios que sí llegan:

\[
FN_{\mathrm{post}}
=
\sum_{e\in\mathcal F}
\mathbb{1}
\left[
G(e)=1
\land
\widehat y(e)\neq\texttt{possible\_sensor\_fault}
\right].
\]

Por contrato, una predicción `indeterminate` sobre una falla real no es
verdadero positivo y forma parte de \(FN_{\mathrm{post}}\).

\[
FN_{\mathrm{E2E}}
=
FN_{\mathrm{gate}}+FN_{\mathrm{post}},
\]

\[
\operatorname{Recall}_{\mathrm{E2E}}
=
1-\frac{FN_{\mathrm{E2E}}}{|\mathcal F|}.
\]

Para la vista binaria de falla, \(TP_{\mathrm{E2E}}\) cuenta episodios de falla
diagnosticados como `possible_sensor_fault` antes del plazo y
\(FP_{\mathrm{E2E}}\) cuenta episodios cuya verdad no es falla pero fueron
diagnosticados como `possible_sensor_fault`. El criterio de selección se calcula
entonces como:

\[
F_{2,\mathrm{E2E}}
=
\frac{5TP_{\mathrm{E2E}}}
{5TP_{\mathrm{E2E}}+4FN_{\mathrm{E2E}}+FP_{\mathrm{E2E}}}.
\]

No se permite eliminar \(FN_{\mathrm{gate}}\) del cálculo. Una salida
`indeterminate` nunca aporta un verdadero positivo ni se presenta como un
diagnóstico correcto; además de la vista binaria, se informa siempre su tasa
y su razón de abstención.

### Dos matrices obligatorias

1. **Matriz extremo a extremo.** Incluye todos los episodios desde la entrada
   del primer filtro. Para las métricas operativas, `not_reached` se interpreta
   como `normal`. Una versión de auditoría conserva una columna adicional
   `not_reached` para mostrar el descarte previo.
2. **Matriz condicional al clasificador.** Incluye solo episodios con
   \(G(e)=1\). Debe titularse explícitamente “condicional a haber alcanzado el
   clasificador” y nunca presentarse como desempeño general.

Se informan además:

- \(FN_{\mathrm{gate}}\), \(FN_{\mathrm{post}}\) y
  \(FN_{\mathrm{E2E}}\);
- cobertura del clasificador;
- tasa y razones de abstención;
- resultados por tipo de entidad y subtipo sugerido;
- latencia hasta la primera decisión;
- cantidad de episodios reales por clase y fuente de etiqueta.

## Limitaciones que permanecen

- Una lista de reglas no descubre una causa no observada.
- Una causa común sin evidencia independiente permanece indeterminada.
- La correlación o coocurrencia contextual no demuestra causalidad.
- Un mes real no representa por sí solo cambios prolongados de rutina ni
  estacionalidad.
- Sin etiquetas reales no es posible calibrar probabilidades ni estimar
  desempeño diagnóstico.
- La fuerza de evidencia definida aquí no debe comunicarse como probabilidad.
- Los subtipos describen patrones compatibles, no causas confirmadas.

## Texto sugerido para reemplazar la sección conceptual del clasificador

```latex
\subsection{Filtro de clasificación diagnóstico basado en reglas versionadas}

El tercer filtro integra la evidencia producida por el filtro
estadístico-temporal y por el filtro contextual espacio-temporal. Detectar una
lectura atípica no determina por sí solo su origen, por lo que la salida se
formula como la categoría más compatible con la evidencia disponible y no como
una confirmación de causa raíz
\cite{jan2021distributed,yasaei2020iot}.

Para asegurar reproducibilidad se adopta la lista de reglas
\texttt{diagnostic-rule-list-v1.0.0}. Esta elección reemplaza la función
conceptual abierta por una función determinista \(C(x)\), donde \(x\) contiene
la sospecha \texttt{suspicion\_score} del primer filtro, la sospecha ajustada,
el resultado contextual, los puntajes de respaldo y contradicción, la
cobertura, la vigencia del mapa, el componente estadístico dominante y los
indicadores de causa común y evidencia independiente. Las etiquetas de verdad
no forman parte de \(x\).

El clasificador emite exactamente
\texttt{normal}, \texttt{environmental\_anomaly},
\texttt{possible\_sensor\_fault} o \texttt{indeterminate}. La última salida es
una abstención. Las nueve reglas se evalúan en el siguiente orden y la primera
condición satisfecha determina la salida:

\begin{enumerate}
  \item[\texttt{R00}] \texttt{R00\_INVALID\_OR\_MISSING\_INPUT}: si un campo
  requerido es inválido o falta el resultado contextual, emite
  \texttt{indeterminate} y conserva los códigos dinámicos de validación.
  \item[\texttt{R01}] \texttt{R01\_AMBIGUOUS\_COMMON\_CAUSE}: si existe una
  causa común candidata sin evidencia independiente, emite
  \texttt{indeterminate} con
  \texttt{possible\_common\_cause\_without\_independent\_evidence}.
  \item[\texttt{R02}] \texttt{R02\_STALE\_CONTEXT\_MAP}: si
  \texttt{map\_quality.stale} es verdadero, emite \texttt{indeterminate} con
  \texttt{stale\_context\_map}.
  \item[\texttt{R03}] \texttt{R03\_INSUFFICIENT\_CONTEXT\_COVERAGE}: si
  \texttt{coverage.ratio} es menor que \(0{,}50\),
  \texttt{coverage.status} es distinto de \texttt{sufficient}, o el resultado
  es \texttt{insufficient\_coverage}, emite \texttt{indeterminate} con
  \texttt{insufficient\_context\_coverage}.
  \item[\texttt{R04}] \texttt{R04\_STRONG\_CONTEXTUAL\_SUPPORT}: si el
  resultado es \texttt{supported} y \texttt{support\_score} es al menos
  \(0{,}50\), emite \texttt{environmental\_anomaly} con
  \texttt{strong\_contextual\_support}.
  \item[\texttt{R05}] \texttt{R05\_CONTEXT\_REJECTS\_SUSPICION}: si
  \texttt{adjusted\_suspicion\_score} es menor que \(0{,}60\), emite
  \texttt{normal} con
  \texttt{adjusted\_suspicion\_below\_classification\_threshold}.
  \item[\texttt{R06}] \texttt{R06\_CONTEXTUAL\_CONTRADICTION}: si el resultado
  es \texttt{contradicted} y \texttt{contradiction\_score} es al menos
  \(0{,}35\), emite \texttt{possible\_sensor\_fault} con
  \texttt{contextual\_contradiction}.
  \item[\texttt{R07}] \texttt{R07\_UNEXPLAINED\_SPECIFIC\_PATTERN}: si el
  resultado es \texttt{unexplained} y el componente estadístico máximo es al
  menos \(0{,}70\), emite \texttt{possible\_sensor\_fault} con
  \texttt{unexplained\_specific\_sensor\_pattern} y
  \texttt{dominant\_component:<componente>}.
  \item[\texttt{R08}] \texttt{R08\_UNRESOLVED\_EVIDENCE}: en cualquier caso
  restante, emite \texttt{indeterminate} con
  \texttt{unexplained\_or\_conflicting\_evidence}.
\end{enumerate}

Sea \(S\) la sospecha estadística, \(A\) la sospecha ajustada, \(U\) el
respaldo, \(D\) la contradicción, \(C\) la cobertura y \(P\) el componente
estadístico máximo. El campo \texttt{confidence} se calcula como fuerza de
evidencia mediante:

\begin{align}
\operatorname{confidence}_{\mathrm{ambiental}}
 &= \operatorname{clip}(0{,}40S+0{,}35U+0{,}25C),\\
\operatorname{confidence}_{\mathrm{normal}}
 &= \operatorname{clip}(0{,}45(1-A)+0{,}35U+0{,}20C),\\
\operatorname{confidence}_{\mathrm{R06}}
 &= \operatorname{clip}(0{,}45S+0{,}35D+0{,}20C),\\
\operatorname{confidence}_{\mathrm{R07}}
 &= \operatorname{clip}(0{,}45S+0{,}35P+0{,}20C).
\end{align}

El valor se restringe a \([0,1]\) y se redondea a seis decimales. Para
\texttt{indeterminate}, \texttt{confidence} es nulo. El campo
\texttt{confidence\_kind} conserva el literal
\texttt{evidence\_strength\_not\_calibrated\_probability}; por tanto, no debe
interpretarse como probabilidad calibrada. Una posible falla con fuerza de
evidencia mayor o igual que \(0{,}75\) recomienda
\texttt{alert\_and\_hold\_sensitive\_automation\_recommended}; bajo ese umbral
recomienda \texttt{inspect\_sensor\_before\_automation}.

El subtipo solo se emite para \texttt{possible\_sensor\_fault}. El catálogo
canónico comprende \texttt{communication\_gap}, \texttt{stuck\_value},
\texttt{stuck\_state\_or\_abnormal\_duration},
\texttt{repeated\_activation\_or\_interference},
\texttt{omission\_or\_expected\_inactivity}, \texttt{persistent\_shift},
\texttt{spike\_or\_extreme} y \texttt{unspecified\_sensor\_pattern}. Estos
subtipos describen patrones compatibles y no confirman una causa raíz.

Cada resultado registra \texttt{diagnosis\_id}, \texttt{diagnostic\_class},
\texttt{subtype}, \texttt{confidence}, \texttt{confidence\_kind},
\texttt{abstained}, \texttt{decision\_rule\_id}, \texttt{reason\_codes},
\texttt{suggested\_action}, \texttt{features\_used}, \texttt{evidence},
\texttt{limitations}, las versiones, la procedencia de entrada y
\texttt{pipeline}. Los campos \texttt{features\_used} y \texttt{evidence}
contienen la misma lista de trazas, mientras que \texttt{pipeline} separa los
estados estadístico, contextual y de clasificación.

Las reglas y sus umbrales se fijan a priori. El histórico real actualmente
disponible cubre un mes y no posee etiquetas diagnósticas independientes, por
lo que se utiliza solo para comprobar factibilidad, cobertura, trazabilidad y
costo operativo en modo sombra. La validación definitiva requerirá al menos
cinco meses de histórico real y episodios etiquetados mediante evidencia
independiente o fallas controladas en tiempo real. El futuro conjunto se
dividirá cronológicamente en desarrollo, validación y prueba con proporciones
\(60/20/20\).

Una candidata deberá alcanzar como mínimo \(0{,}80\) de sensibilidad de falla
extremo a extremo, \(0{,}70\) de sensibilidad ambiental y \(0{,}70\) de
cobertura determinada. La tasa de causas comunes ambiguas clasificadas como
ambientales y la tasa de retención de lecturas legítimamente normales no
podrán superar \(0{,}05\). Entre candidatas elegibles se maximizará
\(F_{2,\mathrm{E2E}}\); los desempates usarán macro-\(F_1\) extremo a extremo,
cobertura determinada y latencia p95. La lista de decisión se conservará por
parsimonia si queda a no más de \(0{,}02\) de macro-\(F_1\) extremo a extremo
del mejor candidato elegible. Antes de abrir la prueba se congelarán
\texttt{code\_hash}, \texttt{configuration\_hash},
\texttt{feature\_schema\_version}, \texttt{split\_boundaries} y
\texttt{context\_map\_version}.

La evaluación se realizará por episodios y desde la entrada del primer filtro.
Las fallas que no alcancen el clasificador se contabilizarán como
\(FN_{\mathrm{gate}}\), mientras que las que lleguen pero no sean clasificadas
como \texttt{possible\_sensor\_fault} se contabilizarán como
\(FN_{\mathrm{post}}\). Por tanto,
\(FN_{\mathrm{E2E}}=FN_{\mathrm{gate}}+FN_{\mathrm{post}}\). Una abstención no
se contará como verdadero positivo. Se presentará una matriz extremo a extremo
y, por separado, una matriz condicional a los episodios que alcanzaron el
clasificador.
```

## Lista de verificación para adaptación

- [ ] Copiar o adaptar el texto sugerido en `Tesis.Tex`.
- [ ] Incorporar la tabla completa de reglas o referenciar este anexo.
- [ ] Añadir `indeterminate` a clases y salida.
- [ ] Sustituir “confianza” probabilística por “fuerza de evidencia” mientras
      no exista calibración real.
- [ ] Registrar `decision_rule_id`, `reason_codes`, `confidence`,
      `model_version`, `features_used`, `evidence` y `pipeline` en cada
      diagnóstico.
- [ ] Mantener el mes actual en modo sombra.
- [ ] Obtener etiquetas mediante evidencia independiente o fallas controladas
      en tiempo real.
- [ ] Alcanzar el horizonte mínimo de cinco meses antes de la validación
      definitiva.
- [ ] Pre-registrar reglas, episodios, métricas, fronteras temporales y hashes.
- [ ] Reportar \(FN_{\mathrm{gate}}\), \(FN_{\mathrm{post}}\) y matrices
      extremo a extremo y condicional.
