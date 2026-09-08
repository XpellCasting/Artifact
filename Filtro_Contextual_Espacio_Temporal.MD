# Segundo filtro: contexto espacio-temporal

Esta implementación operacionaliza la sección **Filtro contextual
espacio-temporal basado en dependencias entre entidades** de `Tesis.Tex`
(líneas 1090–1360). El mapa se aprende desde el histórico normalizado y se
consulta únicamente para sospechas que provienen del filtro
estadístico-temporal.

Las asociaciones son evidencia contextual. No se interpretan como causalidad.
Este límite coincide con la tesis (línea 1141) y con el uso de asociaciones
entre sensores de `yasaei2020iot`, matrices por segmentos de
`xie2014segment`, series multivariadas de `belay2023unsupervised` y contexto
para sensores de estado de `ye2016detecting`.

## Regla reproducible de construcción

La versión `contextual-spatiotemporal-v3` usa estos valores por defecto:

| Parámetro | Valor |
|---|---:|
| Ventanas candidatas | 300, 900 y 3600 s; menor resolución temporal que supera todos los controles |
| Desfases evaluados | Hasta 7200 s hacia el pasado; candidatos por ventana en `lag_candidates_by_window` |
| Observaciones pareadas mínimas | 30 |
| Cobertura histórica mínima del par | 0,60 |
| Asociación absoluta mínima | 0,35 |
| Periodos contiguos para estabilidad | 4 |
| Asociación mínima por periodo estable | 0,20 |
| Periodos estables mínimos | 3 de 4 |
| Consistencia mínima del signo | 0,75 |
| Relaciones conservadas por entidad (`top_k`) | 5 |

Una arista es elegible si cumple simultáneamente:

\[
n_{ij}\ge 30 \land q_{ij}\ge 0{,}60 \land
|a_{ij}|\ge 0{,}35 \land m_{ij}\ge 3 \land g_{ij}\ge 0{,}75.
\]

Después se conservan, como máximo, las cinco relaciones elegibles de mayor
\(|a_{ij}|\) ponderado por estabilidad para cada entidad. La API conserva
también las aristas rechazadas y sus motivos:
`insufficient_observations`, `insufficient_pair_coverage`,
`weak_association`, `temporal_instability`, `sign_instability` u
`outside_top_k`.

La calidad se comprueba **antes** de elegir el desfase: un candidato de mayor
correlación pero inestable o con pocas observaciones no desplaza a otro válido.
Las aristas aceptadas exponen `lag_candidates` con motivos de descarte. En las
rechazadas se conserva el diagnóstico seleccionado, sin duplicar los parámetros
de inferencia de todos los pares en la caché.

Para cada par se evalúan todas las ventanas y se escoge la de menor duración
que supera todos los controles, antes de aplicar top_k. Si ninguna cumple,
se muestra la más fina como diagnóstico rechazado. `window_candidates` conserva
los resultados de cada resolución, incluso en aristas rechazadas. La cadencia
de Desarrollo sigue determinando la antigüedad máxima de las lecturas; elegir
una ventana mayor no amplía ese límite.
El catálogo de ventanas y desfases es una política operacional, no una búsqueda
exhaustiva ni una estimación causal del tiempo de respuesta. La matriz y el
detalle presentan la asociación calculada con la ventana y el desfase de cada
arista. La covarianza descriptiva se ha retirado del mapa y de la API porque no
interviene en sus controles de aceptación. Se conserva el cálculo interno de
covarianza necesario para ajustar la respuesta lineal contextual.

Estos umbrales son decisiones operacionales versionadas del prototipo; las
referencias sustentan las familias de análisis, no estos valores numéricos
específicos. Deben tratarse como hiperparámetros en la evaluación experimental.

## Familias de relación

- Continua–continua: correlación de Pearson sobre medianas por ventana.
- Estado–estado: coeficiente phi, coocurrencia y probabilidad condicional sobre
  el estado activo o sobre indicadores uno-contra-el-resto.
- Mixta continua–estado: correlación punto-biserial. Esto cubre explícitamente
  relaciones como temperatura–climatización y temperatura–presencia.
- Multivalórica: se evalúan indicadores uno-contra-el-resto de los estados más
  frecuentes y se conserva el estado asociado en la arista.

El desfase de una arista dirigida \(e_i\rightarrow e_j\) indica que una
sospecha de \(e_i(t)\) consulta \(e_j(t-\Delta)\). No se usan observaciones
futuras.

## Evidencia contextual \(c_j(t)\)

Para cada entidad relacionada se obtiene una desviación estandarizada robusta:

\[
z_j(t)=\frac{x_j(t)-\operatorname{mediana}(x_j)}
{\max(1{,}4826\,MAD(x_j),\,\sigma_j,\,\epsilon)}.
\]

Para estados categóricos se usa el indicador del estado almacenado en la
arista, centrado por su frecuencia histórica. Sea \(d_i(t)\) el signo de la
desviación de la entidad sospechosa y \(s_{ij}\) el signo de la asociación:

\[
c_j(t)=\operatorname{clip}
\left(\frac{s_{ij}\,d_i(t)\,z_j(t-\Delta_{ij})}{2},-1,1\right).
\]

Por tanto:

- \(c_j(t)>0\): la observación respalda el signo y desfase históricos;
- \(c_j(t)<0\): existe evidencia observable en sentido contrario;
- \(c_j(t)\approx0\): la entidad fue observada, pero no entrega respaldo ni
  contradicción activa;
- sin observación válida: no se calcula \(c_j(t)\) y la relación reduce
  cobertura, no se cuenta como contradicción.

El puntaje neto es:

\[
S_{\mathrm{ctx}}(e_i,t)=
\frac{\sum_j |a_{ij}|\,\operatorname{estab}_{ij}\,c_j(t)}
{\sum_{j\;observados}|a_{ij}|\,\operatorname{estab}_{ij}}.
\]

Cada resultado expone el valor crudo, el normalizado, el peso y el aporte
ponderado de todas las entidades consultadas.

### Consistencia de magnitud (continua–continua)

Las relaciones elegibles ajustan `source = intercept + slope * context` con el
primer 70 % de sus pares de Desarrollo. El último 30 % calibra una tolerancia
empírica: cuantil 0,99 del residuo absoluto, con piso de 0,05 desviaciones estándar
del origen y 1e-6. Se requieren 20 pares de ajuste y 10 de calibración.
La lectura sospechosa puntual se compara con ese rango; una magnitud incompatible
reduce el aporte y no permite cerrar como ambiental aunque otros vecinos apoyen.
El resultado conserva predicción, rango, residuo y si hubo extrapolación fuera
del rango de ajuste. Sin calibración suficiente se mantiene la prueba direccional.

Este ajuste pertenece a F2, no diagnostica fallas ni implementa F3. El rango no es
un intervalo de confianza: seleccionar relaciones y tolerancias en Desarrollo
no constituye una validación independiente. La extrapolación y las relaciones
mixtas/categóricas requieren evaluación adicional; en estas últimas no se ajusta
todavía un rango de magnitud. Una ausencia de lectura (`missing_gap`) no se explica
mediante el signo de los vecinos y permanece pendiente.

## Estados de salida

- `supported`: cobertura suficiente y masa de respaldo mayor o igual a 0,35;
  disminuye la sospecha, sin afirmar causalidad.
- `contradicted`: cobertura suficiente y masa de contradicción efectiva mayor
  o igual a 0,35; aumenta la sospecha.
- `unexplained`: existe cobertura, pero la evidencia es neutral o conflictiva;
  mantiene la sospecha.
- `insufficient_coverage`: menos de 0,50 del peso elegible tiene observación
  válida; mantiene la sospecha.
- `no_learned_relations`: no hay aristas aceptadas para la entidad; F2 registra
  que no es evaluable y mantiene el caso pendiente. No inventa dependencias,
  no afirma una falla y no lo confunde con lecturas actuales ausentes.

La falta de respaldo ya no se confunde con una contradicción.

## Compuerta secuencial de salida

El segundo filtro decide el destino antes de cualquier clasificador. Un resultado
`supported` con mapa vigente (`stale=false`) termina
como `contextual_class="environmental_anomaly"`. Registra una alerta informativa
(`alert.level="informational"`, `delivery="recorded_only"`), con
`routing.terminal=true` y `routing.send_to_classifier=false`.

El umbral único para respaldo es `decision_threshold=0.35`; no hay un segundo
umbral de 0.50 para declarar ambiental. La cobertura mínima sigue siendo 0.50.
`adjusted_suspicion_score` se conserva para trazabilidad, no como decisión de ruta.

Contradicción, evidencia no explicada, cobertura insuficiente, ausencia de
relaciones y mapa obsoleto o desconocido producen `routing.send_to_classifier=true`.
Esto indica que requieren una futura clasificación, no que hayan sido enviados.
Actualmente F3 no está implementado: esos casos aparecen en
`pending_classification_results`, con `classification_status="pending_not_implemented"`.
No se producen tipos de falla, diagnósticos ni notificaciones a Home Assistant.

`main.py` separa `environmental_results` de `pending_classification_results` en
histórico, reanálisis y reproducción causal. Los contadores incluyen todos los
casos, incluso si una lista para visualización está recortada. El resumen informa
`classifier=0` y `classifier_status="not_implemented"`. La caché de análisis usa
versión 20 para no reutilizar decisiones de la versión anterior del filtro.

### Disponibilidad temporal de sensores continuos

Las fechas conservan los subsegundos. Para una sospecha en `t`, cada relación
consulta una referencia `t - lag_seconds`: no se incorpora una lectura posterior
a esa referencia. La mediana se calcula con las observaciones válidas de la
ventana parcial de la arista disponibles hasta ese instante, usando sus centros y
escalas de Desarrollo como referencia fija. Los estados también respetan el corte
exacto `t - lag_seconds`, incluso dentro de una misma ventana.

Si la ventana no tiene observaciones, se permite el último valor conocido cuando
su antigüedad no supera `continuous_max_age_seconds` (3600 segundos inicialmente),
con reemplazos por entidad en `continuous_max_age_seconds_by_entity`. La antigüedad
se mide desde la fecha original hasta la referencia consultada, no desde la fecha
de procesamiento del lote. Una lectura caducada o un último reporte inválido
mantienen la relación sin cobertura. El relleno no se utiliza al construir el mapa.

El TTL aprendido es el menor entre el límite general y dos veces el percentil 95
del intervalo de reportes válidos de Desarrollo. Un límite explícito por entidad
tiene prioridad; sin 30 intervalos se conserva el límite configurado. Un intervalo
entre cambios no necesariamente representa la frecuencia de muestreo, por lo que
el valor inferido no amplía automáticamente el máximo. Los estados no caducan por
esta regla de continuos, pero un reporte inválido interrumpe su disponibilidad.
Ventanas, TTL, desfases y rangos se guardan en el mapa y no se recalibran con
Evaluación. Debe reconstruirse el mapa desde Desarrollo para aplicar otra política.

Cada aporte documenta la fecha original, referencia, antigüedad, límite, método
(`bucket_median` o `last_known`) y motivo de ausencia. Si no hay peso contextual
observado, `context_score` y `coherence_score` se devuelven como `null` y la interfaz
indica **No evaluable**. La ausencia de evidencia no equivale a coherencia 0.5.

## Causa común e identificabilidad

La coincidencia se limita a la entidad y sus vecinos aceptados, con el mismo
patrón sospechoso en los 300 segundos anteriores. Se requieren tres entidades.
Se registra como `contextual_coincidence`, no como causa atribuida. El replay
conserva el último evento por entidad/patrón entre lotes, elimina entradas vencidas
y limpia este estado al reiniciar. El resultado usa el orden cronológico del
flujo y solo sospechas ya procesadas, incluyendo en empates temporales.

Sin telemetría independiente, `candidate=false`, `status="not_evaluated"` e
`identifiable=false`. La coincidencia es descriptiva: no veta un respaldo suficiente
ni aumenta la sospecha. `network`, `power` e `integration` son dominios que haría
falta observar, no dominios detectados. La interfaz ya no los muestra como causas.
Una futura atribución necesita señales reales de infraestructura y reglas de
identificación; esta versión no las simula ni convierte correlación en causalidad.

## Actualización del mapa

El mapa se reconstruye automáticamente cuando:

- no existe o cambia su versión/configuración;
- cambia el hash del catálogo de entidades o del histórico;
- se cruza un límite estacional al incorporar un histórico más reciente;
- supera siete días desde su construcción.

Al reconstruir se compara con el mapa anterior. Una variación media de
asociación de 0,20, una fracción relevante de aristas con ese cambio o una
variación media de cobertura de 0,20 quedan registradas como
`association_drift` o `coverage_drift`. También existe reconstrucción manual
mediante `POST /context/map/rebuild`.

## API y visualización

- `GET /context/map`: mapa, configuración, calidad, nodos, aristas y matrices.
- `POST /context/map/rebuild`: reconstrucción explícita.
- `POST /context/analyze`: vuelve a aplicar el filtro sobre las sospechas en
  memoria.
- `GET /context/results`: resultados contextuales acumulados.

La pestaña **Contextual** muestra el mapa de asociaciones entre sensores
continuos, de estado y mixtos en una escala \([-1,1]\). El detalle identifica
la métrica, ventana, desfase y controles. Las marcas distinguen las aristas
aceptadas de las rechazadas.

## Trazabilidad de las seis debilidades

1. La regla de arista ahora define ventana, \(k\), umbrales, muestra mínima,
   cobertura y estabilidad temporal.
2. El mapa incluye relaciones mixtas punto-biseriales y estados
   uno-contra-el-resto.
3. \(c_j(t)\) está definido, normalizado y combinado explícitamente con signo,
   desfase y estabilidad.
4. Contradicción, evidencia neutral y cobertura insuficiente son salidas
   distintas.
5. Los patrones multientidad activan una advertencia de causa común no
   identificable sin evidencia independiente.
6. La actualización tiene disparadores automáticos por versión, catálogo,
   fuente, edad, estación y deriva, más reconstrucción manual.
