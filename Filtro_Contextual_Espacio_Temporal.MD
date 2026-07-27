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

La versión `contextual-spatiotemporal-v1` usa estos valores por defecto:

| Parámetro | Valor |
|---|---:|
| Tamaño de ventana | 3600 s |
| Desfases evaluados | 0, 3600 y 7200 s, solo hacia contexto presente o pasado |
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

Estos umbrales son decisiones operacionales versionadas del prototipo; las
referencias sustentan las familias de análisis, no estos valores numéricos
específicos. Deben tratarse como hiperparámetros en la evaluación experimental.

## Familias de relación

- Continua–continua: correlación de Pearson sobre medianas por ventana. La
  matriz visible conserva además la covarianza cruda.
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

## Estados de salida

- `supported`: cobertura suficiente y masa de respaldo mayor o igual a 0,35;
  disminuye la sospecha, sin afirmar causalidad.
- `contradicted`: cobertura suficiente y masa de contradicción efectiva mayor
  o igual a 0,35; aumenta la sospecha.
- `unexplained`: existe cobertura, pero la evidencia es neutral o conflictiva;
  mantiene la sospecha.
- `insufficient_coverage`: menos de 0,50 del peso elegible tiene observación
  válida; mantiene la sospecha.

La falta de respaldo ya no se confunde con una contradicción.

## Causa común e identificabilidad

Si tres o más entidades sospechosas coinciden en la misma ventana, el resultado
marca `common_cause.candidate=true`. Mientras el conjunto no incluya evidencia
independiente sobre red, alimentación o integración, el estado es
`unidentifiable_without_independent_evidence`.

Este resguardo impide concluir “anomalía ambiental” solo porque varias entidades
cambiaron de forma coherente. Además bloquea cualquier ajuste contextual de la
sospecha (`adjustment_blocked_by_common_cause=true`): el puntaje estadístico se
mantiene hasta disponer de telemetría independiente. El histórico correlacional
puede ser compatible tanto con un cambio real del entorno como con una falla
común.

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

La pestaña **Contextual** muestra por defecto la matriz de covarianza calculada
del histórico. La vista **Asociación** permite comparar relaciones continuas,
de estado y mixtas en una escala \([-1,1]\). Una celda `N/D` en covarianza
significa “no aplicable”; no se sustituye por cero.

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
