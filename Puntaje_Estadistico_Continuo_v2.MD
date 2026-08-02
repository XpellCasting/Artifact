# Corrección de la saturación del puntaje estadístico continuo

## 1. Propósito

Este documento describe la versión
`statistical-continuous-score-v2.0.0`, aplicada al filtro
estadístico-temporal de entidades continuas.

El cambio busca recuperar capacidad de ordenamiento entre lecturas atípicas
que antes terminaban con el mismo puntaje. No se usó la base simulada para
definir la transformación ni para evaluar sus resultados. La comprobación se
realizó exclusivamente sobre el mes real disponible en
`data/datosreales/history.csv`.

La corrección mejora la resolución numérica del puntaje, pero no demuestra una
mejora de exactitud diagnóstica. Todavía no existe una base de verdad real
suficiente para estimar precisión, sensibilidad, especificidad o \(F_2\).

## 2. Problema observado

La versión anterior aplicaba dos transformaciones consecutivas. Para los
indicadores expresados como una desviación \(z\), primero calculaba:

\[
s_{\mathrm{old}}(z)=
\operatorname{clip}
\left(
\frac{z-2{,}5}{3{,}5-2{,}5},
0,
1
\right).
\]

Luego convertía la evidencia mediante:

\[
S_{\mathrm{old}}=
\frac{s_{\mathrm{old}}}
{s_{\mathrm{old}}+0{,}4}.
\]

Cuando \(z\geq3{,}5\), el primer cálculo quedaba recortado en 1. Por
consiguiente, el puntaje final siempre era:

\[
S_{\mathrm{old,max}}=
\frac{1}{1+0{,}4}
=0{,}714286.
\]

Con un umbral operacional de 0,60, la regla anterior aceptaba en realidad los
indicadores \(z\) desde \(z=3{,}1\), aunque el código presentaba 3,5 como
umbral. Esto ocurría porque \(S_{\mathrm{old}}=0{,}60\) cuando el componente
intermedio vale 0,60.

El problema no estaba limitado a Hampel. También existían recortes exactos a
1 en CUSUM, persistencia constante, brecha de lectura y límites operacionales.

### Diagnóstico previo sobre mayo

| Medida | Versión anterior |
|---|---:|
| Candidatos totales | 291 |
| Candidatos continuos | 235 |
| Candidatos binarios | 56 |
| Continuos con puntaje 0,714286 | 233 |
| Proporción en la moda 0,714286 | 99,15 % |
| Puntajes continuos distintos, a 6 decimales | 3 |
| Rango continuo | 0,662413–0,714286 |
| Mediana continua | 0,714286 |

Los 235 casos son candidatos emitidos después de la agrupación y supresión
temporal. No representan 235 fallas confirmadas ni 235 lecturas crudas.

## 3. Nueva transformación suave

Se fijaron los siguientes parámetros antes de volver a procesar el mes:

\[
z_0=2{,}5,\qquad
z_\star=3{,}1,\qquad
p_\star=0{,}60.
\]

La constante de forma se obtiene de manera analítica:

\[
\lambda=
\frac{1-p_\star}{p_\star}\ln 2
=\frac{2}{3}\ln 2
=0{,}462098120373.
\]

La función común de evidencia es:

\[
\phi(u)=
\begin{cases}
0, & u\leq0,\\[2mm]
\dfrac{\ln(1+u)}
{\ln(1+u)+\lambda}, & u>0.
\end{cases}
\]

Para una magnitud expresada como desviación \(z\):

\[
u_z=
\max\left(
0,
\frac{z-z_0}{z_\star-z_0}
\right),
\qquad
s_z(z)=\phi(u_z).
\]

La elección de \(\lambda\) garantiza que:

\[
s_z(3{,}1)=0{,}60.
\]

La función es continua y estrictamente creciente sobre \(z>2{,}5\). Para
cualquier entrada finita se cumple \(0\leq s_z(z)<1\); por tanto, ya no existe
un recorte que convierta magnitudes distintas en el mismo valor exacto.

Algunos valores de referencia son:

| \(z\) | Evidencia v2 |
|---:|---:|
| 3,1 | 0,600000 |
| 3,5 | 0,679750 |
| 6,5 | 0,815085 |
| 27 | 0,889866 |
| 150 | 0,922607 |
| 4231 | 0,950433 |

## 4. Aplicación por indicador

| Indicador | Regla v2 |
|---|---|
| Hampel | \(s_H=s_z(H_i)\) |
| Tasa robusta | \(s_R=s_z(Z_{R_i})\) |
| CUSUM | \(s_C=\phi(\max(C^+,C^-)/\theta_C)\) |
| Persistencia constante | Antes de \(T_D\), 0; desde \(T_D\), \(\max[s_z(Z_D),\phi(D/T_D)]\) |
| Brecha de lectura | Antes de \(T_M\), 0; desde \(T_M\), \(\max[s_z(Z_M),\phi(M/T_M)]\) |
| DSPOT | \(s_{\mathrm{DSPOT}}=s_z(Z_{\mathrm{DSPOT}})\) |
| Límite operacional | Evidencia suave según la distancia a los límites extremo y operacional configurados |

En CUSUM, \(\phi(1)=0{,}60\), por lo que la activación fuerte ocurre al
alcanzar el parámetro documentado \(\theta_C\). La implementación anterior
podía aceptar desde \(0{,}60\theta_C\) debido a la segunda transformación.
Esta diferencia es intencional y explica parte del cambio en la cantidad de
candidatos.

Para un límite operacional se construye una razón \(u_F=1+d/L\), donde \(d\)
es la distancia desde el límite extremo y \(L\) es la separación entre los
límites extremo y operacional. El cruce del límite extremo comienza en
evidencia 0,60 y la evidencia aumenta suavemente con la excedencia. El valor
se registra como `Z_F_i`, y la salida `component_sources.s_H` permite
distinguir si el componente dominante provino de Hampel o del límite
operacional.

Los campos heredados del código se llaman `physical_low` y `physical_high`,
pero, salvo edición explícita del usuario con conocimiento del dispositivo,
se estiman mediante percentiles del mismo histórico. Por ello no constituyen
límites físicos independientes ni especificaciones del fabricante; en la
documentación se denominan límites operacionales empíricos.

## 5. Agregación y semántica

La evidencia continua se agrega mediante:

\[
E_i(t)=
\max
\left(
s_H,
s_R,
s_C,
s_{D_{\mathrm{const}}},
s_M,
s_{\mathrm{DSPOT}}
\right).
\]

La salida final es:

\[
S_i^{\mathrm{stat}}(t)=E_i(t).
\]

Se conservan los campos `evidence_score` y `suspicion_score` por
compatibilidad con el resto del artefacto, pero ambos contienen el mismo
valor. Este valor representa fuerza de evidencia estadística y no una
probabilidad de falla. Esta interpretación es consistente con la advertencia
existente en `Tesis.Tex`: el puntaje de la primera etapa no debe interpretarse
como una probabilidad definitiva.

Una brecha de lectura sin ningún otro indicador se limita a 0,50. Con el
umbral final 0,60, ese candidato no se envía al filtro contextual. La misma
política se aplica ahora tanto al procesamiento histórico como al incremental;
antes solo estaba presente en la ruta histórica. El artefacto no mantiene una
colección separada de casos “en observación”.

El puntaje horario auxiliar de las entidades continuas también usa \(s_z\).
Así se evita conservar un segundo recorte a 1 dentro de
`temporal_context.hour_of_day_score`. El contexto horario binario permanece
sin cambios.

La versión binaria no fue modificada. Su transformación, validación y posible
saturación deben analizarse como un cambio separado, para no mezclar dos
modelos de evidencia en una misma corrección.

## 6. Trazabilidad y caché

Cada resultado continuo incorpora:

- `score_version: statistical-continuous-score-v2.0.0`;
- `score_kind: statistical_evidence_strength_not_probability`;
- `reading.components`, con los componentes normalizados;
- `reading.raw_components`, con las magnitudes sin transformar;
- `reading.component_sources`, para identificar el origen del componente
  Hampel/límite operacional;
- `reading.operational_limit_score`, con la evidencia específica de ese
  límite.

El resumen del filtro expone la especificación completa: zona muerta, ancla,
\(\lambda\), fórmulas, agregación, política de brecha aislada y declaración
`uses_simulated_dataset: false`.

La identidad de caché aumentó de la versión 15 a la 16. Ahora incluye:

- versión y parámetros del puntaje continuo;
- hash SHA-256 del historial real;
- umbral de sospecha;
- hash SHA-256 del archivo de configuración continuo realmente utilizado;
- versión, configuración y hash de implementación del filtro contextual;
- hash y estado de vigencia del mapa contextual;
- versión y huellas del clasificador.

Esto impide reutilizar resultados calculados con la transformación anterior o
con parámetros continuos o contextuales diferentes. También invalida el
resultado extremo a extremo cuando cambia o vence el mapa contextual. La caché
de normalidad permanece en la versión 2 porque almacena baselines, no puntajes
transformados.

## 7. Resultado después del cambio

La ejecución final usó:

- fuente: `data/datosreales/history.csv`;
- tipo de fuente: real;
- periodo: 1 de mayo de 2026 18:00 UTC a 31 de mayo de 2026 20:59:54 UTC;
- filas: 123.023;
- entidades: 14;
- umbral final: 0,60;
- datos simulados: no utilizados.

| Medida | Antes | v2 |
|---|---:|---:|
| Candidatos totales | 291 | 277 |
| Candidatos continuos | 235 | 221 |
| Candidatos binarios | 56 | 56 |
| Puntajes continuos distintos, a 6 decimales | 3 | 218 |
| Continuos con puntaje 0,714286 | 233 | 0 |
| Scores continuos iguales a 1 | 0 | 0 |
| Componentes continuos iguales a 1 | 233 como mínimo | 0 |
| Scores horarios continuos iguales a 1 | no auditado | 0 |
| Puntaje continuo mínimo | 0,662413 | 0,613040 |
| Mediana continua | 0,714286 | 0,899317 |
| Puntaje continuo máximo | 0,714286 | 0,948709 |
| Diferencias entre `evidence_score` y `suspicion_score` | no aplicable | 0 |

La masa artificial en 0,714286 desapareció. El aumento de la mediana no
significa que el filtro sea más exacto ni que las lecturas sean fallas más
probables; refleja que las magnitudes extremas vuelven a conservar su orden y
que el campo se interpreta como evidencia, no como probabilidad.

Los 14 candidatos continuos menos corresponden a un cambio operacional real.
La diferencia es coherente con el nuevo anclaje de CUSUM en
\(\theta_C\) y con la selección de representantes sobre una escala que ahora
conserva resolución, pero sin una comparación evento a evento no se atribuye
cada descarte a una causa única. No se reajustaron parámetros para conservar
una cantidad predeterminada de alertas.

### Efecto aguas abajo

| Resultado | Antes | v2 |
|---|---:|---:|
| Contexto respaldado | 70 | 75 |
| Contexto no explicado | 57 | 65 |
| Cobertura insuficiente | 156 | 125 |
| Contexto contradicho | 8 | 12 |
| Clasificación: lectura normal | 3 | 0 |
| Clasificación: anomalía ambiental | 60 | 65 |
| Clasificación: posible falla | 55 | 66 |
| Clasificación: indeterminado | 173 | 146 |

Estas cifras describen cómo se redistribuyeron los candidatos. No son métricas
de desempeño, porque no existe una base de verdad real con la cual determinar
aciertos, falsos positivos o falsos negativos.

## 8. Verificación realizada

Se añadieron pruebas para:

- entradas no finitas y zona muerta;
- valor exacto del ancla \(z=3{,}1\);
- valores numéricos de regresión;
- monotonía estricta;
- ausencia de saturación para entradas finitas;
- activación de razones propias en \(\phi(1)=0{,}60\);
- normalización suave de límites operacionales;
- continuidad del mismo estado CUSUM entre batch y tiempo real;
- identidad entre evidencia y sospecha;
- política de brecha aislada;
- invalidación de caché ante cambios de configuración contextual y vencimiento
  del mapa;
- versión, huellas de caché y prohibición de usar datos simulados.

Resultado:

```text
43 pruebas ejecutadas
43 pruebas aprobadas
0 fallos
```

El frontend también fue compilado correctamente con Vite. Su modal de cálculo
muestra la versión, la naturaleza del score y las fórmulas v2.

Para repetir el reprocesamiento y la auditoría:

```powershell
cd Artifact
.\backend\.venv\Scripts\python.exe .\scripts\reprocesar_mes_real_score_v2.py
```

La segunda ejecución debe informar `cache_hit: true` y conservar los mismos
resultados mientras no cambien el historial, la configuración, la versión del
score, el umbral, el filtro contextual, el mapa contextual o el clasificador.

## 9. Cambios que deben trasladarse a `Tesis.Tex`

El pseudocódigo actual calcula:

```text
evidencia <- max(s_H, s_R, s_C, s_D, s_M)
puntaje <- evidencia / (evidencia + tau)
```

Debe reemplazarse conceptualmente por:

```text
lambda <- ((1 - 0.60) / 0.60) * ln(2)

phi(u):
    si u <= 0 retornar 0
    retornar ln(1 + u) / (ln(1 + u) + lambda)

normalizar_z(z):
    u <- max(0, (z - 2.5) / (3.1 - 2.5))
    retornar phi(u)

s_H <- max(normalizar_z(H), normalizar_z(Z_F))
s_R <- normalizar_z(Z_R)
s_C <- phi(C / theta_C)
s_D <- max(normalizar_z(Z_D), phi(D / T_D))
s_M <- max(normalizar_z(Z_M), phi(M / T_M))
s_DSPOT <- normalizar_z(Z_DSPOT)

evidencia <- max(s_H, s_R, s_C, s_D, s_M, s_DSPOT)
puntaje <- evidencia
```

También se recomienda:

1. reemplazar “puntaje probabilístico” por “fuerza de evidencia
   estadística”;
2. declarar \(z_0=2{,}5\), \(z_\star=3{,}1\), \(p_\star=0{,}60\) y
   \(\lambda=\frac{2}{3}\ln2\);
3. agregar DSPOT y el tratamiento de límites operacionales empíricos a la
   tabla de indicadores continuos, sin presentarlos como especificaciones
   físicas del fabricante;
4. describir la política de brecha aislada en 0,50;
5. indicar que la v2 se aplicó provisionalmente al único mes real disponible
   y que su validación diagnóstica queda pendiente;
6. conservar la advertencia de que el score no es una probabilidad definitiva
   de falla.

## 10. Limitaciones pendientes

- Un mes no permite evaluar estacionalidad ni cambios prolongados de rutina.
- El histórico se usa para construir la normalidad y para buscar atipicidades,
  por lo que todavía existe riesgo de fuga temporal.
- La curva reduce el efecto visual de \(z\) enormes, pero no corrige por sí
  sola una MAD demasiado pequeña o un baseline contaminado.
- La agregación por máximo conserva el indicador dominante, pero no acumula
  varios indicios moderados.
- La ruta incremental conserva ahora el estado CUSUM entre ticks, pero todavía
  no mantiene la duración de persistencia constante; `s_D_const` permanece en
  cero durante esa ruta y debe completarse en una iteración posterior.
- El clasificador consume umbrales fijos y debe permanecer en modo sombra
  mientras no existan etiquetas reales suficientes.
- No corresponde ajustar la curva buscando una cantidad conveniente de
  alertas en mayo.
