# Tecnicas para filtro probabilistico

Este documento resume tecnicas estadisticas y probabilisticas mencionadas en los papers de `Papers para el filtro probabilistico/`. El objetivo no es afirmar que todas estan implementadas, sino dejar claro que aportan, cuando sirven y como se comparan con el filtro estadistico-temporal del artefacto.

## Papers consultados

- `2012_08_05_Carter_IEEESSP_FP.pdf`: propone pEWMA como variante probabilistica de EWMA para deteccion en streams.
- `2422-157-2406-3522-2-20180504.pdf`: describe SH-ESD+ como mejora de Seasonal-Hybrid ESD.
- `3097983.3098144.pdf`: describe SPOT y DSPOT con teoria de valores extremos para streams.
- `FULLTEXT02.pdf`: aplica Holt-Winters para pronostico y deteccion por residuales.
- `milets18_paper_19.pdf`: compara Holt-Winters, SARIMA y Kalman Filter en series industriales.
- `zde27.pdf`: revision comparativa amplia de metodos de deteccion de anomalias en series temporales.

## EWMA

**Nombre:** Exponentially Weighted Moving Average.

**Idea:** mantiene una media suavizada de la serie dando mas peso a las lecturas recientes. En forma simple:

```text
mu_t = alpha * x_t + (1 - alpha) * mu_{t-1}
```

**Uso para anomalias:** se compara la lectura observada contra el valor esperado por EWMA. Si el residual es grande, la lectura puede ser sospechosa.

**Ventajas:**

- Bajo costo computacional.
- Funciona online, lectura por lectura.
- Bueno para suavizar ruido menor.

**Limitaciones:**

- Puede adaptarse demasiado rapido a una anomalia.
- Un pico grande puede contaminar temporalmente la media.
- No modela bien estacionalidad por si solo.

**Relacion con el artefacto:** ya se usa EWMA, pero solo como predictor interno para CUSUM sobre residuales. No se usa como indicador final independiente.

## pEWMA

**Nombre:** Probabilistic EWMA.

**Idea:** variante de EWMA que adapta dinamicamente el peso de actualizacion segun la probabilidad de la observacion. Si una lectura parece improbable, el modelo evita incorporarla con demasiado peso.

**Uso para anomalias:** estima una prediccion y una incertidumbre. Una observacion con baja probabilidad bajo el modelo se marca como sospechosa.

**Ventajas:**

- Mas robusto que EWMA frente a picos abruptos.
- Mantiene capacidad online.
- Reduce contaminacion del modelo por valores anomalos.

**Limitaciones:**

- Requiere modelar probabilidad o distribucion de la observacion.
- Mas parametros que EWMA simple.
- Puede ser excesivo si ya se usa Hampel/MAD y CUSUM robusto.

**Relacion con el artefacto:** no esta implementado. Seria candidato futuro si se quiere reemplazar el EWMA interno por un predictor probabilistico.

## MedianMethod

**Nombre:** Metodo basado en mediana.

**Idea:** usa una referencia robusta, normalmente mediana y dispersion robusta, para evitar que la media sea arrastrada por extremos.

**Uso para anomalias:** una lectura es rara si se aleja demasiado de la mediana o de una banda robusta.

```text
z_robusto = |x_t - mediana| / MAD_escalado
```

**Ventajas:**

- Simple y explicable.
- Robusto ante outliers.
- Buen ajuste para sensores continuos domesticos con ruido y picos.

**Limitaciones:**

- No captura bien tendencia o estacionalidad si se usa una sola mediana global.
- Puede fallar si la serie tiene varios regimenes normales.

**Relacion con el artefacto:** ya se usa en espiritu mediante mediana, MAD, percentiles robustos y Hampel.

## ARIMA

**Nombre:** AutoRegressive Integrated Moving Average.

**Idea:** modelo de pronostico que combina autorregresion, diferenciacion e informacion de errores pasados.

```text
ARIMA(p, d, q)
```

**Uso para anomalias:** se pronostica el siguiente valor y se evalua el residual:

```text
residual_t = x_t - prediccion_t
```

Si el residual supera un umbral, se marca sospecha.

**Ventajas:**

- Util para series con dependencia temporal clara.
- Puede modelar tendencia luego de diferenciar.
- SARIMA extiende la idea para estacionalidad.

**Limitaciones:**

- Requiere seleccion de parametros.
- Mas costoso que EWMA/Hampel.
- Sensible a datos faltantes, cambios de regimen y configuracion.

**Relacion con el artefacto:** no esta implementado. Para este artefacto seria demasiado pesado como primer filtro; podria evaluarse despues como predictor alternativo en sensores muy regulares.

## Holt-Winters

**Nombre:** Triple Exponential Smoothing.

**Idea:** extiende suavizado exponencial para modelar nivel, tendencia y estacionalidad.

```text
valor esperado = nivel + tendencia + estacionalidad
```

**Uso para anomalias:** se pronostica la siguiente lectura y se compara el residual contra una distribucion o umbral.

**Ventajas:**

- Adecuado para series con ciclos diarios/semanales.
- Mas interpretable que muchos modelos complejos.
- Puede funcionar online.

**Limitaciones:**

- Requiere estacionalidad relativamente estable.
- Sufre con datos faltantes o historicos contaminados.
- No es ideal para sensores con reportes muy irregulares.

**Relacion con el artefacto:** no esta implementado. Puede servir como filtro posterior para sensores continuos con ciclos repetitivos claros.

## Kalman Filter

**Nombre:** Filtro de Kalman.

**Idea:** estima un estado oculto de un sistema dinamico combinando prediccion y medicion. Cada lectura corrige el estado estimado segun la incertidumbre del modelo y de la medicion.

**Uso para anomalias:** si la medicion se aleja demasiado de la prediccion del estado, el residual o innovacion se marca como sospechoso.

**Ventajas:**

- Bueno para series con ruido de medicion.
- Funciona online.
- Puede separar senal real y ruido si el modelo esta bien calibrado.

**Limitaciones:**

- Requiere definir modelo de estado y ruido.
- Puede ser innecesario para sensores simples.
- Mala calibracion produce falsos positivos o falsos negativos.

**Relacion con el artefacto:** no esta implementado. Podria ser util en una etapa posterior para sensores continuos con dinamica fisica conocida.

## S-H-ESD / SH-ESD+

**Nombre:** Seasonal Hybrid Extreme Studentized Deviate.

**Idea:** combina descomposicion estacional con una prueba estadistica ESD para detectar puntos extremos despues de remover tendencia/estacionalidad.

**Uso para anomalias:** primero separa componentes de la serie; luego aplica un test robusto sobre los residuales.

**Ventajas:**

- Bueno para series con estacionalidad.
- Detecta anomalias locales y globales.
- SH-ESD+ mejora limitaciones del SH-ESD original usando metricas robustas.

**Limitaciones:**

- Mas orientado a analisis por lotes que a tick realtime simple.
- Requiere suficiente historial estacional.
- Mas dificil de explicar que Hampel/MAD directo.

**Relacion con el artefacto:** no esta implementado. Es una referencia fuerte si se quiere formalizar deteccion estacional en una version posterior.

## DSPOT

**Nombre:** Drift SPOT.

**Idea:** usa teoria de valores extremos para calcular umbrales dinamicos en streams. DSPOT adapta SPOT a contextos con drift restando un comportamiento local.

```text
x'_t = x_t - M_t
```

Donde `M_t` representa el comportamiento local esperado.

**Uso para anomalias:** mantiene un umbral extremo para la cola de la distribucion. Si la lectura supera ese umbral, se activa alarma.

**Ventajas:**

- Pensado para streams.
- No asume una distribucion completa de los datos.
- Bueno para eventos extremos.
- DSPOT considera drift.

**Limitaciones:**

- Requiere fase de calibracion.
- Parametros como riesgo `q` y ventana local afectan bastante el resultado.
- Puede ser excesivo para sensores domesticos simples si no se necesita teoria de valores extremos.

**Relacion con el artefacto:** no esta implementado. Seria una buena tecnica futura para umbrales extremos adaptativos en tiempo real.

## Comparacion con el filtro actual

| Tecnica | Implementada | Rol posible |
|---|---:|---|
| EWMA | Parcial | Predictor interno para CUSUM |
| pEWMA | No | Predictor probabilistico futuro |
| MedianMethod | Parcial | Base robusta del filtro actual |
| ARIMA | No | Predictor para series regulares |
| Holt-Winters | No | Predictor con estacionalidad |
| Kalman Filter | No | Estimacion online con ruido |
| S-H-ESD / SH-ESD+ | No | Deteccion estacional por residuales |
| DSPOT | No | Umbrales extremos en streams |

## Decision para el artefacto

Para la etapa actual conviene mantener el filtro implementado: Hampel, tasa de cambio robusta, CUSUM sobre residuales, persistencia constante y brecha temporal. Este enfoque es mas simple, explicable y coherente con la tesis.

Las tecnicas de esta lista sirven como respaldo teorico y como candidatas para evolucionar el filtro probabilistico:

- `pEWMA`: reemplazo robusto del EWMA interno.
- `Holt-Winters` o `S-H-ESD+`: mejor tratamiento de estacionalidad.
- `DSPOT`: umbrales extremos dinamicos para tiempo real.
- `Kalman Filter`: sensores continuos con ruido y comportamiento fisico modelable.
