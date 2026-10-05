# Verificación de la refactorización

Validación local realizada el 30 de septiembre de 2026.

## Código y contratos

- Backend: **61 pruebas correctas** con `scripts/run_tests.py --backend-only`.
- Frontend: **47 pruebas correctas**, incluidos eventos de clic y teclado; `npm run build` correcto.
- Rutas HTTP comparadas con el inventario anterior: se conserva el conjunto, excepto `POST /diagnostics/pretrain`, retirado deliberadamente.
- Arquitectura admitida: `gat_lstm_v3_segments_multiscale`; se rechazan manifiestos sin arquitectura, v1 y solicitudes del método de prototipos.
- Verificados el estado sin modelo y la ausencia de diagnósticos por fallback.
- Entrenamiento corto, guardado y recarga v2 verificados por las pruebas de integración.

## Conservación numérica

El filtro estadístico se comparó con una referencia generada desde el código previo
usando el mismo CSV determinista y la misma configuración. Los resultados completos,
incluida la evidencia, conservan el hash
`2d6efe0dd75001168472ec0cc7decf114375d53b26ee28435416baf2db995329`.
Las pruebas contextuales conservan las comparaciones con la implementación escalar de
referencia, las ventanas causales y los casos de datos inválidos y duplicados.

Se guardaron entradas y salidas del modelo activo antes de mover código. Al recargar
sus mismos pesos desde la nueva estructura, ambas cabezas produjeron salidas
idénticas bit a bit (error máximo 0). Esta comparación corresponde al caso de
regresión determinista, no a un nuevo entrenamiento o estudio de calidad predictiva.

## Almacenamiento

La migración local está completa y su repetición no introdujo conflictos.
El inventario detallado permanece en `storage/migrations/layout-v2.json`.
Se verificaron integridad SQLite, conteos antes/después del traslado y claves foráneas.
Las cuatro bases no presentan violaciones de claves foráneas.

- Revisión conservada: 14 datasets, 5.506.654 eventos, 927 anotaciones y 1.110 revisiones.
- Se conservaron las 32 definiciones de flujo; las que usan un modelo retirado requieren selección de v2.
- Se conservaron dos modelos v2: el activo `6bf93f476f564ce7930b0f402e3306aa` y el publicado `669501ff239c48eaaeafce499bdb8d43`.
- Se retiraron 11 modelos antiguos, 3 datasets exclusivos, 2 configuraciones y 59 referencias generadas del clasificador por patrones, junto con sus resultados asociados.
- El paquete versionado de `artifacts/gat_lstm/` no fue modificado.

Las pruebas de migración cubren planificación, ejecución, repetición, destino
incompatible, conservación de anotaciones y rechazo si la API sigue activa.

## Docker e interfaz

Se construyeron las imágenes de Compose y se probaron en contenedores desechables,
con los pesos conservados montados en solo lectura. El recorrido HTTP cargó 1.152
lecturas históricas, ejecutó el análisis, cargó 144 observaciones futuras y guardó
un flujo. Sus cinco nodos (entrada, estadístico, contextual, clasificador y salida)
procesaron las 144 observaciones; la consulta paginada recuperó los resultados
persistidos. La interfaz se sirvió desde Docker y se comprobó en el navegador.

El recorrido se puede repetir con `scripts/smoke_api.py --disposable` contra una API
de prueba en el puerto 18000. El script modifica únicamente los datos de la API
indicada, por lo que debe usarse sobre almacenamiento desechable.

Los contenedores de comprobación se eliminan al terminar. Para utilizar los datos
migrados de la instalación: `docker compose up --build` desde Artifact.
