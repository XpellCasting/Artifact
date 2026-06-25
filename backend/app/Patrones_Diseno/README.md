# Patrones de diseno aplicados

## Unidad de Trabajo: `UnidadTrabajoIteraciones`

Problema: ejecutar iteraciones una por una provocaba un analisis completo del historico y una actualizacion de timeline por cada tick.

Solucion: agrupar las iteraciones en una unidad de trabajo. La unidad inserta todos los ticks simulados y luego el backend ejecuta el filtro estadistico una sola vez sobre `historico + realtime`.

Efecto esperado: menos llamadas HTTP, menos concatenaciones de DataFrames, menos reanalisis completos y una sola actualizacion de UI al terminar el lote.
