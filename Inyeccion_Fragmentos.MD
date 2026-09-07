# Inyecciones sobre fragmentos reales revisados

El generador crea casos semisintéticos: toma observaciones conservadas de un
CSV y modifica el estado de una entidad durante un intervalo controlado. La
intervención queda documentada por construcción. La fiabilidad del supuesto de
normalidad previo sigue dependiendo de la revisión del fragmento.

## Procedimiento en la interfaz

1. Importa cada partición en **Revisión y etiquetado**. Revisa el fragmento y
   guarda una anotación **Normal**, con responsable y descripción de evidencia
   de al menos 12 caracteres. Usa una descripción concreta; cumplir la longitud
   mínima no acredita su calidad. Una revisión visual puede quedar en borrador.
2. En **Generar inyecciones**, selecciona esa anotación guardada. El generador
   usa su entidad principal, sus entidades de contexto y la revisión indicada.
   Al guardar en el paso 4, la anotación queda seleccionada automáticamente.
   Las anotaciones incompletas también aparecen: el paso 5 indica qué campo
   debes completar antes de habilitar la generación.
3. Define el inicio y fin de la intervención en UTC, dentro del intervalo
   revisado. Estos límites determinan la duración; las fechas admiten precisión
   de microsegundos. El contexto anterior y posterior se conserva.
4. Selecciona el tipo de falla, configura la intensidad cuando corresponda y
   fija la semilla. La intensidad se expresa en las unidades del sensor; no se
   calcula a partir de los umbrales del detector.
5. Pulsa **Generar y guardar caso**. Examina la comparación y las lecturas
   modificadas. Los gráficos y tablas son vistas reducidas; el ZIP conserva la
   resolución completa. Una operación que no cambie ninguna lectura se rechaza.
6. Descarga el ZIP. El caso queda guardado y puede recuperarse después de
   reiniciar el servicio. Editar posteriormente la anotación no cambia el caso
   que ya se generó: hay que crear otro para utilizar la nueva revisión.

Cada caso admite hasta 250 000 filas entre el fragmento y su contexto. Si se
supera ese límite, reduce el intervalo de la anotación o el margen de contexto.

## Tipos de intervención

| Tipo | Entidad | Operación |
| --- | --- | --- |
| Desplazamiento | Continua | Suma una cantidad constante, positiva o negativa, a cada valor válido de la ventana. |
| Deriva | Continua | Añade un desplazamiento que crece linealmente con el tiempo desde cero hasta la intensidad especificada entre los límites solicitados. |
| Ruido | Continua | Añade ruido gaussiano de media cero y desviación estándar positiva indicada, con semilla reproducible. |
| Pico | Continua | Suma la intensidad únicamente a la primera lectura válida dentro de la ventana; la etiqueta corresponde al instante de esa lectura. |
| Valor congelado | Continua o discreta | Mantiene el estado de la primera lectura válida de la ventana durante el resto de la intervención. |
| Indisponibilidad | Continua o discreta | Sustituye las lecturas válidas por `unavailable`. Conserva las filas y los tiempos; no simula ausencia de mensajes. |
| Estado sustituido | Discreta | Impone un estado válido observado de esa entidad durante la ventana. |

Se conservan los estados que ya eran inválidos. Solo cambia la columna `state`
de la entidad objetivo: las otras columnas, entidades y filas fuera de la
intervención mantienen sus valores originales. Para una deriva deben existir
lecturas posteriores al inicio; para congelar debe haber algún cambio original
que pueda suprimirse. Si no hay lecturas modificables, acorta o cambia la ventana.

Las operaciones actúan sobre registros existentes. No añaden reportes periódicos
ni modifican la cadencia de adquisición. Esta versión no simula cambios físicos
ambientales entre varias entidades: esos escenarios requieren relaciones físicas
y reglas de generación explícitas que respalden esa interpretación.

## Archivos de cada caso

- `original.csv`: fragmento revisado con su contexto, conservando los campos
  originales y ordenado por tiempo y fila fuente.
- `injected.csv`: mismo fragmento con las intervenciones aplicadas.
- `ground_truth.csv`: episodio de falla sintética de la entidad intervenida.
  En un pico, `source_row` restringe además la etiqueta a una sola observación
  cuando hay marcas temporales duplicadas.
- `interventions.csv`: fila fuente, entidad, tiempo, estado original y estado
  inyectado de cada lectura que efectivamente cambió.
- `manifest.json`: identificación del caso y de la fuente, revisión y evidencia
  congeladas, partición, semilla, parámetros, versiones del generador y de
  Python, y SHA-256 de cada CSV exportado.

El SHA-256 de la fuente corresponde al CSV importado. Los CSV del ZIP vuelven a
serializarse en UTF-8; sus bytes pueden diferir de los originales aunque los
campos se conserven. Las filas rechazadas al importar permanecen en el archivo
fuente y no se convierten en observaciones válidas para generar casos.

## Interpretación para la evaluación

- Un fragmento **Normal / Borrador** produce un caso exploratorio con
  `evaluable=false`. Conocer la intervención no confirma la normalidad previa.
- Un fragmento **Normal / Verificado** puede producir un caso con
  `evaluable=true` para la evaluación semisintética, según el protocolo del
  estudio. Esa marca conserva la declaración del revisor; no convierte la
  falla inyectada en una falla observada en el mundo real.
- La etiqueta de intervención no se extiende a las entidades de contexto ni
  convierte el resto del fragmento en ejemplos negativos. Conserva las
  anotaciones normales originales para construir los controles que admita el
  protocolo y medir falsas alarmas.
- Todas las variantes heredan la partición de la fuente. No repartas variantes
  del mismo fragmento entre Desarrollo, Validación y Prueba. Define los
  escenarios antes de la evaluación final y evita ajustarlos según sus resultados.
- La misma fuente, revisión, parámetros, semilla y versión del generador
  reproducen los valores alterados. El identificador y la fecha del caso pueden
  cambiar entre ejecuciones.

Los originales y casos generados se guardan bajo `REVIEW_DATA_DIR`
(`data/review` por defecto). Respalda el directorio completo. El generador
prepara archivos; no ejecuta automáticamente el detector ni calcula métricas.
