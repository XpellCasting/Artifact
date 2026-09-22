# Inyecciones sobre fragmentos reales revisados

## Escenario con el histórico completo

Después de generar un caso, el paso **6. Acumular fallas y descargar el histórico
completo** lo añade a la selección actual. Revisa otra entidad, guarda su base
normal y genera la siguiente falla. Puedes incluir o retirar casos guardados
mediante la tabla, sin borrar sus archivos. La selección se conserva en este
navegador para cada histórico.

La previsualización agrupa todas las entidades seleccionadas y muestra cada
intervalo con su nombre de falla, comparación original/alterado y filas cambiadas.
Las vistas muestreadas lo indican. Dos intervenciones sobre la misma entidad no
pueden compartir instantes; entidades diferentes sí pueden intervenirse a la vez.

**Guardar escenario y descargar histórico completo** crea una instantánea
persistente con todas las entidades y filas del archivo fuente. `original.csv`
conserva los bytes originales y `injected.csv` conserva el orden, las columnas y
los valores salvo los cambios registrados. También conserva las filas no
indexables. `ground_truth.csv` registra el tipo de falla (`kind`) y el intervalo
de cada caso; `interventions.csv` identifica cada fila alterada, su falla y sus
valores antes y después. Las etiquetas quedan separadas de los datos de entrada.
El manifiesto conserva parámetros, semillas, revisiones, partición y huellas.
Los registros sin intervención no se convierten automáticamente en normales.

Desde la versión 2 del escenario, todas las anotaciones manuales actuales de
ese histórico se incorporan automáticamente, incluso sin inyecciones. El ZIP
añade `annotations.csv` con las anotaciones originales y sus revisiones fijadas.
`ground_truth.csv` contiene los intervalos efectivos para `injected.csv`, con
origen `historical_review` o `controlled_injection`. Los intervalos normales se
recortan donde se aplica una inyección; las anotaciones originales no cambian.
Pendientes e inciertos se conservan con `evaluable=false`.

Los solapamientos manuales se dividen en tramos sin duplicar su cobertura.
`annotation_references` conserva los identificadores y revisiones que contribuyen
a cada tramo. Las clases conocidas revisadas prevalecen sobre pendientes e
inciertas; clases conocidas contradictorias producen un tramo incierto.
Una revisión actual que contradiga la normalidad de la base inyectada conserva
la intervención, pero la excluye de métricas e indica el motivo.
Los picos usan `source_row` para distinguir una lectura; si hay otras lecturas
con la misma fecha, el tramo manual usa `excluded_source_rows` para excluir solo
esa fila. Las fechas son inclusivas y conservan microsegundos.

Descargar un escenario antiguo conserva su instantánea. Para incorporar las
anotaciones actuales, carga su selección y guarda un nuevo escenario.

El formulario de inyección muestra su fragmento original junto a los parámetros.
Pulsa **Fijar inicio** y una lectura, luego otra para fijar el fin. Los mismos
instantes están disponibles en una lista y en los campos ISO con microsegundos.
Si el gráfico está muestreado se indica: la generación usa todos los registros.

Los escenarios guardados pueden descargarse de nuevo o cargarse como selección
para preparar otra versión. Añadir o retirar casos no modifica descargas anteriores.

El generador crea casos semisintéticos: toma observaciones conservadas de un
CSV y modifica el estado de una entidad durante un intervalo controlado. La
intervención queda documentada por construcción. La fiabilidad del supuesto de
normalidad previo sigue dependiendo de la revisión del fragmento.

## Procedimiento en la interfaz

1. Importa cada partición en **Revisión y etiquetado**. Revisa el fragmento y
   guarda una anotación **Normal** y **Revisado**, con responsable y observaciones
   de al menos 12 caracteres. Usa una descripción concreta; cumplir la longitud
   mínima no acredita su calidad. No se requiere evidencia externa. La normalidad
   se asume a partir del histórico; el origen del caso será **Inyección controlada**.
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

- Un fragmento **Normal / Pendiente** requiere completar la revisión antes de
  generar desde la interfaz. Los casos antiguos exploratorios conservan
  `evaluable=false`.
- Un fragmento **Normal / Revisado** produce un caso con `evaluable=true`
  para la evaluación semisintética. El origen es **Inyección controlada** y
  `baseline_normality_basis=assumed_from_history` documenta la normalidad
  asumida. Conocer la intervención no confirma la normalidad previa ni convierte
  la falla inyectada en una falla observada en el mundo real.
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

## Histórico completo con marcas manuales e inyecciones

El paso 6 permite acumular casos de distintas entidades del mismo CSV y descargar
un único **CSV etiquetado** con `entity_id,state,last_changed,es_sintetica,tipo,tipo_falla`.
Se incluyen todas las filas y todas las marcas manuales actuales. `es_sintetica=true`
identifica exclusivamente valores que cambiaron en una intervención seleccionada;
las demás filas llevan `false`. `tipo` usa `normal`, `anomalia`, `Falla` o `NULL`.
`tipo_falla` contiene el nombre de la falla, o `NULL` si no aplica o no se conoce.

La marca de una fila prevalece sobre una anotación de intervalo. En los registros
modificados por una inyección prevalece la falla sintética. Los datos restantes
no se convierten automáticamente en normales. Los escenarios permiten exportar
marcas manuales sin seleccionar inyecciones. Las pendientes e inciertas quedan
sin etiqueta definitiva en el CSV y se conservan documentadas en el respaldo.

El esquema 3.0 conserva un `labelled.csv` descargable mediante
`GET /review/scenarios/{scenario_id}/csv`. El ZIP de respaldo conserva también
`original.csv`, `injected.csv`, `ground_truth.csv`, `interventions.csv`,
`annotations.csv` y `manifest.json`, junto con los hashes. `ground_truth.csv`
describe episodios; `labelled.csv` es la referencia por registro y distingue los
valores efectivamente modificados de los demás registros del episodio.

Los escenarios anteriores conservan sus archivos. Para usar el CSV nuevo, carga
la selección del escenario y guarda otra versión. La revisión, la generación y
la descarga no ejecutan el clasificador ni los filtros estadísticos.
