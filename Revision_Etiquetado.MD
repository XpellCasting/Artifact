# Revisión y etiquetado de fragmentos reales

Esta herramienta prepara evidencia para la base de verdad de la tesis. Una
anotación registra el juicio del revisor y su fundamento. El software conserva
esa declaración; no confirma por sí mismo la causa de un evento.

## Procedimiento por registro

1. Importa el CSV y selecciona su partición: Desarrollo, Validación o Prueba.
   La revisión y la exportación no ejecutan filtros ni entrenan clasificadores.
2. En **Explorar histórico**, elige una entidad y pulsa un día de **Actividad
   diaria**. La inspección abre ese día en UTC, de 00:00:00 a 23:59:59.999999;
   los extremos se limitan a la cobertura del archivo. Los días sin reportes
   permanecen visibles y no reciben una etiqueta automática.
3. En **Revisar y marcar lecturas**, pulsa un punto. El panel muestra la fecha
   exacta, el estado y la fila original. Elige **Normal**, **Anomalía** o
   **Falla → Tipo de falla**. Pulsar la clase o el tipo guarda la marca de ese
   registro y permite seguir revisando el mismo día. Puedes volver a pulsarlo
   para corregirla; se conserva una nueva revisión de la anotación.
4. Usa **Día anterior / Día siguiente** para continuar con la misma entidad.
   Los puntos verdes son normales; ámbar, anomalías; rojos, fallas; grises,
   registros sin etiqueta definitiva. El selector de lecturas permite elegir
   filas cuyos puntos se superponen. La identidad es la fila de origen, no solo
   su fecha: dos lecturas con la misma fecha se pueden etiquetar por separado.
5. Si necesitas una marca para un tramo completo o una base normal para
   inyectar, abre **Anotación de intervalo (opcional) y base para inyecciones**.
   Documenta el tramo y guarda su revisión. Las marcas puntuales prevalecen
   sobre las de intervalo; una marca de un punto no etiqueta todo el día.
6. Genera las inyecciones opcionales con la herramienta existente. Su gráfica
   original y su selector de lecturas permiten fijar fechas junto al formulario.
7. En el paso 6, selecciona las inyecciones y pulsa **Guardar escenario y
   descargar CSV etiquetado**. Se incluyen todas las anotaciones actuales del
   archivo. También puedes exportar solo las marcas manuales, sin inyecciones.

En **Revisar y marcar lecturas**, usa la rueda sobre la gráfica para acercar
o alejar el eje temporal alrededor del cursor. **Ver intervalo completo** restaura
la vista y cambiar de día reinicia el zoom. La precisión hasta microsegundos
permite separar lecturas cercanas sin modificar sus fechas ni sus etiquetas.
Si la vista estaba muestreada, el zoom consulta las lecturas del área ampliada;
el aviso de muestreo permanece cuando aún hay más registros que puntos visibles. La exportación
conserva todas las filas del CSV de origen; no se limita al día o a la entidad
abiertos. Ninguna navegación, selección de punto o apertura de menú asigna
una etiqueta: se guarda al pulsar una clase o un tipo de falla.

Para continuas, el catálogo ofrece Pico, Desplazamiento, Deriva, Ruido, Valor
congelado e Indisponibilidad. Para binarias y multiestado ofrece Valor congelado,
Indisponibilidad y Estado sustituido. El servidor comprueba la compatibilidad
al guardar una marca puntual. Las categorías originales, incluido `NONE`, se
conservan como estados, separadas de `unknown` y `unavailable`.

## Persistencia de las marcas

Cada marca puntual se confirma en `data/review/review.sqlite3` antes de mostrarla
como guardada. Al abrir el mismo histórico, la interfaz recupera las marcas y
sus colores desde SQLite. Cerrar el navegador o detener y volver a iniciar la
aplicación no las elimina. El directorio `data/review` está montado fuera del
contenedor y debe conservarse al actualizar los servicios.

**Anotaciones del histórico** muestra solo notas de intervalo. Las marcas
puntuales permanecen en SQLite, visibles en las gráficas y en el CSV completo,
sin añadir tarjetas a ese listado. La gráfica muestra el número de registros
marcados y espera a recuperar las marcas antes de permitir nuevas etiquetas.
Las marcas existentes siguen usando su misma identidad y sus revisiones.

## CSV de la base de verdad

La descarga principal contiene exactamente:

```csv
entity_id,state,last_changed,es_sintetica,tipo,tipo_falla
```

- `entity_id`, `state` y `last_changed`: campos originales, en el orden de las
  filas de la fuente. Solo `state` cambia en las intervenciones seleccionadas.
  Si la fuente usa `timestamp`, se exporta como `last_changed` sin reformatearlo.
- `es_sintetica`: `true` exclusivamente si una inyección cambió ese valor;
  `false` para los demás registros, incluidos los que no han sido revisados.
- `tipo`: `normal`, `anomalia` o `Falla`. Sin una marca definitiva es `NULL`.
- `tipo_falla`: nombre de la falla manual o inyectada; `NULL` para normalidad,
  anomalías, datos sin revisar o fallas históricas cuyo tipo quedó indeterminado.

La prioridad por fila es: modificación sintética efectiva, marca manual puntual,
marca manual de intervalo, `NULL`. Una fila sin cambio dentro de una intervención
conserva su marca manual o queda sin etiqueta; no se asume normal. Las anotaciones
pendientes o inciertas se conservan en el respaldo, con etiquetas `NULL` en el CSV.

Cada escenario es una versión guardada. Cambiar una anotación después no altera
una descarga anterior. Para incorporar cambios o usar el formato nuevo en un
escenario antiguo, carga su selección y guarda un nuevo escenario. El **ZIP de
respaldo** añade el original, `labelled.csv`, las intervenciones, las anotaciones
con su revisión y un manifiesto con hashes. Las fuentes y anotaciones no se borran.

## Qué observar

| Observación | Qué investigar antes de etiquetar |
| --- | --- |
| Cambio brusco | Acción física documentada, cambio de unidad, reinicio o error de medición. |
| Deriva | Evolución ambiental, calibración y contraste con una referencia independiente. |
| Valor constante | Estabilidad real del entorno, resolución del sensor y disponibilidad de reportes. |
| Intervalo sin filas | Registro por cambios de estado, pérdida de conexión o falta de adquisición. |
| Cambio de varias entidades | Fenómeno físico compartido o problema común de alimentación, comunicación o registro. |
| Cambio de una sola entidad | Evento local, escasa cobertura contextual o error en la medición. |

La ausencia de reacción en sensores vecinos no demuestra una falla. La
correlación entre varias entidades tampoco demuestra por sí sola un evento
ambiental. Conserva los casos ambiguos para análisis cualitativo.

## Clases y estado de la evidencia

- **Normal:** comportamiento válido y habitual dentro de los criterios del
  estudio. Un período sin modificaciones sintéticas no queda automáticamente
  verificado como normal.
- **Anomalía ambiental:** cambio físico real e inusual según los criterios
  previamente definidos; describe el fenómeno y la evidencia que lo respalda.
- **Falla del sensor:** evidencia de medición no confiable. Especifica si la
  evidencia atribuye el problema al dispositivo, a su comunicación o al
  registro; no atribuyas una falla física sin respaldo.
- **Incierto:** evidencia insuficiente para decidir. Es un estado del
  etiquetado, no una cuarta clase diagnóstica que deba aprender el clasificador.

`evaluable` identifica las anotaciones revisadas de clase conocida. Las
métricas sobre ellas miden acuerdo con la interpretación del histórico, no
certeza sobre la causa real. La normalidad histórica se asume tras la revisión;
no constituye una certificación automática de su veracidad.
La revisión del protocolo de la tesis debe decidir qué evidencia admite en
las métricas principales y conservar separadamente los casos inciertos.

## ZIP individual de una anotación (respaldo opcional)

- `events.csv`: campos originales de las filas de la entidad objetivo y de las
  entidades de contexto, dentro del intervalo observado incluyendo su margen.
  Las filas no contienen etiquetas añadidas para el detector. Se preservan
  valores y precisión temporal; el CSV se serializa de nuevo y puede cambiar
  su formato de comillas respecto al archivo de origen.
- `ground_truth.csv`: etiqueta, entidad objetivo, intervalo central y datos de
  revisión. Fuera de ese intervalo y para las entidades de contexto no se
  presume normalidad.
- `manifest.json`: procedencia del archivo, SHA-256, partición, límites de
  selección, versión de la anotación y evidencia. También documenta incidencias
  de importación y la separación entre datos y etiquetas.

Los archivos originales y las anotaciones se conservan en `data/review`.
Los tiempos se indexan hasta microsegundos; las filas con una precisión
significativa mayor o identificadores con espacios exteriores se informan
como no indexables y permanecen en la fuente original.
La base de datos y su índice permiten consultar fragmentos sin ejecutar el
pipeline estadístico. Para respaldar el trabajo conserva este directorio
completo, preferiblemente con el servicio detenido.

## Uso experimental

Revisa muestras repartidas entre días, horarios, entidades y niveles de
actividad, incluyendo períodos sin alertas. Revisar solamente los casos que
señala el detector sesga la muestra y dificulta estudiar sus omisiones.

Mantén Desarrollo, Validación y Prueba identificados por fecha y finalidad.
Las anotaciones de Prueba sirven a la evaluación final; no deben utilizarse
para ajustar los filtros. Al reproducir los fragmentos, el detector debe seguir
procesando cronológicamente y utilizar solo observaciones disponibles hasta
cada instante, aunque el revisor haya consultado el contexto posterior para
anotar el evento.

Después de guardar un fragmento como **Normal**, puedes usar **Generar
inyecciones** en esta misma pestaña. El generador exige un revisor y una
descripción de la revisión y el estado **Revisado**. La intervención conocida
se exporta con origen **Inyección controlada** y conserva el supuesto de
normalidad de la base. Los casos exploratorios antiguos no se reclasifican. Consulta
[Inyeccion_Fragmentos.MD](Inyeccion_Fragmentos.MD) para configurar la falla y
descargar los datos alterados con sus etiquetas. El cálculo de métricas
experimentales sigue siendo una etapa posterior.

## Continuar un CSV etiquetado

La carga de Revisión y etiquetado reconoce las seis columnas
`entity_id,state,last_changed,es_sintetica,tipo,tipo_falla`. Las etiquetas recibidas
se conservan en las filas indexadas en SQLite y se muestran en las gráficas; no
se crean cientos de miles de anotaciones de intervalo. Las etiquetas nuevas
prevalecen sobre las importadas. El indicador sintético anterior se conserva,
incluso al corregir manualmente una clase. Las nuevas inyecciones sustituyen
solo sus valores modificados. Puedes descargar nuevamente el CSV completo sin
hacer cambios adicionales.

Para inyectar, documenta un intervalo base como Normal y Revisado en el paso 4,
usa el generador del paso 5 y exporta el escenario. El CSV importado es la fuente
de esta nueva etapa: no permite reconstruir los valores previos a las antiguas
inyecciones. El respaldo conserva esa fuente y las intervenciones nuevas.
