# Revisión y etiquetado de fragmentos reales

Esta herramienta prepara evidencia para la base de verdad de la tesis. Una
anotación registra el juicio del revisor y su fundamento. El software conserva
esa declaración; no confirma por sí mismo la causa de un evento.

## Procedimiento

1. Abre **Revisión y etiquetado** e importa el CSV indicando su partición real:
   Desarrollo, Validación o Prueba. Usa las fechas efectivas del archivo. La
   importación para revisión no entrena los filtros ni cambia la base de
   normalidad del módulo Estadístico.
2. Elige la entidad objetivo. Añade otras entidades que puedan aportar
   contexto según su ubicación, magnitud y función física. La selección es
   manual y no utiliza las predicciones del detector como etiquetas.
3. Define el inicio y fin del episodio en UTC. Incluye un margen de
   observación antes y después para examinar el comienzo, la persistencia y la
   recuperación. El margen es contexto; la etiqueta solo afecta al intervalo
   central de la entidad objetivo.
4. Revisa la curva y los estados originales. Compara las entidades por tiempo,
   recordando que pueden tener unidades, cadencias y desfases distintos.
   Consulta la tabla de filas para inspeccionar valores concretos. En períodos
   densos el gráfico presenta una reducción de puntos; acorta la ventana para
   revisar detalles. La exportación conserva todas las filas seleccionadas.
5. Busca evidencia externa a la interpretación de la curva: una acción
   documentada, un mantenimiento, una medición independiente o un registro
   técnico que explique el evento. Escribe qué observaste, qué respalda la
   interpretación y qué dudas permanecen.
6. Guarda la clase propuesta y el estado de revisión. Usa **Incierto** cuando
   falte evidencia y **Borrador** mientras el caso esté en estudio. Para marcar
   **Verificado**, identifica al revisor y registra evidencia independiente
   concreta. El tipo de evidencia por sí solo no prueba una causa.
7. Descarga el ZIP de la anotación. Puedes abrir y corregir anotaciones
   anteriores; cada edición incrementa su revisión. Un ZIP descargado refleja
   la versión de la anotación que se exportó.

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

Las anotaciones basadas únicamente en revisión visual permanecen como
borradores. `evaluable` identifica las anotaciones declaradas verificadas de
clase conocida; no constituye una certificación automática de su veracidad.
La revisión del protocolo de la tesis debe decidir qué evidencia admite en
las métricas principales y conservar separadamente los casos inciertos.

## Contenido de la exportación

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
descripción de la revisión; permite los borradores visuales como casos
exploratorios y conserva esa limitación en la exportación. Consulta
[Inyeccion_Fragmentos.MD](Inyeccion_Fragmentos.MD) para configurar la falla y
descargar los datos alterados con sus etiquetas. El cálculo de métricas
experimentales sigue siendo una etapa posterior.
