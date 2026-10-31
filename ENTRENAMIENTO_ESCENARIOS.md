# Escenarios sintéticos y evidencia opcional — entrega del 29 de octubre de 2026

Los datos y las secuencias de febrero–abril están preparados en
`training/escenarios_20261029_v1/`.
No se inició el entrenamiento del clasificador ni se activó un modelo. Los pesos,
las comparaciones científicas y el candidato exportable se obtendrán cuando
ejecutes los comandos de entrenamiento y evaluación.

La implementación sigue el plan revisado cuando difiere del plan base.

La entrega contiene **25.181 ventanas de entrenamiento, 7.984 de selección y
2.123 de robustez de abril**. La cobertura supervisada de las tres clases y los
siete tipos está comprobada para C/D. Los umbrales de publicación y el recall
solo se validarán después de tu entrenamiento.

## Datos entregados

Se verificaron 1.200.815 observaciones del CSV de seis columnas
`experimento_v2/4mesesDesarrollo_15etiquetado.csv` contra su auditoría: ninguna
quedó sin correspondencia. Esta base es anterior a los CSV de `inyecciones/`;
no se reconstruyeron ni se inventaron valores anteriores a una inyección.

| Partición UTC, fin exclusivo | Escenarios principales | Episodios de falla | Episodios ambientales |
|---|---:|---:|---:|
| Febrero–marzo, entrenamiento | 5 | 280 | 10 |
| Abril, selección | 2 | 70 | 4 |
| Mayo, prueba temporal | 2 | 70 | 2 |
| Robustez de abril | 1 | 35 | 2 |
| Robustez de mayo | 1 | 34 | 1 |

Total: **508 episodios derivados**. Las cuotas principales de los siete tipos
se cumplieron. Se documentan cuatro episodios faltantes: uno ambiental en cada
escenario principal de mayo, y uno ambiental y uno de ruido en robustez de mayo.
No se duplicaron episodios para ocultar esos déficits.

El componente compacto ocupa aproximadamente 331 MiB sin comprimir: una sola base
SQLite y once archivos de intervenciones. Las secuencias NPY se entregan por
separado dentro de esa carpeta; su tamaño y cobertura están en
`preparation_report.json`. Los sensores mantienen sus IDs reales.
`observations` conserva lecturas y filas originales; `labels` conserva las
etiquetas, origen y certeza; `label_evidence`/`evidence_sources` conservan los
vínculos de auditoría. Las variantes contienen únicamente valores cambiados.
La evidencia de filtros se genera como componente separado al preparar las
secuencias, con los filtros reales sobre los valores intervenidos.

Los archivos `manifest.json`, `config.json`, `preprocessor.json`, `graph.json`
y `report.json` guardan hashes, configuración y límites. La normalización,
histogramas, vocabularios y segmentos se ajustaron una sola vez sobre la
normalidad elegible de febrero–marzo, antes del aumento. Abril y mayo no
participan en ese ajuste. Las entidades sin referencia de entrenamiento no
incorporan vocabularios de mayo: se reportan como no evaluables.

Las nuevas inyecciones requieren 72 horas de contexto normal etiquetado dentro
de su partición. Se excluyen estados inválidos, etiquetas desconocidas e
intervenciones sin efecto. Se separan duración programada y filas efectivamente
alteradas. No existe una regla causal verificada para fechar la identificabilidad
del subtipo; se conserva `identifiable_from=null` y se mide la demora desde la
intervención. La recuperación de un pico se usa para elegir el escenario, sin
introducir esa lectura futura en la entrada de una predicción anterior.

## Usar tu entorno Ubuntu/ROCm existente

Desde PowerShell entra en tu distribución:

```powershell
wsl -d Ubuntu-24.04
```

Después, dentro de Ubuntu, extrae esta entrega en un directorio nuevo del disco
Linux y reutiliza el entorno que indicaste:

```bash
mkdir -p ~/tesis-entrenamiento/escenarios_20261029
tar -xzf /mnt/c/Users/BlindWizard/Desktop/Tesis/outputs/Escenarios_entrenamiento_20261029_final.tar.gz \
  -C ~/tesis-entrenamiento/escenarios_20261029
source ~/tesis-entrenamiento/Artifact/.venv-rocm/bin/activate
cd ~/tesis-entrenamiento/escenarios_20261029/Artifact
```

La ruta de activación depende de tu instalación. Los scripts reutilizan el
Python ya activo. También puedes indicar la ruta mediante `ROCM_VENV`; no
instalan paquetes ni reemplazan PyTorch.

```bash
python -c 'import torch; print(torch.__version__, torch.version.hip); assert torch.version.hip and torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -c 'import torch_geometric, pandas, sklearn, scipy, networkx, dtaidistance, focus_cpt; print("Dependencias de preparación disponibles")'
python scripts/escenarios_experimento.py verify
```

Para reproducir las evidencias se necesitan también las dependencias de los
filtros declaradas en `backend/requirements.txt`. Ese archivo es una referencia
de versiones; no ejecutes una reinstalación de PyTorch CPU/CUDA sobre tu entorno
ROCm. El piloto se verificó con tu entorno
`~/tesis-entrenamiento/Artifact/.venv-rocm` de Ubuntu-24.04, con PyTorch
`2.9.1+rocm7.2.0`. En Windows, `focus-cpt` requiere Qhull para compilar; no es
necesario instalarlo allí para utilizar esta entrega en Ubuntu.

## Preparación sin entrenamiento

El piloto de **preparación** ya está incluido y verificado. Este comando
comprueba sus archivos existentes, sin ajustar el clasificador:

```bash
bash preparar_escenarios.sh --pilot 2>&1 | tee piloto_preparacion.log
```

Selecciona hasta dos objetivos por clase/tipo en el primer escenario de entrenamiento. Los filtros
auxiliares se calibran únicamente con el 1–14 de febrero y se reproducen
cronológicamente con estados separados por escenario. Ese período no aporta
objetivos de entrenamiento ni comparación. Un filtro sin referencia suficiente
queda no evaluable, con su máscara correspondiente. El piloto conserva los
datos completos de contexto: puede requerir tiempo de CPU aunque tenga pocos
objetivos.

El piloto real produjo 18 ventanas en unos 136 segundos: dos por cada tipo de
falla, dos normales y dos ambientales. Hubo evidencia estadística utilizable en
4 ventanas, contextual en 8 y ambas simultáneamente en 1. Las otras situaciones
conservaron sus máscaras de ausencia o no evaluabilidad; no se inventaron
referencias para completar las modalidades.

Para comprobar las secuencias completas entregadas, o materializarlas si
usas solamente el formato compacto:

```bash
bash preparar_escenarios.sh --sequences 2>&1 | tee preparacion_completa.log
```

El comando publica `sequences/` al terminar. Informa el número de ventanas y
el espacio estimado antes de materializar cada partición, comprueba espacio
libre y procesa un escenario cada vez. Requiere un destino de secuencias nuevo;
un resultado incompleto no se marca como utilizable. Si ya existe una salida
completa, verifica sus hashes y la reutiliza. No lee mayo para ajustar
preprocesadores ni seleccionar políticas. El piloto se guarda por separado y
no puede usarse como dataset completo.

No es necesario copiar de nuevo el CSV y la auditoría para estos comandos: los
datos preparados ya están incluidos. Para regenerarlos desde los originales:

```bash
python scripts/escenarios_experimento.py prepare \
  --source /ruta/experimento_v2/4mesesDesarrollo_15etiquetado.csv \
  --audit /ruta/experimento_v2/4mesesDesarrollo_auditoria.csv \
  --config configs/escenarios_20261029.json \
  --output training/otra_preparacion
```

## Iniciar y reanudar el entrenamiento

Estos comandos **sí entrenan**; quedan a tu cargo:

```bash
set -o pipefail
bash entrenar_experimento.sh --freeze 2>&1 | tee entrenamiento_escenarios.log
```

Ejecuta secuencialmente A, B, C y D con las semillas 20261029, 20261030 y
20261031. A usa la base y selección por macro-F1 de clase; B usa la base y
selección por subtipos; C incorpora las variantes; D carga y congela C de la
misma semilla y ajusta solamente la rama residual de evidencia. El backbone
queda en evaluación durante toda la etapa D. Las cuatro disponibilidades se
ocultan de forma equilibrada y reproducible. Sin evidencia utilizable la
corrección es exactamente cero; se conserva también la política autónoma de C.

Para reanudar tras una interrupción:

```bash
bash entrenar_experimento.sh --resume --freeze 2>&1 | tee -a entrenamiento_escenarios.log
```

Los checkpoints se guardan por época. La identidad incluye entradas,
secuencias, semilla, configuración, código y dependencias. Las ejecuciones
terminadas se reutilizan con `--resume`; una incompatibilidad se rechaza.
`runs/active.lock` impide dos entrenamientos a la vez. Si el proceso fue terminado
con `kill -9`, comprueba manualmente que su PID ya no existe antes de retirar
ese archivo concreto. No borres carpetas para forzar una reanudación.

Para ejecutar solo una comparación predefinida sin congelar todavía:

```bash
bash entrenar_experimento.sh --experiments C D --seeds 20261029
```

Después completa las doce ejecuciones con `--resume --freeze`. La congelación
requiere A–D con las tres semillas. Una vez creado `freeze.json` no se permite
continuar seleccionando configuraciones en esa entrega.

Febrero–marzo contiene 138.658 normales, 1.063 fallas históricas estimadas y
456.305 desconocidos, pero ninguna anomalía ambiental histórica etiquetada.
A y B declaran esa clase ausente; no se trasladan ejemplos de abril para
completarla. C y D requieren cobertura supervisada de las tres clases y los
siete tipos después de construir las ventanas, o fallan con un diagnóstico.

## Prueba reservada, exportación y limpieza

Solo después de congelar los doce candidatos, prepara mayo y evalúa:

```bash
bash preparar_escenarios.sh --test 2>&1 | tee preparacion_mayo.log
bash evaluar_experimento.sh 2>&1 | tee evaluacion_mayo.log
```

La evaluación comprueba los hashes congelados y utiliza los mismos objetivos
para todos los modelos. D se evalúa en las cuatro modalidades, usando la
política de abril correspondiente a la evidencia realmente disponible. Guarda
predicciones internas/publicadas, matrices, cobertura, desgloses, demoras por
episodio, variación entre semillas y comparación por bootstrap agrupado. Las
réplicas y semillas permanecen juntas dentro de cada contexto original. Si
solo queda un grupo, el intervalo se declara no estimable. La ejecución del
grafo de compuertas usa el `FlowRunner` de producción con las salidas causales
precalculadas de los filtros y cuenta las fallas que nunca llegan al clasificador.

El reporte de falsas alertas usa exclusivamente objetivos normales de la base,
sin aumento. Está identificado como tasa de objetivos muestreados por día de
sensor; con el stride horario no representa alertas sobre todas las lecturas.
No se convierten todas las filas de un episodio en aciertos por detectar una.

Un tipo solo se publica con tres contextos originales de entrenamiento, tres
de abril y recall de al menos 0,5. El conteo de soporte usa ventanas originales
de 72 horas que no se solapan, sin multiplicarlo por copias. La ponderación y
el bootstrap conservan además los componentes de contextos solapados. Una falla
puede quedar sin tipo aunque exista un candidato interno. Los estados ausente,
negativo, positivo, omitido, error y no evaluable tienen canales distintos.

Exporta una ejecución explícita, sin cambiar el modelo activo:

```bash
bash exportar_candidato.sh --run D-20261029 --output candidato_D_20261029.tar.gz
```

El paquete incluye pesos, preprocesador, grafo, manifiestos y contrato de
evidencias. Se verifica la carga en CPU antes de exportar. Es un directorio
portable de candidato; no es una importación ni una activación automática.

Para inventariar checkpoints eliminables:

```bash
bash limpiar_entrenamientos.sh
bash limpiar_entrenamientos.sh --run A-20261029
bash limpiar_entrenamientos.sh --run A-20261029 --apply
```

Solo se eliminan checkpoints de ejecuciones identificadas y terminadas, dentro
de `runs/`. Datos, auditorías, entorno, modelos candidatos, resultados y
manifiestos quedan protegidos. No se borran genéricamente `models`, `data` ni
`storage`. Las ejecuciones antiguas ajenas a este contrato quedan fuera de esta
limpieza; no se infiere que sean eliminables.

## Límites e integridad

Las etiquetas heredadas son estimadas y sus exportaciones pudieron usar contexto
posterior a los cortes. Esa dependencia se registra y sus resultados se separan
de los sintéticos. Las intensidades térmicas están en unidades nativas del
escenario predefinido; no certifican equivalencia física de los dispositivos.
Los estados `NULL` siguen siendo desconocidos. Mayo es una evaluación temporal
reservada para este experimento y pudo utilizarse en trabajos anteriores.

Se validaron contratos de evidencia, generación reproducible de siete tipos,
aislamiento de contextos, salida autónoma idéntica sin evidencia, carga CPU de
la arquitectura nueva y compatibilidad de los componentes anteriores. Los
resultados de entrenamiento, la ejecución de la red en GPU y la mejora científica
no se presentan como obtenidos en esta entrega.
