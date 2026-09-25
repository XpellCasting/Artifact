# Avance GAT-LSTM por entidad

## Corrección del consumo al entrenar candidato — 24 de septiembre de 2026

La primera corrección todavía conservaba mapas de memoria de los NPY durante todo el entrenamiento. Con el dataset `8c1e27ef008b481e85be38a7ec2d8b80` (68.258 ventanas de ajuste y 17.635 de desarrollo), la lectura aleatoria hacía residentes las páginas de esos archivos. Una reproducción acotada pasó de 628,3 MiB a 1.600,1 MiB de RSS; se detuvo al alcanzar ese umbral. La memoria de archivos residentes creció de 75,4 a 1.043,9 MiB.

`FileArray` ahora lee cada ventana por su desplazamiento en el archivo y libera inmediatamente el mapa usado para inspeccionar la cabecera. Solo etiquetas y metadatos pequeños permanecen cargados. Los conteos de episodios, soporte, validaciones temporales y procedencia acceden a resúmenes sin leer características. Se conservan los valores, el tamaño del lote de 16, las particiones y el formato existente: no hace falta reconstruir el dataset.

Verificación sobre el dataset real, montado en solo lectura: 512 lotes de entrenamiento (8.192 ventanas) con gradientes y Adam, seguidos de 1.024 ventanas de evaluación de desarrollo, en un contenedor aislado con límite de 3 GiB y dos CPU. El RSS máximo fue 797,1 MiB y terminó en 783,7 MiB; desde el lote 65 se mantuvo aproximadamente entre 775 y 794 MiB en los puntos medidos. Duración: 93,9 segundos. Es una prueba parcial del entrenamiento, no las 30 épocas ni una medición de toda la aplicación o de la VM Docker. No se publicó ni activó el modelo de prueba. El script reproducible está en `../tmp/gat_memory_check/check.py`.

Las 34 pruebas GAT-LSTM pasan, incluidas las comprobaciones de que los resúmenes no leen las características y las ventanas devueltas no son mapas de memoria. Docker Compose valida y limita el backend a 4 GiB, sin swap adicional. Se reconstruyó la imagen, se inició la aplicación y `/health` confirmó que la API está operativa. El límite efectivo del contenedor se verificó con Docker Inspect.

## Corrección de memoria — 24 de septiembre de 2026

La preparación acumulaba todas las ventanas como listas de Python antes de escribir los NPY; el cargador de entrenamiento también convertía la partición completa a listas pese a abrirla con `mmap`. Esto podía agotar la RAM durante el tramo 35–65 % de preparación.

Ahora `GAT_LSTM/storage.py` escribe cada ventana de forma incremental, genera los mismos NPY y JSONL sin reunir los tensores completos, y carga las particiones bajo demanda. Solo se copia el lote actual a tensores. La corrección incluye ajuste supervisado y preentrenamiento; conserva etiquetas, particiones, características y formato de los datasets existentes. Los eventos originales, metadatos e índices de episodios siguen en memoria. La escritura usa archivos temporales en disco, eliminados al finalizar o fallar la construcción.

Verificación: 34 pruebas GAT-LSTM aprobadas, incluidas equivalencia de arrays, acceso por lotes, entrenamiento integrado y una regresión de memoria. En una prueba sintética de 128 ventanas de 60 pasos × 32 nodos × 19 características, `tracemalloc` midió un pico de 162,92 MiB acumulando listas y 2,04 MiB escribiendo incrementalmente; abrir la partición usó 0,10 MiB y copiar un lote de 16 ventanas, 2,34 MiB. Estas cifras miden asignaciones de Python, no la RAM total del proceso ni del histórico completo. No se ejecutó el entrenamiento completo del CSV del usuario.

La imagen local `artifact-api` se reconstruyó con la corrección. Para iniciar la aplicación desde `Artifact`, ejecutar `docker compose up -d` y volver a preparar el conjunto que quedó interrumpido. Los datasets ya completados son compatibles con el nuevo cargador.

Fecha de verificación: 23 de septiembre de 2026.

## Estado

La implementación GAT-LSTM está conectada al flujo de preparación del investigador, a la API de diagnósticos y al despacho de inferencia de F3. El motor convencional sigue disponible. La selección y activación de un modelo GAT-LSTM son acciones explícitas; entrenar o evaluar no activa candidatos. La interfaz Docker muestra el selector **GAT-LSTM por entidad**, configuración, fuentes canónicas y las acciones de preparación.

El clasificador produce tres clases (`normal`, `environmental_anomaly`, `sensor_fault`) y puede abstenerse (`indeterminate`). Las etiquetas de subtipo se conservan como evidencia. Las puntuaciones softmax no están calibradas y no son probabilidades físicas.

## Componentes terminados

- `backend/app/GAT_LSTM/` incluye configuración validada, auditoría, ingestión causal, características, grafo F2, GAT, LSTM, entrenamiento supervisado, preentrenamiento predictivo opcional, inferencia, evaluación, persistencia de datos y comparación de predicciones.
- `scripts/gat_lstm_*.py` ofrece auditoría, preentrenamiento, entrenamiento, evaluación y comparación reproducibles.
- `Diagnostico_Trabajo.py` expone operaciones de estado, configuración, dataset, preentrenamiento, entrenamiento, evaluación y activación. F3 puede usar el modelo activo y conserva el comportamiento convencional y del catálogo cuando no hay candidato activo.
- La preparación está integrada en **Clasificador diagnóstico → Preparación del investigador**. Permite ajustar y guardar el motor y parámetros, elegir fuentes y anotaciones, preparar un dataset persistido, revisar soporte y lanzar operaciones cuando los datos lo permiten.
- Los datasets registran hashes, identidad de fila, fuente, partición y revisión. Las transformaciones y el mapa F2 se ajustan solo con `train_fit`; `dev_tune` selecciona y ajusta abstención. Validación y prueba solo se usan para evaluación.
- La comparación fija el mismo conjunto objetivo para todos los modelos. Predicciones ausentes permanecen como abstenciones; el informe separa métricas por origen y admite intervalos bootstrap por episodio.

## Auditoría del histórico

El artefacto completo está en `data/review/diagnostics/experiments/gat_lstm_p0_audit.json`. Se verificaron hashes y conteos del manifiesto canónico para Desarrollo (240.751 filas), Validación (86.059), Prueba (77.496) y Reserva (28.333). Se corrigió la resolución dentro del contenedor de las rutas Windows del manifiesto; Docker monta `HistoricoPuro` como solo lectura en `/app/HistoricoPuro`.

El soporte elegible observado fue:

| Partición | Normal | Anomalía ambiental | Falla del sensor |
|---|---:|---:|---:|
| `train_fit` | 0 | 0 | 0 |
| `dev_tune` | 0 | 0 | 0 |
| `validation` | 220 | 123 | 7 |
| `test` | 0 | 0 | 0 |

Por lo tanto, el histórico actual no permite entrenar las tres clases ni completar una evaluación independiente de prueba. Las etiquetas existentes no se reasignaron ni se inventaron. La auditoría encontró siete problemas de fuentes importadas: cinco copias físicas ausentes y dos discrepancias entre partición declarada y fechas canónicas. La preparación verifica los archivos canónicos mediante el manifiesto y no usa esas copias rotas como sustituto.

La aptitud para preentrenamiento predictivo permanece indeterminada: la cobertura de objetivos futuros aún no está auditada. Este modo es opcional y no está activado por defecto. No se entrenó ni activó ningún modelo con los datos reales. Las pruebas de aprendizaje usan fixtures sintéticos identificados como pruebas funcionales.

## Docker y verificación

`docker-compose.yml` habilita GAT con `INSTALL_GAT=1`, instala PyTorch CPU-only (`2.14.0+cpu`) y monta `HistoricoPuro` en solo lectura. El estado de la API en Docker informó `gat_lstm=true`; los endpoints de salud y fuentes respondieron correctamente. La web servida por el contenedor se abrió en `http://localhost:5173/`, y se verificaron visualmente el clasificador, el selector GAT y su configuración. La inspección no construyó un dataset ni modificó la base de revisión.

Verificación final:

- Backend: 28 pruebas aprobadas.
- Frontend: 45 pruebas aprobadas.
- `npm run build`: completó; Vite emitió únicamente el aviso de que el chunk principal supera 500 kB.
- `docker compose config`: válido.
- Imágenes Docker API y web: compiladas; el contenedor API importó el módulo y confirmó PyTorch CPU sin CUDA.
- `git diff --check`: limpio.

Las pruebas usan almacenamiento temporal. Docker montó el histórico de forma de solo lectura y aisló `REVIEW_DATA_DIR` en `/tmp/review`, sin ejecutar acciones de preparación ni escribir en los datos de producción.

## Uso

Desde la raíz del proyecto, la auditoría local se ejecuta con:

```powershell
backend/.venv/Scripts/python.exe scripts/gat_lstm_audit.py --output data/review/diagnostics/experiments/gat_lstm_p0_audit.json
```

La aplicación del usuario se ejecuta con Docker Compose:

```powershell
docker compose up --build
```

Las pruebas backend se ejecutan desde `backend`:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Las pruebas web se ejecutan desde `frontend`:

```powershell
$files = @(Get-ChildItem -LiteralPath './src' -Filter '*.test.js' | ForEach-Object { $_.FullName })
node --test $files
npm run build
```

## Trabajo que depende de nuevas etiquetas

Para entrenar y evaluar un candidato real hacen falta anotaciones revisadas con soporte causal suficiente en Desarrollo (`train_fit` y `dev_tune`) y soporte independiente en Prueba. Una vez incorporadas, la app permite crear dataset, entrenar, evaluar y comparar B0/B1/B2/B3 sobre las mismas muestras objetivo. La comparación empírica queda pendiente hasta que existan esos artefactos de predicción y etiquetas elegibles.
