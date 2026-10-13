# Artifact: filtros y diagnóstico GAT-LSTM por segmentos

Aplicación local de la tesis: revisión de históricos, filtros estadístico y contextual,
diagnóstico GAT-LSTM por segmentos y evaluación de flujos causales. Los sensores binarios
usan FOCuS Bernoulli sobre lecturas 0/1, con estimación basal y calibración separadas
temporalmente. Multiestados conserva sus reglas mientras se revisa su alternativa categórica.
Los sensores continuos usan únicamente
FOCuS sobre residuos mediante `focus-cpt==0.1.10`; se retiraron los detectores
antiguos Hampel/CUSUM/DSPOT, la detección por intervalos y la alternativa EWMA.

El predictor continuo aprende el cambio respecto de la mediana de las doce
lecturas anteriores. Las características de nivel se expresan respecto de esa
misma mediana y el pronóstico se devuelve en unidades originales. Esto evita
que el predictor quede limitado al rango absoluto de temperaturas visto en
Desarrollo; una lectura actual o futura nunca entra en su referencia. Los
residuos de normalización, calibración y evaluación usan el mismo pronóstico.
La nueva implementación invalida las referencias y cachés anteriores.

La decisión contextual continua calibra cada combinación de familias de
evidencia por cantidad de vecinos evaluables en cada familia. Una cobertura
sin suficientes muestras históricas queda sin conclusión; no se reutiliza el
umbral de otra cantidad de vecinos. Los mapas anteriores se reconstruyen.

La evaluación del flujo cuenta falsos positivos por lectura, incluidas las
repeticiones cuya alerta se suprime. El detalle estadístico por entidad muestra
por separado esos errores y las alertas emitidas sobre lecturas normales.
Las ejecuciones guardadas conservan sus resultados originales.

La referencia continua separa entrenamiento, normalización y calibración por
bloques temporales de 24 horas desde el primer evento de cada entidad, con unos
70/15/15 % del tiempo y al menos siete días en cada reserva. El fragmento final
se conserva con su duración real, el último valor y su fecha. La cobertura
excluye estados explícitamente inválidos; un silencio no prueba desconexión.
Los CSV de `last_changed` se interpretan como reportes por cambios. Las columnas
opcionales `reporting_mode` (`on_change`, `periodic`, `unknown`) y
`max_report_gap_seconds` permiten declarar otra política; los reportes periódicos
requieren un intervalo máximo para medir cobertura. La continuidad por cambios
supone un registro completo y no confirma conectividad.

La persistencia se compara con duraciones históricas por valor y hora UTC, tanto
en continuos como en binarios, y se comprueba al recibir una lectura. La interfaz
muestra la preparación y los motivos de exclusión antes de evaluar; distingue
lecturas evaluadas sin sospecha de resultados sin evaluación o sin conclusión.
La reconstrucción cambia la identidad de las referencias y sus cachés.

FOCuS reinicia la acumulación de los canales residuales tras detectar una
desviación, conservando las lecturas del predictor temporal. El reinicio ocurre
después del grupo completo de lecturas simultáneas. Guardar, restaurar o copiar
una ejecución activa reproduce los residuos posteriores al último reinicio.
Un estado inválido o una brecha superior al intervalo periódico declarado
abre una nueva secuencia. Calibración y evaluación comparten esas reglas y
agrupan de la misma manera las lecturas simultáneas. Los umbrales candidatos se
revalidan reproduciendo la calibración con reinicios tras las detecciones; si
un canal supera su presupuesto, se aumenta su umbral y se repite el recorrido.

Los canales puntual (`P`), cambio acumulado (`D`) y ruido (`N`) tienen umbrales
separados, con un presupuesto conjunto máximo de 1 % de lecturas marcadas y
un episodio por día en la partición de calibración. Cada canal recibe un tercio
de ambos límites. Los umbrales normalizados parten de 1, evitando rebajar la
referencia del percentil 99 a la mitad. Estos límites históricos no garantizan
la misma tasa en datos futuros. La tabla de cobertura muestra las tasas
alcanzadas durante la calibración.

`scripts/compare_focus_features.py` compara las características originales y
las que incluyen duraciones sobre las mismas entidades, observaciones y
particiones de 24 horas. Usa los mismos argumentos de archivos que el comando
de verificación siguiente, con `--output` apuntando a un archivo JSON. Las
etiquetas de evaluación se consultan después de decidir y no ajustan umbrales.

El experimento anterior del histórico disponible sobre los cinco sensores revisados,
con FOCuS sin reinicio tras alertas y los nuevos presupuestos por canal, produjo
64.874 falsos positivos con las características originales y 64.853 con las
características de duración. Estas últimas detectaron 20/20 lecturas `bias`,
4/5 `spike` y 2/3 `noise`. El límite de calibración no se sostuvo en la
evaluación futura. Se restauró por ello el reinicio tras detecciones; los
resultados anteriores no miden esta nueva configuración. El informe del
experimento sin reinicios está en `artifacts/daily_coverage/feature_comparison.json`.

Para repetir la comprobación sobre un histórico y etiquetas existentes, usar
`PYTHONPATH=backend python scripts/verify_daily_coverage.py --development <csv>
--evaluation <csv> --truth-database <sqlite> --output <directorio>`. Este comando
guarda referencias y un informe nuevos sin modificar la base de etiquetas.

Para verificar únicamente un sensor, ejecutar `PYTHONPATH=backend python
scripts/verify_false_positive_corrections.py --development <csv> --evaluation
<csv> --truth-database <sqlite> --entity sensor.weatherstation_inside_temp_2
--output <informe.json>`. El informe conserva el modelo de referencia, los
falsos positivos por lectura y por alerta emitida, y las detecciones de fallas
etiquetadas. No ajusta umbrales usando las etiquetas ni cambia ejecuciones.

## Procesamiento contextual

El recorrido contextual de eventos futuros compila las relaciones aceptadas
una vez por ejecución. Sus índices conservan el último reporte anterior a la
ventana y recortan únicamente los sensores cuyo historial vence, sin concatenar
todos los sensores en cada grupo temporal. Las lecturas ya normalizadas evitan
otra conversión y ordenación; las características continuas se reutilizan dentro
del grupo y su caché se descarta al avanzar. Las ejecuciones sin candidatas
contextuales omiten la preparación de sus observaciones.

`PYTHONPATH=backend python scripts/profile_contextual_replay.py --rows 1000
--output <archivo.json>` mide un flujo contextual independiente con el mapa
guardado y los primeros eventos del CSV de validación. No modifica etiquetas
ni ejecuciones guardadas. En la muestra disponible, el recorrido instrumentado
pasó de 223,9 a 30,5 segundos; los conteos de decisiones coincidieron. Estos
tiempos incluyen la sobrecarga del perfilador y no son una medición del CSV
completo. El script también guarda las funciones que consumieron más tiempo.

## Organización

```text
frontend/src/
  app/                  # Arranque y coordinación de la interfaz
  features/             # Históricos, revisión, inyecciones, escenarios,
                        # filtros estadístico/contextual, diagnóstico y flujos
  shared/               # Cliente HTTP, componentes y utilidades compartidas
backend/app/
  main.py               # Composición de FastAPI, middleware y ciclo de vida
  config.py             # Rutas y configuración del almacenamiento
  api/                  # Routers y contratos HTTP
  services/             # Históricos, reproducción, análisis y ciclo del modelo
  flow/                 # Definición, validación, ejecución y evaluación del grafo
  nodos/                # Comportamiento de los nodos
  modulos/
    estadistico/        # FOCuS continuo/Bernoulli binario; reglas multivalor
    contextual/         # Filtro contextual y calibración
    clasificador/       # Red, atención, características, entrenamiento e inferencia v2
  database/             # Repositorios SQLite y migración de almacenamiento
storage/
  database/             # review.sqlite3, diagnostics.sqlite3, flow_lab.sqlite3,
                        # validation_replay_results.sqlite3
  cache/                # Análisis, configuración continua y cachés regenerables
  jobs/                 # Progreso persistido del diagnóstico
  migrations/           # Inventario y registro de la migración local
models/gat_lstm/         # Pesos, metadatos, grafos, preprocesadores y modelo activo
data/
  defaults/             # Configuración inicial versionada
  history/              # Históricos y manifiesto de particiones
  review/               # Fuentes, anotaciones exportadas, inyecciones y escenarios
  gat_lstm/             # Manifiestos de datasets y secuencias numéricas
  fixtures/
  static_history/
scripts/                # Migración, empaquetado, entrenamiento y verificación
tests/backend/
tests/frontend/
tests/fixtures/
artifacts/gat_lstm/      # Paquete publicado en Git LFS, conservado
```

Cada funcionalidad conserva su propia base SQLite. Los datos locales, pesos y bases
no se versionan. `artifacts/gat_lstm/` sigue siendo el paquete compartible mediante
Git LFS; `models/` contiene los modelos de la instalación.

## Históricos del experimento actual

Las fuentes para los filtros están en `data/history/en_uso/`: `2mesesDesarrollo.csv`
(abril-mayo de 2026), `4mesesDesarrollo.csv` (febrero-mayo) y
`EvaluacionJunioAgosto.csv` (junio hasta la última lectura disponible del 25 de
agosto). Se unen todas las entidades de los seis originales conservados en
`../HistoricoPuro/SebaDatos6meses/`, sin selección de sensores. Ambos ajustes usan
la misma Evaluación futura. Los estados de revisión empiezan el 13 de mayo;
por eso el conjunto de 2 meses usa los dos últimos meses de Desarrollo.

`data/history/en_uso/ground_truth.csv` contiene 163.596 lecturas de los 14 sensores
seleccionados, de junio al 25 de agosto, etiquetadas por reglas y revisiones
compatibles. Tiene un mínimo de dos tipos de falla por sensor; las intervenciones
controladas se marcan con `es_sintetica=true`. Las etiquetas sobre datos reales
son estimaciones. `etiquetado_14_sensores/METODOLOGIA.md` y su resumen por sensor
documentan las etapas y los estados originales. Su `LEEME.md` detalla el protocolo;
`manifiesto.json` conserva hashes, cobertura y verificaciones, y `limpieza.json`
registra las particiones y exportaciones antiguas retiradas. Las fuentes internas
de anotaciones y modelos se conservan en `data/review`.

La aplicación carga el Desarrollo elegido en `data/history/cargados/history.csv`
y las lecturas futuras en `data/history/cargados/historico_validacion.csv`.
Estas son copias operativas; los dos CSV de Desarrollo de `en_uso` se conservan
para poder alternar entre experimentos sin sobrescribirlos.

## Migrar una instalación anterior

Detener la API y los trabajos antes de migrar. La primera orden muestra el inventario
sin modificar archivos; la segunda aplica la migración:

```powershell
docker compose stop
backend/.venv/Scripts/python.exe scripts/migrate_layout.py
backend/.venv/Scripts/python.exe scripts/migrate_layout.py --apply --offline
```

La migración verifica integridad SQLite y conteos, mueve los archivos dentro de
Artifact y registra el avance en `storage/migrations/layout-v2.json`. Puede reanudarse
tras un fallo y no sobrescribe destinos incompatibles. Conserva IDs, datos fuente,
anotaciones, revisiones y los modelos v2. Verifica los hashes de sus pesos antes de
retirar artefactos antiguos.

Se eliminan los pesos, configuraciones, referencias generadas y resultados exclusivos
de prototipos robustos y GAT-LSTM v1. Los flujos guardados permanecen; una referencia
a un modelo retirado requiere seleccionar un modelo v2 antes de ejecutar. No se
convierten pesos v1 en v2 ni se reentrena el modelo activo. Los cachés de preparación
que dependían de las rutas antiguas se reconstruyen al cargar Desarrollo.

## Ejecutar con Docker

```powershell
git lfs pull
docker compose up --build
```

Interfaz: http://localhost:5173. API y documentación: http://localhost:8000/docs.
Los volúmenes `data/`, `storage/` y `models/` conservan el estado entre reinicios.
El bootstrap restaura los archivos ausentes del paquete publicado y conserva la
selección de modelo activo existente. No inicia un entrenamiento GAT-LSTM.

## Ejecutar localmente

Requiere Python 3.12 y Node.js compatible con la versión de Vite del lockfile.
Desde Artifact:

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt -r backend/requirements-gat.txt
$env:PYTHONPATH = "$pwd/backend"
backend/.venv/Scripts/python.exe -m app.bootstrap_gat --serve
```

En otra terminal:

```powershell
cd frontend
npm ci
npm run dev
```

`ARTIFACT_ROOT` cambia la raíz completa de los datos; por defecto se deriva una sola
vez de `app/config.py`. También se admiten `DATA_DIR`, `DATABASE_DIR`, `MODELS_DIR`,
`REVIEW_DATA_DIR`, `HISTORICAL_DATA_DIR`, `CACHE_DIR`, `JOBS_DIR` y las rutas específicas
que aparecen en Docker Compose. Un directorio de revisión explícito distinto del
principal crea un espacio aislado, útil para pruebas.

## Contrato del clasificador

- Único método: `gat_lstm`; única arquitectura: `gat_lstm_v3_segments_multiscale`.
- Se usa `schema_version="gat-lstm-segments-v1"`: identifica el esquema de
  datos, no la arquitectura retirada.
- Conserva clases Normal, anomalía ambiental y falla del sensor, más la cabeza de
  tipo de falla y la política de abstención.
- Los artefactos deben declarar la arquitectura de segmentos explícitamente. Los modelos sin arquitectura
  declarada no se reinterpretan como modelos de segmentos.
- Sin modelo de segmentos utilizable, la salida es indeterminada con
  `diagnosis_mode="unavailable"`; no hay fallback a patrones.
- Se retiraron `/diagnostics/pretrain` y su script. `pretrain_enabled=false` y los
  campos históricos de configuración se aceptan por compatibilidad del formato de configuración;
  activar preentrenamiento se rechaza.
- Se mantienen las rutas de las funcionalidades conservadas. Entrenar requiere
  Desarrollo y el soporte supervisado existente, incluido el tipo de falla. La política
  previa que reserva Validación/Prueba para eventos futuros permanece.

## Verificación

```powershell
backend/.venv/Scripts/python.exe scripts/run_tests.py
npm --prefix frontend run build
docker compose config --quiet
```

Las pruebas incluyen causalidad y calibración contextual, una referencia numérica del
filtro estadístico obtenida del código anterior, entrenamiento y recarga de v2,
rechazo de modelos retirados, ausencia de fallback, migración repetible, conflictos
de destino y conservación de anotaciones. Las pruebas de interfaz viven fuera del
bundle y usan `frontend/test-runtime.js` para resolver React, Vite y el DOM de pruebas.

`scripts/package_gat_lstm.py` empaqueta el modelo activo con sus secuencias desde la
nueva distribución y conserva el formato del paquete publicado. El resto de scripts
de entrenamiento, auditoría y evaluación utiliza los mismos servicios que la API.

La comprobación HTTP integrada se puede repetir con
`python scripts/smoke_api.py --base http://127.0.0.1:18000 --disposable`.
Debe apuntar a una API desechable con un modelo v2 activo: carga históricos sintéticos
y guarda un flujo de prueba. El resumen de la validación de esta migración está en
[REFACTOR_VALIDATION.md](REFACTOR_VALIDATION.md).
