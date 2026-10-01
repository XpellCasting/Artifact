# Artifact: filtros y diagnóstico GAT-LSTM v2

Aplicación local de la tesis: revisión de históricos, filtros estadístico y contextual,
diagnóstico GAT-LSTM v2 y evaluación de flujos causales. La reorganización conserva
las reglas de sensores binarios y multivalor. Los sensores continuos usan únicamente
FOCuS sobre residuos mediante `changepoint-online==1.2.1`; se retiraron los detectores
antiguos Hampel/CUSUM/DSPOT, la detección por intervalos y la alternativa EWMA.

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
    estadistico/        # FOCuS para continuos y reglas de binarios/multivalor
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

- Único método: `gat_lstm`; única arquitectura: `gat_lstm_v2_shared_type`.
- Se conserva `schema_version="gat-lstm-entities-v1"`: identifica el esquema de
  datos, no la arquitectura retirada.
- Conserva clases Normal, anomalía ambiental y falla del sensor, más la cabeza de
  tipo de falla y la política de abstención de v2.
- Los artefactos deben declarar v2 explícitamente. Los modelos sin arquitectura
  declarada no se reinterpretan como v2.
- Sin modelo v2 utilizable, la salida es indeterminada con
  `diagnosis_mode="unavailable"`; no hay fallback a patrones.
- Se retiraron `/diagnostics/pretrain` y su script. `pretrain_enabled=false` y los
  campos históricos de configuración se aceptan para cargar artefactos v2 existentes;
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
