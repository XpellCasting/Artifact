# Aplicacion de diagnostico de sensores

Aplicacion local con API FastAPI, interfaz React y filtros estadistico,
contextual y diagnostico GAT-LSTM. Requiere Docker con Docker Compose.

## Iniciar

Desde la carpeta `Artifact` (la raiz de este repositorio):

```powershell
docker compose up --build
```

- Aplicacion: http://localhost:5173
- API y estado: http://localhost:8000/health
- Documentacion de la API: http://localhost:8000/docs

Antes del primer inicio, instalar Git LFS y ejecutar `git lfs pull` desde este
repositorio para descargar `artifacts/gat_lstm/snapshot.tar.gz`.
Docker restaura automaticamente el dataset y el modelo GAT-LSTM activo en
`data/review/diagnostics/`; no necesita reconstruir el dataset ni reentrenar.
El primer inicio necesita aproximadamente 6 GiB adicionales para descomprimirlo.

Docker instala las dependencias y PyTorch CPU. La aplicacion puede iniciar sin
historicos; carga un CSV desde la interfaz para comenzar el analisis.

## Carpetas

- `backend/app/`: codigo de la API y filtros; `backend/requirements*.txt`: dependencias.
- `frontend/src/`: interfaz y recursos utilizados; `frontend/package-lock.json`: dependencias fijadas.
- `HistoricosEnUso/`: historicos locales, excluidos de Git.
- `data/`: caches, revisiones y archivos generados locales, excluidos de Git.
- `artifacts/gat_lstm/`: snapshot versionado con Git LFS, con dataset, pesos y configuracion del modelo activo.
- `backend/tests/` y `scripts/`: pruebas y utilidades del codigo.
- `docs/` y `local/`: documentacion de investigacion y recursos auxiliares conservados localmente, excluidos de Git.

Los Dockerfile y `.dockerignore` limitan cada contexto de construccion al codigo
y sus dependencias. No se incorporan historicos, caches ni entornos virtuales.

## Historicos locales

Para usar las particiones existentes, copiar en `HistoricosEnUso/`:

- `historico_desarrollo.csv`
- `historico_validacion.csv`
- `historico_prueba.csv`
- `historico_reserva_agosto.csv`
- `particiones_seleccionadas_resumen.json`

Los CSV etiquetados y sus resumenes tambien se conservan en esa carpeta.
El manifiesto conserva la procedencia original; la aplicacion resuelve los
archivos trasladados por su nombre dentro de `HistoricosEnUso`.

`HistoricosEnUso/cargados/history.csv` conserva el historico base cargado desde
la interfaz; `HistoricosEnUso/cargados/historico_validacion.csv` conserva la
carga para reproduccion. Son copias de trabajo modificables por la aplicacion.
Docker monta las particiones como solo lectura y `cargados/` como escritura.
El historico base se recupera en segundo plano al reiniciar; una reproduccion
en curso debe iniciarse nuevamente.

Un clon nuevo incluye el dataset y el modelo GAT-LSTM del snapshot mediante
Git LFS. No incluye historicos originales, anotaciones de revision ni caches.
Para continuar un trabajo existente se deben transferir por separado
`HistoricosEnUso/` y `data/`. Docker crea las carpetas necesarias al iniciar.
Los respaldos y originales anteriores permanecen fuera del repositorio,
en `../HistoricoPuro/`. La guia extensa local esta en `docs/Guia_Completa.md`.

## Desarrollo local

La API usa las mismas carpetas de datos por defecto. Las rutas se pueden
sobrescribir con `REAL_HISTORY_PATH`, `VALIDATION_HISTORY_PATH` y
`HISTORICAL_DATA_DIR`. Las pruebas y la compilacion web se ejecutan con:

```powershell
cd backend
.venv/Scripts/python.exe -m unittest discover -s tests -v
cd ../frontend
npm ci
npm run build
```

### Paralelismo del histórico

El servicio `api` configura `HISTORY_TRAINING_WORKERS=4`: entrena hasta cuatro
sensores continuos simultáneamente, con un hilo numérico por proceso hijo.
La cola mantiene como máximo un sensor por proceso para acotar las copias en
memoria. La interfaz muestra los sensores en curso y los completados.
Con `HISTORY_TRAINING_WORKERS=1` se usa ejecución secuencial (también es el valor
predeterminado fuera de Compose).

`CONTEXT_CALIBRATION_WORKERS=4` calibra las relaciones continuas del mapa en
hasta cuatro procesos, con un hilo numérico por proceso. Las relaciones del
mismo sensor de origen se agrupan para reutilizar características. Solo se envían
los sensores necesarios a cada proceso, con un máximo de cuatro grupos en curso.
La extracción usa bloques de 2.048 consultas y una caché de características de
hasta 64 MiB por grupo. La calibración de decisiones también usa características
por bloques, manteniendo su ejecución secuencial. La búsqueda inicial de pares
sigue siendo secuencial.

El progreso contextual muestra relaciones completadas, partición de ajuste o
tolerancia y eventos procesados. `CONTEXT_CALIBRATION_WORKERS=1` permite calibrar
sin procesos hijos (valor predeterminado fuera de Compose).

Los cambios requieren reconstruir y recrear el API; una carga que ya está en
curso conserva el código anterior. Cambiar la implementación invalida las cachés
contextual y causal para evitar reutilizar resultados de otra versión.

Medición local de la calibración presión relativa → temperatura interior:
111.274 y 73.594 registros respectivamente, ventana de 300 s y desfase de 0 s.
En una ejecución aislada por versión, con un hilo numérico, la implementación
escalar tardó 210,0 s y la extracción por bloques 6,4 s (32,8×). Los modelos,
umbrales y cantidades de muestras coincidieron exactamente; el pico de RSS
observado fue de unos 167 MiB por proceso en ambas ejecuciones. Esta medición
corresponde únicamente a esa relación y no estima la duración del histórico
completo ni la memoria conjunta de cuatro procesos. Las pruebas de regresión
están en `backend/tests/test_contextual_calibration.py`, con la implementación
escalar de referencia en `backend/tests/contextual_reference.py`.
