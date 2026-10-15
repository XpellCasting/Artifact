# Artifact: filtros y diagnóstico GAT-LSTM por segmentos

## Home Assistant (HACS)

La integración de HACS conecta Home Assistant con Artifact y permite abrir su
interfaz desde la página del dispositivo. Artifact se ejecuta por separado en
Docker; HACS instala únicamente `custom_components/artifact`, sin copiar los
modelos, históricos, frontend ni backend dentro de Home Assistant.

1. En el equipo que ejecutará Artifact, iniciar `docker compose up -d --build`.
2. Comprobar desde otro equipo de la red las direcciones
   `http://100.72.168.116:8000/health` y `http://100.72.168.116:5173`
   mediante Tailscale.
   La primera debe devolver el estado de la API y la segunda abrir Artifact.
3. En HACS, abrir **⋮ → Repositorios personalizados**, añadir
   `https://github.com/XpellCasting/Artifact` con el tipo **Integración**,
   buscar **Artifact** y descargarlo desde `main`.
4. Reiniciar Home Assistant. En **Configuración → Dispositivos y servicios →
   Añadir integración**, buscar **Artifact**.
5. Confirmar la URL de la API (`http://100.72.168.116:8000`) y la URL de la
   interfaz (`http://100.72.168.116:5173`), sugeridas para este notebook.
   Ambas direcciones deben ser accesibles desde Home Assistant y el navegador.
   Para otra instalación, introducir las direcciones de su equipo de Artifact.
6. Abrir el dispositivo **Artifact** y pulsar **Visitar** (o el enlace web
   equivalente de Home Assistant) para abrir la aplicación en el navegador.

El indicador de conexión consulta `/health` cada 30 segundos y queda no disponible
cuando la API no responde. No carga históricos ni inicia análisis al consultar.
Las direcciones se pueden cambiar desde **Reconfigurar** en la integración.
Esta versión requiere Home Assistant 2024.12 o posterior. La integración 0.3.0
añade **Artifact-Alertas** a la barra lateral izquierda de Home Assistant. Abre
una vista independiente que se actualiza cada tres segundos: rojo para una
falla, con el posible tipo si fue identificado; amarillo con **Se encontró una
anomalía.** para anomalías y sospechas sin diagnóstico. Cada aviso muestra su
entidad y fecha. Las alertas se guardan al llegar a salida, incluso durante una
evaluación en curso, y siguen disponibles al cerrar la evaluación o reiniciar
Artifact. **Ver alertas anteriores** permite consultar el registro por páginas.
Las ejecuciones anteriores también aportan sus alertas al actualizar la API.
La misma vista se puede abrir directamente en `/alertas` de la interfaz de
Artifact; el panel incluye **Abrir en otra ventana** para abrirla por separado.
Para que aparezca la nueva entrada hay que actualizar la API, la interfaz y la
integración en HACS, y recargar la integración.

La integración 0.3.1 corrige la altura del panel para ocupar toda la ventana,
incluidas las pantallas móviles. Después de instalar esta corrección, recargar
la página de Home Assistant con **Ctrl+F5** para cargar el componente actualizado.

A partir de la integración 0.2.0, Home Assistant establece una conexión
saliente con Artifact y proporciona el historial y los cambios de estado usando
sus APIs internas. La interfaz detecta esta conexión y no pide URL ni token de
Home Assistant. La actualización requiere instalar tanto el backend como la
integración; el enlace existente **Visitar** continúa funcionando.

La integración crea y conserva una clave privada, limitada a este vínculo, en
su entrada de configuración. Artifact conserva únicamente su huella en
`storage/home_assistant_bridge.json` y acepta el primer vínculo; las conexiones
posteriores deben presentar la misma clave. Un vínculo existente no se puede
sustituir con otra integración ni con otra clave. Se permiten solo consultas
de entidades/historial y suscripciones de estado, sin llamadas a servicios.
No se crea un token de acceso de Home Assistant ni se comparte la sesión del
navegador. Al reiniciar, la integración restablece la conexión automáticamente.

### Actualizar una instalación existente

HACS actualiza únicamente la integración. Para que los cambios de la aplicación
aparezcan al abrirla desde Home Assistant, también hay que actualizar los
contenedores en el equipo de Artifact. Con el código actualizado, ejecutar
desde la carpeta del repositorio:

```powershell
docker compose up -d --build api web
```

Esto reconstruye la API y la interfaz y conserva los datos de los volúmenes.
Reiniciar contenedores existentes sin reconstruirlos sigue usando su código
anterior. El puerto habitual de la interfaz es **5173**; **5174** corresponde
a una vista de desarrollo independiente.

Después de actualizar los contenedores y la integración en HACS, recargar
**Configuración → Dispositivos y servicios → Artifact → ⋮ → Recargar**
(o reiniciar Home Assistant). Si Home Assistant se reinició antes de actualizar
la API de Artifact, recargar la integración después para iniciar el vínculo.
Finalmente, volver a abrir **Visitar** y recargar la página del navegador.

Para comprobar el vínculo, abrir `/api/history/home-assistant/config` en la
misma dirección de la interfaz. `bridge_connected: true` confirma la conexión;
una respuesta `404` indica que se sigue utilizando una API anterior.

### Selector de Historial y página principal

Artifact abre en **Evaluación de flujos**. El historial que prepara los filtros
se elige en **Selector de Historial**, accesible desde la navegación y desde
la evaluación de flujos:

- **Home Assistant:** con la integración conectada, elegir las fechas del período.
  La conexión se resuelve automáticamente. Al consultar el catálogo se pueden
  desactivar los sensores que no se quieran analizar.
  Se importa únicamente el historial que conserva Home
  Assistant; un período sin lecturas devuelve un error y conserva el CSV anterior.
- **Archivo CSV:** seleccionar un archivo con las columnas `entity_id`, `state`
  y `last_changed`. También se puede revisar el CSV guardado en el servidor.

Ambos orígenes muestran una selección de sensores antes de crear las bases.
Todos comienzan activos. Solo al pulsar **Preparar filtros con N sensores** se
importa o activa el historial elegido y se preparan las bases estadística y
contextual. Las lecturas de los sensores desactivados quedan fuera de ambas
bases y del catálogo de evaluación. Revisar un CSV o consultar el catálogo de
Home Assistant conserva la referencia activa y no inicia su preparación.

La selección permite buscar por nombre o identificador, filtrar por dominio,
tipo y estado de selección, y activar/desactivar todos o todos los resultados
filtrados, incluidas otras páginas. Se muestran 50 sensores por página, incluso
con 5.000 disponibles. Las selecciones grandes muestran un aviso del coste
contextual: 5.000 sensores permiten hasta 12.497.500 pares antes de considerar
retardos. La paginación evita saturar la interfaz, pero no elimina ese coste de
cálculo; el usuario puede reducir el conjunto antes de confirmarlo.

Los CSV se reciben y revisan en streaming en un área temporal, sin guardar sus
lecturas completas en memoria para construir el catálogo. Las preselecciones
caducan tras 24 horas; la referencia confirmada conserva únicamente las lecturas
activas y se recupera al reiniciar. Un CSV guardado sin selección confirmada
espera esa selección y no prepara bases automáticamente al iniciar el servidor.
Las suscripciones en directo se dividen en
lotes de hasta 1.000 entidades para admitir selecciones de 5.000 sin cambiar la
integración de Home Assistant.

La integración consulta Recorder desde Home Assistant, con consultas por día
y lotes de entidades, y entrega los datos a Artifact. La interfaz muestra el
progreso junto con la preparación de los filtros. La procedencia Home
Assistant/CSV se conserva al reiniciar Artifact. Recorder debe estar activo
para consultar históricos; los eventos entrantes usan el bus de estados.

Para una instalación independiente que no utilice la integración, se conserva
la conexión manual mediante la [API de historial de Home Assistant](https://developers.home-assistant.io/docs/api/rest/).
Se puede introducir URL y token de larga duración, o configurar las variables
`HOME_ASSISTANT_URL` y `HOME_ASSISTANT_TOKEN` en el backend o `.env` de Docker
Compose. El token configurado solo se usa para esa URL. El token introducido en
la interfaz no se guarda en el navegador, CSV ni estado de los trabajos.

En **Evaluación de flujos** se diseña el recorrido y se cargan los eventos
futuros para evaluarlo. **Selector de Historial** concentra la elección de
Home Assistant o CSV y las herramientas para consultar entidades,
normalización y parámetros del histórico activo.

La evaluación de eventos futuros permite elegir **Archivo CSV** o **Home Assistant
en directo**. La integración transmite los cambios de las entidades seleccionadas
directamente desde Home Assistant. En una instalación independiente se usa la
[API WebSocket de Home Assistant](https://developers.home-assistant.io/docs/api/websocket/).
Artifact evalúa los cambios recibidos de las entidades del historial y conserva el estado
del flujo entre eventos. La interfaz muestra la conexión, los contadores y los
últimos resultados. **Detener recepción y guardar** termina la sesión y conserva
la ejecución en el listado habitual. La falta de eventos no detiene la recepción.
Si la conexión se interrumpe, la misma sesión reintenta conectarse automáticamente
y conserva el estado del flujo y los resultados recibidos. La interfaz informa
de los cortes; puede haber eventos durante esos intervalos que no se reciban.
Un fallo temporal al consultar el estado desde el navegador tampoco descarta
la sesión. Las
instalaciones independientes pueden usar las mismas credenciales del selector.

Si un flujo incluye un clasificador habilitado sin modelo disponible, la
evaluación no se inicia. Se muestra un aviso junto a los controles de evaluación
para seleccionar un modelo válido o desactivar ese filtro. Los errores de
preparación se informan explícitamente y no se presentan como falta de eventos.

Los eventos se agrupan durante 250 ms para ordenar las lecturas simultáneas. Los
eventos que llegan después de que su instante ya fue evaluado se excluyen y se
cuentan como fuera de orden. Se reciben solo entidades conocidas y cambios
posteriores al historial; los cambios únicamente de atributos se omiten. Una
cola llena detiene la recepción e informa del motivo, conservando los resultados
ya recibidos. Los eventos en directo no llevan etiquetas de referencia.

### Acceso mediante Tailscale

Esta instalación utiliza las siguientes direcciones:

| Equipo o servicio | Dirección |
| --- | --- |
| Home Assistant | `100.70.236.99` |
| Notebook que ejecuta Artifact | `100.72.168.116` |
| API de Artifact | `http://100.72.168.116:8000` |
| Interfaz de Artifact | `http://100.72.168.116:5173` |

Tailscale debe estar activo en ambos equipos y en el dispositivo desde el que
se abre la interfaz, con acceso a la misma red Tailscale. Las conexiones de
Home Assistant hacia Artifact usan la IP del notebook. El notebook debe estar
encendido y con los servicios de Docker ejecutándose para que Artifact responda.

Para cambiar una integración ya instalada, abrir **Configuración → Dispositivos
y servicios → Artifact → ⋮ → Reconfigurar** e introducir las dos URL de Artifact
de la tabla. Esto actualiza la conexión y el enlace **Visitar** del dispositivo.
Las direcciones guardadas se conservan hasta confirmar la reconfiguración;
actualizar el código de HACS no las reemplaza automáticamente.

Docker publica los puertos 8000 y 5173 en las interfaces del notebook, incluida
Tailscale. El proxy `/api` de la interfaz sigue conectándose a `http://api:8000`
dentro de Docker y permite abrir Artifact desde la IP Tailscale sin cambiar
la configuración CORS.

La interfaz envía las peticiones a `/api` en su propio origen. El servidor web
las reenvía a `http://api:8000` dentro de Docker, por lo que no se necesitan
excepciones CORS para cada cliente. Al ejecutar Vite fuera de Docker, el destino
predeterminado es `http://127.0.0.1:8000`; se puede cambiar con
`ARTIFACT_API_TARGET`. Si se publica un build estático, el servidor que lo aloje
debe proporcionar el mismo proxy `/api`. `VITE_API_BASE_URL` permite indicar
una API externa cuando ya existe una configuración de proxy/CORS propia.

Home Assistant y el navegador necesitan acceso de red al equipo de Artifact.
La integración comprueba el contrato real de `/health`; no acepta cualquier
servidor que devuelva `200 OK`. Las URL deben ser HTTP(S), sin credenciales,
parámetros de consulta ni fragmentos. La autenticación de un proxy externo no
está implementada en esta primera versión.

Para verificar la integración en un entorno de desarrollo Linux con Python 3.12:
`pip install homeassistant==2024.12.5 pytest==8.3.4 pytest-asyncio==0.24.0`,
seguido de `python -m pytest tests/home_assistant -q`. El workflow de HACS
comprueba además el manifiesto y la estructura del repositorio.
La validación está orientada a un repositorio personalizado: los requisitos
adicionales del catálogo oficial sobre descripción, temas y licencia quedan
fuera de este workflow. La licencia del proyecto debe decidirla su autor.

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
Desde Tailscale: http://100.72.168.116:5173 y http://100.72.168.116:8000/docs.
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

Para entrenar en otro ordenador con NVIDIA, consulta la [guía de entrenamiento
CUDA](training/README.md): instalación mínima, comprobación de GPU, entrenamiento
desde consola, checkpoints por época y transferencia verificada de datasets y
candidatos. Se reutiliza la arquitectura del backend y los modelos se recargan en
CPU. El snapshot histórico de Git LFS usa una arquitectura anterior; para este
entrenamiento hay que preparar un dataset nuevo por segmentos.

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
