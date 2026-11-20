# Paranoid

Aplicación local para analizar históricos de sensores, detectar comportamientos
inusuales y distinguir entre anomalías ambientales y posibles fallas del sensor.
Permite explorar y etiquetar lecturas, aplicar filtros y diseñar flujos visuales
para evaluar datos en orden temporal, desde archivos CSV o desde Home Assistant.

El clasificador permite importar modelos GAT-LSTM preentrenados desde un ZIP.
Su backend está desarrollado con Python y FastAPI, y la interfaz con
React. El código puede ampliarse para incorporar nuevos tipos de nodos.

## Instalación

Los ejemplos usan PowerShell en Windows. Se puede ejecutar la aplicación con
Docker Compose o instalar el backend y la interfaz directamente en el equipo.

### 1. Descargar el proyecto

En el [repositorio de Paranoid](https://github.com/XpellCasting/Artifact), elegir
**Code → Download ZIP** y extraer el proyecto. Abrir PowerShell en la carpeta
que contiene `docker-compose.yml`. Si ya se dispone de una copia del código,
abrir esa carpeta directamente.

Para la evaluación con el modelo C, descargar también `Modelo_C_20261030.zip`
desde el enlace de Drive proporcionado por quien organiza la prueba. Conservar
ese ZIP sin descomprimir para importarlo desde la aplicación. Esta instalación
funciona sin Git LFS.

### 2. Instalar y ejecutar con Docker

Requiere Docker con Compose v2. En Windows, iniciar Docker Desktop con
contenedores Linux antes de continuar. Esta opción instala Python, Node.js y
las dependencias dentro de los contenedores.

Desde la raíz del repositorio:

```powershell
docker compose up --build
```

La primera ejecución construye las imágenes, instala las dependencias e inicia
la API y la interfaz. En una instalación nueva, el clasificador queda disponible
al importar y activar un modelo.

En Docker, el primer arranque con un volumen nuevo crea bases SQLite vacías. Los
flujos, las evaluaciones y los archivos de resultados que se generen se guardan
en el volumen Linux `active_databases`. Los reinicios y las recreaciones del
contenedor reutilizan ese volumen. Los datos de instalaciones anteriores siguen
en `storage/database/` de Windows y quedan fuera del nuevo historial.
Incluir el volumen en las copias de seguridad. El comando `docker compose down -v`
elimina los volúmenes del proyecto.

Cuando ambos servicios estén disponibles, abrir:

- Aplicación: [http://localhost:5173](http://localhost:5173).
- Documentación de la API: [http://localhost:8000/docs](http://localhost:8000/docs).

Para detener los servicios, presionar `Ctrl+C`. Para iniciarlos después en segundo
plano o detenerlos desde otra terminal, usar desde la raíz del repositorio:

```powershell
docker compose up -d
docker compose stop
```

Los directorios `data/`, `storage/` y `models/` conservan el estado entre reinicios.

### 3. Importar y activar el modelo C

1. En Paranoid, abrir **Clasificador diagnóstico → Modelos y evaluación**.
2. En **Importar modelo entrenado**, elegir `Modelo_C_20261030.zip` y pulsar
   **Importar ZIP**.
3. Pulsar **Activar versión experimental** y comprobar que el encabezado muestra
   **Modelo entrenado activo** con la versión `4d238952b2af4730a52c1c1a355f738c`.

Los flujos que usan **Modelo activo** utilizarán esa versión. La selección se
conserva al reiniciar Paranoid.

### Home Assistant (Marketplace y HACS)

La integración conecta Home Assistant con Paranoid, proporciona históricos y
cambios de estado, publica sensores de confianza y permite abrir la aplicación y
sus paneles de alertas e historial de confianza. Requiere
Home Assistant 2024.12 o posterior. Paranoid debe estar ejecutándose por separado.

1. En Home Assistant 2026.11 o posterior, abrir **Configuración → Marketplace →
   Añadir desde un enlace** y añadir `https://github.com/XpellCasting/Artifact`.
   En versiones anteriores, añadir la misma dirección en **HACS → ⋮ →
   Repositorios personalizados** con el tipo **Integración**.
2. Instalar **Paranoid** y reiniciar Home Assistant si lo solicita.
3. En **Configuración → Dispositivos y servicios → Añadir integración**, buscar
   **Paranoid** e introducir las direcciones del equipo donde está ejecutándose:
   API `http://<equipo-paranoid>:8000` e interfaz `http://<equipo-paranoid>:5173`.
   Reemplazar `<equipo-paranoid>` por su IP o nombre de red. Ambas direcciones
   deben ser accesibles desde Home Assistant y desde el navegador.
4. Una vez cargada la integración, Home Assistant añade **Paranoid** y
   **Paranoid-Alertas** y **Paranoid-Confianza** a la barra lateral. Los accesos se retiran al
   desactivar o eliminar la configuración. El dispositivo **Paranoid** también ofrece **Visitar**.

Las direcciones se pueden cambiar desde **Reconfigurar** en la integración.
Desde la versión 0.4.1, la aplicación y la integración se llaman **Paranoid**.
Al actualizar y recargar la integración, el nombre predeterminado anterior se
actualiza automáticamente; los nombres personalizados se conservan. El dominio
interno `artifact` mantiene el vínculo, las entidades y las rutas existentes.
La integración es la única conexión de Paranoid con Home Assistant para consultar
sensores, importar históricos y recibir eventos en directo. No se introducen URL
ni tokens de Home Assistant en Paranoid. El puente conserva su token interno y
la integración conserva la dirección del servidor Paranoid. La consulta de
históricos requiere Recorder activo. Solo se pueden iniciar consultas o sesiones
cuando la conexión está **Conectada**; la interfaz también muestra **Comprobando
conexión**, **Integración no configurada** y **Desconectada**.
Las direcciones `localhost` de los ejemplos de instalación sirven para abrir
Paranoid desde el mismo equipo donde se ejecuta.

Para incorporar los sensores y los tres paneles a una instalación existente,
actualiza la integración a la versión 0.5.1 o posterior desde Marketplace o HACS
y reinicia Home Assistant. Actualiza también el backend y la interfaz de Paranoid.
Los históricos y vínculos guardados se conservan; no es necesario volver a
emparejar. Una integración anterior todavía entrega el catálogo plano, con un
aviso de que los metadatos de áreas no están disponibles.

### Alternativa: instalación local sin Docker

Requiere Python 3.12 y Node.js 22.13 o superior dentro de la rama 22, o Node.js 24,
con npm. Estas versiones de Node.js satisfacen los requisitos del lockfile de
la interfaz. Descargar primero el proyecto como se indica en el paso 1.

En una terminal PowerShell, desde la raíz del repositorio, crear el entorno,
instalar las dependencias e iniciar la API:

```powershell
py -3.12 -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt -r backend/requirements-gat.txt
$env:PYTHONPATH = "$pwd/backend"
backend/.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Si Python 3.12 se invoca mediante `python` en el equipo, reemplazar `py -3.12`
por `python` en la primera orden. Los comandos usan directamente el intérprete
del entorno virtual, por lo que no es necesario activarlo.

En otra terminal PowerShell, también desde la raíz del repositorio, instalar e
iniciar la interfaz:

```powershell
cd frontend
npm ci
npm run dev
```

Abrir [http://localhost:5173](http://localhost:5173). La documentación de la API
está en [http://localhost:8000/docs](http://localhost:8000/docs). Mantener ambas
terminales abiertas y presionar `Ctrl+C` en cada una para detener la aplicación.
En los siguientes arranques basta con definir `PYTHONPATH`, iniciar Uvicorn y
ejecutar `npm run dev` desde `frontend`; no es necesario reinstalar dependencias.

Para usar el modelo C, seguir el paso 3 en la interfaz. Los históricos que se
quieran analizar se cargan desde **Selector de Historial**.

## Desinstalación completa

### Retirar la integración de Home Assistant

1. Si hay una sesión en directo, pulsar **Detener recepción y guardar**. Si también
   se quiere borrar su historial de Home Assistant, abrir el dispositivo **Paranoid**
   y anotar los identificadores de sus entidades de conexión y confianza, incluidas
   las de prueba. Usar los identificadores actuales si se renombró alguna entidad.
2. En **Configuración → Dispositivos y servicios → Paranoid**, abrir el menú de
   la entrada y elegir **Eliminar**. Esto cierra el vínculo activo y retira los
   tres paneles de la barra lateral.
3. Desinstalar **Paranoid** desde Marketplace o HACS, según el método de
   instalación, y reiniciar Home Assistant.

### Eliminar el servidor y los datos locales

Si se utilizó un servidor compartido, corresponde a quien lo administra decidir
si se eliminan las evaluaciones y los datos de ese servidor.

- **Con Docker Compose**, abrir una terminal en la carpeta de esta instalación
  de `Artifact`, junto a `docker-compose.yml`. El siguiente comando elimina los
  contenedores, las imágenes locales del proyecto y sus volúmenes, **incluidas
  las bases de datos y las evaluaciones guardadas**.

  ```powershell
  docker compose down --volumes --rmi local
  ```

- **Con instalación local sin Docker**, detener la API y la interfaz con
  `Ctrl+C` en cada una de sus terminales.

En ambos casos, eliminar después la **copia del proyecto utilizada para la
instalación**. Allí se guardan los históricos importados en `data/`, el vínculo
con Home Assistant en `storage/home_assistant_bridge.json` y los modelos en
`models/`. Si se configuraron rutas de datos externas, revisar también esas
ubicaciones.

### Borrar el historial de la prueba y los datos del navegador

En Home Assistant, con una cuenta administradora, ejecutar `recorder.purge_entities` con los
identificadores anotados y `keep_days: 0` para borrar todo su historial de
estados. Por ejemplo, para borrar la confianza de una puerta, seleccionar
solo `sensor.confianza_binary_sensor_puerta`. Así se conserva el historial del
sensor original `binary_sensor.puerta`. En **Herramientas → Estadísticas**,
eliminar también las estadísticas de largo plazo de las entidades de Paranoid.
[Purga de entidades en Recorder](https://www.home-assistant.io/actions/recorder.purge_entities/) ·
[Estadísticas de largo plazo](https://www.home-assistant.io/docs/configuration/long-term-statistics/)

Para retirar las copias locales de la prueba, eliminar las exportaciones
descargadas y borrar los datos del sitio de Paranoid en el navegador utilizado.
Las copias de seguridad previas conservan los datos que incluían.

## Uso

1. Abrir **Selector de Historial** y elegir **Home Assistant** o **Archivo CSV**.
   Seleccionar el período o archivo, revisar los sensores y confirmar
   **Preparar filtros con N sensores**. Esperar a que termine la preparación.
2. En **Evaluación de flujos**, elegir una plantilla en **Nuevo desde plantilla**.
   Añadir o conectar nodos y seleccionar cada uno para ajustar sus propiedades.
   En el clasificador, mantener **Modelo activo** para usar el modelo preentrenado.
3. Elegir el origen de los eventos que se evaluarán: **Archivo CSV** para un
   conjunto etiquetado o **Home Assistant en directo** para recibir cambios de
   estado de los sensores seleccionados.
4. Guardar y evaluar el archivo, o iniciar la recepción en directo. En este último
   caso, **Detener recepción y guardar** termina la sesión y conserva la ejecución.
5. Consultar las ejecuciones guardadas, seleccionar un nodo y abrir el detalle de
   evidencias de cada lectura. Los módulos estadístico, contextual y de análisis
   y etiquetado permiten explorar los resultados y revisar los datos.

Los sensores de Home Assistant empiezan sin seleccionar. Las áreas con sensores
compatibles aparecen en paneles inicialmente cerrados; **Sin área asignada** se
muestra como una lista independiente. El área asignada a la entidad tiene
prioridad sobre la de su dispositivo, siguiendo las
[reglas de Home Assistant](https://www.home-assistant.io/docs/organizing/areas/).
La búsqueda incluye nombre, `entity_id` y área, sin distinguir mayúsculas ni
tildes, y se combina con los filtros de área, dominio, tipo y selección.
Cada lista muestra 50 sensores por página. Las acciones de resultados y de un
área afectan a todas las coincidencias, incluidas las otras páginas; **Activar
todos** y **Desactivar todos** afectan al catálogo completo. Plegar paneles,
buscar o cambiar de página conserva la selección. Desde 500 seleccionados se
muestra una advertencia de coste de preparación. CSV conserva su selección
inicial completa.

Consultar sensores no descarga históricos ni prepara filtros. Al confirmar se
importan únicamente los `entity_ids` elegidos, que forman una única referencia
estadística y contextual, incluidas las relaciones entre áreas. Si se corta una
sesión en directo ya iniciada, se reintenta la conexión hasta detenerla y guardar.

El contrato de `POST /history/home-assistant/entities` recibe `start`, `end` y
opcionalmente `entity_ids`, y devuelve `entities`, `entity_count`, `source` y
`areas_available`. Con metadatos disponibles, cada entidad incluye `area_id` y
`area_name` (ambos nulos cuando no tiene área); con metadatos desconocidos esos
campos se omiten. El puente consulta los registros una sola vez y entrega el
catálogo con la operación `entity_catalog` en bloques de hasta 500 entidades.
`POST /history/home-assistant/prepare` y `/history/home-assistant/import/start`
requieren `entity_ids` no vacíos. `POST /flows/{flow_id}/live/start` recibe solo
`entity_ids` opcionales, limitados a la referencia activa. Estas solicitudes
rechazan los campos antiguos `url`, `token` y `connection_mode` con HTTP 422.
`GET /history/home-assistant/config` devuelve únicamente `bridge_configured` y
`bridge_connected`. Docker Compose ya no utiliza variables de conexión directa
a Home Assistant; el vínculo existente continúa en `storage/home_assistant_bridge.json`.

Cada nombre identifica un flujo guardado. Si se intenta guardar otro flujo con
el mismo nombre, se carga el existente sin crear otra copia ni reemplazar su
diseño. Las mayúsculas y los espacios adicionales no crean nombres distintos.
Si esto ocurre al iniciar una evaluación, revisar el flujo cargado y volver a
iniciarla. Los nombres que ya estaban repetidos muestran el guardado más reciente.

Los CSV de lecturas deben incluir `entity_id` (sensor), `state` (valor o estado)
y `last_changed` (fecha y hora, preferiblemente con zona horaria). Los conjuntos
etiquetados incluyen además `es_sintetica`, `tipo` y `tipo_falla`; sus etiquetas
se utilizan para comprobar los resultados.

**Paranoid-Alertas**, también disponible en la ruta `/alertas` de la interfaz,
muestra las detecciones de la última ejecución: rojo para posibles fallas y
amarillo para anomalías y sospechas. Permite filtrar los avisos y consultar
resultados anteriores de esa misma ejecución.

El editor admite ramas y conexiones según la señal de cada nodo. El flujo debe
tener una entrada y una salida, y no puede contener ciclos. Durante la evaluación,
cada lectura utiliza solo la información disponible hasta ese instante.

### Nodos disponibles

| Nodo | Funcionamiento |
| --- | --- |
| Entrada | Introduce en el flujo las lecturas del archivo seleccionado o los eventos recibidos desde Home Assistant. |
| Estadístico FOCuS | Detecta desviaciones respecto del comportamiento histórico. Usa FOCuS sobre residuos en sensores continuos, FOCuS Bernoulli en binarios y reglas para sensores de varios estados. |
| Contextual | Contrasta una lectura con sensores relacionados para comprobar si el contexto respalda o contradice su comportamiento. |
| Clasificador | Usa el modelo GAT-LSTM preentrenado para distinguir comportamiento normal, anomalía ambiental y posible falla del sensor. Puede devolver un resultado indeterminado cuando la evidencia es insuficiente. |
| Unión | Aplica una función lógica a los resultados de una misma lectura: «cualquiera positiva» (O) o «todas positivas» (Y). Conserva las evidencias de cada filtro. |
| Decisión | Evalúa si la señal recibida cumple la condición seleccionada. Devuelve positiva para verdadero y sin señal para falso; seleccionar solo «Sin señal» permite negar una señal evaluable (NO). |
| Salida | Reúne los resultados finales, conserva su recorrido de evidencias y calcula un índice de confianza por lectura con ponderación uniforme. |

Las señales son `positive` (señal positiva), `negative` (sin señal) y `unknown`
(no evaluable). Las conexiones pueden dejar pasar todas o solo algunas de ellas.
Una sospecha indica que conviene revisar la lectura; el diagnóstico expresa
compatibilidad con un patrón de falla.

Unión y Decisión conservan su resultado lógico aunque las evidencias diagnósticas
discrepen. Una entrada ausente o no evaluable representa un valor desconocido:
en O basta un verdadero; en Y basta un falso para resolver el resultado. Si no
puede resolverse, la señal permanece no evaluable. Decisión conserva ese valor
desconocido salvo que se seleccione explícitamente como condición verdadera.
Las conexiones determinan qué resultados pasan a cada rama.

Para comparar los tres filtros, las plantillas incluyen O, `A Y B Y C` y
`(A O B) Y C`, donde A es Estadístico FOCuS, B es Contextual y C es Clasificador.
Se pueden encadenar uniones y decisiones para construir otras fórmulas. En la
Salida, el resultado lógico determina la detección usada en las métricas:
falso produce «Sin alerta» y desconocido produce «Indeterminado». Un verdadero
produce una sospecha o un diagnóstico respaldado por las evidencias. El análisis
diagnóstico combinado se conserva por separado y no sustituye el valor lógico.
Si después de una compuerta se ejecuta otro filtro, ese filtro calcula una nueva
evaluación con la política de evidencias seleccionada.

## Guardado de resultados y evidencias

Las lecturas nuevas conservan el contrato diagnóstico `module-envelope-v1` y
añaden una traza `execution_trace` con versión `filter-execution-v1`. Esta
información alimenta el cálculo de confianza en Salida y conserva las señales
y diagnósticos existentes.

Cada resultado de Salida incluye `flow_id`, `execution_id` y `config_hash`.
`snapshot.execution_context` contiene las mismas referencias y la versión de
la traza; la configuración completa permanece en `snapshot.flow`. `config_hash`
es SHA-256 del grafo y las configuraciones de sus nodos serializados de forma
canónica. Omite nombres visibles, posiciones e identificadores de almacenamiento;
conserva el orden de las listas. Identifica la configuración, no los datos ni
los pesos de un modelo: las identidades de modelos, mapas y referencias siguen
en las evidencias y la instantánea existentes.

La traza de cada lectura contiene:

- `filters`: un registro por nodo de filtro configurado, con tipo, activación,
  estado de ejecución, motivo, versión disponible, secuencia, grupo y referencias
  a las evidencias producidas y recibidas.
- `absent_filter_types`: tipos de filtro que no forman parte del grafo. Un filtro
  configurado pero desactivado se identifica como `disabled` en `filters`.
- `availability_mask`: disponibilidad de resultados por nodo; un filtro sin
  resultado utilizable no equivale a una evaluación normal.
- `execution_order`: filtros cuya ejecución se intentó para esa lectura.
- `groups`: llamadas de filtros, sus timestamps, cantidad de observaciones y
  duración compartida. Son lotes de procesamiento, no grupos de ejecución paralela.
- `evidence`: evidencias originales de filtros ejecutados, incluidas las ramas
  que no llegaron a Salida y las evidencias de control que recibieron otros
  filtros. El archivo reutiliza su conjunto de evidencias compartidas mediante
  referencias. El campo diagnóstico `evidence` de Salida mantiene su recorrido.

`execution_status` está separado del `status` diagnóstico. Sus valores son
`pending` durante la llamada, `completed`, `disabled`, `skipped`,
`insufficient_data` y `error`. Los resultados finales no conservan `pending`.
Una abstención del clasificador con puntuaciones válidas puede estar `completed`
y conservar su diagnóstico `indeterminate`.

Las duraciones de `groups` usan `timing_scope=filter_call`; no se dividen entre
lecturas para simular mediciones individuales. En evaluación causal, la
preparación estadística se registra aparte en `groups[].preparation`, con ámbito
`timestamp_group`. La latencia existente añade `latency_scope=timestamp_group`,
`latency_group_id` y `latency_group_size`, porque es compartida por las lecturas
del mismo grupo temporal.

Un error de filtro conserva el comportamiento de detener la evaluación. El
estado del trabajo añade `filter_execution_context`, con una muestra explícita
de hasta 20 lecturas y el indicador `truncated`; no publica una ejecución
terminada. Las ejecuciones anteriores continúan leyéndose sin inventar trazas.

Las pruebas específicas cubren las siete combinaciones de uno, dos y tres
filtros activos, los seis órdenes, filtros ausentes o deshabilitados, ramas,
datos insuficientes, errores y persistencia. Con el entorno Python del proyecto
y `PYTHONPATH` apuntando a `backend`, ejecutar:

```powershell
python -m unittest discover -s tests/backend -p "test_filter_execution_trace.py" -v
```

### Confianza por lectura en Salida

Salida calcula automáticamente `confidence` para cada lectura nueva, sin añadir
un nodo al grafo. La política `uniform_v1` aplica `1 - media(sospechas)` a los
resultados utilizables, con una aportación por nodo de filtro. Los parámetros
de esta política quedan guardados una vez en `snapshot.confidence_policy`.
La ponderación aprendida y la calibración se reservan para trabajo futuro.

Los adaptadores usan las magnitudes originales de `execution_trace.evidence`:

- Estadístico: para un puntaje relativo a su umbral, `r / (1 + r)`; los puntajes
  ya acotados de binarios y multivalor conservan su escala. Una detección
  explícita de persistencia o indisponibilidad aporta sospecha 1. Un resultado
  `no_suspicion` sin puntaje numérico aporta 0 y registra ese adaptador explícito.
- Contextual: apoyo contextual aporta 0; contradicción usa su intensidad relativa
  al umbral o su indicador acotado. No utiliza `adjusted_suspicion_score`, que
  podría contar por segunda vez la evidencia estadística.
- Clasificador: usa `class_scores.sensor_fault`; el máximo de las clases puede
  corresponder a Normal o a una anomalía ambiental y no se usa como sospecha.

El resultado mantiene `confidence_kind=operational_index` y
`confidence_calibrated=false`. Un valor `0.90` representa un índice del 90 %,
no una probabilidad calibrada de funcionamiento correcto. Los filtros omitidos,
deshabilitados, sin datos o sin un puntaje válido no reciben una aportación cero:
se excluyen de la media. `evidence_coverage` es la proporción de filtros
habilitados que aportaron un puntaje válido; no multiplica la confianza. Los
motivos de exclusión, adaptadores y referencias se guardan en
`confidence_fusion`, con versión `reading-confidence-v1`.

Sin evidencia utilizable, con una lectura inválida o sin ruta de salida,
`confidence` es `null` y se registra el motivo. Una ejecución pendiente o con
error tampoco produce confianza válida. Cada lectura se calcula por separado,
antes de comparar con ground truth. Las uniones y decisiones no aportan votos
adicionales ni cambian el índice; las evidencias ejecutadas en ramas descartadas
siguen disponibles a través de la traza. Los cambios de orden solo afectan al
índice cuando modifican las evidencias producidas.

Los valores se guardan y se consultan mediante los archivos y endpoints de
resultados existentes, tanto en CSV como en directo. Las ejecuciones anteriores
conservan sus datos y no se les asigna retrospectivamente un índice. El benchmark
de replay sigue comparando resultados originales de detectores y diagnósticos
con referencias anteriores; comprueba las referencias de las nuevas anotaciones.
Las pruebas específicas verifican el cálculo de confianza de forma independiente:

```powershell
python -m unittest discover -s tests/backend -p "test_flow_confidence.py" -v
```

### Entidades e historial de confianza

En **Evaluación de flujos → Gestionar sensores de confianza**, selecciona los
sensores inactivos que quieras retirar, pulsa **Borrar selección** y confirma.
El borrado distingue los modos **En directo** y **Test**. Quita la entidad del
catálogo y del registro de Home Assistant. El historial y los resultados de las
ejecuciones guardadas en Paranoid siguen disponibles. Si vuelves a evaluar el
sensor en el mismo modo, se publicará de nuevo.

Para que el borrado llegue a Home Assistant, usa la integración **Paranoid 0.6.4**
o posterior. Si el puente está desconectado, el catálogo se actualiza de inmediato
y la integración procesa el borrado en la siguiente sincronización.

Al iniciar un flujo en directo, el selector lista los sensores existentes en Home
Assistant. Al agregar uno que no está en el historial de referencia, aparece un
aviso. Paranoid evalúa y registra una entidad de confianza por sensor agregado.
Con la selección vacía, evalúa todos los sensores compatibles que coinciden en
Home Assistant y el historial de referencia. Por ejemplo, `sensor.temperatura` genera el identificador
lógico `confianza.sensor_temperatura`. En Home Assistant se publica como
`sensor.confianza_sensor_temperatura`, con unidad `%` y clase de estado
`measurement`, para que Recorder conserve su historial y estadísticas. El
dominio original distingue, por ejemplo, un `sensor.puerta` de un
`binary_sensor.puerta`. El identificador lógico se conserva en el atributo
`confidence_entity_id`; si se cambia el nombre de la entidad en Home Assistant,
el panel utiliza su identificador registrado real.

La entidad muestra `unknown` hasta recibir una lectura evaluable y publica el
índice entre 0 y 100. Una lectura sin evidencia utilizable vuelve a `unknown`.
Al detener el flujo, reiniciar el backend o perder la conexión, pasa a
`unavailable`, evitando utilizar el último valor como si continuase vigente.
Una reconexión sincroniza el catálogo y los estados actuales. Los identificadores
permanecen estables al cambiar de flujo; la nueva sesión en directo toma el
control de los sensores que selecciona. Estas entidades se excluyen de las
entradas del puente para impedir realimentación.

Los atributos incluyen sensor y valor de origen, fecha de la lectura, fecha de
evaluación, flujo, ejecución, hash de configuración, cobertura y cantidad de
filtros utilizados. La confianza sigue siendo un índice operativo sin calibrar.
Home Assistant gestiona `last_changed` cuando cambia el estado; `evaluated_at`
identifica cada lectura procesada, aunque conserve el mismo porcentaje.

Desde la versión 0.6.0 de la integración y con el backend actualizado,
`confidence_change_reason` explica el último cambio y `confidence_change_detail`
guarda las dos lecturas comparadas, la cobertura, el orden real de ejecución
y los aportes de cada filtro. Las lecturas consecutivas con igual porcentaje
conservan el motivo anterior. Una sesión nueva empieza su propia comparación.
El detalle contextual incluye los sensores relacionados y las referencias a la
evidencia completa. Si la lista supera el espacio reservado para Recorder, se
indica el número total de vecinos. Los cambios de disponibilidad de un filtro
aparecen en el motivo.

Con la integración 0.6.1, cada filtro muestra su evidencia al abrir un cambio.
El contextual compara, por entidad relacionada, los valores observados y su
respaldo o contradicción. En el modelo temporal compara sus comprobaciones
de nivel, innovación y tendencia. El estadístico muestra los indicadores que
registró, incluido el valor previsto y la desviación cuando corresponda.
El clasificador de segmentos descompone el cambio en su puntuación de falla
entre las tres ventanas temporales del propio sensor. Calcula el aporte medio
de sustituir cada representación anterior por la nueva en los seis órdenes
posibles en la capa final del modelo. Los conteos de lecturas y transiciones
describen las entradas de cada ventana. La comparación usa la lectura inmediatamente
anterior de la misma sesión, incluso cuando había repetido el porcentaje.
Los registros antiguos muestran la evidencia que guardaron en su momento.

En **Paranoid-Confianza → Motivos de los cambios**, selecciona la entidad de
**En directo** o **Test**, el inicio del intervalo en hora local y **Consultar
15 minutos**. Abre una fila para ver los aportes y la evidencia de ese cambio.
El intervalo usa la hora de publicación en Home Assistant. Dentro de cada
fila figura también la fecha de la lectura original. Los motivos se pueden
consultar para estados publicados con esta versión que Recorder conserve.

**Paranoid-Confianza** usa tarjetas nativas `entities` y `history-graph`, con las
últimos 15 minutos y el eje fijado entre 0 % y 100 %. Requiere Recorder e History
activos y que estos sensores estén incluidos en su configuración de grabación.
El panel se administra desde la integración. También se pueden añadir los
sensores a un dashboard propio o utilizarlos en condiciones numéricas de las
automatizaciones. Ejemplo de tarjeta nativa:

```yaml
type: history-graph
title: Confianza de temperatura
entities:
  - sensor.confianza_sensor_temperatura
hours_to_show: 0.25
min_y_axis: 0
max_y_axis: 100
fit_y_data: false
```

Recorder registra la hora de publicación en Home Assistant. La fecha original
de la lectura permanece en `source_timestamp`; los intervalos desconectados no
se rellenan retrospectivamente en Recorder. Su retención depende de la
configuración de Home Assistant.

### Retención para funcionamiento continuo

Cada sensor tiene una fila de confianza actual por modo en el catálogo. En directo,
Home Assistant publica un nuevo estado cuando cambia el porcentaje, la disponibilidad,
la cobertura o la explicación. Una lectura que solo renueva la fecha o el identificador
se sigue evaluando y actualiza el catálogo. Recorder conserva los motivos de los cambios
según `purge_keep_days` y la purga configurada en Home Assistant.

La API guarda por defecto las últimas **10.000 observaciones por sesión en directo**,
compartidas entre los sensores del flujo. Esta ventana abarca el detalle por nodo,
el gráfico del backend y las alertas de esas observaciones. Los totales de la sesión
incluyen todas las lecturas evaluadas. La mediana y el p95 de latencia en directo
usan las últimas 10.000 latencias válidas. El repositorio del clasificador conserva
las últimas 10.000 decisiones nuevas en directo entre todas las sesiones.

Se conservan las **10 últimas sesiones en directo completadas** que usen esta
política, entre todos los flujos. Al guardar la siguiente se retira la sesión
más antigua junto con su archivo y alertas. Las evaluaciones CSV y las sesiones
anteriores a la actualización conservan su historial. Configurar los
enteros positivos `LIVE_HISTORY_READINGS` y `LIVE_HISTORY_SESSIONS` en el
`.env` de Compose y recrear la API para ajustar estos límites. El volumen ocupado
depende del tamaño de las evidencias, de los CSV guardados y de las bases existentes.

Las evaluaciones CSV con ground truth publican en el grupo **Test** de
**Paranoid-Confianza**. Para una lectura de `sensor.x`, el identificador lógico
es `confianza.test_sensor_x` y la entidad nativa es `sensor.confianza_test_sensor_x`.
Cada evaluación reutiliza la entidad del sensor y publica sus lecturas mientras
procesa el archivo. Al cerrar la evaluación, la entidad queda `unavailable` y
Recorder conserva su historial. El atributo `source_mode` vale `test` y
`source_timestamp` conserva la hora original del CSV. Si Home Assistant está
desconectado, la evaluación sigue disponible en Paranoid.

En **Resultados → Confianza**, elegir un sensor permite ver su gráfico de 0 a
100 %, con huecos para lecturas no evaluables y páginas de 1000 lecturas. La
misma pestaña está disponible al guardar una sesión en directo. Los archivos
nuevos incluyen un índice escalar para consultar el gráfico sin descomprimir
las evidencias completas. Los archivos anteriores siguen siendo legibles y
no reciben valores de confianza inventados.

`GET /confidence/entities` consulta el catálogo actual y
`GET /flow-runs/{run_id}/confidence` consulta una serie guardada, con parámetros
`entity_id` (sensor de origen), `offset` y `limit` (hasta 5000). Las pruebas del
catálogo, persistencia y puente se encuentran en
`tests/backend/test_home_assistant_confidence.py`; las entidades y tarjetas
nativas se prueban con Home Assistant real en `tests/home_assistant/`, y el
gráfico por ejecución en `tests/frontend/confidence-history.test.js`.

Las evaluaciones nuevas escriben un archivo incremental con formato
`flow-flyweight-v1`. Cada lectura comparte su identidad y un conjunto de
evidencias entre las vistas de los nodos; las vistas guardan referencias en
lugar de repetir todo el historial. Los bloques y el manifiesto diagnóstico se
comprimen sin pérdida. Las consultas reconstruyen el detalle completo y las
métricas frente a las etiquetas conservan su representación habitual.

Durante la evaluación se prepara un archivo privado en
`flow_lab.results/.pending/`, junto a la base de flujos. Al terminar se comprueba su
integridad, se sincroniza a disco y se publica junto con los metadatos de la
ejecución. El archivo publicado permanece inmutable. La base central conserva
las métricas, las identidades de modelos y la ubicación de ese archivo. Las
ejecuciones anteriores continúan leyéndose desde `flow_results`.

**Las copias de seguridad deben incluir tanto `flow_lab.sqlite3` como el
directorio `flow_lab.results/`.** No mover ni borrar los archivos publicados por
separado. Los archivos de `.pending/` son temporales y no se anuncian como
ejecuciones terminadas. Una interrupción conserva las ejecuciones confirmadas;
la evaluación en curso sigue necesitando ejecutarse nuevamente.

`scripts/benchmark_flow_storage.py` compara el guardado anterior y el nuevo a
partir de resultados ya evaluados, sin ejecutar detectores ni modificar la base
de producción. Comprueba por consulta todas las filas de todos los nodos y sus
métricas. Usar `--work-dir` en el mismo sistema de archivos que los datos para
medir también el costo de escritura del entorno real.

## Medir el rendimiento de evaluaciones CSV

El clasificador de segmentos prepara las características de las entidades
solicitadas en cada timestamp y comparte la muestra y la inferencia entre los
candidatos de ese grupo. Conserva las dimensiones del modelo y la procedencia
de las entradas. La caché pertenece a cada llamada diagnóstica y admite hasta
ocho grupos; el estado causal permanece en su sesión.

Para comparar esta preparación con la preparación completa de referencia:

```powershell
backend/.venv/Scripts/python.exe scripts/benchmark_causal_flow.py --reference-run ID_EJECUCION --sizes 5000 20000 --runs 3 --output outputs/benchmark-causal.json
```

Usar el identificador de una ejecución guardada cuyo CSV coincida con el archivo
local. El script comprueba su huella, congela el modelo registrado, completa el
último timestamp de cada prefijo y alterna ambas variantes. Verifica las salidas
de todos los nodos, las referencias de evidencia, las decisiones y las métricas
frente a las etiquetas. Guarda tiempos de replay y persistencia, CPU, RSS y
bytes escritos. La carga y la preparación estadística inicial se registran por
separado. Las escrituras del ensayo usan bases y archivos temporales aislados.

`--use-prepared-cache` reutiliza la preparación interna después de verificar su
procedencia estadística. `--flow-run ID` permite estudiar otro flujo congelado
como experimento nuevo sobre esos datos. Para montar los datos en solo lectura,
exportar primero las instantáneas y etiquetas con `--export-reference ruta.json`
y proporcionar ese archivo al ensayo mediante `--reference-file ruta.json`.

## Añadir nuevos nodos

Para incorporar otro tipo de nodo, usar uno de los existentes como referencia y
actualizar los siguientes puntos:

| Archivo | Cambio necesario |
| --- | --- |
| [backend/app/flow/definitions.py](backend/app/flow/definitions.py) | Registrar el tipo en `NODE_TYPES` y validar su configuración en `validate_flow`. |
| [backend/app/nodos/handlers.py](backend/app/nodos/handlers.py) | Implementar el procesamiento de las observaciones y el registro de evidencias. La lógica específica puede vivir en un módulo propio. |
| [backend/app/flow/runner.py](backend/app/flow/runner.py) | Incorporar el tipo en el despacho de `FlowRunner.run` para ejecutar su comportamiento. |
| [backend/app/flow/evaluacion.py](backend/app/flow/evaluacion.py) | Integrar, cuando corresponda, la preparación, el estado y el historial requeridos durante la evaluación causal con `CausalFlowRunner`. |
| [backend/app/flow/evidence.py](backend/app/flow/evidence.py) | Revisar `own_assessment` y `combine` si el nodo aporta una nueva evaluación; `combine` selecciona explícitamente los tipos que participan en la decisión combinada. |
| [frontend/src/features/flows/flow-lab-model.js](frontend/src/features/flows/flow-lab-model.js) | Añadir el nombre en `NODE_LABELS` y adaptar la validación y los valores de configuración que correspondan. |
| [frontend/src/features/flows/FlowLab.jsx](frontend/src/features/flows/FlowLab.jsx) | Incorporar la descripción, la configuración inicial y los controles del nuevo nodo en el editor. |

Cada nodo recibe observaciones y devuelve resultados con una señal `positive`,
`negative` o `unknown`. Debe conservar el identificador de la observación y las
evidencias previas, y registrar su propia evidencia mediante `_with_evidence`.
Si utiliza historial, solo debe consultar lecturas disponibles hasta el instante
evaluado. Las conexiones usan la señal para decidir por qué ramas continúa cada
observación.

Añadir pruebas en `tests/backend/` y `tests/frontend/` que comprueben la validación,
el procesamiento, el enrutamiento y la conservación de evidencias. Como referencia,
consultar `tests/backend/test_module_evidence.py`,
`tests/backend/test_causal_flow_incremental.py` y
`tests/frontend/flow-lab-model.test.js`.

Para comprobar los cambios desde la raíz del repositorio:

```powershell
backend/.venv/Scripts/python.exe scripts/run_tests.py
npm --prefix frontend test
npm --prefix frontend run build
docker compose config --quiet
```

Después de modificar el código, reiniciar los servicios locales o ejecutar
`docker compose up --build` para reconstruir los contenedores.
