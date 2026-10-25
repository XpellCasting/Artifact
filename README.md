# Artifact

Aplicación local para analizar históricos de sensores, detectar comportamientos
inusuales y distinguir entre anomalías ambientales y posibles fallas del sensor.
Permite explorar y etiquetar lecturas, aplicar filtros y diseñar flujos visuales
para evaluar datos en orden temporal.

La aplicación incluye un modelo GAT-LSTM preentrenado que se restaura durante la
instalación. Su backend está desarrollado con Python y FastAPI, y la interfaz con
React. El código puede ampliarse para incorporar nuevos tipos de nodos.

## Instalación

Los ejemplos usan PowerShell en Windows. Se puede ejecutar la aplicación con
Docker Compose o instalar el backend y la interfaz directamente en el equipo.

### 1. Descargar el proyecto

Instalar Git y Git LFS. Luego, desde la carpeta donde se quiera guardar el proyecto:

```powershell
git lfs install
git clone https://github.com/XpellCasting/Artifact.git
cd Artifact
git lfs pull
```

Si el repositorio ya está descargado, abrir PowerShell en su carpeta raíz
(`Artifact`, donde está `docker-compose.yml`) y ejecutar `git lfs install` y
`git lfs pull`. Git LFS descarga el paquete del modelo en
`artifacts/gat_lstm/snapshot.tar.gz`, necesario para la restauración inicial.

### 2. Instalar y ejecutar con Docker

Requiere Docker con Compose v2. En Windows, iniciar Docker Desktop con
contenedores Linux antes de continuar. Esta opción instala Python, Node.js y
las dependencias dentro de los contenedores.

Desde la raíz del repositorio:

```powershell
docker compose up --build
```

La primera ejecución construye las imágenes, instala las dependencias y restaura
los archivos ausentes del paquete del modelo preentrenado. Conserva la selección
de modelo activo existente.

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

### Alternativa: instalación local sin Docker

Requiere Python 3.12 y Node.js 22.13 o superior dentro de la rama 22, o Node.js 24,
con npm. Estas versiones de Node.js satisfacen los requisitos del lockfile de
la interfaz. Completar primero la descarga del proyecto y del paquete Git LFS.

En una terminal PowerShell, desde la raíz del repositorio, crear el entorno,
instalar las dependencias, restaurar el modelo e iniciar la API:

```powershell
py -3.12 -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt -r backend/requirements-gat.txt
$env:PYTHONPATH = "$pwd/backend"
backend/.venv/Scripts/python.exe -m app.bootstrap_gat
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

Los archivos de históricos que se quieran analizar deben cargarse desde la
aplicación. El modelo preentrenado y sus datos asociados se descargan mediante
Git LFS. Si el arranque informa que el paquete está ausente o incompleto, ejecutar
`git lfs pull` desde la raíz del repositorio y volver a iniciar la aplicación.

## Uso

1. Abrir la aplicación y entrar en **Evaluación**. Usar **Subir historico** para
   cargar el CSV de referencia y esperar a que termine la preparación de los filtros.
2. En **Estadistico**, explorar las lecturas y sus sospechas. En **Contextual**,
   revisar las relaciones entre sensores. **Analisis y etiquetado** permite
   inspeccionar y anotar los datos.
3. Abrir **Evaluación de flujos** y elegir una plantilla en **Nuevo desde plantilla**.
   Añadir o conectar nodos y seleccionar cada uno para ajustar sus propiedades.
   En el clasificador, mantener **Modelo activo** para usar el modelo preentrenado.
4. Usar **Cargar eventos futuros** para importar lecturas posteriores al histórico.
   También se puede usar **Cargar conjunto etiquetado** para comparar los resultados
   con etiquetas conocidas.
5. Pulsar **Guardar y evaluar** o **Evaluar flujo**. Consultar la ejecución,
   seleccionar un nodo y abrir el detalle de evidencias de cada lectura.

Los CSV de lecturas deben incluir `entity_id` (sensor), `state` (valor o estado)
y `last_changed` (fecha y hora, preferiblemente con zona horaria). Los conjuntos
etiquetados incluyen además `es_sintetica`, `tipo` y `tipo_falla`; sus etiquetas
se utilizan para comprobar los resultados.

El editor admite ramas y conexiones según la señal de cada nodo. El flujo debe
tener una entrada y una salida, y no puede contener ciclos. Durante la evaluación,
cada lectura utiliza solo la información disponible hasta ese instante.

### Nodos disponibles

| Nodo | Funcionamiento |
| --- | --- |
| Entrada | Introduce en el flujo las lecturas del archivo seleccionado. |
| Estadístico FOCuS | Detecta desviaciones respecto del comportamiento histórico. Usa FOCuS sobre residuos en sensores continuos, FOCuS Bernoulli en binarios y reglas para sensores de varios estados. |
| Contextual | Contrasta una lectura con sensores relacionados para comprobar si el contexto respalda o contradice su comportamiento. |
| Clasificador | Usa el modelo GAT-LSTM preentrenado para distinguir comportamiento normal, anomalía ambiental y posible falla del sensor. Puede devolver un resultado indeterminado cuando la evidencia es insuficiente. |
| Unión | Reúne las evidencias de distintas ramas y combina sus señales según el modo «cualquiera» (O) o «todas» (Y). |
| Decisión | Compara la señal recibida con las seleccionadas en su configuración y permite dirigir las lecturas por distintas ramas. |
| Salida | Reúne los resultados finales y conserva el recorrido de evidencias de cada lectura. |

Las señales son `positive` (señal positiva), `negative` (sin señal) y `unknown`
(no evaluable). Las conexiones pueden dejar pasar todas o solo algunas de ellas.
Una sospecha indica que conviene revisar la lectura; el diagnóstico expresa
compatibilidad con un patrón de falla.

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
