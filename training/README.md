# Entrenar GAT-LSTM en un ordenador con CUDA

El entrenamiento se ejecuta desde consola, reutiliza el modelo y el procesamiento de Artifact, y guarda un candidato sin activarlo. La aplicación puede usar después sus pesos en CPU. No es necesario iniciar el frontend ni el servidor para entrenar.

## 1. Preparar el ordenador NVIDIA

Instala Git, Git LFS, Python 3.12 de 64 bits y el controlador NVIDIA compatible con tu GPU. Comprueba el controlador con `nvidia-smi`.

Después de publicar los cambios en GitHub, ejecuta en PowerShell:

```powershell
git clone https://github.com/XpellCasting/Artifact.git
cd Artifact
git lfs install
git lfs pull
python -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install --upgrade pip
.\.venv-cuda\Scripts\python.exe -m pip install -r backend/requirements-gat-training.txt
```

Si ya clonaste el repositorio, usa `git pull` y `git lfs pull` dentro de esa copia. Git transporta el código y el snapshot publicado; los datasets locales nuevos, checkpoints y candidatos todavía no publicados se trasladan con el exportador descrito más abajo.

Instala **PyTorch con CUDA** siguiendo el [selector oficial de PyTorch](https://pytorch.org/get-started/locally/): elige tu sistema, Pip, Python y una variante CUDA compatible con el controlador. Copia el comando completo que muestra la página y sustituye únicamente su `pip` o `pip3` inicial por `.\.venv-cuda\Scripts\python.exe -m pip`. Conserva `install`, los nombres de paquetes (incluido `torch`) y los argumentos `--index-url` del comando generado. El fragmento que termina en `pip install` no es un comando completo y por sí solo produce el error «You must give at least one requirement to install». La versión de referencia del proyecto es `torch==2.14.0`; busca esa versión en las [instrucciones oficiales por versión](https://pytorch.org/get-started/previous-versions/) si el selector ofrece otra. No se fija aquí un índice `cuXXX` porque depende de la GPU y del controlador del equipo nuevo.

El archivo `requirements-gat-training.txt` contiene las dependencias mínimas para entrenar desde consola. No instala el servidor ni los demás clasificadores de Artifact. Instala después las dependencias neuronales restantes:

```powershell
.\.venv-cuda\Scripts\python.exe -m pip install -r backend/requirements-gat-cuda.txt
.\.venv-cuda\Scripts\python.exe -m pip check
.\.venv-cuda\Scripts\python.exe scripts/gat_lstm_check.py --device cuda --require-cuda --output storage/cuda-check.json
```

`requirements-gat-cuda.txt` conserva la versión de PyTorch CUDA que acabas de elegir y fija `torch-geometric==2.8.0.post1`. No instales `requirements-gat-docker.txt` en este entorno: está destinado a CPU. La instalación mínima de PyG permite este modelo sin extensiones adicionales; si instalas extensiones como `torch-scatter`, sus ruedas deben coincidir con las versiones exactas de PyTorch y CUDA, según la [documentación oficial de PyG](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html).

El comprobador informa versiones, GPU, memoria libre y total, y ejecuta forward, backward y un paso del optimizador con ambas cabezas del GAT-LSTM. También guarda los pesos en memoria y verifica que se recargan y predicen en CPU. Termina con código 0 cuando todo funciona; con código 1 e instrucciones cuando falla. `--device cuda:1` permite elegir otra GPU visible. `--device auto` usa CUDA si está disponible y CPU en caso contrario; `--require-cuda` impide que una comprobación del equipo NVIDIA pase solamente en CPU.

En Linux crea el entorno con `python3 -m venv .venv-cuda` y sustituye `.\.venv-cuda\Scripts\python.exe` por `.venv-cuda/bin/python` en los comandos. Para el bootstrap usa `PYTHONPATH=backend .venv-cuda/bin/python -m app.bootstrap_gat`.

## 2. Preparar y trasladar un dataset compatible

En el ordenador actual, prepara un dataset con la versión actual de Artifact, desde el flujo GAT-LSTM de la aplicación, usando los históricos y las anotaciones seleccionadas. El contrato requerido es `gat-lstm-segments-v1`, con arquitectura `gat_lstm_v3_segments_multiscale`. Conserva su identificador para los siguientes comandos.

El snapshot publicado actualmente en Git LFS contiene el dataset histórico `3c145354dcda4cefb68f1811a21ef488`, con contrato anterior `gat-lstm-entities-v1` y arquitectura `gat_lstm_v2_shared_type`. **Ese dataset y sus pesos no se pueden usar para entrenar esta arquitectura por segmentos.** Hay que preparar un dataset nuevo; cambiar su etiqueta de versión no convierte los datos ni el modelo.

El preparador acepta varios CSV etiquetados de Desarrollo y genera un conjunto con un solo `dataset_id` para entrenar un único modelo. Cada archivo conserva su hash, origen y episodios; sus ventanas se construyen por separado. Los identificadores de sensor deben ser distintos entre archivos y su unión no puede superar `max_nodes` (32). Se rechazan CSV repetidos, entidades compartidas y fechas fuera de Desarrollo. No es necesario concatenar ni renombrar los dos CSV real/sintético suministrados: ya usan 14 identificadores diferentes cada uno.

### Preparación directa desde los dos CSV

Puedes preparar los datos sin iniciar la aplicación. Copia los archivos al ordenador CUDA como `storage/gat-mixto/inputs/real.csv` y `storage/gat-mixto/inputs/sintetico.csv`. El paquete local `storage/gat-mixto/entradas-mixto.zip`, si está disponible, contiene esa estructura: extráelo en la raíz de Artifact. Los CSV no se publican en Git.

Después de actualizar el código e instalar el entorno CUDA, ejecuta desde PowerShell:

```powershell
.\scripts\train_gat_lstm_mixto.ps1
```

El script comprueba la GPU, prepara ambos CSV y entrena un único candidato. Usa `training/config-mixto.json` para declarar los valores numéricos/categóricos de los 28 sensores, Desarrollo del 1 de febrero al 21 de mayo de 2026 (fin exclusivo) y la separación cronológica 80/20 del proyecto. No requiere escribir `ID_DATASET`: lo obtiene automáticamente. El resultado queda en `storage/checkpoints/mixto-01/result.json`. La preparación de secuencias usa CPU; el ajuste de pesos usa CUDA y guarda checkpoints al terminar cada época.

Si se interrumpe después de terminar la preparación:

```powershell
.\scripts\train_gat_lstm_mixto.ps1 -Resume
```

Si se interrumpe durante la preparación, vuelve a ejecutar sin `-Resume`. El preparador publica solo secuencias completas. Para otro experimento usa `-RunDir storage/checkpoints/mixto-02`; `-BatchSize`, `-MaxEpochs` y `-Patience` permiten cambiar parámetros de una ejecución nueva. Al reanudar conserva los parámetros usados originalmente.

Los comandos equivalentes, para configurar rutas o límites distintos, son:

```powershell
.\.venv-cuda\Scripts\python.exe scripts/gat_lstm_prepare.py --csv storage/gat-mixto/inputs/real.csv --csv storage/gat-mixto/inputs/sintetico.csv --config training/config-mixto.json --development-start 2026-02-01T00:00:00Z --development-end 2026-05-21T00:00:00Z --result storage/gat-mixto/dataset.json
$dataset = (Get-Content storage/gat-mixto/dataset.json -Raw | ConvertFrom-Json).dataset_id
.\.venv-cuda\Scripts\python.exe scripts/gat_lstm_train.py $dataset --device cuda --checkpoint-dir storage/checkpoints/mixto-01 --result storage/checkpoints/mixto-01/result.json
```

Añade `--inspect-only` al preparador para comprobar columnas, fechas, entidades y etiquetas sin generar secuencias. El resultado entrenado incluye `metrics.dev_metrics_by_source` para revisar cada histórico por separado con la misma política de decisión. Estas métricas siguen siendo de ajuste `dev_tune`.

Para trasladar el dataset nuevo, exporta en el ordenador actual usando el Python que ya ejecuta Artifact:

```powershell
python scripts/gat_lstm_transfer.py export --dataset-id ID_DATASET --output storage/exports/dataset-cuda.tar.gz
```

Copia ese archivo al ordenador CUDA e impórtalo:

```powershell
.\.venv-cuda\Scripts\python.exe scripts/gat_lstm_transfer.py import storage/exports/dataset-cuda.tar.gz
$dataset = "ID_DATASET"
```

El paquete contiene el manifiesto y las secuencias preparadas, con comprobaciones de integridad. La importación adapta las rutas al equipo de destino. No copies manualmente rutas absolutas del ordenador anterior.

Si necesitas restaurar el snapshot histórico para conservarlo, `git lfs pull` descarga sus archivos y el siguiente comando comprueba su integridad y restaura lo que falta. Esta operación no prepara un dataset nuevo ni convierte la arquitectura anterior:

```powershell
$env:PYTHONPATH = "backend"
.\.venv-cuda\Scripts\python.exe -m app.bootstrap_gat
Remove-Item Env:\PYTHONPATH
```

La restauración inicial incluye los archivos de selección activa del snapshot; conserva los existentes en una instalación previa. Para el entrenamiento actual utiliza siempre el identificador del dataset por segmentos que acabas de importar.

## 3. Entrenar y reanudar

Ejecuta desde la raíz del repositorio. Puedes cambiar el nombre `run-01` para cada experimento:

```powershell
.\.venv-cuda\Scripts\python.exe scripts/gat_lstm_train.py $dataset --device cuda --checkpoint-dir storage/checkpoints/run-01 --result storage/checkpoints/run-01/result.json
```

La configuración del dataset preparado se reutiliza. Para modificar parámetros de entrenamiento, crea un JSON con los valores que quieras cambiar, por ejemplo `storage/train-config.json`:

```json
{
  "max_epochs": 50,
  "patience": 8,
  "batch_size": 16,
  "learning_rate": 0.001,
  "seed": 42
}
```

Añade `--config storage/train-config.json` al comando. Conserva la representación y las particiones del dataset; un cambio del procesamiento de datos requiere preparar otro dataset. El candidato guarda configuración, métricas y datos de la ejecución para poder identificar el entrenamiento. `--device cpu` permite una prueba en el ordenador actual. Solicitar `cuda` o `cuda:N` explícitamente falla si ese dispositivo no está disponible.

Si el proceso se interrumpe, repite el mismo comando y añade `--resume`:

```powershell
.\.venv-cuda\Scripts\python.exe scripts/gat_lstm_train.py $dataset --device cuda --checkpoint-dir storage/checkpoints/run-01 --resume --result storage/checkpoints/run-01/result.json
```

Si usaste `--config`, vuelve a incluir ese mismo archivo. Los checkpoints guardan el progreso, el estado del modelo y del optimizador, y el estado aleatorio para continuar desde una época completada. Una interrupción dentro de una época puede requerir repetir esa época. Usa otro directorio para un entrenamiento nuevo. Conserva el mismo código, dataset y configuración para reanudar el mismo experimento; no actualices dependencias a mitad de la ejecución.

## 4. Revisar y devolver el candidato

El entrenamiento imprime el identificador del candidato y lo guarda en el JSON indicado por `--result`. Revisa las métricas de ajuste `dev_tune` y el historial de entrenamiento de ese resultado; esta partición sirve para elegir el candidato y su política de decisión. Usa su identificador para exportarlo:

```powershell
$model = "ID_MODELO_DEVUELTO"
.\.venv-cuda\Scripts\python.exe scripts/gat_lstm_transfer.py export --dataset-id $dataset --model-id $model --output storage/exports/candidate-cuda.tar.gz
```

El paquete del candidato incluye pesos en CPU, preprocesador, grafo, configuración, política de decisión y dataset preparado. Copia el paquete al ordenador que ejecuta Artifact e impórtalo usando su entorno Python:

```powershell
python scripts/gat_lstm_transfer.py import storage/exports/candidate-cuda.tar.gz
```

La importación registra el modelo y conserva sus métricas y evaluaciones; no lo selecciona ni cambia la activación local. Revisa sus resultados y actívalo desde el flujo habitual de Artifact cuando decidas utilizarlo. La evaluación del clasificador sobre las particiones históricas `validation` y `test` está deshabilitada por el contrato actual: quedan reservadas para el flujo de reproducción de eventos futuros. La validación operacional se realiza con los flujos de la aplicación que ya soportan ese análisis; las métricas `dev_tune` no sustituyen esa validación.

Si trabajas con una ubicación de revisión personalizada, añade el mismo `--review-dir RUTA` a los comandos de entrenamiento y transferencia en ese equipo. Los pesos y checkpoints se guardan fuera de Git, dentro de las rutas locales de datos/modelos o `storage/`; el código y esta guía sí se publican en GitHub.

## Problemas frecuentes

| Síntoma | Acción |
| --- | --- |
| `CUDA no disponible` / `torch.version.cuda` es `null` | Comprueba `nvidia-smi`, el Python del entorno y la instalación de una rueda PyTorch CUDA con el selector oficial. |
| `invalid device ordinal` | Selecciona un índice menor que `cuda_device_count`; `CUDA_VISIBLE_DEVICES` puede cambiar los índices visibles. |
| `no kernel image` / arquitectura CUDA incompatible | Usa una rueda PyTorch que soporte la arquitectura de la GPU y un controlador compatible. |
| `CUDA out of memory` | Cierra otros procesos GPU y reduce `batch_size` en una ejecución nueva. No cambies la configuración de un checkpoint existente para continuar otro experimento. |
| PyG: DLL ausente / `undefined symbol` | Revisa que PyTorch y cualquier extensión compilada tengan la misma variante CUDA; empieza con la instalación mínima de esta guía. |
| `focus-cpt` solicita `nmake` o compilador C++ | Para entrenar GAT-LSTM usa `requirements-gat-training.txt`; `focus-cpt` pertenece a la aplicación completa y no es necesario para esta consola. |
| Snapshot ausente o incompleto | Ejecuta `git lfs pull` y vuelve a lanzar el bootstrap. |
| Falla al reanudar por dataset o configuración | Usa los mismos datos, configuración y código del checkpoint, o comienza otro directorio de ejecución. |

Una comprobación pequeña no mide el consumo del dataset completo ni garantiza idénticas métricas bit a bit entre GPU, controladores y versiones diferentes. El primer paso en el ordenador nuevo es ejecutar el comprobador con `--require-cuda`.
