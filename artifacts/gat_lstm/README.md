# Dataset y modelo GAT-LSTM reutilizables

El snapshot contiene el modelo activo `669501ff239c48eaaeafce499bdb8d43` y su
dataset `3c145354dcda4cefb68f1811a21ef488`: 68.258 ventanas de entrenamiento y
17.635 de ajuste. Conserva arrays, metadatos, configuracion, grafo,
preprocesador, pesos, evaluaciones y seleccion activa. El modelo mantiene su
estado experimental original; el empaquetado no lo valida ni lo reentrena.

`snapshot.tar.gz` se versiona mediante Git LFS. `manifest.json` registra los
archivos y sus SHA-256. Descargar con `git lfs pull` si se desea restaurar
este paquete histórico. Para la instalación con el modelo C, seguir la
[guía de instalación](../../README.md#instalación).
Al publicar un commit, `git push` debe transferir tambien el objeto LFS.

La restauración manual distribuye los archivos entre `models/gat_lstm/`,
`data/gat_lstm/` y `storage/jobs/` según su tipo. Conserva los archivos
existentes y la selección activa local.
No incluye otros datasets, modelos anteriores, trabajos ni caches.

Restauracion manual desde la raiz del repositorio:

```powershell
$env:PYTHONPATH = "$pwd/backend"
backend/.venv/Scripts/python.exe -m app.bootstrap_gat
```

Para actualizar el snapshot con el modelo activo local, detener primero los
entrenamientos y ejecutar:

```powershell
backend/.venv/Scripts/python.exe scripts/package_gat_lstm.py
git add artifacts/gat_lstm
```

La restauracion no necesita los CSV originales. Para analizar nuevas lecturas
hay que cargar el historico correspondiente desde la aplicacion.
