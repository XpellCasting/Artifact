# Dataset y modelo GAT-LSTM reutilizables

El snapshot contiene el modelo activo `669501ff239c48eaaeafce499bdb8d43` y su
dataset `3c145354dcda4cefb68f1811a21ef488`: 68.258 ventanas de entrenamiento y
17.635 de ajuste. Conserva arrays, metadatos, configuracion, grafo,
preprocesador, pesos, evaluaciones y seleccion activa. El modelo mantiene su
estado experimental original; el empaquetado no lo valida ni lo reentrena.

`snapshot.tar.gz` se versiona mediante Git LFS. `manifest.json` registra los
archivos y sus SHA-256. Descargar con `git lfs pull` antes de iniciar Docker.
Al publicar un commit, `git push` debe transferir tambien el objeto LFS.

Docker restaura los archivos faltantes en `data/review/diagnostics/` antes de
iniciar la API. Conserva archivos existentes y la seleccion activa local.
No incluye otros datasets, modelos anteriores, trabajos ni caches.

Restauracion manual desde la raiz del repositorio:

```powershell
backend/.venv/Scripts/python.exe backend/app/bootstrap_gat.py
```

Para actualizar el snapshot con el modelo activo local, detener primero los
entrenamientos y ejecutar:

```powershell
backend/.venv/Scripts/python.exe scripts/package_gat_lstm.py
git add artifacts/gat_lstm
```

La restauracion no necesita los CSV originales. Para analizar nuevas lecturas
hay que cargar el historico correspondiente desde la aplicacion.
