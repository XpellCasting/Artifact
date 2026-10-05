# Dependencias de detección secuencial

`focus-cpt==0.1.10` proporciona FOCuS mediante `Detector(type="univariate")`
y `get_statistics(family="gaussian", theta0=0.0)` sobre residuos estandarizados.
La rama binaria usa `get_statistics(family="bernoulli", theta0=p0)` sobre
reportes válidos 0/1. El primer 60 % de Desarrollo estima p0 con suavizado
Jeffreys y el resto calibra el umbral; nunca se mezclan timestamps entre
esas particiones. Detalles y revisión pendiente: `../docs/BINARY_BERNOULLI.md`.
Documentación: https://github.com/gtromano/unified_focus/tree/main/focus_cpt
Es software GPL-3.0 según su distribución. Su compilación requiere un compilador
C++ y Qhull; Docker instala `g++` y `libqhull-dev`. En Windows se requiere
configurar esas herramientas antes de instalar el paquete desde fuente.
La identidad de calibración incluye la implementación y versión de la librería:
las referencias anteriores se recalibran, sin reutilizar sus umbrales.
La ausencia o inicialización fallida de esta dependencia hace fallar el nodo
**Estadístico FOCuS** explícitamente; nunca se sustituye por EWMA.
