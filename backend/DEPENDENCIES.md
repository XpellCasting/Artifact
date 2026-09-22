# Dependencias de detección secuencial

`changepoint-online==1.2.1` proporciona el adaptador FOCuS mediante
`Focus(Gaussian(loc=0))`. Es software GPL-3.0-or-later según su distribución.
La ausencia o inicialización fallida de esta dependencia hace fallar el nodo
**Estadístico FOCuS** explícitamente; nunca se sustituye por EWMA.
