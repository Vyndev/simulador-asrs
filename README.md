# Simulador ASRS

Simulación de un sistema automatizado de almacenamiento y recuperación (ASRS) con una grúa, una estantería de celdas con canales de varias capas y un punto de entrada/salida (E/S). Compara **políticas de asignación de almacenamiento** sobre múltiples réplicas reproducibles.

## Uso

```bash
pip install -r requirements.txt
python main.py
```

Genera `resultados_simulacion.csv` (separador `;`) con una fila por réplica y política. Los parámetros (tamaño de la estantería, velocidades, ocupación, demanda por clase, políticas, número de réplicas, semilla) están en `asrs/config.py`.

Visualización (carrera de políticas, como los visualizadores de algoritmos de ordenamiento):

```bash
python visualizar.py --abrir
```

Genera `visualizacion.html`, una página autónoma en la que todas las políticas ejecutan la misma lista de tareas sobre un mismo reloj de simulación. Cada política tiene su estantería y su grúa animadas en vista isométrica (fila 1 y columna 1 abajo a la izquierda, E/S como plataforma en el pasillo), una tira con las tareas, un marcador en vivo y una tabla final. La versión 2D anterior está en `original/plantilla_visualizacion_2d.html`. Opciones: `--replicas N` (réplicas incluidas, por defecto 3), `--semilla S` (semilla de la primera) y `--salida archivo.html`. Usa las mismas semillas que la simulación principal, así que los tiempos coinciden con el CSV.

Pruebas:

```bash
python -m unittest discover -s tests
```

## Modelo de políticas

Toda combinación de decisiones es una **política de asignación de almacenamiento**. La misma política decide dónde almacenar y dónde reubicar: reubicar es almacenar un ítem que ya está en la estantería, tomando como referencia la celda de ese ítem. La recuperación es siempre FCFS (se retira el ítem más antiguo de la clase solicitada).

| Clave de configuración | Nombre en resultados | Criterio |
|---|---|---|
| `aleatoria` | `Aleatoria` | Canal elegido al azar |
| `basada_en_clases` | `BasadaEnClases` | Zonas fijas por clase, la clase 1 recibe las celdas más cercanas al E/S, la 2 las siguientes, etc. |
| `full_turnover` | `FullTurnover` | Zonas fijas por clase, pero las clases se ordenan de mayor a menor demanda y la de mayor demanda recibe las ubicaciones más cercanas (Hausman, Graves y Schwarz, 1976) |
| `mas_cercana_es` | `MasCercanaAlES` | Celda libre más cercana al E/S |
| `mas_cercana_grua` | `MasCercanaALaGrua` | Celda libre más cercana a la grúa: en reposo al almacenar, la celda del ítem que retira al reubicar |

La cercanía es el tiempo de viaje Chebyshev en el plano xy, `max(x/vx, y/vy)`.

**Regla de minimización de varianza** (en todas las políticas): entre las celdas que la política considera equivalentes se elige la libre más profunda, así los canales se llenan desde el fondo. Solo se consideran celdas accesibles, es decir sin ítems delante. Si aún hay empate, se desempata al azar.

Si no hay celda factible al reubicar, el ítem se lleva temporalmente al E/S y se devuelve a la estantería al terminar la operación.

## Arquitectura

```
main.py                         Punto de entrada
asrs/
  config.py                     Parámetros del modelo
  metricas.py                   Conteos y tiempos (Metricas)
  dominio/                      Entidades físicas
    item.py  estanteria.py  grua.py  punto_es.py
  politicas/
    asignacion.py                 PoliticaAsignacion: base común (candidatas, varianza, desempate)
    aleatoria.py  basada_en_clases.py  full_turnover.py
    mas_cercana_es.py  mas_cercana_grua.py
    zonificacion.py               Construcción de zonas por clase
    recuperacion.py               PoliticaRecuperacionFCFS
    restriccion.py                Ocupación y accesibilidad
    registro.py                   Registro y fábrica de políticas
  generacion/
    secuencia.py                  GeneradorSecuencia
    probabilidades.py             Probabilidades acumuladas (Hausman), sin uso actual
  control/
    controlador_grua.py         Ejecuta la secuencia usando las políticas
  simulacion/
    simulacion.py               Réplicas, escenarios y ejecución en paralelo
    resultados.py               Escritura del CSV
  visualizacion/
    generador.py                Graba los eventos de la grúa y escribe el HTML
    plantilla.html              Página de la carrera (JavaScript, sin dependencias)
visualizar.py                   Genera visualizacion.html
tests/                          Pruebas de regresión
original/                       Versiones previas en un solo archivo
```

Dependencias entre capas (solo hacia abajo): `simulacion` → `control` → `politicas` → `dominio` / `metricas`. Para añadir una política: crear una subclase de `PoliticaAsignacion` que implemente `matrizCriterio`, registrarla en `politicas/registro.py` y añadir su clave a `vector_politicas_asignacion`.

## Modelo de réplicas

Cada réplica `i` usa la semilla `escalar_semilla + i` y genera una secuencia de llenado y una de operación. Cada política parte de una estantería vacía, la llena con esa misma secuencia bajo su propia lógica y procesa la secuencia de operación (comparación pareada). Las métricas del llenado se descartan.
