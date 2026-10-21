# Simulador ASRS

Simulación de un sistema automatizado de almacenamiento y recuperación (ASRS) con una grúa, una estantería de celdas con canales de varias capas y un punto de entrada/salida (E/S). Compara **políticas de asignación de almacenamiento** sobre múltiples réplicas reproducibles.

## Uso

```bash
pip install -r requirements.txt
python main.py
```

Genera `resultados_simulacion.csv` (separador `;`) con una fila por réplica y política. Los parámetros (tamaño de la estantería, velocidades, ocupación, demanda por clase, políticas, número de réplicas, semilla) están en `asrs/config.py`.

## Herramienta visual

```bash
python visualizar.py --servir
```

Abre en el navegador una herramienta interactiva (servidor local en `127.0.0.1`, sin dependencias nuevas). Se trabaja en tres pasos, que están siempre a la vista en la barra superior junto al botón **Simular**:

1. **Escenario.** Todo lo que se puede ajustar, a la vista y a todo el ancho, en cuatro tarjetas: Almacén (con un esquema a escala de la estantería), Demanda, Políticas y Ejecución. No hay pestañas que escondan parámetros. Si cambias algo después de simular, la barra avisa "Cambios sin simular".
2. **Carrera.** Al terminar de simular se pasa solo a este paso: las políticas ejecutan la misma lista de tareas sobre un mismo reloj, en 3D (Three.js, incluido en `asrs/visualizacion/vendor/`, funciona sin internet). La cámara es la misma en todos los paneles: arrastra para girar, rueda para acercar, doble clic para restablecer, o usa los botones Isométrica, Frente, Lateral y Superior.
3. **Resultados.** Primero la conclusión en una frase (qué política gana, por cuánto y si la diferencia es creíble o los intervalos de confianza se solapan) y después la estadística de las réplicas y la tabla de la réplica vista.

Si recargas la página mientras corre una simulación, se reconecta y sigue su progreso. En el menú ⋯ están restablecer, guardar y cargar el escenario.

**Parámetros editables**
- Estantería: filas, columnas, capas de profundidad, medidas de la celda, posición y centro de referencia de la celda.
- Origen de las tareas: generadas con una demanda por SKU (Hausman o lista propia) o **datos reales** desde un CSV con una tarea por fila y dos columnas, `SKU` y `Task` (A o R; separador y encabezado se detectan solos). Con datos reales la cantidad de SKUs, la demanda, la cantidad de tareas y la mezcla de operaciones salen del archivo, y los SKUs se numeran por frecuencia (el 1 es el más frecuente) conservando su código en la animación. Las políticas planifican (llenado inicial y clases) con el primer porcentaje de la lista que se indique como historial, y después se ejecuta la lista completa, lo que permite medir cuánto afecta que la demanda real cambie respecto de la conocida. Todas las réplicas usan la misma lista y solo cambia el llenado inicial.
- Grúa y E/S: velocidades, aceleraciones y deceleraciones en filas, columnas y canales, tiempo de carga y descarga, separación del riel. Cada eje acelera, avanza a la velocidad máxima y frena (perfil trapezoidal, o triangular en distancias cortas); el tramo xy dura lo que tarda el eje más lento y luego se recorre z. Con aceleración o deceleración 0 no hay límite (velocidad constante). Por defecto son 1 m/s². Ver `asrs/cinematica.py`.
- Ocupación inicial y perfil de inventario (cuántas unidades de cada SKU hay guardadas, ver más abajo).
- Productos (SKU): cantidad (hasta 200), sesgo de la demanda según la curva de Hausman o una lista de demanda propia, con una vista previa de la demanda por SKU.
- Clases de la política basada en clases (dentro de la propia política): la política agrupa los SKUs sola con la demanda definida en la pestaña Demanda (no la modifica) y solo se elige cómo formar las clases. *Misma demanda en cada clase* (de 1 a 8 clases, cada una con la misma parte de la demanda; es lo predeterminado, con 3 clases), *Pareto 80 / 15 / 5* (A con el 80 % de la demanda, B con el 15 % y C con el 5 %) o *a mano* con el porcentaje de SKUs de cada clase. Con los dos primeros no hace falta indicar cuántos SKUs tiene cada clase: salen de la curva de demanda acumulada (`agrupar_skus_por_demanda` en `asrs/politicas/zonificacion.py`), y la vista previa muestra el resultado.
- Lista de tareas: cantidad (automática o fija) y porcentaje de almacenamientos.
- Políticas a comparar: cualquier subconjunto de las cinco.
- Semilla y réplicas para la estadística (hasta 2000). La carrera anima siempre una réplica, la primera.

Los parámetros se validan antes de ejecutar y los errores se muestran junto al formulario. Se pueden guardar y cargar como archivo JSON, y el navegador recuerda los últimos usados.

**Color de los ítems.** Depende de la política y cada panel trae su leyenda: en la política basada en clases es la clase del SKU (A, B, C), y en las demás es la rotación del SKU, en un degradado de amarillo (más rotación) a violeta (menos). Así, en `full_turnover` se ve cómo los SKUs más rotativos ocupan las posiciones más cercanas al E/S sin que aparezcan clases.

**Qué muestra**
- La carrera: todas las políticas ejecutan la misma lista de tareas sobre un mismo reloj. Cada una tiene su estantería y su grúa animadas, una tira con las tareas, un marcador en vivo y una tabla final de la réplica.
- La estadística: promedio con intervalo de confianza del 95 %, desviación, mínimo, máximo y en cuántas réplicas gana cada política (por tiempo total), para la métrica que elijas. Se puede descargar como CSV.

**Cómo se decide si una política es mejor.** El intervalo de confianza de cada promedio usa la t de Student con n − 1 grados de libertad (la desviación se estima con las mismas réplicas; con pocas réplicas la normal daría intervalos demasiado estrechos). Como todas las políticas corren las mismas secuencias en cada réplica, la comparación con la mejor es **pareada**: se calcula la diferencia réplica a réplica y su intervalo de confianza del 95 %. Si ese intervalo incluye el cero, la diferencia no es significativa: el veredicto dice que las políticas "quedan muy parejas" y la tabla la marca con *n. s.* Este criterio es más exigente con los datos que comparar si dos intervalos se solapan, porque descuenta la variación que comparten las políticas (la misma secuencia afecta a todas). Ver `asrs/simulacion/estadistica.py`.

La simulación se ejecuta en Python con el mismo código que `main.py`, así que los resultados coinciden con el CSV para las mismas semillas. Las réplicas de la estadística corren en paralelo y la página muestra su progreso, con opción de cancelar.

Opciones: `--puerto N` (por defecto 8765) y `--sin-abrir` (no abre el navegador).

**Archivo autónomo.** Sin `--servir`, `python visualizar.py --abrir` genera `visualizacion.html`, que reproduce la carrera con la configuración de `asrs/config.py` y no necesita servidor (opciones `--replicas`, `--semilla`, `--salida`). La versión 2D anterior está en `original/plantilla_visualizacion_2d.html`.

Pruebas:

```bash
python -m unittest discover -s tests
```

## Modelo de políticas

Toda combinación de decisiones es una **política de asignación de almacenamiento**. La misma política decide dónde almacenar y dónde reubicar: reubicar es almacenar un ítem que ya está en la estantería, tomando como referencia la celda de ese ítem.

La **regla de recuperación** decide qué unidad del SKU pedido se retira y es la misma para todas las políticas (`texto_regla_recuperacion`, `asrs/politicas/recuperacion.py`):
- *FCFS estricto* (`fcfs`, por defecto): la unidad más antigua del SKU, aunque haya que reubicar lo que tiene delante. Corresponde a un almacén con control de lote o de vencimiento.
- *FCFS accesible* (`fcfs_accesible`): la más antigua entre las que no tienen nada delante; solo si ninguna está libre se retira la más antigua y se reubica. Corresponde a un almacén donde las unidades de un SKU son intercambiables.

Como los canales se llenan desde el fondo, con FCFS estricto la unidad más antigua suele quedar detrás de otra más nueva, y casi todas las recuperaciones obligan a reubicar.

| Clave de configuración | Nombre en resultados | Criterio |
|---|---|---|
| `aleatoria` | `Aleatoria` | Canal elegido al azar |
| `basada_en_clases` | `BasadaEnClases` | La política agrupa los SKUs en clases por demanda (la clase A con los más demandados) y reserva una zona a cada clase; la clase A recibe las celdas más cercanas al E/S |
| `full_turnover` | `FullTurnover` | Se hace una lista de SKUs ordenada por rotación y se empareja con la lista de posiciones ordenada por cercanía al E/S: el SKU más rotativo recibe las posiciones más cercanas, el siguiente las que siguen, y así sucesivamente (Hausman, Graves y Schwarz, 1976). No hay clases |
| `mas_cercana_es` | `MasCercanaAlES` | Celda libre más cercana al E/S |
| `mas_cercana_grua` | `MasCercanaALaGrua` | Celda libre más cercana a la grúa: en reposo al almacenar, la celda del ítem que retira al reubicar |

La cercanía es el tiempo de viaje Chebyshev en el plano xy, `max(x/vx, y/vy)`.

Las zonas se reparten por **canales completos** y el espacio de cada clase o SKU es proporcional a su inventario, con al menos un canal por zona. El orden de las zonas (cuál queda más cerca del E/S) es por demanda.

**Perfil de inventario.** La demanda dice cuántas veces se pide cada SKU; el perfil de inventario dice cuántas unidades suyas hay guardadas. Define el llenado inicial y el tamaño de las zonas (`perfil_inventario` en `asrs/generacion/probabilidades.py`):
- *Lote económico (EOQ)*, por defecto. Cada SKU se repone en lotes de `Q = √(2·D·S/H)` y su inventario promedio es `Q/2`, así que crece con la raíz de la demanda: un SKU que se pide 4 veces más tiene 2 veces más unidades y cada una se pide 2 veces más seguido. Se supone el mismo costo de pedir (S) y de mantener (H) para todos los SKUs, por eso se cancelan al normalizar y no hace falta conocerlos. Como demanda ÷ inventario = √demanda, ordenar por demanda es lo mismo que ordenar por rotación (índice COI de Heskett).
- *Proporcional a la demanda*. Cada unidad guardada se pide con la misma frecuencia, sea del SKU que sea, así que ninguna asignación por rotación puede reducir el recorrido: solo cuenta tener el inventario compacto cerca del E/S.

Durante la lista de tareas cada SKU entra y sale al ritmo de su demanda, así que el perfil inicial se conserva en promedio. Con FCFS, `full_turnover` suele reubicar más que `basada_en_clases`: el ítem más antiguo de un SKU queda al fondo del canal, con ítems del mismo SKU delante.

## Lista de tareas

La lista de tareas solo conoce **SKUs**: cada tarea es un par `(actividad, SKU)`, donde la actividad es almacenar o recuperar. Agrupar los SKUs en clases, si hace falta, es decisión de cada política y no del generador.

- La demanda de cada SKU sigue la curva de Hausman `F(i) = (i/n)^σ` (σ = 1 es uniforme; menor σ concentra la demanda en pocos SKUs) o una lista propia. El SKU 1 es el más demandado.
- Las cantidades son exactas (método del mayor resto): se reparte primero entre almacenar y recuperar según la mezcla y luego entre SKUs según su demanda. El azar solo decide el orden.
- Cada réplica genera dos listas: una de llenado, solo almacenamientos, que deja la estantería en el nivel de ocupación inicial, y la lista de operación que se compara.

**Regla de minimización de varianza** (en todas las políticas): entre las celdas que la política considera equivalentes se elige la libre más profunda, así los canales se llenan desde el fondo. Solo se consideran celdas accesibles, es decir sin ítems delante. Si aún hay empate, se desempata al azar.

Si no hay celda factible al reubicar, el ítem se lleva temporalmente al E/S y se devuelve a la estantería al terminar la operación.

**Recorrido de una recuperación con reubicación.** La grúa va a la estantería, saca cada ítem que obstruye y lo deja en la celda que elige la política. Si la última reubicación fue dentro de la estantería, sigue desde esa celda directo a la celda objetivo, sin volver a la esquina; ese tramo cuenta como tiempo de recuperación. Si la última reubicación dejó el ítem en el E/S, la grúa sale desde ahí. Después retira el ítem, lo entrega en el E/S y vuelve al reposo.

## Arquitectura

```
main.py                         Punto de entrada
asrs/
  config.py                     Parámetros del modelo, con validación y conversión a JSON
  metricas.py                   Conteos y tiempos (Metricas)
  utilidades.py                 Reparto entero por mayor resto
  dominio/                      Entidades físicas
    item.py  estanteria.py  grua.py  punto_es.py
  politicas/
    asignacion.py                 PoliticaAsignacion: base común (candidatas, varianza, desempate)
    aleatoria.py  basada_en_clases.py  full_turnover.py
    mas_cercana_es.py  mas_cercana_grua.py
    zonificacion.py               Zonas por canales completos y agrupación de SKUs en clases
    recuperacion.py               PoliticaRecuperacionFCFS
    restriccion.py                Ocupación y accesibilidad
    registro.py                   Registro y fábrica de políticas
  generacion/
    secuencia.py                  GeneradorSecuencia: lista de tareas por SKU
    probabilidades.py             Demanda por SKU (curva de Hausman)
    datos_reales.py               Lista de tareas desde un CSV (SKU y Task)
  control/
    controlador_grua.py         Ejecuta la secuencia usando las políticas
  simulacion/
    simulacion.py               Réplicas, escenarios y ejecución en paralelo
    estadistica.py              Resumen de varias réplicas (promedio, IC 95 %, victorias)
    resultados.py               Escritura del CSV
  visualizacion/
    generador.py                Graba los eventos de la grúa y prepara los datos de la animación
    servidor.py                 Servidor local: sirve la página y ejecuta simulaciones como trabajos
    plantilla.html              Página con el formulario y la carrera (Three.js incluido en vendor/)
visualizar.py                   Herramienta visual (--servir) o archivo autónomo
tests/                          Pruebas de regresión
original/                       Versiones previas en un solo archivo
```

Dependencias entre capas (solo hacia abajo): `simulacion` → `control` → `politicas` → `dominio` / `metricas`. Para añadir una política: crear una subclase de `PoliticaAsignacion` que implemente `matrizCriterio`, registrarla en `politicas/registro.py` y añadir su clave a `vector_politicas_asignacion`.

## Modelo de réplicas

Cada réplica `i` usa la semilla `escalar_semilla + i` y genera una secuencia de llenado y una de operación. Cada política parte de una estantería vacía, la llena con esa misma secuencia bajo su propia lógica y procesa la secuencia de operación (comparación pareada). Las métricas del llenado se descartan.
