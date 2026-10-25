# Simulador ASRS

Simulación de un sistema automatizado de almacenamiento y recuperación (ASRS) con una grúa, una estantería de celdas con canales de varias capas y un punto de entrada/salida (E/S). Compara **políticas de asignación de almacenamiento** sobre múltiples réplicas reproducibles.

## Uso

```bash
pip install -r requirements.txt
python main.py
```

Genera `resultados_simulacion.csv` (separador `;`) con una fila por réplica, política de asignación y política de recuperación. Los parámetros (tamaño de la estantería, velocidades, ocupación, demanda por clase, políticas, número de réplicas, semilla) están en `asrs/config.py`.

## Instancia por defecto

Los parámetros por defecto (`asrs/config.py`) son la instancia del capítulo 6 del documento de la tesis, tomada de la literatura (Lerher et al. 2010, Lehmann y Hußmann 2023, Gagliardi et al. 2012):

| Parámetro | Valor |
|---|---|
| Estantería | 18 filas × 20 columnas × 2 capas |
| Celda (alto × ancho × fondo) | 1,67 × 1,1 × 1,2 m |
| Velocidades (filas; columnas; canales) | 1; 1,5; 0,7 m/s |
| Aceleración y deceleración | 0,5 m/s² en filas y columnas; velocidad constante en canales |
| Toma o depósito | 3,43 s en el E/S y en la capa delantera; 6,86 s en la trasera |
| E/S | 1 m sobre la base y 1 m antes de la primera columna; riel en la cara frontal |
| SKU y demanda | 180 SKU, curva de Hausman con σ = 0,4 |
| Clases | 3, con el 20 %, 20 % y 60 % de los SKU |
| Tareas por corrida | 100 000, tras un llenado inicial que hace de calentamiento |
| Ocupación | 85 % por defecto; la instancia recorre 5 %, 10 %, …, 95 % (`NIVELES_OCUPACION`) |
| Corridas | 1 por defecto |

Una corrida completa de 100 000 tareas tarda minutos por política. La herramienta visual (`--servir`) arranca con una lista corta (el 10 % de las celdas) para que la primera ejecución sea ágil, y las pruebas usan 72 tareas.

## Herramienta visual

```bash
python visualizar.py --servir
```

Abre en el navegador una herramienta interactiva (servidor local en `127.0.0.1`, sin dependencias nuevas). Se trabaja en tres pasos, que están siempre a la vista en la barra superior junto al botón **Simular**:

1. **Escenario.** Todo lo que se puede ajustar, a la vista y a todo el ancho, en cuatro tarjetas: Almacén (con un esquema a escala de la estantería y del punto E/S), Demanda, Políticas y Ejecución, con un resumen siempre visible. No hay pestañas que escondan parámetros. Si cambias algo después de simular, la barra avisa "Cambios sin simular".
2. **Carrera.** Al terminar de simular se pasa solo a este paso: las políticas ejecutan la misma lista de tareas sobre un mismo reloj, en 3D (Three.js, incluido en `asrs/visualizacion/vendor/`, funciona sin internet). La cámara es la misma en todos los paneles: arrastra para girar, rueda para acercar, doble clic para restablecer, o usa los botones ISO, FRT, LAT y SUP (isométrica, frente, lateral y superior). Cada visor trae cotas del largo y la altura de la estantería y un triedro con los ejes de columnas, filas y profundidad. El punto E/S se dibuja como una placa frente a la grúa, con una zona de espera al lado para los ítems que aguardan su turno (el que llega mientras otro sale y los que se reubican temporalmente). Cuando la estantería es grande, un recuadro de detalle en la esquina del visor sigue a la grúa y muestra ampliados el carro, la horquilla y la carga; se oculta con el botón GRÚA o la tecla D. Con tiempo de carga y descarga 0 la grúa toma y entrega en el mismo instante, así que esas transferencias se dibujan con una animación breve que no altera los tiempos; con un tiempo mayor que 0 duran lo que dice el modelo. Con listas muy largas la carrera anima solo las primeras tareas y lo avisa.
3. **Resultados.** Primero la conclusión en una frase (qué política gana, por cuánto y si la diferencia es creíble según la comparación pareada de las réplicas) y después la estadística de las réplicas y la tabla de la carrera.

Si recargas la página mientras corre una simulación, se reconecta y sigue su progreso. En el menú ⋯ están restablecer, guardar y cargar el escenario; el botón de tema alterna entre automático, claro y oscuro.

**Parámetros editables**
- Estantería: filas, columnas, capas de profundidad, medidas de la celda, posición y centro de referencia de la celda.
- Origen de las tareas: generadas con una demanda por SKU (Hausman o lista propia) o **datos reales** desde un CSV con una tarea por fila y dos columnas, `SKU` y `Task` (A o R; separador y encabezado se detectan solos). Con datos reales la cantidad de SKUs, la demanda, la cantidad de tareas y la mezcla de operaciones salen del archivo, y los SKUs se numeran por frecuencia (el 1 es el más frecuente) conservando su código en la animación. Las políticas planifican (llenado inicial y clases) con el primer porcentaje de la lista que se indique como historial, y después se ejecuta la lista completa, lo que permite medir cuánto afecta que la demanda real cambie respecto de la conocida. Todas las réplicas usan la misma lista y solo cambia el llenado inicial.
- Grúa y E/S: velocidades, aceleraciones y deceleraciones en filas, columnas y canales, tiempo de carga y descarga, separación del riel. Cada eje acelera, avanza a la velocidad máxima y frena (perfil trapezoidal, o triangular en distancias cortas); el tramo xy dura lo que tarda el eje más lento y luego se recorre z. Con aceleración o deceleración 0 no hay límite (velocidad constante). Por defecto son 1 m/s². Ver `asrs/cinematica.py`.
- Ocupación inicial y perfil de inventario (cuántas unidades de cada SKU hay guardadas, ver más abajo).
- Productos (SKU): cantidad (hasta 200), sesgo de la demanda según la curva de Hausman o una lista de demanda propia, con una vista previa de la demanda por SKU.
- Clases de la política basada en clases (dentro de la propia política): la política agrupa los SKUs sola con la demanda definida en la pestaña Demanda (no la modifica) y solo se elige cómo formar las clases. *Misma demanda en cada clase* (de 1 a 8 clases, cada una con la misma parte de la demanda; es lo predeterminado, con 3 clases), *Pareto 80 / 15 / 5* (A con el 80 % de la demanda, B con el 15 % y C con el 5 %) o *a mano* con el porcentaje de SKUs de cada clase. Con los dos primeros no hace falta indicar cuántos SKUs tiene cada clase: salen de la curva de demanda acumulada (`agrupar_skus_por_demanda` en `asrs/politicas/zonificacion.py`), y la vista previa muestra el resultado.
- Lista de tareas: cantidad (automática o fija) y porcentaje de almacenamientos.
- Políticas a comparar: cualquier subconjunto de las cinco de asignación, combinadas con FCFS estricto, FCFS entre accesibles o las dos.
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

La **recuperación** decide qué unidad del SKU pedido se retira (`vector_politicas_recuperacion`, `asrs/politicas/recuperacion.py`). Tiene dos variantes y cada política de asignación se ejecuta con las que se elijan; por defecto solo FCFS estricto:

- **FCFS estricto** (`fcfs`): sale la unidad más antigua del SKU aunque tenga ítems delante, que se reubican. Representa un orden de salida obligatorio, como en un almacén con control de lote o de vencimiento.
- **FCFS entre accesibles** (`fcfs_accesible`): sale la más antigua entre las unidades del SKU que no tienen nada delante; solo si todas están bloqueadas se recurre a la más antigua de todas. Representa un almacén donde las unidades de un SKU son intercambiables.

Como los canales se llenan desde el fondo, con FCFS estricto la unidad más antigua suele quedar detrás de otra más nueva, y casi todas las recuperaciones obligan a reubicar. Las dos variantes de una misma política de asignación usan la misma semilla, así la diferencia entre ellas es pareada. Después de reubicar, la grúa va directo desde la celda donde dejó el último ítem hasta la celda objetivo; ese tramo cuenta en el tiempo de recuperación. Las configuraciones guardadas con la regla única anterior (`texto_regla_recuperacion`) siguen cargando.

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

Durante la lista de tareas cada SKU entra y sale al ritmo de su demanda, así que el perfil inicial se conserva en promedio. Con FCFS estricto, `full_turnover` suele reubicar más que `basada_en_clases`: el ítem más antiguo de un SKU queda al fondo del canal, con ítems del mismo SKU delante.

### Supuestos que conviene tener presentes

- **La cercanía ignora la profundidad.** Las políticas comparan celdas por el tiempo en filas y columnas, sin contar el recorrido de la horquilla dentro del canal. Entre celdas igual de cercanas se elige la del fondo (minimización de varianza), aunque sea la más lejana en profundidad: así los canales se llenan desde atrás y no quedan huecos detrás de un ítem.
- **`mas_cercana_grua` y `mas_cercana_es` almacenan igual.** Cada tarea termina con la grúa en reposo, que está en el E/S, así que al almacenar la celda más cercana a la grúa es la más cercana al E/S. Solo se diferencian al reubicar; si casi no hay reubicaciones (por ejemplo con FCFS entre accesibles), sus resultados prácticamente coinciden.
- **Geometría.** El punto E/S es el origen de coordenadas y la posición de la estantería indica dónde está respecto de él. El centro de celda (0,5 por defecto en los tres ejes) indica, a partir de la posición de la estantería, en qué punto de cada celda se mide su posición: 0 es el borde inicial y 0,5 el centro. No mueve la estantería ni el E/S, que sigue en el origen; solo cambia el punto de cada celda al que va la grúa y, con él, la distancia desde el E/S y hasta dónde entra la horquilla en el canal. El riel queda delante de la cara frontal de la estantería, a la distancia de separación indicada, aunque la estantería se desplace en profundidad. La grúa va directo de un punto a otro por el pasillo, sin pasar por la esquina de la estantería.
- **Conteo de reubicaciones.** `Cant_Reubicaciones` cuenta ítems movidos. Las columnas de reubicaciones durante el almacenamiento o la recuperación (y sus desgloses regulares y predeterminadas) cuentan tareas: con canales de 3 capas o más una tarea puede mover varios ítems.

## Lista de tareas

La lista de tareas solo conoce **SKUs**: cada tarea es un par `(actividad, SKU)`, donde la actividad es almacenar o recuperar. Agrupar los SKUs en clases, si hace falta, es decisión de cada política y no del generador.

- La demanda de cada SKU sigue la curva de Hausman `F(i) = (i/n)^σ` (σ = 1 es uniforme; menor σ concentra la demanda en pocos SKUs) o una lista propia. El SKU 1 es el más demandado.
- Las cantidades son exactas (método del mayor resto): se reparte primero entre almacenar y recuperar según la mezcla y luego entre SKUs según su demanda. El azar solo decide el orden.
- Cada réplica genera dos listas: una de llenado, solo almacenamientos, que deja la estantería en el nivel de ocupación inicial, y la lista de operación que se compara.

**Regla de minimización de varianza** (en todas las políticas): entre las celdas que la política considera equivalentes se elige la libre más profunda, así los canales se llenan desde el fondo. Solo se consideran celdas accesibles, es decir sin ítems delante. Si aún hay empate, se desempata al azar.

Si no hay celda factible al reubicar, el ítem se lleva temporalmente al E/S y se devuelve a la estantería al terminar la operación.

**Recorrido de una recuperación con reubicación.** La grúa va a la estantería, saca cada ítem que obstruye y lo deja en la celda que elige la política. Si la última reubicación fue dentro de la estantería, sigue desde esa celda directo a la celda objetivo; ese tramo cuenta como tiempo de recuperación. Si la última reubicación dejó el ítem en el E/S, la grúa sale desde ahí. Después retira el ítem, lo entrega en el E/S y vuelve al reposo.

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
    recuperacion.py               PoliticaRecuperacionFCFS (estricto) y PoliticaRecuperacionFCFSAccesible
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
