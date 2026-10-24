# Simulador ASRS

Simulación de un sistema automatizado de almacenamiento y recuperación (ASRS) con una grúa, una estantería de celdas con canales de varias capas y un punto de entrada/salida (E/S). Compara **políticas de asignación de almacenamiento** sobre múltiples réplicas reproducibles.

## Uso

```bash
pip install -r requirements.txt
python main.py
```

Genera `resultados_simulacion.csv` (separador `;`) con una fila por réplica, política de asignación y política de recuperación. Los parámetros (tamaño de la estantería, velocidades, ocupación, demanda por clase, políticas, número de réplicas, semilla) están en `asrs/config.py`.

## Herramienta visual

```bash
python visualizar.py --servir
```

Abre en el navegador una herramienta interactiva (servidor local en `127.0.0.1`, sin dependencias nuevas). A la izquierda están **todos los parámetros de la simulación**, organizados en cuatro pestañas (Almacén, Demanda, Políticas y Ejecución) con un resumen siempre visible; a la derecha, la carrera de políticas en 3D (Three.js, incluido en `asrs/visualizacion/vendor/`, funciona sin internet) y la estadística. La cámara es la misma en todos los paneles: arrastra para girar, rueda para acercar, doble clic para restablecer, o usa los botones ISO, FRT, LAT y SUP (isométrica, frente, lateral y superior). Cada visor trae cotas del largo y la altura de la estantería y un triedro con los ejes de columnas, filas y profundidad. El punto E/S se dibuja como una placa frente a la grúa, con una zona de espera al lado para los ítems que aguardan su turno (el que llega mientras otro sale y los que se reubican temporalmente). Cuando la estantería es grande, un recuadro de detalle en la esquina del visor sigue a la grúa y muestra ampliados el carro, la horquilla y la carga; se oculta con el botón GRÚA de la barra o la tecla D. Con tiempo de carga y descarga 0 la grúa toma y entrega en el mismo instante, así que esas transferencias se dibujan con una animación breve que no altera los tiempos; con un tiempo mayor que 0 duran lo que dice el modelo. El botón de tema de la barra superior alterna entre automático, claro y oscuro. Si recargas la página mientras corre una simulación, se reconecta y sigue su progreso.

**Parámetros editables**
- Estantería: filas, columnas, capas de profundidad, medidas de la celda, posición y centro de referencia de la celda.
- Origen de las tareas: generadas con una demanda por SKU (Hausman o lista propia) o **datos reales** desde un CSV con una tarea por fila y dos columnas, `SKU` y `Task` (A o R; separador y encabezado se detectan solos). Con datos reales la cantidad de SKUs, la demanda, la cantidad de tareas y la mezcla de operaciones salen del archivo, y los SKUs se numeran por frecuencia (el 1 es el más frecuente) conservando su código en la animación. Las políticas planifican (llenado inicial y clases) con el primer porcentaje de la lista que se indique como historial, y después se ejecuta la lista completa, lo que permite medir cuánto afecta que la demanda real cambie respecto de la conocida. Todas las réplicas usan la misma lista y solo cambia el llenado inicial.
- Grúa y E/S: velocidades, aceleraciones y deceleraciones en filas, columnas y canales, tiempo de carga y descarga, separación del riel. Cada eje acelera, avanza a la velocidad máxima y frena (perfil trapezoidal, o triangular en distancias cortas); el tramo xy dura lo que tarda el eje más lento y luego se recorre z. Con aceleración o deceleración 0 no hay límite (velocidad constante). Por defecto son 1 m/s². Ver `asrs/cinematica.py`.
- Ocupación inicial.
- Productos (SKU): cantidad (hasta 200), sesgo de la demanda según la curva de Hausman o una lista de demanda propia, con una vista previa de la demanda por SKU.
- Clases de la política basada en clases: de 1 a 8, con la proporción de SKUs de cada una.
- Lista de tareas: cantidad (automática o fija) y porcentaje de almacenamientos.
- Políticas a comparar: cualquier subconjunto de las cinco de asignación, combinadas con FCFS estricto, FCFS entre accesibles o las dos.
- Semilla y réplicas para la estadística (hasta 2000). La carrera anima la primera réplica.

Los parámetros se validan antes de ejecutar y los errores se muestran junto al formulario. Se pueden guardar y cargar como archivo JSON, y el navegador recuerda los últimos usados.

**Color de los ítems.** Depende de la política y cada panel trae su leyenda: en la política basada en clases es la clase del SKU (A, B, C), y en las demás es la rotación del SKU, en un degradado de amarillo (más rotación) a violeta (menos). Así, en `full_turnover` se ve cómo los SKUs más rotativos ocupan las posiciones más cercanas al E/S sin que aparezcan clases.

**Qué muestra**
- La carrera: todas las políticas ejecutan la misma lista de tareas sobre un mismo reloj. Cada una tiene su estantería y su grúa animadas, una tira con las tareas, un marcador en vivo y una tabla final de la réplica.
- La estadística: promedio con intervalo de confianza del 95 %, desviación, mínimo, máximo y en cuántas réplicas gana cada política, para la métrica que elijas. Se puede descargar como CSV.

La simulación se ejecuta en Python con el mismo código que `main.py`, así que los resultados coinciden con el CSV para las mismas semillas. Las réplicas de la estadística corren en paralelo y la página muestra su progreso, con opción de cancelar.

Opciones: `--puerto N` (por defecto 8765) y `--sin-abrir` (no abre el navegador).

**Archivo autónomo.** Sin `--servir`, `python visualizar.py --abrir` genera `visualizacion.html`, que reproduce la carrera con la configuración de `asrs/config.py` y no necesita servidor (opciones `--replicas`, `--semilla`, `--salida`). La versión 2D anterior está en `original/plantilla_visualizacion_2d.html`.

Pruebas:

```bash
python -m unittest discover -s tests
```

## Modelo de políticas

Toda combinación de decisiones es una **política de asignación de almacenamiento**. La misma política decide dónde almacenar y dónde reubicar: reubicar es almacenar un ítem que ya está en la estantería, tomando como referencia la celda de ese ítem. La recuperación tiene dos variantes y cada política de asignación se ejecuta con las que se elijan (por defecto, las dos):

- **FCFS estricto** (`fcfs`): sale el ítem más antiguo del SKU aunque tenga ítems delante, que se reubican. Representa un orden de salida obligatorio (perecederos, trazabilidad por lote).
- **FCFS entre accesibles** (`fcfs_accesible`): sale el más antiguo entre los ítems del SKU que no tienen nada delante; solo si todos están bloqueados se recurre al más antiguo de todos.

Las dos variantes de una misma política de asignación usan la misma semilla, así la diferencia entre ellas es pareada. Después de reubicar, la grúa va directo desde la celda donde dejó el último ítem hasta la celda objetivo; ese tramo cuenta en el tiempo de recuperación.

| Clave de configuración | Nombre en resultados | Criterio |
|---|---|---|
| `aleatoria` | `Aleatoria` | Canal elegido al azar |
| `basada_en_clases` | `BasadaEnClases` | La política agrupa los SKUs en clases por demanda (la clase A con los más demandados) y reserva una zona a cada clase; la clase A recibe las celdas más cercanas al E/S |
| `full_turnover` | `FullTurnover` | Se hace una lista de SKUs ordenada por rotación y se empareja con la lista de posiciones ordenada por cercanía al E/S: el SKU más rotativo recibe las posiciones más cercanas, el siguiente las que siguen, y así sucesivamente (Hausman, Graves y Schwarz, 1976). No hay clases |
| `mas_cercana_es` | `MasCercanaAlES` | Celda libre más cercana al E/S |
| `mas_cercana_grua` | `MasCercanaALaGrua` | Celda libre más cercana a la grúa: en reposo al almacenar, la celda del ítem que retira al reubicar |

La cercanía es el tiempo de viaje Chebyshev en el plano xy, `max(x/vx, y/vy)`.

### Supuestos que conviene tener presentes

- **La cercanía ignora la profundidad.** Las políticas comparan celdas por el tiempo en filas y columnas, sin contar el recorrido de la horquilla dentro del canal. Entre celdas igual de cercanas se elige la del fondo (minimización de varianza), aunque sea la más lejana en profundidad: así los canales se llenan desde atrás y no quedan huecos detrás de un ítem.
- **`mas_cercana_grua` y `mas_cercana_es` almacenan igual.** Cada tarea termina con la grúa en reposo, que está en el E/S, así que al almacenar la celda más cercana a la grúa es la más cercana al E/S. Solo se diferencian al reubicar; si casi no hay reubicaciones (por ejemplo con FCFS entre accesibles), sus resultados prácticamente coinciden.
- **Geometría.** El punto E/S es el origen de coordenadas y la posición de la estantería indica dónde está respecto de él. El centro de celda (0,5 por defecto en los tres ejes) indica, a partir de la posición de la estantería, en qué punto de cada celda se mide su posición: 0 es el borde inicial y 0,5 el centro. No mueve la estantería ni el E/S, que sigue en el origen; solo cambia el punto de cada celda al que va la grúa y, con él, la distancia desde el E/S y hasta dónde entra la horquilla en el canal. El riel queda delante de la cara frontal de la estantería, a la distancia de separación indicada, aunque la estantería se desplace en profundidad. La grúa va directo de un punto a otro por el pasillo, sin pasar por la esquina de la estantería.
- **Conteo de reubicaciones.** `Cant_Reubicaciones` cuenta ítems movidos. Las columnas de reubicaciones durante el almacenamiento o la recuperación (y sus desgloses regulares y predeterminadas) cuentan tareas: con canales de 3 capas o más una tarea puede mover varios ítems.

Las zonas se reparten por **canales completos** y el espacio de cada clase o SKU es proporcional a su demanda (el inventario sigue a la demanda), con al menos un canal por zona. Con FCFS estricto, `full_turnover` suele reubicar más que `basada_en_clases`: el ítem más antiguo de un SKU queda al fondo del canal, con ítems del mismo SKU delante.

## Lista de tareas

La lista de tareas solo conoce **SKUs**: cada tarea es un par `(actividad, SKU)`, donde la actividad es almacenar o recuperar. Agrupar los SKUs en clases, si hace falta, es decisión de cada política y no del generador.

- La demanda de cada SKU sigue la curva de Hausman `F(i) = (i/n)^σ` (σ = 1 es uniforme; menor σ concentra la demanda en pocos SKUs) o una lista propia. El SKU 1 es el más demandado.
- Las cantidades son exactas (método del mayor resto): se reparte primero entre almacenar y recuperar según la mezcla y luego entre SKUs según su demanda. El azar solo decide el orden.
- Cada réplica genera dos listas: una de llenado, solo almacenamientos, que deja la estantería en el nivel de ocupación inicial, y la lista de operación que se compara.

**Regla de minimización de varianza** (en todas las políticas): entre las celdas que la política considera equivalentes se elige la libre más profunda, así los canales se llenan desde el fondo. Solo se consideran celdas accesibles, es decir sin ítems delante. Si aún hay empate, se desempata al azar.

Si no hay celda factible al reubicar, el ítem se lleva temporalmente al E/S y se devuelve a la estantería al terminar la operación.

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
    vista_entrenamiento.js      Vista Entrenamiento: curvas y red neuronal en vivo
    gestor_entrenamiento.py     Lanza y vigila el proceso de entrenamiento
  rl/                           Aprendizaje por refuerzo (experimento aparte)
    entorno.py                    Entorno gymnasium de la asignación de almacenamiento
    entrenamiento.py              Creación de DQN y PPO y entrenamiento con entornos en paralelo
    entrenamiento_vivo.py         Proceso que entrena y emite su estado como JSON
    evaluacion.py  comparacion.py Evaluación y tabla comparativa con diferencias pareadas
visualizar.py                   Herramienta visual (--servir) o archivo autónomo
entrenar_rl.py  comparar_rl.py  Entrenamiento y comparación de los agentes de RL
tests/                          Pruebas de regresión
original/                       Versiones previas en un solo archivo
```

Dependencias entre capas (solo hacia abajo): `simulacion` → `control` → `politicas` → `dominio` / `metricas`. Para añadir una política: crear una subclase de `PoliticaAsignacion` que implemente `matrizCriterio`, registrarla en `politicas/registro.py` y añadir su clave a `vector_politicas_asignacion`.

## Modelo de réplicas

Cada réplica `i` usa la semilla `escalar_semilla + i` y genera una secuencia de llenado y una de operación. Cada política parte de una estantería vacía, la llena con esa misma secuencia bajo su propia lógica y procesa la secuencia de operación (comparación pareada). Las métricas del llenado se descartan.

## Aprendizaje por refuerzo (DQN y PPO)

**Objetivo:** comprobar si un agente entrenado con DQN o PPO puede **superar a las políticas clásicas** en el tiempo total de la lista de tareas. Es un experimento aparte del simulador, que no necesita estas librerías: `pip install -r requirements-rl.txt`.

### Cómo se plantea

- **Estado** `X_t = ((l, c, d), (i, e))` más la tarea actual y el perfil de la demanda:
  - *Configuración física:* para cada posición (nivel `l`, columna `c`, profundidad `d`, con 1 frontal) el SKU almacenado `i` y su orden de ingreso `e`, necesario para FIFO. Se añade la cercanía de cada posición al E/S, que es la geometría de las coordenadas.
  - *Tarea actual:* el SKU `i_t` y el tipo de operación `θ_t` (almacenamiento o recuperación). Cuando la decisión es reubicar un ítem que obstruye, la tarea en curso es una recuperación y se informa además el SKU del ítem que hay que ubicar.
  - *Perfil de la demanda:* el balance `p_t` entre almacenamientos y recuperaciones y los perfiles de SKUs que entran `P_t^S` y que salen `P_t^R`, calculados con las últimas 30 tareas y suavizados con la demanda de planificación mientras hay poco historial.
- **Acción:** una de las posiciones disponibles de almacenamiento, es decir celdas libres y accesibles (sin ítems delante). No se impone la regla de la celda más profunda: el agente elige la celda exacta.
- **Cuándo decide:** al almacenar cada ítem y al reubicar cada ítem que obstruye una recuperación. Las recuperaciones son FCFS estricto. Si no queda ninguna posición disponible no hay decisión: la tarea se omite o el ítem espera en el E/S, igual que en el simulador.
- **Recompensa:** el tiempo de ciclo con signo negativo, es decir lo que tarda la grúa desde una decisión hasta la siguiente. El retorno del episodio es el tiempo total de la lista, la misma métrica con la que se compara a las políticas clásicas.
- **Protocolo de comparación:** todos (agente y políticas clásicas) parten de la misma estantería, llenada con la política aleatoria, y ejecutan la misma lista de tareas. El agente se evalúa en estados de prueba que no vio al entrenar. Esto es distinto de la carrera del visualizador, donde cada política llena su propia estantería.
- **Algoritmos:** DQN toma el mejor valor Q entre las posiciones disponibles; PPO usa su máscara de acciones (`MaskablePPO` de sb3-contrib).
- **Implementación:** el simulador corre en un hilo que se detiene en cada decisión y espera la acción (`asrs/rl/entorno.py`), así se reutiliza el controlador de la grúa sin modificarlo.

### Cómo se usa

| Para… | Comando |
|---|---|
| Entrenar y comparar de una vez | `python entrenar_rl.py --algoritmo ambos --pasos 20000` |
| Entrenar viendo la red por dentro | `python visualizar.py --servir` y la vista **Entrenamiento** |
| Comparar los modelos ya guardados (`modelos/`) con un número fiable de estados | `python comparar_rl.py --pruebas 30 --csv resultados_rl.csv` |

El entrenamiento usa un entorno por núcleo (`--entornos`, por defecto los núcleos menos uno), porque lo más lento es la simulación y no las redes: con 7 entornos PPO pasó de 146 a unos 280 pasos por segundo y DQN llegó a unos 235. `--dispositivo cuda` entrena las redes en la tarjeta gráfica, pero hace falta PyTorch con CUDA (aquí se instaló el de CPU) y con redes tan pequeñas la ganancia es mínima.

La tabla de `comparar_rl.py` da, para cada política, el tiempo medio, la **diferencia pareada** contra la mejor clásica con su intervalo de confianza del 95 %, en cuántos estados le gana y cuántas tareas omitió. Si el intervalo no incluye 0, la diferencia es real.

### Dónde estamos

Con la formulación actual solo se probó una corrida corta (6 000 pasos, 18×20×2, 72 tareas, 6 estados de prueba): PPO queda un 24 % por encima de la mejor clásica (más cercana a la grúa) y DQN un 27 %, al nivel de la política aleatoria. Con tan pocos pasos no se puede concluir nada sobre su techo. El agente ahora elige posiciones exactas y también decide las reubicaciones, un problema más difícil que el anterior (que elegía solo el canal y dejaba las reubicaciones a una heurística).

### Siguientes pasos

1. **Entrenar en serio** (200 000 pasos o más) y medir con `comparar_rl.py` en 30 o más estados.
2. **Mejorar al agente antes de gastar más cómputo:** que arranque imitando a la mejor política clásica (así parte de su nivel y solo tiene que mejorarla), y una red que puntúe cada posición con los mismos pesos, que aprende mucho más rápido que mirar los más de 4000 números a la vez.
3. **Unificar el protocolo con la carrera** (que el agente también llene su estantería) para poder ponerlo junto a las demás políticas en el visualizador.

### Vista Entrenamiento del visualizador

En la herramienta interactiva hay dos vistas: Carrera y Entrenamiento. En la segunda eliges DQN o PPO, los pasos y los entornos en paralelo, y el servidor entrena en un proceso aparte (`asrs/rl/entrenamiento_vivo.py`) mientras la página muestra en vivo:

- la curva de aprendizaje, con el tiempo total por episodio y las políticas clásicas como líneas de referencia;
- las pérdidas (y epsilon en DQN);
- la red por dentro: los mapas por posición que ve el agente, cómo se activan las neuronas de cada capa oculta y el mapa de posiciones con lo que decide;
- cómo cambian los pesos: histogramas, a qué grupo de entradas mira cada neurona de la primera capa y un mapa de los pesos entre capas.

La red se muestra siempre ante la misma situación de la estantería, y un deslizador permite comparar el principio del entrenamiento con el final. Usa el almacén, la demanda y las tareas del formulario. Solo se entrena un agente a la vez y se puede detener en cualquier momento.
