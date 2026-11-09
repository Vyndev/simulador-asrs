# Simulador ASRS

Simulación de un sistema automatizado de almacenamiento y recuperación (ASRS) con una grúa, una estantería de celdas con canales de varias capas y un punto de entrada/salida (E/S). Compara **políticas de asignación de almacenamiento** sobre múltiples réplicas reproducibles.

## Uso

```bash
pip install -e .
python scripts/main.py
```

`pip install -e .` instala el paquete `asrs` y sus dependencias (`numpy`, `pandas`); `pip install -r requirements.txt` hace lo mismo. Todo lo que generan los scripts va a `salidas/`, que git ignora. Para repetir un experimento, guarda sus parámetros en un JSON dentro de `experimentos/` y lanza `python scripts/main.py --config experimentos/ejemplo.json` (también lo acepta `visualizar.py`).

Genera `salidas/resultados_simulacion.csv` (separador `;`) con una fila por réplica, política de asignación y política de recuperación, y `salidas/resultados_escenario.csv` con una fila por réplica y los indicadores de su escenario de demanda (ver "Modelo dinámico de suministro y demanda"). Los parámetros (tamaño de la estantería, velocidades, ocupación, demanda por clase, políticas, número de réplicas, semilla) están en `asrs/config.py`.

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
| SKU y demanda | surtido abierto de 180 productos activos en promedio, ciclo de vida de Bass con estacionalidad; concentración media de la curva de Hausman con σ = 0,4 |
| Clases | 3, con el 20 %, 20 % y 60 % de los SKU |
| Tareas por corrida | 525 600 (un año de 365 días a 1 440 tareas por día), tras un calentamiento de 30 días y un llenado inicial |
| Ocupación | 85 % de media, con un techo del 95 % |
| Políticas de asignación | 5 (aleatoria, basada en clases, dedicada por SKU, posición libre más cercana al E/S y más cercana a la grúa), todas con plan estático |
| Corridas | 1 por defecto |

Un año completo (525 600 tareas) tarda unos 3,5 minutos por política en un solo proceso y hasta unos 6,5 minutos con varios procesos en paralelo (se midieron 352 a 387 s con siete), así que las siete políticas que había entonces (antes de retirar las replanificadas) sumaban unos 25 minutos en un solo núcleo; con las cinco actuales es menos; `main.py` reparte las réplicas entre los núcleos. La herramienta visual (`--servir`) arranca con una lista corta (el 10 % de las celdas) para que la primera ejecución sea ágil, y las pruebas usan 72 tareas.

## Modelo dinámico de suministro y demanda

La lista de tareas generada sigue el modelo de la sección 6.2.3 del capítulo 6 (decisiones de control, tablas 6 y 7) y su diseño (`documento/disenos/2026-10-25-modelo-suministro-demanda.md`). El generador produce el escenario y las políticas solo ven la lista; el escenario es el mismo para todas las políticas de una réplica. Código en `asrs/generacion/ciclo_vida.py`, `reposicion.py` y `escenario.py`.

- **Surtido abierto y ciclo de vida.** Cada producto nace en un día de lanzamiento, recorre una curva de Bass (p = 0,03, q = 0,38) de 18 meses (un mes son 365/12 días) y sale al cerrar el ciclo, sin que nadie lo reemplace. Nacen n / L productos por día (n = `escalar_numero_skus`, el surtido medio, y L la duración del ciclo) en un proceso de Poisson, así que el surtido activo promedia n y fluctúa. El día 0 hay n productos con edades repartidas de forma uniforme. El SKU de la lista es el número del producto, que no se reutiliza, de modo que hay más de n SKU distintos en una corrida. Cada producto nuevo toma su altura de un rango de la curva de Hausman de n rangos y sortea su propia estacionalidad.
- **Estacionalidad.** Cada producto tiene una amplitud (cuantiles de Butters et al. 2025) y un mes de pico (44 % de diciembre a febrero, 37 % de junio a agosto, 19 % el resto). El día 0 del año es el 1 de enero.
- **Demanda y mezcla.** Cada día hay en promedio R = 720 recuperaciones de demanda (60 ciclos por hora × 24 horas ÷ 2), repartidas entre los productos activos según su participación. La proporción de almacenamientos no es un parámetro (`vector_prob_acum_AR` no se aplica a la lista generada): resulta del modelo.
- **Reposición.** Lotes EOQ, `Q = max(1, redondeo(κ √D))`, con `κ` recalculada cada día para que el inventario esperado sea el 85 % de las celdas. Punto de pedido y plazo 0: el lote entra justo después de la recuperación que deja un producto sin unidades. Hay un primer lote al nacer el producto, un lote final cuando la demanda restante del producto es menor que Q, y un retiro de fin de ciclo que vacía al producto cuando sale.
- **Techo de ocupación.** Un lote que llevaría la ocupación por encima del 95 % se aplaza hasta que haya espacio. Si un producto sin existencias tiene el lote aplazado y le llega demanda, esa recuperación no se emite y se cuenta como demanda no atendida.
- **Calentamiento y horizonte.** 30 días de prehistoria que el generador lleva sin que las políticas la ejecuten; su inventario final define la lista de llenado. El año evaluado son las 525 600 tareas siguientes.
- **Popularidad móvil.** Es lo único que las políticas ven de la demanda: las recuperaciones de cada SKU en los últimos 30 días, y qué SKU existen (el catálogo). La popularidad de un producto nuevo arranca en cero.
- **Datos reales.** El camino de datos reales no cambia: sin escenario no hay avisos de día ni retiros.

**Cinco políticas de asignación.** Son heurísticas clásicas y estáticas: no se adaptan a la demanda que cambia (esa adaptación es lo que aporta el agente de aprendizaje). La basada en clases (`basada_en_clases`) y la dedicada por SKU (`full_turnover`) planifican una sola vez, antes del llenado inicial, con la popularidad móvil calculada con el historial (con datos reales, el primer tramo de la lista, que no se evalúa), y el plan no cambia durante el año. Solo planifican los SKU que existen en ese momento. Un SKU que nace después de planificar no tiene zona propia y se trata como el de menor demanda: la basada en clases lo pone en la clase de menor demanda y la dedicada, en la zona del SKU de menor demanda del plan. El tamaño de cada zona es proporcional a la raíz de la popularidad (con el perfil EOQ el inventario medio crece con la raíz de la demanda) y, si la zona no tiene una celda libre accesible, se relaja la zonificación, así ninguna política descarta almacenamientos. Las otras tres (`aleatoria`, `mas_cercana_es`, `mas_cercana_grua`) no usan la demanda. Con las dos reglas de recuperación (FCFS estricto y FCFS entre accesibles) son 10 combinaciones.

**Claves nuevas de configuración** (`asrs/config.py`, valores por defecto de la instancia del capítulo 6):

| Clave | Por defecto | Significado |
|---|---|---|
| `texto_modelo_demanda` | `ciclo_vida` | `ciclo_vida` (Bass y estacionalidad) o `estatica` (participaciones constantes de Hausman) |
| `escalar_bass_p`, `escalar_bass_q` | 0,03; 0,38 | Coeficientes de innovación y de imitación de Bass |
| `escalar_meses_ciclo_vida` | 18 | Duración del ciclo de vida |
| `vector_cuantiles_amplitud` | [0,05; 0,10; 0,20; 0,47; 0,67] | Amplitud estacional (puntos logarítmicos) |
| `vector_probabilidades_amplitud` | [0; 0,10; 0,50; 0,90; 1] | Probabilidad acumulada de cada cuantil |
| `vector_reparto_mes_pico` | [0,44; 0,37; 0,19] | Posiciones con el pico en dic–feb, jun–ago y el resto |
| `escalar_ciclos_por_hora`, `escalar_horas_por_dia` | 60; 24 | Fijan R = ciclos × horas ÷ 2 recuperaciones de demanda al día |
| `escalar_dias_ventana_popularidad` | 30 | Ventana de la popularidad móvil |
| `escalar_dias_calentamiento` | 30 | Días de prehistoria antes del año evaluado |
| `escalar_techo_ocupacion` | 0,95 | Techo de ocupación de la estantería |

**Salidas nuevas.** `resultados_simulacion.csv` suma dos columnas, `Cant_RetirosFinCiclo` y `TiempoTotal_RetirosFinCiclo`: los retiros de fin de ciclo se cuentan además como recuperaciones, y su tiempo ya está incluido en el de recuperación. `resultados_escenario.csv` tiene una fila por réplica con `semilla`, `demanda_no_atendida`, `lotes_aplazados`, `retiros`, `lanzamientos`, `ocupacion_media`, `ocupacion_minima`, `ocupacion_maxima`, `participacion_top20_media`, `sigma_alturas`, `dias_operacion`, `productos` (SKU distintos que existen), `surtido_medio`, `surtido_minimo` y `surtido_maximo`. `retiros` cuenta tareas de retiro (una por unidad retirada), no productos; `lanzamientos` cuenta productos nacidos en la operación.

**Límites.** `TAREAS_POR_CORRIDA` es ahora un año (525 600) y `LIMITE_TAREAS_GENERADAS` es 600 000 (para datos reales `LIMITE_TAREAS` vale lo mismo; antes era 100 000, lo que dejaba fuera a calzado).

**Limitaciones conocidas**
- El flujo diario R = 720 sale de los 60 ciclos por hora y no escala con la estantería. En estanterías muy pequeñas (por ejemplo 6×6×2 con 10 SKU) una parte grande de la demanda queda sin atender: hay que reducir `escalar_ciclos_por_hora`.
- La página web envía siempre la clave del techo (0,95): con origen generado rechaza ocupaciones iguales o mayores al 95 %. Su lista de políticas sigue mostrando solo las cinco originales.
- Con una lista generada, en la página el campo "% almacenar" se ignora (la mezcla es un resultado del modelo) y "Sesgo σ (Hausman)" es ahora la concentración objetivo 0,2^σ del 20 % de SKUs más demandados.

## Capa del agente: perfil de SKU y celdas candidatas

El paquete `asrs/agente/` es lo que el agente de aprendizaje observa; no cambia la simulación. Se apoya en dos datos que el escenario conserva: las recuperaciones de demanda por SKU y día (`escenario.recuperaciones(desde, hasta)`, sin los retiros de fin de ciclo, con días de prehistoria negativos) y el día de la primera llegada de cada SKU (`escenario.vector_dia_primera_llegada`: -10⁹ si ya tenía existencias el día 0).

- **`perfil.py`.** `perfil_de_skus(escenario, día)` da, por SKU y solo con días anteriores al dado, el nivel ln(1 + r) con r la tasa por día dentro de su historia, la tendencia (tasa de los últimos 15 días frente a los 15 anteriores) y los días de historia acotados a la ventana de 30 días y divididos entre ella. Con `escalar_duracion_ciclo` añade la edad real como cuarta columna, la variable extra de la ablación.
- **`candidatos.py`.** `celdas_candidatas(estantería, tiempos al E/S, perfiles)` lista las celdas libres y accesibles con sus características: tiempo al E/S, capa, celdas libres del canal, si el fondo quedaría inaccesible, si la unidad de atrás quedaría bloqueada y el perfil de esa unidad. El agente solo tiene modo libre: todas las celdas libres y accesibles son candidatas y puede elegir cualquiera, también el frente de un canal con el fondo libre. No aplica la minimización de varianza.
- **`scripts/separabilidad_perfil.py`.** Mide qué tan bien separa el perfil a un SKU creciente de uno en declive. Con nivel y tendencia acierta 0,95 en semillas de prueba y con la edad real 1,00; el nivel solo no distingue (0,53).

**Por qué se diseñó así (deducciones).**
1. El surtido es abierto y el SKU no se reutiliza, así que una entrada con un elemento por SKU cambiaría de tamaño y la identidad no serviría para un SKU nuevo: el agente observa el perfil, que sí se repite entre SKU.
2. La popularidad tiene cola pesada (concentración de Hausman): el nivel usa ln(1 + r). Un SKU nuevo tiene la ventana incompleta: r se mide dentro de su historia. La curva de Bass es simétrica, así que un SKU al lanzamiento y otro al cierre tienen el mismo nivel y solo la tendencia los distingue (el script de separabilidad lo mide). La historia se acota a la ventana W porque con ella completa no informa sobre la confianza de la tasa y sin tope revelaría la edad real, que el supuesto S7 excluye.
3. Cada celda libre y accesible es una candidata que se puntúa con las mismas características, así que la acción es independiente del tamaño de la estantería y del surtido. El costo de una asignación viene del bloqueo, por eso cada candidata incluye el perfil de la unidad que quedaría detrás.
4. Decisión de diseño: las políticas clásicas usan siempre la minimización de varianza, como regla interna del simulador que no se configura. El agente, en cambio, es libre y puede decidir cualquier celda disponible. Por eso el agente tiene una única forma de operar, la libre, y no hay un modo restringido con el que igualar las reglas.
5. Con la configuración por defecto nada cambia: los tiempos de cuatro corridas de un año coinciden exactamente con los del piloto de réplicas.

## Herramienta visual

```bash
python scripts/visualizar.py --servir
```

Abre en el navegador una herramienta interactiva (servidor local en `127.0.0.1`, sin dependencias nuevas). Se trabaja en tres pasos, que están siempre a la vista en la barra superior junto al botón **Simular**:

1. **Escenario.** Todo lo que se puede ajustar, a la vista y a todo el ancho, en cuatro tarjetas: Almacén (con un esquema a escala de la estantería y del punto E/S), Demanda, Políticas y Ejecución, con un resumen siempre visible. No hay pestañas que escondan parámetros. Si cambias algo después de simular, la barra avisa "Cambios sin simular".
2. **Carrera.** Al terminar de simular se pasa solo a este paso: las políticas ejecutan la misma lista de tareas sobre un mismo reloj, en 3D (Three.js, incluido en `asrs/visualizacion/vendor/`, funciona sin internet). La cámara es la misma en todos los paneles: arrastra para girar, rueda para acercar, doble clic para restablecer, o usa los botones ISO, FRT, LAT y SUP (isométrica, frente, lateral y superior). Cada visor trae cotas del largo y la altura de la estantería y un triedro con los ejes de columnas, filas y profundidad. El punto E/S se dibuja como una placa frente a la grúa, con una zona de espera al lado para los ítems que aguardan su turno (el que llega mientras otro sale y los que se reubican temporalmente). Cuando la estantería es grande, un recuadro de detalle en la esquina del visor sigue a la grúa y muestra ampliados el carro, la horquilla y la carga; se oculta con el botón GRÚA o la tecla D. Con tiempo de carga y descarga 0 la grúa toma y entrega en el mismo instante, así que esas transferencias se dibujan con una animación breve que no altera los tiempos; con un tiempo mayor que 0 duran lo que dice el modelo. Con listas muy largas la carrera anima solo las primeras tareas y lo avisa.
3. **Resultados.** Primero la conclusión en una frase (qué política gana, por cuánto y si la diferencia es creíble según la comparación pareada de las réplicas) y después la estadística de las réplicas y la tabla de la carrera.

Si recargas la página mientras corre una simulación, se reconecta y sigue su progreso. En el menú ⋯ están restablecer, guardar y cargar el escenario; el botón de tema alterna entre automático, claro y oscuro.

**Parámetros editables**
- Estantería: filas, columnas, capas de profundidad, medidas de la celda, posición y centro de referencia de la celda.
- Origen de las tareas: generadas con el modelo dinámico de suministro y demanda (ciclo de vida de Bass y estacionalidad; con una lista de demanda propia esta fija la altura de la curva de cada SKU) o **datos reales** desde un CSV con una tarea por fila y dos columnas, `SKU` y `Task` (A o R; separador y encabezado se detectan solos). También se acepta un CSV solo de demanda, de una columna `SKU` o de dos columnas `SKU,Dia` con el día del modelo de cada recuperación (como lo escribe `scripts/extraer_demanda.py`; ver "Demanda de las fuentes por réplica"): el simulador genera los almacenamientos con la reposición por lotes EOQ del modelo de suministro (el lote de cada SKU sale de su participación en el historial y el llenado inicial es la mitad del lote), de modo que la lista de operación tiene cerca del doble de tareas. Con datos reales la cantidad de SKUs, la demanda, la cantidad de tareas y la mezcla de operaciones salen del archivo, y los SKUs se numeran por frecuencia (el 1 es el más frecuente) conservando su código en la animación. Las políticas planifican (llenado inicial y clases) con el historial: con días, las primeras R·B = 21 600 recuperaciones (B = `escalar_dias_calentamiento`); sin días, el primer porcentaje de la lista que se indique (por defecto 30 días de 395, el 7,59 %); ese tramo es anterior a lo que se evalúa y no se ejecuta: se ejecuta lo que viene después, de modo que ninguna política conoce demanda posterior al momento actual, y se mide cuánto afecta que la demanda real cambie respecto de la conocida. Un historial del 100 % no deja nada por evaluar y es un error. Con un CSV todas las réplicas usan la misma lista y solo cambia el llenado inicial, salvo que se elija un nivel de variabilidad (la mezcla diaria se sortea en cada réplica).
- Grúa y E/S: velocidades, aceleraciones y deceleraciones en filas, columnas y canales, tiempo de carga y descarga, separación del riel. Cada eje acelera, avanza a la velocidad máxima y frena (perfil trapezoidal, o triangular en distancias cortas); el tramo xy dura lo que tarda el eje más lento y luego se recorre z. Con aceleración o deceleración 0 no hay límite (velocidad constante). Por defecto son 1 m/s². Ver `asrs/cinematica.py`.
- Ocupación inicial y perfil de inventario (cuántas unidades de cada SKU hay guardadas, ver más abajo).
- Productos (SKU): cantidad (hasta 200), sesgo de la demanda según la curva de Hausman o una lista de demanda propia, con una vista previa de la demanda por SKU.
- Clases de la política basada en clases (dentro de la propia política): la política agrupa los SKUs sola con la demanda definida en la pestaña Demanda (no la modifica) y solo se elige cómo formar las clases. *Misma demanda en cada clase* (de 1 a 8 clases, cada una con la misma parte de la demanda; es lo predeterminado, con 3 clases), *Pareto 80 / 15 / 5* (A con el 80 % de la demanda, B con el 15 % y C con el 5 %) o *a mano* con el porcentaje de SKUs de cada clase. Con los dos primeros no hace falta indicar cuántos SKUs tiene cada clase: salen de la curva de demanda acumulada (`agrupar_skus_por_demanda` en `asrs/politicas/zonificacion.py`), y la vista previa muestra el resultado.
- Lista de tareas: cantidad (automática o fija) y porcentaje de almacenamientos.
- Políticas a comparar: cualquier subconjunto de las cinco políticas de asignación, combinadas con FCFS estricto, FCFS entre accesibles o las dos.
- Semilla y réplicas para la estadística (hasta 2000). La carrera anima siempre una réplica, la primera.

Los parámetros se validan antes de ejecutar y los errores se muestran junto al formulario. Se pueden guardar y cargar como archivo JSON, y el navegador recuerda los últimos usados.

**Color de los ítems.** Depende de la política y cada panel trae su leyenda: en la política basada en clases es la clase del SKU (A, B, C), y en las demás es la rotación del SKU, en un degradado de amarillo (más rotación) a violeta (menos). Así, en `full_turnover` se ve cómo los SKUs más rotativos ocupan las posiciones más cercanas al E/S sin que aparezcan clases.

**Qué muestra**
- La carrera: todas las políticas ejecutan la misma lista de tareas sobre un mismo reloj. Cada una tiene su estantería y su grúa animadas, una tira con las tareas, un marcador en vivo y una tabla final de la réplica.
- La estadística: promedio con intervalo de confianza del 95 %, desviación, mínimo, máximo y en cuántas réplicas gana cada política (por tiempo total), para la métrica que elijas. Se puede descargar como CSV.

**Cómo se decide si una política es mejor.** El intervalo de confianza de cada promedio usa la t de Student con n − 1 grados de libertad (la desviación se estima con las mismas réplicas; con pocas réplicas la normal daría intervalos demasiado estrechos). Como todas las políticas corren las mismas secuencias en cada réplica, la comparación con la mejor es **pareada**: se calcula la diferencia réplica a réplica y su intervalo de confianza del 95 %. Si ese intervalo incluye el cero, la diferencia no es significativa: el veredicto dice que las políticas "quedan muy parejas" y la tabla la marca con *n. s.* Este criterio es más exigente con los datos que comparar si dos intervalos se solapan, porque descuenta la variación que comparten las políticas (la misma secuencia afecta a todas). Ver `asrs/simulacion/estadistica.py`.

La simulación se ejecuta en Python con el mismo código que `scripts/main.py`, así que los resultados coinciden con el CSV para las mismas semillas. Las réplicas de la estadística corren en paralelo y la página muestra su progreso, con opción de cancelar.

Opciones: `--puerto N` (por defecto 8765) y `--sin-abrir` (no abre el navegador).

**Archivo autónomo.** Sin `--servir`, `python scripts/visualizar.py --abrir` genera `salidas/visualizacion.html`, que reproduce la carrera con la configuración de `asrs/config.py` y no necesita servidor (opciones `--replicas`, `--semilla`, `--salida`).

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
| `basada_en_clases` | `BasadaEnClases` | La política agrupa los SKUs en clases por popularidad móvil (la clase A con los más demandados) y reserva una zona a cada clase; la clase A recibe las celdas más cercanas al E/S. Plan fijo: las clases se forman una vez, antes del llenado |
| `full_turnover` | `FullTurnover` | Se hace una lista de SKUs ordenada por rotación y se empareja con la lista de posiciones ordenada por cercanía al E/S: el SKU más rotativo recibe las posiciones más cercanas, el siguiente las que siguen, y así sucesivamente (Hausman, Graves y Schwarz, 1976). No hay clases. Plan fijo: la lista se ordena una vez, antes del llenado |
| `mas_cercana_es` | `MasCercanaAlES` | Celda libre más cercana al E/S |
| `mas_cercana_grua` | `MasCercanaALaGrua` | Celda libre más cercana a la grúa: en reposo al almacenar, la celda del ítem que retira al reubicar |

La cercanía es el tiempo de viaje Chebyshev en el plano xy, `max(x/vx, y/vy)`.

Las zonas se reparten por **canales completos** y el espacio de cada clase o SKU es proporcional a su inventario, con al menos un canal por zona. El orden de las zonas (cuál queda más cerca del E/S) es por demanda.

**Perfil de inventario.** La demanda dice cuántas veces se pide cada SKU; el perfil de inventario dice cuántas unidades suyas hay guardadas. Define el llenado inicial y el tamaño de las zonas (`perfil_inventario` en `asrs/generacion/probabilidades.py`):
- *Lote económico (EOQ)*, por defecto. Cada SKU se repone en lotes de `Q = √(2·D·S/H)` y su inventario promedio es `Q/2`, así que crece con la raíz de la demanda: un SKU que se pide 4 veces más tiene 2 veces más unidades y cada una se pide 2 veces más seguido. Se supone el mismo costo de pedir (S) y de mantener (H) para todos los SKUs, por eso se cancelan al normalizar y no hace falta conocerlos. Como demanda ÷ inventario = √demanda, ordenar por demanda es lo mismo que ordenar por rotación (índice COI de Heskett).
- *Proporcional a la demanda*. Cada unidad guardada se pide con la misma frecuencia, sea del SKU que sea, así que ninguna asignación por rotación puede reducir el recorrido: solo cuenta tener el inventario compacto cerca del E/S.

Durante la lista de tareas cada SKU entra y sale al ritmo de su demanda, así que el perfil inicial se conserva en promedio. Con FCFS estricto, `full_turnover` suele reubicar más que `basada_en_clases`: el ítem más antiguo de un SKU queda al fondo del canal, con ítems del mismo SKU delante.

### Supuestos que conviene tener presentes

- **La cercanía ignora la profundidad.** Las políticas comparan celdas por el tiempo en filas y columnas, sin contar el recorrido de la horquilla dentro del canal. La profundidad la fija antes la minimización de varianza: primero se ocupa la capa libre más profunda de todos los canales candidatos y la cercanía solo desempata dentro de esa capa, aunque la celda sea la más lejana en profundidad.
- **`mas_cercana_grua` y `mas_cercana_es` almacenan igual.** Cada tarea termina con la grúa en reposo, que está en el E/S, así que al almacenar la celda más cercana a la grúa es la más cercana al E/S. Solo se diferencian al reubicar; si casi no hay reubicaciones (por ejemplo con FCFS entre accesibles), sus resultados prácticamente coinciden.
- **Geometría.** El punto E/S es el origen de coordenadas y la posición de la estantería indica dónde está respecto de él. El centro de celda (0,5 por defecto en los tres ejes) indica, a partir de la posición de la estantería, en qué punto de cada celda se mide su posición: 0 es el borde inicial y 0,5 el centro. No mueve la estantería ni el E/S, que sigue en el origen; solo cambia el punto de cada celda al que va la grúa y, con él, la distancia desde el E/S y hasta dónde entra la horquilla en el canal. El riel queda delante de la cara frontal de la estantería, a la distancia de separación indicada, aunque la estantería se desplace en profundidad. La grúa va directo de un punto a otro por el pasillo, sin pasar por la esquina de la estantería.
- **Conteo de reubicaciones.** `Cant_Reubicaciones` cuenta ítems movidos. Las columnas de reubicaciones durante el almacenamiento o la recuperación (y sus desgloses regulares y predeterminadas) cuentan tareas: con canales de 3 capas o más una tarea puede mover varios ítems.

## Lista de tareas

La lista de tareas solo conoce **SKUs**: cada tarea es un par `(actividad, SKU)`, donde la actividad es almacenar o recuperar. Agrupar los SKUs en clases, si hace falta, es decisión de cada política y no del generador.

- Con el modelo `ciclo_vida` (por defecto) la lista sale del escenario descrito en "Modelo dinámico de suministro y demanda": la demanda cambia día a día y la mezcla de almacenamientos y recuperaciones resulta de la reposición.
- Con el modelo `estatica` la demanda de cada SKU es constante y sigue la curva de Hausman `F(i) = (i/n)^σ` (σ = 1 es uniforme; menor σ concentra la demanda en pocos SKUs) o una lista propia, y el SKU 1 es el más demandado. Es el caso de referencia del modelo dinámico.
- Cada réplica genera dos listas: una de llenado, solo almacenamientos, que deja la estantería con el inventario del final del calentamiento (cerca del 85 % de ocupación), y la lista de operación que se compara, con exactamente `escalar_largo_secuencia` tareas.

**Regla de minimización de varianza** (fija en todas las políticas clásicas, no se configura; el agente no está sujeto a ella): entre todas las celdas candidatas (libres y accesibles, es decir sin ítems delante; en las políticas con zonas, las de la zona del SKU) se elige primero la capa libre más profunda, y dentro de esa capa el criterio de la política (cercanía, azar, etc.). Así se ocupan los fondos de todos los canales antes que cualquier frente, que minimiza la varianza del llenado de los canales (Lehmann y Hußmann, 2023) y da reubicaciones nulas hasta el 50 % de ocupación con 2 capas (Lerher et al., 2010). Si aún hay empate, se desempata al azar. Hasta el 2026-11-02 la regla solo desempataba dentro del canal; ver `docs/decisiones.md`.

Si no hay celda factible al reubicar, el ítem se lleva temporalmente al E/S y se devuelve a la estantería al terminar la operación.

**Recorrido de una recuperación con reubicación.** La grúa va a la estantería, saca cada ítem que obstruye y lo deja en la celda que elige la política. Si la última reubicación fue dentro de la estantería, sigue desde esa celda directo a la celda objetivo; ese tramo cuenta como tiempo de recuperación. Si la última reubicación dejó el ítem en el E/S, la grúa sale desde ahí. Después retira el ítem, lo entrega en el E/S y vuelve al reposo.

## Demanda de las fuentes por réplica

Implementa las secciones 8.2.1, 8.2.3 y 8.2.4 del documento (tratamiento de las fuentes, variación entre réplicas y escenarios de variabilidad).

- **Día de cada recuperación.** El calendario de la fuente (L días, del primero al último con demanda) se reescala a los D = Y + B = 395 días del modelo: la recuperación del día r de la fuente pasa al día ⌊r·D/L⌋, y al escalar el volumen a N = R(Y + B) = 284 400 recuperaciones cada copia hereda su día. La secuencia se escribe como `SKU,Dia`.
- **Historial por volumen.** Con días, el historial son las primeras R·B = 21 600 recuperaciones (cuántos días abarca depende del ritmo de la fuente y de la rotación) y la operación, con R·Y = 262 800 recuperaciones, empieza el día de la siguiente, que es su día 0. Las heurísticas planifican con la demanda del historial. El escenario de la réplica lleva la popularidad móvil de cada día, Pop_i(τ) = recuperaciones de los W = 30 días anteriores (ec. 8.3), el día de cada tarea (los almacenamientos de la reposición llevan el día de la recuperación que los provoca) y el registro de la demanda pasada que observa el agente. Se eligió por volumen y no por días porque el comercio electrónico concentra su demanda en noviembre y un historial de 30 días quedaba casi vacío en muchas réplicas (ver `docs/decisiones.md`).
- **Origen `fuente`.** Con `texto_origen_tareas = "fuente"` y `texto_fuente_demanda` (`ecommerce`, `calzado` o `inventario_sintetico`), cada réplica trata la fuente de `texto_carpeta_datos` (por defecto `../tesis/datos`, que no se versiona) con su semilla: el muestreo ponderado del surtido (o la ampliación del caso sintético) y una **rotación circular** de k días, k uniforme entre 0 y L − 1 (`default_rng([semilla, 1])`), aplicada antes del reescalado y del escalado. El corte cae entre dos días. El archivo crudo se lee una vez por proceso.
- **Variabilidad.** `texto_variabilidad` = `baja`, `real` o `alta` sortea en cada réplica (`default_rng([semilla, 2])`) la mezcla de cada día, p(d) ~ Dirichlet(η ρ̃(d)), con ρ̃(d) la participación en la ventana de W días centrada en d (desplazada hacia adentro en los bordes) y η = (1 − ω)/ω; las n_d recuperaciones del día son un sorteo multinomial en orden aleatorio. ω vale ω̂/2, ω̂ y (1 + ω̂)/2. ω̂ es uno por fuente: se estima por momentos (ec. 8.8) una vez, sobre la secuencia de referencia (tratada y escalada con la semilla 0, sin rotar), y vale para todas las réplicas (con un CSV, sobre el archivo), salvo que se fije `escalar_omega_fuente`. Con los datos: calzado 0,0997, comercio electrónico 0,0733 y caso sintético 0. `ninguna` (por defecto) usa la secuencia tratada tal cual. Funciona con el origen `fuente` y con un CSV `SKU,Dia`.
- `resultados_escenario.csv` informa por réplica la fuente, el desplazamiento, ω̂, el ω aplicado, los SKU, las recuperaciones del historial, el inventario inicial y las tareas de operación.

```bash
python scripts/extraer_demanda.py                 # salidas/demanda/<fuente>.csv (SKU,Dia), <fuente>_skus.csv, <fuente>_experimento.json e imprime ω̂
python scripts/main.py --config salidas/demanda/ecommerce_experimento.json   # origen 'fuente'; añade "texto_variabilidad": "real" para el escenario base
```

## Arquitectura

```
scripts/
  main.py                       Punto de entrada: ejecuta la simulación
  visualizar.py                 Herramienta visual (--servir) o archivo autónomo
  separabilidad_perfil.py       Mide qué tan bien separa el perfil a un SKU creciente de uno en declive
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
    ciclo_vida.py                 Surtido abierto: curva de Bass, estacionalidad y nacimientos de productos
    reposicion.py                 Lotes EOQ, techo, retiros de fin de ciclo y escenario de un año
    escenario.py                  EscenarioDemanda: listas de llenado y de operación, días, retiros y popularidad
    datos_reales.py               Lista de tareas desde un CSV (SKU y Task, o SKU y opcionalmente Dia de las recuperaciones), historial, reposición EOQ y escenario con días
    variabilidad.py               Tendencia, mezcla diaria con la Dirichlet y estimación de ω̂ (sección 8.2.4)
    demanda_replica.py            Demanda propia de cada réplica: fuente tratada con su semilla y nivel de variabilidad
    fuentes/                      Un adaptador por fuente (calzado, ecommerce e inventario sintético) y el núcleo que extrae la secuencia de SKU, una recuperación por unidad de cada línea (`scripts/extraer_demanda.py`)
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
tests/                          Pruebas de regresión, una por módulo (test_dominio, test_politicas, ...); auxiliar.py tiene lo común
experimentos/                   Configuraciones de experimentos en JSON, para repetir cada resultado
salidas/                        Lo generado (CSV, HTML); lo ignora git
docs/                           Planes de implementación con fecha y registro de decisiones (decisiones.md)
pyproject.toml                  Paquete asrs y sus dependencias
```

Dependencias entre capas (solo hacia abajo): `simulacion` → `control` → `politicas` → `dominio` / `metricas`. Para añadir una política: crear una subclase de `PoliticaAsignacion` que implemente `matrizCriterio`, registrarla en `politicas/registro.py` y añadir su clave a `vector_politicas_asignacion`.

## Modelo de réplicas

Cada réplica `i` usa la semilla `escalar_semilla + i` y genera una secuencia de llenado y una de operación (con el modelo `ciclo_vida`, un escenario de un año y su resumen; con el origen `fuente` o un nivel de variabilidad, su propia demanda, ver "Demanda de las fuentes por réplica"). Cada política tiene una semilla derivada de su posición en el registro (no de su posición en la lista elegida). Cada política parte de una estantería vacía, la llena con esa misma secuencia bajo su propia lógica y procesa la secuencia de operación (comparación pareada). Las métricas del llenado se descartan.
