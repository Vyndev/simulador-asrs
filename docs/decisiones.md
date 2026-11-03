# Registro de decisiones

Una entrada por decisión que afecta a los resultados o al texto de la tesis. Es el lugar para anotar el **porqué**
mientras se recuerda: el README explica cómo funciona el simulador, no por qué se eligió así.

Plantilla para añadir una entrada nueva (las más recientes arriba):

```
## AAAA-MM-DD · Título corto
- **Contexto:** qué problema o duda había.
- **Decisión:** qué se hizo.
- **Alternativas descartadas:** y por qué.
- **Efecto en los resultados:** qué cambia si se decide distinto.
- **Dónde está en el código:** archivo o sección del README.
```

## 2026-11-02 · Secuencia de demanda extraída de las fuentes reales
- **Contexto:** antes de generar escenarios a partir de datos reales hace falta una entrada común: solo la demanda, ordenada, sin fechas ni cantidades. Los 15 345 SKU de ecommerce y los 208 de calzado no caben en el surtido del modelo (180).
- **Decisión:** un adaptador por fuente (`asrs/generacion/fuentes/`) dice qué filas son demanda; el núcleo ordena por tiempo (estable: los empates conservan el orden del archivo), descarta el tiempo y escribe `salidas/demanda/<fuente>.csv` (una columna `SKU`). `leer_tareas` acepta ese CSV (todas las tareas son recuperaciones). Cada línea con demanda cuenta como una recuperación, sea cual sea su cantidad. Filtros: calzado, 33 filas con cantidad cero (SKU = referencia; referencia-talla con `--sku-con-talla`); ecommerce, 240 con fecha de pedido fuera de agosto a diciembre de 2015 y 14 con cantidad cero; inventario sintético (ventas diarias, cada fila con ventas es una recuperación), 620 sin ventas; almacén inteligente (filas con `outgoing_items` > 0), 83 sin salidas. Inventario sintético y almacén inteligente son datos simulados o aleatorios según sus `calidad.md`. Almacén logístico no tiene adaptador: una fila por artículo, sin eventos de demanda ni orden temporal, y fabricar una secuencia sería inventar datos.
- **Condensación a 180 etiquetas como máximo** (`agrupar_surtido`, sin quitar recuperaciones; `--grupos`, 0 lo desactiva): los SKU que por sí solos superan `tope` veces la demanda media por etiqueta (n / 180) quedan individuales y la cola se agrupa con k-means (numpy, semilla 0) sobre log(1 + frecuencia) y primera, última y media de aparición; se parte con el 60 % del cupo y se divide primero el grupo que pasa del tope de demanda, luego el más disperso en el tiempo. **El tope es 0,1** en calzado y ecommerce (`--tope`, o `escalar_tope` del adaptador): así la cabeza de la demanda, que es lo que más pesa en las políticas, queda individual y solo se agrupa la cola larga. **En almacén inteligente es 1,5:** su demanda es casi uniforme, con 0,1 casi todos los SKU pasarían el umbral y no quedaría cupo para la cola (el código limita los individuales a 180 - 1 por si ocurre). Inventario sintético (50 SKU) no se agrupa.
- **Codificación:** los SKU se renombran SKU001, SKU002... por demanda decreciente; la correspondencia con el nombre original (o con el grupo) está en `<fuente>_skus.csv` (`--sin-codificar` la desactiva).
- **Alternativas descartadas:** quedarse con los N SKU más frecuentes (borra el 37 % de las tareas en ecommerce); k-means sobre todo el surtido sin individuales (el mayor grupo sumaba el 22,7 % de las tareas, una masa gris); tope 1,5 también en ecommerce y calzado (solo 14 SKU individuales en ecommerce, el 69,5 % de las tareas en grupos y 41 grupos con SKU de momentos distintos); «cubetas» que juntan los SKU más demandados (borran la cabeza de la demanda); repetir el SKU tantas veces como unidades (la cantidad es de la línea, no del movimiento del ASRS); excluir las líneas repetidas de una misma orden (calidad.md no prueba que sean errores); ordenar las líneas de una orden por `orderToCollect` (el pedido lo da el archivo).
- **Efecto en los resultados:** ninguno en la simulación por defecto. Cifras con el tope elegido: ecommerce 99 746 tareas, 150 etiquetas individuales y 30 grupos, el mayor con el 1,9 % de las tareas, el 39,3 % de las tareas en grupos y el 20 % de las etiquetas haciendo el 62,4 % (la secuencia completa: 84 %); calzado 122 337 tareas, 140 individuales y 40 grupos (22 de un solo SKU), el 0,5 % de las tareas en grupos y el 20 % de las etiquetas haciendo el 71,5 %. En calzado la hora no es fiable (768 marcas de tiempo para 122 370 líneas): el orden dentro de un mismo instante es el del archivo, en el que `orderToCollect` decrece dentro de cada orden. Una etiqueta agrupada es una familia, no un producto. Se subió `LIMITE_TAREAS` de 100 000 a 600 000 (el de una lista generada), porque el tope de datos reales no tenía justificación y dejaba fuera a calzado.
- **Dónde está en el código:** `asrs/generacion/fuentes/`, `scripts/extraer_demanda.py`, `leer_tareas` en `asrs/generacion/datos_reales.py`, `LIMITE_TAREAS` en `asrs/config.py`; pruebas en `tests/test_fuentes.py`.

## 2026-11-02 · Minimización de varianza: fondos antes que frentes entre canales
- **Contexto:** la regla solo desempataba por profundidad entre celdas de igual criterio, y el criterio (tiempo xy) ignora la profundidad. Eso la limitaba a cada canal: el frente del canal más cercano se ocupaba antes que el fondo de uno lejano, y con ocupación superior al 50 % la capa profunda no estaba llena.
- **Decisión:** la capa libre más profunda entre todas las candidatas (de la zona, en las políticas con zonas) se elige primero; el criterio de la política desempata dentro de esa capa y después el azar.
- **Evidencia:** Lehmann y Hußmann (2023) definen la varianza mínima como la estrategia que minimiza la varianza del llenado de los canales (resumen del artículo, ficha `lehmann2023`), y la varianza máxima como la contraria; el desempate solo por canal se parece más a la máxima. Lerher et al. (2010) la describen como llenar primero la segunda profundidad, con reubicación nula hasta α = 0,5, y la tesis (cap. 7, validación) usa ese umbral como referencia. Con el desempate por canal esa referencia no se cumple. Falta contrastar con el PDF de Lehmann 2023, que no está en el equipo (la ficha es una lectura del texto, no una verificación).
- **Alternativas descartadas:** dejar el desempate solo por canal (incumple la referencia de validación); añadir un peso de profundidad al criterio (agrega un parámetro sin respaldo en la literatura).
- **Efecto en los resultados:** cambian los números de **todas** las políticas clásicas, también `aleatoria` (ahora sortea entre los canales con la capa libre más profunda) y las zonificadas (fondos antes que frentes dentro de la zona). Las corridas anteriores a esta fecha no son comparables. El agente no cambia, pero su comparación con las clásicas sí.
- **Dónde está en el código:** `PoliticaAsignacion.asignar` en `asrs/politicas/asignacion.py`; prueba `test_minimizacion_de_varianza_llena_todos_los_fondos_antes_que_los_frentes`.

## 2026-11-01 · Salidas y experimentos fuera del código
- **Contexto:** los scripts escribían CSV y HTML en la carpeta desde la que se lanzaban y el `.gitignore` los listaba uno a uno.
- **Decisión:** todo lo generado va a `salidas/` (ignorada entera). Cada experimento se guarda como JSON en `experimentos/` y se lanza con `--config`.
- **Alternativas descartadas:** seguir editando `asrs/config.py` por experimento, porque no deja rastro de qué configuración produjo cada resultado.
- **Efecto en los resultados:** ninguno en los números; solo los hace reproducibles con un archivo.
- **Dónde está en el código:** `asrs/utilidades.py` (`ruta_salida`), `asrs/config.py` (`cargar_configuracion_archivo`), `experimentos/LEEME.md`.

## Comparación pareada entre políticas
- **Contexto:** comparar políticas con secuencias distintas mezclaría el efecto de la política con el del azar.
- **Decisión:** en cada réplica todas las políticas procesan la misma secuencia de llenado y la misma de operación, con la semilla `escalar_semilla + i`. Cada política tiene una semilla derivada de su política base, no de su posición en la lista.
- **Efecto en los resultados:** las diferencias entre políticas son pareadas y las variantes replanificadas parten con los mismos desempates al azar que su variante de plan fijo.
- **Dónde está en el código:** README, sección "Modelo de réplicas"; `asrs/politicas/registro.py`.
