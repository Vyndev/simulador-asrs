# Plan de implementación: simulador con el modelo dinámico de suministro y demanda

> **Para agentes:** SUB-SKILL REQUERIDA: usar superpowers:subagent-driven-development (recomendado) o superpowers:executing-plans para ejecutar este plan tarea por tarea. Los pasos usan casillas (`- [ ]`) para el seguimiento.

**Objetivo:** alinear `simulador-asrs/` con la formulación nueva del capítulo 6. Hay que lograr cinco cosas:
- demanda por ciclo de vida y estacionalidad;
- reposición con lotes EOQ;
- un año de operación con 30 días de calentamiento;
- siete políticas, con las variantes de plan fijo y de replanificación cada 30 días;
- ninguna tarea descartada.

**Arquitectura:**
- **Generación:** dos módulos nuevos en `asrs/generacion/`.
  - `ciclo_vida.py` arma la cartera de 180 posiciones con sus curvas.
  - `reposicion.py` arma, día por día, la prehistoria, la lista de llenado y la lista de operación, y lo entrega en un `EscenarioDemanda` (`escenario.py`).
- **Simulación:** `Simulacion` crea el escenario por réplica y se lo pasa a las políticas por la configuración. Al controlador le pasa los días de cada tarea y los retiros de fin de ciclo.
- **Formato de las listas:** sigue siendo `(actividad, sku)`. El código del SKU es la posición del surtido (1 a 180).

**Herramientas:** Python 3, NumPy y `unittest` (`.venv/bin/python -m unittest discover -s tests -q`).

**Especificación:**
- El diseño: `documento/disenos/2026-10-25-modelo-suministro-demanda.md`.
- El capítulo 6 ya reescrito, que manda sobre el código.
- Las "secciones 5.x" citadas son del diseño.

## Restricciones globales

- **Estilo:** el del código existente. Nombres con prefijo de tipo (`escalar_`, `vector_`, `matriz_`, `txt_`, `booleano_`, `dic_`), comentarios en español y la densidad de comentarios de los archivos vecinos.
- **Valores por defecto:** la instancia del capítulo 6, copiada tal cual:
  - p = 0.03, q = 0.38 y 18 meses de ciclo;
  - cuantiles de amplitud [0.05, 0.10, 0.20, 0.47, 0.67] en las probabilidades [0, 0.10, 0.50, 0.90, 1.0];
  - reparto de meses de pico [0.44, 0.37, 0.19] (diciembre a febrero, junio a agosto, resto);
  - 60 ciclos por hora y 24 horas por día;
  - ventana de 30 días y calentamiento de 30 días;
  - techo de 0.95;
  - `TAREAS_POR_CORRIDA = 525_600`.
- **Calendario:** el año evaluado empieza el 1 de enero. El día 0 del generador es el primero del calentamiento, y el día del año es `(dia - dias_calentamiento) mod 365`. Un mes son 365/12 días.
- **Reproducibilidad:** todo el azar del escenario sale de un `np.random.Generator` creado con `np.random.default_rng(rnd.getrandbits(64))`, después del `rnd.seed(semilla)` de la réplica. Misma semilla, mismo escenario.
- **Rendimiento:** un escenario de un año se genera en menos de 10 s. Los de las pruebas (72 tareas) tardan menos de 0,5 s.
- **Sin commits:** no hacer commit, push ni ramas nuevas; Kevin los pide. Se trabaja en la rama actual del simulador (`comparacion-clasicas`).

## Foco de revisión

1. **Lista más corta que un día** (72 tareas en las pruebas y por defecto en la herramienta web): exactamente N tareas, todas del día 0, y una fila de popularidad. Prueba en la tarea 2.
2. **Varios cierres de ciclo por posición dentro del horizonte** (ciclos cortos): el retiro deja la posición en cero antes del primer lote del sucesor, y la popularidad del sucesor arranca en cero. Prueba en la tarea 2.
3. **Techo que actúa a menudo** (ocupación media cercana al techo): los lotes se aplazan y se cuentan, la ocupación nunca pasa el techo, la demanda no atendida se cuenta y el generador no se bloquea. Prueba en la tarea 2.
4. **Estantería pequeña y demanda personalizada de la herramienta web** (6×6×2 con 10 SKU, o `vector_demanda_sku_personalizada`): sin errores y con lotes de al menos una unidad. Prueba en la tarea 2.
5. **Datos reales:** sin escenario, el controlador no da avisos de día ni cuenta retiros, y las políticas planifican con `vector_demanda_sku` como hoy. Prueba en la tarea 4.

---

### Tarea 1: Cartera de productos con ciclo de vida y estacionalidad

**Archivos:**
- Crear: `simulador-asrs/asrs/generacion/ciclo_vida.py`
- Crear: `simulador-asrs/tests/test_suministro_demanda.py` (con el mismo encabezado de `sys.path` de `tests/test_simulacion.py`)
- Modificar: `simulador-asrs/asrs/config.py`, agregando a `cargar_configuracion` las claves nuevas con sus valores por defecto. La validación en `config_desde_json` queda para la tarea 4.

**Interfaces:**
- **Consume:** `demanda_hausman(escalar_cantidad_skus, escalar_sigma)` de `asrs/generacion/probabilidades.py`.
- **Produce:**
  - **Claves de configuración, con sus valores por defecto:**
    - `texto_modelo_demanda="ciclo_vida"`;
    - `escalar_bass_p=0.03` y `escalar_bass_q=0.38`;
    - `escalar_meses_ciclo_vida=18`;
    - `vector_cuantiles_amplitud=np.array([0.05, 0.10, 0.20, 0.47, 0.67])` y `vector_probabilidades_amplitud=np.array([0, 0.10, 0.50, 0.90, 1.0])`;
    - `vector_reparto_mes_pico=np.array([0.44, 0.37, 0.19])`;
    - `escalar_ciclos_por_hora=60` y `escalar_horas_por_dia=24`;
    - `escalar_dias_ventana_popularidad=30` y `escalar_dias_calentamiento=30`;
    - `escalar_techo_ocupacion=0.95`.
  - Cada clave lleva un comentario con su fuente (diseño, sección 6).
  - **Constantes:** `DIAS_POR_ANIO = 365`, `DIAS_POR_MES = 365 / 12` y `MESES_POR_GRUPO_PICO = ((11, 0, 1), (5, 6, 7), (2, 3, 4, 8, 9, 10))`, con los meses numerados de 0 (enero) a 11.
  - `densidad_bass(vector_edad, escalar_duracion, escalar_razon_qp) -> np.ndarray`: la ecuación `eq:densidad-bass` con β = 2 ln r / L. Vale 0 fuera de [0, L). Si la duración es `np.inf`, devuelve unos.
  - `factor_estacional(escalar_dia_del_anio, vector_amplitud, vector_dia_pico) -> np.ndarray`: la ecuación `eq:factor-estacional`.
  - `muestrear_amplitudes(generador, escalar_n, vector_cuantiles, vector_probabilidades) -> np.ndarray`: `np.interp(U(0,1), vector_probabilidades, vector_cuantiles)`.
  - `muestrear_dias_pico(generador, escalar_n, vector_reparto) -> np.ndarray`: el grupo sale de `vector_reparto`; el mes, uniforme dentro del grupo; el día, `(mes + U(0,1)) · DIAS_POR_MES`.
  - `participacion_superior(vector_participacion, escalar_fraccion=0.2) -> float`: participación de los `floor(fraccion · n)` mayores.
  - `class CarteraProductos(vector_altura, vector_edad_inicial, escalar_duracion, escalar_razon_qp, vector_amplitud, vector_dia_pico, escalar_dia_inicio_anio)` con:
    - los métodos `edades(dia)`, `generaciones(dia)` (ciclos completos desde el día 0), `intensidades(dia)` (μ) y `participaciones(dia)` (ρ);
    - el atributo `escalar_sigma_alturas`.
  - `calibrar_sigma_alturas(matriz_base, vector_orden_alturas, escalar_objetivo) -> float`.
  - `crear_cartera(config, generador) -> CarteraProductos`.

- [ ] **Paso 1: Escribir las pruebas que fallan** en la clase `PruebasCicloDeVida`. Con `L = 18 * 365 / 12` y `r = 0.38 / 0.03`:
  - `test_densidad_pico_simetria_y_area`: en una malla de 20 001 puntos sobre [0, L):
    - el máximo está en `L/2 ± 0.002 L`;
    - `g(0) / g(L/2)` vale `0.2710 ± 0.001`;
    - `g(0) ≈ g(L - 1e-9)` con `rtol=1e-6`;
    - la integral por trapecios es `1 - 1/r ± 0.002`, es decir 0,921.
  - `test_densidad_fuera_del_ciclo_y_ciclo_infinito`: `densidad_bass([-1, L], L, r)` da `[0, 0]`, y `densidad_bass([0, 500], np.inf, r)` da `[1, 1]`.
  - `test_factor_estacional_amplitud_pico_y_periodo`: con A = 0.2 y pico en el día 100:
    - el máximo sobre los días 0 a 364 cae en el día 100;
    - `log(max/min)` vale `0.2 ± 1e-3`;
    - el día 365 vale lo mismo que el día 0.
  - `test_amplitudes_siguen_los_cuantiles`: con 200 000 muestras, los percentiles 10, 50 y 90 dan 0.10, 0.20 y 0.47 (`± 0.005`); el mínimo es al menos 0.05 y el máximo a lo sumo 0.67.
  - `test_meses_de_pico`: con 200 000 muestras, la fracción con mes en {11, 0, 1} es `0.44 ± 0.005` y en {5, 6, 7} es `0.37 ± 0.005`. El mes es `floor(dia / DIAS_POR_MES)`.
  - `test_cartera_estatica_es_hausman`: `crear_cartera` con `texto_modelo_demanda="estatica"` da `participaciones(0)` y `participaciones(200)` iguales a `demanda_hausman(180, 0.4)`, con `allclose`.
  - `test_sucesion_aumenta_la_generacion`: si una posición tiene edad L − 0.5 en el día 0, su generación es 0 en el día 0 y 1 en el día 1, y su edad en el día 1 es 0.5.
  - `test_calibracion_da_la_concentracion_de_sigma_04`: `crear_cartera(config_por_defecto, np.random.default_rng(3))` da una media de `participacion_superior(participaciones(d))` para d de 30 a 394 igual a `0.2 ** 0.4 ± 0.003`, y `escalar_sigma_alturas` entre 0.38 y 0.48.

- [ ] **Paso 2: Ejecutar y ver que fallan**

Ejecutar: `cd simulador-asrs && .venv/bin/python -m unittest discover -s tests -p "test_suministro_demanda.py" -q`
Resultado esperado: FAIL o ERROR por `ModuleNotFoundError: asrs.generacion.ciclo_vida`.

- [ ] **Paso 3: Implementar `asrs/generacion/ciclo_vida.py`**
  - **Edades:** `edades(dia) = (vector_edad_inicial + dia) mod L` (o la edad inicial fija si L es infinita), y `generaciones(dia) = floor((vector_edad_inicial + dia) / L)`.
  - **Intensidades:** `intensidades(dia) = altura · densidad_bass(edades) · factor_estacional((dia - dia_inicio_anio) mod 365, ...)`.
  - **Calibración:** `calibrar_sigma_alturas` hace bisección en σ ∈ [0.05, 1.0] con la función `media_filas(participacion_superior(matriz_base · demanda_hausman(n, σ)[orden]))`, que decrece con σ. Hasta `|media − objetivo| ≤ 1e-4` o 30 iteraciones. `matriz_base` tiene forma (365, n) y es g·ω del año evaluado.
  - **`crear_cartera`** saca del generador, en este orden: la permutación de edades, las amplitudes, los días de pico y la permutación de alturas.
    - Las edades se estratifican: `(perm + 0.5) / n · L`.
    - **Modelo `estatica`:** L = `np.inf`, amplitudes 0, alturas `demanda_hausman(n, σ)` en orden de SKU, sin permutación, y `escalar_sigma_alturas = σ`.
    - **Demanda personalizada** (`vector_demanda_sku_personalizada` no vacío): las alturas son ese vector normalizado, en orden de SKU y sin calibrar.

- [ ] **Paso 4: Ejecutar y ver que pasan.** El mismo comando del paso 2 debe dar `OK`.

---

### Tarea 2: Escenario de demanda y reposición

**Archivos:**
- Crear: `simulador-asrs/asrs/generacion/escenario.py`
- Crear: `simulador-asrs/asrs/generacion/reposicion.py`
- Modificar: `simulador-asrs/tests/test_suministro_demanda.py`, agregando la clase `PruebasEscenarioDemanda`.

**Interfaces:**
- **Consume:** `crear_cartera` y `participacion_superior` de la tarea 1.
- **Produce:**
  - `@dataclass EscenarioDemanda` con los campos:
    - `lista_llenado: list`, de tuplas `(0, sku)`;
    - `lista_operacion: list`, de tuplas `(actividad, sku)`, con exactamente `escalar_largo_secuencia` tareas;
    - `vector_dia_de_tarea: list`, día de cada tarea de la operación, con 0 el 1 de enero;
    - `conjunto_retiros: set`, índices de `lista_operacion` que son retiros de fin de ciclo;
    - `matriz_popularidad: np.ndarray`, de forma (días de operación generados, n) y enteros, con la popularidad móvil al empezar cada día;
    - `vector_inventario_inicial: np.ndarray`, las unidades de cada posición al final del calentamiento;
    - `resumen: dict`, con las claves `demanda_no_atendida`, `lotes_aplazados`, `retiros`, `lanzamientos`, `ocupacion_media`, `ocupacion_minima`, `ocupacion_maxima`, `participacion_top20_media`, `sigma_alturas` y `dias_operacion`.
  - El método `popularidad(escalar_dia) -> np.ndarray`, que devuelve la fila del día, acotada a la última generada.
  - `recuperaciones_diarias(config) -> int`, igual a `round(ciclos_por_hora · horas_por_dia / 2)` (720).
  - `calibrar_kappa(vector_demanda_esperada, escalar_inventario_objetivo) -> float`: bisección para que `Σ (lotes_eoq + 1) / 2` quede lo más cerca posible del objetivo.
  - `lotes_eoq(vector_demanda_esperada, escalar_kappa) -> np.ndarray`, igual a `max(1, rint(κ · √ν))` como enteros.
  - `generar_escenario(config, generador) -> EscenarioDemanda`.

- [ ] **Paso 1: Escribir las pruebas que fallan.** Todas con la configuración por defecto y `escalar_largo_secuencia = 60 * 1440`, salvo que se diga otra cosa. El generador es `np.random.default_rng(11)`.
  - `test_kappa_y_lotes_del_caso_estatico`: con `ν = 720 · demanda_hausman(180, 0.4)` y objetivo 612:
    - `|Σ (Q + 1) / 2 − 612| ≤ 3`;
    - `min(Q) ≥ 1`;
    - `Q[0]` está en [29, 33] y la mediana de Q en [4, 6].
  - `test_largo_y_dias`: `len(lista_operacion) == len(vector_dia_de_tarea) == 86400`; los días no decrecen y empiezan en 0.
  - `test_misma_semilla_mismo_escenario`: dos escenarios con `default_rng(11)` tienen listas y `resumen` iguales.
  - `test_inventario_nunca_negativo_ni_sobre_el_techo`:
    - recorriendo `lista_llenado` y luego `lista_operacion` con un contador por SKU, ninguna recuperación encuentra cero unidades;
    - el total nunca pasa `floor(0.95 · 720) = 684`;
    - `len(lista_llenado) == vector_inventario_inicial.sum()`.
  - `test_ocupacion_media_cerca_del_85`: `resumen["ocupacion_media"] / 720` está en `0.85 ± 0.03`.
  - `test_recuperaciones_de_demanda_por_dia`: sin contar los retiros, la media de recuperaciones de los días completos es `720 ± 2` y cada día completo está en [700, 740].
  - `test_mezcla_cerca_del_50`: los almacenamientos son entre el 45 % y el 55 % de la operación.
  - `test_demanda_no_atendida_excepcional`: `resumen["demanda_no_atendida"]` es a lo sumo 0.001 de las recuperaciones de demanda.
  - `test_popularidad_movil`: para el día 45 y una posición sin cierre entre los días 15 y 45, `matriz_popularidad[45, i - 1]` es igual a las recuperaciones de i que no son retiros en los días 15 a 44 de la lista de operación.
  - `test_retiros_vacian_la_posicion`:
    - cada índice de `conjunto_retiros` es una tupla `(1, i)`;
    - después del último retiro de un bloque, el contador de i vale 0, y la tarea siguiente de i, si existe, es un almacenamiento.
  - `test_modelo_estatico_sin_retiros_ni_lanzamientos`: con `texto_modelo_demanda="estatica"`, `conjunto_retiros` está vacío y `resumen["lanzamientos"]` es 0.
  - **Foco 1,** `test_lista_corta`: con `escalar_largo_secuencia = 72`, hay 72 tareas, todos los días valen 0 y `matriz_popularidad.shape[0]` es 1.
  - **Foco 2,** `test_varias_sucesiones`:
    - con `escalar_meses_ciclo_vida = 1` y `escalar_largo_secuencia = 90 * 1440`, `resumen["lanzamientos"]` es al menos 180 · 2;
    - la prueba de los retiros vale para todos los bloques;
    - `popularidad` de la posición, el día siguiente a un relanzamiento, cuenta solo días del sucesor.
  - **Foco 3,** `test_techo_activo`: con `escalar_nivel_ocupacion = 0.93` el generador termina, el total nunca pasa 684 y `resumen["lotes_aplazados"]` es mayor que 0.
  - **Foco 4,** `test_estanteria_pequena_y_demanda_personalizada`:
    - con `vector_configuracion_estanteria = [6, 6, 2]`, `escalar_numero_skus = 10` y `escalar_largo_secuencia = 2000`, termina sin errores y ningún lote vale menos de 1;
    - lo mismo con `vector_demanda_sku_personalizada = [5, 3, 2]` y `escalar_numero_skus = 3` sobre esa estantería.

- [ ] **Paso 2: Ejecutar y ver que fallan**

Ejecutar: `cd simulador-asrs && .venv/bin/python -m unittest discover -s tests -p "test_suministro_demanda.py" -q`
Resultado esperado: ERROR por `ModuleNotFoundError: asrs.generacion.reposicion`.

- [ ] **Paso 3: Implementar `escenario.py` y `reposicion.py`.**

Este es el algoritmo que no determinan las pruebas. Las cantidades salen de la configuración:
- celdas = F·C·K;
- objetivo = φ · celdas;
- techo = `floor(φ̄ · celdas)`;
- R = `recuperaciones_diarias(config)`;
- N = `escalar_largo_secuencia`;
- calentamiento = `escalar_dias_calentamiento`;
- W = `escalar_dias_ventana_popularidad`.

1. **Preparación**
   - Crear la cartera con `crear_cartera(config, generador)`.
   - Precalcular la matriz ν = R · participaciones para los días 0 a `calentamiento + ceil(N / R) + ceil(L) + 1`, sin `ceil(L)` si L es infinita, y sus sumas acumuladas por posición, para la demanda restante hasta el cierre.
   - Sacar del generador un desfase uniforme en [0, 1) por posición, para el redondeo acumulado.
2. **Día 0.** El inventario de cada posición es `ceil(lotes_eoq(ν(0), κ(0)) / 2)`.
3. **Cada día d**, en este orden:
   - (a) Si d es igual a `calentamiento`:
     - copiar el inventario a `vector_inventario_inicial`;
     - armar `lista_llenado` en orden aleatorio;
     - desde aquí las tareas van a `lista_operacion`, con día `d - calentamiento`.
   - (b) Calcular `κ(d) = calibrar_kappa(ν(d), objetivo)` y `Q(d) = lotes_eoq(ν(d), κ(d))`.
   - (c) **Cierres.** Para cada posición cuya generación aumentó en d, en orden de posición:
     - emitir sus unidades como recuperaciones (y, en la operación, agregar sus índices a `conjunto_retiros`);
     - dejar su inventario en 0, cancelar su lote pendiente y reiniciar su historia de popularidad;
     - insertar el lote inicial del sucesor, de tamaño `Q_i(d)`, y contar un lanzamiento si d está en la operación.
   - (d) Si d está en la operación, anotar la fila de popularidad del día: recuperaciones de demanda de la generación actual en los días `[max(lanzamiento, d − W), d − 1]`.
   - (e) **Recuperaciones del día.** La cantidad de cada posición es `floor(E_i(d) + desfase_i) − floor(E_i(d−1) + desfase_i)`, con E la ν acumulada. Se ponen en orden aleatorio y se procesan una por una:
     - si la posición no tiene unidades, se suma 1 a la demanda no atendida (solo en la operación) y se pasa a la siguiente;
     - si tiene, se emite `(1, i)`, se resta 1 al inventario y se cuenta en la popularidad del día;
     - si el inventario quedó en 0, se calcula la demanda esperada restante hasta el cierre de la generación (infinita en el modelo estático) menos lo ya emitido hoy de i. El lote es `Q_i(d)` si esa demanda es al menos `Q_i(d)`; `ceil(restante)` si es mayor que 0; y ninguno si no.
     - después de cada tarea emitida se intentan los lotes pendientes en orden de llegada, mientras quepan bajo el techo.
   - (f) **Insertar un lote** de tamaño `lote` para i: si `total + lote ≤ techo`, se emiten `lote` tuplas `(0, i)` seguidas; si no, el lote queda pendiente y se cuenta como lote aplazado (solo en la operación).
   - (g) El generador se detiene en cuanto la operación llega a N tareas, aunque sea a mitad de día.
4. **Resumen.**
   - La ocupación se mide con el total después de cada tarea de la operación.
   - `participacion_top20_media` es la media, en los días de operación generados, de `participacion_superior(participaciones(d))`.
   - `sigma_alturas` sale de la cartera.

- [ ] **Paso 4: Ejecutar y ver que pasan.** El mismo comando del paso 2 debe dar `OK`. Medir además el tiempo de un año:

Ejecutar: `cd simulador-asrs && .venv/bin/python -c "import time,numpy as np; from asrs.config import cargar_configuracion; from asrs.generacion.reposicion import generar_escenario; c=cargar_configuracion(); c['escalar_largo_secuencia']=525600; t=time.time(); e=generar_escenario(c, np.random.default_rng(1)); print(round(time.time()-t,1), e.resumen)"`
Resultado esperado:
- menos de 10 s;
- `ocupacion_media` entre 586 y 638;
- `participacion_top20_media` en 0.525 ± 0.005;
- `demanda_no_atendida` a lo sumo 0.001 de las recuperaciones.

---

### Tarea 3: Controlador, métricas y resultados

**Archivos:**
- Modificar: `simulador-asrs/asrs/control/controlador_grua.py`, en el constructor y en `ejecutarSecuencia`.
- Modificar: `simulador-asrs/asrs/politicas/asignacion.py`, agregando `iniciarDia`.
- Modificar: `simulador-asrs/asrs/metricas.py`, en el constructor, un registro nuevo y `recuperarInformcion`.
- Modificar: `simulador-asrs/asrs/simulacion/resultados.py`, en el encabezado.
- Modificar: `simulador-asrs/tests/test_simulacion.py`, agregando la clase `PruebasControladorDinamico` y ajustando `test_replica_reproducible_y_completa`.

**Interfaces:**
- **Produce:**
  - `ControladorGrua(E, G, P, S, politica_asignacion, politica_recuperacion=None, vector_dia_de_tarea=None, conjunto_retiros=None)`.
  - `PoliticaAsignacion.iniciarDia(self, escalar_dia) -> None`, que no hace nada por defecto.
  - `Metricas.registrarRetiro(self, tiempo)`.
  - La fila de métricas pasa a `[..., I0, I1, I2, R0, R1, O0, O1]`, con 31 valores, o 33 con los nombres. Los índices negativos −1 y −2 siguen siendo el tiempo total y las operaciones.

- [ ] **Paso 1: Escribir las pruebas que fallan**
  - `test_controlador_avisa_el_inicio_de_cada_dia`:
    - usar una subclase de `PoliticaMasCercanaAlES` que anota los días en una lista;
    - la secuencia tiene 6 tareas válidas y `vector_dia_de_tarea = [0, 0, 1, 1, 1, 2]`;
    - los días anotados son `[0, 1, 2]`;
    - sin `vector_dia_de_tarea` no se anota ninguno.
  - `test_retiros_se_cuentan_aparte`:
    - se guarda una unidad de SKU 1 y se recupera con `conjunto_retiros = {1}`;
    - en la fila, la cantidad de retiros es 1, el tiempo de retiros es igual al de la recuperación y la recuperación sigue contándose en su columna.
  - `test_zona_llena_se_relaja_al_almacenar`:
    - estantería [2, 3, 2] con 3 SKU y política `full_turnover`;
    - se llenan a mano todas las celdas de la zona del SKU 1;
    - un almacenamiento de SKU 1 se ejecuta, porque sube la cantidad de almacenamientos, y queda fuera de su zona;
    - un almacenamiento de SKU 2 queda dentro de la suya.
  - En `test_replica_reproducible_y_completa`: `len(r1[0]) == 33`.
  - `test_encabezado_de_resultados`: `guardar_resultados_csv([], ruta_temporal)` escribe 33 columnas; las dos anteriores a `Cant_Operaciones` son `Cant_RetirosFinCiclo` y `TiempoTotal_RetirosFinCiclo`.
  - `test_politicas_con_zonas_respetan_su_zona`: con la regla de zona llena, pasa a comprobar que, al terminar el llenado, toda unidad fuera de su zona es de un SKU o clase cuya zona no tiene ninguna celda libre accesible. Durante el llenado no hay recuperaciones, así que una zona que se llenó sigue llena.

- [ ] **Paso 2: Ejecutar y ver que fallan**

Ejecutar: `cd simulador-asrs && .venv/bin/python -m unittest discover -s tests -p "test_simulacion.py" -q`
Resultado esperado: FAIL o ERROR en las pruebas nuevas.

- [ ] **Paso 3: Implementar**
  - `ejecutarSecuencia` recorre con índice.
    - Con `vector_dia_de_tarea`, llama `politica.iniciarDia(dia)` antes de la primera tarea y cada vez que el día cambia.
    - En un almacenamiento, si `asignar(sku)` no devuelve celda, reintenta con `booleano_restringir_zona=False` antes de descartar (regla de zona llena).
    - En una recuperación cuyo índice está en `conjunto_retiros`, registra con `registrarRetiro` la diferencia de `tiempo_total_recuperacion` antes y después de recuperar.

- [ ] **Paso 4: Ejecutar toda la suite.**

Ejecutar: `cd simulador-asrs && .venv/bin/python -m unittest discover -s tests -q`
Resultado esperado: `OK`.

---

### Tarea 4: Configuración, simulación y visualización con el escenario

**Archivos:**
- Modificar: `simulador-asrs/asrs/config.py`, en `TAREAS_POR_CORRIDA` y `config_desde_json`.
- Modificar: `simulador-asrs/asrs/simulacion/simulacion.py`, en `__init__`, `generarSecuencias`, `prepararEscenario`, `ejecutarEscenario`, `cargarReplica` y `ejecutarSimulacion`, más los métodos nuevos.
- Modificar: `simulador-asrs/asrs/simulacion/resultados.py`, agregando `guardar_resumen_escenarios_csv`.
- Modificar: `simulador-asrs/asrs/visualizacion/generador.py`, en `grabar_escenario` y `generar_datos`.
- Modificar: `simulador-asrs/.gitignore`, agregando `resultados_escenario.csv` junto a `resultados_simulacion.csv`.
- Modificar: `simulador-asrs/tests/test_simulacion.py`.

**Interfaces:**
- **Consume:** `generar_escenario` (tarea 2), las claves de configuración (tarea 1) y los argumentos nuevos del controlador (tarea 3).
- **Produce:**
  - `TAREAS_POR_CORRIDA = 525_600`, comentada como "un año de 365 días a 1 440 tareas por día".
  - **Métodos nuevos de `Simulacion`:** `escenario_demanda` (atributo, `None` al crear la instancia), `configPolitica() -> dict` y `argumentosOperacion(escalar_tareas) -> dict`.
    - `configPolitica()` devuelve `dict(self.config, escenario_demanda=self.escenario_demanda)`.
    - `argumentosOperacion(n)` devuelve `{}` si no hay escenario, o `{"vector_dia_de_tarea": dias[:n], "conjunto_retiros": retiros}` si lo hay.
  - `Simulacion.cargarReplicaConEscenario(vector_argumentos_replica) -> (filas, dic_resumen)`.
    - `dic_resumen` es el `resumen` del escenario más la clave `semilla`, o `{"semilla": s}` con datos reales.
    - `cargarReplica` la llama y devuelve solo las filas, como hoy.
  - `guardar_resumen_escenarios_csv(vector_resumenes, ruta='resultados_escenario.csv')`: una fila por réplica, con las claves del resumen como columnas y `;` como separador. `ejecutarSimulacion` la llama después de `guardar_resultados_csv`.

- [ ] **Paso 1: Ajustar y escribir las pruebas**
  - `test_la_instancia_por_defecto_es_la_del_capitulo_6`:
    - `escalar_largo_secuencia == 525_600`;
    - las claves nuevas con los valores de la tarea 1.
    - La lista de políticas se comprueba en la tarea 5.
  - En `test_rechaza_valores_invalidos`, que cada uno de estos casos lance `ValueError`:
    - `{"texto_modelo_demanda": "otro"}`;
    - `{"escalar_bass_q": 0.01}` (q ≤ p);
    - `{"escalar_techo_ocupacion": 0.8}` (techo ≤ ocupación);
    - `{"escalar_horas_por_dia": 25}`;
    - `{"vector_cuantiles_amplitud": [0.2, 0.1, 0.3, 0.4, 0.5]}`;
    - `{"vector_reparto_mes_pico": [1, -1, 1]}`;
    - `{"escalar_dias_calentamiento": 10}` (menor que la ventana).
  - `test_ida_y_vuelta_conserva_los_valores`: también conserva `texto_modelo_demanda`, `escalar_meses_ciclo_vida` y `vector_cuantiles_amplitud`.
  - `test_generar_secuencias_usa_el_escenario`: con la configuración de prueba (72 tareas):
    - `SA == simulacion.escenario_demanda.lista_llenado`;
    - `S == simulacion.escenario_demanda.lista_operacion`;
    - `len(S) == 72`.
  - **Foco 5,** `test_datos_reales_sin_escenario`: con el CSV de `PruebasDatosReales`, `escenario_demanda is None` y `argumentosOperacion(10) == {}`.
  - `test_resumen_de_escenario_por_replica`: `Simulacion.cargarReplicaConEscenario((config_de_prueba, 7))` devuelve las mismas filas que `cargarReplica((config_de_prueba, 7))` y un resumen con las claves `semilla`, `ocupacion_media` y `demanda_no_atendida`. `guardar_resumen_escenarios_csv([resumen], ruta_temporal)` escribe un encabezado con esas claves y una fila.
  - `test_llenado_alcanza_la_ocupacion_fijada` pasa a llamarse `test_llenado_reproduce_el_inventario_del_calentamiento`: para cada política, las unidades guardadas son igual a `len(SA)`, porque con la regla de zona llena tampoco la dedicada descarta.
  - `test_el_llenado_sigue_el_perfil_de_inventario` pasa a comprobar que el conteo de SA por SKU es igual a `escenario_demanda.vector_inventario_inicial`.
  - `preparar_estado` construye la política con `simulacion.configPolitica()`.

- [ ] **Paso 2: Ejecutar y ver que fallan**

Ejecutar: `cd simulador-asrs && .venv/bin/python -m unittest discover -s tests -p "test_simulacion.py" -q`
Resultado esperado: FAIL en las pruebas ajustadas.

- [ ] **Paso 3: Implementar**
  - **`config_desde_json`** valida, con el estilo de `_numero` y `_vector` y mensajes en español:
    - el modelo en {"ciclo_vida", "estatica"};
    - 0 < p ≤ 1, 0 < q ≤ 5 y q > p;
    - meses en (0, 600];
    - cuantiles: 5 valores en [0, 3], no decrecientes;
    - probabilidades: 5 valores de 0 a 1, estrictamente crecientes;
    - reparto: 3 valores ≥ 0 con suma > 0, que se normalizan;
    - ciclos por hora en (0, 1000] y horas en (0, 24];
    - ventana entera en [1, 365] y calentamiento entero en [ventana, 730];
    - techo en (ocupación, 1].
  - **`generarSecuencias`:**
    - con datos reales hace lo de hoy y deja `escenario_demanda = None`;
    - con lista generada crea el escenario con `np.random.default_rng(rnd.getrandbits(64))` y devuelve `(lista_llenado, lista_operacion)`.
  - **`prepararEscenario`** usa `configPolitica()`.
  - **`ejecutarEscenario`** y **`grabar_escenario`** crean el controlador de la operación con `**argumentosOperacion(len(S))`.

- [ ] **Paso 4: Ejecutar toda la suite.**

Ejecutar: `cd simulador-asrs && .venv/bin/python -m unittest discover -s tests -q`
Resultado esperado: `OK`.

---

### Tarea 5: Políticas con zonas que planifican con la popularidad, en dos variantes

**Archivos:**
- Modificar: `simulador-asrs/asrs/politicas/asignacion.py`, agregando `demandaPlanificacion`.
- Modificar: `simulador-asrs/asrs/politicas/basada_en_clases.py` y `full_turnover.py`, extrayendo `planificar` y agregando las subclases que replanifican.
- Modificar: `simulador-asrs/asrs/politicas/registro.py`.
- Modificar: `simulador-asrs/asrs/config.py`, en la lista por defecto de políticas, la clave nueva `escalar_dias_replanificacion` y su validación en `config_desde_json`.
- Modificar: `simulador-asrs/tests/test_suministro_demanda.py`, agregando la clase `PruebasPoliticasConPopularidad`, y `tests/test_simulacion.py`.

**Interfaces:**
- **Consume:** `escenario.popularidad(dia)`, que es lo único que las políticas usan del escenario, y `iniciarDia` de la tarea 3.
- **Produce:**
  - `PoliticaAsignacion.demandaPlanificacion(self, escalar_dia=0) -> np.ndarray`: `escenario.popularidad(dia)` como float si `config.get("escenario_demanda")` existe; si no, `config["vector_demanda_sku"]`.
  - `PoliticaBasadaEnClases.planificar(self, vector_demanda)` y `PoliticaFullTurnover.planificar(self, vector_demanda)`: el código actual del constructor, con `vector_demanda` como parámetro. El constructor llama `self.planificar(self.demandaPlanificacion(0))`.
  - **Clave de configuración nueva:** `escalar_dias_replanificacion = 30`, comentada con su fuente (periodo igual a la ventana de la popularidad móvil, Manzini et al. 2015; la asignación se recalcula una vez por periodo, Roodbergen y Vis 2009). `config_desde_json` la valida como entero en [1, 365].
  - `PoliticaBasadaEnClasesReplanificada` (nombre `'BasadaEnClasesReplanificada'`) y `PoliticaFullTurnoverReplanificada` (nombre `'FullTurnoverReplanificada'`): redefinen `iniciarDia(dia)` para que, cuando hay escenario y `dia > 0 and dia % config["escalar_dias_replanificacion"] == 0`, ejecute `self.planificar(self.demandaPlanificacion(dia))`; en los demás días no hacen nada. Durante el llenado y los primeros 30 días usan el plan inicial.
  - **Claves del registro:** `'basada_en_clases_replanificada'` y `'full_turnover_replanificada'`.
  - **Lista por defecto:** `('aleatoria', 'basada_en_clases', 'basada_en_clases_replanificada', 'full_turnover', 'full_turnover_replanificada', 'mas_cercana_es', 'mas_cercana_grua')`.

- [ ] **Paso 1: Escribir las pruebas que fallan.** Usan un objeto mínimo con `popularidad(dia)`: para `dia < 30` devuelve una popularidad decreciente con la posición (la posición 1 es la más popular) y para `dia >= 30`, la inversa.
  - `test_variantes_registradas`: `POLITICAS_ASIGNACION` tiene las siete claves y `crear_politica` construye cada una con y sin escenario.
  - `test_plan_fijo_no_cambia_sus_zonas`: con `basada_en_clases` y `full_turnover`, `matriz_zonificacion` sigue igual después de `iniciarDia(30)` y de `iniciarDia(60)`.
  - `test_replanificada_cambia_sus_zonas_solo_cada_30_dias`: en ambas variantes, después de `iniciarDia(0)` y de `iniciarDia(15)` la zonificación sigue igual que al construir la política; después de `iniciarDia(30)` la etiqueta de la celda más cercana al E/S es la de la última posición (clase 1 de la basada en clases, zona del SKU n de la dedicada; antes era la de la posición 1). Con `escalar_dias_replanificacion = 10` en la configuración, `iniciarDia(10)` ya replanifica pero con el stub (`dia < 30`) no cambia nada: usa para este caso un stub que invierte en `dia >= 10`.
  - `test_planifica_con_la_popularidad_del_primer_dia`: `vector_clase_de_sku` de la basada en clases es igual a `clasificar_skus(popularidad(0), config)`.
  - En `test_simulacion.py`:
    - `test_rechaza_valores_invalidos` incluye `{"escalar_dias_replanificacion": 0}` y `{"escalar_dias_replanificacion": 400}`; la instancia por defecto trae 30;
    - `test_replica_reproducible_y_completa` espera los nombres `['Aleatoria', 'BasadaEnClases', 'BasadaEnClasesReplanificada', 'FullTurnover', 'FullTurnoverReplanificada', 'MasCercanaAlES', 'MasCercanaALaGrua']`;
    - `test_politicas_con_zonas_respetan_su_zona` (ya ajustada en la tarea 3) cubre las cuatro claves con zonas;
    - `test_la_instancia_por_defecto_es_la_del_capitulo_6` comprueba `vector_politicas_asignacion` con las siete claves, en el orden de la lista por defecto.

- [ ] **Paso 2: Ejecutar y ver que fallan**

Ejecutar: `cd simulador-asrs && .venv/bin/python -m unittest discover -s tests -q`
Resultado esperado: FAIL o ERROR en las pruebas nuevas.

- [ ] **Paso 3: Implementar** según las interfaces. La herramienta web no necesita cambios, porque `config_desde_json` valida las claves contra el registro.

- [ ] **Paso 4: Ejecutar toda la suite**

Ejecutar: `cd simulador-asrs && .venv/bin/python -m unittest discover -s tests -q`
Resultado esperado: `OK`.

---

### Tarea 6: Verificación de punta a punta y documentación

**Archivos:**
- Modificar: `simulador-asrs/README.md`, en la sección que describe la demanda y la lista de tareas. Hay que describir el modelo nuevo en pocas líneas, con referencia al capítulo 6, y listar las claves nuevas de configuración.

- [ ] **Paso 1: Suite completa.** Ejecutar `cd simulador-asrs && .venv/bin/python -m unittest discover -s tests -q`; debe dar `OK`.
- [ ] **Paso 2: Una réplica de un año con las siete políticas**, en segundo plano, porque tarda unos 25 minutos en un núcleo:

```bash
cd simulador-asrs && .venv/bin/python -c "
import random as rnd, time
from asrs.config import cargar_configuracion
from asrs.simulacion.simulacion import Simulacion
c = cargar_configuracion(); s = Simulacion(c); rnd.seed(c['escalar_semilla'])
SA, S = s.generarSecuencias(); print(s.escenario_demanda.resumen, flush=True)
for i, pol in enumerate(c['vector_politicas_asignacion']):
    rnd.seed(c['escalar_semilla'] * 1000 + i); t = time.time()
    fila = s.ejecutarEscenario(pol, SA, S, 'fcfs'); print(pol, fila[-2], round(fila[-1]), round(time.time() - t), flush=True)
"
```

Resultado esperado:
- el resumen tiene la ocupación media entre 586 y 638 y la demanda no atendida a lo sumo 0.001 de las recuperaciones;
- todas las políticas ejecutan las 525 600 operaciones (`fila[-2]`), porque la demanda no atendida nunca entra a la lista y ninguna política descarta almacenamientos;
- cada política tarda menos de 10 minutos.

- [ ] **Paso 3: Herramienta web.**
  - `SALIDA=$(mktemp --suffix=.html) && .venv/bin/python visualizar.py --salida "$SALIDA" --replicas 1; rm -f "$SALIDA"` termina sin errores.
  - `PruebasServidor` y `PruebasVisualizacion` pasan en la suite.
- [ ] **Paso 4: Actualizar `README.md`.** Dejar todo sin commit y reportar `git -C simulador-asrs status --short`.
