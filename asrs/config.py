"""Parámetros del modelo de simulación."""
import numpy as np

from asrs.generacion.datos_reales import derivar_lista_demanda, es_solo_demanda, leer_dias, leer_tareas
from asrs.generacion.probabilidades import PERFILES_INVENTARIO, demanda_hausman

# Límites que protegen contra configuraciones que tardarían demasiado o no tienen sentido
LIMITE_CELDAS = 6000
LIMITE_TAREAS = 600_000 # Tareas de un archivo de datos reales: el mismo tope que una lista generada (un año de 525 600 tareas más margen)
LIMITE_TAREAS_GENERADAS = 600_000 # Tareas de una lista generada: un año de 525 600 tareas más margen
LIMITE_SKUS = 200
LIMITE_CLASES = 8
LIMITE_REPLICAS_ESTADISTICA = 2000

TAREAS_POR_CORRIDA = 525_600 # Cantidad de tareas evaluadas por corrida en la instancia del capítulo 6: un año de 365 días a 1 440 tareas por día
DIAS_HORIZONTE = 365 # Horizonte evaluado de la instancia, en días
DIAS_HISTORIAL = 30 # Días de historial de una lista real: los mismos del calentamiento y de la ventana de popularidad; no se evalúan
RECUPERACIONES_POR_CORRIDA = TAREAS_POR_CORRIDA // 2 # Mitad de las tareas con la mezcla 50 % almacenamientos / 50 % recuperaciones: la demanda de un año (R x Y = 720 x 365)


def recuperaciones_por_corrida(config):
    # Recuperaciones de demanda de un año evaluado: R = ciclos por hora x horas por día / 2 (mezcla 50 / 50) por día, durante el horizonte
    return int(round(config["escalar_ciclos_por_hora"] * config["escalar_horas_por_dia"] / 2)) * DIAS_HORIZONTE


def recuperaciones_con_historial(config):
    # Largo de una lista de demanda real: el historial más el año evaluado, R x (Y + B) = 720 x 395 = 284 400 recuperaciones
    return int(round(config["escalar_ciclos_por_hora"] * config["escalar_horas_por_dia"] / 2)) * (DIAS_HORIZONTE + DIAS_HISTORIAL)


def dias_lista_real(config):
    # Días del modelo de una lista real con días: el año evaluado más el historial, Y + B = 395
    return DIAS_HORIZONTE + int(config["escalar_dias_calentamiento"])


def demanda_varia_por_replica(config):
    # True si cada réplica tiene su propia demanda: la fuente se trata con la semilla de la réplica o la mezcla diaria se sortea con ella
    return config.get("texto_origen_tareas") == "fuente" or (config.get("texto_origen_tareas") == "datos" and config.get("texto_variabilidad", "ninguna") != "ninguna")


def lista_real_de_replica(config, escalar_semilla):
    """Claves de la configuración que salen de la lista real de una réplica: la secuencia de operación con su reposición, la demanda del historial,
    los SKU, los días y el resumen de la demanda. Con origen 'fuente' la fuente se trata con la semilla; con días y un nivel de variabilidad la mezcla
    diaria se sortea con ella. `config` ya está validada."""
    from asrs.generacion.demanda_replica import demanda_de_replica, omega_de_fuente, secuencia_de_fuente
    from asrs.generacion.variabilidad import estimar_omega, recuperaciones_por_dia
    dic_resumen = {}
    escalar_recuperaciones = int(round(config["escalar_ciclos_por_hora"] * config["escalar_horas_por_dia"] / 2)) * dias_lista_real(config)
    # ω̂ de la fuente: el fijado a mano o uno por fuente, estimado una vez (ver `omega_de_fuente`); con un CSV, el de la lista del archivo
    escalar_omega_fuente = config["escalar_omega_fuente"]
    if config["texto_origen_tareas"] == "fuente":
        resultado = secuencia_de_fuente(config["texto_fuente_demanda"], config["texto_carpeta_datos"], escalar_semilla,
                                        escalar_recuperaciones, dias_lista_real(config))
        if escalar_omega_fuente is None:
            escalar_omega_fuente = omega_de_fuente(config["texto_fuente_demanda"], config["texto_carpeta_datos"], escalar_recuperaciones,
                                                   dias_lista_real(config), int(config["escalar_dias_ventana_popularidad"]))
        matriz_tareas, vector_dias, bool_solo_demanda = [(1, txt_sku) for txt_sku in resultado.vector_skus], resultado.vector_dias, True
        dic_resumen.update(fuente=config["texto_fuente_demanda"], skus_fuente=resultado.dic_muestreo["skus"],
                           desplazamiento_dias=resultado.escalar_desplazamiento, dias_fuente=resultado.escalar_dias_fuente)
    else:
        matriz_tareas = leer_tareas(config["texto_tareas_csv"], LIMITE_TAREAS)
        vector_dias = leer_dias(config["texto_tareas_csv"], LIMITE_TAREAS)
        bool_solo_demanda = es_solo_demanda(config["texto_tareas_csv"])
        if vector_dias is not None and escalar_omega_fuente is None:
            escalar_omega_fuente = estimar_omega(recuperaciones_por_dia([txt_sku for _, txt_sku in matriz_tareas], vector_dias)[0],
                                                 config["escalar_dias_ventana_popularidad"])
    if vector_dias is None and config["texto_variabilidad"] != "ninguna":
        raise ValueError("Variabilidad: la Dirichlet necesita el día de cada recuperación; usa un CSV con columnas SKU y Dia o el origen 'fuente'.")
    if vector_dias is not None:
        vector_skus, vector_dias, dic_variabilidad = demanda_de_replica([txt_sku for _, txt_sku in matriz_tareas], vector_dias, config["texto_variabilidad"],
                                                                        config["escalar_dias_ventana_popularidad"], escalar_semilla, escalar_omega_fuente)
        matriz_tareas = [(1, txt_sku) for txt_sku in vector_skus]
        dic_resumen.update(dic_variabilidad)
    escalar_celdas = int(np.prod(config["vector_configuracion_estanteria"]))
    # Con días, el historial son las primeras R·B recuperaciones (R por día durante los B días de historial), 21 600 en la instancia
    escalar_recuperaciones_historial = int(round(config["escalar_ciclos_por_hora"] * config["escalar_horas_por_dia"] / 2)) * int(config["escalar_dias_calentamiento"])
    dic = derivar_lista_demanda(matriz_tareas, vector_dias, bool_solo_demanda, config["escalar_porcentaje_historial"], escalar_recuperaciones_historial,
                                int(round(config["escalar_ciclos_por_hora"] * config["escalar_horas_por_dia"] / 2)),
                                config["escalar_nivel_ocupacion"] * escalar_celdas, LIMITE_TAREAS)
    if dic["escalar_numero_skus"] > LIMITE_SKUS:
        raise ValueError(f"Datos reales: hay {dic['escalar_numero_skus']} SKUs distintos, el máximo es {LIMITE_SKUS}.")
    dic_resumen.update(skus=dic["escalar_numero_skus"], recuperaciones_historial=dic["escalar_tareas_historial"],
                       skus_con_historial=int((np.asarray(dic["vector_demanda_sku"]) > 0).sum()), dia_inicio_operacion=dic["escalar_dia_inicio_operacion"],
                       inventario_inicial=int(sum(dic["vector_inventario_inicial"] or [])), tareas_operacion=len(dic["secuencia_real"]))
    escalar_almacenes = sum(1 for escalar_actividad, _ in dic["secuencia_real"] if escalar_actividad == 0)
    dic["vector_prob_acum_AR"] = np.array([escalar_almacenes / len(dic["secuencia_real"]), 1.0])
    dic["escalar_largo_secuencia"] = len(dic["secuencia_real"])
    dic["dic_resumen_demanda"] = dic_resumen
    dic["escalar_semilla_demanda"] = escalar_semilla
    return dic


def cargar_configuracion():
    # Se cargan los párametros del modelo
    config = {

        # Nota en los párametros espaciales las componentes verticales y horizontales son intercambiables, solo se debe tener congruencia en todos los demás párametros

        # Se recomienda utilizar la convención de ejemplo, esto por que la matrices tradicionalmente se recorrecoren por filas y luego columnas
        # Sin embargo como se meciona, siempre y cuando se conserve congruencia las componente  verticales y horizontales son intercambiales sin implicaciones negativas

        "vector_posicion_estanteria": np.array([-1.0, 1.0, 0]), # (posición vertical, posición horizontal, posición de profundidad), el E/S queda 1 m sobre la base de la estantería y 1 m antes de la primera columna
        "vector_configuracion_estanteria": np.array([18, 20, 2]), # (filas, columnas, capas de profundidad)
        "vector_cotas_estanteria": np.array([1.67, 1.1, 1.2]), # (alto de fila, ancho de columna, profundid de celda), 30 m de altura en 18 niveles, bahía de 1,1 m por 1,2 m
        "vector_centro_celda_estanteria": np.array([0.5, 0.5, 0.5]), # (% verical, % horizontal, % profundidad), representa un traslación del punto de referencia dentro de la celda
        "escalar_nivel_ocupacion": 0.85, # Porcentaje de la estatería que esta ocupada, intervalo [0, 1]
        "texto_perfil_inventario": "eoq", # Cuánto inventario tiene cada SKU: 'eoq' (lote económico, crece con la raíz de la demanda) o 'demanda' (proporcional)
        "escalar_separacion_riel_grua": 0, # Unidades de separación entre la cara frontal de la estantería y el riel de la grúa
        "vector_velocidad_grua": np.array([1, 1.5, 0.7]), #(Velocidad en filas, velocidad en columnas, velocidad a lo largo de canales)
        "vector_aceleracion_grua": np.array([0.5, 0.5, 0]), # Aceleración de cada eje (filas, columnas, canales), 0 significa sin límite: el eje alcanza la velocidad al instante
        "vector_deceleracion_grua": np.array([0.5, 0.5, 0]), # Deceleración de cada eje, 0 significa sin límite
        "escalar_tiempo_toma_deposito_es": 3.43, # Tiempo de tomar o depositar un ítem en el punto E/S
        "escalar_tiempo_toma_deposito_capa_delantera": 3.43, # Tiempo de tomar o depositar un ítem en la capa de la estantería más cercana al pasillo
        "escalar_tiempo_toma_deposito_capa_trasera": 6.86, # Tiempo de tomar o depositar un ítem en las capas más profundas

        # Demanda por SKU: la lista de tareas solo conoce SKUs, el SKU 1 es el más demandado
        "escalar_numero_skus": 180, # Surtido medio de productos activos, relación de 4 celdas por producto; nacen n / L por día y el surtido es abierto
        "escalar_sigma_hausman": 0.4, # Sesgo de la demanda según la curva de Hausman, 1 es demanda uniforme y menor es más concentrada
        "vector_demanda_sku_personalizada": None, # Opcional, demanda de cada SKU, reemplaza a los dos parámetros anteriores
        "vector_demanda_sku": None, # Se calcula a partir de lo anterior, suma 1

        # Modelo dinámico de suministro y demanda de la lista generada, fuente de cada valor: diseño del modelo de suministro y demanda, sección 6
        "texto_modelo_demanda": "ciclo_vida", # 'ciclo_vida' (curva de Bass y estacionalidad) o 'estatica' (participaciones constantes de Hausman)
        "escalar_bass_p": 0.03, # Coeficiente de innovación de Bass, promedio de Sultan et al. (1990) y Mahajan et al. (1995)
        "escalar_bass_q": 0.38, # Coeficiente de imitación de Bass, mismas fuentes
        "escalar_meses_ciclo_vida": 18, # Duración del ciclo de vida L, el caso más corto de Manzini et al. (2015); un mes son 365/12 días
        "vector_cuantiles_amplitud": np.array([0.05, 0.10, 0.20, 0.47, 0.67]), # Amplitud estacional en puntos logarítmicos: mínimo, p10, mediana, p90 y máximo de Butters et al. (2025)
        "vector_probabilidades_amplitud": np.array([0, 0.10, 0.50, 0.90, 1.0]), # Probabilidad acumulada de cada cuantil anterior, interpolación lineal entre ellos
        "vector_reparto_mes_pico": np.array([0.44, 0.37, 0.19]), # Proporción de posiciones con el pico en diciembre a febrero, en junio a agosto y en el resto (Butters et al., 2025; Ang et al., 2012)
        "escalar_ciclos_por_hora": 60, # Ciclos simples por hora, capacidad demandada del AS/RS de doble profundidad de Lerher et al. (2010)
        "escalar_horas_por_dia": 24, # Operación continua, 24 horas los 7 días (Gagliardi et al., 2012)
        "escalar_dias_ventana_popularidad": 30, # Ventana de la popularidad móvil (Manzini et al., 2015)
        "escalar_dias_calentamiento": 30, # Días de prehistoria antes del año evaluado (Meneghetti et al., 2015; Lehmann y Hußmann, 2023)
        "escalar_techo_ocupacion": 0.95, # Techo de ocupación de la estantería, el mayor factor de llenado de Lerher et al. (2010) y Rizqi y Chou (2025)

        # Agrupación de SKUs en clases, solo la usa la política basada en clases. La clase 1 agrupa a los más demandados
        "texto_criterio_clases": "skus", # 'iguales': N clases con la misma parte de la demanda; 'pareto': 80/15/5 de la demanda; 'skus': porcentaje de SKUs de cada clase a mano
        "escalar_numero_clases": 3, # Número de clases con el criterio 'iguales'
        "vector_proporcion_skus_por_clase": np.array([0.2, 0.2, 0.6]), # Con el criterio 'skus', proporción de SKUs de cada clase

        "vector_prob_acum_AR": np.array([0.5, 1]), # Vector de probabilidad acumulada, para el almacenamiento y recuperación

        # Origen de la lista de tareas: 'generada' (con la demanda y la mezcla de arriba), 'datos' (un CSV real con actividad y SKU de cada tarea, o solo
        # la demanda con SKU y, opcionalmente, el día del modelo) o 'fuente' (una fuente de tesis/datos tratada en cada réplica con su semilla, capítulo 8)
        # Con datos reales la cantidad de SKUs, la demanda, la cantidad de tareas y la mezcla salen del archivo
        "texto_origen_tareas": "generada",
        "texto_tareas_csv": "", # Contenido del CSV, con una tarea por fila: actividad (A o R) y SKU, o solo SKU (y Dia)
        "texto_fuente_demanda": "", # Con origen 'fuente': 'ecommerce', 'calzado' o 'inventario_sintetico'. Cada réplica muestrea el surtido y rota el punto de inicio con su semilla
        "texto_carpeta_datos": "", # Con origen 'fuente': carpeta con las fuentes (cada una con crudo/); vacío es ../tesis/datos
        "texto_variabilidad": "ninguna", # Nivel de variabilidad de la Dirichlet (sección 8.2.4) para una demanda con días: 'ninguna' (la secuencia tratada tal cual), 'baja' (ω̂/2), 'real' (ω̂) o 'alta' ((1 + ω̂)/2)
        "escalar_omega_fuente": None, # ω̂ de la fuente; None lo estima una vez por fuente (ec. 8.8) sobre su secuencia tratada y escalada con la semilla 0, o sobre el CSV
        "escalar_porcentaje_historial": 100 * DIAS_HISTORIAL / (DIAS_HORIZONTE + DIAS_HISTORIAL), # Con datos reales sin días, porcentaje inicial de la lista que sirve de historial (llenado y clases) y no se evalúa: 30 días de 395 (7,59 %), como el calentamiento y la ventana de popularidad; las políticas no ven demanda posterior al momento actual
        "escalar_largo_secuencia": 1,  # Valor inicial, se ajustará en la inicialización
        "escalar_cantidad_replicas": 1, # una corrida larga, como en la instancia del capítulo 6; con más corridas la estadística resume el promedio y el intervalo de confianza. Corrida de la instancia: # la simulación se repetira según la cantidad de replicas
        "escalar_replicas_animadas": 1, # la carrera muestra una réplica; la estadística usa escalar_cantidad_replicas
        "escalar_semilla": 2024, # Semilla base, la réplica i usa semilla + i, esto hace reproducibles los resultados

        # Políticas de asignación de almacenamiento a comparar, también gobiernan la reubicación
        # Disponibles: 'aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua'
        # Las dos con zonas tienen plan fijo: se forma antes del llenado con la popularidad móvil del historial y no cambia (las heurísticas clásicas son estáticas)
        "vector_politicas_asignacion": ('aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua'),

        # Políticas de recuperación: cada política de asignación se ejecuta con cada una
        # 'fcfs': sale el ítem más antiguo del SKU aunque esté bloqueado (FCFS estricto; control de lote)
        # 'fcfs_accesible': sale el más antiguo entre los que no tienen nada delante, y solo si todos están bloqueados el más antiguo de todos
        "vector_politicas_recuperacion": ('fcfs',),
    }
    # Se calcula el largo de secuencia y la demanda de cada SKU
    config["escalar_largo_secuencia"] = TAREAS_POR_CORRIDA
    config["vector_demanda_sku"] = demanda_hausman(config["escalar_numero_skus"], config["escalar_sigma_hausman"])

    return config


def cargar_configuracion_archivo(txt_ruta):
    # Lee un experimento guardado en JSON: basta con poner los parámetros que cambian, el resto toma el valor por defecto
    import json
    with open(txt_ruta, encoding="utf-8") as archivo:
        return config_desde_json(json.load(archivo))


def config_a_json(config):
    # Convierte la configuración en un diccionario con tipos básicos, listo para serializar
    resultado = {}
    for txt_clave, valor in config.items():
        if isinstance(valor, np.ndarray):
            resultado[txt_clave] = valor.tolist()
        elif isinstance(valor, (tuple, list)):
            resultado[txt_clave] = list(valor)
        elif isinstance(valor, np.generic):
            resultado[txt_clave] = valor.item()
        else:
            resultado[txt_clave] = valor
    return resultado


def _numero(valor, txt_nombre, entero=False, minimo=None, maximo=None, minimo_estricto=False):
    # Convierte y valida un número, el mensaje de error nombra el parámetro tal como lo ve el usuario
    if isinstance(valor, bool) or valor is None or valor == "":
        raise ValueError(f"{txt_nombre}: falta el valor.")
    try:
        escalar = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"{txt_nombre}: '{valor}' no es un número.")
    if not np.isfinite(escalar):
        raise ValueError(f"{txt_nombre}: debe ser un número finito.")
    if entero:
        if escalar != int(escalar):
            raise ValueError(f"{txt_nombre}: debe ser un número entero.")
        escalar = int(escalar)
    if minimo is not None and (escalar <= minimo if minimo_estricto else escalar < minimo):
        raise ValueError(f"{txt_nombre}: debe ser {'mayor que' if minimo_estricto else 'al menos'} {minimo}.")
    if maximo is not None and escalar > maximo:
        raise ValueError(f"{txt_nombre}: no puede superar {maximo}.")
    return escalar


def _vector(valor, cantidad, txt_nombre, **restricciones):
    if not isinstance(valor, (list, tuple, np.ndarray)) or len(valor) != cantidad:
        raise ValueError(f"{txt_nombre}: se esperan {cantidad} valores.")
    return [_numero(escalar, f"{txt_nombre} ({indice + 1})", **restricciones) for indice, escalar in enumerate(valor)]


def config_desde_json(diccionario):
    # Construye una configuración válida a partir de un diccionario, completando con los valores por defecto lo que falte
    # Lanza ValueError con un mensaje en español si algún parámetro no es válido
    from asrs.politicas.registro import POLITICAS_ASIGNACION, POLITICAS_RECUPERACION

    if not isinstance(diccionario, dict):
        raise ValueError("La configuración debe ser un objeto.")
    base = config_a_json(cargar_configuracion())
    datos = {**base, **{k: v for k, v in diccionario.items() if k in base}}
    config = {}

    vector_configuracion = _vector(datos["vector_configuracion_estanteria"], 3, "Dimensiones de la estantería", entero=True, minimo=1, maximo=200)
    escalar_celdas = int(np.prod(vector_configuracion))
    if escalar_celdas > LIMITE_CELDAS:
        raise ValueError(f"Dimensiones de la estantería: {escalar_celdas} celdas, el máximo es {LIMITE_CELDAS}.")
    config["vector_configuracion_estanteria"] = np.array(vector_configuracion, dtype=int)
    config["vector_cotas_estanteria"] = np.array(_vector(datos["vector_cotas_estanteria"], 3, "Medidas de la celda", minimo=0, minimo_estricto=True, maximo=100), dtype=float)
    config["vector_centro_celda_estanteria"] = np.array(_vector(datos["vector_centro_celda_estanteria"], 3, "Centro de la celda", minimo=0, maximo=1), dtype=float)
    config["vector_posicion_estanteria"] = np.array(_vector(datos["vector_posicion_estanteria"], 3, "Posición de la estantería", minimo=-1000, maximo=1000), dtype=float)

    config["escalar_nivel_ocupacion"] = _numero(datos["escalar_nivel_ocupacion"], "Nivel de ocupación", minimo=0, maximo=1)
    if datos["texto_perfil_inventario"] not in PERFILES_INVENTARIO:
        raise ValueError(f"Perfil de inventario: debe ser {' o '.join(repr(p) for p in PERFILES_INVENTARIO)}.")
    config["texto_perfil_inventario"] = datos["texto_perfil_inventario"]
    config["escalar_separacion_riel_grua"] = _numero(datos["escalar_separacion_riel_grua"], "Separación del riel", minimo=0, maximo=100)
    config["vector_velocidad_grua"] = np.array(_vector(datos["vector_velocidad_grua"], 3, "Velocidad de la grúa", minimo=0, minimo_estricto=True, maximo=1000), dtype=float)
    config["vector_aceleracion_grua"] = np.array(_vector(datos["vector_aceleracion_grua"], 3, "Aceleración de la grúa", minimo=0, maximo=1000), dtype=float)
    config["vector_deceleracion_grua"] = np.array(_vector(datos["vector_deceleracion_grua"], 3, "Deceleración de la grúa", minimo=0, maximo=1000), dtype=float)
    # Las configuraciones guardadas antes tenían un solo tiempo de carga y descarga para todo
    if "escalar_tiempo_carga_descarga" in diccionario and not any(k.startswith("escalar_tiempo_toma_deposito") for k in diccionario):
        for txt_clave in ("escalar_tiempo_toma_deposito_es", "escalar_tiempo_toma_deposito_capa_delantera", "escalar_tiempo_toma_deposito_capa_trasera"):
            datos[txt_clave] = diccionario["escalar_tiempo_carga_descarga"]
    config["escalar_tiempo_toma_deposito_es"] = _numero(datos["escalar_tiempo_toma_deposito_es"], "Tiempo de toma y depósito en el E/S", minimo=0, maximo=3600)
    config["escalar_tiempo_toma_deposito_capa_delantera"] = _numero(datos["escalar_tiempo_toma_deposito_capa_delantera"], "Tiempo de toma y depósito en la capa delantera", minimo=0, maximo=3600)
    config["escalar_tiempo_toma_deposito_capa_trasera"] = _numero(datos["escalar_tiempo_toma_deposito_capa_trasera"], "Tiempo de toma y depósito en las capas traseras", minimo=0, maximo=3600)

    # Modelo dinámico de suministro y demanda: solo lo usa la lista generada, con datos reales se valida igual para conservar la configuración completa
    if datos["texto_modelo_demanda"] not in ("ciclo_vida", "estatica"):
        raise ValueError("Modelo de demanda: debe ser 'ciclo_vida' o 'estatica'.")
    config["texto_modelo_demanda"] = datos["texto_modelo_demanda"]
    config["escalar_bass_p"] = _numero(datos["escalar_bass_p"], "Coeficiente de innovación de Bass", minimo=0, minimo_estricto=True, maximo=1)
    config["escalar_bass_q"] = _numero(datos["escalar_bass_q"], "Coeficiente de imitación de Bass", minimo=0, minimo_estricto=True, maximo=5)
    if config["escalar_bass_q"] <= config["escalar_bass_p"]:
        raise ValueError("Coeficientes de Bass: el de imitación debe ser mayor que el de innovación.")
    config["escalar_meses_ciclo_vida"] = _numero(datos["escalar_meses_ciclo_vida"], "Duración del ciclo de vida (meses)", minimo=0, minimo_estricto=True, maximo=600)
    vector_cuantiles = _vector(datos["vector_cuantiles_amplitud"], 5, "Cuantiles de la amplitud estacional", minimo=0, maximo=3)
    if any(vector_cuantiles[i] > vector_cuantiles[i + 1] for i in range(4)):
        raise ValueError("Cuantiles de la amplitud estacional: no pueden decrecer.")
    config["vector_cuantiles_amplitud"] = np.array(vector_cuantiles, dtype=float)
    vector_probabilidades = _vector(datos["vector_probabilidades_amplitud"], 5, "Probabilidades de los cuantiles", minimo=0, maximo=1)
    if any(vector_probabilidades[i] >= vector_probabilidades[i + 1] for i in range(4)):
        raise ValueError("Probabilidades de los cuantiles: deben ser estrictamente crecientes.")
    config["vector_probabilidades_amplitud"] = np.array(vector_probabilidades, dtype=float)
    vector_reparto = _vector(datos["vector_reparto_mes_pico"], 3, "Reparto del mes de pico", minimo=0)
    if sum(vector_reparto) <= 0:
        raise ValueError("Reparto del mes de pico: la suma debe ser mayor que cero.")
    config["vector_reparto_mes_pico"] = np.array(vector_reparto, dtype=float) / sum(vector_reparto)
    config["escalar_ciclos_por_hora"] = _numero(datos["escalar_ciclos_por_hora"], "Ciclos por hora", minimo=0, minimo_estricto=True, maximo=1000)
    config["escalar_horas_por_dia"] = _numero(datos["escalar_horas_por_dia"], "Horas por día", minimo=0, minimo_estricto=True, maximo=24)
    if round(config["escalar_ciclos_por_hora"] * config["escalar_horas_por_dia"] / 2) < 1:
        raise ValueError("Ciclos por hora y horas por día: dan menos de una recuperación diaria.")
    config["escalar_dias_ventana_popularidad"] = _numero(datos["escalar_dias_ventana_popularidad"], "Ventana de la popularidad (días)", entero=True, minimo=1, maximo=365)
    config["escalar_dias_calentamiento"] = _numero(datos["escalar_dias_calentamiento"], "Calentamiento (días)", entero=True,
                                                   minimo=config["escalar_dias_ventana_popularidad"], maximo=730)
    escalar_techo = datos["escalar_techo_ocupacion"]
    if "escalar_techo_ocupacion" not in diccionario and config["escalar_nivel_ocupacion"] >= base["escalar_techo_ocupacion"]:
        escalar_techo = 1.0 # La herramienta web no tiene campo para el techo: con ocupaciones altas se sube a 1 en lugar de rechazar una configuración antes válida
    config["escalar_techo_ocupacion"] = _numero(escalar_techo, "Techo de ocupación", minimo=0, minimo_estricto=True, maximo=1)
    # Con datos reales no hay generador de escenario y el techo no interviene
    if datos["texto_origen_tareas"] == "generada" and config["escalar_techo_ocupacion"] <= config["escalar_nivel_ocupacion"]:
        raise ValueError(f"Techo de ocupación ({config['escalar_techo_ocupacion']:g}): debe ser mayor que el nivel de ocupación ({config['escalar_nivel_ocupacion']:g}); "
                         "sube el techo o reduce la ocupación.")

    origen = datos["texto_origen_tareas"]
    if origen not in ("generada", "datos", "fuente"):
        raise ValueError("Origen de las tareas: debe ser 'generada', 'datos' o 'fuente'.")
    config["texto_origen_tareas"] = origen
    config["texto_tareas_csv"] = datos["texto_tareas_csv"] if isinstance(datos["texto_tareas_csv"], str) else ""
    config["texto_fuente_demanda"] = datos["texto_fuente_demanda"] if isinstance(datos["texto_fuente_demanda"], str) else ""
    config["texto_carpeta_datos"] = datos["texto_carpeta_datos"] if isinstance(datos["texto_carpeta_datos"], str) else ""
    from asrs.generacion.variabilidad import NIVELES_VARIABILIDAD
    if datos["texto_variabilidad"] not in NIVELES_VARIABILIDAD:
        raise ValueError(f"Variabilidad: debe ser {', '.join(repr(n) for n in NIVELES_VARIABILIDAD)}.")
    config["texto_variabilidad"] = datos["texto_variabilidad"]
    config["escalar_omega_fuente"] = None if datos["escalar_omega_fuente"] in (None, "") else _numero(datos["escalar_omega_fuente"], "ω de la fuente", minimo=0, maximo=1)
    if origen == "fuente" and not config["texto_fuente_demanda"]:
        raise ValueError("Fuente de demanda: con origen 'fuente' indica 'texto_fuente_demanda'.")
    config["escalar_porcentaje_historial"] = _numero(datos["escalar_porcentaje_historial"], "Historial para planificar", minimo=0, minimo_estricto=True, maximo=100)

    # Demanda por SKU: curva de Hausman o una lista personalizada, siempre se normaliza para que sume 1
    personalizada = datos["vector_demanda_sku_personalizada"]
    if origen in ("datos", "fuente"):
        # Con datos reales todo sale de la lista: las tareas, los SKUs y la demanda que conocen las políticas
        # La lista de la primera réplica (semilla base) queda en la configuración; las demás réplicas rehacen la suya si su demanda varía
        config["escalar_semilla"] = _numero(datos["escalar_semilla"], "Semilla", entero=True, minimo=0, maximo=2 ** 31 - 1)
        config.update(lista_real_de_replica(config, config["escalar_semilla"]))
        config["escalar_sigma_hausman"] = _numero(datos["escalar_sigma_hausman"], "Sesgo de la demanda", minimo=0, minimo_estricto=True, maximo=1)
        config["vector_demanda_sku_personalizada"] = None
    elif personalizada is not None and personalizada != [] and personalizada != "":
        if not isinstance(personalizada, (list, tuple)) or not 1 <= len(personalizada) <= LIMITE_SKUS:
            raise ValueError(f"Demanda de los SKUs: indica entre 1 y {LIMITE_SKUS} valores.")
        vector_demanda = np.array(_vector(personalizada, len(personalizada), "Demanda del SKU", minimo=0), dtype=float)
        if vector_demanda.sum() <= 0:
            raise ValueError("Demanda de los SKUs: la suma debe ser mayor que cero.")
        config["escalar_numero_skus"] = len(vector_demanda)
        config["escalar_sigma_hausman"] = _numero(datos["escalar_sigma_hausman"], "Sesgo de la demanda", minimo=0, minimo_estricto=True, maximo=1)
        config["vector_demanda_sku_personalizada"] = vector_demanda.tolist()
        config["vector_demanda_sku"] = vector_demanda / vector_demanda.sum()
    else:
        config["escalar_numero_skus"] = _numero(datos["escalar_numero_skus"], "Cantidad de SKUs", entero=True, minimo=1, maximo=LIMITE_SKUS)
        config["escalar_sigma_hausman"] = _numero(datos["escalar_sigma_hausman"], "Sesgo de la demanda", minimo=0, minimo_estricto=True, maximo=1)
        config["vector_demanda_sku_personalizada"] = None
        config["vector_demanda_sku"] = demanda_hausman(config["escalar_numero_skus"], config["escalar_sigma_hausman"])
    escalar_canales = int(vector_configuracion[0] * vector_configuracion[1])
    if config["escalar_numero_skus"] > escalar_canales:
        raise ValueError(f"Cantidad de SKUs: hay {config['escalar_numero_skus']} SKUs y solo {escalar_canales} canales, la política full-turnover reserva al menos un canal por SKU.")

    criterio = datos["texto_criterio_clases"]
    if criterio not in ("iguales", "pareto", "skus"):
        raise ValueError("Criterio de las clases: debe ser 'iguales', 'pareto' o 'skus'.")
    config["texto_criterio_clases"] = criterio
    config["escalar_numero_clases"] = _numero(datos["escalar_numero_clases"], "Número de clases", entero=True, minimo=1, maximo=LIMITE_CLASES)
    proporciones = datos["vector_proporcion_skus_por_clase"]
    if not isinstance(proporciones, (list, tuple, np.ndarray)) or not 1 <= len(proporciones) <= LIMITE_CLASES:
        raise ValueError(f"Clases de la política: debe haber entre 1 y {LIMITE_CLASES}.")
    proporciones = _vector(list(proporciones), len(proporciones), "Proporción de SKUs de la clase", minimo=0, minimo_estricto=True)
    config["vector_proporcion_skus_por_clase"] = np.array(proporciones, dtype=float) / sum(proporciones)
    # Cada clase necesita al menos un SKU, según el criterio elegido
    escalar_clases_usadas = {"iguales": config["escalar_numero_clases"], "pareto": 3, "skus": len(proporciones)}[criterio]
    if escalar_clases_usadas > config["escalar_numero_skus"]:
        raise ValueError(f"Clases de la política: hay {escalar_clases_usadas} clases y solo {config['escalar_numero_skus']} SKUs.")

    if origen in ("datos", "fuente"):
        pass # La mezcla de operaciones es la de la lista (ver `lista_real_de_replica`)
    else:
        probabilidad = _vector(datos["vector_prob_acum_AR"], 2, "Mezcla de operaciones", minimo=0, maximo=1)
        if probabilidad[1] != 1:
            raise ValueError("Mezcla de operaciones: la probabilidad acumulada debe terminar en 1.")
        config["vector_prob_acum_AR"] = np.array(probabilidad, dtype=float)

    # Si no se indica la cantidad de tareas se usa el 10 % de la capacidad de la estantería
    largo = diccionario.get("escalar_largo_secuencia")
    if largo is None or largo == "":
        largo = max(1, round(escalar_celdas * 0.1))
    config["escalar_largo_secuencia"] = len(config["secuencia_real"]) if origen in ("datos", "fuente") else _numero(largo, "Cantidad de tareas", entero=True, minimo=1, maximo=LIMITE_TAREAS_GENERADAS)

    config["escalar_semilla"] = _numero(datos["escalar_semilla"], "Semilla", entero=True, minimo=0, maximo=2 ** 31 - 1)
    config["escalar_replicas_animadas"] = 1      # ya no es un parámetro: la carrera anima siempre la primera réplica
    config["escalar_cantidad_replicas"] = _numero(datos["escalar_cantidad_replicas"], "Réplicas para la estadística", entero=True, minimo=0, maximo=LIMITE_REPLICAS_ESTADISTICA)

    politicas = datos["vector_politicas_asignacion"]
    if not isinstance(politicas, (list, tuple)) or not politicas:
        raise ValueError("Políticas: elige al menos una.")
    for txt_politica in politicas:
        if txt_politica not in POLITICAS_ASIGNACION:
            raise ValueError(f"Políticas: '{txt_politica}' no existe. Disponibles: {', '.join(POLITICAS_ASIGNACION)}.")
    if len(set(politicas)) != len(politicas):
        raise ValueError("Políticas: hay políticas repetidas.")
    config["vector_politicas_asignacion"] = tuple(politicas)

    # Las configuraciones guardadas antes admitían una sola regla de recuperación en 'texto_regla_recuperacion'
    if "vector_politicas_recuperacion" not in diccionario and isinstance(diccionario.get("texto_regla_recuperacion"), str):
        datos["vector_politicas_recuperacion"] = [diccionario["texto_regla_recuperacion"]]
    recuperaciones = datos["vector_politicas_recuperacion"]
    if not isinstance(recuperaciones, (list, tuple)) or not recuperaciones:
        raise ValueError("Recuperación: elige al menos una.")
    for txt_recuperacion in recuperaciones:
        if txt_recuperacion not in POLITICAS_RECUPERACION:
            raise ValueError(f"Recuperación: '{txt_recuperacion}' no existe. Disponibles: {', '.join(POLITICAS_RECUPERACION)}.")
    if len(set(recuperaciones)) != len(recuperaciones):
        raise ValueError("Recuperación: hay políticas repetidas.")
    config["vector_politicas_recuperacion"] = tuple(recuperaciones)

    return config
