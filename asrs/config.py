"""Parámetros del modelo de simulación."""
import numpy as np

from asrs.generacion.probabilidades import demanda_hausman

# Límites que protegen contra configuraciones que tardarían demasiado o no tienen sentido
LIMITE_CELDAS = 6000
LIMITE_TAREAS = 5000
LIMITE_SKUS = 200
LIMITE_CLASES = 8
LIMITE_REPLICAS_ANIMADAS = 10
LIMITE_REPLICAS_ESTADISTICA = 2000


def cargar_configuracion():
    # Se cargan los párametros del modelo
    config = {

        # Nota en los párametros espaciales las componentes verticales y horizontales son intercambiables, solo se debe tener congruencia en todos los demás párametros

        # Se recomienda utilizar la convención de ejemplo, esto por que la matrices tradicionalmente se recorrecoren por filas y luego columnas
        # Sin embargo como se meciona, siempre y cuando se conserve congruencia las componente  verticales y horizontales son intercambiales sin implicaciones negativas

        "vector_posicion_estanteria": np.array([0, 0, 0]), # (posición vertical, posición horizontal, posición de profundidad)
        "vector_configuracion_estanteria": np.array([18, 20, 2]), # (filas, columnas, capas de profundidad)
        "vector_cotas_estanteria": np.array([1.6, 0.9, 1.1]), # (alto de fila, ancho de columna, profundid de celda)
        "vector_centro_celda_estanteria": np.array([0, 0, 0]), # (% verical, % horizontal, % profundidad), representa un traslación del punto de referencia dentro de la celda
        "escalar_nivel_ocupacion": 0.85, # Porcentaje de la estatería que esta ocupada, intervalo [0, 1]
        "escalar_separacion_riel_grua": 0.5, # Unidades de separación entre la cara frontal de la estantería y el riel de la grúa
        "vector_velocidad_grua": np.array([1, 1.5, 0.7]), #(Velocidad en filas, velocidad en columnas, velocidad a lo largo de canales)
        "escalar_tiempo_carga_descarga": 0, # Tiempo que demora la grúa en cargar o descargar algún ítem

        # Demanda por SKU: la lista de tareas solo conoce SKUs, el SKU 1 es el más demandado
        "escalar_numero_skus": 30, # Cantidad de SKUs
        "escalar_sigma_hausman": 0.5, # Sesgo de la demanda según la curva de Hausman, 1 es demanda uniforme y menor es más concentrada
        "vector_demanda_sku_personalizada": None, # Opcional, demanda de cada SKU, reemplaza a los dos parámetros anteriores
        "vector_demanda_sku": None, # Se calcula a partir de lo anterior, suma 1

        # Agrupación de SKUs en clases, solo la usa la política basada en clases
        "vector_proporcion_skus_por_clase": np.array([0.2, 0.3, 0.5]), # Proporción de SKUs de cada clase, la clase 1 agrupa a los más demandados

        "vector_prob_acum_AR": np.array([0.5, 1]), # Vector de probabilidad acumulada, para el almacenamiento y recuperación
        "escalar_largo_secuencia": 1,  # Valor inicial, se ajustará en la inicialización
        "escalar_cantidad_replicas": 100, # la simulación se repetira según la cantidad de replicas
        "escalar_replicas_animadas": 3, # réplicas que se graban para la animación del visualizador
        "escalar_semilla": 2024, # Semilla base, la réplica i usa semilla + i, esto hace reproducibles los resultados

        # Políticas de asignación de almacenamiento a comparar, también gobiernan la reubicación
        # Disponibles: 'aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua'
        "vector_politicas_asignacion": ('aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua'),
    }
    # Se calcula el largo de secuencia y la demanda de cada SKU
    config["escalar_largo_secuencia"] = round(np.prod(config["vector_configuracion_estanteria"]) * 0.1)
    config["vector_demanda_sku"] = demanda_hausman(config["escalar_numero_skus"], config["escalar_sigma_hausman"])

    return config


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
    from asrs.politicas.registro import POLITICAS_ASIGNACION

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
    config["escalar_separacion_riel_grua"] = _numero(datos["escalar_separacion_riel_grua"], "Separación del riel", minimo=0, maximo=100)
    config["vector_velocidad_grua"] = np.array(_vector(datos["vector_velocidad_grua"], 3, "Velocidad de la grúa", minimo=0, minimo_estricto=True, maximo=1000), dtype=float)
    config["escalar_tiempo_carga_descarga"] = _numero(datos["escalar_tiempo_carga_descarga"], "Tiempo de carga y descarga", minimo=0, maximo=3600)

    # Demanda por SKU: curva de Hausman o una lista personalizada, siempre se normaliza para que sume 1
    personalizada = datos["vector_demanda_sku_personalizada"]
    if personalizada is not None and personalizada != [] and personalizada != "":
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

    proporciones = datos["vector_proporcion_skus_por_clase"]
    if not isinstance(proporciones, (list, tuple, np.ndarray)) or not 1 <= len(proporciones) <= LIMITE_CLASES:
        raise ValueError(f"Clases de la política: debe haber entre 1 y {LIMITE_CLASES}.")
    proporciones = _vector(list(proporciones), len(proporciones), "Proporción de SKUs de la clase", minimo=0, minimo_estricto=True)
    if len(proporciones) > config["escalar_numero_skus"]:
        raise ValueError(f"Clases de la política: hay {len(proporciones)} clases y solo {config['escalar_numero_skus']} SKUs.")
    config["vector_proporcion_skus_por_clase"] = np.array(proporciones, dtype=float) / sum(proporciones)

    probabilidad = _vector(datos["vector_prob_acum_AR"], 2, "Mezcla de operaciones", minimo=0, maximo=1)
    if probabilidad[1] != 1:
        raise ValueError("Mezcla de operaciones: la probabilidad acumulada debe terminar en 1.")
    config["vector_prob_acum_AR"] = np.array(probabilidad, dtype=float)

    # Si no se indica la cantidad de tareas se usa el 10 % de la capacidad de la estantería
    largo = diccionario.get("escalar_largo_secuencia")
    if largo is None or largo == "":
        largo = max(1, round(escalar_celdas * 0.1))
    config["escalar_largo_secuencia"] = _numero(largo, "Cantidad de tareas", entero=True, minimo=1, maximo=LIMITE_TAREAS)

    config["escalar_semilla"] = _numero(datos["escalar_semilla"], "Semilla", entero=True, minimo=0, maximo=2 ** 31 - 1)
    config["escalar_replicas_animadas"] = _numero(datos["escalar_replicas_animadas"], "Réplicas animadas", entero=True, minimo=1, maximo=LIMITE_REPLICAS_ANIMADAS)
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

    return config
