"""Parámetros del modelo de simulación."""
import numpy as np

# Límites que protegen contra configuraciones que tardarían demasiado o no tienen sentido
LIMITE_CELDAS = 6000
LIMITE_TAREAS = 5000
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
        "vector_capacidad_clase": np.array([0.2, 0.4, 0.4]), # Proporción de celdas que ocupa cada clase en las políticas con zonas fijas
        "vector_demanda_clase": np.array([0.2, 0.4, 0.4]), # Proporción de la demanda (llegadas y solicitudes) de cada clase, ordena las clases en full-turnover
        "vector_prob_acum_AR": np.array([0.5, 1]), # Vector de probabilidad acumulada, para el almacenamiento y recuperación
        "escalar_largo_secuencia": 1,  # Valor inicial, se ajustará en la inicialización
        "escalar_cantidad_replicas": 100, # la simulación se repetira según la cantidad de replicas
        "escalar_replicas_animadas": 3, # réplicas que se graban para la animación del visualizador
        "escalar_semilla": 2024, # Semilla base, la réplica i usa semilla + i, esto hace reproducibles los resultados

        # Políticas de asignación de almacenamiento a comparar, también gobiernan la reubicación
        # Disponibles: 'aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua'
        "vector_politicas_asignacion": ('aleatoria', 'basada_en_clases', 'full_turnover', 'mas_cercana_es', 'mas_cercana_grua'),
    }
    # Se calcula el largo de secuencia
    config["escalar_largo_secuencia"] = round(np.prod(config["vector_configuracion_estanteria"]) * 0.1)

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
    if int(np.prod(vector_configuracion)) > LIMITE_CELDAS:
        raise ValueError(f"Dimensiones de la estantería: {int(np.prod(vector_configuracion))} celdas, el máximo es {LIMITE_CELDAS}.")
    config["vector_configuracion_estanteria"] = np.array(vector_configuracion, dtype=int)
    config["vector_cotas_estanteria"] = np.array(_vector(datos["vector_cotas_estanteria"], 3, "Medidas de la celda", minimo=0, minimo_estricto=True, maximo=100), dtype=float)
    config["vector_centro_celda_estanteria"] = np.array(_vector(datos["vector_centro_celda_estanteria"], 3, "Centro de la celda", minimo=0, maximo=1), dtype=float)
    config["vector_posicion_estanteria"] = np.array(_vector(datos["vector_posicion_estanteria"], 3, "Posición de la estantería", minimo=-1000, maximo=1000), dtype=float)

    config["escalar_nivel_ocupacion"] = _numero(datos["escalar_nivel_ocupacion"], "Nivel de ocupación", minimo=0, maximo=1)
    config["escalar_separacion_riel_grua"] = _numero(datos["escalar_separacion_riel_grua"], "Separación del riel", minimo=0, maximo=100)
    config["vector_velocidad_grua"] = np.array(_vector(datos["vector_velocidad_grua"], 3, "Velocidad de la grúa", minimo=0, minimo_estricto=True, maximo=1000), dtype=float)
    config["escalar_tiempo_carga_descarga"] = _numero(datos["escalar_tiempo_carga_descarga"], "Tiempo de carga y descarga", minimo=0, maximo=3600)

    # Clases: capacidad y demanda se normalizan para que sumen 1
    capacidad = datos["vector_capacidad_clase"]
    demanda = datos["vector_demanda_clase"]
    if not isinstance(capacidad, (list, tuple)) or not 1 <= len(capacidad) <= LIMITE_CLASES:
        raise ValueError(f"Clases: debe haber entre 1 y {LIMITE_CLASES}.")
    if not isinstance(demanda, (list, tuple)) or len(demanda) != len(capacidad):
        raise ValueError("Clases: la capacidad y la demanda deben tener la misma cantidad de clases.")
    capacidad = _vector(capacidad, len(capacidad), "Capacidad de la clase", minimo=0)
    demanda = _vector(demanda, len(demanda), "Demanda de la clase", minimo=0)
    if sum(capacidad) <= 0:
        raise ValueError("Capacidad de las clases: la suma debe ser mayor que cero.")
    if sum(demanda) <= 0:
        raise ValueError("Demanda de las clases: la suma debe ser mayor que cero.")
    config["vector_capacidad_clase"] = np.array(capacidad, dtype=float) / sum(capacidad)
    config["vector_demanda_clase"] = np.array(demanda, dtype=float) / sum(demanda)

    probabilidad = _vector(datos["vector_prob_acum_AR"], 2, "Mezcla de operaciones", minimo=0, maximo=1)
    if probabilidad[1] != 1:
        raise ValueError("Mezcla de operaciones: la probabilidad acumulada debe terminar en 1.")
    config["vector_prob_acum_AR"] = np.array(probabilidad, dtype=float)

    escalar_celdas = int(np.prod(vector_configuracion))
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
