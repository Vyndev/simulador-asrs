"""Lista de tareas a partir de datos reales: un archivo CSV con dos columnas, el SKU y la tarea (almacenar o recuperar), en orden.
También acepta un CSV solo de demanda (todas las tareas son recuperaciones), que es la salida de `asrs/generacion/fuentes`: una columna con
encabezado `SKU`, o dos columnas `SKU,Dia` con el día del modelo de cada recuperación (capítulo 8, sección 8.2)."""
import csv
import math
from collections import Counter

import numpy as np

PALABRAS_ALMACENAR = {"a", "0", "almacenar", "almacenamiento", "entrada", "ingreso", "in", "store", "put", "putaway", "e"}
PALABRAS_RECUPERAR = {"r", "1", "recuperar", "recuperacion", "recuperación", "salida", "despacho", "out", "retrieve", "pick", "get", "s"}
NOMBRES_COLUMNA_ACTIVIDAD = ("task", "tarea", "act", "tipo", "oper", "movim", "type")
NOMBRES_COLUMNA_SKU = ("sku", "art", "prod", "ref", "item", "cod", "cód")
ENCABEZADOS_UNA_COLUMNA = {"sku", "articulo", "artículo", "producto", "referencia", "item", "codigo", "código"}   # nombre exacto de la columna única
ENCABEZADOS_DIA = {"dia", "día", "day"}   # nombre exacto de la segunda columna de un CSV de demanda con días


def _es_una_columna(vector_lineas):
    # Un CSV de una sola columna con encabezado SKU: solo trae la demanda (recuperaciones)
    if not vector_lineas:
        return False
    txt_separador = max(",;\t|", key=lambda s: vector_lineas[0].count(s))
    return vector_lineas[0].count(txt_separador) == 0 and vector_lineas[0].strip().strip('"').lower() in ENCABEZADOS_UNA_COLUMNA


def _es_sku_dia(vector_lineas):
    # Un CSV de dos columnas con encabezado SKU y Dia: la demanda con el día del modelo de cada recuperación
    if not vector_lineas:
        return False
    vector_encabezado = [txt.strip().strip('"').lower() for txt in next(csv.reader([vector_lineas[0]]))]
    return len(vector_encabezado) == 2 and vector_encabezado[0] in ENCABEZADOS_UNA_COLUMNA and vector_encabezado[1] in ENCABEZADOS_DIA


def _leer_sku_dia(vector_lineas, escalar_limite_tareas):
    # Devuelve las recuperaciones (1, SKU) y el día de cada una. Los días deben ser enteros no negativos y no decrecer
    matriz_tareas, vector_dias = [], []
    for escalar_fila, vector_fila in enumerate(csv.reader(vector_lineas[1:]), start=2):
        if len(vector_fila) != 2:
            raise ValueError(f"Datos reales, fila {escalar_fila}: se esperan dos columnas, SKU y Dia.")
        txt_sku, txt_dia = vector_fila[0].strip(), vector_fila[1].strip()
        if not txt_sku:
            raise ValueError(f"Datos reales, fila {escalar_fila}: falta el SKU.")
        try:
            escalar_dia = int(txt_dia)
        except ValueError:
            raise ValueError(f"Datos reales, fila {escalar_fila}: el día '{txt_dia}' no es un entero.")
        if escalar_dia < 0 or (vector_dias and escalar_dia < vector_dias[-1]):
            raise ValueError(f"Datos reales, fila {escalar_fila}: los días deben ser enteros no negativos y estar en orden.")
        matriz_tareas.append((1, txt_sku))
        vector_dias.append(escalar_dia)
        if escalar_limite_tareas is not None and len(matriz_tareas) > escalar_limite_tareas:
            raise ValueError(f"Datos reales: hay más de {escalar_limite_tareas} tareas, el máximo es {escalar_limite_tareas}.")
    if not matriz_tareas:
        raise ValueError("Datos reales: el archivo no tiene tareas.")
    return matriz_tareas, vector_dias


def leer_dias(txt_csv, escalar_limite_tareas=None):
    """Día del modelo de cada recuperación de un CSV `SKU,Dia`; None si el archivo no trae días."""
    vector_lineas = [txt_linea for txt_linea in txt_csv.splitlines() if txt_linea.strip()]
    return _leer_sku_dia(vector_lineas, escalar_limite_tareas)[1] if _es_sku_dia(vector_lineas) else None


def _es_actividad(txt_valor):
    return txt_valor.strip().lower() in PALABRAS_ALMACENAR | PALABRAS_RECUPERAR


def _leer_una_columna(vector_lineas, escalar_limite_tareas):
    # Una columna con encabezado SKU: cada fila es la recuperación de ese SKU. Se lee con un delimitador que no aparece
    # para respetar las comillas y que un SKU con comas, que el escritor de fuentes cita, no se parta en columnas
    matriz_tareas = []
    for escalar_fila, vector_fila in enumerate(csv.reader(vector_lineas[1:], delimiter="\x1f"), start=2):
        txt_sku = vector_fila[0].strip()
        if not txt_sku:
            raise ValueError(f"Datos reales, fila {escalar_fila}: falta el SKU.")
        matriz_tareas.append((1, txt_sku))
        if escalar_limite_tareas is not None and len(matriz_tareas) > escalar_limite_tareas:
            raise ValueError(f"Datos reales: hay más de {escalar_limite_tareas} tareas, el máximo es {escalar_limite_tareas}.")
    if not matriz_tareas:
        raise ValueError("Datos reales: el archivo no tiene tareas.")
    return matriz_tareas


def leer_tareas(txt_csv, escalar_limite_tareas=None):
    # Devuelve la lista de tareas como pares (actividad, etiqueta del SKU): actividad 0 es almacenar y 1 es recuperar
    # Las columnas son SKU y Task, en ese orden. El separador (coma, punto y coma, tabulación o barra vertical) y el encabezado se detectan solos
    # Con encabezado las columnas se buscan por su nombre; sin encabezado se toma SKU y Task, salvo que solo la primera columna sea una actividad
    vector_lineas = [txt_linea for txt_linea in txt_csv.splitlines() if txt_linea.strip()]
    if not vector_lineas:
        raise ValueError("Datos reales: el archivo está vacío.")
    txt_separador = max(",;\t|", key=lambda s: vector_lineas[0].count(s))
    if _es_una_columna(vector_lineas):
        return _leer_una_columna(vector_lineas, escalar_limite_tareas)
    if _es_sku_dia(vector_lineas):
        return _leer_sku_dia(vector_lineas, escalar_limite_tareas)[0]
    if vector_lineas[0].count(txt_separador) == 0:
        raise ValueError("Datos reales: se esperan dos columnas, SKU y Task, separadas por coma, punto y coma o tabulación.")
    matriz_filas = list(csv.reader(vector_lineas, delimiter=txt_separador))

    # Si ninguna celda de la primera fila es una actividad reconocible se toma como encabezado y se buscan las columnas por su nombre
    escalar_columna_sku, escalar_columna_actividad, escalar_primera = 0, 1, 0
    vector_primera = matriz_filas[0]
    if len(vector_primera) > 1 and _es_actividad(vector_primera[0]) and not _es_actividad(vector_primera[1]):
        escalar_columna_sku, escalar_columna_actividad = 1, 0      # sin encabezado y con la actividad primero
    elif not any(_es_actividad(txt) for txt in vector_primera):
        vector_nombres = [txt_nombre.strip().lower() for txt_nombre in matriz_filas[0]]
        escalar_primera = 1
        encontrada = next((i for i, n in enumerate(vector_nombres) if any(p in n for p in NOMBRES_COLUMNA_ACTIVIDAD)), None)
        escalar_sku = next((i for i, n in enumerate(vector_nombres) if any(p in n for p in NOMBRES_COLUMNA_SKU)), None)
        if encontrada is not None and escalar_sku is not None and encontrada != escalar_sku:
            escalar_columna_actividad, escalar_columna_sku = encontrada, escalar_sku
        elif len(vector_nombres) != 2:
            raise ValueError("Datos reales: no encuentro las columnas. El encabezado debe nombrarlas, por ejemplo 'SKU' y 'Task'.")

    matriz_tareas = []
    for escalar_fila, vector_fila in enumerate(matriz_filas[escalar_primera:], start=escalar_primera + 1):
        if len(vector_fila) <= max(escalar_columna_actividad, escalar_columna_sku):
            raise ValueError(f"Datos reales, fila {escalar_fila}: faltan columnas.")
        txt_actividad, txt_sku = vector_fila[escalar_columna_actividad].strip(), vector_fila[escalar_columna_sku].strip()
        if txt_actividad.lower() in PALABRAS_ALMACENAR:
            escalar_actividad = 0
        elif txt_actividad.lower() in PALABRAS_RECUPERAR:
            escalar_actividad = 1
        else:
            raise ValueError(f"Datos reales, fila {escalar_fila}: no reconozco la actividad '{txt_actividad}'. "
                             "Usa A o R, almacenar o recuperar, entrada o salida, o 0 (almacenar) y 1 (recuperar).")
        if not txt_sku:
            raise ValueError(f"Datos reales, fila {escalar_fila}: falta el SKU.")
        matriz_tareas.append((escalar_actividad, txt_sku))
        if escalar_limite_tareas is not None and len(matriz_tareas) > escalar_limite_tareas:
            raise ValueError(f"Datos reales: hay más de {escalar_limite_tareas} tareas, el máximo es {escalar_limite_tareas}.")
    if not matriz_tareas:
        raise ValueError("Datos reales: el archivo no tiene tareas.")
    return matriz_tareas


def es_solo_demanda(txt_csv):
    """True si el CSV es de una sola columna SKU o de dos columnas SKU y Dia: trae solo la demanda (recuperaciones) y los almacenamientos hay que generarlos."""
    vector_lineas = [txt for txt in txt_csv.splitlines() if txt.strip()]
    return _es_una_columna(vector_lineas) or _es_sku_dia(vector_lineas)


def reponer_con_lotes_eoq(matriz_secuencia, vector_demanda, escalar_recuperaciones_diarias, escalar_inventario_esperado, escalar_limite_tareas=None,
                          vector_dias=None):
    """Genera los almacenamientos de una lista que solo trae demanda, con la reposición por lotes EOQ del modelo de suministro (capítulo 8).
    `matriz_secuencia` son las recuperaciones (1, SKU) con el SKU numerado de 1 a n y `vector_demanda` la participación de cada SKU en el historial.
    Demanda esperada ν = R ρ (R recuperaciones por día); lote Q = max(1, round(κ √ν)) con κ calibrada para que el inventario esperado
    Σ (Q + 1) / 2 sea `escalar_inventario_esperado` (φ F C K); punto de pedido y plazo de entrega cero: cuando una recuperación deja al SKU sin
    unidades, el lote entra justo después, unidad tras unidad. Cada SKU con demanda en el historial parte con la mitad de su lote, redondeada hacia arriba, que es el
    llenado inicial; los demás nacen con su primer lote. Devuelve la lista de operación y las unidades iniciales de cada SKU (posición i = SKU i + 1).
    Si se da `vector_dias` (el día de cada recuperación) devuelve además el día de cada tarea de la operación: un almacenamiento lleva el día de la
    recuperación que lo provoca.
    Supuestos de los datos reales: ν es constante (la participación del historial), sin ciclo de vida, cierres ni techo de ocupación."""
    from asrs.generacion.reposicion import calibrar_kappa, lotes_eoq
    vector_nu = escalar_recuperaciones_diarias * np.asarray(vector_demanda, dtype=float)
    vector_lote = lotes_eoq(vector_nu, calibrar_kappa(vector_nu, escalar_inventario_esperado)).tolist()
    # Un SKU sin demanda en el historial (ν = 0) es un producto que todavía no existe: no parte con existencias y, como en el generador,
    # su primer lote entra justo antes de su primera recuperación (el inventario precede a la demanda)
    vector_inicial = [(escalar_lote + 1) // 2 if escalar_nu > 0 else 0 for escalar_lote, escalar_nu in zip(vector_lote, vector_nu.tolist())]
    vector_existencias = list(vector_inicial)
    matriz_operacion, vector_dias_operacion = [], []
    for escalar_j, (_, escalar_sku) in enumerate(matriz_secuencia):
        escalar_i = escalar_sku - 1
        if vector_existencias[escalar_i] <= 0:
            matriz_operacion.extend([(0, escalar_sku)] * vector_lote[escalar_i])
            vector_existencias[escalar_i] += vector_lote[escalar_i]
        matriz_operacion.append((1, escalar_sku))
        vector_existencias[escalar_i] -= 1
        if vector_existencias[escalar_i] <= 0:
            matriz_operacion.extend([(0, escalar_sku)] * vector_lote[escalar_i])
            vector_existencias[escalar_i] += vector_lote[escalar_i]
        if vector_dias is not None:
            vector_dias_operacion.extend([vector_dias[escalar_j]] * (len(matriz_operacion) - len(vector_dias_operacion)))
        if escalar_limite_tareas is not None and len(matriz_operacion) > escalar_limite_tareas:
            raise ValueError(f"Datos reales: con su reposición la lista tiene más de {escalar_limite_tareas} tareas, el máximo es {escalar_limite_tareas}.")
    if vector_dias is not None:
        return matriz_operacion, vector_inicial, vector_dias_operacion
    return matriz_operacion, vector_inicial


def tamano_historial(escalar_tareas, escalar_porcentaje_historial):
    # Tareas del tramo inicial de la lista que sirve de historial: ese tramo es anterior a lo que se evalúa y no se ejecuta
    # La resta evita que un error de coma flotante en el producto agregue una tarea de más (8,219...% de 262 800 son 21 600 y no 21 601)
    return max(1, math.ceil(escalar_tareas * escalar_porcentaje_historial / 100 - 1e-9))


def preparar_datos_reales(matriz_tareas, escalar_porcentaje_historial=50, escalar_tareas_historial=None):
    # Convierte las etiquetas de SKU en números y calcula la demanda que conocen las políticas
    # Las políticas solo conocen el pasado: planifican (llenado inicial, clases y zonas) con el historial, el primer porcentaje de la lista,
    # y lo que se evalúa es lo que viene después (la configuración descarta el historial de la lista que se ejecuta)
    # El SKU 1 es el más frecuente en el historial, a igual frecuencia el que aparece primero; los que no aparecen en el historial
    # van al final (con demanda cero para las políticas), por orden de aparición: no se usa la frecuencia posterior al historial
    # `escalar_tareas_historial`, si se da, reemplaza al porcentaje: con días es la cantidad de recuperaciones de los B primeros días
    escalar_historial = tamano_historial(len(matriz_tareas), escalar_porcentaje_historial) if escalar_tareas_historial is None else escalar_tareas_historial
    cuenta_historial = Counter(txt_sku for _, txt_sku in matriz_tareas[:escalar_historial])
    cuenta_total = Counter(txt_sku for _, txt_sku in matriz_tareas)
    primera_aparicion = {}
    for escalar_indice, (_, txt_sku) in enumerate(matriz_tareas):
        primera_aparicion.setdefault(txt_sku, escalar_indice)
    vector_etiquetas = sorted(cuenta_total, key=lambda s: (-cuenta_historial.get(s, 0), primera_aparicion[s]))
    dic_numero = {txt_sku: escalar_indice + 1 for escalar_indice, txt_sku in enumerate(vector_etiquetas)}

    vector_demanda = np.array([cuenta_historial.get(txt_sku, 0) for txt_sku in vector_etiquetas], dtype=float)
    vector_demanda /= vector_demanda.sum()
    matriz_secuencia = [(escalar_actividad, dic_numero[txt_sku]) for escalar_actividad, txt_sku in matriz_tareas]
    return matriz_secuencia, vector_demanda, vector_etiquetas


def resumir_tareas(matriz_tareas, escalar_porcentaje_historial=50):
    # Resumen para mostrar al usuario antes de simular
    matriz_secuencia, vector_demanda, vector_etiquetas = preparar_datos_reales(matriz_tareas, escalar_porcentaje_historial)
    escalar_almacenar = sum(1 for escalar_actividad, _ in matriz_tareas if escalar_actividad == 0)
    return {
        "tareas": len(matriz_tareas),
        "skus": len(vector_etiquetas),
        "pct_almacenar": round(100 * escalar_almacenar / len(matriz_tareas), 1),
        "historial": tamano_historial(len(matriz_tareas), escalar_porcentaje_historial),
        "evaluadas": len(matriz_tareas) - tamano_historial(len(matriz_tareas), escalar_porcentaje_historial),
        "sin_historial": int((vector_demanda == 0).sum()),
        "principales": [[vector_etiquetas[i], round(float(vector_demanda[i]) * 100, 1)] for i in range(min(5, len(vector_etiquetas)))],
    }


def derivar_lista_demanda(matriz_tareas, vector_dias, bool_solo_demanda, escalar_porcentaje_historial, escalar_recuperaciones_historial,
                          escalar_recuperaciones_diarias, escalar_inventario_esperado, escalar_limite_tareas=None):
    """Todo lo que la simulación necesita de una lista real, como claves de la configuración (las usa `config_desde_json` y cada réplica con su
    propia demanda). Con días (`vector_dias`, día del modelo de cada recuperación) el historial son las primeras `escalar_recuperaciones_historial`
    (R·B) recuperaciones y la operación empieza el día de la siguiente, que es su día 0; sin días, el historial es el porcentaje inicial de la lista.
    La lista que se ejecuta empieza después del historial y, si solo trae demanda, se repone con lotes EOQ."""
    dic = {}
    if vector_dias is not None:
        escalar_historial = int(escalar_recuperaciones_historial)
        if escalar_historial < 1:
            raise ValueError("Datos reales: el historial debe tener al menos una recuperación.")
    else:
        escalar_historial = tamano_historial(len(matriz_tareas), escalar_porcentaje_historial)
    matriz_secuencia, vector_demanda, vector_etiquetas = preparar_datos_reales(matriz_tareas, escalar_porcentaje_historial, escalar_historial)
    # El historial es un tramo inicial anterior a lo que se evalúa: las políticas planifican con él y la lista que se ejecuta empieza después
    # Así ninguna política conoce demanda posterior al momento actual
    dic["escalar_tareas_historial"] = escalar_historial
    if escalar_historial >= len(matriz_secuencia):
        raise ValueError(f"Datos reales: el historial ({escalar_historial} de {len(matriz_secuencia)} tareas) no deja tareas por evaluar; "
                         + ("la lista tiene menos recuperaciones que el historial." if vector_dias is not None else "usa un porcentaje menor que 100."))
    matriz_completa = matriz_secuencia
    matriz_secuencia = matriz_secuencia[escalar_historial:]
    # Día de operación de cada tarea: el día 0 es el de la primera recuperación evaluada
    dic["escalar_dia_inicio_operacion"] = None if vector_dias is None else int(vector_dias[escalar_historial])
    vector_dias_evaluados = None if vector_dias is None else [escalar_d - dic["escalar_dia_inicio_operacion"] for escalar_d in vector_dias[escalar_historial:]]
    dic["vector_inventario_inicial"] = None
    vector_dias_operacion = vector_dias_evaluados
    if bool_solo_demanda:
        # El archivo solo trae demanda: los almacenamientos se generan con la reposición por lotes EOQ del capítulo 8
        vector_resultado = reponer_con_lotes_eoq(matriz_secuencia, vector_demanda, escalar_recuperaciones_diarias, escalar_inventario_esperado,
                                                 escalar_limite_tareas, vector_dias_evaluados)
        matriz_secuencia, dic["vector_inventario_inicial"] = vector_resultado[0], vector_resultado[1]
        if vector_dias is not None:
            vector_dias_operacion = vector_resultado[2]
    dic["secuencia_real"] = matriz_secuencia
    dic["vector_demanda_sku"] = vector_demanda
    dic["etiquetas_sku"] = vector_etiquetas
    dic["escalar_numero_skus"] = len(vector_etiquetas)
    dic["vector_dias_operacion"] = vector_dias_operacion
    dic["matriz_recuperaciones_dia"] = None
    if vector_dias is not None:
        # Recuperaciones de demanda de cada SKU (columna SKU - 1) en cada día de la lista, del primer día del historial al último
        matriz = np.zeros((max(vector_dias) + 1, len(vector_etiquetas)), dtype=np.int64)
        np.add.at(matriz, (np.asarray(vector_dias, dtype=np.int64), np.array([escalar_sku - 1 for escalar_actividad, escalar_sku in matriz_completa], dtype=np.int64)),
                  np.array([escalar_actividad for escalar_actividad, _ in matriz_completa], dtype=np.int64))
        dic["matriz_recuperaciones_dia"] = matriz
    return dic


def escenario_datos_reales(config, lista_llenado):
    """Escenario de una lista real con días: la popularidad móvil de cada día de operación (ec. 8.3), Pop_i(τ) = recuperaciones de los W días
    anteriores, el día de cada tarea y el registro de la demanda pasada que observa el agente. El día de operación 0 es el de la primera
    recuperación evaluada. Todos los SKU existen desde el principio (el surtido es fijo, supuesto D4). None si la lista no trae días.
    Las heurísticas no planifican con este escenario: planifican con la demanda del historial (ver `PoliticaAsignacion.demandaPlanificacion`)."""
    from asrs.generacion.escenario import EscenarioDemanda
    matriz_dias = config.get("matriz_recuperaciones_dia")
    if matriz_dias is None or config.get("vector_dias_operacion") is None:
        return None
    matriz_dias = np.asarray(matriz_dias, dtype=np.int64)
    escalar_b, escalar_w = int(config["escalar_dia_inicio_operacion"]), int(config["escalar_dias_ventana_popularidad"])
    escalar_n = matriz_dias.shape[1]
    escalar_dias_operacion = max(matriz_dias.shape[0] - escalar_b, 1)
    matriz_acumulada = np.vstack([np.zeros((1, escalar_n), dtype=np.int64), np.cumsum(matriz_dias, axis=0)])
    vector_hasta = np.clip(np.arange(escalar_dias_operacion) + escalar_b, 0, matriz_dias.shape[0])
    vector_desde = np.clip(vector_hasta - escalar_w, 0, matriz_dias.shape[0])
    matriz_popularidad = matriz_acumulada[vector_hasta] - matriz_acumulada[vector_desde]
    vector_inicial = np.asarray(config["vector_inventario_inicial"] if config.get("vector_inventario_inicial") is not None else np.zeros(escalar_n), dtype=int)
    # Primera llegada de cada SKU: antes del día 0 si parte con existencias, si no el día de su primer almacenamiento (infinito si nunca entra)
    vector_primera = np.where(vector_inicial > 0, -1e9, np.inf)
    for (escalar_actividad, escalar_sku), escalar_dia in zip(config["secuencia_real"], config["vector_dias_operacion"]):
        if escalar_actividad == 0 and not np.isfinite(vector_primera[escalar_sku - 1]):
            vector_primera[escalar_sku - 1] = escalar_dia
    dic_resumen = dict(config.get("dic_resumen_demanda") or {})
    return EscenarioDemanda(list(lista_llenado), list(config["secuencia_real"]), list(config["vector_dias_operacion"]), set(),
                            matriz_popularidad, vector_inicial, dic_resumen,
                            np.full(escalar_n, -10 ** 9, dtype=np.int64), np.full(escalar_n, 10 ** 9, dtype=np.int64),
                            matriz_dias, vector_primera, escalar_b)
