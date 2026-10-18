"""Lista de tareas a partir de datos reales: un archivo CSV con dos columnas, el SKU y la tarea (almacenar o recuperar), en orden."""
import csv
import math
from collections import Counter

import numpy as np

PALABRAS_ALMACENAR = {"a", "0", "almacenar", "almacenamiento", "entrada", "ingreso", "in", "store", "put", "putaway", "e"}
PALABRAS_RECUPERAR = {"r", "1", "recuperar", "recuperacion", "recuperación", "salida", "despacho", "out", "retrieve", "pick", "get", "s"}
NOMBRES_COLUMNA_ACTIVIDAD = ("task", "tarea", "act", "tipo", "oper", "movim", "type")
NOMBRES_COLUMNA_SKU = ("sku", "art", "prod", "ref", "item", "cod", "cód")


def _es_actividad(txt_valor):
    return txt_valor.strip().lower() in PALABRAS_ALMACENAR | PALABRAS_RECUPERAR


def leer_tareas(txt_csv, escalar_limite_tareas=None):
    # Devuelve la lista de tareas como pares (actividad, etiqueta del SKU): actividad 0 es almacenar y 1 es recuperar
    # Las columnas son SKU y Task, en ese orden. El separador (coma, punto y coma, tabulación o barra vertical) y el encabezado se detectan solos
    # Con encabezado las columnas se buscan por su nombre; sin encabezado se toma SKU y Task, salvo que solo la primera columna sea una actividad
    vector_lineas = [txt_linea for txt_linea in txt_csv.splitlines() if txt_linea.strip()]
    if not vector_lineas:
        raise ValueError("Datos reales: el archivo está vacío.")
    txt_separador = max(",;\t|", key=lambda s: vector_lineas[0].count(s))
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


def preparar_datos_reales(matriz_tareas, escalar_porcentaje_historial=100):
    # Convierte las etiquetas de SKU en números y calcula la demanda que conocen las políticas
    # Las políticas planifican con el historial, es decir, con el primer porcentaje de la lista (con 100 usan toda la lista):
    # el llenado inicial y la clasificación se hacen con esa demanda. Después se ejecuta la lista completa
    # El SKU 1 es el más frecuente en el historial, a igual frecuencia el que aparece primero; los que no aparecen en el historial
    # van al final (con demanda cero para las políticas), por frecuencia total
    escalar_historial = max(1, math.ceil(len(matriz_tareas) * escalar_porcentaje_historial / 100))
    cuenta_historial = Counter(txt_sku for _, txt_sku in matriz_tareas[:escalar_historial])
    cuenta_total = Counter(txt_sku for _, txt_sku in matriz_tareas)
    primera_aparicion = {}
    for escalar_indice, (_, txt_sku) in enumerate(matriz_tareas):
        primera_aparicion.setdefault(txt_sku, escalar_indice)
    vector_etiquetas = sorted(cuenta_total, key=lambda s: (-cuenta_historial.get(s, 0), -cuenta_total[s], primera_aparicion[s]))
    dic_numero = {txt_sku: escalar_indice + 1 for escalar_indice, txt_sku in enumerate(vector_etiquetas)}

    vector_demanda = np.array([cuenta_historial.get(txt_sku, 0) for txt_sku in vector_etiquetas], dtype=float)
    vector_demanda /= vector_demanda.sum()
    matriz_secuencia = [(escalar_actividad, dic_numero[txt_sku]) for escalar_actividad, txt_sku in matriz_tareas]
    return matriz_secuencia, vector_demanda, vector_etiquetas


def resumir_tareas(matriz_tareas, escalar_porcentaje_historial=100):
    # Resumen para mostrar al usuario antes de simular
    matriz_secuencia, vector_demanda, vector_etiquetas = preparar_datos_reales(matriz_tareas, escalar_porcentaje_historial)
    escalar_almacenar = sum(1 for escalar_actividad, _ in matriz_tareas if escalar_actividad == 0)
    return {
        "tareas": len(matriz_tareas),
        "skus": len(vector_etiquetas),
        "pct_almacenar": round(100 * escalar_almacenar / len(matriz_tareas), 1),
        "historial": max(1, math.ceil(len(matriz_tareas) * escalar_porcentaje_historial / 100)),
        "sin_historial": int((vector_demanda == 0).sum()),
        "principales": [[vector_etiquetas[i], round(float(vector_demanda[i]) * 100, 1)] for i in range(min(5, len(vector_etiquetas)))],
    }
