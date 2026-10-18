"""Zonas fijas para las políticas que reservan espacio, y agrupación de SKUs en clases."""
import numpy as np

from asrs.utilidades import repartir_entero


def agrupar_skus_en_clases(vector_demanda_sku, vector_proporcion_skus_por_clase):
    # Agrupa los SKUs en clases por demanda: la clase 1 recibe los SKUs más demandados, la última los menos demandados
    # vector_proporcion_skus_por_clase: proporción de SKUs de cada clase, por ejemplo [0.2, 0.3, 0.5]
    # Devuelve un arreglo con la clase (1, 2, ...) de cada SKU, indexado por SKU - 1
    vector_demanda = np.asarray(vector_demanda_sku, dtype=float)
    escalar_skus = len(vector_demanda)
    vector_cantidades = repartir_entero(escalar_skus, vector_proporcion_skus_por_clase, escalar_minimo=1)

    # Los SKUs se ordenan de mayor a menor demanda, a igual demanda se conserva el orden por número de SKU
    vector_orden_skus = np.argsort(-vector_demanda, kind='stable')
    vector_clase_de_sku = np.zeros(escalar_skus, dtype=int)
    escalar_inicio = 0
    for escalar_clase, escalar_cantidad in enumerate(vector_cantidades, start=1):
        vector_clase_de_sku[vector_orden_skus[escalar_inicio:escalar_inicio + escalar_cantidad]] = escalar_clase
        escalar_inicio += escalar_cantidad
    return vector_clase_de_sku


def construir_zonificacion(matriz_tiempos, vector_espacio_por_etiqueta, vector_orden_etiquetas):
    # Reparte las celdas en zonas contiguas, de las más cercanas (menor tiempo) a las más lejanas
    # matriz_tiempos: tiempo de viaje a cada celda, con la forma de la estantería
    # vector_espacio_por_etiqueta: espacio que necesita cada etiqueta (clase o SKU), indexado por etiqueta - 1, no hace falta que sume 1
    # vector_orden_etiquetas: etiquetas en el orden en que reciben las celdas más cercanas
    # Cada etiqueta recibe al menos un canal. Devuelve una matriz con la etiqueta de cada celda

    escalar_cantidad_celdas = matriz_tiempos.size
    vector_indices = np.indices(matriz_tiempos.shape).reshape(3, -1)

    # Orden de las celdas: tiempo, fila y columna, y por último capa (las profundas primero, minimización de varianza)
    # Las capas de un mismo canal quedan consecutivas, así una zona no deja el fondo de un canal en una etiqueta y el frente en otra
    vector_orden_celdas = np.lexsort((vector_indices[2], vector_indices[1], vector_indices[0], matriz_tiempos.ravel()))

    # Las zonas se reparten por canales completos (todas las capas de un canal pertenecen a la misma etiqueta)
    # Así una frontera entre zonas nunca deja el fondo de un canal detrás del frente de otro, lo que obligaría a reubicar
    escalar_capas = matriz_tiempos.shape[2]
    vector_canales_por_etiqueta = repartir_entero(escalar_cantidad_celdas // escalar_capas, vector_espacio_por_etiqueta, escalar_minimo=1)
    vector_celdas_por_etiqueta = vector_canales_por_etiqueta * escalar_capas

    vector_zonificacion = np.zeros(escalar_cantidad_celdas, dtype=int)
    escalar_inicio = 0
    for escalar_etiqueta in vector_orden_etiquetas:
        escalar_cantidad = vector_celdas_por_etiqueta[escalar_etiqueta - 1]
        vector_zonificacion[vector_orden_celdas[escalar_inicio:escalar_inicio + escalar_cantidad]] = escalar_etiqueta
        escalar_inicio += escalar_cantidad

    return vector_zonificacion.reshape(matriz_tiempos.shape)
