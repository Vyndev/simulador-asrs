"""Construcción de zonas fijas por clase para las políticas basadas en clases."""
import numpy as np


def construir_zonificacion(matriz_tiempos, vector_proporcion_clases, vector_orden_clases):
    # Reparte las celdas en zonas contiguas, de las más cercanas (menor tiempo) a las más lejanas
    # matriz_tiempos: tiempo de viaje a cada celda, con la forma de la estantería
    # vector_proporcion_clases: proporción de celdas a asignar a cada clase, indexada por etiqueta - 1
    # vector_orden_clases: etiquetas de clase en el orden en que reciben las celdas más cercanas
    # Devuelve una matriz con la etiqueta de clase de cada celda

    escalar_cantidad_celdas = matriz_tiempos.size
    vector_indices = np.indices(matriz_tiempos.shape).reshape(3, -1)

    # Orden de las celdas: tiempo, fila y columna, y por último capa (las profundas primero, minimización de varianza)
    # Las capas de un mismo canal quedan consecutivas, así una zona no deja el fondo de un canal en una clase y el frente en otra
    vector_orden_celdas = np.lexsort((vector_indices[2], vector_indices[1], vector_indices[0], matriz_tiempos.ravel()))

    # Cantidad de celdas por clase, las sobrantes por redondeo se asignan a la primera clase del orden
    vector_celdas_por_clase = np.floor(np.asarray(vector_proporcion_clases, dtype=float) * escalar_cantidad_celdas + 1e-9).astype(int)
    vector_celdas_por_clase[vector_orden_clases[0] - 1] += escalar_cantidad_celdas - vector_celdas_por_clase.sum()

    vector_zonificacion = np.zeros(escalar_cantidad_celdas, dtype=int)
    escalar_inicio = 0
    for escalar_etiqueta_clase in vector_orden_clases:
        escalar_cantidad = vector_celdas_por_clase[escalar_etiqueta_clase - 1]
        vector_zonificacion[vector_orden_celdas[escalar_inicio:escalar_inicio + escalar_cantidad]] = escalar_etiqueta_clase
        escalar_inicio += escalar_cantidad

    return vector_zonificacion.reshape(matriz_tiempos.shape)
