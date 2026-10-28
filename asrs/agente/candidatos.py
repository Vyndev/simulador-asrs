"""Celdas candidatas para almacenar una unidad y sus características: lo que el agente puntúa para decidir."""
import numpy as np

from asrs.dominio.item import Item
from asrs.politicas.restriccion import MatricesRestriccion


def mascara_candidatas(instancia_estanteria, booleano_restringido):
    # Celdas libres y accesibles, es decir, sin unidades delante
    # Restringido: solo la más profunda de cada canal, que es la celda que elige la regla de minimización de varianza
    # Libre: todas, de modo que el agente también decide si ocupa el frente de un canal con el fondo libre
    matriz_ocupacion = MatricesRestriccion().ocupacion(instancia_estanteria)
    matriz_candidatas = np.logical_not(matriz_ocupacion) & MatricesRestriccion().accesibilidad(matriz_ocupacion)
    if booleano_restringido:
        escalar_capas = matriz_candidatas.shape[2]
        matriz_capas = np.broadcast_to(np.arange(escalar_capas), matriz_candidatas.shape)
        matriz_capa_minima = np.where(matriz_candidatas, matriz_capas, escalar_capas).min(axis=2, keepdims=True)
        matriz_candidatas = matriz_candidatas & (matriz_capas == matriz_capa_minima)
    return matriz_candidatas


def celdas_candidatas(instancia_estanteria, matriz_tiempos_es, matriz_perfil, booleano_restringido):
    # Devuelve (índices, características) de las celdas candidatas
    #   índices: matriz (M, 3) con la fila, la columna y la capa de cada celda
    #   características: matriz (M, 5 + d) con, por cada celda,
    #     0  tiempo de viaje desde el E/S dividido entre el máximo de la estantería
    #     1  capa normalizada: 0 en la más profunda y 1 en la delantera
    #     2  fracción de celdas libres del canal
    #     3  1 si la celda de atrás está libre: ocupar esta celda dejaría ese fondo inaccesible
    #     4  1 si la celda de atrás está ocupada: la unidad nueva la bloquearía
    #     5… perfil del SKU de la unidad de atrás (ceros si no hay), de largo d
    # matriz_perfil: perfil de cada SKU (una fila por SKU - 1), del que sale el perfil de la unidad vecina
    matriz_candidatas = mascara_candidatas(instancia_estanteria, booleano_restringido)
    matriz_indices = np.argwhere(matriz_candidatas)
    matriz_ocupacion = MatricesRestriccion().ocupacion(instancia_estanteria)
    escalar_capas = matriz_ocupacion.shape[2]
    escalar_dim_perfil = matriz_perfil.shape[1]

    vector_filas, vector_columnas, vector_capas = matriz_indices.T
    vector_tiempo = matriz_tiempos_es[vector_filas, vector_columnas, vector_capas] / max(float(matriz_tiempos_es.max()), 1e-12)
    vector_capa = vector_capas / max(escalar_capas - 1, 1)
    vector_libres = (np.logical_not(matriz_ocupacion).sum(axis=2)[vector_filas, vector_columnas]) / escalar_capas

    vector_hay_atras = vector_capas > 0
    vector_atras = np.where(vector_hay_atras, vector_capas - 1, 0)
    vector_ocupada_atras = vector_hay_atras & matriz_ocupacion[vector_filas, vector_columnas, vector_atras]
    vector_libre_atras = vector_hay_atras & np.logical_not(vector_ocupada_atras)

    matriz_perfil_vecino = np.zeros((len(matriz_indices), escalar_dim_perfil))
    for escalar_i in np.flatnonzero(vector_ocupada_atras):
        instancia_item = instancia_estanteria.matriz_contenedor_items[vector_filas[escalar_i], vector_columnas[escalar_i], vector_atras[escalar_i]]
        if isinstance(instancia_item, Item) and instancia_item.escalar_sku <= len(matriz_perfil):
            matriz_perfil_vecino[escalar_i] = matriz_perfil[instancia_item.escalar_sku - 1]

    matriz_caracteristicas = np.column_stack([vector_tiempo, vector_capa, vector_libres, vector_libre_atras.astype(float),
                                              vector_ocupada_atras.astype(float), matriz_perfil_vecino])
    return matriz_indices, matriz_caracteristicas
