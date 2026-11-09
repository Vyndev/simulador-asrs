"""Celdas candidatas para almacenar una unidad y sus características: lo que el agente puntúa para decidir."""
import numpy as np

from asrs.dominio.item import Item
from asrs.politicas.restriccion import MatricesRestriccion


def mascara_candidatas(instancia_estanteria, matriz_extra=None):
    # Celdas entre las que elige el agente, con las mismas reglas que las heurísticas:
    #   1. libres y accesibles, es decir, sin unidades delante;
    #   2. dentro de la restricción adicional, si la hay (al reubicar se excluye el canal que se despeja);
    #   3. la regla de minimización de varianza: solo la capa libre más profunda entre todas las candidatas, de modo que se ocupan los
    #      fondos de todos los canales antes que cualquier frente y ningún fondo queda inaccesible.
    # Sin la regla, a ocupación alta el agente podría dejar fondos inaccesibles, descartar almacenamientos por falta de celda y
    # reducir el tiempo total con menos operaciones (docs/decisiones.md, 2026-11-08)
    matriz_ocupacion = MatricesRestriccion().ocupacion(instancia_estanteria)
    matriz_candidatas = np.logical_not(matriz_ocupacion) & MatricesRestriccion().accesibilidad(matriz_ocupacion)
    if matriz_extra is not None:
        matriz_candidatas = matriz_candidatas & matriz_extra
    vector_capas = np.flatnonzero(matriz_candidatas.any(axis=(0, 1)))
    if len(vector_capas):
        matriz_candidatas[:, :, vector_capas[0] + 1:] = False
    return matriz_candidatas


def celdas_candidatas(instancia_estanteria, matriz_tiempos, matriz_perfil, matriz_extra=None):
    # Devuelve (índices, características) de las celdas candidatas (ver `mascara_candidatas`)
    #   índices: matriz (M, 3) con la fila, la columna y la capa de cada celda
    #   características: matriz (M, 2 + d) con, por cada celda,
    #     0  tiempo de viaje desde el punto de referencia (`matriz_tiempos`) dividido entre el máximo de la estantería
    #     1  capa normalizada: 0 en la más profunda y 1 en la delantera
    #     2… perfil del SKU de la unidad de atrás (ceros si no hay), de largo d: la unidad que la nueva bloquearía
    # Con la regla de varianza todas las candidatas de una decisión están en la misma capa y la de atrás está ocupada salvo en la capa más
    # profunda, de modo que la fracción libre del canal y los indicadores de la celda de atrás quedan determinados por la capa
    # matriz_perfil: perfil de cada SKU (una fila por SKU - 1), del que sale el perfil de la unidad vecina
    matriz_indices = np.argwhere(mascara_candidatas(instancia_estanteria, matriz_extra))
    escalar_capas = instancia_estanteria.matriz_contenedor_items.shape[2]
    escalar_dim_perfil = matriz_perfil.shape[1]

    vector_filas, vector_columnas, vector_capas = matriz_indices.T
    vector_tiempo = matriz_tiempos[vector_filas, vector_columnas, vector_capas] / max(float(matriz_tiempos.max()), 1e-12)
    vector_capa = vector_capas / max(escalar_capas - 1, 1)

    matriz_perfil_vecino = np.zeros((len(matriz_indices), escalar_dim_perfil))
    for escalar_i in np.flatnonzero(vector_capas > 0):
        instancia_item = instancia_estanteria.matriz_contenedor_items[vector_filas[escalar_i], vector_columnas[escalar_i], vector_capas[escalar_i] - 1]
        if isinstance(instancia_item, Item) and instancia_item.escalar_sku <= len(matriz_perfil):
            matriz_perfil_vecino[escalar_i] = matriz_perfil[instancia_item.escalar_sku - 1]

    matriz_caracteristicas = np.column_stack([vector_tiempo, vector_capa, matriz_perfil_vecino])
    return matriz_indices, matriz_caracteristicas
