"""Política de asignación de almacenamiento: clase base.

Una política de asignación decide en qué celda libre se coloca un ítem. Se usa tanto para almacenar
como para reubicar: reubicar es almacenar un ítem que ya está en la estantería, desde la posición actual de la grúa.

Todas las políticas aplican la regla de minimización de varianza: entre las celdas que la política considera
igual de buenas, se prefiere la libre más profunda, de modo que los canales se llenan desde el fondo.
"""
import random as rnd

import numpy as np

from asrs.politicas.restriccion import MatricesRestriccion


class PoliticaAsignacion:
    # Nombre representativo, se usa para identificar la política en los resultados
    nombre = ''

    def __init__(self, instancia_estanteria, instancia_grua, instancia_puntoES, config):
        self.instancia_estanteria = instancia_estanteria
        self.instancia_grua = instancia_grua
        self.instancia_puntoES = instancia_puntoES
        self.config = config

        # Las políticas que usan zonas fijas rellenan esta matriz con la etiqueta de zona de cada celda
        self.matriz_zonificacion = None

    def etiquetaZona(self, escalar_sku):
        # Zona que corresponde a un SKU, cada política con zonas decide cómo agrupa sus SKUs
        return escalar_sku

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Devuelve una matriz con la forma de la estantería, los valores menores son los preferidos
        # vector_posicion_referencia: punto desde el cual se mide la cercanía, si es None cada política usa su referencia natural
        raise NotImplementedError

    def tiempoChebyshevXY(self, vector_posicion_referencia):
        # Tiempo mínimo de viaje en el plano xy desde la referencia a cada celda, max(x/vx, y/vy)
        matriz_posiciones = self.instancia_estanteria.posicionesCeldas()
        matriz_distancias_xy = np.abs(matriz_posiciones[:, :, :, :2] - np.asarray(vector_posicion_referencia, dtype=float)[:2])
        matriz_tiempos_xy = matriz_distancias_xy / self.instancia_grua.vector_velocidad[:2]

        # Se redondea para que las celdas equidistantes empaten de verdad
        return np.round(np.max(matriz_tiempos_xy, axis=3), 9)

    def asignar(self, escalar_sku, matriz_extra=None, booleano_restringir_zona=True, vector_posicion_referencia=None):
        # Devuelve el índice de la celda elegida para un ítem del SKU dado, o una tupla vacía si no hay celda factible
        # vector_posicion_referencia: al reubicar es la posición del ítem que se retira, al almacenar es None (la grúa en reposo)
        # matriz_extra: restricción lógica adicional, por ejemplo excluir el canal que se está despejando
        # booleano_restringir_zona: si es False se ignoran las zonas fijas de la política

        matriz_ocupacion = MatricesRestriccion().ocupacion(self.instancia_estanteria)

        # Solo se consideran celdas libres y accesibles, es decir sin ítems delante
        matriz_candidatas = np.logical_not(matriz_ocupacion) & MatricesRestriccion().accesibilidad(matriz_ocupacion)

        if booleano_restringir_zona and self.matriz_zonificacion is not None:
            matriz_candidatas = matriz_candidatas & (self.matriz_zonificacion == self.etiquetaZona(escalar_sku))

        if matriz_extra is not None:
            matriz_candidatas = matriz_candidatas & matriz_extra

        matriz_indices = np.argwhere(matriz_candidatas)
        if not len(matriz_indices):
            return ()

        # Se prefieren las celdas de menor valor del criterio de la política
        vector_valores = self.matrizCriterio(vector_posicion_referencia)[tuple(matriz_indices.T)]
        matriz_indices = matriz_indices[vector_valores == vector_valores.min()]

        # Regla de minimización de varianza: entre las equivalentes, la libre más profunda (índice de capa menor)
        matriz_indices = matriz_indices[matriz_indices[:, 2] == matriz_indices[:, 2].min()]

        # Si aún hay empate se desempata al azar
        return tuple(matriz_indices[rnd.randrange(len(matriz_indices))])
