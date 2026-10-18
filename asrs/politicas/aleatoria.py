"""Política aleatoria: el canal se elige al azar."""
import random as rnd

import numpy as np

from asrs.politicas.asignacion import PoliticaAsignacion


class PoliticaAleatoria(PoliticaAsignacion):
    nombre = 'Aleatoria'

    def matrizCriterio(self, vector_posicion_referencia = None):
        # Cada canal recibe un valor aleatorio, las capas de un mismo canal comparten el valor
        # La regla de minimización de varianza elige luego la celda libre más profunda del canal sorteado
        escalar_filas, escalar_columnas, escalar_capas = self.instancia_estanteria.matriz_contenedor_items.shape
        generador = np.random.default_rng(rnd.getrandbits(64))
        matriz_canales = generador.random((escalar_filas, escalar_columnas))

        return np.repeat(matriz_canales[:, :, np.newaxis], escalar_capas, axis=2)
